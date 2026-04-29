"""Frozen DINOv3 encoder with trainable multi-layer fusion head for E1/E2."""
from __future__ import annotations

import math
from typing import Sequence

import torch
import torch.nn as nn
from torch import Tensor
from transformers import AutoModel

from src.models.encoders.base import BaseEncoder


class LoRALinear(nn.Module):
    """Low-rank adapter for a frozen Linear layer used by E2 LoRA runs."""

    def __init__(
        self,
        base: nn.Linear,
        rank: int,
        alpha: float | None = None,
    ) -> None:
        super().__init__()
        if rank <= 0:
            raise ValueError(f"LoRA rank must be positive, got {rank}")

        self.base = base
        self.rank = int(rank)
        self.alpha = float(alpha) if alpha is not None else float(rank)
        self.scaling = self.alpha / float(self.rank)

        self.lora_A = nn.Linear(base.in_features, self.rank, bias=False)
        self.lora_B = nn.Linear(self.rank, base.out_features, bias=False)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)
        self.enable_lora_training()

    def enable_lora_training(self) -> None:
        """Freeze the base layer while keeping only LoRA weights trainable."""
        for param in self.base.parameters():
            param.requires_grad = False
        self.lora_A.weight.requires_grad = True
        self.lora_B.weight.requires_grad = True

    def forward(self, x: Tensor) -> Tensor:
        return self.base(x) + self.scaling * self.lora_B(self.lora_A(x))


class DINOv3MultiLayerEncoder(BaseEncoder):
    """DINOv3 ViT-B encoder with trainable multi-layer fusion.

    E1 keeps the DINOv3 backbone frozen, reads several intermediate hidden
    states, strips CLS/register tokens, and fuses them into a single token
    sequence of shape (B, N, C). This lets the rest of the pipeline stay
    unchanged while giving RWKV richer multi-level features.

    Args:
        model_name: HuggingFace model ID.
        patch_size: Patch size of the backbone (16 for DINOv3 ViT-B).
        embed_dim: Output feature dimension (768 for DINOv3 ViT-B).
        layer_ids: Transformer block indices to extract, e.g. [3, 6, 9, 11].
        fusion_mode: 'sum' or 'concat'.
        local_files_only: If True, never attempt to download from HuggingFace.
    """

    def __init__(
        self,
        model_name: str = "facebook/dinov3-vitb16-pretrain-lvd1689m",
        patch_size: int = 16,
        embed_dim: int = 768,
        layer_ids: Sequence[int] = (3, 6, 9, 11),
        fusion_mode: str = "sum",
        local_files_only: bool = False,
        anchor_last_layer: bool = True,
        residual_scale_init: float = 0.0,
        use_output_norm: bool = False,
        enable_lora: bool = False,
        lora_rank: int = 4,
        lora_alpha: float | None = None,
        lora_target_layers: Sequence[int] = (8, 9, 10, 11),
        lora_target_modules: Sequence[str] = ("q_proj", "v_proj"),
    ) -> None:
        super().__init__()
        if fusion_mode not in {"sum", "concat"}:
            raise ValueError(f"fusion_mode must be 'sum' or 'concat', got {fusion_mode!r}")
        if not layer_ids:
            raise ValueError("layer_ids must contain at least one layer index")

        self._patch_size = patch_size
        self._embed_dim = embed_dim
        self.layer_ids = list(layer_ids)
        self.fusion_mode = fusion_mode
        self.anchor_last_layer = bool(anchor_last_layer)
        self.use_output_norm = bool(use_output_norm)
        self.enable_lora = bool(enable_lora)
        self.lora_rank = int(lora_rank)
        self.lora_alpha = lora_alpha
        self.lora_target_layers = [int(layer_id) for layer_id in lora_target_layers]
        self.lora_target_modules = [str(name) for name in lora_target_modules]

        self.backbone = AutoModel.from_pretrained(
            model_name,
            local_files_only=local_files_only,
        )
        self.num_register_tokens = self.backbone.config.num_register_tokens

        n_backbone_layers = len(self.backbone.model.layer)
        for layer_id in self.layer_ids:
            if layer_id < 0 or layer_id >= n_backbone_layers:
                raise ValueError(
                    f"layer_id {layer_id} out of range for DINOv3 with {n_backbone_layers} layers"
                )

        # [E2 MOD] Optional LoRA only targets the deeper q_proj / v_proj blocks
        # used in the E2 experiments. The pretrained backbone remains frozen;
        # only the injected low-rank matrices are trainable.
        if self.enable_lora:
            if self.lora_rank <= 0:
                raise ValueError(f"lora_rank must be positive, got {self.lora_rank}")
            invalid_modules = sorted(set(self.lora_target_modules) - {"q_proj", "v_proj"})
            if invalid_modules:
                raise ValueError(
                    "lora_target_modules only supports q_proj/v_proj, got "
                    f"{invalid_modules}"
                )
            for layer_id in self.lora_target_layers:
                if layer_id < 0 or layer_id >= n_backbone_layers:
                    raise ValueError(
                        f"LoRA target layer {layer_id} out of range for DINOv3 with "
                        f"{n_backbone_layers} layers"
                    )
            self._inject_lora_into_backbone()

        # hidden_states[0] is the embedding output, so block k lives at k + 1.
        self.hidden_state_indices = [layer_id + 1 for layer_id in self.layer_ids]

        if self.anchor_last_layer and len(self.layer_ids) < 2:
            raise ValueError("anchor_last_layer=True requires at least two layer_ids")

        self.register_buffer(
            "pixel_mean",
            torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "pixel_std",
            torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )

        # [E1 MOD] PSNR-oriented E1: keep the last layer as the anchor feature
        # and only learn residual corrections from lower/mid-level features.
        # This preserves the baseline feature distribution at initialization,
        # which makes init_from_checkpoint from E0b much more effective.
        self.anchor_index = len(self.layer_ids) - 1 if self.anchor_last_layer else None
        self.aux_layer_ids = self.layer_ids[:-1] if self.anchor_last_layer else self.layer_ids
        self.aux_projs = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(embed_dim),
                nn.Linear(embed_dim, embed_dim),
            )
            for _ in self.aux_layer_ids
        ])

        # [E1 MOD] Two E1 fusion variants:
        # - sum: last-layer anchor + gated residuals from lower layers
        # - concat: last-layer anchor + zero-init residual delta from concat fusion
        if self.fusion_mode == "sum":
            self.residual_scales = nn.Parameter(
                torch.full((len(self.aux_layer_ids),), float(residual_scale_init))
            )
            self.concat_fuse = None
        else:
            self.residual_scales = None
            concat_width = embed_dim * (len(self.aux_layer_ids) + (1 if self.anchor_last_layer else 0))
            self.concat_fuse = nn.Sequential(
                nn.LayerNorm(concat_width),
                nn.Linear(concat_width, embed_dim),
                nn.GELU(),
                nn.Linear(embed_dim, embed_dim),
            )
            # [E1 MOD] Zero-init the final delta head so concat fusion starts as
            # the baseline last-layer encoder, then learns only useful residuals.
            nn.init.zeros_(self.concat_fuse[-1].weight)
            nn.init.zeros_(self.concat_fuse[-1].bias)
        self.output_norm = nn.LayerNorm(embed_dim) if self.use_output_norm else nn.Identity()

        self.freeze()

    # [E2 MOD] Inject LoRA wrappers into selected deep attention projections.
    def _inject_lora_into_backbone(self) -> None:
        for layer_id in self.lora_target_layers:
            attention = self.backbone.model.layer[layer_id].attention
            for module_name in self.lora_target_modules:
                module = getattr(attention, module_name, None)
                if module is None:
                    raise AttributeError(
                        f"DINOv3 attention layer {layer_id} has no attribute {module_name!r}"
                    )
                if isinstance(module, LoRALinear):
                    module.enable_lora_training()
                    continue
                if not isinstance(module, nn.Linear):
                    raise TypeError(
                        f"Expected nn.Linear at layer {layer_id}.{module_name}, got "
                        f"{type(module).__name__}"
                    )
                setattr(
                    attention,
                    module_name,
                    LoRALinear(module, rank=self.lora_rank, alpha=self.lora_alpha),
                )

    def freeze(self) -> None:
        """Freeze the DINOv3 backbone but keep the E1 fusion head trainable."""
        # [E1 MOD] Only the pretrained backbone is frozen. The new fusion head
        # remains trainable so E1 can adapt encoder features to enhancement.
        self.backbone.eval()
        for param in self.backbone.parameters():
            param.requires_grad = False
        if self.enable_lora:
            # [E2 MOD] Re-enable gradients only for the injected LoRA weights.
            for layer_id in self.lora_target_layers:
                attention = self.backbone.model.layer[layer_id].attention
                for module_name in self.lora_target_modules:
                    module = getattr(attention, module_name)
                    if isinstance(module, LoRALinear):
                        module.enable_lora_training()

    def train(self, mode: bool = True) -> "DINOv3MultiLayerEncoder":
        """Train fusion modules while forcing the frozen backbone to stay in eval."""
        # [E1 MOD] Override BaseEncoder.train(): backbone stays in eval mode,
        # but the trainable fusion layers still follow the Lightning mode flag.
        # [E2 MOD] LoRA lives inside the frozen backbone modules, but keeping
        # backbone.eval() is still intentional: we want deterministic pretrained
        # features while only updating the low-rank attention residuals.
        nn.Module.train(self, mode)
        self.backbone.eval()
        return self

    def _strip_special_tokens(self, x: Tensor) -> Tensor:
        """Remove CLS and register tokens, leaving only patch tokens."""
        return x[:, 1 + self.num_register_tokens :, :]

    def forward(self, x: Tensor) -> Tensor:
        """Extract and fuse multi-layer DINOv3 patch token features."""
        x = x.clamp(0.0, 1.0)
        mean = self.pixel_mean.to(device=x.device, dtype=x.dtype)
        std = self.pixel_std.to(device=x.device, dtype=x.dtype)
        x = (x - mean) / std

        # [E2 MOD] E1 can keep the frozen backbone under no_grad, but E2 LoRA
        # needs gradients through the selected deep attention projections.
        if self.enable_lora and self.training:
            outputs = self.backbone(pixel_values=x, output_hidden_states=True)
        else:
            with torch.no_grad():
                outputs = self.backbone(pixel_values=x, output_hidden_states=True)

        hidden_states = outputs.hidden_states
        per_layer_feats = [
            self._strip_special_tokens(hidden_states[idx])
            for idx in self.hidden_state_indices
        ]

        if self.anchor_last_layer:
            # [E1 MOD] Use the deepest layer as the stable baseline-compatible
            # anchor, then add trainable residuals from lower layers.
            base_feat = per_layer_feats[-1]
            aux_feats = per_layer_feats[:-1]
            projected_aux = [
                proj(feat) for proj, feat in zip(self.aux_projs, aux_feats)
            ]

            if self.fusion_mode == "sum":
                fused = base_feat
                for scale, feat in zip(self.residual_scales, projected_aux):
                    fused = fused + scale * feat
            else:
                delta = self.concat_fuse(torch.cat(projected_aux + [base_feat], dim=-1))
                fused = base_feat + delta
        else:
            projected_feats = [
                proj(feat) for proj, feat in zip(self.aux_projs, per_layer_feats)
            ]
            if self.fusion_mode == "sum":
                fused = projected_feats[0].new_zeros(projected_feats[0].shape)
                for scale, feat in zip(self.residual_scales, projected_feats):
                    fused = fused + scale * feat
            else:
                fused = self.concat_fuse(torch.cat(projected_feats, dim=-1))

        return self.output_norm(fused)

    @property
    def embed_dim(self) -> int:
        return self._embed_dim

    @property
    def patch_size(self) -> int:
        return self._patch_size
