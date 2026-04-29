"""Composite loss: Charbonnier + MS-SSIM + LPIPS + DeltaE Lab."""
from __future__ import annotations

from typing import Dict, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def _gaussian_kernel(window_size: int, sigma: float, channels: int) -> Tensor:
    coords = torch.arange(window_size, dtype=torch.float32) - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    kernel_2d = g.outer(g)
    return kernel_2d.unsqueeze(0).unsqueeze(0).repeat(channels, 1, 1, 1)


def _ssim_and_cs(
    pred: Tensor,
    target: Tensor,
    window: Tensor,
    window_size: int,
    channels: int,
    c1: float,
    c2: float,
) -> tuple[Tensor, Tensor]:
    pad = window_size // 2
    mu_x = F.conv2d(pred, window, padding=pad, groups=channels)
    mu_y = F.conv2d(target, window, padding=pad, groups=channels)

    mu_x_sq = mu_x * mu_x
    mu_y_sq = mu_y * mu_y
    mu_xy = mu_x * mu_y

    sigma_x_sq = F.conv2d(pred * pred, window, padding=pad, groups=channels) - mu_x_sq
    sigma_y_sq = F.conv2d(target * target, window, padding=pad, groups=channels) - mu_y_sq
    sigma_xy = F.conv2d(pred * target, window, padding=pad, groups=channels) - mu_xy

    cs_map = (2.0 * sigma_xy + c2) / (sigma_x_sq + sigma_y_sq + c2)
    ssim_map = ((2.0 * mu_xy + c1) / (mu_x_sq + mu_y_sq + c1)) * cs_map

    ssim_val = ssim_map.mean(dim=(1, 2, 3))
    cs_val = cs_map.mean(dim=(1, 2, 3))
    return ssim_val, cs_val


def ms_ssim_index(
    pred: Tensor,
    target: Tensor,
    levels: int = 5,
    window_size: int = 11,
    sigma: float = 1.5,
    data_range: float = 1.0,
    scale_weights: Sequence[float] | None = None,
) -> Tensor:
    """Compute batch-mean MS-SSIM in [0, 1] (higher is better)."""
    if pred.shape != target.shape:
        raise ValueError(f"pred and target shape mismatch: {pred.shape} vs {target.shape}")

    channels = int(pred.size(1))
    window = _gaussian_kernel(window_size, sigma, channels).to(pred.device, pred.dtype)

    if scale_weights is None:
        scale_weights = (0.0448, 0.2856, 0.3001, 0.2363, 0.1333)

    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2

    x = pred
    y = target
    ssim_vals: list[Tensor] = []
    cs_vals: list[Tensor] = []

    max_levels = min(int(levels), len(scale_weights))
    for level in range(max_levels):
        ssim_val, cs_val = _ssim_and_cs(x, y, window, window_size, channels, c1, c2)
        ssim_vals.append(ssim_val)
        cs_vals.append(cs_val)

        if level < max_levels - 1:
            h, w = int(x.size(-2)), int(x.size(-1))
            if h < 2 or w < 2:
                break
            x = F.avg_pool2d(x, kernel_size=2, stride=2)
            y = F.avg_pool2d(y, kernel_size=2, stride=2)

    actual_levels = len(ssim_vals)
    if actual_levels == 0:
        return pred.new_tensor(0.0)

    weights = torch.tensor(scale_weights[:actual_levels], device=pred.device, dtype=pred.dtype)
    weights = weights / weights.sum()

    eps = pred.new_tensor(1e-6)
    ssim_last = torch.clamp(ssim_vals[-1], min=float(eps.item()), max=1.0)

    if actual_levels == 1:
        return ssim_last.mean()

    mcs = torch.stack(cs_vals[:-1], dim=0)
    mcs = torch.clamp(mcs, min=float(eps.item()), max=1.0)

    mcs_term = (mcs ** weights[:-1].view(-1, 1)).prod(dim=0)
    ssim_term = ssim_last ** weights[-1]
    return (mcs_term * ssim_term).mean()


class CharbonnierMSSSIMLPIPSLoss(nn.Module):
    """Weighted composite loss used for paired image enhancement.

    total = (
        w_char * Charbonnier
        + w_ms * (1 - MS-SSIM)
        + w_lpips * LPIPS
        + w_de * DeltaE_Lab
    )
    """

    def __init__(
        self,
        charbonnier_weight: float = 1.0,
        ms_ssim_weight: float = 0.2,
        lpips_weight: float = 0.03,
        delta_e_lab_weight: float = 0.1,
        charbonnier_eps: float = 1e-3,
        ms_ssim_levels: int = 5,
        ms_ssim_window_size: int = 11,
        ms_ssim_sigma: float = 1.5,
        lpips_net: str = "alex",
        lpips_pretrained: bool = True,
        components: object | None = None,
        **_: object,
    ) -> None:
        super().__init__()
        # Keep compatibility with config systems that may still pass loss.components.*.
        self._ignored_components = components

        self.charbonnier_weight = float(charbonnier_weight)
        self.ms_ssim_weight = float(ms_ssim_weight)
        self.lpips_weight = float(lpips_weight)
        self.delta_e_lab_weight = float(delta_e_lab_weight)

        self.charbonnier_eps = float(charbonnier_eps)
        self.ms_ssim_levels = int(ms_ssim_levels)
        self.ms_ssim_window_size = int(ms_ssim_window_size)
        self.ms_ssim_sigma = float(ms_ssim_sigma)

        self.lpips: nn.Module | None = None
        if self.lpips_weight > 0.0:
            try:
                import lpips as lpips_lib
            except Exception as exc:  # pragma: no cover - runtime dependency guard
                raise ImportError(
                    "LPIPS is required for CharbonnierMSSSIMLPIPSLoss when lpips_weight > 0. "
                    "Install via: pip install lpips"
                ) from exc

            self.lpips = lpips_lib.LPIPS(
                pretrained=bool(lpips_pretrained),
                net=lpips_net,
                pnet_rand=not bool(lpips_pretrained),
                verbose=False,
            )
            self.lpips.eval()
            for p in self.lpips.parameters():
                p.requires_grad = False

    def _charbonnier(self, pred: Tensor, target: Tensor) -> Tensor:
        diff = pred - target
        loss = torch.sqrt(diff * diff + self.charbonnier_eps ** 2)
        return loss.mean()

    def _lpips(self, pred: Tensor, target: Tensor) -> Tensor:
        if self.lpips is None:
            return pred.new_zeros(())
        pred_11 = pred.clamp(0.0, 1.0) * 2.0 - 1.0
        target_11 = target.clamp(0.0, 1.0) * 2.0 - 1.0
        return self.lpips(pred_11.float(), target_11.float()).mean()

    @staticmethod
    def _srgb_to_linear(x: Tensor) -> Tensor:
        x = x.clamp(0.0, 1.0)
        return torch.where(
            x <= 0.04045,
            x / 12.92,
            ((x + 0.055) / 1.055) ** 2.4,
        )

    def _rgb_to_lab(self, x: Tensor) -> Tensor:
        """Convert RGB in [0, 1] to CIE Lab (D65)."""
        x_lin = self._srgb_to_linear(x)
        r = x_lin[:, 0:1, :, :]
        g = x_lin[:, 1:2, :, :]
        b = x_lin[:, 2:3, :, :]

        xx = 0.4124564 * r + 0.3575761 * g + 0.1804375 * b
        yy = 0.2126729 * r + 0.7151522 * g + 0.0721750 * b
        zz = 0.0193339 * r + 0.1191920 * g + 0.9503041 * b

        xr = xx / 0.95047
        yr = yy / 1.00000
        zr = zz / 1.08883

        delta = 6.0 / 29.0
        delta3 = delta ** 3
        inv_3delta2 = 1.0 / (3.0 * delta * delta)
        shift = 4.0 / 29.0

        def f(t: Tensor) -> Tensor:
            return torch.where(t > delta3, t.pow(1.0 / 3.0), t * inv_3delta2 + shift)

        fx = f(xr)
        fy = f(yr)
        fz = f(zr)

        ll = 116.0 * fy - 16.0
        aa = 500.0 * (fx - fy)
        bb = 200.0 * (fy - fz)
        return torch.cat([ll, aa, bb], dim=1)

    def _delta_e_lab(self, pred: Tensor, target: Tensor) -> Tensor:
        pred_lab = self._rgb_to_lab(pred)
        target_lab = self._rgb_to_lab(target)
        d = pred_lab - target_lab
        de = torch.sqrt((d * d).sum(dim=1) + 1e-12)
        return de.mean()

    def component_losses(self, pred: Tensor, target: Tensor) -> Dict[str, Tensor]:
        zero = pred.new_zeros(())

        if self.charbonnier_weight > 0.0:
            charbonnier = self.charbonnier_weight * self._charbonnier(pred, target)
        else:
            charbonnier = zero

        if self.ms_ssim_weight > 0.0:
            ms_ssim_v = ms_ssim_index(
                pred,
                target,
                levels=self.ms_ssim_levels,
                window_size=self.ms_ssim_window_size,
                sigma=self.ms_ssim_sigma,
            )
            ms_ssim_loss = self.ms_ssim_weight * (1.0 - ms_ssim_v)
        else:
            ms_ssim_loss = zero

        if self.lpips_weight > 0.0:
            lpips_loss = self.lpips_weight * self._lpips(pred, target)
        else:
            lpips_loss = zero

        if self.delta_e_lab_weight > 0.0:
            delta_e_loss = self.delta_e_lab_weight * self._delta_e_lab(pred, target)
        else:
            delta_e_loss = zero

        return {
            "charbonnier": charbonnier,
            "ms_ssim": ms_ssim_loss,
            "lpips": lpips_loss,
            "delta_e_lab": delta_e_loss,
        }

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        parts = self.component_losses(pred, target)
        total = pred.new_zeros(())
        for v in parts.values():
            total = total + v
        return total
