"""RWKV bottleneck using official RWKV-v4 implementation.

This module directly uses code from https://github.com/BlinkDL/RWKV-LM/RWKV-v4
with minimal modifications to support CPU fallback.
"""
from __future__ import annotations

import math
import os
import warnings
from pathlib import Path
from torch import Tensor

import torch
import torch.nn as nn

from src.models.bottlenecks.base import BaseBottleneck

########################################################################################################
# CUDA Kernel - From official RWKV-v4/src/model.py
########################################################################################################

T_MAX = 1024  # maximum context length for CUDA kernel
_wkv_cuda = None  # Module-level cache for the loaded CUDA extension
_wkv_cuda_failed = False
_verbose_first_load = True
_fallback_warning_emitted = False

# ******* auotograd******* begin
def _resolve_cuda_sources() -> list[str]:
    """Resolve WKV CUDA source files with robust absolute paths."""
    project_root = Path(__file__).resolve().parents[3]
    candidate_dirs = [
        project_root / "configs" / "CUDA",
        project_root / "configs" / "cuda",
        project_root / "cuda",
        project_root / "CUDA",
    ]
    for cuda_dir in candidate_dirs:
        op = cuda_dir / "wkv_op.cpp"
        cu = cuda_dir / "wkv_cuda.cu"
        if op.is_file() and cu.is_file():
            return [str(op), str(cu)]
    tried = ", ".join(str(d) for d in candidate_dirs)
    raise FileNotFoundError(f"RWKV CUDA sources not found. Tried: {tried}")


def _rwkv_float_mode() -> str:
    """Return RWKV float mode with a safe default."""
    return os.environ.get("RWKV_FLOAT_MODE", "fp32").strip().lower()
# ******* auotograd******* end


def _load_cuda_extension():
    """Load the CUDA extension, using cached version if available."""
    global _wkv_cuda, _wkv_cuda_failed, _verbose_first_load

    if _wkv_cuda is not None:
        return _wkv_cuda
    if _wkv_cuda_failed:
        return None

    try:
        import torch.utils.cpp_extension as cpp_extension

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available on this system")

        # Use cached extension if available
        _wkv_cuda = cpp_extension.load(
            name="wkv",
            sources=_resolve_cuda_sources(),
            verbose=False,  # Don't spam output every time
            extra_cuda_cflags=[
                "-res-usage",
                "--maxrregcount=60",
                "--use_fast_math",
                "-O3",
                "-Xptxas -O3",
                f"-DTmax={T_MAX}",
            ],
        )

        if _verbose_first_load:
            print("✓ RWKV CUDA kernel loaded successfully")
            _verbose_first_load = False

        return _wkv_cuda
    except Exception as e:
        _wkv_cuda_failed = True
        if _verbose_first_load:
            warnings.warn(
                f"Failed to load RWKV CUDA kernel: {e}\n"
                "Falling back to pure PyTorch implementation (slower).\n"
                "To enable CUDA acceleration, ensure:\n"
                "  1. CUDA toolkit is installed and nvcc is in PATH\n"
                "  2. PyTorch is built with CUDA support\n"
                "  3. You have write permissions for the build cache\n"
                "  4. ninja is installed: pip install ninja\n"
                "Note: First-time compilation may take several minutes.",
                stacklevel=2,
            )
            _verbose_first_load = False
        return None


# ******* auotograd******* begin
class WKV(torch.autograd.Function):
    """Autograd wrapper for the RWKV CUDA extension."""

    @staticmethod
    def forward(ctx, B, T, C, w, u, k, v):
        if T > T_MAX:
            raise RuntimeError(f"T={T} exceeds T_MAX={T_MAX}. Increase T_MAX to use CUDA WKV.")
        if (B * C) % min(C, 1024) != 0:
            raise RuntimeError(
                f"Invalid launch shape for CUDA WKV: B={B}, C={C}. "
                f"Require B*C % min(C,1024) == 0."
            )

        wkv_module = _load_cuda_extension()
        if wkv_module is None:
            raise RuntimeError("RWKV CUDA extension is unavailable.")

        mode = _rwkv_float_mode()
        ctx.B = B
        ctx.T = T
        ctx.C = C
        ctx.mode = mode
        ctx.w_dtype = w.dtype
        ctx.u_dtype = u.dtype
        ctx.k_dtype = k.dtype
        ctx.v_dtype = v.dtype

        if "32" in mode:
            w_cuda = -torch.exp(w.contiguous())
            u_cuda = u.contiguous()
            k_cuda = k.contiguous()
            v_cuda = v.contiguous()
        else:
            w_cuda = -torch.exp(w.float().contiguous())
            u_cuda = u.float().contiguous()
            k_cuda = k.float().contiguous()
            v_cuda = v.float().contiguous()

        ctx.save_for_backward(w_cuda, u_cuda, k_cuda, v_cuda)
        y = torch.empty(
            (B, T, C),
            device=k.device,
            dtype=torch.float32,
            memory_format=torch.contiguous_format,
        )
        with torch.cuda.device(k.device):
            wkv_module.forward(B, T, C, w_cuda, u_cuda, k_cuda, v_cuda, y)

        if "32" in mode:
            return y
        if mode == "fp16":
            return y.half()
        if mode == "bf16":
            return y.bfloat16()
        return y

    @staticmethod
    def backward(ctx, gy):
        B = ctx.B
        T = ctx.T
        C = ctx.C
        mode = ctx.mode
        if T > T_MAX:
            raise RuntimeError(f"T={T} exceeds T_MAX={T_MAX}.")
        if (B * C) % min(C, 1024) != 0:
            raise RuntimeError(
                f"Invalid launch shape for CUDA WKV backward: B={B}, C={C}. "
                f"Require B*C % min(C,1024) == 0."
            )

        wkv_module = _load_cuda_extension()
        if wkv_module is None:
            raise RuntimeError("RWKV CUDA extension is unavailable for backward.")

        w, u, k, v = ctx.saved_tensors
        gw = torch.zeros((B, C), device=gy.device, dtype=torch.float32).contiguous()
        gu = torch.zeros((B, C), device=gy.device, dtype=torch.float32).contiguous()
        gk = torch.zeros((B, T, C), device=gy.device, dtype=torch.float32).contiguous()
        gv = torch.zeros((B, T, C), device=gy.device, dtype=torch.float32).contiguous()

        gy_in = gy.contiguous() if "32" in mode else gy.float().contiguous()
        with torch.cuda.device(gy.device):
            wkv_module.backward(B, T, C, w, u, k, v, gy_in, gw, gu, gk, gv)

        gw = torch.sum(gw, dim=0).to(dtype=ctx.w_dtype)
        gu = torch.sum(gu, dim=0).to(dtype=ctx.u_dtype)
        gk = gk.to(dtype=ctx.k_dtype)
        gv = gv.to(dtype=ctx.v_dtype)
        return (None, None, None, gw, gu, gk, gv)
# ******* auotograd******* end


def RUN_CUDA_OR_CPU(B, T, C, w, u, k, v):
    """WKV forward pass - uses CUDA if available, otherwise pure PyTorch.

    The PyTorch implementation is adapted from the official RWKV-v4
    RNN mode (src/model_run.py SA method), converted to batch processing.

    See: https://github.com/BlinkDL/RWKV-LM/blob/RWKV-v4/src/model_run.py#L318-L354
    """
    global _fallback_warning_emitted

    # Try CUDA first if tensor is on CUDA
    if k.is_cuda:
        try:
            wkv_module = _load_cuda_extension()
            if wkv_module is not None and T <= T_MAX:
                return WKV.apply(B, T, C, w, u, k, v)
            if wkv_module is not None and T > T_MAX and not _fallback_warning_emitted:
                warnings.warn(
                    f"T={T} exceeds CUDA T_MAX={T_MAX}; falling back to PyTorch WKV.",
                    stacklevel=2,
                )
                _fallback_warning_emitted = True
        except Exception as exc:
            if not _fallback_warning_emitted:
                warnings.warn(
                    f"CUDA WKV forward failed ({exc}); falling back to PyTorch WKV.",
                    stacklevel=2,
                )
                _fallback_warning_emitted = True
            # Fall through to PyTorch implementation
            pass

    # Pure PyTorch implementation - adapted from official RWKV_RNN.SA
    dtype = k.dtype
    k = k.float()
    v = v.float()
    w = -torch.exp(w.float())
    u = u.float()

    outputs = []
    for b in range(B):
        # Initialize RNN state (from official SA: pp, aa, bb)
        pp = torch.full((C,), -1e38, device=k.device, dtype=k.dtype)
        aa = torch.zeros(C, device=k.device, dtype=k.dtype)
        bb = torch.zeros(C, device=k.device, dtype=k.dtype)

        out_t = []
        for t in range(T):
            kt = k[b, t, :]
            vt = v[b, t, :]

            # WKV recurrence (from official SA lines 335-350)
            ww = u + kt
            p = torch.maximum(pp, ww)
            e1 = torch.exp(pp - p)
            e2 = torch.exp(ww - p)
            a = e1 * aa + e2 * vt
            b_coef = e1 * bb + e2

            yt = a / b_coef.clamp_min(1e-9)
            out_t.append(yt)

            ww = w + pp
            p = torch.maximum(ww, kt)
            e1 = torch.exp(ww - p)
            e2 = torch.exp(kt - p)
            aa = e1 * aa + e2 * vt
            bb = e1 * bb + e2
            pp = p

        outputs.append(torch.stack(out_t, dim=0))

    return torch.stack(outputs, dim=0).to(dtype=dtype)


########################################################################################################
# RWKV: RWKV Time-mix + RWKV Channel-mix
# From official RWKV-v4/src/model.py (lines 164-272)
########################################################################################################


class RWKV_TimeMix(nn.Module):
    """RWKV Time-Mixing module - from official RWKV-v4.

    Source: https://github.com/BlinkDL/RWKV-LM/blob/RWKV-v4/src/model.py#L164-L232
    """

    def __init__(self, config, layer_id):
        super().__init__()
        self.layer_id = layer_id
        self.ctx_len = config.ctx_len
        self.n_embd = config.n_embd

        attn_sz = config.n_embd

        with torch.no_grad():  # fancy init
            ratio_0_to_1 = (layer_id / (config.n_layer - 1)) if config.n_layer > 1 else 0
            ratio_1_to_almost0 = 1.0 - (layer_id / config.n_layer)

            # fancy time_decay
            decay_speed = torch.ones(attn_sz)
            for h in range(attn_sz):
                decay_speed[h] = -5 + 8 * (h / (attn_sz - 1)) ** (0.7 + 1.3 * ratio_0_to_1)
            self.time_decay = nn.Parameter(decay_speed)

            # fancy time_first
            zigzag = (torch.tensor([(i + 1) % 3 - 1 for i in range(attn_sz)]) * 0.5)
            self.time_first = nn.Parameter(torch.ones(attn_sz) * math.log(0.3) + zigzag)

            # fancy time_mix
            x = torch.ones(1, 1, config.n_embd)
            for i in range(config.n_embd):
                x[0, 0, i] = i / config.n_embd
            self.time_mix_k = nn.Parameter(torch.pow(x, ratio_1_to_almost0))
            self.time_mix_v = nn.Parameter(torch.pow(x, ratio_1_to_almost0) + 0.3 * ratio_0_to_1)
            self.time_mix_r = nn.Parameter(torch.pow(x, 0.5 * ratio_1_to_almost0))

        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))

        self.key = nn.Linear(config.n_embd, attn_sz, bias=False)
        self.value = nn.Linear(config.n_embd, attn_sz, bias=False)
        self.receptance = nn.Linear(config.n_embd, attn_sz, bias=False)

        self.output = nn.Linear(attn_sz, config.n_embd, bias=False)

        self.key.scale_init = 0
        self.receptance.scale_init = 0
        self.output.scale_init = 0

    def jit_func(self, x):
        """Mix x with the previous timestep to produce xk, xv, xr.

        From official RWKV-v4/src/model.py jit_func (lines 208-223).
        """
        xx = self.time_shift(x)
        xk = x * self.time_mix_k + xx * (1 - self.time_mix_k)
        xv = x * self.time_mix_v + xx * (1 - self.time_mix_v)
        xr = x * self.time_mix_r + xx * (1 - self.time_mix_r)

        k = self.key(xk)
        v = self.value(xv)
        r = self.receptance(xr)
        sr = torch.sigmoid(r)

        return sr, k, v

    def forward(self, x):
        """Forward pass - from official RWKV-v4/src/model.py (lines 225-232)."""
        B, T, C = x.size()

        sr, k, v = self.jit_func(x)

        rwkv = sr * RUN_CUDA_OR_CPU(B, T, C, self.time_decay, self.time_first, k, v)
        rwkv = self.output(rwkv)
        return rwkv


class RWKV_ChannelMix(nn.Module):
    """RWKV Channel-Mixing module - from official RWKV-v4.

    Source: https://github.com/BlinkDL/RWKV-LM/blob/RWKV-v4/src/model.py#L235-L271
    """

    def __init__(self, config, layer_id):
        super().__init__()
        self.layer_id = layer_id

        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))

        with torch.no_grad():  # fancy init of time_mix
            ratio_1_to_almost0 = 1.0 - (layer_id / config.n_layer)

            x = torch.ones(1, 1, config.n_embd)
            for i in range(config.n_embd):
                x[0, 0, i] = i / config.n_embd

            self.time_mix_k = nn.Parameter(torch.pow(x, ratio_1_to_almost0))
            self.time_mix_r = nn.Parameter(torch.pow(x, ratio_1_to_almost0))

        hidden_sz = 4 * config.n_embd
        self.key = nn.Linear(config.n_embd, hidden_sz, bias=False)
        self.receptance = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.value = nn.Linear(hidden_sz, config.n_embd, bias=False)

        self.value.scale_init = 0
        self.receptance.scale_init = 0

    def forward(self, x):
        """Forward pass - from official RWKV-v4/src/model.py (lines 260-271)."""
        xx = self.time_shift(x)
        xk = x * self.time_mix_k + xx * (1 - self.time_mix_k)
        xr = x * self.time_mix_r + xx * (1 - self.time_mix_r)

        k = self.key(xk)
        k = torch.square(torch.relu(k))
        kv = self.value(k)

        rkv = torch.sigmoid(self.receptance(xr)) * kv
        return rkv


########################################################################################################
# Block and Config - From official RWKV-v4/src/model.py
########################################################################################################


class GPTConfig:
    """Configuration class for RWKV models.

    From official RWKV-v4/src/model.py (lines 278-284).
    """

    def __init__(self, vocab_size, ctx_len, **kwargs):
        self.vocab_size = vocab_size
        self.ctx_len = ctx_len
        for k, v in kwargs.items():
            setattr(self, k, v)


class Block(nn.Module):
    """RWKV Block - from official RWKV-v4.

    Source: https://github.com/BlinkDL/RWKV-LM/blob/RWKV-v4/src/model.py#L286-L313
    """

    def __init__(self, config, layer_id, drop_rate: float = 0.1):
        super().__init__()
        self.config = config
        self.layer_id = layer_id

        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)

        if self.layer_id == 0:
            self.ln0 = nn.LayerNorm(config.n_embd)

        self.att = RWKV_TimeMix(config, layer_id)
        self.ffn = RWKV_ChannelMix(config, layer_id)

        self.drop1 = nn.Dropout(drop_rate)
        self.drop2 = nn.Dropout(drop_rate)

    def forward(self, x):
        """Forward pass - from official RWKV-v4/src/model.py (lines 305-313)."""
        if self.layer_id == 0:
            x = self.ln0(x)
        x = x + self.drop1(self.att(self.ln1(x)))
        x = x + self.drop2(self.ffn(self.ln2(x)))
        return x


########################################################################################################
# Bottleneck Interface
########################################################################################################


class RWKVBottleneckV1(BaseBottleneck):
    """RWKV bottleneck using official RWKV-v4 implementation.

    This bottleneck directly uses RWKV components from the official
    RWKV-v4 implementation at https://github.com/BlinkDL/RWKV-LM/RWKV-v4

    Components used:
    - RWKV_TimeMix: Time-mixing with WKV recurrence
    - RWKV_ChannelMix: Channel-mixing feed-forward
    - Block: Complete RWKV block with residual connections
    - GPTConfig: Configuration class

    CUDA Acceleration:
        If CUDA kernel compilation succeeds, WKV operations will be accelerated.
        Otherwise, falls back to pure PyTorch implementation.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 384,
        num_layers: int = 6,
        drop_rate: float = 0.1,
        enable_bidirectional_scan: bool = False,
        enable_row_column_scan: bool = False,
        enable_2d_adapter: bool = False,
        adapter_kernel_size: int = 3,
        adapter_alpha_init: float = 0.0,
    ) -> None:
        super().__init__()
        self._output_dim = hidden_dim
        self._hidden_dim = hidden_dim
        self.enable_bidirectional_scan = bool(enable_bidirectional_scan)
        self.enable_row_column_scan = bool(enable_row_column_scan)
        self.enable_2d_adapter = bool(enable_2d_adapter)

        # Project input to hidden dimension
        self.proj_in = nn.Linear(input_dim, hidden_dim)

        # Create RWKV config using official GPTConfig
        config = GPTConfig(
            vocab_size=1,  # Not used for bottleneck (no token embedding)
            ctx_len=4096,  # Large enough for any sequence length
            n_embd=hidden_dim,
            n_layer=num_layers,
            model_type="RWKV",
        )

        # Create RWKV blocks using official implementation
        self.blocks = nn.ModuleList(
            [Block(config, i, drop_rate=drop_rate) for i in range(num_layers)]
        )

        # Output projection and normalization
        self.norm = nn.LayerNorm(hidden_dim)
        self.proj_out = nn.Linear(hidden_dim, hidden_dim)

        # ******** MOD START: Optional 2D local adapter (config-driven) ********
        if self.enable_2d_adapter:
            if adapter_kernel_size % 2 == 0 or adapter_kernel_size < 1:
                raise ValueError(
                    f"adapter_kernel_size must be an odd positive integer, got {adapter_kernel_size}"
                )
            padding = adapter_kernel_size // 2
            self.adapter_dw = nn.Conv2d(
                hidden_dim,
                hidden_dim,
                kernel_size=adapter_kernel_size,
                padding=padding,
                groups=hidden_dim,
                bias=True,
            )
            self.adapter_pw = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1, bias=True)
            self.adapter_alpha = nn.Parameter(torch.tensor(float(adapter_alpha_init)))
        else:
            self.adapter_dw = None
            self.adapter_pw = None
            self.adapter_alpha = None
        # ******** MOD END: Optional 2D local adapter (config-driven) ********

    # ******** MOD START: Helper methods for optional scan modes ********
    def _run_rwkv_stack(self, x: Tensor) -> Tensor:
        for block in self.blocks:
            x = block(x)
        return x

    def _run_bidirectional(self, x: Tensor) -> Tensor:
        y_fwd = self._run_rwkv_stack(x)
        y_bwd = torch.flip(self._run_rwkv_stack(torch.flip(x, dims=[1])), dims=[1])
        return 0.5 * (y_fwd + y_bwd)

    def _resolve_hw(self, n_tokens: int, h: int | None, w: int | None) -> tuple[int, int]:
        if h is not None and w is not None:
            if h * w != n_tokens:
                raise ValueError(
                    f"Token count mismatch: h*w={h*w}, n_tokens={n_tokens}. "
                    "Check encoder patch grid size and passed h,w."
                )
            return h, w

        side = int(round(math.sqrt(n_tokens)))
        if side * side != n_tokens:
            raise ValueError(
                "h,w are required for non-square token grids when row/column scan "
                "or 2D adapter is enabled."
            )
        return side, side
    # ******** MOD END: Helper methods for optional scan modes ********

    def forward(self, x: Tensor, h: int | None = None, w: int | None = None) -> Tensor:
        """Process token features through RWKV bottleneck.

        Args:
            x: Encoder output token features of shape (B, N, C_in).
            h: Optional token-grid height.
            w: Optional token-grid width.

        Returns:
            Processed token features of shape (B, N, hidden_dim).
        """
        x = self.proj_in(x)

        # ******** MOD START: Optional row/column scan + bidirectional scan ********
        if self.enable_row_column_scan:
            B, N, C = x.shape
            h_eff, w_eff = self._resolve_hw(N, h, w)

            x2d = x.reshape(B, h_eff, w_eff, C)
            row_tokens = x2d.reshape(B, h_eff * w_eff, C)
            col_tokens = x2d.permute(0, 2, 1, 3).contiguous().reshape(B, h_eff * w_eff, C)

            row_out = (
                self._run_bidirectional(row_tokens)
                if self.enable_bidirectional_scan
                else self._run_rwkv_stack(row_tokens)
            )
            col_out = (
                self._run_bidirectional(col_tokens)
                if self.enable_bidirectional_scan
                else self._run_rwkv_stack(col_tokens)
            )
            col_out = (
                col_out.reshape(B, w_eff, h_eff, C)
                .permute(0, 2, 1, 3)
                .contiguous()
                .reshape(B, h_eff * w_eff, C)
            )
            x = 0.5 * (row_out + col_out)
        elif self.enable_bidirectional_scan:
            x = self._run_bidirectional(x)
        else:
            x = self._run_rwkv_stack(x)
        # ******** MOD END: Optional row/column scan + bidirectional scan ********

        # ******** MOD START: Optional 2D local adapter (config-driven) ********
        if self.enable_2d_adapter:
            B, N, C = x.shape
            h_eff, w_eff = self._resolve_hw(N, h, w)
            x2d = x.reshape(B, h_eff, w_eff, C).permute(0, 3, 1, 2).contiguous()
            x2d = x2d + self.adapter_alpha * self.adapter_pw(self.adapter_dw(x2d))
            x = x2d.permute(0, 2, 3, 1).contiguous().reshape(B, h_eff * w_eff, C)
        # ******** MOD END: Optional 2D local adapter (config-driven) ********

        x = self.norm(x)
        return self.proj_out(x)

    @property
    def output_dim(self) -> int:
        return self._output_dim


if __name__ == "__main__":
    # Test CUDA kernel loading
    print(f"CUDA Available: {torch.cuda.is_available()}")

    # Quick test
    model = RWKVBottleneckV1(input_dim=768, hidden_dim=384, num_layers=2)
    if torch.cuda.is_available():
        model = model.cuda()
        x = torch.randn(1, 10, 768, device="cuda")
    else:
        x = torch.randn(1, 10, 768)
    y = model(x)
    print(f"Test output shape: {y.shape}")
