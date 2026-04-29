# RWKV_DINO_github

`DINOv2 + RWKV + CNN` image enhancement pipeline for paired retouching (FiveK-style `raw -> expertC`).

This README is written to match the **current code status** in this repository (Hydra config + Lightning training).

## 1. Current Pipeline (Code-Accurate)

### 1.1 Overall architecture

```
Input image (RGB, [0,1])
  -> Frozen DINOv2 encoder (patch tokens)
  -> RWKV bottleneck (trainable)
  -> CNN decoder (trainable, residual prediction)
  -> Output enhanced image (RGB, [0,1])
```

Reference:
- `src/models/pipeline.py`
- `src/models/encoders/dinov2.py`
- `src/models/bottlenecks/rwkv_bottleneck.py`
- `src/models/decoders/cnn_decoder.py`

### 1.2 Encoder

- Default: `DINOv2Encoder` (`vit_base_patch14_dinov2`)
- Frozen in pipeline (`no_grad` + `freeze`)
- Outputs patch tokens `(B, N, C)`

Config:
- `configs/encoder/dinov2_b.yaml`

### 1.3 RWKV bottleneck (current status)

Default config:
- `configs/bottleneck/rwkv.yaml`

Key options:
- `use_2d_adapter: true`  
  Row/col positional injection + local depthwise/pointwise conv in token-map space.
- `use_hierarchical_multiscale: false` (default currently off)  
  If turned on, runs 3-stage high/low/high token modeling with downsample and fuse.
- `block_local_conv: false`  
  Note: pre-TimeMix depthwise path in `RWKVBlock` is currently kept as commented scaffold and bypassed in code.

ChannelMix:
- **Current active version is `ReLU^2`**
- `SwiGLU` code path is kept as comment for quick rollback/switch.

### 1.4 Decoder

- CNN decoder with PixelShuffle upsampling
- Multi-scale image skip guidance
- Predicts residual delta, final output is `clamp(input + delta, 0, 1)`

Config:
- `configs/decoder/cnn.yaml`

---

## 2. Data Format

Expected dataset root (one of these naming styles is accepted):

```
<data_dir>/
  raw/ or input/
    a0001.jpg
    ...
  c/ or expertC/
    a0001.jpg
    ...
  splits/               # optional
    train.txt
    val.txt
    test.txt
```

Behavior:
- If `splits/*.txt` exists, loader uses them directly.
- Otherwise, it auto-splits by ratio (`train_ratio/val_ratio/test_ratio`).

Reference:
- `src/data/fivek_dataset.py`
- `configs/data/fivek.yaml`

---

## 3. Environment Setup

## 3.1 Python

Recommended interpreter in this repo:
- `E:\Anaconda\python.exe`

## 3.2 Install dependencies

There is no pinned `requirements.txt` here currently, so install the essentials:

```powershell
E:\Anaconda\python.exe -m pip install --upgrade pip
E:\Anaconda\python.exe -m pip install torch torchvision pytorch-lightning hydra-core omegaconf timm pillow numpy lpips pytest
```

Notes:
- `lpips` is required if you use LPIPS-related losses/metrics.
- DINOv2 backbone loading relies on `timm` (with torch.hub fallback).

---

## 4. Training

Main entry:
- `scripts/train.py`

Default behavior:
- deterministic training enabled
- `CUBLAS_WORKSPACE_CONFIG` auto-set in script (`:4096:8`)
- early stopping by `val/psnr`

Default training config:
- `configs/base.yaml` (currently `max_epochs=20`, `patience=5`)
- `configs/config.yaml` (default composition)

### 4.1 Standard train

```powershell
cd D:\USYD\usyd-26s1\5703_capstone\RWKV_DINO_github
$env:DATA_DIR="D:/USYD/usyd-26s1/5703_capstone/datasets/archive"

E:\Anaconda\python.exe scripts\train.py experiment_name=rwkv_run bottleneck=rwkv
```

### 4.2 20-image subset + Stage1 manual loss weights

```powershell
E:\Anaconda\python.exe scripts\train.py `
  experiment_name=run20_stage1_manual `
  data.data_dir=D:/USYD/usyd-26s1/5703_capstone/datasets/archive/subsets/fivek_c_20 `
  bottleneck=rwkv `
  loss=stage1 `
  loss.components.l1.weight=1.0 `
  loss.components.ssim.weight=0.2 `
  loss.components.perceptual.weight=0.02 `
  data.train_subset_size=20 `
  data.val_subset_size=5 `
  data.test_subset_size=5
```

### 4.3 20-image subset + Charbonnier/MS-SSIM/LPIPS manual weights

```powershell
E:\Anaconda\python.exe scripts\train.py `
  experiment_name=run20_charb_manual `
  data.data_dir=D:/USYD/usyd-26s1/5703_capstone/datasets/archive/subsets/fivek_c_20 `
  bottleneck=rwkv `
  loss=Charbonnier_MS-SSIM_LPIPS `
  loss.charbonnier_weight=1.0 `
  loss.ms_ssim_weight=0.2 `
  loss.lpips_weight=0.03 `
  loss.delta_e_lab_weight=0.1 `
  data.train_subset_size=20 `
  data.val_subset_size=5 `
  data.test_subset_size=5
```

---

## 5. Evaluation and Visualization

### 5.1 Evaluate checkpoint

```powershell
E:\Anaconda\python.exe scripts\evaluate.py `
  --checkpoint outputs/checkpoints/<your.ckpt> `
  --config outputs/logs/<exp>/config/config.yaml `
  --data-dir D:/USYD/usyd-26s1/5703_capstone/datasets/archive `
  --output-dir outputs/eval `
  --device cuda
```

### 5.2 Generate qualitative comparisons

```powershell
E:\Anaconda\python.exe scripts\generate_comparison.py `
  --checkpoint outputs/checkpoints/<your.ckpt> `
  --config outputs/logs/<exp>/config/config.yaml `
  --num-batches 4 `
  --num-samples 8 `
  --device cuda
```

### 5.3 Export JSON to LaTeX tables

```powershell
E:\Anaconda\python.exe scripts\export_results.py --input results --output paper/tables
```

---

## 6. VSCode Launch (already configured)

File:
- `.vscode/launch.json`

Included:
- full A/B/C/D pipeline entries via `scripts/run_pipeline.py`
- two Hydra training entries (Stage1 and Charbonnier loss)
- dataset selector input (`archive`, `fivek_c_20`, `fivek_c_200`)
- deterministic env consistency:
  - `CUDA_VISIBLE_DEVICES=0`
  - `CUBLAS_WORKSPACE_CONFIG=:4096:8`

---

## 7. Important Config Knobs

### 7.1 Data split behavior

- `data.overfit_on_train_subset=true`  
  Reuses train subset for train/val/test (sanity overfit mode).
- `data.train_subset_size=null`  
  Means "no train truncation", **not** "no split".

### 7.2 RWKV ablation knobs

In `configs/bottleneck/rwkv.yaml`:

- `use_2d_adapter`: enable/disable token-space 2D adapter
- `use_hierarchical_multiscale`: enable/disable hierarchical multiscale branch
- `block_local_conv`: config flag for pre-TimeMix local conv path (currently code path is bypassed)

Recommended quick ablation examples:

```powershell
# baseline-like RWKV
... bottleneck.use_2d_adapter=false bottleneck.use_hierarchical_multiscale=false

# 2D adapter only
... bottleneck.use_2d_adapter=true bottleneck.use_hierarchical_multiscale=false

# hierarchical only
... bottleneck.use_2d_adapter=false bottleneck.use_hierarchical_multiscale=true

# both
... bottleneck.use_2d_adapter=true bottleneck.use_hierarchical_multiscale=true
```

---

## 8. Common Issues

### 8.1 `FileNotFoundError` in DataLoader worker

Usually caused by missing paired files or wrong directory names.
Check:
- input dir exists (`raw/` or `input/`)
- target dir exists (`c/` or `expertC/`)
- filenames match by stem

### 8.2 Deterministic CUDA errors

This repo sets deterministic mode in trainer.  
Ensure `CUBLAS_WORKSPACE_CONFIG=:4096:8` is present (already set in `train.py` and VSCode launch env).

### 8.3 DINO weights download/network issue

If `timm` model loading fails due network restrictions, verify:
- network access
- `timm` installation
- torch hub cache permissions

---

## 9. Minimal Project Structure

```
RWKV_DINO_github/
  configs/
    encoder/dinov2_b.yaml
    bottleneck/rwkv.yaml
    decoder/cnn.yaml
    loss/stage1.yaml
    loss/Charbonnier_MS-SSIM_LPIPS.yaml
    data/fivek.yaml
    config.yaml
    base.yaml
  scripts/
    train.py
    evaluate.py
    generate_comparison.py
    export_results.py
    run_pipeline.py
  src/
    lit_module.py
    data/
    models/
    losses/
    evaluation/
    utils/
```

---

## 10. Reproducibility Notes

- Seed is set by `src/utils/seed.py`.
- Trainer runs with `deterministic=True`.
- Config snapshots are saved to:
  - `outputs/logs/<experiment_name>/config/config.yaml`
  - `outputs/logs/<experiment_name>/config/config.json`

This makes it straightforward to reproduce a run with exact overrides.

