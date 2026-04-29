"""Save a fully-resolved Hydra config snapshot at training start."""
from __future__ import annotations

import json
from pathlib import Path

from omegaconf import DictConfig, OmegaConf


def save_config_snapshot(
    cfg: DictConfig,
    out_dir: str | Path | None = None,
) -> Path:
    """Persist the fully-resolved config to disk for reproducibility.

    Writes two files:
    - config.yaml  — human-readable, includes all override values
    - config.json  — machine-readable, for programmatic result linking

    Args:
        cfg: Hydra DictConfig with all command-line overrides applied.
        out_dir: Output directory. Defaults to
                 outputs/logs/{experiment_name}/config/.

    Returns:
        Path to the saved YAML file.
    """
    if out_dir is None:
        log_dir = Path(cfg.logging.log_dir) / cfg.experiment_name / "config"
    else:
        log_dir = Path(out_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    yaml_path = log_dir / "config.yaml"
    OmegaConf.save(cfg, yaml_path, resolve=True)

    json_path = log_dir / "config.json"
    with open(json_path, "w") as f:
        json.dump(OmegaConf.to_container(cfg, resolve=True), f, indent=2)

    return yaml_path
