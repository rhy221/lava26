"""OmegaConf-based config loader for LAVA 2026.

Usage:
    cfg = load_config("config.yaml", ["retriever.max_pages=10", "vlm.enabled=true"])
    print(cfg.retriever.max_pages)              # 10
    snapshot(cfg, Path("runs/12345"))           # saves config_resolved.yaml
"""
from pathlib import Path
from omegaconf import DictConfig, OmegaConf


def load_config(
    path: str = "config.yaml",
    cli_overrides: list[str] | None = None,
) -> DictConfig:
    """Load YAML config and apply CLI dotlist overrides.

    Overrides use dotlist format: ["key.nested=value", "flag=true"]
    Supports OmegaConf interpolations like ${oc.env:USER}.
    """
    cfg = OmegaConf.load(path)
    if cli_overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(cli_overrides))
    OmegaConf.resolve(cfg)
    return cfg


def snapshot(cfg: DictConfig, run_dir: Path) -> None:
    """Save resolved config to run_dir/config_resolved.yaml for reproducibility."""
    run_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, run_dir / "config_resolved.yaml")
