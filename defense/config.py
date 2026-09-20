import yaml
from pathlib import Path
from defense.utils.logger import log

DEFAULT_CONFIG_PATH = Path(__file__).parent.parent / "config" / "default.yaml"

def load_config(config_path: str = None) -> dict:
    """
    Load YAML config. Falls back to default.yaml if no path given.
    Returns a plain dict.
    """
    path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    log("Config", f"Loaded config from: {path}")
    return cfg

def get_camera_config(cfg: dict) -> dict:
    """Shortcut to get cfg['camera']."""
    return cfg.get("camera", {})

def get_detection_config(cfg: dict) -> dict:
    """Shortcut to get cfg['detection']."""
    return cfg.get("detection", {})

def get_localization_config(cfg: dict) -> dict:
    """Shortcut to get cfg['localization']."""
    return cfg.get("localization", {})

def get_tracking_config(cfg: dict) -> dict:
    """Shortcut to get cfg['tracking']."""
    return cfg.get("tracking", {})
