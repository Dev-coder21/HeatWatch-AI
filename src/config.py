"""Single place to load project configuration from config/."""

from functools import lru_cache
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = PROJECT_ROOT / "config"


@lru_cache(maxsize=None)
def _load(name):
    with open(CONFIG_DIR / name, "r") as file:
        return yaml.safe_load(file)


def settings():
    return _load("settings.yaml")


def thresholds():
    return _load("thresholds.yaml")


def project_path(relative):
    """Resolve a path from config relative to the project root."""
    return PROJECT_ROOT / relative
