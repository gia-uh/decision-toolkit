"""`decision_toolkit.config` — toolkit home and ``config.json``.

Configuration lives in ``<home>/config.json`` (default home
``~/.local/share/decision-toolkit/``), created on first use with the defaults
already written so you can see and edit them (e.g. point ``cache_dir`` at
another disk). There are **no environment variables**; pass ``home=`` /
``--home`` for non-default installations such as tests.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

#: Default toolkit home (no environment variables).
DEFAULT_HOME = Path.home() / ".local/share/decision-toolkit"

CONFIG_NAME = "config.json"


@dataclass
class Config:
    """The toolkit configuration: where models live, which one is in use."""

    home: Path
    cache_dir: Path
    current: str | None = None
    device: str = "auto"


def load_config(home: str | Path | None = None) -> Config:
    """Load ``<home>/config.json``, creating it with defaults on first use."""
    home_path = Path(home).expanduser() if home is not None else DEFAULT_HOME
    home_path = home_path.resolve()
    config_path = home_path / CONFIG_NAME

    if not config_path.is_file():
        home_path.mkdir(parents=True, exist_ok=True)
        config = Config(home=home_path, cache_dir=home_path / "models")
        save_config(config)
        return config

    data = json.loads(config_path.read_text(encoding="utf-8"))
    cache_dir = Path(data.get("cache_dir") or (home_path / "models")).expanduser()
    if not cache_dir.is_absolute():
        cache_dir = (home_path / cache_dir).resolve()
    return Config(
        home=home_path,
        cache_dir=cache_dir,
        current=data.get("current"),
        device=data.get("device", "auto"),
    )


def save_config(config: Config) -> None:
    """Persist a :class:`Config` to ``<home>/config.json``."""
    config.home.mkdir(parents=True, exist_ok=True)
    payload = {
        "cache_dir": str(config.cache_dir),
        "current": config.current,
        "device": config.device,
    }
    (config.home / CONFIG_NAME).write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )


def update_config(home: str | Path | None = None, **changes: object) -> Config:
    """Load, patch and save the configuration in one call."""
    config = load_config(home)
    for key, value in changes.items():
        if not hasattr(config, key):
            raise AttributeError(f"unknown configuration key: {key}")
        setattr(config, key, value)
    save_config(config)
    return config


__all__ = ["Config", "DEFAULT_HOME", "CONFIG_NAME", "load_config", "save_config", "update_config"]
