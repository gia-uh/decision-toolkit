"""Toolkit home and config.json: created on first use, no environment variables."""

from __future__ import annotations

import json

from decision_toolkit.config import load_config, save_config, update_config


def test_first_use_creates_home_and_config(home):
    config = load_config(home)
    assert config.home == home.resolve()
    assert config.cache_dir == (home / "models").resolve()
    assert config.current is None
    assert config.device == "auto"
    payload = json.loads((home / "config.json").read_text())
    assert payload == {
        "cache_dir": str((home / "models").resolve()),
        "current": None,
        "device": "auto",
    }


def test_roundtrip_persists_changes(home):
    load_config(home)
    update_config(home, current="laya", device="cpu")
    reloaded = load_config(home)
    assert reloaded.current == "laya"
    assert reloaded.device == "cpu"


def test_relative_cache_dir_resolves_against_home(home):
    load_config(home)
    payload = {"cache_dir": "elsewhere", "current": None, "device": "auto"}
    (home / "config.json").write_text(json.dumps(payload))
    config = load_config(home)
    assert config.cache_dir == (home / "elsewhere").resolve()


def test_save_config_writes_pretty_json(home):
    config = load_config(home)
    config.current = "von"
    save_config(config)
    text = (home / "config.json").read_text()
    assert '"current": "von"' in text
    assert text.endswith("\n")


def test_update_config_rejects_unknown_keys(home):
    import pytest

    with pytest.raises(AttributeError):
        update_config(home, nope="x")
