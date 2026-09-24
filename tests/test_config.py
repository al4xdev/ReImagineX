import json
from pathlib import Path

import pytest

from src.config import LoraConfig, Settings, load_settings, save_settings


def test_settings_are_saved_atomically(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    settings = Settings()
    settings.system_prompt = "A neutral prompt"
    settings.openrouter_api_key = "secret"

    save_settings(settings)

    saved = json.loads((tmp_path / "config.json").read_text(encoding="utf-8"))
    assert saved["system_prompt"] == "A neutral prompt"
    assert saved["openrouter_api_key"] == "secret"
    assert load_settings().system_prompt == "A neutral prompt"
    assert not list(tmp_path.glob(".config.json.*"))


def test_lora_and_sampler_settings_round_trip(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    settings = Settings()
    settings.loras = [LoraConfig(name="qwen21/style.safetensors", strength=-0.75)]
    settings.sampler_name = "euler"
    settings.scheduler = "simple"

    save_settings(settings)
    restored = load_settings()

    assert restored.loras == settings.loras
    assert restored.sampler_name == "euler"
    assert restored.scheduler == "simple"
