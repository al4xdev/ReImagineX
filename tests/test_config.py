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


def test_cfg_setting_round_trip(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    settings = Settings()
    settings.cfg = 3.5

    save_settings(settings)
    restored = load_settings()

    assert restored.cfg == 3.5


def test_custom_models_persist_without_being_reverted(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    settings = Settings()
    settings.diffusion_model_name = "noctQUncensoredRealismBuiltOn_v40Base_int8.safetensors"
    settings.clip_model_name = "q06.safetensors"
    settings.vae_model_name = "flux2-vae.safetensors"

    save_settings(settings)
    restored = load_settings()

    assert restored.diffusion_model_name == "noctQUncensoredRealismBuiltOn_v40Base_int8.safetensors"
    assert restored.clip_model_name == "q06.safetensors"
    assert restored.vae_model_name == "flux2-vae.safetensors"

