import json
from pathlib import Path

import pytest

from src.config import Settings, load_settings, save_settings


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
