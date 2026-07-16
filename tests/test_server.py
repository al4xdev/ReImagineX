import asyncio
from pathlib import Path

import pytest
from fastapi import HTTPException

from src.server import ConfigSchema, _safe_child, get_config, get_ws_url, update_config


def test_websocket_url_uses_matching_transport() -> None:
    assert get_ws_url("http://localhost:8001").startswith("ws://localhost:8001/ws")
    assert get_ws_url("https://example.test").startswith("wss://example.test/ws")


def test_safe_child_rejects_parent_traversal(tmp_path: Path) -> None:
    with pytest.raises(HTTPException):
        _safe_child(tmp_path, "../secret.png")


def test_safe_child_accepts_nested_paths(tmp_path: Path) -> None:
    assert _safe_child(tmp_path, "root/image.jpg") == tmp_path / "root" / "image.jpg"


def test_config_api_never_returns_openrouter_key(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setenv("OPENROUTER_API_KEY", "server-secret")

    payload = asyncio.run(get_config())

    assert payload["openrouter_configured"] is True
    assert "openrouter_api_key" not in payload


def test_blank_key_update_preserves_existing_secret(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setenv("OPENROUTER_API_KEY", "server-secret")
    current = asyncio.run(get_config())
    request = ConfigSchema(
        **current,
        openrouter_api_key=None,
        clear_openrouter_api_key=False,
    )

    asyncio.run(update_config(request))

    saved = (tmp_path / "config.json").read_text(encoding="utf-8")
    assert "server-secret" in saved
