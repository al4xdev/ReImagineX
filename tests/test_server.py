import asyncio
import json
from pathlib import Path

import pytest
from fastapi import HTTPException

from src.server import (
    ConfigSchema,
    _clean_llm_response,
    _first_image_batch,
    _safe_child,
    get_config,
    get_ws_url,
    update_config,
)


def test_websocket_url_uses_matching_transport() -> None:
    assert get_ws_url("http://localhost:8001").startswith("ws://localhost:8001/ws")
    assert get_ws_url("https://example.test").startswith("wss://example.test/ws")


def test_safe_child_rejects_parent_traversal(tmp_path: Path) -> None:
    with pytest.raises(HTTPException):
        _safe_child(tmp_path, "../secret.png")


def test_safe_child_accepts_nested_paths(tmp_path: Path) -> None:
    assert _safe_child(tmp_path, "root/image.jpg") == tmp_path / "root" / "image.jpg"


def test_config_api_never_returns_openrouter_key(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setenv("OPENROUTER_API_KEY", "server-secret")

    payload = asyncio.run(get_config())

    assert payload["openrouter_configured"] is True
    assert "openrouter_api_key" not in payload


def test_config_api_never_returns_deepseek_key(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setenv("DEEPSEEK_API_KEY", "server-deepseek-secret")

    payload = asyncio.run(get_config())

    assert payload["deepseek_configured"] is True
    assert "deepseek_api_key" not in payload


def test_blank_key_update_preserves_existing_secret(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
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


def test_clear_flag_removes_stored_secret(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setenv("OPENROUTER_API_KEY", "server-secret")
    current = asyncio.run(get_config())
    request = ConfigSchema(**current, clear_openrouter_api_key=True)

    asyncio.run(update_config(request))

    saved = json.loads((tmp_path / "config.json").read_text(encoding="utf-8"))
    assert saved["openrouter_api_key"] == ""
    assert asyncio.run(get_config())["openrouter_configured"] is False


def test_clear_flag_removes_stored_deepseek_secret(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setenv("DEEPSEEK_API_KEY", "server-deepseek-secret")
    current = asyncio.run(get_config())
    request = ConfigSchema(**current, clear_deepseek_api_key=True)

    asyncio.run(update_config(request))

    saved = json.loads((tmp_path / "config.json").read_text(encoding="utf-8"))
    assert saved["deepseek_api_key"] == ""
    assert asyncio.run(get_config())["deepseek_configured"] is False


def test_new_key_replaces_stored_secret(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setenv("OPENROUTER_API_KEY", "server-secret")
    current = asyncio.run(get_config())
    request = ConfigSchema(**current, openrouter_api_key="fresh-key")

    asyncio.run(update_config(request))

    saved = json.loads((tmp_path / "config.json").read_text(encoding="utf-8"))
    assert saved["openrouter_api_key"] == "fresh-key"


def test_new_deepseek_key_replaces_stored_secret(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setenv("DEEPSEEK_API_KEY", "server-deepseek-secret")
    current = asyncio.run(get_config())
    request = ConfigSchema(**current, deepseek_api_key="fresh-deepseek-key")

    asyncio.run(update_config(request))

    saved = json.loads((tmp_path / "config.json").read_text(encoding="utf-8"))
    assert saved["deepseek_api_key"] == "fresh-deepseek-key"


def test_first_image_batch_reads_flat_executed_payload() -> None:
    payload = {"images": [{"filename": "a.png", "subfolder": "", "type": "output"}]}

    assert _first_image_batch(payload) == payload["images"]


def test_first_image_batch_reads_per_node_history_payload() -> None:
    images = [{"filename": "b.png", "subfolder": "", "type": "output"}]
    payload = {"459:474": {"text": ["..."]}, "461": {"images": images}}

    assert _first_image_batch(payload) == images


def test_first_image_batch_prefers_flat_and_ignores_empty_batches() -> None:
    images = [{"filename": "c.png", "subfolder": "s", "type": "temp"}]

    assert _first_image_batch({"461": {"images": []}, "images": images}) == images
    assert _first_image_batch({"461": {"images": []}, "3": {"text": ["x"]}}) is None
    assert _first_image_batch({}) is None


def test_clean_llm_response_strips_code_fences_and_whitespace() -> None:
    assert _clean_llm_response("```\nA scenic sunset\n```") == "A scenic sunset"
    assert _clean_llm_response("```markdown\nA scenic sunset\n```") == "A scenic sunset"
    assert _clean_llm_response("  A clean prompt  ") == "A clean prompt"
