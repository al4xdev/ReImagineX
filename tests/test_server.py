import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi import HTTPException
from pydantic import ValidationError

from src.config import LoraConfig
from src.server import (
    ConfigSchema,
    PromptRequest,
    _clean_llm_response,
    _first_image_batch,
    _safe_child,
    expand_prompt_deepseek,
    generate,
    get_config,
    get_state,
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


def test_config_rejects_more_than_five_loras(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    current = asyncio.run(get_config())

    with pytest.raises(ValidationError):
        ConfigSchema(**{**current, "loras": [{"name": f"style-{i}.safetensors"} for i in range(6)]})


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


def test_deepseek_prompt_expansion_disables_reasoning_and_uses_temperature_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payloads: list[dict[str, Any]] = []

    class FakeResponse:
        status_code = 200

        def json(self) -> dict[str, Any]:
            return {"choices": [{"message": {"content": "expanded prompt"}}]}

    class FakeClient:
        def __init__(self, **_kwargs: Any) -> None:
            pass

        async def __aenter__(self) -> "FakeClient":
            return self

        async def __aexit__(self, *_args: Any) -> None:
            pass

        async def post(self, _url: str, **kwargs: Any) -> FakeResponse:
            payloads.append(kwargs["json"])
            return FakeResponse()

    monkeypatch.setattr(
        "src.server.load_settings",
        lambda: SimpleNamespace(deepseek_api_key="test-key", deepseek_model="deepseek-flash", system_prompt="Normalize"),
    )
    monkeypatch.setattr("src.server.httpx.AsyncClient", FakeClient)

    assert asyncio.run(expand_prompt_deepseek("original prompt")) == "expanded prompt"
    assert payloads[0]["temperature"] == 1.0
    assert payloads[0]["reasoning_effort"] == "none"


def test_album_keeps_root_first_and_other_images_newest_first(monkeypatch: pytest.MonkeyPatch) -> None:
    # Stored state is newest first; the newest image can belong to an older branch.
    state = [
        {"id": "new-grandchild", "parent_id": "older-child", "status": "completed"},
        {"id": "other-root-child", "parent_id": "other-root", "status": "completed"},
        {"id": "new-child", "parent_id": "root", "status": "completed"},
        {"id": "older-child", "parent_id": "root", "status": "completed"},
        {"id": "root", "parent_id": None, "status": "completed"},
        {"id": "other-root", "parent_id": None, "status": "completed"},
    ]

    async def fake_load_state() -> list[dict[str, Any]]:
        return state

    monkeypatch.setattr("src.server.load_state", fake_load_state)
    result = asyncio.run(get_state(root_id="root"))

    assert [item["id"] for item in result] == ["root", "new-grandchild", "new-child", "older-child"]


def test_legacy_upscale_exposes_source_prompt_for_frontend(monkeypatch: pytest.MonkeyPatch) -> None:
    state = [
        {"id": "upscaled", "parent_id": "original", "upscaled": True,
         "prompt": "technical upscale prompt", "prompt_original": "Upscale (2MP)"},
        {"id": "original", "parent_id": "root", "prompt": "expanded image prompt",
         "prompt_original": "my image prompt"},
        {"id": "root", "parent_id": None, "prompt": "Initial Upload"},
    ]

    async def fake_load_state() -> list[dict[str, Any]]:
        return state

    monkeypatch.setattr("src.server.load_state", fake_load_state)
    result = asyncio.run(get_state(root_id="root"))
    upscaled = next(item for item in result if item["id"] == "upscaled")

    assert upscaled["source_prompt"] == "expanded image prompt"
    assert upscaled["source_prompt_original"] == "my image prompt"
    assert upscaled["prompt"] == "technical upscale prompt"


def test_reprocess_reuses_seed_and_expanded_prompt(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    base = {"id": "base", "parent_id": None, "filename": "base.png", "comfyName": "base.png"}
    saved: list[dict[str, Any]] = []
    submitted: list[dict[str, Any]] = []

    async def fake_load_state() -> list[dict[str, Any]]:
        return [base]

    def fake_load() -> list[dict[str, Any]]:
        return [base]

    def fake_save(state: list[dict[str, Any]]) -> None:
        saved.extend(state)

    class FakeResponse:
        status_code = 200

        def json(self) -> dict[str, str]:
            return {"prompt_id": "comfy-prompt"}

    class FakeClient:
        async def __aenter__(self) -> "FakeClient":
            return self

        async def __aexit__(self, *_args: Any) -> None:
            pass

        async def post(self, _url: str, json: dict[str, Any]) -> FakeResponse:
            submitted.append(json["prompt"])
            return FakeResponse()

    monkeypatch.setattr("src.server.load_state", fake_load_state)
    monkeypatch.setattr("src.server._load", fake_load)
    monkeypatch.setattr("src.server._save", fake_save)
    monkeypatch.setattr("src.server.httpx.AsyncClient", FakeClient)

    request = PromptRequest(
        base_id="base", prompt="expanded prompt", original_prompt="original prompt",
        bypass_llm=True, seed=123, additional_images=["ref.png"],
        loras=[LoraConfig(name="qwen21/style.safetensors", strength=0.65)],
        sampler_name="euler", scheduler="simple",
    )
    result = asyncio.run(generate(request))

    assert result["seed"] == 123
    assert result["prompt"] == "expanded prompt"
    assert result["prompt_original"] == "original prompt"
    assert saved[0]["seed"] == 123
    assert submitted[0]["459:458"]["inputs"]["seed"] == 123
    assert submitted[0]["459:474"]["inputs"]["prompt"] == "expanded prompt"
    assert submitted[0]["load_ref_2"]["inputs"]["image"] == "ref.png"
    assert submitted[0]["generation_lora_1"]["inputs"]["lora_name"] == "qwen21/style.safetensors"
    assert submitted[0]["459:458"]["inputs"]["sampler_name"] == "euler"
    assert submitted[0]["459:458"]["inputs"]["scheduler"] == "simple"
    assert result["loras"] == [{"name": "qwen21/style.safetensors", "strength": 0.65}]
    assert result["sampler_name"] == "euler"
    assert result["scheduler"] == "simple"
