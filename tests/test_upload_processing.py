import asyncio
from io import BytesIO
from pathlib import Path
from typing import Any

import pytest
from PIL import Image

from src.server import prepare_uploaded_image
from src.upload_processing import MAX_UPLOAD_PIXELS, build_downscale_workflow


def png_bytes(width: int, height: int) -> bytes:
    output = BytesIO()
    Image.new("RGB", (width, height), "blue").save(output, format="PNG")
    return output.getvalue()


def test_downscale_workflow_uses_uploaded_image_and_is_independent() -> None:
    first = build_downscale_workflow("first.png")
    first["load"]["inputs"]["image"] = "changed.png"
    second = build_downscale_workflow("second.png")

    assert second["load"]["inputs"]["image"] == "second.png"
    assert second["scale"]["class_type"] == "ImageScaleToTotalPixels"
    assert second["scale"]["inputs"]["megapixels"] < 2.0
    assert second["save"]["inputs"]["images"] == ["scale", 0]


def test_small_jpeg_upload_is_saved_as_png_without_downscale(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    image = BytesIO()
    Image.new("RGB", (100, 80), "red").save(image, format="JPEG")
    filepath = tmp_path / "root" / "small.png"
    uploads: list[str] = []

    async def fake_upload_to_comfy(path: str, filename: str) -> str:
        uploads.append(filename)
        with Image.open(path) as saved:
            assert saved.format == "PNG"
            assert saved.size == (100, 80)
        return "comfy-small.png"

    monkeypatch.setattr("src.server.upload_to_comfy", fake_upload_to_comfy)
    monkeypatch.setattr("src.server.write_thumbnail", lambda _path: None)

    result = asyncio.run(prepare_uploaded_image(image.getvalue(), "small.jpg", "root/small.png", str(filepath), "image/jpeg"))

    assert result == "comfy-small.png"
    assert uploads == ["root/small.png"]


def test_large_upload_runs_comfy_downscale_and_stores_result(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    source = png_bytes(1800, 1200)
    reduced = png_bytes(1600, 1200)
    filepath = tmp_path / "root" / "large.png"
    submitted: list[dict[str, Any]] = []
    uploaded: list[str] = []

    async def fake_upload_bytes(_data: bytes, filename: str, _content_type: str) -> str:
        uploaded.append(filename)
        return "input-large.png"

    async def fake_upload_to_comfy(path: str, filename: str) -> str:
        with Image.open(path) as saved:
            assert saved.width * saved.height <= MAX_UPLOAD_PIXELS
        uploaded.append(filename)
        return "input-reduced.png"

    class FakeResponse:
        def __init__(self, payload: dict[str, Any] | None = None, content: bytes = b"") -> None:
            self.payload = payload or {}
            self.content = content

        def json(self) -> dict[str, Any]:
            return self.payload

        def raise_for_status(self) -> None:
            pass

    class FakeClient:
        def __init__(self, **_kwargs: Any) -> None:
            pass

        async def __aenter__(self) -> "FakeClient":
            return self

        async def __aexit__(self, *_args: Any) -> None:
            pass

        async def post(self, _url: str, json: dict[str, Any]) -> FakeResponse:
            submitted.append(json["prompt"])
            return FakeResponse({"prompt_id": "downscale-job"})

        async def get(self, url: str, **_kwargs: Any) -> FakeResponse:
            if "/history/" in url:
                return FakeResponse({"downscale-job": {"outputs": {"save": {"images": [
                    {"filename": "reduced.png", "subfolder": "", "type": "output"}
                ]}}}})
            return FakeResponse(content=reduced)

    monkeypatch.setattr("src.server.upload_bytes_to_comfy", fake_upload_bytes)
    monkeypatch.setattr("src.server.upload_to_comfy", fake_upload_to_comfy)
    monkeypatch.setattr("src.server.httpx.AsyncClient", FakeClient)
    monkeypatch.setattr("src.server.write_thumbnail", lambda _path: None)

    result = asyncio.run(prepare_uploaded_image(source, "large.png", "root/large.png", str(filepath), "image/png"))

    assert result == "input-reduced.png"
    assert uploaded == ["large.png", "root/large.png"]
    assert submitted[0]["load"]["inputs"]["image"] == "input-large.png"
    assert submitted[0]["scale"]["inputs"]["megapixels"] == 1.99
    assert filepath.read_bytes() == reduced
