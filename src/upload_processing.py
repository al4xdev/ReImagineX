"""ComfyUI workflow for reducing oversized uploaded images before use."""

from typing import Any

MAX_UPLOAD_PIXELS = 2_000_000


def build_downscale_workflow(comfy_name: str) -> dict[str, Any]:
    """Save an uploaded image near 2 MP without introducing model inference."""
    return {
        "load": {"class_type": "LoadImage", "inputs": {"image": comfy_name}},
        "scale": {
            "class_type": "ImageScaleToTotalPixels",
            "inputs": {
                "image": ["load", 0],
                "upscale_method": "lanczos",
                # A small margin keeps integer rounding below the 2 MP ceiling.
                "megapixels": 1.99,
                "resolution_steps": 1,
            },
        },
        "save": {
            "class_type": "SaveImage",
            "inputs": {"images": ["scale", 0], "filename_prefix": "ReImagineX_upload"},
        },
    }
