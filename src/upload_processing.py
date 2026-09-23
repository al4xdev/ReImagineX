"""ComfyUI workflow for reducing oversized uploaded images before use."""

import math
from typing import Any

from src.workflow import calculate_resolution

MAX_UPLOAD_PIXELS = 2_000_000
MAX_COMFY_DIMENSION = 16_384


def downscale_dimensions(width: int, height: int) -> tuple[int, int]:
    """Fit an image within 2 million pixels and ComfyUI's dimension limit."""
    if width <= 0 or height <= 0:
        raise ValueError("Image dimensions must be positive")
    if width * height <= MAX_UPLOAD_PIXELS and max(width, height) <= MAX_COMFY_DIMENSION:
        return width, height
    target_width, target_height = calculate_resolution(width, height, mp=2.0, multiple=1)
    scale = min(
        1.0,
        MAX_COMFY_DIMENSION / target_width,
        MAX_COMFY_DIMENSION / target_height,
        math.sqrt(MAX_UPLOAD_PIXELS / (target_width * target_height)),
    )
    return max(1, math.floor(target_width * scale)), max(1, math.floor(target_height * scale))


def build_downscale_workflow(comfy_name: str, width: int, height: int) -> dict[str, Any]:
    """Save an uploaded image below 2 million pixels without model inference."""
    target_width, target_height = downscale_dimensions(width, height)
    return {
        "load": {"class_type": "LoadImage", "inputs": {"image": comfy_name}},
        "scale": {
            "class_type": "ImageScale",
            "inputs": {
                "image": ["load", 0],
                "upscale_method": "lanczos",
                "width": target_width,
                "height": target_height,
                "crop": "disabled",
            },
        },
        "save": {
            "class_type": "SaveImage",
            "inputs": {"images": ["scale", 0], "filename_prefix": "ReImagineX_upload"},
        },
    }
