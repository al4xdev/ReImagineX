import copy
import json
import math
from pathlib import Path
from typing import Any

Workflow = dict[str, Any]


def calculate_resolution(w: int, h: int, mp: float = 2.0, multiple: int = 32) -> tuple[int, int]:
    if w <= 0 or h <= 0:
        raise ValueError(f"Image dimensions must be positive, got {w}x{h}")
    scale = math.sqrt((mp * 1_000_000) / (w * h))
    # Clamp to one `multiple` step: extreme aspect ratios would otherwise round
    # a dimension down to 0 and produce an invalid latent size.
    new_w = max(multiple, round(w * scale / multiple) * multiple)
    new_h = max(multiple, round(h * scale / multiple) * multiple)
    return new_w, new_h


def _load_workflow_json(filename: str) -> Workflow:
    json_path = Path(__file__).resolve().parent.parent / filename
    with json_path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"{filename} must contain a JSON object")
    return value


def _load_workflow_base() -> Workflow:
    return _load_workflow_json("workflow_api.json")


_WORKFLOW_CACHE: Workflow | None = None


def _get_workflow_base() -> Workflow:
    global _WORKFLOW_CACHE
    if _WORKFLOW_CACHE is None:
        _WORKFLOW_CACHE = _load_workflow_base()
    return _WORKFLOW_CACHE


def _load_upscale_workflow_base() -> Workflow:
    return _load_workflow_json("workflow_upscale_api.json")


_UPSCALE_WORKFLOW_CACHE: Workflow | None = None


def _get_upscale_workflow_base() -> Workflow:
    global _UPSCALE_WORKFLOW_CACHE
    if _UPSCALE_WORKFLOW_CACHE is None:
        _UPSCALE_WORKFLOW_CACHE = _load_upscale_workflow_base()
    return _UPSCALE_WORKFLOW_CACHE


def build_generation_workflow(
    prompt: str,
    base_image_comfy_name: str,
    seed: int,
    additional_images: list[str] | None = None,
    custom_size: bool = False,
    aspect_ratio: str = "1:1 (Square)",
    megapixels: float = 1.0,
    steps: int = 25,
    cfg: float = 1.0,
    denoise: float = 1.0,
    diffusion_model_name: str = "qwen_image_2.1_int8_convrot.safetensors",
    clip_model_name: str = "qwen3vl_8b_int8_convrot.safetensors",
    vae_model_name: str = "qwen_image_2.1_vae_bf16.safetensors",
) -> Workflow:
    wf = copy.deepcopy(_get_workflow_base())

    # 1. Models
    if "459:451" in wf and "unet_name" in wf["459:451"]["inputs"]:
        wf["459:451"]["inputs"]["unet_name"] = diffusion_model_name
    if "459:453" in wf and "clip_name" in wf["459:453"]["inputs"]:
        wf["459:453"]["inputs"]["clip_name"] = clip_model_name
    if "459:454" in wf and "vae_name" in wf["459:454"]["inputs"]:
        wf["459:454"]["inputs"]["vae_name"] = vae_model_name

    # 2. Text Prompt & Seed
    if "459:474" in wf:
        wf["459:474"]["inputs"]["prompt"] = prompt
    if "459:458" in wf:
        wf["459:458"]["inputs"]["seed"] = seed
        wf["459:458"]["inputs"]["steps"] = steps
        wf["459:458"]["inputs"]["cfg"] = cfg
        wf["459:458"]["inputs"]["denoise"] = denoise

    # 3. Base Image (image 1)
    if "470" in wf:
        wf["470"]["inputs"]["image"] = base_image_comfy_name
    if "459:474" in wf:
        wf["459:474"]["inputs"]["images.image_1"] = ["470", 0]

    # 4. Additional Reference Images (images 2..10)
    if "459:474" in wf:
        text_node_inputs = wf["459:474"]["inputs"]
        for k in list(text_node_inputs.keys()):
            if k.startswith("images.image_") and k != "images.image_1":
                del text_node_inputs[k]

    if additional_images:
        for idx, img_name in enumerate(additional_images[:9], start=2):
            node_id = f"load_ref_{idx}"
            wf[node_id] = {
                "class_type": "LoadImage",
                "inputs": {"image": img_name},
                "_meta": {"title": f"Load Image Ref {idx}"}
            }
            if "459:474" in wf:
                wf["459:474"]["inputs"][f"images.image_{idx}"] = [node_id, 0]

    # 5. Custom Size / Aspect Ratio Switch
    if "459:468" in wf:
        wf["459:468"]["inputs"]["switch"] = bool(custom_size)
    if custom_size and "13" in wf:
        wf["13"]["inputs"]["aspect_ratio"] = aspect_ratio
        wf["13"]["inputs"]["megapixels"] = float(megapixels)
        wf["13"]["inputs"]["multiple"] = 32

    return wf


def build_upscale_workflow(
    base_image_comfy_name: str,
    original_width: int,
    original_height: int,
    seed: int,
    mp: float = 2.0,
    steps: int = 25,
    cfg: float = 1.0,
    denoise: float = 1.0,
    diffusion_model_name: str = "qwen_image_2.1_int8_convrot.safetensors",
    clip_model_name: str = "qwen3vl_8b_int8_convrot.safetensors",
    vae_model_name: str = "qwen_image_2.1_vae_bf16.safetensors",
) -> Workflow:
    wf = copy.deepcopy(_get_upscale_workflow_base())

    # 1. Models
    if "459:451" in wf and "unet_name" in wf["459:451"]["inputs"]:
        wf["459:451"]["inputs"]["unet_name"] = diffusion_model_name
    if "459:453" in wf and "clip_name" in wf["459:453"]["inputs"]:
        wf["459:453"]["inputs"]["clip_name"] = clip_model_name
    if "459:454" in wf and "vae_name" in wf["459:454"]["inputs"]:
        wf["459:454"]["inputs"]["vae_name"] = vae_model_name

    # 2. Base Image
    if "470" in wf:
        wf["470"]["inputs"]["image"] = base_image_comfy_name
    if "459:474" in wf:
        wf["459:474"]["inputs"]["images.image_1"] = ["470", 0]

    # 3. Calculate 2MP resolution
    new_w, new_h = calculate_resolution(original_width, original_height, mp=mp, multiple=32)
    if "459:456" in wf:
        wf["459:456"]["inputs"]["width"] = new_w
        wf["459:456"]["inputs"]["height"] = new_h

    # 4. Latent Switch (ensures on_true / empty latent at 2MP is used)
    if "459:468" in wf:
        wf["459:468"]["inputs"]["switch"] = True

    # 5. KSampler parameters
    if "459:458" in wf:
        wf["459:458"]["inputs"]["seed"] = seed
        wf["459:458"]["inputs"]["steps"] = steps
        wf["459:458"]["inputs"]["cfg"] = cfg
        wf["459:458"]["inputs"]["denoise"] = denoise

    return wf

