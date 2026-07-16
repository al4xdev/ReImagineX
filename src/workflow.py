import copy
import json
from pathlib import Path
from typing import Any

Workflow = dict[str, Any]


def _load_workflow_base() -> Workflow:
    json_path = Path(__file__).resolve().parent.parent / "workflow_api.json"
    with json_path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError("workflow_api.json must contain a JSON object")
    return value


_WORKFLOW_CACHE: Workflow | None = None


def _get_workflow_base() -> Workflow:
    global _WORKFLOW_CACHE
    if _WORKFLOW_CACHE is None:
        _WORKFLOW_CACHE = _load_workflow_base()
    return _WORKFLOW_CACHE


def build_generation_workflow(
    prompt: str,
    base_image_comfy_name: str,
    seed: int,
    upscale_input: bool,
    diffusion_model_name: str,
    clip_model_name: str,
    vae_model_name: str,
    upscale_model_name: str,
    input_upscale_model_name: str,
) -> Workflow:
    wf = copy.deepcopy(_get_workflow_base())

    # Inject dynamic model configurations
    wf["25"]["inputs"]["model_name"] = diffusion_model_name
    wf["5"]["inputs"]["clip_name"] = clip_model_name
    wf["7"]["inputs"]["vae_name"] = vae_model_name
    wf["16"]["inputs"]["model_name"] = upscale_model_name

    # Basic inputs
    wf["2"]["inputs"]["text"] = prompt
    wf["22"]["inputs"]["image"] = base_image_comfy_name
    wf["15"]["inputs"]["seed"] = seed

    # Input Upscale injection
    if upscale_input:
        wf["27"] = {
            "inputs": {
                "model_name": input_upscale_model_name
            },
            "class_type": "UpscaleModelLoader"
        }
        wf["26"] = {
            "inputs": {
                "upscale_model": ["27", 0],
                "image": ["22", 0]
            },
            "class_type": "ImageUpscaleWithModel"
        }
        wf["23"]["inputs"]["image"] = ["26", 0]
        wf["17"]["inputs"]["image_b"] = ["26", 0]

    return wf
