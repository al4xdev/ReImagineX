import copy
import json
import os


def _load_workflow_base() -> dict:
    json_path = os.path.join(os.path.dirname(__file__), "..", "workflow_api.json")
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


_WORKFLOW_CACHE: dict | None = None


def _get_workflow_base() -> dict:
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
    input_upscale_model_name: str
) -> dict:
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
