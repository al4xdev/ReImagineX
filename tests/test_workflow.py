from src.workflow import build_generation_workflow


def build_workflow(*, upscale_input: bool = False):
    return build_generation_workflow(
        prompt="Turn the sky violet",
        base_image_comfy_name="base.png",
        seed=42,
        upscale_input=upscale_input,
        diffusion_model_name="diffusion.safetensors",
        clip_model_name="clip.safetensors",
        vae_model_name="vae.safetensors",
        upscale_model_name="upscale.pth",
        input_upscale_model_name="input-upscale.pth",
    )


def test_workflow_injects_runtime_values_without_leaking_between_calls() -> None:
    first = build_workflow()
    first["2"]["inputs"]["text"] = "mutated"
    second = build_workflow()

    assert second["2"]["inputs"]["text"] == "Turn the sky violet"
    assert second["22"]["inputs"]["image"] == "base.png"
    assert second["15"]["inputs"]["seed"] == 42
    assert second["25"]["inputs"]["model_name"] == "diffusion.safetensors"


def test_input_upscale_adds_loader_and_rewires_reference() -> None:
    workflow = build_workflow(upscale_input=True)

    assert workflow["27"]["inputs"]["model_name"] == "input-upscale.pth"
    assert workflow["23"]["inputs"]["image"] == ["26", 0]
    assert workflow["17"]["inputs"]["image_b"] == ["26", 0]
