import copy
from typing import Any

import pytest

from src.workflow import (
    DEFAULT_SAGE_ATTENTION,
    DEFAULT_UPSCALE_LORA,
    DEFAULT_UPSCALE_LORA_STRENGTH,
    DEFAULT_UPSCALE_PROMPT,
    _get_upscale_workflow_base,
    _get_workflow_base,
    build_generation_workflow,
    build_upscale_workflow,
    calculate_resolution,
)

DEFAULT_PROMPT = "Turn the sky violet"

# Real node ids in workflow_api.json / workflow_upscale_api.json for the
# "Qwen Image 2.1" graphs.
PROMPT_NODE = "459:474"  # TextEncodeQwenImage21
SAMPLER_NODE = "459:458"  # KSampler
UNET_NODE = "459:451"  # UNETLoader
CLIP_NODE = "459:453"  # CLIPLoader
VAE_NODE = "459:454"  # VAELoader
LATENT_NODE = "459:456"  # EmptyLatentImage
SWITCH_NODE = "459:468"  # ComfySwitchNode
SELECTOR_NODE = "13"  # ResolutionSelector
BASE_IMAGE_NODE = "470"  # LoadImage
LORA_NODE = "459:475"  # LoraLoader
SAGE_NODE = "459:476"  # PathchSageAttentionKJ
CACHE_NODE = "459:469"  # QwenImage21Cache


def _build_generation(**overrides: Any) -> dict[str, Any]:
    params: dict[str, Any] = {
        "prompt": DEFAULT_PROMPT,
        "base_image_comfy_name": "base.png",
        "seed": 42,
        "steps": 30,
        "cfg": 5.5,
        "denoise": 0.8,
        "diffusion_model_name": "diffusion.safetensors",
        "clip_model_name": "clip.safetensors",
        "vae_model_name": "vae.safetensors",
    }
    params.update(overrides)
    return build_generation_workflow(**params)


def test_generation_workflow_does_not_leak_mutations_into_next_call_or_cache() -> None:
    cache_before = copy.deepcopy(_get_workflow_base())

    first = _build_generation(
        additional_images=["ref-a.png", "ref-b.png"],
        custom_size=True,
        aspect_ratio="16:9 (Widescreen)",
        megapixels=2.0,
    )
    # Deeply mutate the returned structure: strings, ints, wiring, extra dynamic
    # nodes and a removed node must all be confined to this call.
    first[PROMPT_NODE]["inputs"]["prompt"] = "MUTATED"
    first[SAMPLER_NODE]["inputs"]["seed"] = -1
    first[UNET_NODE]["inputs"]["unet_name"] = "mutated.safetensors"
    first[BASE_IMAGE_NODE]["inputs"]["image"] = "mutated.png"
    first["load_ref_2"]["inputs"]["image"] = "mutated-ref.png"
    first[SELECTOR_NODE]["inputs"]["megapixels"] = 99.0
    del first[CLIP_NODE]

    second = _build_generation()

    assert second[PROMPT_NODE]["inputs"]["prompt"] == DEFAULT_PROMPT
    assert second[SAMPLER_NODE]["inputs"]["seed"] == 42
    assert second[UNET_NODE]["inputs"]["unet_name"] == "diffusion.safetensors"
    assert second[BASE_IMAGE_NODE]["inputs"]["image"] == "base.png"
    assert second[CLIP_NODE]["inputs"]["clip_name"] == "clip.safetensors"
    assert second[SELECTOR_NODE]["inputs"]["megapixels"] == 1.0
    assert _get_workflow_base() == cache_before


def test_generation_workflow_injects_prompt_and_sampler_settings_into_real_nodes() -> None:
    workflow = _build_generation(prompt="A red fox on a rock", seed=1234, steps=12, cfg=3.75, denoise=0.6)

    prompt_node = workflow[PROMPT_NODE]
    assert prompt_node["class_type"] == "TextEncodeQwenImage21"
    assert prompt_node["inputs"]["prompt"] == "A red fox on a rock"

    sampler = workflow[SAMPLER_NODE]
    assert sampler["class_type"] == "KSampler"
    assert sampler["inputs"]["seed"] == 1234
    assert sampler["inputs"]["steps"] == 12
    assert sampler["inputs"]["cfg"] == 3.75
    assert sampler["inputs"]["denoise"] == 0.6
    # Settings this function does not own must survive untouched.
    assert sampler["inputs"]["sampler_name"] == "euler"
    assert sampler["inputs"]["scheduler"] == "simple"


def test_generation_workflow_injects_model_names_into_loader_nodes() -> None:
    workflow = _build_generation(
        diffusion_model_name="custom-diffusion.safetensors",
        clip_model_name="custom-clip.safetensors",
        vae_model_name="custom-vae.safetensors",
    )

    assert workflow[UNET_NODE]["class_type"] == "UNETLoader"
    assert workflow[UNET_NODE]["inputs"]["unet_name"] == "custom-diffusion.safetensors"
    assert workflow[CLIP_NODE]["class_type"] == "CLIPLoader"
    assert workflow[CLIP_NODE]["inputs"]["clip_name"] == "custom-clip.safetensors"
    assert workflow[VAE_NODE]["class_type"] == "VAELoader"
    assert workflow[VAE_NODE]["inputs"]["vae_name"] == "custom-vae.safetensors"
    # Loader-specific settings from the shipped graph are preserved.
    assert workflow[CLIP_NODE]["inputs"]["type"] == "qwen_image"


def test_generation_workflow_wires_base_image_into_load_and_text_encode() -> None:
    workflow = _build_generation(base_image_comfy_name="uploaded-base.png")

    loader = workflow[BASE_IMAGE_NODE]
    assert loader["class_type"] == "LoadImage"
    assert loader["inputs"]["image"] == "uploaded-base.png"
    assert workflow[PROMPT_NODE]["inputs"]["images.image_1"] == [BASE_IMAGE_NODE, 0]


def test_generation_workflow_creates_load_ref_nodes_for_additional_images() -> None:
    workflow = _build_generation(additional_images=["ref-1.png", "ref-2.png"])
    text_inputs = workflow[PROMPT_NODE]["inputs"]

    assert workflow["load_ref_2"]["class_type"] == "LoadImage"
    assert workflow["load_ref_2"]["inputs"]["image"] == "ref-1.png"
    assert workflow["load_ref_3"]["class_type"] == "LoadImage"
    assert workflow["load_ref_3"]["inputs"]["image"] == "ref-2.png"
    assert text_inputs["images.image_2"] == ["load_ref_2", 0]
    assert text_inputs["images.image_3"] == ["load_ref_3", 0]
    assert text_inputs["images.image_1"] == [BASE_IMAGE_NODE, 0]


def test_generation_workflow_caps_additional_images_at_nine_refs() -> None:
    workflow = _build_generation(additional_images=[f"ref-{i}.png" for i in range(12)])
    text_inputs = workflow[PROMPT_NODE]["inputs"]

    ref_ids = {node_id for node_id in workflow if node_id.startswith("load_ref_")}
    assert ref_ids == {f"load_ref_{i}" for i in range(2, 11)}
    assert workflow["load_ref_10"]["inputs"]["image"] == "ref-8.png"
    assert "images.image_11" not in text_inputs


def test_generation_workflow_without_additional_images_has_no_leaked_ref_nodes() -> None:
    with_refs = _build_generation(additional_images=["ref-1.png", "ref-2.png"])
    assert "load_ref_2" in with_refs  # guard: the previous call really created refs

    without_refs = _build_generation()

    assert not [node_id for node_id in without_refs if node_id.startswith("load_ref_")]
    text_inputs = without_refs[PROMPT_NODE]["inputs"]
    assert "images.image_2" not in text_inputs
    assert "images.image_3" not in text_inputs
    assert text_inputs["images.image_1"] == [BASE_IMAGE_NODE, 0]


def test_generation_workflow_custom_size_on_sets_switch_and_selector() -> None:
    workflow = _build_generation(
        custom_size=True,
        aspect_ratio="16:9 (Widescreen)",
        megapixels=2.5,
    )

    assert workflow[SWITCH_NODE]["class_type"] == "ComfySwitchNode"
    assert workflow[SWITCH_NODE]["inputs"]["switch"] is True
    selector = workflow[SELECTOR_NODE]
    assert selector["class_type"] == "ResolutionSelector"
    assert selector["inputs"]["aspect_ratio"] == "16:9 (Widescreen)"
    assert selector["inputs"]["megapixels"] == 2.5
    assert selector["inputs"]["multiple"] == 32


def test_generation_workflow_custom_size_off_forces_switch_false_and_keeps_selector(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The shipped graph defaults the switch to False, so flip it in a copy to
    # prove the function actively writes the value rather than inheriting it.
    base = copy.deepcopy(_get_workflow_base())
    base[SWITCH_NODE]["inputs"]["switch"] = True
    monkeypatch.setattr("src.workflow._get_workflow_base", lambda: base)

    workflow = _build_generation(custom_size=False)

    assert workflow[SWITCH_NODE]["inputs"]["switch"] is False
    selector = workflow[SELECTOR_NODE]
    assert selector["inputs"]["aspect_ratio"] == "1:1 (Square)"
    assert selector["inputs"]["megapixels"] == 1.0


def test_upscale_workflow_sets_target_resolution_and_latent_switch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The shipped upscale graph already sets switch=True, so force it False in a
    # copy to prove build_upscale_workflow writes the switch itself.
    base = copy.deepcopy(_get_upscale_workflow_base())
    base[SWITCH_NODE]["inputs"]["switch"] = False
    monkeypatch.setattr("src.workflow._get_upscale_workflow_base", lambda: base)

    workflow = build_upscale_workflow(
        base_image_comfy_name="upscale-input.png",
        original_width=1000,
        original_height=1000,
        seed=7,
        mp=2.0,
        steps=18,
        cfg=2.5,
        denoise=0.35,
        diffusion_model_name="d.safetensors",
        clip_model_name="c.safetensors",
        vae_model_name="v.safetensors",
    )

    latent = workflow[LATENT_NODE]
    assert latent["class_type"] == "EmptyLatentImage"
    assert (latent["inputs"]["width"], latent["inputs"]["height"]) == (1408, 1408)
    assert latent["inputs"]["width"] % 32 == 0
    assert latent["inputs"]["height"] % 32 == 0

    assert workflow[SWITCH_NODE]["inputs"]["switch"] is True

    sampler = workflow[SAMPLER_NODE]
    assert sampler["class_type"] == "KSampler"
    assert sampler["inputs"]["seed"] == 7
    assert sampler["inputs"]["steps"] == 18
    assert sampler["inputs"]["cfg"] == 2.5
    assert sampler["inputs"]["denoise"] == 0.35

    assert workflow[BASE_IMAGE_NODE]["inputs"]["image"] == "upscale-input.png"
    assert workflow[PROMPT_NODE]["inputs"]["images.image_1"] == [BASE_IMAGE_NODE, 0]
    assert workflow[PROMPT_NODE]["inputs"]["prompt"] == DEFAULT_UPSCALE_PROMPT
    assert workflow[UNET_NODE]["inputs"]["unet_name"] == "d.safetensors"
    assert workflow[CLIP_NODE]["inputs"]["clip_name"] == "c.safetensors"
    assert workflow[VAE_NODE]["inputs"]["vae_name"] == "v.safetensors"

    # LoRA & SageAttention injection defaults
    assert workflow[LORA_NODE]["class_type"] == "LoraLoader"
    assert workflow[LORA_NODE]["inputs"]["lora_name"] == DEFAULT_UPSCALE_LORA
    assert workflow[LORA_NODE]["inputs"]["strength_model"] == DEFAULT_UPSCALE_LORA_STRENGTH
    assert workflow[LORA_NODE]["inputs"]["strength_clip"] == DEFAULT_UPSCALE_LORA_STRENGTH
    assert workflow[LORA_NODE]["inputs"]["model"] == [UNET_NODE, 0]
    assert workflow[LORA_NODE]["inputs"]["clip"] == [CLIP_NODE, 0]

    assert workflow[SAGE_NODE]["class_type"] == "PathchSageAttentionKJ"
    assert workflow[SAGE_NODE]["inputs"]["model"] == [LORA_NODE, 0]
    assert workflow[SAGE_NODE]["inputs"]["sage_attention"] == DEFAULT_SAGE_ATTENTION
    assert workflow[CACHE_NODE]["inputs"]["model"] == [SAGE_NODE, 0]
    assert workflow[PROMPT_NODE]["inputs"]["clip"] == [LORA_NODE, 1]


def test_upscale_workflow_custom_lora_and_prompt() -> None:
    workflow = build_upscale_workflow(
        base_image_comfy_name="custom.png",
        original_width=512,
        original_height=512,
        seed=99,
        lora_name="Custom_Slider.safetensors",
        lora_strength=-2.5,
        prompt="Custom high resolution restoration prompt",
    )

    assert workflow[LORA_NODE]["inputs"]["lora_name"] == "Custom_Slider.safetensors"
    assert workflow[LORA_NODE]["inputs"]["strength_model"] == -2.5
    assert workflow[LORA_NODE]["inputs"]["strength_clip"] == -2.5
    assert workflow[PROMPT_NODE]["inputs"]["prompt"] == "Custom high resolution restoration prompt"
    assert workflow[SAGE_NODE]["inputs"]["model"] == [LORA_NODE, 0]
    assert workflow[CACHE_NODE]["inputs"]["model"] == [SAGE_NODE, 0]


def test_upscale_workflow_bypasses_lora_keeps_sage() -> None:
    workflow = build_upscale_workflow(
        base_image_comfy_name="bypass.png",
        original_width=512,
        original_height=512,
        seed=1,
        lora_strength=0.0,
    )

    assert LORA_NODE not in workflow
    assert workflow[SAGE_NODE]["inputs"]["model"] == [UNET_NODE, 0]
    assert workflow[CACHE_NODE]["inputs"]["model"] == [SAGE_NODE, 0]
    assert workflow[PROMPT_NODE]["inputs"]["clip"] == [CLIP_NODE, 0]


def test_upscale_workflow_bypasses_sage_when_disabled() -> None:
    workflow = build_upscale_workflow(
        base_image_comfy_name="bypass_sage.png",
        original_width=512,
        original_height=512,
        seed=1,
        sage_attention="disabled",
    )

    assert SAGE_NODE not in workflow
    assert workflow[CACHE_NODE]["inputs"]["model"] == [LORA_NODE, 0]


def test_generation_workflow_injects_sage_attention() -> None:
    workflow = _build_generation()

    assert workflow[SAGE_NODE]["class_type"] == "PathchSageAttentionKJ"
    assert workflow[SAGE_NODE]["inputs"]["model"] == [UNET_NODE, 0]
    assert workflow[SAGE_NODE]["inputs"]["sage_attention"] == "auto"
    assert workflow[CACHE_NODE]["inputs"]["model"] == [SAGE_NODE, 0]


def test_generation_workflow_bypasses_sage_when_disabled() -> None:
    workflow = _build_generation(sage_attention="disabled")

    assert SAGE_NODE not in workflow
    assert workflow[CACHE_NODE]["inputs"]["model"] == [UNET_NODE, 0]


def test_upscale_workflow_tracks_non_square_original_aspect_ratio() -> None:
    workflow = build_upscale_workflow(
        base_image_comfy_name="wide.png",
        original_width=800,
        original_height=600,
        seed=1,
        mp=2.0,
    )

    latent = workflow[LATENT_NODE]
    assert (latent["inputs"]["width"], latent["inputs"]["height"]) == (1632, 1216)
    assert latent["inputs"]["width"] > latent["inputs"]["height"]


def test_calculate_resolution_hits_target_megapixels_on_multiples_of_32() -> None:
    width, height = calculate_resolution(1000, 1000, mp=2.0, multiple=32)

    assert (width, height) == (1408, 1408)
    assert width % 32 == 0
    assert height % 32 == 0
    assert abs(width * height - 2_000_000) / 2_000_000 < 0.02


def test_calculate_resolution_preserves_aspect_ratio_within_rounding() -> None:
    width, height = calculate_resolution(1920, 1080, mp=2.0, multiple=32)

    assert (width, height) == (1888, 1056)
    assert width % 32 == 0
    assert height % 32 == 0
    assert abs(width / height - 1920 / 1080) < 0.02


def test_calculate_resolution_rejects_non_positive_dimensions() -> None:
    with pytest.raises(ValueError):
        calculate_resolution(0, 100)
    with pytest.raises(ValueError):
        calculate_resolution(100, -1)


def test_calculate_resolution_never_returns_a_zero_dimension() -> None:
    # An extreme aspect ratio used to round the short side down to 0, which
    # ComfyUI rejects as an invalid latent size.
    width, height = calculate_resolution(10000, 1, mp=1.0, multiple=32)

    assert width % 32 == 0
    assert height == 32
    assert height > 0
