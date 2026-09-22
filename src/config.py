import json
import os
from typing import Any

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # Core system settings (can be overridden via env)
    comfy_url: str = Field(default="http://127.0.0.1:8001", validation_alias="COMFY_URL")
    data_dir: str = Field(default="gallery_data", validation_alias="DATA_DIR")
    comfy_root: str = Field(default="", validation_alias="COMFY_ROOT")
    openrouter_api_key: str = Field(default="", validation_alias="OPENROUTER_API_KEY")
    # Network binding. 0.0.0.0 exposes the gallery on the local network (Wi-Fi).
    host: str = Field(default="0.0.0.0", validation_alias="REIMAGINEX_HOST")
    port: int = Field(default=8888, validation_alias="REIMAGINEX_PORT")
    
    # Dynamic settings (saved to/loaded from config.json)
    system_prompt: str = """You are a prompt normalizer for Qwen Image 2.1.

Your job is to rewrite the user's request into clear, concise and natural English
suitable for image generation or image editing.

The user may write informally, with typos, poor grammar, repetition, incomplete
sentences or in Portuguese. Understand the intended request and normalize it
without changing its meaning.

Rules:

1. Preserve the user's exact intent.

2. Translate non-English input into natural English.

3. Correct grammar, spelling, ambiguity and unnecessary repetition.

4. Remove redundant or obvious instructions when they do not add useful
   information.

5. Do not make the prompt more artistic, sophisticated or descriptive than the
   user's original request. Improve precision, not creativity.

6. Do not invent details that the user did not request.

7. Do not automatically add terms such as:
   • masterpiece
   • best quality
   • 8K
   • ultra detailed
   • cinematic
   • professional photography
   • camera models
   • lenses
   • focal lengths
   • depth of field
   • dramatic lighting
   • award-winning photography
   unless the user explicitly requests them.

8. Do not invent or modify:
   • clothing
   • body features
   • facial features
   • hairstyle
   • pose
   • expression
   • objects
   • scenery
   • colors
   • lighting
   • camera angle
   • composition
   • artistic style
   unless requested by the user.

9. Prefer direct natural-language instructions instead of keyword lists.

10. For image editing:
    • State the requested change clearly and first.
    • Modify only what the user requested.
    • Preserve unrelated elements.
    • Preserve facial identity, body proportions, pose, hairstyle, background,
      framing, composition and lighting when they are not part of the requested
      change.
    • Do not redescribe the entire reference image unnecessarily.

11. For text-to-image generation:
    • Preserve all concrete details provided by the user.
    • Organize them clearly.
    • Do not add stylistic or photographic details that were not requested.

12. When multiple reference images are mentioned, preserve references such as
    <image1>, <image2>, <image3>, etc., and clearly describe the role of each
    referenced image.

13. Do not explain what you changed.

14. Do not answer the user's request conversationally.

15. Return only the final rewritten English prompt."""
    llm_provider: str = "deepseek"  # "deepseek" or "openrouter"
    deepseek_api_key: str = ""
    deepseek_model: str = "deepseek-flash"
    openrouter_models: list[str] = [
        "deepseek/deepseek-v4-flash",
        "cognitivecomputations/dolphin-mistral-24b-venice-edition:free"
    ]
    diffusion_model_name: str = "qwen_image_2.1_int8_convrot.safetensors"
    clip_model_name: str = "qwen3vl_8b_int8_convrot.safetensors"
    vae_model_name: str = "qwen_image_2.1_vae_bf16.safetensors"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

def get_config_path(data_dir: str) -> str:
    return os.path.join(data_dir, "config.json")

# Checkpoint fields that must reference a Qwen model after the Qwen Image migration.
_QWEN_CHECKPOINT_FIELDS: tuple[str, ...] = (
    "diffusion_model_name",
    "clip_model_name",
    "vae_model_name",
)

# Marker identifying the pre-Qwen (Flux) default system prompt.
_LEGACY_PROMPT_MARKER = "Flux"


def _apply_persisted_settings(settings: Settings, data: dict[str, Any]) -> bool:
    """Apply saved config values, migrating legacy pre-Qwen settings.

    Returns True when migration changed something that must be written back.
    """
    dirty = False
    for field, value in data.items():
        if not hasattr(settings, field):
            continue
        if field in _QWEN_CHECKPOINT_FIELDS and "qwen" not in str(value).lower():
            # Legacy checkpoint from before the Qwen migration: fall back to default.
            value = getattr(settings, field)
            dirty = True
        elif field == "system_prompt" and _LEGACY_PROMPT_MARKER in str(value):
            # Only the old Flux-era default prompt is upgraded; custom prompts survive.
            value = settings.system_prompt
            dirty = True
        setattr(settings, field, value)

    # Configs written before the DeepSeek provider existed need a refresh.
    if not data.get("deepseek_api_key") or not data.get("llm_provider"):
        dirty = True
    return dirty


def load_settings() -> Settings:
    settings = Settings()
    config_path = get_config_path(settings.data_dir)
    if os.path.exists(config_path):
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                data: Any = json.load(f)
            if not isinstance(data, dict):
                raise ValueError("config.json must contain a JSON object")
            if _apply_persisted_settings(settings, data):
                save_settings(settings)
        except Exception as e:
            print(f"Error loading {config_path}: {e}")
    return settings

def save_settings(settings: Settings) -> None:
    config_path = get_config_path(settings.data_dir)
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    data = {
        "system_prompt": settings.system_prompt,
        "llm_provider": settings.llm_provider,
        "deepseek_api_key": settings.deepseek_api_key,
        "deepseek_model": settings.deepseek_model,
        "openrouter_api_key": settings.openrouter_api_key,
        "openrouter_models": settings.openrouter_models,
        "diffusion_model_name": settings.diffusion_model_name,
        "clip_model_name": settings.clip_model_name,
        "vae_model_name": settings.vae_model_name,
        "comfy_url": settings.comfy_url,
        "comfy_root": settings.comfy_root,
    }
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

