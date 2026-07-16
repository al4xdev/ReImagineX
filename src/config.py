import json
import os
import tempfile
from pathlib import Path
from typing import Any

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # Core system settings (can be overridden via env)
    comfy_url: str = Field(default="http://127.0.0.1:8001", validation_alias="COMFY_URL")
    data_dir: str = Field(default="gallery_data", validation_alias="DATA_DIR")
    comfy_root: str = Field(default="", validation_alias="COMFY_ROOT")
    openrouter_api_key: str = Field(default="", validation_alias="OPENROUTER_API_KEY")
    
    # Dynamic settings (saved to/loaded from config.json)
    system_prompt: str = (
        "You are an expert prompt engineer for Flux image-to-image workflows.\n"
        "Translate the user's intent to English and turn it into a direct visual prompt while preserving details that the user did not ask to change.\n"
        "CRITICAL STRUCTURE RULES:\n"
        "- NEVER include meta-explanations, parenthetical notes, or phrases from the user like 'only difference', 'apenas isso', or 'just change this'. Convert everything into direct descriptions.\n"
        "- NEVER invent or alter identity, age, ethnicity, hair color, or facial features unless requested.\n"
        "- Output ONLY the final English prompt string. No intros, no explanations, no quotes."
    )
    openrouter_models: list[str] = [
        "cognitivecomputations/dolphin-mistral-24b-venice-edition:free",
        "deepseek/deepseek-v4-flash"
    ]
    diffusion_model_name: str = "flux-2-klein-9b-nvfp4.safetensors"
    clip_model_name: str = "qwen_3_8b_fp8mixed.safetensors"
    vae_model_name: str = "full_encoder_small_decoder.safetensors"
    upscale_model_name: str = "4x-UltraSharpV2.pth"
    input_upscale_model_name: str = "4x_foolhardy_Remacri.pth"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

def get_config_path(data_dir: str) -> Path:
    return Path(data_dir).expanduser() / "config.json"


def _atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise

def load_settings() -> Settings:
    settings = Settings()
    config_path = get_config_path(settings.data_dir)
    if os.path.exists(config_path):
        try:
            with config_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
                if not isinstance(data, dict):
                    raise ValueError("config.json must contain a JSON object")
                for field, val in data.items():
                    if hasattr(settings, field):
                        setattr(settings, field, val)
        except Exception as e:
            print(f"Error loading {config_path}: {e}")
    return settings

def save_settings(settings: Settings) -> None:
    config_path = get_config_path(settings.data_dir)
    data = {
        "system_prompt": settings.system_prompt,
        "openrouter_models": settings.openrouter_models,
        "diffusion_model_name": settings.diffusion_model_name,
        "clip_model_name": settings.clip_model_name,
        "vae_model_name": settings.vae_model_name,
        "upscale_model_name": settings.upscale_model_name,
        "input_upscale_model_name": settings.input_upscale_model_name,
        "comfy_url": settings.comfy_url,
        "openrouter_api_key": settings.openrouter_api_key,
        "comfy_root": settings.comfy_root,
    }
    _atomic_write_json(config_path, data)
