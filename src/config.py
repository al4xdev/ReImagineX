import os
import json
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field

class Settings(BaseSettings):
    # Core system settings (can be overridden via env)
    comfy_url: str = Field(default="http://127.0.0.1:8001", validation_alias="COMFY_URL")
    data_dir: str = Field(default="gallery_data", validation_alias="DATA_DIR")
    comfy_root: str = Field(default="", validation_alias="COMFY_ROOT")
    openrouter_api_key: str = Field(
        default="",
        validation_alias="OPENROUTER_API_KEY"
    )
    
    # Dynamic settings (saved to/loaded from config.json)
    system_prompt: str = (
        "You are an expert prompt engineer for a custom Flux Klein model using image-to-image.\n"
        "Your job is to translate the user's intent to English and wrap it inside the creator's exact photographic prompt structure, without describing the woman's face or body.\n"
        "CRITICAL STRUCTURE RULES:\n"
        "- NEVER include meta-explanations, parenthetical notes, or phrases from the user like 'only difference', 'apenas isso', or 'just change this'. Convert everything into direct descriptions.\n"
        "- NEVER invent or alter her age, ethnicity, hair color (unless requested), or face.\n"
        "- Output ONLY the final English prompt string. No intros, no explanations, no quotes."
    )
    openrouter_models: list[str] = [
        "cognitivecomputations/dolphin-mistral-24b-venice-edition:free",
        "deepseek/deepseek-v4-flash"
    ]
    diffusion_model_name: str = "flux1-dev-fp8.safetensors"
    clip_model_name: str = "clip_l.safetensors"
    vae_model_name: str = "ae.safetensors"
    upscale_model_name: str = "4x_foolhardy_Remacri.pth"
    input_upscale_model_name: str = "4x_foolhardy_Remacri.pth"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

def get_config_path(data_dir: str) -> str:
    return os.path.join(data_dir, "config.json")

def load_settings() -> Settings:
    settings = Settings()
    config_path = get_config_path(settings.data_dir)
    if os.path.exists(config_path):
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                for field, val in data.items():
                    if hasattr(settings, field):
                        setattr(settings, field, val)
        except Exception as e:
            print(f"Error loading {config_path}: {e}")
    return settings

def save_settings(settings: Settings) -> None:
    config_path = get_config_path(settings.data_dir)
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
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
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
