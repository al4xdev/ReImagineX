import copy

WORKFLOW_BASE = {
  "1": {
    "inputs": {
      "cfg": 1,
      "model": ["25", 0],
      "positive": ["19", 0],
      "negative": ["21", 0]
    },
    "class_type": "CFGGuider"
  },
  "2": {
    "inputs": {
      "text": "PROMPT_HERE",
      "clip": ["5", 0]
    },
    "class_type": "CLIPTextEncode"
  },
  "3": {
    "inputs": {
      "filename_prefix": "Flux2-Klein-Edit",
      "images": ["18", 0]
    },
    "class_type": "SaveImage"
  },
  "4": {
    "inputs": {
      "sampler_name": "euler"
    },
    "class_type": "KSamplerSelect"
  },
  "5": {
    "inputs": {
      "clip_name": "qwen_3_8b_fp8mixed.safetensors",
      "type": "flux2",
      "device": "default"
    },
    "class_type": "CLIPLoader"
  },
  "6": {
    "inputs": {
      "steps": 4,
      "width": ["9", 0],
      "height": ["9", 1]
    },
    "class_type": "Flux2Scheduler"
  },
  "7": {
    "inputs": {
      "vae_name": "full_encoder_small_decoder.safetensors"
    },
    "class_type": "VAELoader"
  },
  "8": {
    "inputs": {
      "value": 1
    },
    "class_type": "PrimitiveInt"
  },
  "9": {
    "inputs": {
      "image": ["23", 0]
    },
    "class_type": "GetImageSize"
  },
  "10": {
    "inputs": {
      "noise": ["11", 0],
      "guider": ["1", 0],
      "sampler": ["4", 0],
      "sigmas": ["6", 0],
      "latent_image": ["12", 0]
    },
    "class_type": "SamplerCustomAdvanced"
  },
  "11": {
    "inputs": {
      "noise_seed": ["15", 0]
    },
    "class_type": "RandomNoise"
  },
  "12": {
    "inputs": {
      "width": ["9", 0],
      "height": ["9", 1],
      "batch_size": ["8", 0]
    },
    "class_type": "EmptyFlux2LatentImage"
  },
  "13": {
    "inputs": {
      "samples": ["10", 0],
      "vae": ["7", 0]
    },
    "class_type": "VAEDecode"
  },
  "14": {
    "inputs": {
      "pixels": ["23", 0],
      "vae": ["7", 0]
    },
    "class_type": "VAEEncode"
  },
  "15": {
    "inputs": {
      "seed": 0
    },
    "class_type": "Seed (rgthree)"
  },
  "16": {
    "inputs": {
      "model_name": "4x_foolhardy_Remacri.pth"
    },
    "class_type": "UpscaleModelLoader"
  },
  "17": {
    "inputs": {
      "rgthree_comparer": {
        "images": [
          {"name": "A", "selected": True, "url": ""},
          {"name": "B", "selected": True, "url": ""}
        ]
      },
      "image_a": ["18", 0],
      "image_b": ["22", 0]
    },
    "class_type": "Image Comparer (rgthree)"
  },
  "18": {
    "inputs": {
      "upscale_model": ["16", 0],
      "image": ["13", 0]
    },
    "class_type": "ImageUpscaleWithModel"
  },
  "19": {
    "inputs": {
      "conditioning": ["2", 0],
      "latent": ["14", 0]
    },
    "class_type": "ReferenceLatent"
  },
  "20": {
    "inputs": {
      "conditioning": ["2", 0]
    },
    "class_type": "ConditioningZeroOut"
  },
  "21": {
    "inputs": {
      "conditioning": ["20", 0],
      "latent": ["14", 0]
    },
    "class_type": "ReferenceLatent"
  },
  "22": {
    "inputs": {
      "image": "BASE_IMAGE_HERE"
    },
    "class_type": "LoadImage"
  },
  "23": {
    "inputs": {
      "upscale_method": "nearest-exact",
      "megapixels": 1,
      "resolution_steps": 1,
      "image": ["22", 0]
    },
    "class_type": "ImageScaleToTotalPixels"
  },
  "24": {
    "inputs": {
      "upscale_method": "nearest-exact",
      "megapixels": 1,
      "resolution_steps": 1
    },
    "class_type": "ImageScaleToTotalPixels"
  },
  "25": {
    "inputs": {
      "model_name": "miracleinNSFWGeneration_30Bf16Fp8.safetensors",
      "weight_dtype": "default",
      "compute_dtype": "default",
      "patch_cublaslinear": False,
      "sage_attention": "sageattn3",
      "enable_fp16_accumulation": False
    },
    "class_type": "DiffusionModelLoaderKJ"
  }
}

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
    wf = copy.deepcopy(WORKFLOW_BASE)

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
        # Node 27: UpscaleModelLoader for input upscale model
        wf["27"] = {
            "inputs": {
                "model_name": input_upscale_model_name
            },
            "class_type": "UpscaleModelLoader"
        }
        # Node 26: ImageUpscaleWithModel applying model to the loaded image (Node 22)
        wf["26"] = {
            "inputs": {
                "upscale_model": ["27", 0],
                "image": ["22", 0]
            },
            "class_type": "ImageUpscaleWithModel"
        }
        # Redirect downstream dependencies of LoadImage node 22 to the upscaled image node 26
        wf["23"]["inputs"]["image"] = ["26", 0]
        wf["17"]["inputs"]["image_b"] = ["26", 0]
        
    return wf
