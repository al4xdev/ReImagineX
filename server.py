import json
import os
import copy
import uuid
import asyncio
from contextlib import asynccontextmanager
from pydantic import BaseModel
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import httpx
import uvicorn

# ── Configurações ────────────────────────────────────────────────────────────
COMFY_URL = "http://127.0.0.1:8001"
DATA_DIR  = "galeria_dados"
DB_FILE   = os.path.join(DATA_DIR, "state.json")
IMG_DIR   = os.path.join(DATA_DIR, "imagens")
os.makedirs(IMG_DIR, exist_ok=True)
OPENROUTER_API_KEY = ""
state_lock = asyncio.Lock()

# ── Resolução alvo ────────────────────────────────────────────────────────────
# Aspect ratio 9:20 (1080×2400) arredondado para múltiplos de 64
# 768×1664 ≈ 1.28MP — bom balanço qualidade/VRAM para Flux
TARGET_W = 832
TARGET_H = 1248

# ── Workflow Base ─────────────────────────────────────────────────────────────
WORKFLOW_BASE = {
  "1": {
    "inputs": {
      "cfg": 1,
      "model": ["33", 0],
      "positive": ["21", 0],
      "negative": ["23", 0]
    },
    "class_type": "CFGGuider"
  },
  "2": {
    "inputs": {
      "text": "PROMPT_AQUI",
      "clip": ["33", 1]
    },
    "class_type": "CLIPTextEncode"
  },
  "3": {
    "inputs": {
      "filename_prefix": "Flux2-Klein-Edit",
      "images": ["20", 0]
    },
    "class_type": "SaveImage"
  },
  "5": {
    "inputs": {
      "sampler_name": "euler"
    },
    "class_type": "KSamplerSelect"
  },
  "6": {
    "inputs": {
      "clip_name": "qwen_3_8b_fp8mixed.safetensors",
      "type": "flux2",
      "device": "default"
    },
    "class_type": "CLIPLoader"
  },
  "7": {
    "inputs": {
      "steps": 12,
      "width": ["10", 0],
      "height": ["10", 1]
    },
    "class_type": "Flux2Scheduler"
  },
  "8": {
    "inputs": {
      "vae_name": "flux2-vae.safetensors"
    },
    "class_type": "VAELoader"
  },
  "9": {
    "inputs": {
      "value": 1
    },
    "class_type": "PrimitiveInt"
  },
  "10": {
    "inputs": {
      "image": ["30", 0]
    },
    "class_type": "GetImageSize"
  },
  "11": {
    "inputs": {
      "noise": ["12", 0],
      "guider": ["1", 0],
      "sampler": ["5", 0],
      "sigmas": ["7", 0],
      "latent_image": ["13", 0]
    },
    "class_type": "SamplerCustomAdvanced"
  },
  "12": {
    "inputs": {
      "noise_seed": ["16", 0]
    },
    "class_type": "RandomNoise"
  },
  "13": {
    "inputs": {
      "width": ["10", 0],
      "height": ["10", 1],
      "batch_size": ["9", 0]
    },
    "class_type": "EmptyFlux2LatentImage"
  },
  "14": {
    "inputs": {
      "samples": ["11", 0],
      "vae": ["8", 0]
    },
    "class_type": "VAEDecode"
  },
  "15": {
    "inputs": {
      "pixels": ["30", 0],
      "vae": ["8", 0]
    },
    "class_type": "VAEEncode"
  },
  "16": {
    "inputs": {
      "seed": 0
    },
    "class_type": "Seed (rgthree)"
  },
  "18": {
    "inputs": {
      "model_name": "4x-UltraSharp.pth"
    },
    "class_type": "UpscaleModelLoader"
  },
  "19": {
    "inputs": {
      "rgthree_comparer": {
        "images": [
          {"name": "A", "selected": True, "url": ""},
          {"name": "B", "selected": True, "url": ""}
        ]
      },
      "image_a": ["20", 0],
      "image_b": ["25", 0]
    },
    "class_type": "Image Comparer (rgthree)"
  },
  "20": {
    "inputs": {
      "upscale_model": ["18", 0],
      "image": ["14", 0]
    },
    "class_type": "ImageUpscaleWithModel"
  },
  "21": {
    "inputs": {
      "conditioning": ["2", 0],
      "latent": ["15", 0]
    },
    "class_type": "ReferenceLatent"
  },
  "22": {
    "inputs": {
      "conditioning": ["2", 0]
    },
    "class_type": "ConditioningZeroOut"
  },
  "23": {
    "inputs": {
      "conditioning": ["22", 0],
      "latent": ["15", 0]
    },
    "class_type": "ReferenceLatent"
  },
  "25": {
    "inputs": {
      "image": "IMAGEM_BASE_AQUI"
    },
    "class_type": "LoadImage"
  },
  "30": {
    "inputs": {
      "upscale_method": "nearest-exact",
      "megapixels": 1,
      "resolution_steps": 1,
      "image": ["25", 0]
    },
    "class_type": "ImageScaleToTotalPixels"
  },
  "31": {
    "inputs": {
      "upscale_method": "nearest-exact",
      "megapixels": 1,
      "resolution_steps": 1
    },
    "class_type": "ImageScaleToTotalPixels"
  },
  "33": {
    "inputs": {
      "lora_name": "Realism_Engine_Klein_V2.safetensors",
      "strength_model": 1,
      "strength_clip": 1,
      "model": ["34", 0],
      "clip": ["6", 0]
    },
    "class_type": "LoraLoader"
  },
  "34": {
    "inputs": {
      "model_name": "miracleinNSFWGeneration_30Bf16Fp8.safetensors",
      "weight_dtype": "fp8_e4m3fn",
      "compute_dtype": "default",
      "patch_cublaslinear": False,
      "sage_attention": "sageattn3",
      "enable_fp16_accumulation": False
    },
    "class_type": "DiffusionModelLoaderKJ"
  }
}

# ── Estado ────────────────────────────────────────────────────────────────────
def _load():
    if os.path.exists(DB_FILE):
        with open(DB_FILE, "r", encoding="utf-8") as f:
            try: return json.load(f)
            except: return []
    return []

def _save(state):
    with open(DB_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, ensure_ascii=False)

async def load_state():
    async with state_lock: return _load()

async def save_state(state):
    async with state_lock: _save(state)

# ── ComfyUI ───────────────────────────────────────────────────────────────────
async def upload_to_comfy(filepath: str, filename: str) -> str:
    async with httpx.AsyncClient() as client:
        with open(filepath, "rb") as f:
            res = await client.post(f"{COMFY_URL}/upload/image", files={"image": (filename, f, "image/png")})
            res.raise_for_status()
            return res.json()["name"]

async def check_comfy_queue():
    while True:
        try:
            state = await load_state()
            pendentes = [i for i in state if i["status"] == "pending"]
            if pendentes:
                async with httpx.AsyncClient(timeout=10.0) as client:
                    history = (await client.get(f"{COMFY_URL}/history")).json()
                    changed = False
                    for item in pendentes:
                        pid = item["prompt_id"]
                        if pid in history:
                            outputs = history[pid].get("outputs", {})
                            if "3" in outputs and "images" in outputs["3"]:
                                img_data = outputs["3"]["images"][0]
                                url = f"{COMFY_URL}/view?filename={img_data['filename']}&subfolder={img_data['subfolder']}&type={img_data['type']}"
                                res_img = await client.get(url)
                                local_fn = f"{item['id']}.png"
                                local_path = os.path.join(IMG_DIR, local_fn)
                                with open(local_path, "wb") as f: f.write(res_img.content)
                                comfy_name = await upload_to_comfy(local_path, local_fn)
                                item["status"] = "completed"
                                item["filename"] = local_fn
                                item["comfyName"] = comfy_name
                                changed = True
                    if changed: await save_state(state)
        except httpx.ConnectError: pass
        except Exception as e: print(f"Polling erro: {e}")
        await asyncio.sleep(1.5)

@asynccontextmanager
async def lifespan(app: FastAPI):
    task = asyncio.create_task(check_comfy_queue())
    yield
    task.cancel()
    try: await task
    except asyncio.CancelledError: pass

app = FastAPI(lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/api/state")
async def get_state(): return await load_state()

@app.post("/api/upload")
async def handle_upload(file: UploadFile = File(...)):
    item_id  = str(uuid.uuid4())
    filename = f"{item_id}.png"
    filepath = os.path.join(IMG_DIR, filename)
    with open(filepath, "wb") as f: f.write(await file.read())
    comfy_name = await upload_to_comfy(filepath, filename)
    new_item = {"id": item_id, "status": "completed", "filename": filename,
                "comfyName": comfy_name, "prompt_id": None, "prompt": "Upload Inicial"}
    state = await load_state()
    state.insert(0, new_item)
    await save_state(state)
    return new_item

class PromptRequest(BaseModel):
    base_id: str
    prompt: str

async def expandir_prompt_openrouter(prompt_usuario: str) -> str:
    # Lista de modelos free/descensurados em ordem de prioridade
    modelos_fallback = [
        "cognitivecomputations/dolphin-mistral-24b-venice-edition:free",
        "deepseek/deepseek-v4-flash",
        "deepseek/deepseek-v4-flash"
    ]
    system_instruction = (
        "You are an expert prompt engineer for Flux image-to-image modifications.\n"
        "The user wants to make minor changes to a reference image containing a specific woman.\n"
        "Your job is to output a short, raw description in English focusing ONLY on the changes.\n"
        "CRITICAL RULES:\n"
        "- Structure example: 'The woman from the reference image, now with pink hair, lying on a bed, soft bedroom lighting.'\n"
        "- Output ONLY the final English string. No intros, no quotes, no explanations."
  )
    async with httpx.AsyncClient(timeout=12.0) as client:
        for modelo in modelos_fallback:
            try:
                print(f"Tentando otimizar prompt com o modelo: {modelo}")
                res = await client.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {OPENROUTER_API_KEY}".strip(),
                        "Content-Type": "application/json",
                        "HTTP-Referer": "http://localhost:8888",
                        "X-Title": "ComfyUI Gallery Proxy"
                    },
                    json={
                        "model": modelo,
                        "messages": [
                            {"role": "system", "content": system_instruction},
                            {"role": "user", "content": prompt_usuario}
                        ],
                        "temperature": 0.7
                    }
                )
                
                if res.status_code == 200:
                    prompt_otimizado = res.json()["choices"][0]["message"]["content"].strip()
                    print(f"Sucesso com o modelo [{modelo}]: {prompt_otimizado}")
                    return prompt_otimizado
                else:
                    print(f"Modelo [{modelo}] falhou com status {res.status_code}. Tentando próximo...")
            except Exception as e:
                print(f"Erro ao conectar com o modelo [{modelo}]: {e}. Tentando próximo...")
                
    print("Todos os modelos do OpenRouter falharam. Usando o prompt original do usuário.")
    return prompt_usuario

@app.post("/api/generate")
async def generate(req: PromptRequest):
    state     = await load_state()
    base_item = next((i for i in state if i["id"] == req.base_id), None)
    if not base_item or not base_item.get("comfyName"):
        raise HTTPException(status_code=400, detail="Imagem base inválida.")
        
    # Executa a busca em cascata de modelos
    prompt_final = await expandir_prompt_openrouter(req.prompt)

    # ── Injeção no Workflow do ComfyUI ────────────────────────────────────────
    wf = copy.deepcopy(WORKFLOW_BASE)
    wf["2"]["inputs"]["text"]  = prompt_final
    wf["25"]["inputs"]["image"] = base_item["comfyName"]
    wf["16"]["inputs"]["seed"]  = int.from_bytes(os.urandom(4), byteorder="little")
    
    async with httpx.AsyncClient() as client:
        res = await client.post(f"{COMFY_URL}/prompt", json={"prompt": wf})
        if res.status_code != 200:
            raise HTTPException(status_code=500, detail="Erro ao enviar ao ComfyUI")
        prompt_id = res.json()["prompt_id"]
        
    new_item = {"id": str(uuid.uuid4()), "status": "pending", "filename": None,
                "comfyName": None, "prompt_id": prompt_id, "prompt": prompt_final}
                
    state.insert(0, new_item)
    await save_state(state)
    return new_item

# @app.post("/api/generate")
# async def generate(req: PromptRequest):
#     state     = await load_state()
#     base_item = next((i for i in state if i["id"] == req.base_id), None)
#     if not base_item or not base_item.get("comfyName"):
#         raise HTTPException(status_code=400, detail="Imagem base inválida.")
#     wf = copy.deepcopy(WORKFLOW_BASE)
#     wf["2"]["inputs"]["text"]  = req.prompt
#     wf["25"]["inputs"]["image"] = base_item["comfyName"]
#     wf["16"]["inputs"]["seed"]  = int.from_bytes(os.urandom(4), byteorder="little")
#     async with httpx.AsyncClient() as client:
#         res = await client.post(f"{COMFY_URL}/prompt", json={"prompt": wf})
#         if res.status_code != 200:
#             raise HTTPException(status_code=500, detail="Erro ao enviar ao ComfyUI")
#         prompt_id = res.json()["prompt_id"]
#     new_item = {"id": str(uuid.uuid4()), "status": "pending", "filename": None,
#                 "comfyName": None, "prompt_id": prompt_id, "prompt": req.prompt}
#     state.insert(0, new_item)
#     await save_state(state)
#     return new_item

@app.delete("/api/item/{item_id}")
async def delete_item(item_id: str):
    state = await load_state()
    item  = next((i for i in state if i["id"] == item_id), None)
    if item:
        if item["status"] == "pending" and item["prompt_id"]:
            async with httpx.AsyncClient() as client:
                try:
                    await client.post(f"{COMFY_URL}/queue", json={"delete": [item["prompt_id"]]})
                    await client.post(f"{COMFY_URL}/interrupt")
                except: pass
        state = [i for i in state if i["id"] != item_id]
        await save_state(state)
        if item.get("filename"):
            try: os.remove(os.path.join(IMG_DIR, item["filename"]))
            except: pass
    return {"status": "ok"}

app.mount("/images", StaticFiles(directory=IMG_DIR), name="images")

# ── Frontend ──────────────────────────────────────────────────────────────────
FRONTEND_HTML = r"""<!DOCTYPE html>
<html lang="pt-BR">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no, viewport-fit=cover">
<meta name="mobile-web-app-capable" content="yes">
<meta name="apple-mobile-web-app-capable" content="yes">
<meta name="apple-mobile-web-app-status-bar-style" content="black-translucent">
<title>Galeria ComfyUI</title>
<script src="https://cdn.tailwindcss.com"></script>
<style>
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

  body {
    background: #000;
    color: #f3f4f6;
    min-height: 100dvh;
    -webkit-tap-highlight-color: transparent;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    overflow-x: hidden;
  }

  /* ── Modal fullscreen ───────────────────────────────────────────── */
  #imageModal {
    /* começa escondido via display:none — JS alterna entre none e flex */
    display: none;
    position: fixed;
    top: 0; left: 0; right: 0; bottom: 0;
    width: 100vw;
    height: 100vh;
    height: 100dvh;
    background: #000;
    z-index: 9999;
    flex-direction: column;
    overflow: hidden;
    padding-top:    env(safe-area-inset-top,    0px);
    padding-bottom: env(safe-area-inset-bottom, 0px);
    padding-left:   env(safe-area-inset-left,   0px);
    padding-right:  env(safe-area-inset-right,  0px);
  }

  /* Quando JS adiciona .open o modal aparece */
  #imageModal.open { display: flex; }

  #modalTopBar {
    flex-shrink: 0;
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 12px 16px 8px;
    background: #000;
  }

  #modalImageContainer {
    flex: 1 1 0;
    min-height: 0;
    display: flex;
    align-items: center;
    justify-content: center;
    overflow: hidden;
    position: relative;
    background: #000;
  }

  #viewFullImage {
    /* Ocupa toda a área disponível mantendo proporção */
    width: 100%;
    height: 100%;
    object-fit: contain;
    display: block;
    user-select: none;
    -webkit-user-drag: none;
    /* Renderiza sem suavização — mantém pixels nítidos na resolução nativa */
    image-rendering: -webkit-optimize-contrast;
    image-rendering: crisp-edges;
  }

  /* Overlay do prompt — flutua sobre a imagem, não empurra ela para cima */
  #modalFooter {
    position: absolute;
    bottom: 0;
    left: 0;
    right: 0;
    padding: 32px 16px 14px;
    /* fade de baixo para cima */
    background: linear-gradient(to top, rgba(0,0,0,0.85) 0%, rgba(0,0,0,0) 100%);
    display: flex;
    flex-direction: column;
    gap: 4px;
    pointer-events: none; /* não bloqueia swipe na imagem */
    z-index: 5;
  }

  #modalFooter p, #modalFooter #viewFullPrompt {
    pointer-events: auto; /* mas o texto pode ser selecionado */
  }

  /* ── Gallery cards (Proporção 2:3 Vertical) ────────────────────── */
  .gallery-card {
    background: #111;
    border-radius: 10px;
    overflow: hidden;
    /* Altera de 1 (quadrado) para 2/3 (vertical) */
    aspect-ratio: 2 / 3; 
    cursor: pointer;
    position: relative;
    border: 1px solid #1f1f1f;
    transition: transform 0.12s;
  }
  .gallery-card :active { transform: scale(0.96); }
  /* Garante que a imagem preencha o retângulo vertical */
  .gallery-card img { width: 100%; height: 100%; object-fit: cover; display: block; }

  /* ── Spinner ────────────────────────────────────────────────────── */
  .spinner {
    border: 3px solid rgba(255,255,255,0.1);
    border-left-color: #3b82f6;
    border-radius: 50%;
    width: 32px; height: 32px;
    animation: spin 1s linear infinite;
  }
  @keyframes spin { to { transform: rotate(360deg); } }

  .no-scrollbar::-webkit-scrollbar { display: none; }
  .no-scrollbar { -ms-overflow-style: none; scrollbar-width: none; }
</style>
</head>
<body>
<!-- ── GALERIA ─────────────────────────────────────────────────────────── -->
<div class="max-w-7xl mx-auto px-2 py-4 sm:px-6 sm:py-8">
  <header class="flex justify-between items-center mb-4 sm:mb-8 border-b border-gray-900 pb-3">
    <h1 class="text-xl sm:text-3xl font-bold tracking-tight text-white">Galeria ComfyUI</h1>
    <div class="flex items-center gap-2">
      <input type="file" id="uploadInput" class="hidden" accept="image/png,image/jpeg">
      <button onclick="document.getElementById('uploadInput').click()"
        class="bg-blue-600 hover:bg-blue-500 active:scale-95 px-3 py-2 sm:px-5 sm:py-2.5 rounded-lg text-xs sm:text-sm font-semibold transition shadow-md">
        + Upload
      </button>
    </div>
  </header>
  <div id="gallery" class="grid grid-cols-2 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-2 sm:gap-6"></div>
</div>

<!-- ── MODAL 1: VISUALIZADOR FULLSCREEN ──────────────────────────────────── -->
<div id="imageModal">

  <!-- Topo -->
  <div id="modalTopBar">
    <button id="btnFecharModal"
      style="background:#1f2937;color:#fff;border:none;border-radius:50%;width:42px;height:42px;font-size:18px;font-weight:bold;cursor:pointer;display:flex;align-items:center;justify-content:center;">
      ✕
    </button>
    <span id="modalCounter" style="color:#6b7280;font-size:12px;font-family:monospace;"></span>
    <button id="btnAbrirPrompt"
      style="background:#2563eb;color:#fff;border:none;border-radius:50%;width:42px;height:42px;cursor:pointer;display:flex;align-items:center;justify-content:center;"
      title="Criar variação">
      <svg width="20" height="20" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2.5"
          d="M15.232 5.232l3.536 3.536m-2.036-5.036a2.5 2.5 0 113.536 3.536L6.5 21.036H3v-3.572L16.732 3.732z"/>
      </svg>
    </button>
  </div>

  <!-- Imagem + overlay de prompt dentro do mesmo container -->
  <div id="modalImageContainer">
    <!-- Nav desktop -->
    <button onclick="slidePrev()"
      style="position:absolute;left:12px;top:50%;transform:translateY(-50%);background:rgba(0,0,0,0.6);color:#fff;border:none;border-radius:50%;width:44px;height:44px;font-size:18px;cursor:pointer;display:none;align-items:center;justify-content:center;z-index:10;"
      id="btnPrev">◀</button>
    <button onclick="slideNext()"
      style="position:absolute;right:12px;top:50%;transform:translateY(-50%);background:rgba(0,0,0,0.6);color:#fff;border:none;border-radius:50%;width:44px;height:44px;font-size:18px;cursor:pointer;display:none;align-items:center;justify-content:center;z-index:10;"
      id="btnNext">▶</button>

    <img id="viewFullImage" src="" alt="Imagem selecionada">

    <!-- Prompt como overlay no rodapé da imagem -->
    <div id="modalFooter">
      <p id="viewFullPrompt" style="font-size:12px;color:rgba(255,255,255,0.85);line-height:1.5;user-select:text;text-shadow:0 1px 3px rgba(0,0,0,0.8);"></p>
      <p style="font-size:10px;color:rgba(255,255,255,0.3);letter-spacing:0.05em;">← deslize →</p>
    </div>
  </div>
</div>

<!-- ── MODAL 2: PROMPT / VARIAÇÃO ────────────────────────────────────────── -->
<div id="promptModal" style="display:none;position:fixed;inset:0;background:rgba(0,0,0,0.9);z-index:10000;align-items:center;justify-content:center;padding:8px;backdrop-filter:blur(4px);">
  <div style="background:#111827;border:1px solid #1f2937;border-radius:16px;width:100%;max-width:900px;display:flex;flex-direction:column;overflow:hidden;max-height:95dvh;position:relative;">
    <button id="btnFecharPrompt"
      style="position:absolute;top:12px;right:12px;background:#1f2937;color:#fff;border:none;border-radius:50%;width:36px;height:36px;font-weight:bold;font-size:15px;cursor:pointer;z-index:10;display:flex;align-items:center;justify-content:center;">
      ✕
    </button>

    <!-- Preview da imagem base -->
    <div style="background:#000;display:flex;align-items:center;justify-content:center;padding:12px;height:38vh;border-bottom:1px solid #1f2937;">
      <img id="promptRefImage" src="" alt="Imagem base" style="max-width:100%;max-height:100%;object-fit:contain;border-radius:8px;">
    </div>

    <!-- Inputs -->
    <div style="padding:20px;display:flex;flex-direction:column;gap:14px;">
      <div>
        <h2 style="font-size:18px;font-weight:700;color:#fff;">Nova Variação</h2>
        <p style="font-size:11px;color:#6b7280;margin-top:2px;">A imagem acima será usada como referência.</p>
      </div>
      <div>
        <label style="display:block;font-size:10px;font-weight:700;color:#6b7280;text-transform:uppercase;letter-spacing:0.1em;margin-bottom:6px;">Prompt</label>
        <textarea id="promptText" rows="4"
          style="width:100%;background:#1f2937;border:1px solid #374151;border-radius:10px;padding:10px 12px;color:#fff;font-size:14px;resize:none;outline:none;font-family:inherit;"
          placeholder="Descreva as modificações..."></textarea>
      </div>
      <button id="btnEnviarPrompt"
        style="width:100%;background:#2563eb;color:#fff;border:none;border-radius:10px;padding:14px;font-size:14px;font-weight:600;cursor:pointer;transition:background 0.15s;">
        Gerar Variação ✦
      </button>
    </div>
  </div>
</div>

<script>
// ── Estado global ─────────────────────────────────────────────────────────────
let galeria      = [];
let modalIndex   = 0;
let currentBaseId = null;
let touchStartX  = 0;
let touchStartY  = 0;
let isDesktop    = window.matchMedia('(min-width: 768px)').matches;

// ── Refs ──────────────────────────────────────────────────────────────────────
const imageModal   = document.getElementById('imageModal');
const promptModal  = document.getElementById('promptModal');
const viewImage    = document.getElementById('viewFullImage');
const viewPrompt   = document.getElementById('viewFullPrompt');
const modalCounter = document.getElementById('modalCounter');
const promptText   = document.getElementById('promptText');
const btnPrev      = document.getElementById('btnPrev');
const btnNext      = document.getElementById('btnNext');

// Mostra setas laterais apenas em desktop
if (isDesktop) {
  btnPrev.style.display = 'flex';
  btnNext.style.display = 'flex';
}

// ── Botões (listeners explícitos, sem onclick inline no modal) ────────────────
document.getElementById('btnFecharModal').addEventListener('click',  fecharImageModal);
document.getElementById('btnAbrirPrompt').addEventListener('click',  abrirPromptModal);
document.getElementById('btnFecharPrompt').addEventListener('click', fecharPromptModal);
document.getElementById('btnEnviarPrompt').addEventListener('click', enviarPrompt);

// ── Polling ───────────────────────────────────────────────────────────────────
async function carregarGaleria() {
  try {
    const res = await fetch('/api/state');
    galeria = await res.json();
    renderGaleria();
  } catch(e) {}
}

setInterval(carregarGaleria, 2000);
carregarGaleria();

// ── Render galeria ────────────────────────────────────────────────────────────
function renderGaleria() {
  const container = document.getElementById('gallery');
  container.innerHTML = '';

  galeria.forEach((item, idx) => {
    const card = document.createElement('div');
    card.className = 'gallery-card';

    if (item.status === 'pending') {
      card.innerHTML = `
        <div style="width:100%;height:100%;display:flex;flex-direction:column;align-items:center;justify-content:center;gap:8px;background:#0a0a0a;">
          <div class="spinner"></div>
          <p style="font-size:10px;color:#6b7280;text-align:center;padding:0 8px;line-height:1.4;">${item.prompt || 'Gerando...'}</p>
        </div>`;
    } else {
      const img = document.createElement('img');
      img.src     = `/images/${item.filename}`;
      img.loading = 'lazy';
      img.alt     = item.prompt || '';
      card.appendChild(img);
      card.addEventListener('click', () => abrirImageModal(idx));
    }

    // Botão deletar
    const btnDel = document.createElement('button');
    btnDel.innerHTML = '✕';
    btnDel.style.cssText = 'position:absolute;top:6px;right:6px;background:rgba(0,0,0,0.75);color:#fff;border:none;border-radius:50%;width:24px;height:24px;font-size:11px;font-weight:bold;cursor:pointer;z-index:5;display:flex;align-items:center;justify-content:center;';
    btnDel.addEventListener('click', (e) => { e.stopPropagation(); deletarItem(item.id); });
    card.appendChild(btnDel);

    container.appendChild(card);
  });
}

// ── Deletar ───────────────────────────────────────────────────────────────────
async function deletarItem(id) {
  if (!confirm('Remover esta imagem?')) return;
  await fetch(`/api/item/${id}`, { method: 'DELETE' });
  await carregarGaleria();
}

// ── Modal fullscreen ──────────────────────────────────────────────────────────
function abrirImageModal(idx) {
  // Garante que idx aponta para item completo
  if (galeria[idx] && galeria[idx].status !== 'completed') return;

  modalIndex = idx;
  document.body.style.overflow = 'hidden';
  imageModal.classList.add('open');       // display:flex via CSS .open

  // Fullscreen nativo no mobile (quando o browser permitir)
  if (!document.fullscreenElement && document.documentElement.requestFullscreen) {
    document.documentElement.requestFullscreen().catch(() => {});
  }

  atualizarModalConteudo();
}

function fecharImageModal() {
  imageModal.classList.remove('open');   // volta a display:none
  document.body.style.overflow = '';

  if (document.fullscreenElement) {
    document.exitFullscreen().catch(() => {});
  }
}

function atualizarModalConteudo() {
  const item = galeria[modalIndex];
  if (!item || item.status !== 'completed') return;

  // Limpa src antes de setar novo para forçar reload caso seja a mesma url
  viewImage.src = '';
  viewImage.src = `/images/${item.filename}`;
  viewPrompt.textContent   = item.prompt || '—';
  modalCounter.textContent = `${modalIndex + 1} / ${galeria.length}`;
  currentBaseId = item.id;
}

function slideNext() {
  let next = modalIndex + 1;
  while (next < galeria.length && galeria[next].status !== 'completed') next++;
  if (next < galeria.length) { modalIndex = next; atualizarModalConteudo(); }
}

function slidePrev() {
  let prev = modalIndex - 1;
  while (prev >= 0 && galeria[prev].status !== 'completed') prev--;
  if (prev >= 0) { modalIndex = prev; atualizarModalConteudo(); }
}

// ── Swipe touch (no container da imagem, sem clonar nada) ────────────────────
const container = document.getElementById('modalImageContainer');
container.addEventListener('touchstart', (e) => {
  touchStartX = e.touches[0].clientX;
  touchStartY = e.touches[0].clientY;
}, { passive: true });

container.addEventListener('touchend', (e) => {
  const dx = e.changedTouches[0].clientX - touchStartX;
  const dy = e.changedTouches[0].clientY - touchStartY;
  if (Math.abs(dx) > Math.abs(dy) && Math.abs(dx) > 40) {
    dx < 0 ? slideNext() : slidePrev();
  }
}, { passive: true });

// ── Teclado desktop ───────────────────────────────────────────────────────────
document.addEventListener('keydown', (e) => {
  if (!imageModal.classList.contains('open')) return;
  if (e.key === 'ArrowRight') slideNext();
  if (e.key === 'ArrowLeft')  slidePrev();
  if (e.key === 'Escape')     fecharImageModal();
});

// ── Modal de prompt ───────────────────────────────────────────────────────────
function abrirPromptModal() {
  const item = galeria[modalIndex];
  if (!item) return;
  document.getElementById('promptRefImage').src = `/images/${item.filename}`;
  currentBaseId = item.id;
  promptText.value = '';
  promptModal.style.display = 'flex';
}

function fecharPromptModal() {
  promptModal.style.display = 'none';
}

async function enviarPrompt() {
  const prompt = promptText.value.trim();
  if (!prompt || !currentBaseId) return;

  fecharPromptModal();
  fecharImageModal();

  await fetch('/api/generate', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ base_id: currentBaseId, prompt })
  });

  await carregarGaleria();
}

// ── Upload ────────────────────────────────────────────────────────────────────
document.getElementById('uploadInput').addEventListener('change', async (e) => {
  const file = e.target.files[0];
  if (!file) return;
  const fd = new FormData();
  fd.append('file', file);
  await fetch('/api/upload', { method: 'POST', body: fd });
  e.target.value = '';
  await carregarGaleria();
});
</script>
</body>
</html>"""

@app.get("/", response_class=HTMLResponse)
def serve_frontend():
    return HTMLResponse(content=FRONTEND_HTML)

if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8888, reload=False)
