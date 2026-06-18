import os
import uuid
import copy
import asyncio
import httpx
import uvicorn
from contextlib import asynccontextmanager
from PIL import Image
from pydantic import BaseModel
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

# Modularized Imports
from config import load_settings, save_settings
from state_manager import load_state, save_state, state_lock, _load, _save, delete_item_recursive
from workflow import build_generation_workflow

# ── Configurations & Initial State ───────────────────────────────────────────
startup_settings = load_settings()
STARTUP_IMG_DIR = os.path.join(startup_settings.data_dir, "imagens")
STARTUP_THUMB_DIR = os.path.join(startup_settings.data_dir, "thumbnails")
os.makedirs(STARTUP_IMG_DIR, exist_ok=True)
os.makedirs(STARTUP_THUMB_DIR, exist_ok=True)

def save_image_with_thumbnail(filepath: str, data: bytes) -> None:
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "wb") as f:
        f.write(data)
    # Generate thumbnail (400px max, JPEG ~80% quality)
    thumb_path = filepath.replace(STARTUP_IMG_DIR, STARTUP_THUMB_DIR)
    thumb_path = os.path.splitext(thumb_path)[0] + ".jpg"
    os.makedirs(os.path.dirname(thumb_path), exist_ok=True)
    try:
        from io import BytesIO
        img = Image.open(BytesIO(data))
        img.thumbnail((400, 400))
        if img.mode in ("RGBA", "P"):
            img = img.convert("RGB")
        img.save(thumb_path, "JPEG", quality=80)
    except Exception as e:
        print(f"Thumbnail error: {e}")

# ── ComfyUI Integration ───────────────────────────────────────────────────────
async def upload_to_comfy(filepath: str, filename: str) -> str:
    settings = load_settings()
    # Use only the basename for ComfyUI upload to prevent path traversal issues on ComfyUI side
    upload_name = os.path.basename(filename)
    async with httpx.AsyncClient() as client:
        with open(filepath, "rb") as f:
            res = await client.post(f"{settings.comfy_url}/upload/image", files={"image": (upload_name, f, "image/png")})
            res.raise_for_status()
            return res.json()["name"]

async def check_comfy_queue():
    while True:
        try:
            settings = load_settings()
            state = await load_state()
            pending = [i for i in state if i["status"] == "pending"]
            if pending:
                async with httpx.AsyncClient(timeout=10.0) as client:
                    history = (await client.get(f"{settings.comfy_url}/history")).json()
                    queue_data = (await client.get(f"{settings.comfy_url}/queue")).json()
                    
                    # Extract active ComfyUI task IDs
                    running_ids = [q_item[1] for q_item in queue_data.get("queue_running", [])]
                    pending_ids = [q_item[1] for q_item in queue_data.get("queue_pending", [])]
                    active_comfy_ids = set(running_ids + pending_ids)
                    
                    ghost_ids = []
                    for item in pending:
                        pid = item["prompt_id"]
                        if pid in history:
                            outputs = history[pid].get("outputs", {})
                            # Node "3" remains the SaveImage node in our workflow
                            if "3" in outputs and "images" in outputs["3"]:
                                img_data = outputs["3"]["images"][0]
                                url = f"{settings.comfy_url}/view?filename={img_data['filename']}&subfolder={img_data['subfolder']}&type={img_data['type']}"
                                res_img = await client.get(url)
                                
                                # Resolve local path and create directory lineage structure
                                local_fn = item["filename"]
                                local_path = os.path.join(STARTUP_IMG_DIR, local_fn)

                                save_image_with_thumbnail(local_path, res_img.content)
                                comfy_name = await upload_to_comfy(local_path, local_fn)
                                
                                # Atomic update under lock
                                async with state_lock:
                                    current_state = _load()
                                    for entry in current_state:
                                        if entry["id"] == item["id"]:
                                            entry["status"] = "completed"
                                            entry["comfyName"] = comfy_name
                                            break
                                    _save(current_state)
                        elif pid not in active_comfy_ids:
                            # If prompt_id is neither in history nor active in queue, it is a ghost request
                            ghost_ids.append(item["id"])
                    
                    # Clean up ghost requests from local state history (prevent infinite loops)
                    if ghost_ids:
                        async with state_lock:
                            current_state = _load()
                            current_state = [entry for entry in current_state if entry["id"] not in ghost_ids]
                            _save(current_state)
                            print(f"Cleaned up {len(ghost_ids)} ghost requests from state history.")
                            
        except httpx.ConnectError:
            pass
        except Exception as e:
            print(f"Polling error: {e}")
        await asyncio.sleep(1.5)

@asynccontextmanager
async def lifespan(app: FastAPI):
    task = asyncio.create_task(check_comfy_queue())
    yield
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass

app = FastAPI(lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

# ── API Endpoints ─────────────────────────────────────────────────────────────
@app.get("/api/state")
async def get_state(limit: int = 20, offset: int = 0, root_id: str = None):
    state = await load_state()
    if root_id:
        # Return a specific item + all its descendants (lineage view)
        def get_descendants(parent_id):
            children = []
            for item in state:
                if item.get("parent_id") == parent_id:
                    children.append(item)
                    children.extend(get_descendants(item["id"]))
            return children

        root = next((i for i in state if i["id"] == root_id), None)
        if not root:
            raise HTTPException(status_code=404, detail="Image not found")
        subtree = [root] + get_descendants(root_id)
        return subtree[offset : offset + limit]
    else:
        # Return only root images (no parent)
        roots = [i for i in state if i.get("parent_id") is None]
        return roots[offset : offset + limit]

@app.post("/api/upload")
async def handle_upload(file: UploadFile = File(...)):
    item_id = str(uuid.uuid4())
    # Initial upload root images are stored in gallery_data/imagens/root/
    filename = f"root/{item_id}.png"
    filepath = os.path.join(STARTUP_IMG_DIR, filename)
    save_image_with_thumbnail(filepath, await file.read())
    comfy_name = await upload_to_comfy(filepath, filename)
    new_item = {
        "id": item_id,
        "parent_id": None,
        "status": "completed",
        "filename": filename,
        "comfyName": comfy_name,
        "prompt_id": None,
        "prompt": "Initial Upload",
        "prompt_original": "Initial Upload",
        "bypass_llm": False,
        "force_consistency": False,
        "upscale_input": False
    }
    async with state_lock:
        state = _load()
        state.insert(0, new_item)
        _save(state)
    return new_item

class PromptRequest(BaseModel):
    base_id: str
    prompt: str
    upscale_input: bool = False
    bypass_llm: bool = False
    force_consistency: bool = False

async def expand_prompt_openrouter(user_prompt: str) -> str:
    settings = load_settings()
    fallback_models = settings.openrouter_models
    system_instruction = settings.system_prompt
    
    async with httpx.AsyncClient(timeout=12.0) as client:
        for model in fallback_models:
            try:
                print(f"Trying to expand prompt with model: {model}")
                res = await client.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {settings.openrouter_api_key}".strip(),
                        "Content-Type": "application/json",
                        "HTTP-Referer": "http://localhost:8888",
                        "X-Title": "ComfyUI Gallery Proxy"
                    },
                    json={
                        "model": model,
                        "messages": [
                            {"role": "system", "content": system_instruction},
                            {"role": "user", "content": user_prompt}
                        ],
                        "temperature": 0.7
                    }
                )
                
                if res.status_code == 200:
                    optimized_prompt = res.json()["choices"][0]["message"]["content"].strip()
                    print(f"Success with model [{model}]: {optimized_prompt}")
                    return optimized_prompt
                else:
                    print(f"Model [{model}] failed with status {res.status_code}. Trying next...")
            except Exception as e:
                print(f"Error connecting with model [{model}]: {e}. Trying next...")
                
    print("All OpenRouter models failed. Using original user prompt.")
    return user_prompt

@app.post("/api/generate")
async def generate(req: PromptRequest):
    settings = load_settings()
    state = await load_state()
    base_item = next((i for i in state if i["id"] == req.base_id), None)
    if not base_item or not base_item.get("comfyName"):
        raise HTTPException(status_code=400, detail="Invalid base image.")
        
    # Re-upload the base image to ComfyUI to guarantee it exists in ComfyUI's input directory (in case ComfyUI restarted/cleared input)
    local_path = os.path.join(STARTUP_IMG_DIR, base_item["filename"])
    if os.path.exists(local_path):
        try:
            comfy_name = await upload_to_comfy(local_path, base_item["filename"])
            base_item["comfyName"] = comfy_name
        except Exception as e:
            print(f"Failed to auto-upload base image to ComfyUI: {e}")
    
    # ── LLM Bypass / Expansion ────────────────────────────────────────────────
    if req.bypass_llm:
        prompt_final = req.prompt
    else:
        prompt_final = await expand_prompt_openrouter(req.prompt)

    # ── Consistency Formatting (Force identical image) ──────────────────────
    # We apply this after LLM expansion so that the LLM's system prompt doesn't strip it.
    if req.force_consistency:
        prompt_final = f"Keep everything identical, only modify: {prompt_final}"

    # ── ComfyUI Workflow Generation ───────────────────────────────────────────
    wf = build_generation_workflow(
        prompt=prompt_final,
        base_image_comfy_name=base_item["comfyName"],
        seed=int.from_bytes(os.urandom(4), byteorder="little"),
        upscale_input=req.upscale_input,
        diffusion_model_name=settings.diffusion_model_name,
        clip_model_name=settings.clip_model_name,
        vae_model_name=settings.vae_model_name,
        upscale_model_name=settings.upscale_model_name,
        input_upscale_model_name=settings.input_upscale_model_name
    )

    async with httpx.AsyncClient() as client:
        res = await client.post(f"{settings.comfy_url}/prompt", json={"prompt": wf})
        if res.status_code != 200:
            raise HTTPException(status_code=500, detail="Error sending request to ComfyUI")
        prompt_id = res.json()["prompt_id"]
        
    new_id = str(uuid.uuid4())
    # The file is generated inside a dedicated folder named after parent image (Lineage)
    filename = f"{req.base_id}/{new_id}.png"
    
    new_item = {
        "id": new_id,
        "parent_id": req.base_id,
        "status": "pending",
        "filename": filename,
        "comfyName": None,
        "prompt_id": prompt_id,
        "prompt": prompt_final,
        "prompt_original": req.prompt,
        "bypass_llm": req.bypass_llm,
        "force_consistency": req.force_consistency,
        "upscale_input": req.upscale_input
    }
                
    async with state_lock:
        state = _load()
        state.insert(0, new_item)
        _save(state)
    return new_item

# ── Bulk Delete (Cascade Deletion with ComfyUI Cancellation) ──────────────────
class DeleteRequest(BaseModel):
    ids: list[str]

@app.post("/api/items/delete")
async def delete_items(req: DeleteRequest):
    settings = load_settings()
    async with state_lock:
        state = _load()
        all_removed_ids = []
        current_state = state
        
        for item_id in req.ids:
            if any(i["id"] == item_id for i in current_state):
                current_state, removed_ids = delete_item_recursive(item_id, current_state)
                all_removed_ids.extend(removed_ids)
                
        _save(current_state)
        
        # Clean up files and directories recursively on disk, and cancel pending tasks
        for r_id in all_removed_ids:
            item = next((i for i in state if i["id"] == r_id), None)
            if item:
                # 1. Cancel in ComfyUI queue if item is currently pending execution
                if item.get("status") == "pending" and item.get("prompt_id"):
                    async with httpx.AsyncClient() as client:
                        try:
                            await client.post(f"{settings.comfy_url}/queue", json={"delete": [item["prompt_id"]]})
                            await client.post(f"{settings.comfy_url}/interrupt")
                        except Exception as ex:
                            print(f"Error canceling pending ComfyUI task {item['prompt_id']}: {ex}")
                
                # 2. Remove the corresponding image file
                if item.get("filename"):
                    file_path = os.path.join(STARTUP_IMG_DIR, item["filename"])
                    if os.path.exists(file_path):
                        try:
                            os.remove(file_path)
                        except:
                            pass
            
            # 3. Remove the directory of descendants associated with this item ID
            dir_path = os.path.join(STARTUP_IMG_DIR, r_id)
            if os.path.exists(dir_path):
                import shutil
                try:
                    shutil.rmtree(dir_path, ignore_errors=True)
                except:
                    pass
                
    return {"status": "ok", "deleted_count": len(all_removed_ids)}

# ── Dynamic Settings ──────────────────────────────────────────────────────────
class ConfigSchema(BaseModel):
    system_prompt: str
    openrouter_models: list[str]
    diffusion_model_name: str
    clip_model_name: str
    vae_model_name: str
    upscale_model_name: str
    input_upscale_model_name: str
    comfy_url: str
    openrouter_api_key: str

@app.get("/api/config")
async def get_config():
    settings = load_settings()
    return {
        "system_prompt": settings.system_prompt,
        "openrouter_models": settings.openrouter_models,
        "diffusion_model_name": settings.diffusion_model_name,
        "clip_model_name": settings.clip_model_name,
        "vae_model_name": settings.vae_model_name,
        "upscale_model_name": settings.upscale_model_name,
        "input_upscale_model_name": settings.input_upscale_model_name,
        "comfy_url": settings.comfy_url,
        "openrouter_api_key": settings.openrouter_api_key
    }

@app.post("/api/config")
async def update_config(cfg: ConfigSchema):
    settings = load_settings()
    settings.system_prompt = cfg.system_prompt
    settings.openrouter_models = cfg.openrouter_models
    settings.diffusion_model_name = cfg.diffusion_model_name
    settings.clip_model_name = cfg.clip_model_name
    settings.vae_model_name = cfg.vae_model_name
    settings.upscale_model_name = cfg.upscale_model_name
    settings.input_upscale_model_name = cfg.input_upscale_model_name
    settings.comfy_url = cfg.comfy_url
    settings.openrouter_api_key = cfg.openrouter_api_key
    save_settings(settings)
    return {"status": "ok"}

@app.get("/api/comfy/models")
async def get_comfy_models():
    settings = load_settings()
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            res = await client.get(f"{settings.comfy_url}/object_info")
            res.raise_for_status()
            info = res.json()

        def extract_options(node_name, field_name):
            node = info.get(node_name, {})
            field = node.get("input", {}).get("required", {}).get(field_name)
            if not field or not isinstance(field, list) or len(field) == 0:
                return []
            # Format: [["opt1", "opt2", ...]] — simple list in first element
            if isinstance(field[0], list):
                return field[0]
            # Format: ["COMBO", {"options": [...]}] — COMBO type with options dict
            if len(field) > 1 and isinstance(field[1], dict) and "options" in field[1]:
                return field[1]["options"]
            return []

        return {
            "diffusion_models": extract_options("DiffusionModelLoaderKJ", "model_name"),
            "clip_models": extract_options("CLIPLoader", "clip_name"),
            "clip_types": extract_options("CLIPLoader", "type"),
            "vae_models": extract_options("VAELoader", "vae_name"),
            "upscale_models": extract_options("UpscaleModelLoader", "model_name"),
        }
    except Exception as e:
        print(f"Error fetching ComfyUI models: {e}")
        return {
            "diffusion_models": [],
            "clip_models": [],
            "clip_types": [],
            "vae_models": [],
            "upscale_models": [],
        }

app.mount("/images", StaticFiles(directory=STARTUP_IMG_DIR), name="images")

# ── Thumbnail serving (lazy generation) ──────────────────────────────────────
@app.get("/thumbnails/{path:path}")
async def serve_thumbnail(path: str):
    # Map thumbnail path to image path (thumb: .jpg → image: .png)
    thumb_path = os.path.join(STARTUP_THUMB_DIR, path)
    if os.path.exists(thumb_path):
        return FileResponse(thumb_path)

    # Generate on demand from full image
    img_path = os.path.join(STARTUP_IMG_DIR, os.path.splitext(path)[0] + ".png")
    if not os.path.exists(img_path):
        raise HTTPException(status_code=404, detail="Image not found")

    os.makedirs(os.path.dirname(thumb_path), exist_ok=True)
    try:
        img = Image.open(img_path)
        img.thumbnail((400, 400))
        if img.mode in ("RGBA", "P"):
            img = img.convert("RGB")
        img.save(thumb_path, "JPEG", quality=80)
        return FileResponse(thumb_path)
    except Exception as e:
        print(f"Thumbnail gen error: {e}")
        return FileResponse(img_path)

# ── Dynamic Frontend HTML Loading ─────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
def serve_frontend():
    frontend_path = os.path.join(os.path.dirname(__file__), "frontend.html")
    if os.path.exists(frontend_path):
        with open(frontend_path, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    else:
        raise HTTPException(status_code=404, detail="frontend.html not found.")

if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8888, reload=False)
