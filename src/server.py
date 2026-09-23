import asyncio
import json
import os
import shutil
import sys
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Optional, cast

import httpx
import uvicorn
import websockets
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
from pydantic import BaseModel, Field

from src.config import load_settings, save_settings
from src.state_manager import _load, _save, delete_item_reparent, load_state, state_lock
from src.workflow import DEFAULT_UPSCALE_PROMPT, build_generation_workflow, build_upscale_workflow

# ── Configurations & Initial State ───────────────────────────────────────────
startup_settings = load_settings()
STARTUP_IMG_DIR = os.path.join(startup_settings.data_dir, "imagens")
STARTUP_THUMB_DIR = os.path.join(startup_settings.data_dir, "thumbnails")
os.makedirs(STARTUP_IMG_DIR, exist_ok=True)
os.makedirs(STARTUP_THUMB_DIR, exist_ok=True)

THUMBNAIL_SIZE = (400, 400)
THUMBNAIL_QUALITY = 80


def thumbnail_path_for(image_path: str) -> str:
    """Map an image path inside STARTUP_IMG_DIR to its JPEG thumbnail path."""
    thumb_path = image_path.replace(STARTUP_IMG_DIR, STARTUP_THUMB_DIR)
    return os.path.splitext(thumb_path)[0] + ".jpg"


def render_thumbnail(image_path: str | os.PathLike[str], thumb_path: str | os.PathLike[str]) -> None:
    """Write a JPEG thumbnail of `image_path` to `thumb_path`."""
    thumb = Path(thumb_path)
    thumb.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(image_path) as opened:
        opened.thumbnail(THUMBNAIL_SIZE)
        img = opened.convert("RGB") if opened.mode != "RGB" else opened
        img.save(thumb, "JPEG", quality=THUMBNAIL_QUALITY)


def write_thumbnail(image_path: str) -> None:
    """Generate the JPEG thumbnail for an image that already exists on disk."""
    try:
        render_thumbnail(image_path, thumbnail_path_for(image_path))
    except Exception as e:
        print(f"Thumbnail error: {e}")


def save_image_with_thumbnail(filepath: str, data: bytes) -> None:
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "wb") as f:
        f.write(data)
    write_thumbnail(filepath)

# ── ComfyUI Integration ───────────────────────────────────────────────────────
def _first_image_batch(outputs: dict[str, Any]) -> list[dict[str, Any]] | None:
    """Return the first non-empty `images` batch in a ComfyUI output payload.

    `outputs` is either a single node output (WebSocket `executed` event) or the
    `outputs` map of a history entry, so both the flat and the per-node shapes
    are inspected.
    """
    direct = outputs.get("images")
    if isinstance(direct, list) and direct:
        return cast(list[dict[str, Any]], direct)
    for value in outputs.values():
        if isinstance(value, dict):
            images = value.get("images")
            if isinstance(images, list) and images:
                return cast(list[dict[str, Any]], images)
    return None


async def upload_to_comfy(filepath: str, filename: str) -> str:
    settings = load_settings()
    # Use only the basename for ComfyUI upload to prevent path traversal issues on ComfyUI side
    upload_name = os.path.basename(filename)
    async with httpx.AsyncClient() as client:
        with open(filepath, "rb") as f:
            res = await client.post(f"{settings.comfy_url}/upload/image", files={"image": (upload_name, f, "image/png")})
            res.raise_for_status()
            return str(res.json()["name"])

CLIENT_ID = str(uuid.uuid4())

def get_ws_url(comfy_url: str) -> str:
    url = comfy_url.rstrip("/")
    if url.startswith("https://"):
        return f"wss://{url[8:]}/ws?clientId={CLIENT_ID}"
    elif url.startswith("http://"):
        return f"ws://{url[7:]}/ws?clientId={CLIENT_ID}"
    else:
        return f"ws://{url}/ws?clientId={CLIENT_ID}"

async def handle_node_executed(pid: str, images_list: list[dict[str, Any]]) -> None:
    try:
        settings = load_settings()
        state = await load_state()
        pending = [i for i in state if i["status"] == "pending" and i.get("prompt_id") == pid]
        if not pending:
            return

        item = pending[0]
        img_data = images_list[0]
        local_fn = item["filename"]
        local_path = os.path.join(STARTUP_IMG_DIR, local_fn)
        url = f"{settings.comfy_url}/view?filename={img_data['filename']}&subfolder={img_data['subfolder']}&type={img_data['type']}"

        linked = False
        if settings.comfy_root:
            comfy_output_file = os.path.join(settings.comfy_root, "output", img_data.get("subfolder", ""), img_data["filename"])
            if os.path.exists(comfy_output_file):
                try:
                    os.makedirs(os.path.dirname(local_path), exist_ok=True)
                    if os.path.exists(local_path):
                        os.unlink(local_path)
                    os.link(comfy_output_file, local_path)
                    write_thumbnail(local_path)
                    linked = True
                    print(f"Created Unix hard link via WebSocket from {comfy_output_file} to {local_path}")
                except Exception as e_link:
                    print(f"Failed to create hard link via WebSocket: {e_link}")

        if not linked:
            async with httpx.AsyncClient(timeout=10.0) as client:
                res_img = await client.get(url)
                save_image_with_thumbnail(local_path, res_img.content)

        comfy_name = await upload_to_comfy(local_path, local_fn)

        async with state_lock:
            current_state = _load()
            for entry in current_state:
                if entry["id"] == item["id"]:
                    entry["status"] = "completed"
                    entry["comfyName"] = comfy_name
                    break
            _save(current_state)
            print(f"Updated item {item['id']} status to completed via WebSocket executed event.")
    except Exception as e:
        print(f"Error in handle_node_executed: {e}")

async def handle_prompt_failure(pid: str) -> None:
    async with state_lock:
        current_state = _load()
        updated = False
        for entry in current_state:
            if entry.get("prompt_id") == pid and entry.get("status") == "pending":
                entry["status"] = "failed"
                updated = True
        if updated:
            _save(current_state)
            print(f"Marked prompt {pid} as failed via WebSocket execution error.")

active_progress: dict[str, float] = {}

async def websocket_listener() -> None:
    backoff = 1.0
    while True:
        try:
            settings = load_settings()
            ws_url = get_ws_url(settings.comfy_url)
            print(f"Connecting to ComfyUI WebSocket at {ws_url}...")
            async with websockets.connect(ws_url) as ws:
                print("Connected to ComfyUI WebSocket.")
                backoff = 1.0
                async for message in ws:
                    if isinstance(message, bytes):
                        continue
                    try:
                        event = json.loads(message)
                    except Exception:
                        continue
                    
                    etype = event.get("type")
                    data = event.get("data", {})
                    if etype == "progress":
                        pid = data.get("prompt_id")
                        val = data.get("value", 0)
                        mx = data.get("max", 1)
                        if pid:
                            active_progress[pid] = val / mx if mx > 0 else 0.0
                    elif etype == "executing":
                        pid = data.get("prompt_id")
                        node = data.get("node")
                        if pid:
                            if node is None:
                                active_progress.pop(pid, None)
                            else:
                                if pid not in active_progress:
                                    active_progress[pid] = 0.0
                    elif etype == "executed":
                        pid = data.get("prompt_id")
                        node = data.get("node")
                        if pid:
                            active_progress.pop(pid, None)
                        output = data.get("output", {})
                        images = _first_image_batch(output) if isinstance(output, dict) else None
                        if pid and images:
                            asyncio.create_task(handle_node_executed(pid, images))
                    elif etype == "execution_error":
                        pid = data.get("prompt_id")
                        if pid:
                            active_progress.pop(pid, None)
                            await handle_prompt_failure(pid)
        except Exception as e:
            print(f"WebSocket listener error: {e}. Reconnecting in {backoff}s...")
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 30.0)

async def check_comfy_queue() -> None:
    while True:
        try:
            settings = load_settings()
            state = await load_state()
            pending = [i for i in state if i["status"] == "pending"]
            if pending:
                async with httpx.AsyncClient(timeout=10.0) as client:
                    history = cast(dict[str, Any], (await client.get(f"{settings.comfy_url}/history")).json())
                    queue_data = cast(dict[str, Any], (await client.get(f"{settings.comfy_url}/queue")).json())
                    
                    # Extract active ComfyUI task IDs
                    running_ids = [q_item[1] for q_item in queue_data.get("queue_running", [])]
                    pending_ids = [q_item[1] for q_item in queue_data.get("queue_pending", [])]
                    active_comfy_ids = set(running_ids + pending_ids)
                    
                    ghost_ids = []
                    for item in pending:
                        pid = item["prompt_id"]
                        history_entry = history.get(pid)
                        if isinstance(history_entry, dict):
                            outputs = history_entry.get("outputs", {})
                            batch = _first_image_batch(outputs) if isinstance(outputs, dict) else None
                            img_data = batch[0] if batch else None
                            if img_data:
                                url = f"{settings.comfy_url}/view?filename={img_data['filename']}&subfolder={img_data['subfolder']}&type={img_data['type']}"
                                
                                # Resolve local path and create directory lineage structure
                                local_fn = item["filename"]
                                local_path = os.path.join(STARTUP_IMG_DIR, local_fn)

                                # Try hard link fallback or fetch via network
                                linked = False
                                if settings.comfy_root:
                                    comfy_output_file = os.path.join(settings.comfy_root, "output", img_data.get("subfolder", ""), img_data["filename"])
                                    if os.path.exists(comfy_output_file):
                                        try:
                                            os.makedirs(os.path.dirname(local_path), exist_ok=True)
                                            if os.path.exists(local_path):
                                                os.unlink(local_path)
                                            os.link(comfy_output_file, local_path)
                                            write_thumbnail(local_path)
                                            linked = True
                                            print(f"Created Unix hard link from {comfy_output_file} to {local_path}")
                                        except Exception as e_link:
                                            print(f"Failed to create hard link: {e_link}")

                                if not linked:
                                    res_img = await client.get(url)
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
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    task = asyncio.create_task(check_comfy_queue())
    ws_task = asyncio.create_task(websocket_listener())
    yield
    task.cancel()
    ws_task.cancel()
    try:
        await asyncio.gather(task, ws_task, return_exceptions=True)
    except asyncio.CancelledError:
        pass

app = FastAPI(lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

# ── API Endpoints ─────────────────────────────────────────────────────────────
@app.get("/api/state")
async def get_state(limit: int = 20, offset: int = 0, root_id: Optional[str] = None) -> list[dict[str, Any]]:
    state = await load_state()
    if root_id:
        # Return a specific item + all its descendants (lineage view)
        def get_descendants(parent_id: str) -> list[dict[str, Any]]:
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
        for item in subtree:
            pid = item.get("prompt_id")
            if pid and pid in active_progress:
                item["progress"] = active_progress[pid]
        return subtree[offset : offset + limit]
    else:
        # Return only root images (no parent)
        roots = [i for i in state if i.get("parent_id") is None]
        for item in roots:
            pid = item.get("prompt_id")
            if pid and pid in active_progress:
                item["progress"] = active_progress[pid]
        return roots[offset : offset + limit]

@app.post("/api/upload")
async def handle_upload(file: UploadFile = File(...)) -> dict[str, Any]:
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
        "bypass_llm": False
    }
    async with state_lock:
        state = _load()
        state.insert(0, new_item)
        _save(state)
    return new_item

@app.post("/api/upload-reference")
async def handle_upload_reference(file: UploadFile = File(...)) -> dict[str, Any]:
    ref_id = str(uuid.uuid4())
    ext = os.path.splitext(file.filename or "")[1] or ".png"
    filename = f"references/ref_{ref_id}{ext}"
    filepath = os.path.join(STARTUP_IMG_DIR, filename)
    data = await file.read()
    save_image_with_thumbnail(filepath, data)
    comfy_name = await upload_to_comfy(filepath, filename)
    return {
        "id": ref_id,
        "filename": filename,
        "comfyName": comfy_name
    }

class PromptRequest(BaseModel):
    base_id: str
    prompt: str
    additional_images: list[str] = Field(default_factory=list)
    custom_size: bool = True
    aspect_ratio: str = "1:1 (Square)"
    megapixels: float = 0.5
    bypass_llm: bool = False
    llm_provider: Optional[str] = None

def _clean_llm_response(text: str) -> str:
    """Strip markdown code block fences and trailing whitespace from LLM output."""
    result = text.strip()
    if result.startswith("```"):
        lines = result.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        result = "\n".join(lines).strip()
    return result


async def expand_prompt_deepseek(user_prompt: str) -> Optional[str]:
    settings = load_settings()
    if not settings.deepseek_api_key:
        return None

    model_name = settings.deepseek_model or "deepseek-flash"
    # Auto-normalize model if passed as DeepSeek-V4.1-Flash or deepseek-v4-flash
    if "flash" in model_name.lower() or "v4" in model_name.lower():
        model_name = "deepseek-flash"

    try:
        print(f"Calling official DeepSeek API with model: {model_name}")
        async with httpx.AsyncClient(timeout=30.0) as client:
            res = await client.post(
                "https://api.deepseek.com/chat/completions",
                headers={
                    "Authorization": f"Bearer {settings.deepseek_api_key.strip()}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": model_name,
                    "messages": [
                        {"role": "system", "content": settings.system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    "temperature": 0.5,
                    "reasoning_effort": "low",
                }
            )
            if res.status_code == 200:
                result = _clean_llm_response(str(res.json()["choices"][0]["message"]["content"]))
                print(f"Success with DeepSeek: {result}")
                return result
            else:
                print(f"DeepSeek API error {res.status_code}: {res.text}")
    except Exception as e:
        print(f"Error calling DeepSeek API: {e}")
    return None

async def expand_prompt_openrouter(user_prompt: str) -> Optional[str]:
    settings = load_settings()
    if not settings.openrouter_api_key:
        return None
    fallback_models = settings.openrouter_models
    system_instruction = settings.system_prompt
    
    async with httpx.AsyncClient(timeout=20.0) as client:
        for model in fallback_models:
            try:
                print(f"Trying to expand prompt with OpenRouter model: {model}")
                res = await client.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {settings.openrouter_api_key.strip()}",
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
                    optimized_prompt = _clean_llm_response(str(res.json()["choices"][0]["message"]["content"]))
                    print(f"Success with OpenRouter [{model}]: {optimized_prompt}")
                    return optimized_prompt
                else:
                    print(f"OpenRouter model [{model}] failed with status {res.status_code}. Trying next...")
            except Exception as e:
                print(f"Error connecting with OpenRouter model [{model}]: {e}. Trying next...")
    return None

async def expand_prompt(user_prompt: str, preferred_provider: Optional[str] = None) -> str:
    settings = load_settings()
    provider = preferred_provider or settings.llm_provider or "deepseek"

    if provider == "deepseek":
        res = await expand_prompt_deepseek(user_prompt)
        if res:
            return res
        print("DeepSeek failed. Trying OpenRouter fallback...")
        res = await expand_prompt_openrouter(user_prompt)
        if res:
            return res
    else:
        res = await expand_prompt_openrouter(user_prompt)
        if res:
            return res
        print("OpenRouter failed. Trying DeepSeek fallback...")
        res = await expand_prompt_deepseek(user_prompt)
        if res:
            return res

    print("All configured LLM providers failed. Using original user prompt.")
    return user_prompt

@app.post("/api/generate")
async def generate(req: PromptRequest) -> dict[str, Any]:
    settings = load_settings()
    state = await load_state()
    
    base_item = next((i for i in state if i["id"] == req.base_id), None)
    if not base_item:
        raise HTTPException(status_code=400, detail="Invalid base image.")
        
    # Re-upload the base image to ComfyUI to guarantee it exists in ComfyUI's input directory
    local_path = os.path.join(STARTUP_IMG_DIR, base_item["filename"])
    if os.path.exists(local_path):
        try:
            comfy_name = await upload_to_comfy(local_path, base_item["filename"])
            base_item["comfyName"] = comfy_name
        except Exception as e:
            print(f"Failed to auto-upload base image to ComfyUI: {e}")
            
    if not base_item.get("comfyName"):
        raise HTTPException(status_code=400, detail="Invalid base image.")
    
    # ── LLM Bypass / Expansion ────────────────────────────────────────────────
    if req.bypass_llm:
        prompt_final = req.prompt
    else:
        prompt_final = await expand_prompt(req.prompt, req.llm_provider)

    # ── ComfyUI Workflow Generation ───────────────────────────────────────────
    wf = build_generation_workflow(
        prompt=prompt_final,
        base_image_comfy_name=base_item["comfyName"],
        seed=int.from_bytes(os.urandom(4), byteorder="little"),
        additional_images=req.additional_images,
        custom_size=req.custom_size,
        aspect_ratio=req.aspect_ratio,
        megapixels=req.megapixels,
        diffusion_model_name=settings.diffusion_model_name,
        clip_model_name=settings.clip_model_name,
        vae_model_name=settings.vae_model_name
    )

    async with httpx.AsyncClient() as client:
        res = await client.post(f"{settings.comfy_url}/prompt", json={"prompt": wf, "client_id": CLIENT_ID})
        if res.status_code != 200:
            raise HTTPException(status_code=500, detail="Error sending request to ComfyUI")
        prompt_id = res.json()["prompt_id"]
        
    new_id = str(uuid.uuid4())
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
        "additional_images": req.additional_images,
        "custom_size": req.custom_size,
        "aspect_ratio": req.aspect_ratio,
        "megapixels": req.megapixels
    }
                
    async with state_lock:
        state = _load()
        state.insert(0, new_item)
        _save(state)
    return new_item

# ── Qwen 2.1 Prompt-Based 2MP Upscaler ────────────────────────────────────────
class UpscaleRequest(BaseModel):
    item_id: str

@app.post("/api/upscale")
async def upscale_image(req: UpscaleRequest) -> dict[str, Any]:
    settings = load_settings()
    state = await load_state()

    base_item = next((i for i in state if i["id"] == req.item_id), None)
    if not base_item:
        raise HTTPException(status_code=404, detail="Image item not found.")

    if base_item.get("upscaled") or (base_item.get("megapixels") and float(base_item.get("megapixels", 0)) >= 2.0):
        raise HTTPException(status_code=400, detail="This image is already at maximum resolution (2MP).")

    local_path = os.path.join(STARTUP_IMG_DIR, base_item["filename"])
    if not os.path.exists(local_path):
        raise HTTPException(status_code=400, detail="Image file not found on disk.")

    # Re-upload base image to ComfyUI to ensure it exists in input directory
    try:
        comfy_name = await upload_to_comfy(local_path, base_item["filename"])
        base_item["comfyName"] = comfy_name
    except Exception as e:
        print(f"Failed to auto-upload base image to ComfyUI for upscale: {e}")
        if not base_item.get("comfyName"):
            raise HTTPException(status_code=500, detail="Failed to upload base image to ComfyUI.")

    # Read original image dimensions to scale to 2MP
    try:
        with Image.open(local_path) as img:
            orig_w, orig_h = img.size
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read image dimensions: {e}")

    seed = int.from_bytes(os.urandom(4), byteorder="little")
    wf = build_upscale_workflow(
        base_image_comfy_name=base_item["comfyName"],
        original_width=orig_w,
        original_height=orig_h,
        seed=seed,
        mp=2.0,
        diffusion_model_name=settings.diffusion_model_name,
        clip_model_name=settings.clip_model_name,
        vae_model_name=settings.vae_model_name,
    )

    async with httpx.AsyncClient() as client:
        res = await client.post(f"{settings.comfy_url}/prompt", json={"prompt": wf, "client_id": CLIENT_ID})
        if res.status_code != 200:
            raise HTTPException(status_code=500, detail="Error sending request to ComfyUI")
        prompt_id = res.json()["prompt_id"]

    new_id = str(uuid.uuid4())
    filename = f"{req.item_id}/{new_id}.png"

    new_item = {
        "id": new_id,
        "parent_id": req.item_id,
        "status": "pending",
        "filename": filename,
        "comfyName": None,
        "prompt_id": prompt_id,
        "prompt": DEFAULT_UPSCALE_PROMPT,
        "prompt_original": "Upscale (2MP)",
        "bypass_llm": True,
        "upscaled": True,
        "megapixels": 2.0
    }

    async with state_lock:
        current_state = _load()
        current_state.insert(0, new_item)
        _save(current_state)
    return new_item

# ── Bulk Delete (Cascade Deletion with ComfyUI Cancellation) ──────────────────
class DeleteRequest(BaseModel):
    ids: list[str]

@app.post("/api/items/delete")
async def delete_items(req: DeleteRequest) -> dict[str, Any]:
    settings = load_settings()
    async with state_lock:
        state = _load()
        all_removed_ids = []
        current_state = state
        
        for item_id in req.ids:
            if any(i["id"] == item_id for i in current_state):
                current_state, removed_ids = delete_item_reparent(item_id, current_state)
                all_removed_ids.extend(removed_ids)
                
        _save(current_state)
        
        # Clean up files and directories recursively on disk, and cancel pending tasks
        for r_id in all_removed_ids:
            item = next((i for i in state if i["id"] == r_id), None)
            # Clean up active progress tracking if any
            if item and item.get("prompt_id"):
                active_progress.pop(item["prompt_id"], None)

            if item:
                # 1. Cancel in ComfyUI queue if item is currently pending execution
                if item.get("status") == "pending" and item.get("prompt_id"):
                    async with httpx.AsyncClient() as client:
                        try:
                            await client.post(f"{settings.comfy_url}/queue", json={"delete": [item["prompt_id"]]})
                            await client.post(f"{settings.comfy_url}/interrupt")
                        except Exception as ex:
                            print(f"Error canceling pending ComfyUI task {item['prompt_id']}: {ex}")
                
                # 2. Remove the corresponding image file and its thumbnail
                if item.get("filename"):
                    file_path = os.path.join(STARTUP_IMG_DIR, item["filename"])
                    if os.path.exists(file_path):
                        try:
                            os.remove(file_path)
                        except OSError:
                            pass
                    thumb_path = thumbnail_path_for(file_path)
                    if os.path.exists(thumb_path):
                        try:
                            os.remove(thumb_path)
                        except OSError:
                            pass
            
            # 3. Remove the directory of descendants associated with this item ID (images and thumbnails)
            for base_dir in (STARTUP_IMG_DIR, STARTUP_THUMB_DIR):
                dir_path = os.path.join(base_dir, r_id)
                if os.path.exists(dir_path):
                    try:
                        shutil.rmtree(dir_path, ignore_errors=True)
                    except OSError:
                        pass
                
    return {"status": "ok", "deleted_count": len(all_removed_ids)}

# ── Dynamic Settings ──────────────────────────────────────────────────────────
class ConfigSchema(BaseModel):
    system_prompt: str
    llm_provider: str = "deepseek"
    deepseek_model: str = "deepseek-flash"
    openrouter_models: list[str] = []
    diffusion_model_name: str
    clip_model_name: str
    vae_model_name: str
    comfy_url: str
    comfy_root: Optional[str] = None
    # API keys are write-only: omit them (or send an empty string) to keep the
    # stored secret, send a new value to replace it, or set the matching
    # clear_* flag to delete it.
    deepseek_api_key: Optional[str] = None
    openrouter_api_key: Optional[str] = None
    clear_deepseek_api_key: bool = False
    clear_openrouter_api_key: bool = False

@app.get("/api/config")
async def get_config() -> dict[str, Any]:
    settings = load_settings()
    return {
        "system_prompt": settings.system_prompt,
        "llm_provider": settings.llm_provider,
        "deepseek_model": settings.deepseek_model,
        "openrouter_models": settings.openrouter_models,
        "diffusion_model_name": settings.diffusion_model_name,
        "clip_model_name": settings.clip_model_name,
        "vae_model_name": settings.vae_model_name,
        "comfy_url": settings.comfy_url,
        "comfy_root": settings.comfy_root,
        "deepseek_configured": bool(settings.deepseek_api_key),
        "openrouter_configured": bool(settings.openrouter_api_key),
    }

@app.post("/api/config")
async def update_config(cfg: ConfigSchema) -> dict[str, str]:
    settings = load_settings()
    settings.system_prompt = cfg.system_prompt
    settings.llm_provider = cfg.llm_provider
    settings.deepseek_model = cfg.deepseek_model
    settings.openrouter_models = cfg.openrouter_models
    settings.diffusion_model_name = cfg.diffusion_model_name
    settings.clip_model_name = cfg.clip_model_name
    settings.vae_model_name = cfg.vae_model_name
    settings.comfy_url = cfg.comfy_url
    settings.comfy_root = cfg.comfy_root or ""
    if cfg.clear_deepseek_api_key:
        settings.deepseek_api_key = ""
    elif cfg.deepseek_api_key:
        settings.deepseek_api_key = cfg.deepseek_api_key
    if cfg.clear_openrouter_api_key:
        settings.openrouter_api_key = ""
    elif cfg.openrouter_api_key:
        settings.openrouter_api_key = cfg.openrouter_api_key
    save_settings(settings)
    return {"status": "ok"}

@app.get("/api/comfy/models")
async def get_comfy_models() -> dict[str, list[str]]:
    settings = load_settings()
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            res = await client.get(f"{settings.comfy_url}/object_info")
            res.raise_for_status()
            info = cast(dict[str, Any], res.json())

        def extract_options(node_name: str, field_name: str) -> list[str]:
            node = info.get(node_name, {})
            field = node.get("input", {}).get("required", {}).get(field_name)
            if not field or not isinstance(field, list) or len(field) == 0:
                return []
            # Format: [["opt1", "opt2", ...]] — simple list in first element
            if isinstance(field[0], list):
                return cast(list[str], field[0])
            # Format: ["COMBO", {"options": [...]}] — COMBO type with options dict
            if len(field) > 1 and isinstance(field[1], dict) and "options" in field[1]:
                return cast(list[str], field[1]["options"])
            return []

        return {
            "diffusion_models": extract_options("UNETLoader", "unet_name"),
            "clip_models": extract_options("CLIPLoader", "clip_name"),
            "clip_types": extract_options("CLIPLoader", "type"),
            "vae_models": extract_options("VAELoader", "vae_name"),
        }
    except Exception as e:
        print(f"Error fetching ComfyUI models: {e}")
        return {
            "diffusion_models": [],
            "clip_models": [],
            "clip_types": [],
            "vae_models": [],
        }

@app.get("/api/comfy-status")
async def get_comfy_status() -> dict[str, str]:
    settings = load_settings()
    try:
        async with httpx.AsyncClient(timeout=1.0) as client:
            resp = await client.get(f"{settings.comfy_url}/queue")
            if resp.status_code == 200:
                return {"status": "connected"}
    except Exception:
        pass
    return {"status": "disconnected"}

app.mount("/images", StaticFiles(directory=STARTUP_IMG_DIR), name="images")

# ── Thumbnail serving (lazy generation) ──────────────────────────────────────
def _safe_child(root: Path | str, relative: str) -> Path:
    """Resolve `relative` beneath `root`, rejecting any escape via `..` or symlinks."""
    root_path = Path(root).resolve()
    candidate = (root_path / relative).resolve()
    if not candidate.is_relative_to(root_path):
        raise HTTPException(status_code=404, detail="Image not found")
    return candidate


@app.get("/thumbnails/{path:path}")
async def serve_thumbnail(path: str) -> FileResponse:
    # Map thumbnail path to image path (thumb: .jpg → image: .png)
    thumb_path = _safe_child(STARTUP_THUMB_DIR, path)
    if thumb_path.exists():
        return FileResponse(thumb_path)

    # Generate on demand from full image
    img_path = _safe_child(STARTUP_IMG_DIR, str(Path(path).with_suffix(".png")))
    if not img_path.exists():
        raise HTTPException(status_code=404, detail="Image not found")

    try:
        render_thumbnail(img_path, thumb_path)
        return FileResponse(thumb_path)
    except Exception as e:
        print(f"Thumbnail gen error: {e}")
        return FileResponse(img_path)

# ── Dynamic Frontend HTML Loading ─────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
def serve_frontend() -> HTMLResponse:
    frontend_path = Path(__file__).with_name("frontend.html")
    if frontend_path.exists():
        with frontend_path.open("r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    else:
        raise HTTPException(status_code=404, detail="frontend.html not found.")

@app.get("/manifest.json")
async def get_manifest() -> FileResponse:
    manifest_path = Path(__file__).with_name("manifest.json")
    if manifest_path.exists():
        return FileResponse(manifest_path, media_type="application/json")
    raise HTTPException(status_code=404, detail="manifest.json not found")

@app.get("/sw.js")
async def get_sw() -> FileResponse:
    sw_path = Path(__file__).with_name("sw.js")
    if sw_path.exists():
        return FileResponse(sw_path, media_type="application/javascript")
    raise HTTPException(status_code=404, detail="sw.js not found")

@app.get("/icon.svg")
async def get_icon() -> FileResponse:
    icon_path = Path(__file__).with_name("icon.svg")
    if icon_path.exists():
        return FileResponse(icon_path, media_type="image/svg+xml")
    raise HTTPException(status_code=404, detail="icon.svg not found")

def _parse_bind_args(default_host: str, default_port: int) -> tuple[str, int]:
    """Read --host/--bind/--port from argv, falling back to the configured defaults."""
    host, port = default_host, default_port
    argv = sys.argv[1:]
    for index, arg in enumerate(argv):
        value = argv[index + 1] if index + 1 < len(argv) else ""
        if arg in ("--host", "--bind") and value:
            host = value
        elif arg.startswith(("--host=", "--bind=")):
            host = arg.split("=", 1)[1]
        elif arg == "--port" and value:
            port = int(value)
        elif arg.startswith("--port="):
            port = int(arg.split("=", 1)[1])
    return host, port


if __name__ == "__main__":
    # Binds to all interfaces by default so the gallery is reachable over the
    # local network (Wi-Fi). Override via REIMAGINEX_HOST/REIMAGINEX_PORT or
    # with --host/--bind and --port.
    bind_host, bind_port = _parse_bind_args(startup_settings.host, startup_settings.port)
    uvicorn.run("src.server:app", host=bind_host, port=bind_port, reload=False)
