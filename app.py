import streamlit as st
import requests
import json
import uuid
import random
from PIL import Image
import io

st.set_page_config(page_title="Galeria ComfyUI", layout="wide")

COMFY_URL = "http://127.0.0.1:8001"

# --- INICIALIZAÇÃO DE ESTADOS ---
# O grid contém dicionários: id, status ('completed' ou 'pending'), img, prompt_id, nome_comfy
if "grid" not in st.session_state:
    st.session_state.grid = []
if "visualizar_id" not in st.session_state:
    st.session_state.visualizar_id = None
WORKFLOW_BASE = {
  "1":  {"inputs": {"cfg": 1, "model": ["50", 0], "positive": ["21", 0], "negative": ["23", 0]}, "class_type": "CFGGuider"},
  "2":  {"inputs": {"text": "PROMPT_AQUI", "clip": ["50", 1]}, "class_type": "CLIPTextEncode"},
  "3":  {"inputs": {"filename_prefix": "Flux2-Klein-Edit", "images": ["14", 0]}, "class_type": "SaveImage"},
  "5":  {"inputs": {"sampler_name": "euler"}, "class_type": "KSamplerSelect"},
  "6":  {"inputs": {"clip_name": "qwen_3_8b_fp8mixed.safetensors", "type": "flux2", "device": "default"}, "class_type": "CLIPLoader"},
  "7":  {"inputs": {"steps": 12, "width": ["40", 0], "height": ["41", 0]}, "class_type": "Flux2Scheduler"},
  "8":  {"inputs": {"vae_name": "flux2-vae.safetensors"}, "class_type": "VAELoader"},
  "9":  {"inputs": {"value": 1}, "class_type": "PrimitiveInt"},
  "10": {"inputs": {"image": ["30", 0]}, "class_type": "GetImageSize"},
  "11": {"inputs": {"noise": ["12", 0], "guider": ["1", 0], "sampler": ["5", 0], "sigmas": ["7", 0], "latent_image": ["13", 0]}, "class_type": "SamplerCustomAdvanced"},
  "12": {"inputs": {"noise_seed": ["16", 0]}, "class_type": "RandomNoise"},
  "13": {"inputs": {"width": ["40", 0], "height": ["41", 0], "batch_size": ["9", 0]}, "class_type": "EmptyFlux2LatentImage"},
  "14": {"inputs": {"samples": ["11", 0], "vae": ["8", 0]}, "class_type": "VAEDecode"},
  "15": {"inputs": {"pixels": ["30", 0], "vae": ["8", 0]}, "class_type": "VAEEncode"},
  "16": {"inputs": {"seed": 0}, "class_type": "Seed (rgthree)"},
  "19": {"inputs": {"rgthree_comparer": {"images": [{"name":"A","selected":True,"url":""},{"name":"B","selected":True,"url":""}]}, "image_a": ["14", 0], "image_b": ["25", 0]}, "class_type": "Image Comparer (rgthree)"},
  "21": {"inputs": {"conditioning": ["2", 0], "latent": ["15", 0]}, "class_type": "ReferenceLatent"},
  "22": {"inputs": {"conditioning": ["2", 0]}, "class_type": "ConditioningZeroOut"},
  "23": {"inputs": {"conditioning": ["22", 0], "latent": ["15", 0]}, "class_type": "ReferenceLatent"},
  "25": {"inputs": {"image": "IMAGEM_BASE_AQUI"}, "class_type": "LoadImage"},
  "30": {"inputs": {"upscale_method": "lanczos", "megapixels": 1, "resolution_steps": 64, "image": ["25", 0]}, "class_type": "ImageScaleToTotalPixels"},
  "40": {"inputs": {"value": TARGET_W}, "class_type": "PrimitiveInt"},
  "41": {"inputs": {"value": TARGET_H}, "class_type": "PrimitiveInt"},
  "34": {"inputs": {"model_name": "miracleinNSFWGeneration_30Bf16Fp8.safetensors", "weight_dtype": "fp8_e4m3fn", "compute_dtype": "default", "patch_cublaslinear": False, "sage_attention": "sageattn3", "enable_fp16_accumulation": False}, "class_type": "DiffusionModelLoaderKJ"},
  "50": {
    "inputs": {
      "lora_name": "seu_lora_aqui.safetensors", 
      "strength_model": 1.0, 
      "strength_clip": 1.0, 
      "model": ["34", 0], 
      "clip": ["6", 0]
    }, 
    "class_type": "LoraLoader"
  }
}

# --- FUNÇÕES DE COMUNICAÇÃO ---
def upload_imagem_comfyui(imagem_pil):
    """Envia a imagem para o servidor e retorna o nome no backend."""
    buf = io.BytesIO()
    imagem_pil.save(buf, format="PNG")
    buf.seek(0)
    files = {"image": ("base_image.png", buf, "image/png")}
    try:
        response = requests.post(f"{COMFY_URL}/upload/image", files=files)
        response.raise_for_status()
        return response.json()["name"]
    except Exception as e:
        st.error(f"Erro no upload: {e}")
        return None

def enviar_para_fila(prompt_text, nome_imagem_base):
    """Envia o workflow para o ComfyUI e retorna o prompt_id imediatamente, sem bloquear."""
    workflow = json.loads(json.dumps(workflow_base))
    workflow["2"]["inputs"]["text"] = prompt_text
    workflow["25"]["inputs"]["image"] = nome_imagem_base
    workflow["16"]["inputs"]["seed"] = random.randint(1, 999999999999999)
    
    try:
        res = requests.post(f"{COMFY_URL}/prompt", json={"prompt": workflow})
        res.raise_for_status()
        return res.json()["prompt_id"]
    except Exception as e:
        st.error(f"Erro ao enviar para a fila: {e}")
        return None

def atualizar_tarefas_pendentes():
    """Consulta o histórico global do ComfyUI e atualiza as imagens que terminaram."""
    pendentes = [item for item in st.session_state.grid if item["status"] == "pending"]
    if not pendentes:
        return

    try:
        res_hist = requests.get(f"{COMFY_URL}/history")
        res_hist.raise_for_status()
        historico = res_hist.json()
        
        for item in pendentes:
            pid = item["prompt_id"]
            if pid in historico:
                outputs = historico[pid].get("outputs", {})
                if "3" in outputs and "images" in outputs["3"]:
                    img_data = outputs["3"]["images"][0]
                    url_img = f"{COMFY_URL}/view?filename={img_data['filename']}&subfolder={img_data['subfolder']}&type={img_data['type']}"
                    
                    img_req = requests.get(url_img)
                    nova_img = Image.open(io.BytesIO(img_req.content))
                    
                    # Atualiza o item na memória
                    item["img"] = nova_img
                    item["status"] = "completed"
                    # Faz o upload silencioso para ser possível usá-la como base no futuro
                    item["nome_comfy"] = upload_imagem_comfyui(nova_img)
    except Exception as e:
        st.sidebar.error(f"Erro ao verificar estado: {e}")

# Executa a verificação sempre que a interface recarregar
atualizar_tarefas_pendentes()


# --- INTERFACE ---
st.sidebar.title("Ações")
if st.sidebar.button("🔄 Atualizar Estado", use_container_width=True):
    st.rerun()

# 1. VISUALIZAÇÃO MAXIMIZADA (MODO DE EDIÇÃO)
if st.session_state.visualizar_id is not None:
    # Procura a imagem pelo ID
    item_atual = next((x for x in st.session_state.grid if x["id"] == st.session_state.visualizar_id), None)
    
    if item_atual and item_atual["status"] == "completed":
        st.button("⬅️ Voltar à Galeria", on_click=lambda: st.session_state.update(visualizar_id=None))
        
        col_img, col_prompt = st.columns([2, 1])
        with col_img:
            st.image(item_atual["img"], use_container_width=True)
            
        with col_prompt:
            st.subheader("Gerar Variação")
            novo_prompt = st.text_area("Insira o prompt:")
            
            if st.button("Colocar na Fila 🚀", type="primary"):
                if novo_prompt:
                    prompt_id = enviar_para_fila(novo_prompt, item_atual["nome_comfy"])
                    if prompt_id:
                        novo_item = {
                            "id": str(uuid.uuid4()),
                            "status": "pending",
                            "img": None,
                            "nome_comfy": None,
                            "prompt_id": prompt_id
                        }
                        st.session_state.grid.append(novo_item)
                        st.session_state.visualizar_id = None # Volta para a galeria
                        st.rerun()
                else:
                    st.error("Insira um prompt.")
    else:
        st.session_state.visualizar_id = None
        st.rerun()

# 2. GALERIA (GRID)
else:
    st.subheader("Galeria")
    
    upload = st.file_uploader("Upload de imagem inicial", type=["png", "jpg", "jpeg"])
    if upload and st.button("Adicionar à Galeria"):
        img_pil = Image.open(upload)
        nome = upload_imagem_comfyui(img_pil)
        if nome:
            st.session_state.grid.append({
                "id": str(uuid.uuid4()),
                "status": "completed",
                "img": img_pil,
                "nome_comfy": nome,
                "prompt_id": None
            })
            st.rerun()

    st.divider()

    if not st.session_state.grid:
        st.info("A galeria está vazia.")
    else:
        cols = st.columns(4)
        for i, item in enumerate(reversed(st.session_state.grid)): # Mais recentes primeiro
            with cols[i % 4]:
                if item["status"] == "pending":
                    st.info("⏳ A processar no ComfyUI...")
                else:
                    st.image(item["img"], use_container_width=True)
                    if st.button("Expandir", key=f"btn_{item['id']}"):
                        st.session_state.visualizar_id = item["id"]
                        st.rerun()
