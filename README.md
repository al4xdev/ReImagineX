# ReImagineX ✦

[![CI](https://github.com/al4xdev/ReImagineX/actions/workflows/ci.yml/badge.svg)](https://github.com/al4xdev/ReImagineX/actions/workflows/ci.yml)
[![Container](https://github.com/al4xdev/ReImagineX/actions/workflows/docker-publish.yml/badge.svg)](https://github.com/al4xdev/ReImagineX/actions/workflows/docker-publish.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-10b981.svg)](LICENSE)

ReImagineX is a local-first, mobile-friendly image-to-image gallery for ComfyUI. Upload an image, create variations, inspect prompts, upscale results, and navigate the complete generation lineage without leaving the browser.

The interface is inspired by the fast, gesture-driven workflow of modern image tools while keeping generation, images, and configuration under your control.

> [!NOTE]
> ReImagineX is a portfolio release provided as open-source software. It has no support SLA, but issues and community contributions are welcome.

## Preview

| Main gallery | Prompt details | Variation editor |
| :---: | :---: | :---: |
| ![Main gallery](assets/main_menu.png) | ![Prompt details drawer](assets/show_prompts.png) | ![Variation editor](assets/edit_menu.png) |

## Why ReImagineX

Typical ComfyUI image-to-image iteration involves moving repeatedly between a workflow canvas, an output folder, and a prompt editor. ReImagineX keeps that loop in one visual history:

- each uploaded image becomes the root of a lineage;
- every variation remains connected to its source;
- prompts and generation options stay attached to each result;
- reprocessing and upscaling are available from the image viewer;
- pending jobs expose progress and can be cancelled;
- the UI is installable as a PWA and designed for touch gestures.

## Features

- Lineage-based galleries for root images and their variations.
- Mobile gestures: swipe up for details, down to close, and sideways to navigate.
- Optional OpenRouter prompt expansion with configurable fallback models.
- Direct prompt mode when no API key is configured or LLM expansion is bypassed.
- Dynamic discovery of diffusion, CLIP, VAE, and upscale models from ComfyUI.
- Input upscaling, output upscaling, reprocessing, and consistency controls.
- ComfyUI queue monitoring through HTTP and WebSocket events.
- Local JSON persistence with atomic writes.
- Thumbnail generation and optional hard-link synchronization with ComfyUI output.
- Docker images for AMD64 and ARM64.

## Requirements

- A running ComfyUI instance.
- The custom nodes referenced by [`workflow_api.json`](workflow_api.json).
- Compatible Flux 2 / Klein model, text encoder, VAE, and upscale models.
- Python 3.12+ and [uv](https://docs.astral.sh/uv/) when running from source.

The bundled defaults match the workflow used during development:

| Component | Default filename |
|---|---|
| Diffusion model | `flux-2-klein-9b-nvfp4.safetensors` |
| Text encoder | `qwen_3_8b_fp8mixed.safetensors` |
| VAE | `full_encoder_small_decoder.safetensors` |
| Output upscaler | `4x-UltraSharpV2.pth` |
| Input upscaler | `4x_foolhardy_Remacri.pth` |

These are defaults, not hardcoded requirements. The Settings panel discovers compatible models from the connected ComfyUI instance and lets you replace every filename.

## Run with Docker

```bash
docker run -d \
  --name reimaginex \
  --restart unless-stopped \
  -p 8888:8888 \
  --add-host host.docker.internal:host-gateway \
  -e COMFY_URL=http://host.docker.internal:8001 \
  -v reimaginex-data:/app/gallery_data \
  ghcr.io/al4xdev/reimaginex:latest
```

Open `http://localhost:8888`.

The same setup is available through Compose:

```bash
docker compose up -d
```

When ComfyUI runs on the host, set its URL to `http://host.docker.internal:8001` in the container. Change the port if your ComfyUI installation uses another one.

## Run from source

```bash
git clone https://github.com/al4xdev/ReImagineX.git
cd ReImagineX
cp .env.example .env
uv sync
./start.sh
```

Open `http://127.0.0.1:8888`.

The application binds to localhost by default. To expose it on a trusted private network:

```bash
REIMAGINEX_HOST=0.0.0.0 ./start.sh
```

## Configuration

Most settings are editable from the frontend:

- ComfyUI URL and optional local root path;
- OpenRouter API key, model fallback list, and system prompt;
- diffusion model, text encoder, VAE, and upscale models.

Environment variables can provide initial values:

```ini
COMFY_URL=http://127.0.0.1:8001
DATA_DIR=gallery_data
COMFY_ROOT=
OPENROUTER_API_KEY=
```

Runtime images, state, and saved configuration live under `gallery_data/` and are excluded from Git. The OpenRouter key is stored server-side and is never returned by the configuration API.

### ComfyUI workflow

[`workflow_api.json`](workflow_api.json) is the API-format workflow used by the application. It currently expects nodes including:

- `DiffusionModelLoaderKJ`;
- `Flux2Scheduler`;
- `ReferenceLatent`;
- `CFGGuider`;
- `EmptyFlux2LatentImage`;
- rgthree seed and image comparison nodes;
- standard CLIP, VAE, image, sampler, and upscale nodes.

Use ComfyUI Manager to identify and install missing custom nodes. Model options are loaded dynamically from ComfyUI after the workflow nodes are available.

## Development

```bash
uv sync --dev
uv run ruff check .
uv run mypy
uv run pytest
```

CI runs the same Ruff, strict mypy, and pytest checks on pushes and pull requests. A separate workflow builds and publishes multi-platform images to GitHub Container Registry.

## Security

ReImagineX has no authentication layer. Keep it bound to localhost or a trusted private network, especially when an OpenRouter key is configured. Do not expose the service directly to the public internet.

Generated images and prompt content remain local unless they are sent to ComfyUI or, when enabled, OpenRouter for prompt expansion.

## License

[MIT](LICENSE)
