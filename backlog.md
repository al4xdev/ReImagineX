# Backlog

## Para publicar no GitHub (prioridade alta)

- **[x] Criar README.md** — explicar o que é o projeto, como rodar localmente, e os pré-requisitos do ComfyUI.
- **[ ] Mover hardcoded model names para lugar mais visível** — hoje os nomes de modelos estão espalhados entre `config.py` (upscale/diffusion defaults), `workflow.py` (CLIP, VAE, upscale de input). O ideal é centralizar tudo num lugar só (`config.py` ou um `models.toml`) pra facilitar troca.
- **[x] README: documentar ambiente ComfyUI necessário** — avisar que a pessoa precisa desses modelos e custom nodes (ver seção abaixo).

## ComfyUI — Pré-requisitos

### Modelos que precisam estar na pasta `models/` do ComfyUI

| Modelo | Tipo | Usado em |
|---|---|---|
| `flux1-dev-fp8.safetensors` | Diffusion (Flux2) | `config.py`, node 25 |
| `clip_l.safetensors` | CLIP (flux2) | node 5 |
| `ae.safetensors` | VAE | node 7 |
| `4x_foolhardy_Remacri.pth` | Upscale (output) | `config.py`, node 16 |
| `4x_foolhardy_Remacri.pth` | Upscale (input) | node 27 |

### Custom Nodes necessários no ComfyUI

- **rgthree** — `Seed (rgthree)`, `Image Comparer (rgthree)`
- **DiffusionModelLoaderKJ** — loader customizado para o modelo Flux2 Klein
- **Flux2 nodes** — `Flux2Scheduler`, `ReferenceLatent`, `CFGGuider`, `EmptyFlux2LatentImage`

## Melhorias de código (médio/baixo)

- **[ ] Separar frontend.html** — 877 linhas com HTML+CSS+JS inline. Ideal extrair CSS e JS pra arquivos próprios ou migrar pra um bundler simples.
- **[ ] Tipagem** — `state_manager.py` e partes do `server.py` usam `list`/`dict` sem type hints. Adicionar `TypedDict` ou model Pydantic pro schema de item da galeria.
- **[ ] Tratamento de erro no polling** — `check_comfy_queue` só dá `print` nos erros. Podia ter retry com backoff e um log estruturado.
- **[ ] Testes** — zero cobertura. Mínimo: testar `delete_item_recursive` e `build_generation_workflow`.
- **[ ] Config.json em runtime salva API key em plain text** — o endpoint `POST /api/config` persiste a key no `gallery_data/config.json`. Seria melhor usar um secret manager ou pelo menos `keyring`.
