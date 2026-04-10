# Ovi Fusion Model – Fine-tuning Guide (v2)

> **Key difference from v1:** This v2 pipeline is built directly on top of Ovi's own codebase — it calls the same `init_fusion_score_model_ovi`, `load_fusion_checkpoint`, `FusionModel.forward()`, `NAME_TO_MODEL_SPECS_MAP`, VAE/T5 loaders, and processing utilities that the inference engine uses. Nothing is reimplemented; everything is imported.

---

## Architecture Recap (from code study)

| Component | Ovi Module | What it does |
|---|---|---|
| **FusionModel** | `ovi.modules.fusion.FusionModel` | Wraps a video `WanModel` and an audio `WanModel` (30 transformer blocks each). Cross-modal fusion is done via injected `k_fusion` / `v_fusion` linear projections + `pre_attn_norm_fusion` in every block. |
| **WanModel** | `ovi.modules.model.WanModel` | DiT backbone with 3D-RoPE (video) or 1D-RoPE (audio), sinusoidal time embeddings, per-token modulation, self-attention + cross-attention + FFN per block. |
| **Forward signature** | `FusionModel.forward(vid, audio, t, vid_context, audio_context, vid_seq_len, audio_seq_len, ...)` | `vid`/`audio` are **lists of tensors** (one per sample). `t` is a 1-D timestep tensor. Returns `(List[pred_vid], List[pred_aud])`. |
| **Video VAE** | `Wan2.2_VAE` via `init_wan_vae_2_2` | Spatial 16× downsample, temporal 4× downsample. Latent channels = 48. |
| **Audio VAE** | `MMAudio FeaturesUtils` via `init_mmaudio_vae` | Waveform → latent with 20 channels. |
| **Text encoder** | `T5EncoderModel` via `init_text_model` | UMT5-XXL, max 512 tokens, bf16. |

### Model Variants

| `model_name` | Checkpoint | Video latents (F) | Audio latents (L) | Target pixel area |
|---|---|---|---|---|
| `720x720_5s` | `model.safetensors` | 31 | 157 | 518 400 |
| `960x960_5s` | `model_960x960.safetensors` | 31 | 157 | 921 600 |
| `960x960_10s` | `model_960x960_10s.safetensors` | 61 | 314 | 921 600 |

---

## Prerequisites

```bash
pip install -r requirements.txt
pip install omegaconf tensorboard opencv-python torchaudio
```

Download pretrained checkpoints into `./ckpts/` (see `download_weights.py`). The expected layout:

```
ckpts/
├── Ovi/
│   ├── model.safetensors            (720x720_5s)
│   ├── model_960x960.safetensors    (960x960_5s)
│   └── model_960x960_10s.safetensors(960x960_10s)
├── Wan2.2-TI2V-5B/
│   ├── Wan2.2_VAE.pth
│   ├── models_t5_umt5-xxl-enc-bf16.pth
│   └── google/umt5-xxl/
└── MMAudio/
    └── ext_weights/
        ├── v1-16.pth
        └── best_netG.pt
```

---

## Step 1 – Prepare Your Data

### 1a. Create a raw manifest

```jsonl
{"video_path": "/data/videos/clip001.mp4", "text_prompt": "A dog running on a beach at sunset"}
{"video_path": "/data/videos/clip002.mp4", "text_prompt": "Two people dancing in a kitchen", "image_path": "/data/images/clip002_frame0.png"}
```

### 1b. Run the v2 preparation script

```bash
python prepare_training_data_v2.py \
    --input_manifest  training_data/raw_manifest.jsonl \
    --output_dir      training_data \
    --ckpt_dir        ./ckpts \
    --model_name      960x960_5s \
    --device          0
```

This calls Ovi's own `init_wan_vae_2_2`, `init_mmaudio_vae`, and `init_text_model` to encode each sample. Outputs:

```
training_data/
├── latents/
│   ├── 000000.pt   # {video_latent, audio_latent, text_embedding, [first_frame_latent]}
│   ├── 000001.pt
│   └── ...
└── manifest.jsonl   # consumed by train_v2.py
```

---

## Step 2 – Train

### Single GPU

```bash
python train_v2.py --config ovi/configs/training/training_fusion_v2.yaml
```

### Multi-GPU (e.g. 4 GPUs)

```bash
torchrun --nproc_per_node=4 train_v2.py --config ovi/configs/training/training_fusion_v2.yaml
```

### What the training script does internally

1. **Builds the model** with `init_fusion_score_model_ovi(meta_init=True)` — same factory as `OviFusionEngine.__init__`.
2. **Loads pretrained weights** with `load_fusion_checkpoint(model, ckpt_path, from_meta=True)` — same loader.
3. **Calls `model.set_rope_params()`** after `.to(device)` — same as inference.
4. **Enables gradient checkpointing** via `video_model.set_gradient_checkpointing(True)` and `audio_model.set_gradient_checkpointing(True)` — Ovi's own method.
5. **Forward pass** calls `FusionModel.forward(vid=..., audio=..., t=..., vid_context=..., audio_context=..., vid_seq_len=..., audio_seq_len=..., first_frame_is_clean=...)` — the **exact same signature** used in `OviFusionEngine.generate()`.
6. **Loss** is MSE on the flow-matching velocity target: `v = noise - x_0`.
7. **Timestep sampling** uses logit-normal + the same shift formula from Ovi's scheduler.

---

## Config Reference (`training_fusion_v2.yaml`)

| Parameter | Default | Description |
|---|---|---|
| `ckpt_dir` | `./ckpts` | Checkpoint root (same as inference) |
| `model_name` | `960x960_5s` | Must match `NAME_TO_MODEL_SPECS_MAP` |
| `training.batch_size` | 1 | Per-GPU batch size |
| `training.gradient_accumulation_steps` | 8 | Effective batch = batch_size × accum × num_gpus |
| `training.learning_rate` | 1e-5 | Peak learning rate |
| `training.max_steps` | 50000 | Total training steps |
| `training.warmup_steps` | 1000 | Linear warmup steps |
| `training.lr_scheduler` | `cosine` | `cosine` or `constant` |
| `training.save_every` | 2000 | Checkpoint interval |
| `finetune.mode` | `full` | `full` = all params; `fusion_only` = only k/v_fusion + norms |
| `finetune.gradient_checkpointing` | `true` | Reduces VRAM ~40% |
| `flow.shift` | 5.0 | Timestep distribution shift (matches inference) |
| `flow.cfg_drop_rate` | 0.1 | Classifier-free guidance dropout probability |
| `loss.video_weight` | 1.0 | Weight for video velocity loss |
| `loss.audio_weight` | 1.0 | Weight for audio velocity loss |

---

## Step 3 – Resume Training

Edit the config or pass via CLI:

```bash
# Load from checkpoint, then continue
python train_v2.py --config ovi/configs/training/training_fusion_v2.yaml
# (manually load checkpoint_latest.pt inside train_v2.py — see "resume" section in code)
```

Or modify `train_v2.py` to accept `--resume` flag pointing to `checkpoint_latest.pt`. The checkpoint contains both `model` and `optimizer` state dicts.

---

## Step 4 – Inference with Your Fine-tuned Model

Because our checkpoints use `FusionModel.state_dict()` (the same keys as Ovi's `.safetensors`), you can load them directly with `load_fusion_checkpoint`:

```python
from ovi.ovi_fusion_engine import OviFusionEngine
from ovi.utils.model_loading_utils import load_fusion_checkpoint

# Option A: patch the engine after init
engine = OviFusionEngine(config)
load_fusion_checkpoint(engine.model, "training_outputs_v2/checkpoint_final.pt", from_meta=False)

# Option B: use inference.py with custom ckpt_dir
# Copy your checkpoint_final.pt → ckpts/Ovi/model_960x960.safetensors (or the appropriate name)
```

Or from the CLI:

```bash
python inference.py --ckpt_dir ./training_outputs_v2 --task t2v --prompt "Your prompt"
```

---

## Monitoring

```bash
tensorboard --logdir training_outputs_v2/tb_logs
```

Tracked: `train/loss`, `train/video_loss`, `train/audio_loss`, `train/lr`.

---

## Tips

- **VRAM**: With `gradient_checkpointing: true` and `batch_size: 1`, the 960×960 model fits on ~40 GB. Use `fusion_only` mode for ~24 GB.
- **fusion_only** mode freezes all parameters except `k_fusion`, `v_fusion`, `pre_attn_norm_fusion`, and `norm_k_fusion` — the cross-modal attention layers injected by `FusionModel.inject_cross_attention_kv_projections()`. This is the most parameter-efficient option.
- **Audio format**: The text prompt should include audio descriptions using the format expected by your model variant (e.g., `Audio: ...` for 960×960 models, `<AUDCAP>...<ENDAUDCAP>` for 720×720).
- **Effective batch size**: `batch_size × gradient_accumulation_steps × num_gpus`. The default (1 × 8 × 1 = 8) is a reasonable starting point.

---

## Files Created

| File | Purpose |
|---|---|
| `ovi/configs/training/training_fusion_v2.yaml` | Training configuration |
| `prepare_training_data_v2.py` | Data encoding (uses Ovi's VAE/T5 loaders) |
| `train_v2.py` | Training loop (uses Ovi's model factory + forward) |
| `TRAINING_V2.md` | This document |
