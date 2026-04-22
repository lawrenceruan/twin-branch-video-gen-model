# Ovi Fusion Model – Training Guide

This document describes how to fine-tune the Ovi Fusion Model (joint video + audio DiT) on your own data.

---

## Overview

The training pipeline has **two stages**:

1. **Data preparation** (`prepare_training_data.py`) – Encodes raw videos, audio, and text prompts into latent tensors using the frozen VAEs and T5 encoder. This is done **once** offline.
2. **Training** (`train.py`) – Trains the FusionModel DiT on the pre-encoded latents using a flow-matching velocity objective.

---

## Prerequisites

```bash
pip install -r requirements.txt
pip install omegaconf tensorboard opencv-python torchaudio
```

You also need the pretrained checkpoints downloaded into `./ckpts/` (see `download_weights.py`).

---

## Step 1 – Prepare Your Data

### 1a. Create a raw manifest

Create a JSONL file where each line is a JSON object:

```jsonl
{"video_path": "/data/videos/clip001.mp4", "text_prompt": "A dog running on a beach at sunset"}
{"video_path": "/data/videos/clip002.mp4", "text_prompt": "Two people dancing in a kitchen", "image_path": "/data/images/clip002_frame0.png"}
```

- `video_path` (required) – path to an MP4 video file
- `text_prompt` (required) – the text description
- `image_path` (optional) – first-frame image for Image-to-Video (I2V) training

### 1b. Run the preparation script

```bash
python prepare_training_data.py \
    --input_manifest  /project/llmsvgen/share/data_videoaudio/filtered_videos/captions.jsonl \
    --output_dir      training_data \
    --ckpt_dir        ./ckpts \
    --model_name      960x960_5s \
    --device          0
```

This produces:
- `training_data/latents/*.pt` – one file per sample containing `video_latent`, `audio_latent`, `text_embedding`, and optionally `first_frame_latent`
- `training_data/manifest.jsonl` – the manifest consumed by the training dataloader

---

## Step 2 – Train

### Single GPU

```bash
python train.py --config ovi/configs/training/training_fusion.yaml
```

### Multi-GPU (e.g. 4 GPUs)

```bash
torchrun --nproc_per_node=4 train.py --config ovi/configs/training/training_fusion.yaml
```

### Key config options (`ovi/configs/training/training_fusion.yaml`)

| Parameter | Default | Description |
|---|---|---|
| `model_name` | `960x960_5s` | Model variant (`720x720_5s`, `960x960_5s`, `960x960_10s`) |
| `training.batch_size` | 1 | Per-GPU batch size |
| `training.gradient_accumulation_steps` | 8 | Accumulation steps |
| `training.learning_rate` | 1e-5 | Learning rate |
| `training.max_steps` | 100000 | Total training steps |
| `model.finetune_mode` | `full` | `full` (all params) or `fusion_only` (cross-modal layers only) |
| `model.gradient_checkpointing` | true | Save VRAM via gradient checkpointing |
| `mixed_precision` | `bf16` | `bf16` or `fp32` |
| `loss.video_weight` | 1.0 | Weight for video loss |
| `loss.audio_weight` | 1.0 | Weight for audio loss |

---

## Step 3 – Resume Training

Set `model.resume_from` in the config to the path of a saved checkpoint:

```yaml
model:
  resume_from: ./training_outputs/checkpoint_latest.pt
```

---

## Step 4 – Use Your Trained Model for Inference

After training, point the inference script to your checkpoint:

```bash
python inference.py --ckpt_dir ./training_outputs --task t2v --prompt "rabbit"
```

Or load it in code by replacing the default fusion checkpoint path with your `checkpoint_final.pt`.

---

## Monitoring

TensorBoard logs are written to `training_outputs/tb_logs/` by default:

```bash
tensorboard --logdir training_outputs/tb_logs
```

Tracked metrics: `train/loss`, `train/video_loss`, `train/audio_loss`, `train/lr`.
