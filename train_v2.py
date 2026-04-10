#!/usr/bin/env python3
"""
train_v2.py  –  Fine-tune the Ovi FusionModel using Ovi's native modules.

This script re-uses Ovi's own:
  • init_fusion_score_model_ovi   → build the DiT
  • load_fusion_checkpoint        → load pretrained weights
  • FusionModel.forward()         → exact same forward signature as inference
  • NAME_TO_MODEL_SPECS_MAP       → latent sizes, patch sizes, etc.

Usage (single GPU):
    python train_v2.py --config ovi/configs/training/training_fusion_v2.yaml

Usage (multi-GPU):
    torchrun --nproc_per_node=4 train_v2.py --config ovi/configs/training/training_fusion_v2.yaml
"""

import argparse
import gc
import json
import logging
import math
import os
import random
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from omegaconf import OmegaConf

# ── Ovi imports (reuse everything) ───────────────────────────────────────────
from ovi.ovi_fusion_engine import NAME_TO_MODEL_SPECS_MAP
from ovi.utils.model_loading_utils import (
    init_fusion_score_model_ovi,
    load_fusion_checkpoint,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# Dataset  – loads pre-encoded latents produced by prepare_training_data_v2.py
# ═══════════════════════════════════════════════════════════════════════════════

class OviLatentDataset(Dataset):
    """
    Each .pt file contains:
        video_latent     : [C=48, F, H_lat, W_lat]
        audio_latent     : [L, C_audio=20]
        text_embedding   : [text_len, D=4096]
        (opt) first_frame_latent : [C=48, 1, H_lat, W_lat]
    """

    def __init__(self, manifest_path: str):
        with open(manifest_path) as f:
            self.records = [json.loads(l) for l in f if l.strip()]
        logger.info(f"Dataset: {len(self.records)} samples from {manifest_path}")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        data = torch.load(rec["latent_path"], map_location="cpu", weights_only=True)
        return {
            "video_latent": data["video_latent"],        # [C, F, H, W]
            "audio_latent": data["audio_latent"],        # [L, C_audio]
            "text_embedding": data["text_embedding"],    # [text_len, D]
            "has_first_frame": rec.get("has_first_frame", False),
            "first_frame_latent": data.get("first_frame_latent", None),  # [C, 1, H, W] or None
        }


def collate_fn(batch):
    """Custom collate – keep lists (FusionModel.forward expects List[Tensor])."""
    return {
        "video_latent": [s["video_latent"] for s in batch],
        "audio_latent": [s["audio_latent"] for s in batch],
        "text_embedding": [s["text_embedding"] for s in batch],
        "has_first_frame": [s["has_first_frame"] for s in batch],
        "first_frame_latent": [s["first_frame_latent"] for s in batch],
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Flow-matching noise utilities  (matches Ovi inference exactly)
# ═══════════════════════════════════════════════════════════════════════════════

def sample_logit_normal_timesteps(batch_size: int, shift: float = 5.0, device="cpu"):
    """
    Sample t ∈ (0,1) from a logit-normal distribution then apply the
    shift  σ = shift·σ_raw / (1 + (shift-1)·σ_raw)  used in Ovi / Wan.
    Returns t ∈ (0, 1) as a 1-D tensor of shape [B].
    """
    u = torch.randn(batch_size, device=device)
    sigma_raw = torch.sigmoid(u)  # logit-normal in (0,1)
    sigma = shift * sigma_raw / (1.0 + (shift - 1.0) * sigma_raw)
    return sigma


def build_noisy_sample(x0, noise, t):
    """
    Flow-matching interpolation:  x_t = (1 - t) · x_0  +  t · noise
    velocity target:               v  = noise - x_0
    `t` is broadcastable to x0's shape.
    """
    x_t = (1.0 - t) * x0 + t * noise
    velocity = noise - x0
    return x_t, velocity


# ═══════════════════════════════════════════════════════════════════════════════
# Freeze helpers
# ═══════════════════════════════════════════════════════════════════════════════

def freeze_non_fusion(model: nn.Module):
    """Freeze everything except cross-modal fusion layers (k_fusion, v_fusion, etc.)."""
    for name, param in model.named_parameters():
        if "fusion" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info(f"fusion_only mode: {trainable:,} / {total:,} params trainable "
                f"({100*trainable/total:.2f}%)")


# ═══════════════════════════════════════════════════════════════════════════════
# Learning-rate schedule
# ═══════════════════════════════════════════════════════════════════════════════

def get_lr(step: int, warmup: int, max_steps: int, base_lr: float, schedule: str):
    if step < warmup:
        return base_lr * step / max(warmup, 1)
    if schedule == "constant":
        return base_lr
    # cosine decay
    progress = (step - warmup) / max(max_steps - warmup, 1)
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))


# ═══════════════════════════════════════════════════════════════════════════════
# Main training loop
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="ovi/configs/training/training_fusion_v2.yaml")
    parser.add_argument("--local_rank", type=int, default=-1)  # for torchrun
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)

    # ── DDP setup ────────────────────────────────────────────────────────
    distributed = int(os.environ.get("WORLD_SIZE", 1)) > 1
    if distributed:
        torch.distributed.init_process_group("nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        is_main = local_rank == 0
    else:
        local_rank = 0
        device = torch.device("cuda", 0)
        is_main = True

    # ── seed ─────────────────────────────────────────────────────────────
    seed = cfg.training.seed
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # ── model specs ──────────────────────────────────────────────────────
    model_name = cfg.get("model_name", "960x960_5s")
    assert model_name in NAME_TO_MODEL_SPECS_MAP, f"Unknown model_name: {model_name}"
    model_specs = NAME_TO_MODEL_SPECS_MAP[model_name]

    # ── build FusionModel using Ovi's own factory ────────────────────────
    logger.info("Initialising FusionModel via init_fusion_score_model_ovi …")
    model, video_config, audio_config = init_fusion_score_model_ovi(rank=0, meta_init=True)

    # ── load pretrained checkpoint ───────────────────────────────────────
    ckpt_path = os.path.join(cfg.ckpt_dir, "Ovi", model_specs["path"])
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Pretrained checkpoint not found: {ckpt_path}")
    logger.info(f"Loading pretrained weights from {ckpt_path}")
    load_fusion_checkpoint(model, checkpoint_path=ckpt_path, from_meta=True)

    # move to device + dtype
    target_dtype = torch.bfloat16 if cfg.get("mixed_precision", "bf16") == "bf16" else torch.float32
    model = model.to(dtype=target_dtype, device=device)
    model.set_rope_params()

    # ── gradient checkpointing ───────────────────────────────────────────
    if cfg.finetune.get("gradient_checkpointing", True):
        model.video_model.set_gradient_checkpointing(True)
        model.audio_model.set_gradient_checkpointing(True)
        # Also set on FusionModel for the fusion block loop
        model.gradient_checkpointing = True
        logger.info("Gradient checkpointing enabled for both video & audio models")

    # ── freeze policy ────────────────────────────────────────────────────
    finetune_mode = cfg.finetune.get("mode", "full")
    if finetune_mode == "fusion_only":
        freeze_non_fusion(model)
    else:
        model.requires_grad_(True)
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"full fine-tune: {trainable:,} params trainable")

    model.train()

    # ── DDP wrapper ──────────────────────────────────────────────────────
    if distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank], find_unused_parameters=False,
        )
        raw_model = model.module
    else:
        raw_model = model

    # ── dataset / dataloader ─────────────────────────────────────────────
    dataset = OviLatentDataset(cfg.data_manifest)
    if distributed:
        sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=True)
    else:
        sampler = None

    dataloader = DataLoader(
        dataset,
        batch_size=cfg.training.batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=cfg.data.get("num_workers", 4),
        pin_memory=cfg.data.get("pin_memory", True),
        collate_fn=collate_fn,
        drop_last=True,
    )

    # ── optimizer ────────────────────────────────────────────────────────
    trainable_params = [p for p in raw_model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=cfg.training.learning_rate,
        weight_decay=cfg.training.get("weight_decay", 0.01),
        betas=(0.9, 0.999),
    )

    # ── tensorboard ──────────────────────────────────────────────────────
    os.makedirs(cfg.output_dir, exist_ok=True)
    tb_dir = os.path.join(cfg.output_dir, "tb_logs")
    writer = SummaryWriter(tb_dir) if is_main else None

    # ── latent dims from configs ─────────────────────────────────────────
    video_latent_channel = video_config["in_dim"]   # 48
    audio_latent_channel = audio_config["in_dim"]   # 20
    video_latent_length = model_specs["video_latent_length"]
    audio_latent_length = model_specs["audio_latent_length"]
    _patch_h = raw_model.video_model.patch_size[1]
    _patch_w = raw_model.video_model.patch_size[2]

    # ── training config shortcuts ────────────────────────────────────────
    grad_accum = cfg.training.get("gradient_accumulation_steps", 1)
    max_steps = cfg.training.max_steps
    max_grad_norm = cfg.training.get("max_grad_norm", 1.0)
    flow_shift = cfg.flow.get("shift", 5.0)
    cfg_drop = cfg.flow.get("cfg_drop_rate", 0.1)
    vid_loss_w = cfg.loss.get("video_weight", 1.0)
    aud_loss_w = cfg.loss.get("audio_weight", 1.0)
    save_every = cfg.training.get("save_every", 2000)
    log_every = cfg.training.get("log_every", 50)

    # ── training loop ────────────────────────────────────────────────────
    global_step = 0
    optimizer.zero_grad()
    epoch = 0

    logger.info(f"Starting training: max_steps={max_steps}, batch={cfg.training.batch_size}, "
                f"accum={grad_accum}, lr={cfg.training.learning_rate}, mode={finetune_mode}")

    while global_step < max_steps:
        if distributed:
            sampler.set_epoch(epoch)

        for batch in dataloader:
            if global_step >= max_steps:
                break

            B = len(batch["video_latent"])

            # ── move to device + dtype ───────────────────────────────────
            video_latents = [v.to(device=device, dtype=target_dtype) for v in batch["video_latent"]]
            audio_latents = [a.to(device=device, dtype=target_dtype) for a in batch["audio_latent"]]
            text_embs     = [t.to(device=device, dtype=target_dtype) for t in batch["text_embedding"]]

            first_frame_latents = []
            has_ff = batch["has_first_frame"]
            for i in range(B):
                if has_ff[i] and batch["first_frame_latent"][i] is not None:
                    first_frame_latents.append(
                        batch["first_frame_latent"][i].to(device=device, dtype=target_dtype)
                    )
                else:
                    first_frame_latents.append(None)

            # ── sample timestep (flow-matching with shift) ───────────────
            t = sample_logit_normal_timesteps(B, shift=flow_shift, device=device)
            # t shape: [B], values in (0,1)
            # Ovi uses 0-1000 integer timesteps for the scheduler but inside the
            # model, sinusoidal_embedding_1d handles the raw float.
            # The model's time_embedding expects a float timestep per token.
            # In inference: t_v comes from the scheduler (0-1000 range).
            # We scale our (0,1) sigma to the 0-1000 range to match:
            t_1000 = t * 1000.0  # [B]

            # ── build noisy inputs ───────────────────────────────────────
            noisy_videos = []
            video_velocities = []
            noisy_audios = []
            audio_velocities = []

            for i in range(B):
                v0 = video_latents[i]  # [C, F, H, W]
                a0 = audio_latents[i]  # [L, C_audio]

                # video noise
                v_noise = torch.randn_like(v0)
                t_i = t[i]
                v_noisy = (1.0 - t_i) * v0 + t_i * v_noise
                v_vel = v_noise - v0

                # for I2V: replace first frame with clean latent
                is_i2v = first_frame_latents[i] is not None
                if is_i2v:
                    v_noisy[:, :1] = first_frame_latents[i]

                noisy_videos.append(v_noisy)
                video_velocities.append(v_vel)

                # audio noise
                a_noise = torch.randn_like(a0)
                a_noisy = (1.0 - t_i) * a0 + t_i * a_noise
                a_vel = a_noise - a0

                noisy_audios.append(a_noisy)
                audio_velocities.append(a_vel)

            # ── CFG dropout (drop text with probability cfg_drop) ────────
            context_vid = []
            context_aud = []
            for i in range(B):
                if random.random() < cfg_drop:
                    # zero-out text embedding for unconditional training
                    context_vid.append(torch.zeros_like(text_embs[i]))
                    context_aud.append(torch.zeros_like(text_embs[i]))
                else:
                    context_vid.append(text_embs[i])
                    context_aud.append(text_embs[i])

            # ── compute seq_len (same formula as inference engine) ───────
            v0_shape = video_latents[0]
            video_latent_h = v0_shape.shape[2]  # H_lat
            video_latent_w = v0_shape.shape[3]  # W_lat
            vid_seq_len = video_latent_length * video_latent_h * video_latent_w // (_patch_h * _patch_w)
            aud_seq_len = audio_latent_length

            # ── forward through FusionModel ──────────────────────────────
            # Uses the EXACT same signature as OviFusionEngine.generate()
            any_i2v = any(ff is not None for ff in first_frame_latents)
            timestep_input = t_1000  # [B]  (model broadcasts internally)

            with torch.amp.autocast("cuda", enabled=(target_dtype != torch.float32), dtype=target_dtype):
                pred_vid_list, pred_aud_list = model(
                    vid=noisy_videos,
                    audio=noisy_audios,
                    t=timestep_input,
                    vid_context=context_vid,
                    audio_context=context_aud,
                    vid_seq_len=vid_seq_len,
                    audio_seq_len=aud_seq_len,
                    first_frame_is_clean=any_i2v,
                )

                # ── loss (MSE on velocity) ───────────────────────────────
                video_loss = torch.tensor(0.0, device=device)
                audio_loss = torch.tensor(0.0, device=device)

                for i in range(B):
                    # video loss
                    pred_v = pred_vid_list[i]   # [C, F, H_lat, W_lat]
                    tgt_v = video_velocities[i]
                    # Ensure shapes match (pred may differ if I2V first frame was clean)
                    min_f = min(pred_v.shape[1], tgt_v.shape[1])
                    video_loss = video_loss + F.mse_loss(
                        pred_v[:, :min_f].float(),
                        tgt_v[:, :min_f].float(),
                    )

                    # audio loss
                    pred_a = pred_aud_list[i]   # [L, C_audio]
                    tgt_a = audio_velocities[i]
                    min_l = min(pred_a.shape[0], tgt_a.shape[0])
                    audio_loss = audio_loss + F.mse_loss(
                        pred_a[:min_l].float(),
                        tgt_a[:min_l].float(),
                    )

                video_loss = video_loss / B
                audio_loss = audio_loss / B
                loss = vid_loss_w * video_loss + aud_loss_w * audio_loss
                loss = loss / grad_accum

            # ── backward ─────────────────────────────────────────────────
            loss.backward()

            if (global_step + 1) % grad_accum == 0:
                if max_grad_norm > 0:
                    nn.utils.clip_grad_norm_(trainable_params, max_grad_norm)

                # LR schedule
                lr = get_lr(
                    global_step // grad_accum,
                    cfg.training.get("warmup_steps", 1000),
                    max_steps // grad_accum,
                    cfg.training.learning_rate,
                    cfg.training.get("lr_scheduler", "cosine"),
                )
                for pg in optimizer.param_groups:
                    pg["lr"] = lr

                optimizer.step()
                optimizer.zero_grad()

            # ── logging ──────────────────────────────────────────────────
            if is_main and global_step % log_every == 0:
                logger.info(
                    f"step={global_step:>6d}  loss={loss.item()*grad_accum:.5f}  "
                    f"v_loss={video_loss.item():.5f}  a_loss={audio_loss.item():.5f}  "
                    f"lr={optimizer.param_groups[0]['lr']:.2e}  "
                    f"t_mean={t.mean().item():.3f}"
                )
                if writer:
                    writer.add_scalar("train/loss", loss.item() * grad_accum, global_step)
                    writer.add_scalar("train/video_loss", video_loss.item(), global_step)
                    writer.add_scalar("train/audio_loss", audio_loss.item(), global_step)
                    writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], global_step)

            # ── save checkpoint ──────────────────────────────────────────
            if is_main and global_step > 0 and global_step % save_every == 0:
                save_path = os.path.join(cfg.output_dir, f"checkpoint_{global_step}.pt")
                torch.save({
                    "step": global_step,
                    "model": raw_model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                }, save_path)
                logger.info(f"Saved checkpoint → {save_path}")

                # also save a "latest" symlink-style copy
                latest_path = os.path.join(cfg.output_dir, "checkpoint_latest.pt")
                torch.save({
                    "step": global_step,
                    "model": raw_model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                }, latest_path)

            global_step += 1

        epoch += 1

    # ── final save ───────────────────────────────────────────────────────
    if is_main:
        final_path = os.path.join(cfg.output_dir, "checkpoint_final.pt")
        torch.save({
            "step": global_step,
            "model": raw_model.state_dict(),
        }, final_path)
        logger.info(f"Training complete at step {global_step}. Final → {final_path}")

    if writer:
        writer.close()
    if distributed:
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
