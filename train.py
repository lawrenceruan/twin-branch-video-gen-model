"""
train.py  –  Training script for the Ovi Fusion Model (joint video + audio diffusion)
=====================================================================================

This script fine-tunes the FusionModel (a pair of coupled DiT transformers for
video and audio) using a **flow-matching** objective on pre-encoded latents.

Key design choices
------------------
1. **Pre-encoded latents** – Video, audio, and text are encoded offline by
   `prepare_training_data.py`.  The VAEs and T5 are *not* loaded at training
   time, keeping VRAM usage manageable.

2. **Flow-matching loss** – At each step a random timestep *t* ∈ [0, 1] is
   sampled (with an optional shifted schedule matching inference), Gaussian
   noise *ε* is mixed with the clean latent at ratio *t*, and the model
   predicts the velocity (i.e. the clean-minus-noise direction).  This is
   the same formulation used by the existing inference scheduler.

3. **Classifier-free guidance preparation** – A fraction of samples have
   their text embeddings zeroed out so the model learns both the conditional
   and unconditional distributions (required for CFG at inference).

4. **Gradient checkpointing + bf16 mixed precision** – Keeps memory usage
   under control even on 24 GB GPUs.

Usage
-----
Single GPU:
    python train.py --config ovi/configs/training/training_fusion.yaml

Multi-GPU (torchrun):
    torchrun --nproc_per_node=4 train.py --config ovi/configs/training/training_fusion.yaml
"""

import argparse
import json
import logging
import math
import os
import random
import time
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from omegaconf import OmegaConf

from ovi.modules.fusion import FusionModel
from ovi.utils.model_loading_utils import (
    init_fusion_score_model_ovi,
    load_fusion_checkpoint,
)
from ovi.ovi_fusion_engine import NAME_TO_MODEL_SPECS_MAP

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ============================================================================
# Dataset
# ============================================================================

class LatentDataset(Dataset):
    """Reads pre-encoded .pt files listed in a JSONL manifest.

    Each .pt file is expected to contain:
        video_latent       : Tensor [C, F, H, W]
        audio_latent       : Tensor [L, C_a]
        text_embedding     : Tensor [S, D]
        first_frame_latent : Tensor [C, 1, H, W] or None
    """

    def __init__(self, manifest_path: str, base_dir: str = None, max_samples: int = None):
        self.base_dir = base_dir or os.path.dirname(manifest_path)
        with open(manifest_path) as f:
            self.entries = [json.loads(line) for line in f if line.strip()]
        if max_samples is not None:
            self.entries = self.entries[:max_samples]
        logger.info(f"LatentDataset: {len(self.entries)} samples from {manifest_path}")

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        entry = self.entries[idx]
        latent_path = os.path.join(self.base_dir, entry["latent_path"])
        data = torch.load(latent_path, map_location="cpu", weights_only=True)
        return data


def collate_fn(batch):
    """Simple collator – keeps lists of tensors (variable lengths possible)."""
    keys = batch[0].keys()
    out = {}
    for k in keys:
        vals = [b[k] for b in batch]
        # Stack if all same shape, otherwise keep as list
        if all(v is not None for v in vals):
            shapes = [v.shape for v in vals]
            if len(set(shapes)) == 1:
                out[k] = torch.stack(vals, dim=0)
            else:
                out[k] = vals
        else:
            out[k] = vals
    return out


# ============================================================================
# Flow-matching helpers
# ============================================================================

def sample_timesteps(batch_size: int, shift: float = 5.0, weighting: str = "uniform",
                     device: torch.device = "cpu"):
    """Sample diffusion timesteps in [0, 1000] using the shifted schedule.

    The *shift* parameter mirrors the inference scheduler: it re-maps a
    uniform u ∈ [0,1] to  t = shift * u / (1 + (shift-1)*u)  before
    scaling to [0, 1000].
    """
    u = torch.rand(batch_size, device=device)  # uniform in [0, 1]

    if weighting == "logit_normal":
        # Logit-normal: sample from logit-normal then sigmoid
        u = torch.sigmoid(torch.randn(batch_size, device=device))
    elif weighting == "mode":
        # Mode-seeking: beta-like weighting toward middle
        u = 1.0 - torch.sqrt(torch.rand(batch_size, device=device))

    # Apply shift  (same as FlowUniPC shift mapping)
    t_shifted = shift * u / (1.0 + (shift - 1.0) * u)

    # Scale to [0, 1000] as the model expects
    timesteps = t_shifted * 1000.0
    timesteps = timesteps.clamp(0.001, 999.999)
    return timesteps  # shape [B]


def flow_matching_noise(clean, noise, t):
    """Interpolate between noise and clean at time t (in [0,1000]).

    Uses the standard flow-matching / rectified flow interpolation:
        x_t = (1 - t/1000) * noise  +  (t/1000) * clean
    The velocity target is:  v = clean - noise

    (Note: In the Ovi codebase the scheduler convention is that t=1000 → clean
    and t=0 → noise, matching σ_t = 1 - t/1000.)
    """
    # t: [B] or scalar
    # Normalize to [0, 1]
    sigma = 1.0 - t / 1000.0  # σ(t) → 1 at t=0 (pure noise), 0 at t=1000 (clean)

    # For video latents: [B, C, F, H, W] or [C, F, H, W]
    # For audio latents: [B, L, C] or [L, C]
    # We need to broadcast t appropriately
    while sigma.dim() < clean.dim():
        sigma = sigma.unsqueeze(-1)

    noisy = (1.0 - sigma) * clean + sigma * noise  # x_t
    velocity_target = clean - noise  # v = dx/dt target for the model
    return noisy, velocity_target


# ============================================================================
# Training step
# ============================================================================

def training_step(model: FusionModel, batch: dict, model_specs: dict,
                  shift: float, weighting: str, cfg_drop_prob: float,
                  device: torch.device, target_dtype: torch.dtype,
                  video_loss_weight: float = 1.0, audio_loss_weight: float = 1.0,
                  mode: str = "t2v"):
    """Execute one forward pass and return the scalar loss."""

    video_latent = batch["video_latent"].to(device, dtype=target_dtype)   # [B, C, F, H, W]
    audio_latent = batch["audio_latent"].to(device, dtype=target_dtype)   # [B, L, Ca]
    text_emb = batch["text_embedding"]  # list or [B, S, D]

    # Handle text embeddings (may be list of variable-length tensors)
    if isinstance(text_emb, list):
        text_emb_list = [t.to(device, dtype=target_dtype) for t in text_emb]
    else:
        text_emb_list = [text_emb[i].to(device, dtype=target_dtype) for i in range(text_emb.shape[0])]

    B = video_latent.shape[0]

    # First frame latent for i2v mode
    first_frame_latent = batch.get("first_frame_latent", [None] * B)
    is_i2v = mode == "i2v" and first_frame_latent[0] is not None

    # ---- Sample timesteps ----
    timesteps = sample_timesteps(B, shift=shift, weighting=weighting, device=device)  # [B]

    # ---- Generate noise ----
    video_noise = torch.randn_like(video_latent)
    audio_noise = torch.randn_like(audio_latent)

    # ---- Create noisy samples via flow-matching interpolation ----
    noisy_video, video_target = flow_matching_noise(video_latent, video_noise, timesteps)
    noisy_audio, audio_target = flow_matching_noise(audio_latent, audio_noise, timesteps)

    # ---- I2V: replace first frame in noisy video with clean latent ----
    if is_i2v:
        for i in range(B):
            if first_frame_latent[i] is not None:
                ffl = first_frame_latent[i].to(device, dtype=target_dtype)
                noisy_video[i, :, :1] = ffl

    # ---- CFG: randomly drop text conditioning ----
    context_video = []
    context_audio = []
    for i in range(B):
        if random.random() < cfg_drop_prob:
            # Drop: zero embedding
            context_video.append(torch.zeros_like(text_emb_list[i]))
            context_audio.append(torch.zeros_like(text_emb_list[i]))
        else:
            context_video.append(text_emb_list[i])
            context_audio.append(text_emb_list[i])

    # ---- Compute sequence lengths ----
    _ph, _pw = model.video_model.patch_size[1], model.video_model.patch_size[2]
    vid_seq_len = video_latent.shape[2] * video_latent.shape[3] * video_latent.shape[4] // (_ph * _pw)
    audio_seq_len = audio_latent.shape[1]

    # ---- Forward pass ----
    # Model expects list inputs per sample
    vid_input = [noisy_video[i] for i in range(B)]      # list of [C, F, H, W]
    audio_input = [noisy_audio[i] for i in range(B)]     # list of [L, Ca]

    t_input = timesteps  # [B]

    pred_vid, pred_audio = model(
        vid=vid_input,
        audio=audio_input,
        t=t_input,
        vid_context=context_video,
        audio_context=context_audio,
        vid_seq_len=vid_seq_len,
        audio_seq_len=audio_seq_len,
        first_frame_is_clean=is_i2v,
    )

    # ---- Compute losses (MSE between predicted and target velocities) ----
    video_loss = torch.tensor(0.0, device=device)
    audio_loss = torch.tensor(0.0, device=device)

    for i in range(B):
        # Video loss
        v_pred = pred_vid[i]  # [C, F, H, W]
        v_tgt = video_target[i]  # [C, F, H, W]
        if is_i2v:
            # Don't compute loss on the clean first frame
            v_pred = v_pred[:, 1:]
            v_tgt = v_tgt[:, 1:]
        video_loss = video_loss + nn.functional.mse_loss(v_pred.float(), v_tgt.float())

        # Audio loss
        a_pred = pred_audio[i]  # [L, Ca]
        a_tgt = audio_target[i]  # [L, Ca]
        audio_loss = audio_loss + nn.functional.mse_loss(a_pred.float(), a_tgt.float())

    video_loss = video_loss / B
    audio_loss = audio_loss / B

    total_loss = video_loss_weight * video_loss + audio_loss_weight * audio_loss

    return total_loss, video_loss.item(), audio_loss.item()


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Train Ovi Fusion Model")
    parser.add_argument("--config", type=str,
                        default="ovi/configs/training/training_fusion.yaml",
                        help="Path to training config YAML")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="Local rank for distributed training")
    args = parser.parse_args()

    # ---- Load config ----
    cfg = OmegaConf.load(args.config)

    # ---- Distributed setup ----
    distributed = int(os.environ.get("WORLD_SIZE", 1)) > 1
    if distributed:
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
        world_size = dist.get_world_size()
        rank = dist.get_rank()
    else:
        local_rank = 0
        device = torch.device("cuda:0")
        world_size = 1
        rank = 0

    is_main = (rank == 0)

    # ---- Seed ----
    seed = cfg.training.seed
    random.seed(seed + rank)
    torch.manual_seed(seed + rank)

    # ---- Dtype ----
    target_dtype = torch.bfloat16 if cfg.mixed_precision == "bf16" else torch.float32

    # ---- Model name / specs ----
    model_name = cfg.model_name
    assert model_name in NAME_TO_MODEL_SPECS_MAP
    model_specs = NAME_TO_MODEL_SPECS_MAP[model_name]

    # ---- Build model ----
    if is_main:
        logger.info("Initializing FusionModel …")
    model, video_config, audio_config = init_fusion_score_model_ovi(rank=device, meta_init=True)

    # Optionally load pretrained fusion checkpoint
    if cfg.model.get("load_pretrained", True):
        basename = model_specs["path"]
        ckpt_path = os.path.join(cfg.ckpt_dir, "Ovi", basename)
        if os.path.exists(ckpt_path):
            if is_main:
                logger.info(f"Loading pretrained fusion checkpoint: {ckpt_path}")
            load_fusion_checkpoint(model, checkpoint_path=ckpt_path, from_meta=True)
        else:
            if is_main:
                logger.warning(f"Pretrained checkpoint not found at {ckpt_path}, training from scratch.")
            # Need to materialize from meta
            model = model.to(dtype=target_dtype, device=device)
    else:
        model = model.to(dtype=target_dtype, device=device)

    # Move to device & dtype (if loaded from checkpoint, assign already placed on CPU)
    model = model.to(dtype=target_dtype, device=device)
    model.set_rope_params()

    # Gradient checkpointing
    if cfg.model.get("gradient_checkpointing", True):
        model.gradient_checkpointing = True
        if hasattr(model.video_model, "set_gradient_checkpointing"):
            model.video_model.set_gradient_checkpointing(True)
        if hasattr(model.audio_model, "set_gradient_checkpointing"):
            model.audio_model.set_gradient_checkpointing(True)
        if is_main:
            logger.info("Gradient checkpointing enabled.")
    else:
        model.gradient_checkpointing = False

    # Finetune mode
    finetune_mode = cfg.model.get("finetune_mode", "full")
    if finetune_mode == "fusion_only":
        # Freeze everything except the injected fusion cross-attention parameters
        for name, param in model.named_parameters():
            if "fusion" not in name:
                param.requires_grad = False
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        if is_main:
            logger.info(f"Fusion-only mode: {trainable:,} / {total:,} params trainable")
    else:
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        if is_main:
            logger.info(f"Full fine-tuning: {trainable:,} params trainable")

    model.train()

    # Wrap with DDP if distributed
    if distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank], find_unused_parameters=False
        )
        raw_model = model.module
    else:
        raw_model = model

    # ---- Dataset & DataLoader ----
    manifest_path = cfg.dataset.manifest_path
    dataset = LatentDataset(
        manifest_path=manifest_path,
        max_samples=cfg.dataset.get("max_samples", None),
    )

    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True) if distributed else None
    dataloader = DataLoader(
        dataset,
        batch_size=cfg.training.batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=2,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=True,
    )

    # ---- Optimizer ----
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg.training.learning_rate,
        weight_decay=cfg.training.weight_decay,
        betas=(0.9, 0.999),
        eps=1e-8,
    )

    # ---- LR Scheduler ----
    max_steps = cfg.training.max_steps
    warmup_steps = cfg.training.warmup_steps

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        if cfg.training.lr_scheduler == "cosine":
            progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
            return 0.5 * (1.0 + math.cos(math.pi * progress))
        return 1.0  # constant

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # ---- Resume from checkpoint ----
    start_step = 0
    resume_path = cfg.model.get("resume_from", None)
    if resume_path and os.path.exists(resume_path):
        if is_main:
            logger.info(f"Resuming from training checkpoint: {resume_path}")
        ckpt = torch.load(resume_path, map_location="cpu")
        raw_model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_step = ckpt.get("step", 0)
        del ckpt

    # ---- TensorBoard ----
    tb_writer = None
    if is_main:
        try:
            from torch.utils.tensorboard import SummaryWriter
            tb_dir = cfg.logging.get("tensorboard_dir", os.path.join(cfg.output_dir, "tb_logs"))
            os.makedirs(tb_dir, exist_ok=True)
            tb_writer = SummaryWriter(tb_dir)
            logger.info(f"TensorBoard logging to {tb_dir}")
        except ImportError:
            logger.warning("tensorboard not installed; skipping TB logging.")

    # ---- Output dir ----
    os.makedirs(cfg.output_dir, exist_ok=True)

    # ---- Training config summary ----
    if is_main:
        logger.info("=" * 60)
        logger.info("Training Configuration")
        logger.info("=" * 60)
        logger.info(f"  Model name:          {model_name}")
        logger.info(f"  Finetune mode:       {finetune_mode}")
        logger.info(f"  Batch size/GPU:      {cfg.training.batch_size}")
        logger.info(f"  Grad accumulation:   {cfg.training.gradient_accumulation_steps}")
        logger.info(f"  Effective batch:     {cfg.training.batch_size * cfg.training.gradient_accumulation_steps * world_size}")
        logger.info(f"  Learning rate:       {cfg.training.learning_rate}")
        logger.info(f"  Max steps:           {max_steps}")
        logger.info(f"  Warmup steps:        {warmup_steps}")
        logger.info(f"  Mixed precision:     {cfg.mixed_precision}")
        logger.info(f"  World size:          {world_size}")
        logger.info(f"  Mode:                {cfg.mode}")
        logger.info(f"  Video loss weight:   {cfg.loss.video_weight}")
        logger.info(f"  Audio loss weight:   {cfg.loss.audio_weight}")
        logger.info("=" * 60)

    # ---- Training loop ----
    grad_accum_steps = cfg.training.gradient_accumulation_steps
    log_every = cfg.logging.log_every
    save_every = cfg.logging.save_every
    max_grad_norm = cfg.training.max_grad_norm
    shift = cfg.flow_matching.shift
    weighting = cfg.flow_matching.get("timestep_weighting", "uniform")
    cfg_drop_prob = 0.1  # 10% unconditional dropout for CFG
    video_loss_weight = cfg.loss.video_weight
    audio_loss_weight = cfg.loss.audio_weight

    global_step = start_step
    epoch = 0
    running_loss = 0.0
    running_vloss = 0.0
    running_aloss = 0.0
    step_time_start = time.time()

    scaler = torch.amp.GradScaler('cuda', enabled=(target_dtype == torch.float16))

    while global_step < max_steps:
        if distributed:
            sampler.set_epoch(epoch)

        for batch in dataloader:
            if global_step >= max_steps:
                break

            # ---- Forward + backward ----
            with torch.amp.autocast('cuda', enabled=(target_dtype != torch.float32), dtype=target_dtype):
                loss, vloss, aloss = training_step(
                    model=raw_model,
                    batch=batch,
                    model_specs=model_specs,
                    shift=shift,
                    weighting=weighting,
                    cfg_drop_prob=cfg_drop_prob,
                    device=device,
                    target_dtype=target_dtype,
                    video_loss_weight=video_loss_weight,
                    audio_loss_weight=audio_loss_weight,
                    mode=cfg.mode,
                )
                loss = loss / grad_accum_steps

            if scaler.is_enabled():
                scaler.scale(loss).backward()
            else:
                loss.backward()

            running_loss += loss.item() * grad_accum_steps
            running_vloss += vloss
            running_aloss += aloss

            # ---- Gradient accumulation step ----
            if (global_step + 1) % grad_accum_steps == 0 or global_step == 0:
                if max_grad_norm > 0:
                    if scaler.is_enabled():
                        scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in model.parameters() if p.requires_grad],
                        max_grad_norm,
                    )

                if scaler.is_enabled():
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()

                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            global_step += 1

            # ---- Logging ----
            if is_main and global_step % log_every == 0:
                elapsed = time.time() - step_time_start
                steps_per_sec = log_every / elapsed
                avg_loss = running_loss / log_every
                avg_vloss = running_vloss / log_every
                avg_aloss = running_aloss / log_every
                lr = scheduler.get_last_lr()[0]
                vram_gb = torch.cuda.max_memory_allocated(device) / 1e9

                logger.info(
                    f"Step {global_step}/{max_steps} | "
                    f"loss={avg_loss:.4f} (video={avg_vloss:.4f}, audio={avg_aloss:.4f}) | "
                    f"lr={lr:.2e} | {steps_per_sec:.2f} step/s | VRAM={vram_gb:.1f}GB"
                )

                if tb_writer:
                    tb_writer.add_scalar("train/loss", avg_loss, global_step)
                    tb_writer.add_scalar("train/video_loss", avg_vloss, global_step)
                    tb_writer.add_scalar("train/audio_loss", avg_aloss, global_step)
                    tb_writer.add_scalar("train/lr", lr, global_step)

                running_loss = 0.0
                running_vloss = 0.0
                running_aloss = 0.0
                step_time_start = time.time()

            # ---- Save checkpoint ----
            if is_main and global_step % save_every == 0:
                save_path = os.path.join(cfg.output_dir, f"checkpoint_step{global_step}.pt")
                torch.save({
                    "step": global_step,
                    "model": raw_model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "config": OmegaConf.to_container(cfg),
                }, save_path)
                logger.info(f"Saved checkpoint: {save_path}")

                # Also save a "latest" symlink / copy for easy resume
                latest_path = os.path.join(cfg.output_dir, "checkpoint_latest.pt")
                if os.path.exists(latest_path):
                    os.remove(latest_path)
                import shutil
                shutil.copy2(save_path, latest_path)

        epoch += 1

    # ---- Final save ----
    if is_main:
        final_path = os.path.join(cfg.output_dir, "checkpoint_final.pt")
        torch.save({
            "step": global_step,
            "model": raw_model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "config": OmegaConf.to_container(cfg),
        }, final_path)
        logger.info(f"Training complete at step {global_step}. Final checkpoint: {final_path}")

        if tb_writer:
            tb_writer.close()

    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
