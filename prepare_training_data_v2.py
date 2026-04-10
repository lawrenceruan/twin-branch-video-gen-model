#!/usr/bin/env python3
"""
prepare_training_data_v2.py  –  Encode raw videos into latents using Ovi's own
VAE / text encoder utilities.

Usage:
    python prepare_training_data_v2.py \
        --input_manifest  training_data/raw_manifest.jsonl \
        --output_dir      training_data \
        --ckpt_dir        ./ckpts \
        --model_name      960x960_5s \
        --device          0

Each line of raw_manifest.jsonl:
    {"video_path": "...", "text_prompt": "...", "image_path": "..." (optional)}

Outputs per sample  →  training_data/latents/<idx>.pt  containing:
    video_latent   : [C, F, H_lat, W_lat]
    audio_latent   : [L, C_audio]
    text_embedding : [text_len, D]
    (optional) first_frame_latent : [C, 1, H_lat, W_lat]

Also writes  training_data/manifest.jsonl  consumed by the training dataloader.
"""

import argparse
import json
import logging
import os
import sys

import cv2
import numpy as np
import torch
import torchaudio

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# ── Ovi imports ──────────────────────────────────────────────────────────────
from ovi.ovi_fusion_engine import NAME_TO_MODEL_SPECS_MAP
from ovi.utils.model_loading_utils import (
    init_wan_vae_2_2,
    init_mmaudio_vae,
    init_text_model,
)
from ovi.utils.processing_utils import preprocess_image_tensor, snap_hw_to_multiple_of_32


# ── helpers ──────────────────────────────────────────────────────────────────

def load_video_frames(video_path: str, target_area: int, num_frames: int):
    """
    Load *num_frames* evenly-spaced frames from an MP4, resize so that
    H*W ≈ target_area (keeping aspect ratio, snapped to 32-multiples),
    and return a float32 tensor [C, F, H, W] in [-1, 1].
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video {video_path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        raise RuntimeError(f"Video {video_path} has 0 frames")

    indices = np.linspace(0, total - 1, num_frames, dtype=int)
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()

    if len(frames) < num_frames:
        # pad by repeating last frame
        while len(frames) < num_frames:
            frames.append(frames[-1])

    h_orig, w_orig = frames[0].shape[:2]
    h, w = snap_hw_to_multiple_of_32(h_orig, w_orig, area=target_area)

    resized = [cv2.resize(f, (w, h), interpolation=cv2.INTER_AREA) for f in frames]
    arr = np.stack(resized, axis=0)  # [F, H, W, 3]
    tensor = torch.from_numpy(arr).permute(3, 0, 1, 2).float() / 127.5 - 1.0  # [C, F, H, W]
    return tensor, h, w


def load_audio_waveform(video_path: str, target_sr: int = 16000, duration_sec: float = 5.0):
    """
    Extract audio from video, resample to target_sr, and return [1, T] tensor.
    """
    try:
        waveform, sr = torchaudio.load(video_path)
    except Exception:
        # silent fallback
        T = int(target_sr * duration_sec)
        return torch.zeros(1, T)

    if sr != target_sr:
        waveform = torchaudio.functional.resample(waveform, sr, target_sr)
    # mono
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    # trim / pad to duration
    T = int(target_sr * duration_sec)
    if waveform.shape[1] > T:
        waveform = waveform[:, :T]
    elif waveform.shape[1] < T:
        waveform = torch.nn.functional.pad(waveform, (0, T - waveform.shape[1]))
    return waveform


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Prepare Ovi training latents (v2)")
    parser.add_argument("--input_manifest", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--ckpt_dir", type=str, default="./ckpts")
    parser.add_argument("--model_name", type=str, default="960x960_5s",
                        choices=list(NAME_TO_MODEL_SPECS_MAP.keys()))
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()

    device = args.device
    model_specs = NAME_TO_MODEL_SPECS_MAP[args.model_name]
    target_area = model_specs["video_area"]
    video_latent_length = model_specs["video_latent_length"]
    audio_latent_length = model_specs["audio_latent_length"]
    # 5s models → 5 sec, 10s models → 10 sec
    duration_sec = 10.0 if "10s" in args.model_name else 5.0
    # raw video frames needed  (4× latent +1 for 5s, etc.)
    num_raw_frames = (video_latent_length - 1) * 4 + 1

    latent_dir = os.path.join(args.output_dir, "latents")
    os.makedirs(latent_dir, exist_ok=True)

    # ── load encoders ────────────────────────────────────────────────────
    logging.info("Loading video VAE …")
    vae_video = init_wan_vae_2_2(args.ckpt_dir, rank=device)
    vae_video.model.requires_grad_(False).eval().bfloat16()

    logging.info("Loading audio VAE …")
    vae_audio = init_mmaudio_vae(args.ckpt_dir, rank=device)
    vae_audio.requires_grad_(False).eval().bfloat16()

    logging.info("Loading T5 text encoder …")
    text_model = init_text_model(args.ckpt_dir, rank=device)

    # ── read manifest ────────────────────────────────────────────────────
    with open(args.input_manifest) as f:
        samples = [json.loads(line) for line in f if line.strip()]
    logging.info(f"Found {len(samples)} samples in {args.input_manifest}")

    out_manifest = []

    for idx, sample in enumerate(samples):
        video_path = sample["video_path"]
        text_prompt = sample["text_prompt"]
        image_path = sample.get("image_path", None)

        logging.info(f"[{idx+1}/{len(samples)}] {video_path}")

        # ── text ─────────────────────────────────────────────────────────
        with torch.no_grad():
            text_emb = text_model([text_prompt], text_model.device)[0]  # [L, D]
            text_emb = text_emb.cpu()

        # ── video ────────────────────────────────────────────────────────
        video_tensor, h, w = load_video_frames(video_path, target_area, num_raw_frames)
        video_tensor = video_tensor.unsqueeze(0).to(device=device, dtype=torch.bfloat16)
        with torch.no_grad():
            video_latent = vae_video.wrapped_encode(video_tensor).squeeze(0).cpu()
            # video_latent shape: [C, F_lat, H_lat, W_lat]

        # ── first frame (optional, for I2V) ──────────────────────────────
        first_frame_latent = None
        if image_path is not None:
            ff = preprocess_image_tensor(image_path, device, torch.bfloat16, resize_total_area=target_area)
            with torch.no_grad():
                first_frame_latent = vae_video.wrapped_encode(ff[:, :, None]).squeeze(0).cpu()

        # ── audio ────────────────────────────────────────────────────────
        waveform = load_audio_waveform(video_path, target_sr=16000, duration_sec=duration_sec)
        waveform = waveform.unsqueeze(0).to(device=device, dtype=torch.bfloat16)
        with torch.no_grad():
            audio_latent = vae_audio.wrapped_encode(waveform).squeeze(0)  # [C_audio, L] or [L, C_audio]
            # transpose if needed to get [L, C_audio]
            if audio_latent.dim() == 2 and audio_latent.shape[0] < audio_latent.shape[1]:
                audio_latent = audio_latent.transpose(0, 1)
            audio_latent = audio_latent.cpu()

        # ── save ─────────────────────────────────────────────────────────
        save_dict = {
            "video_latent": video_latent.float(),
            "audio_latent": audio_latent.float(),
            "text_embedding": text_emb.float(),
        }
        if first_frame_latent is not None:
            save_dict["first_frame_latent"] = first_frame_latent.float()

        pt_path = os.path.join(latent_dir, f"{idx:06d}.pt")
        torch.save(save_dict, pt_path)

        record = {
            "latent_path": pt_path,
            "text_prompt": text_prompt,
            "has_first_frame": first_frame_latent is not None,
            "video_h": h,
            "video_w": w,
        }
        out_manifest.append(record)

    # ── write output manifest ────────────────────────────────────────────
    manifest_path = os.path.join(args.output_dir, "manifest.jsonl")
    with open(manifest_path, "w") as f:
        for rec in out_manifest:
            f.write(json.dumps(rec) + "\n")

    logging.info(f"Done. Wrote {len(out_manifest)} entries → {manifest_path}")


if __name__ == "__main__":
    main()
