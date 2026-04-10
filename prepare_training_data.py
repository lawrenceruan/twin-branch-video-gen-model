"""
prepare_training_data.py  –  Pre-encode video / audio / text into latents
=========================================================================

This script reads raw training samples (video files + text prompts, and
optionally first-frame images for I2V), encodes them through the frozen
VAEs and T5 text encoder, and writes per-sample `.pt` files together with
a JSONL manifest that the training dataloader can consume.

Why pre-encode?
  • The VAEs + T5 are frozen during training so there is no need to keep
    them in GPU memory while the DiT is being trained.
  • Pre-encoding reduces training-time I/O and VRAM requirements
    dramatically.

Usage
-----
    python prepare_training_data.py \
        --input_manifest  training_data/raw_manifest.jsonl \
        --output_dir      training_data \
        --ckpt_dir        ./ckpts \
        --model_name      960x960_5s \
        --device          0

Input manifest format (one JSON object per line):
    {"video_path": "path/to/video.mp4", "text_prompt": "A cat ...", "image_path": "optional/path.png"}

Output per sample:
    training_data/latents/000000.pt   – dict with keys:
        video_latent   (Tensor)  [C, F, H, W]
        audio_latent   (Tensor)  [L, C_a]
        text_embedding (Tensor)  [S, D]
        first_frame_latent (Tensor or None)  [C, 1, H, W]

    training_data/manifest.jsonl  – one JSON per line:
        {"idx": 0, "latent_path": "latents/000000.pt"}
"""

import argparse
import json
import logging
import os

import subprocess
import tempfile

import cv2
import numpy as np
import torch
import torchaudio

from ovi.utils.model_loading_utils import (
    init_mmaudio_vae,
    init_text_model,
    init_wan_vae_2_2,
)
from ovi.utils.processing_utils import preprocess_image_tensor
from ovi.ovi_fusion_engine import NAME_TO_MODEL_SPECS_MAP

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


# ---------------------------------------------------------------------------
# Video helpers
# ---------------------------------------------------------------------------

def load_video_frames(video_path: str, num_frames: int, target_h: int, target_w: int):
    """Load *num_frames* evenly-spaced frames from a video file, return [C,F,H,W] in [-1,1]."""
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        raise RuntimeError(f"Cannot read video: {video_path}")

    indices = np.linspace(0, total - 1, num_frames, dtype=int)
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if not ret:
            raise RuntimeError(f"Failed to read frame {idx} from {video_path}")
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (target_w, target_h))
        frames.append(frame)
    cap.release()

    # [F, H, W, C] -> [C, F, H, W], float in [-1, 1]
    video = np.stack(frames, axis=0).astype(np.float32) / 255.0 * 2.0 - 1.0
    video = torch.from_numpy(video).permute(3, 0, 1, 2)  # C, F, H, W
    return video


def load_audio_from_video(video_path: str, target_sr: int = 16000, duration_sec: float = 5.0):
    """Extract audio from a video file using ffmpeg, return 1-D tensor at target_sr."""
    num_samples = int(target_sr * duration_sec)
    try:
        # Use ffmpeg to extract audio as mono WAV at the target sample rate.
        # This handles MP4 and other container formats that soundfile/torchaudio
        # cannot open directly.
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name
        cmd = [
            "ffmpeg", "-y", "-i", video_path,
            "-vn",                          # no video
            "-ac", "1",                     # mono
            "-ar", str(target_sr),          # resample
            "-t", str(duration_sec),        # trim to duration
            "-f", "wav",
            tmp_path,
        ]
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if result.returncode != 0:
            raise RuntimeError(result.stderr.decode(errors="replace"))

        waveform, sr = torchaudio.load(tmp_path)
        os.remove(tmp_path)
    except Exception as e:
        logging.warning(f"Could not extract audio from {video_path} ({e}); returning silence.")
        return torch.zeros(num_samples)

    # to mono (should already be mono from ffmpeg, but just in case)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    waveform = waveform.squeeze(0)

    # trim / pad to exact duration
    if waveform.shape[0] > num_samples:
        waveform = waveform[:num_samples]
    elif waveform.shape[0] < num_samples:
        waveform = torch.nn.functional.pad(waveform, (0, num_samples - waveform.shape[0]))

    return waveform


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Pre-encode training data for Ovi Fusion training")
    parser.add_argument("--input_manifest", type=str, required=True,
                        help="Path to raw manifest JSONL (video_path, text_prompt, optional image_path)")
    parser.add_argument("--output_dir", type=str, default="./training_data",
                        help="Directory to write latents + manifest")
    parser.add_argument("--ckpt_dir", type=str, default="./ckpts",
                        help="Checkpoint directory (for VAEs and T5)")
    parser.add_argument("--model_name", type=str, default="960x960_5s",
                        choices=list(NAME_TO_MODEL_SPECS_MAP.keys()))
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()

    device = args.device
    model_specs = NAME_TO_MODEL_SPECS_MAP[args.model_name]
    target_area = model_specs["video_area"]
    video_latent_length = model_specs["video_latent_length"]
    # Compute the number of pixel-space frames.  The VAE temporally downsamples by 4
    # with the first frame handled specially:  num_frames = (latent_length - 1) * 4 + 1
    num_pixel_frames = (video_latent_length - 1) * 4 + 1

    # Approximate pixel h/w from target_area (square for simplicity; the VAE handles any multiple of 16)
    import math
    side = int(math.sqrt(target_area))
    # snap to 32
    side = (side // 32) * 32
    target_h, target_w = side, side

    duration_sec = 5.0 if "10s" not in args.model_name else 10.0

    latent_dir = os.path.join(args.output_dir, "latents")
    os.makedirs(latent_dir, exist_ok=True)

    # ---- Load encoders ----
    logging.info("Loading video VAE …")
    vae_video = init_wan_vae_2_2(args.ckpt_dir, rank=device)
    vae_video.model.requires_grad_(False).eval()
    vae_video.model = vae_video.model.bfloat16()

    logging.info("Loading audio VAE …")
    vae_audio = init_mmaudio_vae(args.ckpt_dir, rank=device)
    vae_audio.requires_grad_(False).eval()
    vae_audio = vae_audio.bfloat16()

    logging.info("Loading T5 text encoder …")
    text_model = init_text_model(args.ckpt_dir, rank=device)

    # ---- Read raw manifest ----
    with open(args.input_manifest) as f:
        samples = [json.loads(line) for line in f if line.strip()]
    logging.info(f"Found {len(samples)} samples in {args.input_manifest}")

    manifest_lines = []

    for idx, sample in enumerate(samples):
        video_path = sample["video_path"]
        text_prompt = sample["text_prompt"]
        image_path = sample.get("image_path", None)

        logging.info(f"[{idx+1}/{len(samples)}] Processing {video_path}")

        # ---- Encode video ----
        video_tensor = load_video_frames(video_path, num_pixel_frames, target_h, target_w)
        # video_tensor: [C, F, H, W]  add batch dim
        with torch.no_grad():
            video_latent = vae_video.wrapped_encode(
                video_tensor.unsqueeze(0).to(device, dtype=torch.bfloat16)
            ).float().squeeze(0)  # [C, F_lat, H_lat, W_lat]

        # ---- Encode first frame (if i2v) ----
        first_frame_latent = None
        if image_path and os.path.isfile(image_path):
            first_frame = preprocess_image_tensor(
                image_path, device, torch.bfloat16, resize_total_area=target_area
            )
            with torch.no_grad():
                first_frame_latent = vae_video.wrapped_encode(
                    first_frame[:, :, None].to(device)
                ).float().squeeze(0)  # [C, 1, H_lat, W_lat]

        # ---- Encode audio ----
        waveform = load_audio_from_video(video_path, target_sr=16000, duration_sec=duration_sec)
        with torch.no_grad():
            audio_latent = vae_audio.wrapped_encode(
                waveform.unsqueeze(0).to(device, dtype=torch.float32)
            ).float().squeeze(0).transpose(0, 1)  # [L, C_a]

        # ---- Encode text ----
        with torch.no_grad():
            text_emb_list = text_model([text_prompt], text_model.device)
            text_embedding = text_emb_list[0].float().cpu()  # [S, D]

        # ---- Save ----
        save_dict = {
            "video_latent": video_latent.cpu(),         # [C, F, Hl, Wl]
            "audio_latent": audio_latent.cpu(),          # [L, C_a]
            "text_embedding": text_embedding,            # [S, D]
            "first_frame_latent": first_frame_latent.cpu() if first_frame_latent is not None else None,
        }
        save_name = f"{idx:06d}.pt"
        save_path = os.path.join(latent_dir, save_name)
        torch.save(save_dict, save_path)

        manifest_lines.append(json.dumps({"idx": idx, "latent_path": f"latents/{save_name}"}))

    # ---- Write manifest ----
    manifest_path = os.path.join(args.output_dir, "manifest.jsonl")
    with open(manifest_path, "w") as f:
        f.write("\n".join(manifest_lines) + "\n")
    logging.info(f"Done! Wrote {len(manifest_lines)} entries to {manifest_path}")


if __name__ == "__main__":
    main()