#!/usr/bin/env python3
"""
prepare_training_data_v3.py
===========================

Re-encode raw videos into v2-style training data while:
1. rewriting audio descriptions from `Audio: ...` to `<AUDCAP>...<ENDAUDCAP>`
2. creating deterministic train / validation / test splits up front
3. writing split-specific latent files and manifests directly
4. optionally resuming from an interrupted run

Usage:
    python prepare_training_data_v4.py \
        --input_manifest /project/llmsvgen/share/data_videoaudio/5s_videos_filtered/manifest.jsonl \
        --output_dir training_data_v2 \
        --ckpt_dir ./ckpts \
        --model_name 720x720_5s \
        --device 0 \
        --resume

Output layout:
    training_data_v2/
      source_manifest_clean.jsonl
      source_manifest_removed_errors.jsonl
      manifest.jsonl
      manifest_train.jsonl
      latents/*.pt
      validation/
        manifest.jsonl
        latents/*.pt
      testing/
        manifest.jsonl
        latents/*.pt
      split_summary.json
"""

import argparse
import json
import logging
import os
import random
import re
import shutil
import subprocess
import tempfile
from collections import Counter
from contextlib import ExitStack
from pathlib import Path

import cv2
import numpy as np
import torch
import torchaudio

from ovi.ovi_fusion_engine import NAME_TO_MODEL_SPECS_MAP
from ovi.utils.model_loading_utils import (
    init_mmaudio_vae,
    init_text_model,
    init_wan_vae_2_2,
)
from ovi.utils.processing_utils import preprocess_image_tensor, snap_hw_to_multiple_of_32


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


CORRUPTED_TEXT_PATTERNS = (
    "ERROR: CUDA out of memory",
    "CUDA out of memory",
)


def normalize_audio_caption(text_prompt: str) -> str:
    """
    Convert prompts from:
        "... Audio: some description"
    to:
        "... <AUDCAP>some description<ENDAUDCAP>"

    If the prompt already contains AUDCAP tags, normalize spacing and keep it.
    """
    text_prompt = text_prompt.strip()

    existing_match = re.search(
        r"<AUDCAP>\s*(.*?)\s*<ENDAUDCAP>",
        text_prompt,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if existing_match:
        audio_caption = existing_match.group(1).strip()
        visual_caption = re.sub(
            r"\s*<AUDCAP>\s*.*?\s*<ENDAUDCAP>\s*",
            " ",
            text_prompt,
            flags=re.IGNORECASE | re.DOTALL,
        ).strip()
        return join_visual_and_audio_caption(visual_caption, audio_caption)

    audio_match = re.search(r"\bAudio:\s*(.*)", text_prompt, flags=re.IGNORECASE | re.DOTALL)
    if not audio_match:
        return text_prompt

    visual_caption = text_prompt[:audio_match.start()].strip()
    audio_caption = audio_match.group(1).strip()
    return join_visual_and_audio_caption(visual_caption, audio_caption)


def join_visual_and_audio_caption(visual_caption: str, audio_caption: str) -> str:
    if not audio_caption:
        return visual_caption.strip()
    if not visual_caption:
        return f"<AUDCAP>{audio_caption}<ENDAUDCAP>"
    return f"{visual_caption} <AUDCAP>{audio_caption}<ENDAUDCAP>"


def is_corrupted_text_prompt(text_prompt: str) -> bool:
    normalized = (text_prompt or "").strip()
    if not normalized:
        return False
    return any(pattern.lower() in normalized.lower() for pattern in CORRUPTED_TEXT_PATTERNS)


def split_clean_and_removed_samples(samples):
    clean_samples = []
    removed_samples = []

    for sample in samples:
        text_prompt = sample.get("text_prompt", "")
        if is_corrupted_text_prompt(text_prompt):
            removed_samples.append(sample)
        else:
            clean_samples.append(sample)

    return clean_samples, removed_samples


def write_jsonl(path: Path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_jsonl_rows(path: Path):
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_video_frames(video_path: str, target_area: int, num_frames: int):
    """
    Load `num_frames` evenly spaced frames from a video, resize to the target
    area while preserving aspect ratio, and return [C, F, H, W] in [-1, 1].
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

    if not frames:
        raise RuntimeError(f"Failed to decode any frames from {video_path}")

    while len(frames) < num_frames:
        frames.append(frames[-1])

    h_orig, w_orig = frames[0].shape[:2]
    h, w = snap_hw_to_multiple_of_32(h_orig, w_orig, area=target_area)

    resized = [cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA) for frame in frames]
    arr = np.stack(resized, axis=0)
    tensor = torch.from_numpy(arr).permute(3, 0, 1, 2).float() / 127.5 - 1.0
    return tensor, h, w


def load_audio_waveform(video_path: str, target_sr: int = 16000, duration_sec: float = 5.0):
    """
    Extract audio from a video. Prefer ffmpeg for MP4 robustness, then fall back
    to torchaudio directly. Returns a mono [1, T] tensor.
    """
    target_num_samples = int(target_sr * duration_sec)
    ffmpeg_path = shutil.which("ffmpeg")

    if ffmpeg_path is not None:
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
                tmp_path = tmp_file.name

            cmd = [
                ffmpeg_path,
                "-y",
                "-i",
                video_path,
                "-vn",
                "-ac",
                "1",
                "-ar",
                str(target_sr),
                "-t",
                str(duration_sec),
                "-f",
                "wav",
                tmp_path,
            ]
            subprocess.run(
                cmd,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            waveform, _ = torchaudio.load(tmp_path)
            if waveform.ndim == 1:
                waveform = waveform.unsqueeze(0)
            return trim_or_pad_waveform(waveform, target_num_samples)
        except Exception as exc:
            logger.warning("ffmpeg audio extraction failed for %s: %s", video_path, exc)
        finally:
            if tmp_path is not None and os.path.exists(tmp_path):
                os.unlink(tmp_path)

    try:
        waveform, sr = torchaudio.load(video_path)
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)
        if sr != target_sr:
            waveform = torchaudio.functional.resample(waveform, sr, target_sr)
        return trim_or_pad_waveform(waveform, target_num_samples)
    except Exception as exc:
        logger.warning("Falling back to silence for %s: %s", video_path, exc)
        return torch.zeros(1, target_num_samples)


def trim_or_pad_waveform(waveform: torch.Tensor, target_num_samples: int) -> torch.Tensor:
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if waveform.shape[1] > target_num_samples:
        return waveform[:, :target_num_samples]
    if waveform.shape[1] < target_num_samples:
        return torch.nn.functional.pad(waveform, (0, target_num_samples - waveform.shape[1]))
    return waveform


def build_split_lookup(num_samples: int, validation_count: int, test_count: int, seed: int):
    if validation_count < 0 or test_count < 0:
        raise ValueError("validation_count and test_count must be non-negative")
    if validation_count + test_count >= num_samples:
        raise ValueError(
            f"Need at least one training sample, but got {num_samples} samples with "
            f"{validation_count} validation + {test_count} test requested."
        )

    indices = list(range(num_samples))
    rng = random.Random(seed)
    rng.shuffle(indices)

    split_lookup = {}
    for idx in indices[:validation_count]:
        split_lookup[idx] = "validation"
    for idx in indices[validation_count:validation_count + test_count]:
        split_lookup[idx] = "testing"
    for idx in indices[validation_count + test_count:]:
        split_lookup[idx] = "train"
    return split_lookup


def prepare_output_directories(output_dir: Path, overwrite: bool, resume: bool):
    if output_dir.exists():
        if not overwrite and not resume:
            raise FileExistsError(
                f"{output_dir} already exists. Re-run with --resume to continue or "
                "--overwrite to rebuild it from scratch."
            )
        if overwrite:
            logger.info("Removing existing output directory: %s", output_dir)
            shutil.rmtree(output_dir)

    (output_dir / "latents").mkdir(parents=True, exist_ok=True)
    (output_dir / "validation" / "latents").mkdir(parents=True, exist_ok=True)
    (output_dir / "testing" / "latents").mkdir(parents=True, exist_ok=True)


def get_split_paths(output_dir: Path):
    return {
        "train": {
            "root_dir": output_dir,
            "latent_dir": output_dir / "latents",
            "manifest_path": output_dir / "manifest.jsonl",
            "extra_manifest_paths": [output_dir / "manifest_train.jsonl"],
        },
        "validation": {
            "root_dir": output_dir / "validation",
            "latent_dir": output_dir / "validation" / "latents",
            "manifest_path": output_dir / "validation" / "manifest.jsonl",
            "extra_manifest_paths": [],
        },
        "testing": {
            "root_dir": output_dir / "testing",
            "latent_dir": output_dir / "testing" / "latents",
            "manifest_path": output_dir / "testing" / "manifest.jsonl",
            "extra_manifest_paths": [],
        },
    }


def latent_filename_for_sample(video_path: str, sample_idx: int, basename_counts) -> str:
    video_stem = Path(video_path).stem
    if basename_counts[video_stem] == 1:
        return f"{video_stem}.pt"
    return f"{video_stem}__{sample_idx:06d}.pt"


def load_existing_split_entries(split_paths, resume: bool):
    if not resume:
        return {split_name: [] for split_name in split_paths}

    existing_entries = {}
    for split_name, split_cfg in split_paths.items():
        rows = load_jsonl_rows(split_cfg["manifest_path"])
        if not rows:
            for extra_manifest_path in split_cfg["extra_manifest_paths"]:
                rows = load_jsonl_rows(extra_manifest_path)
                if rows:
                    break
        existing_entries[split_name] = rows
    return existing_entries


def backfill_missing_manifest_copies(split_paths, existing_entries_by_split, resume: bool):
    if not resume:
        return

    for split_name, split_cfg in split_paths.items():
        rows = existing_entries_by_split[split_name]
        if not rows:
            continue
        if not split_cfg["manifest_path"].exists():
            write_jsonl(split_cfg["manifest_path"], rows)
        for extra_manifest_path in split_cfg["extra_manifest_paths"]:
            if not extra_manifest_path.exists():
                write_jsonl(extra_manifest_path, rows)


def main():
    parser = argparse.ArgumentParser(description="Prepare Ovi training latents (v3)")
    parser.add_argument(
        "--input_manifest",
        type=str,
        default="/project/llmsvgen/share/data_videoaudio/5s_videos_filtered/manifest.jsonl",
    )
    parser.add_argument("--output_dir", type=str, default="training_data_v2")
    parser.add_argument("--ckpt_dir", type=str, default="./ckpts")
    parser.add_argument(
        "--model_name",
        type=str,
        default="720x720_5s",
        choices=list(NAME_TO_MODEL_SPECS_MAP.keys()),
    )
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--validation_count", type=int, default=6)
    parser.add_argument("--test_count", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.resume and args.overwrite:
        parser.error("--resume and --overwrite cannot be used together")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for prepare_training_data_v3.py")

    output_dir = Path(args.output_dir).resolve()
    prepare_output_directories(output_dir, overwrite=args.overwrite, resume=args.resume)
    split_paths = get_split_paths(output_dir)

    rank = args.device
    torch_device = torch.device(f"cuda:{args.device}")
    model_specs = NAME_TO_MODEL_SPECS_MAP[args.model_name]
    target_area = model_specs["video_area"]
    video_latent_length = model_specs["video_latent_length"]
    duration_sec = 10.0 if "10s" in args.model_name else 5.0
    num_raw_frames = (video_latent_length - 1) * 4 + 1

    with open(args.input_manifest, encoding="utf-8") as f:
        samples = [json.loads(line) for line in f if line.strip()]
    if args.max_samples is not None:
        samples = samples[:args.max_samples]
    logger.info("Loaded %d samples from %s", len(samples), args.input_manifest)

    clean_samples, removed_samples = split_clean_and_removed_samples(samples)
    logger.info(
        "Filtered corrupted caption rows before encoding: kept=%d removed=%d",
        len(clean_samples),
        len(removed_samples),
    )

    write_jsonl(output_dir / "source_manifest_clean.jsonl", clean_samples)
    write_jsonl(output_dir / "source_manifest_removed_errors.jsonl", removed_samples)

    samples = clean_samples

    split_lookup = build_split_lookup(
        num_samples=len(samples),
        validation_count=args.validation_count,
        test_count=args.test_count,
        seed=args.seed,
    )
    basename_counts = Counter(Path(sample["video_path"]).stem for sample in samples)
    existing_entries_by_split = load_existing_split_entries(split_paths, resume=args.resume)
    backfill_missing_manifest_copies(split_paths, existing_entries_by_split, resume=args.resume)
    processed_video_paths_by_split = {
        split_name: {entry["video_path"] for entry in rows if "video_path" in entry}
        for split_name, rows in existing_entries_by_split.items()
    }

    logger.info("Loading video VAE ...")
    vae_video = init_wan_vae_2_2(args.ckpt_dir, rank=rank)
    vae_video.model.requires_grad_(False).eval().bfloat16()

    logger.info("Loading audio VAE ...")
    vae_audio = init_mmaudio_vae(args.ckpt_dir, rank=rank)
    vae_audio.requires_grad_(False).eval().bfloat16()

    logger.info("Loading T5 text encoder ...")
    text_model = init_text_model(args.ckpt_dir, rank=rank)

    counters = {
        "train": len(existing_entries_by_split["train"]),
        "validation": len(existing_entries_by_split["validation"]),
        "testing": len(existing_entries_by_split["testing"]),
    }
    if args.resume:
        logger.info(
            "Resume mode: existing processed samples train=%d validation=%d testing=%d",
            counters["train"],
            counters["validation"],
            counters["testing"],
        )

    with ExitStack() as stack:
        manifest_writers = {}
        for split_name, split_cfg in split_paths.items():
            handles = [
                stack.enter_context(
                    open(split_cfg["manifest_path"], "a" if args.resume else "w", encoding="utf-8")
                )
            ]
            for extra_manifest_path in split_cfg["extra_manifest_paths"]:
                handles.append(
                    stack.enter_context(
                        open(extra_manifest_path, "a" if args.resume else "w", encoding="utf-8")
                    )
                )
            manifest_writers[split_name] = handles

        for idx, sample in enumerate(samples):
            video_path = sample["video_path"]
            raw_text_prompt = sample["text_prompt"]
            normalized_text_prompt = normalize_audio_caption(raw_text_prompt)
            formatted_text_prompt = model_specs["formatter"](normalized_text_prompt)
            image_path = sample.get("image_path")
            split_name = split_lookup[idx]
            split_cfg = split_paths[split_name]

            if video_path in processed_video_paths_by_split[split_name]:
                logger.info(
                    "[%d/%d] SKIP (%s already processed): %s",
                    idx + 1,
                    len(samples),
                    split_name,
                    video_path,
                )
                continue

            logger.info("[%d/%d] %s -> %s", idx + 1, len(samples), split_name, video_path)

            with torch.no_grad():
                text_emb = text_model([formatted_text_prompt], text_model.device)[0].cpu()

            video_tensor, h, w = load_video_frames(video_path, target_area, num_raw_frames)
            video_tensor = video_tensor.unsqueeze(0).to(device=torch_device, dtype=torch.bfloat16)
            with torch.no_grad():
                video_latent = vae_video.wrapped_encode(video_tensor).squeeze(0).cpu()

            first_frame_latent = None
            if image_path is not None:
                if not os.path.isfile(image_path):
                    logger.warning("image_path does not exist for %s: %s", video_path, image_path)
                else:
                    ff = preprocess_image_tensor(
                        image_path,
                        rank,
                        torch.bfloat16,
                        resize_total_area=target_area,
                    )
                    with torch.no_grad():
                        first_frame_latent = vae_video.wrapped_encode(ff[:, :, None]).squeeze(0).cpu()

            waveform = load_audio_waveform(
                video_path,
                target_sr=16000,
                duration_sec=duration_sec,
            )
            waveform = waveform.to(device=torch_device, dtype=torch.bfloat16)
            with torch.no_grad():
                audio_latent = vae_audio.wrapped_encode(waveform).squeeze(0)
                if audio_latent.dim() == 2 and audio_latent.shape[0] < audio_latent.shape[1]:
                    audio_latent = audio_latent.transpose(0, 1)
                audio_latent = audio_latent.cpu()

            save_dict = {
                "video_latent": video_latent.float(),
                "audio_latent": audio_latent.float(),
                "text_embedding": text_emb.float(),
            }
            if first_frame_latent is not None:
                save_dict["first_frame_latent"] = first_frame_latent.float()

            latent_filename = latent_filename_for_sample(video_path, idx, basename_counts)
            pt_path = split_cfg["latent_dir"] / latent_filename
            torch.save(save_dict, pt_path)

            record = {
                "latent_path": os.path.relpath(pt_path, split_cfg["root_dir"]),
                "video_path": video_path,
                "text_prompt": normalized_text_prompt,
                "has_first_frame": first_frame_latent is not None,
                "video_h": h,
                "video_w": w,
            }

            record_line = json.dumps(record, ensure_ascii=False) + "\n"
            for manifest_handle in manifest_writers[split_name]:
                manifest_handle.write(record_line)
                manifest_handle.flush()

            processed_video_paths_by_split[split_name].add(video_path)
            counters[split_name] += 1

    split_summary = {
        "input_manifest": os.path.abspath(args.input_manifest),
        "output_dir": str(output_dir),
        "model_name": args.model_name,
        "seed": args.seed,
        "source_manifest_paths": {
            "clean": str(output_dir / "source_manifest_clean.jsonl"),
            "removed_errors": str(output_dir / "source_manifest_removed_errors.jsonl"),
        },
        "source_counts": {
            "loaded_total": len(clean_samples) + len(removed_samples),
            "removed_corrupted": len(removed_samples),
            "clean_total": len(clean_samples),
        },
        "counts": {
            "train": counters["train"],
            "validation": counters["validation"],
            "testing": counters["testing"],
            "total": len(samples),
        },
        "manifest_paths": {
            "train": str(split_paths["train"]["manifest_path"]),
            "train_copy": str(split_paths["train"]["extra_manifest_paths"][0]),
            "validation": str(split_paths["validation"]["manifest_path"]),
            "testing": str(split_paths["testing"]["manifest_path"]),
        },
    }
    with open(output_dir / "split_summary.json", "w", encoding="utf-8") as f:
        json.dump(split_summary, f, indent=2)
        f.write("\n")

    logger.info(
        "Done. Train=%d Validation=%d Testing=%d Total=%d",
        counters["train"],
        counters["validation"],
        counters["testing"],
        len(samples),
    )


if __name__ == "__main__":
    main()
