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
import cv2
import importlib.util
import json
import logging
import math
import numpy as np
import os
import random
import re
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from omegaconf import OmegaConf
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

# diffusers in this environment expects torch.xpu to exist even on non-XPU installs.
if not hasattr(torch, "xpu"):
    class _TorchXpuCompat:
        @staticmethod
        def empty_cache():
            return None

        @staticmethod
        def device_count():
            return 0

        @staticmethod
        def manual_seed(_seed):
            return None

        @staticmethod
        def reset_peak_memory_stats():
            return None

        @staticmethod
        def max_memory_allocated():
            return 0

        @staticmethod
        def synchronize():
            return None

    torch.xpu = _TorchXpuCompat()

from ovi.modules.fusion import FusionModel
from ovi.utils.fm_solvers import (
    FlowDPMSolverMultistepScheduler,
    get_sampling_sigmas,
    retrieve_timesteps,
)
from ovi.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
from ovi.utils.io_utils import save_video
from ovi.utils.model_loading_utils import (
    init_fusion_score_model_ovi,
    init_mmaudio_vae,
    init_text_model,
    init_wan_vae_2_2,
    load_fusion_checkpoint,
)
from ovi.utils.processing_utils import format_prompt_for_filename, snap_hw_to_multiple_of_32

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


NAME_TO_MODEL_SPECS_MAP = {
    "720x720_5s": {
        "path": "model.safetensors",
        "video_latent_length": 31,
        "audio_latent_length": 157,
        "video_area": 720 * 720,
        "formatter": lambda text: re.sub(r"Audio:\s*(.*)", r"<AUDCAP>\1<ENDAUDCAP>", text, flags=re.S),
    },
    "960x960_5s": {
        "path": "model_960x960.safetensors",
        "video_latent_length": 31,
        "audio_latent_length": 157,
        "video_area": 960 * 960,
        "formatter": lambda text: re.sub(r"<AUDCAP>(.*?)<ENDAUDCAP>", r"Audio: \1", text, flags=re.S),
    },
    "960x960_10s": {
        "path": "model_960x960_10s.safetensors",
        "video_latent_length": 61,
        "audio_latent_length": 314,
        "video_area": 960 * 960,
        "formatter": lambda text: re.sub(r"<AUDCAP>(.*?)<ENDAUDCAP>", r"Audio: \1", text, flags=re.S),
    },
}


def _patch_ib_audio_loader(ib_module) -> None:
    dataset_cls = getattr(ib_module, "ImageBindAudioFromVideoDataset", None)
    if dataset_cls is None or getattr(dataset_cls, "_ovi_audio_loader_patched", False):
        return

    original_load_audio = dataset_cls._load_audio

    def _load_audio_with_ffmpeg(self, video_path: Path) -> torch.Tensor:
        ffmpeg_path = shutil.which("ffmpeg")
        load_errors = []

        if ffmpeg_path is not None:
            tmp_path = None
            try:
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
                    tmp_path = tmp_file.name

                cmd = [
                    ffmpeg_path,
                    "-y",
                    "-i",
                    str(video_path),
                    "-vn",
                    "-ac",
                    "1",
                    "-ar",
                    str(self.sample_rate),
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
                waveform, sr = ib_module.torchaudio.load(tmp_path)
                if waveform.ndim == 1:
                    waveform = waveform.unsqueeze(0)
                if sr != self.sample_rate:
                    waveform = ib_module.torchaudio.functional.resample(
                        waveform,
                        orig_freq=sr,
                        new_freq=self.sample_rate,
                    )

                target_samples = int(self.sample_rate * self.duration_sec)
                if waveform.shape[1] < target_samples:
                    waveform = torch.nn.functional.pad(waveform, (0, target_samples - waveform.shape[1]))
                else:
                    waveform = waveform[:, :target_samples]

                return waveform
            except Exception as e:
                stderr_tail = ""
                if isinstance(e, subprocess.CalledProcessError) and e.stderr:
                    stderr_tail = f" | ffmpeg stderr: {e.stderr.strip().splitlines()[-1]}"
                load_errors.append(f"ffmpeg_cli: {e}{stderr_tail}")
            finally:
                if tmp_path is not None:
                    try:
                        os.unlink(tmp_path)
                    except OSError:
                        pass
        else:
            load_errors.append("ffmpeg_cli: ffmpeg not found on PATH")

        try:
            return original_load_audio(self, video_path)
        except Exception as e:
            load_errors.append(f"torchaudio_loader: {e}")
            raise RuntimeError(
                f"Failed to load audio track from {video_path}. "
                f"Errors: {' | '.join(load_errors)}"
            ) from e

    dataset_cls._load_audio = _load_audio_with_ffmpeg
    dataset_cls._ovi_audio_loader_patched = True


def _patch_ib_video_loader(ib_module) -> None:
    dataset_cls = getattr(ib_module, "ImageBindVideoDataset", None)
    if dataset_cls is None or getattr(dataset_cls, "_ovi_video_loader_patched", False):
        return

    original_sample = dataset_cls._sample

    def _sample_with_opencv(self, idx: int):
        video_path = self.video_paths[idx]
        load_errors = []
        cap = None

        try:
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                raise RuntimeError(f"OpenCV failed to open {video_path}")

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total_frames <= 0:
                raise RuntimeError(f"OpenCV reported 0 frames for {video_path}")

            fps = float(cap.get(cv2.CAP_PROP_FPS))
            if not math.isfinite(fps) or fps <= 0:
                fps = None

            if fps is not None:
                sample_period_sec = 1.0 / float(ib_module._IMAGEBIND_FPS)
                frame_indices = [
                    min(int(round(i * sample_period_sec * fps)), total_frames - 1)
                    for i in range(self.expected_length)
                ]
            else:
                frame_indices = np.linspace(0, total_frames - 1, self.expected_length, dtype=int).tolist()

            frames = []
            for frame_idx in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
                ok, frame = cap.read()
                if not ok or frame is None:
                    raise RuntimeError(f"OpenCV failed to read frame {frame_idx} from {video_path}")
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            if len(frames) < self.expected_length:
                raise RuntimeError(
                    f"OpenCV decoded too few frames for ImageBind: {video_path}, "
                    f"expected {self.expected_length}, got {len(frames)}"
                )

            video = torch.from_numpy(np.stack(frames, axis=0)).permute(0, 3, 1, 2)
            video = self.transform(video)
            video = self.crop([video])
            video = torch.stack(video)
            return {
                "name": video_path.stem,
                "ib_video": video,
            }
        except Exception as e:
            load_errors.append(f"opencv_loader: {e}")
            logger.debug("OpenCV ImageBind video fallback failed for %s", video_path, exc_info=True)
        finally:
            if cap is not None:
                cap.release()

        try:
            return original_sample(self, idx)
        except Exception as e:
            load_errors.append(f"torchaudio_loader: {e}")
            raise RuntimeError(
                f"Failed to load video frames from {video_path}. "
                f"Errors: {' | '.join(load_errors)}"
            ) from e

    dataset_cls._sample = _sample_with_opencv
    dataset_cls._ovi_video_loader_patched = True


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


def load_jsonl(path: str, max_samples: int = None):
    with open(path, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    if max_samples is not None:
        rows = rows[:max_samples]
    return rows


def get_scheduler_time_steps(sampling_steps: int, solver_name: str, device: torch.device,
                             shift: float = 5.0):
    torch.manual_seed(4)

    if solver_name == "unipc":
        sample_scheduler = FlowUniPCMultistepScheduler(
            num_train_timesteps=1000,
            shift=1,
            use_dynamic_shifting=False,
        )
        sample_scheduler.set_timesteps(sampling_steps, device=device, shift=shift)
        timesteps = sample_scheduler.timesteps
    elif solver_name == "dpm++":
        sample_scheduler = FlowDPMSolverMultistepScheduler(
            num_train_timesteps=1000,
            shift=1,
            use_dynamic_shifting=False,
        )
        sampling_sigmas = get_sampling_sigmas(sampling_steps, shift=shift)
        timesteps, _ = retrieve_timesteps(
            sample_scheduler,
            device=device,
            sigmas=sampling_sigmas,
        )
    elif solver_name == "euler":
        from diffusers import FlowMatchEulerDiscreteScheduler

        sample_scheduler = FlowMatchEulerDiscreteScheduler(shift=shift)
        timesteps, _ = retrieve_timesteps(
            sample_scheduler,
            sampling_steps,
            device=device,
        )
    else:
        raise NotImplementedError(f"Unsupported solver: {solver_name}")

    return sample_scheduler, timesteps


class ValidationRunner:
    def __init__(self, cfg, model: FusionModel, device: torch.device, target_dtype: torch.dtype,
                 model_specs: dict, video_config: dict, audio_config: dict,
                 output_root: str):
        self.cfg = cfg
        self.validation_cfg = cfg.get("validation", {})
        self.model = model
        self.device = device
        self.target_dtype = target_dtype
        self.model_specs = model_specs
        self.video_config = video_config
        self.audio_config = audio_config
        self.output_root = output_root
        self.entries = load_jsonl(
            self.validation_cfg.get("manifest_path", "./training_data/validation/manifest.jsonl"),
            max_samples=self.validation_cfg.get("max_samples", None),
        )
        self.text_formatter = model_specs["formatter"]
        self.target_area = model_specs["video_area"]
        self.video_latent_length = model_specs["video_latent_length"]
        self.audio_latent_length = model_specs["audio_latent_length"]
        self.video_latent_channel = video_config["in_dim"]
        self.audio_latent_channel = audio_config["in_dim"]
        self.text_model = None
        self.vae_model_video = None
        self.vae_model_audio = None
        self.ib_module = None
        self.ib_model = None
        os.makedirs(self.output_root, exist_ok=True)

    def _load_inference_components(self):
        if self.text_model is None:
            logger.info("Loading validation text encoder and VAEs...")
            self.text_model = init_text_model(self.cfg.ckpt_dir, rank=self.device)
            self.vae_model_video = init_wan_vae_2_2(self.cfg.ckpt_dir, rank=self.device)
            self.vae_model_video.model.requires_grad_(False).eval()
            self.vae_model_video.model = self.vae_model_video.model.bfloat16()
            self.vae_model_audio = init_mmaudio_vae(self.cfg.ckpt_dir, rank=self.device)
            self.vae_model_audio.requires_grad_(False).eval()
            self.vae_model_audio = self.vae_model_audio.bfloat16()

    def _load_ib_components(self):
        if self.ib_module is not None and self.ib_model is not None:
            return

        ib_script_path = Path(
            self.validation_cfg.get(
                "ib_script_path",
                "/project/llmsvgen/pengjun/MMAudio_dev/av-benchmark/ib_score_from_videos.py",
            )
        )
        ib_root = ib_script_path.parent
        if str(ib_root) not in sys.path:
            sys.path.insert(0, str(ib_root))

        spec = importlib.util.spec_from_file_location("ib_score_from_videos", ib_script_path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Failed to import ImageBind scoring script from {ib_script_path}")
        ib_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(ib_module)
        _patch_ib_audio_loader(ib_module)
        _patch_ib_video_loader(ib_module)

        ib_device = self.validation_cfg.get(
            "ib_device",
            f"cuda:{self.device.index}" if self.device.type == "cuda" else "cpu",
        )
        logger.info("Loading ImageBind model for validation scoring on %s...", ib_device)
        self.ib_module = ib_module
        self.ib_model = ib_module.imagebind_model.imagebind_huge(pretrained=True).to(ib_device).eval()

    @torch.inference_mode()
    def _generate_single(self, text_prompt: str, video_frame_height_width, seed: int):
        solver_name = self.validation_cfg.get("solver_name", "unipc")
        sample_steps = self.validation_cfg.get("sample_steps", 50)
        shift = self.validation_cfg.get("shift", self.cfg.flow_matching.shift)
        video_guidance_scale = self.validation_cfg.get("video_guidance_scale", 4.0)
        audio_guidance_scale = self.validation_cfg.get("audio_guidance_scale", 3.0)
        slg_layer = self.validation_cfg.get("slg_layer", 11)
        video_negative_prompt = self.validation_cfg.get(
            "video_negative_prompt", "jitter, bad hands, blur, distortion"
        )
        audio_negative_prompt = self.validation_cfg.get(
            "audio_negative_prompt", "robotic, muffled, echo, distorted"
        )

        scheduler_video, timesteps_video = get_scheduler_time_steps(
            sampling_steps=sample_steps,
            solver_name=solver_name,
            device=self.device,
            shift=shift,
        )
        scheduler_audio, timesteps_audio = get_scheduler_time_steps(
            sampling_steps=sample_steps,
            solver_name=solver_name,
            device=self.device,
            shift=shift,
        )

        video_h, video_w = snap_hw_to_multiple_of_32(
            int(video_frame_height_width[0]),
            int(video_frame_height_width[1]),
            area=self.target_area,
        )
        video_latent_h, video_latent_w = video_h // 16, video_w // 16

        formatted_text_prompt = self.text_formatter(text_prompt)
        text_embeddings = self.text_model(
            [formatted_text_prompt, video_negative_prompt, audio_negative_prompt],
            self.text_model.device,
        )
        text_embeddings = [emb.to(self.target_dtype).to(self.device) for emb in text_embeddings]
        text_embeddings_audio_pos = text_embeddings[0]
        text_embeddings_video_pos = text_embeddings[0]
        text_embeddings_video_neg = text_embeddings[1]
        text_embeddings_audio_neg = text_embeddings[2]

        generator = torch.Generator(device=self.device).manual_seed(seed)
        video_noise = torch.randn(
            (self.video_latent_channel, self.video_latent_length, video_latent_h, video_latent_w),
            device=self.device,
            dtype=self.target_dtype,
            generator=generator,
        )
        audio_noise = torch.randn(
            (self.audio_latent_length, self.audio_latent_channel),
            device=self.device,
            dtype=self.target_dtype,
            generator=generator,
        )

        max_seq_len_audio = audio_noise.shape[0]
        patch_h, patch_w = self.model.video_model.patch_size[1], self.model.video_model.patch_size[2]
        max_seq_len_video = (
            video_noise.shape[1] * video_noise.shape[2] * video_noise.shape[3] // (patch_h * patch_w)
        )

        with torch.amp.autocast("cuda", enabled=(self.target_dtype != torch.float32), dtype=self.target_dtype):
            for t_v, t_a in zip(timesteps_video, timesteps_audio):
                timestep_input = torch.full((1,), t_v, device=self.device)

                pred_vid_pos, pred_audio_pos = self.model(
                    vid=[video_noise],
                    audio=[audio_noise],
                    t=timestep_input,
                    vid_context=[text_embeddings_video_pos],
                    audio_context=[text_embeddings_audio_pos],
                    vid_seq_len=max_seq_len_video,
                    audio_seq_len=max_seq_len_audio,
                    first_frame_is_clean=False,
                )

                pred_vid_neg, pred_audio_neg = self.model(
                    vid=[video_noise],
                    audio=[audio_noise],
                    t=timestep_input,
                    vid_context=[text_embeddings_video_neg],
                    audio_context=[text_embeddings_audio_neg],
                    vid_seq_len=max_seq_len_video,
                    audio_seq_len=max_seq_len_audio,
                    first_frame_is_clean=False,
                    slg_layer=slg_layer,
                )

                pred_video_guided = pred_vid_neg[0] + video_guidance_scale * (pred_vid_pos[0] - pred_vid_neg[0])
                pred_audio_guided = pred_audio_neg[0] + audio_guidance_scale * (
                    pred_audio_pos[0] - pred_audio_neg[0]
                )

                video_noise = scheduler_video.step(
                    pred_video_guided.unsqueeze(0), t_v, video_noise.unsqueeze(0), return_dict=False
                )[0].squeeze(0)
                audio_noise = scheduler_audio.step(
                    pred_audio_guided.unsqueeze(0), t_a, audio_noise.unsqueeze(0), return_dict=False
                )[0].squeeze(0)

            audio_latents_for_vae = audio_noise.unsqueeze(0).transpose(1, 2)
            generated_audio = self.vae_model_audio.wrapped_decode(audio_latents_for_vae)
            generated_audio = generated_audio.squeeze().cpu().float().numpy()

            video_latents_for_vae = video_noise.unsqueeze(0)
            generated_video = self.vae_model_video.wrapped_decode(video_latents_for_vae)
            generated_video = generated_video.squeeze(0).cpu().float().numpy()

        return generated_video, generated_audio, (video_h, video_w)

    def _score_videos(self, video_dir: Path, output_json: Path):
        self._load_ib_components()
        ib_device = self.validation_cfg.get(
            "ib_device",
            f"cuda:{self.device.index}" if self.device.type == "cuda" else "cpu",
        )
        audio_length = float(self.validation_cfg.get("audio_length", 5.0))
        batch_size = int(self.validation_cfg.get("ib_batch_size", 1))
        num_workers = int(self.validation_cfg.get("ib_num_workers", 4))

        video_paths = self.ib_module.list_video_files(video_dir)
        if not video_paths:
            raise RuntimeError(f"No generated validation videos found in {video_dir}")

        video_features = self.ib_module.extract_video_features(
            video_paths,
            imagebind=self.ib_model,
            duration_sec=audio_length,
            batch_size=batch_size,
            num_workers=num_workers,
            device=ib_device,
        )
        audio_features = self.ib_module.extract_audio_features(
            video_paths,
            imagebind=self.ib_model,
            duration_sec=audio_length,
            batch_size=batch_size,
            num_workers=num_workers,
            device=ib_device,
        )

        shared_names = sorted(set(video_features) & set(audio_features))
        scores = {}
        for name in shared_names:
            scores[name] = torch.cosine_similarity(
                video_features[name].unsqueeze(0),
                audio_features[name].unsqueeze(0),
                dim=-1,
            ).item()

        args = SimpleNamespace(
            video_path=video_dir,
            output_json=output_json,
            audio_length=audio_length,
            device=ib_device,
        )
        output = self.ib_module.build_output(
            scores,
            video_features,
            audio_features,
            len(video_paths),
            args,
        )
        output_json.parent.mkdir(parents=True, exist_ok=True)
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        return output

    def run(self, step: int):
        self._load_inference_components()

        step_dir = Path(self.output_root) / f"step_{step:06d}"
        videos_dir = step_dir / "videos"
        videos_dir.mkdir(parents=True, exist_ok=True)

        metadata = []
        base_seed = int(self.validation_cfg.get("seed", 103))
        fps = int(self.validation_cfg.get("fps", 24))
        sample_rate = int(self.validation_cfg.get("sample_rate", 16000))

        was_training = self.model.training
        self.model.eval()
        try:
            logger.info("Running validation inference at step %s on %s prompts...", step, len(self.entries))
            for idx, entry in enumerate(self.entries):
                text_prompt = entry["text_prompt"]
                seed = base_seed + idx
                video_h = entry.get("video_h")
                video_w = entry.get("video_w")
                if video_h is None or video_w is None:
                    raise ValueError(f"Validation manifest entry is missing video_h/video_w: {entry}")

                generated_video, generated_audio, actual_hw = self._generate_single(
                    text_prompt=text_prompt,
                    video_frame_height_width=(video_h, video_w),
                    seed=seed,
                )
                stem = f"{idx:03d}_{format_prompt_for_filename(text_prompt)}"
                output_path = videos_dir / f"{stem}.mp4"
                save_video(
                    str(output_path),
                    generated_video,
                    generated_audio,
                    fps=fps,
                    sample_rate=sample_rate,
                )
                metadata.append(
                    {
                        "index": idx,
                        "prompt": text_prompt,
                        "seed": seed,
                        "video_path": str(output_path),
                        "requested_hw": [int(video_h), int(video_w)],
                        "generated_hw": [int(actual_hw[0]), int(actual_hw[1])],
                    }
                )

            metadata_path = step_dir / "metadata.json"
            with open(metadata_path, "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)

            if self.device.type == "cuda":
                torch.cuda.empty_cache()

            ib_output = self._score_videos(videos_dir, step_dir / "ib_scores.json")
            summary = {
                "step": step,
                "num_prompts": len(self.entries),
                "average_ib_score": ib_output.get("average_ib_score"),
                "ib_scores_json": str(step_dir / "ib_scores.json"),
                "metadata_json": str(metadata_path),
                "video_dir": str(videos_dir),
            }
            with open(step_dir / "summary.json", "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)

            history_path = Path(self.output_root) / "history.jsonl"
            with open(history_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(summary, ensure_ascii=False) + "\n")

            logger.info(
                "Validation step %s complete. Average ImageBind score: %s",
                step,
                "n/a" if summary["average_ib_score"] is None else f"{summary['average_ib_score']:.6f}",
            )
            return summary
        finally:
            if was_training:
                self.model.train()


def capture_trainable_parameter_reference(model: nn.Module):
    return {
        name: param.detach().float().cpu().clone()
        for name, param in model.named_parameters()
        if param.requires_grad
    }


def compute_parameter_anchor_loss(model: nn.Module, reference_params: dict):
    if not reference_params:
        device = next(model.parameters()).device
        return torch.tensor(0.0, device=device)

    total_sq = None
    total_numel = 0
    for name, param in model.named_parameters():
        if not param.requires_grad or name not in reference_params:
            continue
        ref = reference_params[name].to(device=param.device, dtype=torch.float32)
        diff_sq = (param.float() - ref).pow(2).sum()
        total_sq = diff_sq if total_sq is None else total_sq + diff_sq
        total_numel += param.numel()

    if total_sq is None or total_numel == 0:
        device = next(model.parameters()).device
        return torch.tensor(0.0, device=device)
    return total_sq / total_numel


# ============================================================================
# Training step
# ============================================================================

def training_step(model: FusionModel, batch: dict, model_specs: dict,
                  shift: float, weighting: str, cfg_drop_prob: float,
                  device: torch.device, target_dtype: torch.dtype,
                  video_loss_weight: float = 1.0, audio_loss_weight: float = 1.0,
                  mode: str = "t2v"):
    """Execute one forward pass and return the scalar loss."""

    def _to_sample_list(value, *, dtype):
        if isinstance(value, list):
            return [v.to(device, dtype=dtype) if v is not None else None for v in value]
        return [value[i].to(device, dtype=dtype) for i in range(value.shape[0])]

    video_latent = _to_sample_list(batch["video_latent"], dtype=target_dtype)
    audio_latent = _to_sample_list(batch["audio_latent"], dtype=target_dtype)
    text_emb = batch["text_embedding"]  # list or [B, S, D]

    # Handle text embeddings (may be list of variable-length tensors)
    if isinstance(text_emb, list):
        text_emb_list = [t.to(device, dtype=target_dtype) for t in text_emb]
    else:
        text_emb_list = [text_emb[i].to(device, dtype=target_dtype) for i in range(text_emb.shape[0])]

    B = len(video_latent)

    # First frame latent for i2v mode
    first_frame_latent = batch.get("first_frame_latent", [None] * B)
    if isinstance(first_frame_latent, list):
        first_frame_latent = [
            ff.to(device, dtype=target_dtype) if ff is not None else None
            for ff in first_frame_latent
        ]
    else:
        first_frame_latent = [
            first_frame_latent[i].to(device, dtype=target_dtype)
            for i in range(first_frame_latent.shape[0])
        ]
    is_i2v = mode == "i2v" and any(ff is not None for ff in first_frame_latent)

    # ---- Sample timesteps ----
    timesteps = sample_timesteps(B, shift=shift, weighting=weighting, device=device)  # [B]

    # ---- Generate noise ----
    video_noise = [torch.randn_like(v) for v in video_latent]
    audio_noise = [torch.randn_like(a) for a in audio_latent]

    # ---- Create noisy samples via flow-matching interpolation ----
    noisy_video = []
    video_target = []
    noisy_audio = []
    audio_target = []
    for i in range(B):
        nv, vt = flow_matching_noise(video_latent[i], video_noise[i], timesteps[i])
        na, at = flow_matching_noise(audio_latent[i], audio_noise[i], timesteps[i])
        noisy_video.append(nv)
        video_target.append(vt)
        noisy_audio.append(na)
        audio_target.append(at)

    # ---- I2V: replace first frame in noisy video with clean latent ----
    if is_i2v:
        for i in range(B):
            if first_frame_latent[i] is not None:
                noisy_video[i][:, :1] = first_frame_latent[i]

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
    vid_seq_len = max(v.shape[1] * v.shape[2] * v.shape[3] // (_ph * _pw) for v in video_latent)
    audio_seq_len = max(a.shape[0] for a in audio_latent)

    # ---- Forward pass ----
    t_input = timesteps  # [B]

    pred_vid, pred_audio = model(
        vid=noisy_video,
        audio=noisy_audio,
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

    regularization_cfg = cfg.get("regularization", {})
    fusion_anchor_weight = float(regularization_cfg.get("fusion_anchor_weight", 0.0))
    trainable_param_reference = None
    if fusion_anchor_weight > 0:
        trainable_param_reference = capture_trainable_parameter_reference(model)
        if is_main:
            logger.info(
                "Fusion anchor regularization enabled with weight %.2e over %s trainable tensors",
                fusion_anchor_weight,
                len(trainable_param_reference),
            )

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
        base_dir="./",  # Set base_dir to current directory since manifest paths already include 'training_data/'
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
    
    # ---- Create timestamped run directory for latents ----
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(cfg.output_dir, f"run_{run_timestamp}")
    record_dir = os.path.join(run_dir, "record")
    os.makedirs(record_dir, exist_ok=True)
    if is_main:
        logger.info(f"Latents will be saved to: {record_dir}")
    
    # ---- Loss tracking ----
    loss_history = []  # List of (step, total_loss, video_loss, audio_loss)
    
    # ---- Get checkpoint steps from config ----
    checkpoint_steps = cfg.logging.get("checkpoint_steps", [])
    if is_main and checkpoint_steps:
        logger.info(f"Additional checkpoints will be saved at steps: {checkpoint_steps}")

    validation_runner = None
    validation_steps = set(int(step) for step in checkpoint_steps)
    validation_cfg = cfg.get("validation", {})
    best_ib_score = None
    if is_main and validation_cfg.get("enabled", False):
        validation_runner = ValidationRunner(
            cfg=cfg,
            model=raw_model,
            device=device,
            target_dtype=target_dtype,
            model_specs=model_specs,
            video_config=video_config,
            audio_config=audio_config,
            output_root=os.path.join(run_dir, "validation"),
        )
        logger.info(
            "Validation enabled with %s prompts from %s",
            len(validation_runner.entries),
            validation_cfg.get("manifest_path", "./training_data/validation/manifest.jsonl"),
        )

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
        logger.info(f"  Max optimizer steps: {max_steps}")
        logger.info(f"  Warmup steps:        {warmup_steps}")
        logger.info(f"  Mixed precision:     {cfg.mixed_precision}")
        logger.info(f"  World size:          {world_size}")
        logger.info(f"  Mode:                {cfg.mode}")
        logger.info(f"  Video loss weight:   {cfg.loss.video_weight}")
        logger.info(f"  Audio loss weight:   {cfg.loss.audio_weight}")
        logger.info(f"  Fusion anchor wt:    {fusion_anchor_weight}")
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

    optimizer_step = start_step
    micro_step = start_step * grad_accum_steps
    epoch = 0
    running_loss = 0.0
    running_vloss = 0.0
    running_aloss = 0.0
    running_anchor_loss = 0.0
    running_micro_batches = 0
    step_time_start = time.time()

    scaler = torch.amp.GradScaler('cuda', enabled=(target_dtype == torch.float16))
    optimizer.zero_grad(set_to_none=True)

    while optimizer_step < max_steps:
        if distributed:
            sampler.set_epoch(epoch)

        for batch in dataloader:
            if optimizer_step >= max_steps:
                break

            micro_step += 1

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
                anchor_loss = compute_parameter_anchor_loss(raw_model, trainable_param_reference)
                loss = loss + fusion_anchor_weight * anchor_loss
                loss = loss / grad_accum_steps

            if scaler.is_enabled():
                scaler.scale(loss).backward()
            else:
                loss.backward()

            running_loss += loss.item() * grad_accum_steps
            running_vloss += vloss
            running_aloss += aloss
            running_anchor_loss += anchor_loss.item()
            running_micro_batches += 1

            # ---- Gradient accumulation step ----
            if micro_step % grad_accum_steps != 0:
                continue

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

            optimizer_step += 1

            # ---- Logging ----
            if is_main and optimizer_step % log_every == 0:
                elapsed = time.time() - step_time_start
                steps_per_sec = log_every / elapsed
                avg_loss = running_loss / max(1, running_micro_batches)
                avg_vloss = running_vloss / max(1, running_micro_batches)
                avg_aloss = running_aloss / max(1, running_micro_batches)
                avg_anchor_loss = running_anchor_loss / max(1, running_micro_batches)
                lr = scheduler.get_last_lr()[0]
                vram_gb = torch.cuda.max_memory_allocated(device) / 1e9

                logger.info(
                    f"Step {optimizer_step}/{max_steps} | "
                    f"loss={avg_loss:.4f} (video={avg_vloss:.4f}, audio={avg_aloss:.4f}, anchor={avg_anchor_loss:.4f}) | "
                    f"lr={lr:.2e} | {steps_per_sec:.2f} step/s | VRAM={vram_gb:.1f}GB"
                )
                
                # Track loss history
                loss_history.append((optimizer_step, avg_loss, avg_vloss, avg_aloss))

                if tb_writer:
                    tb_writer.add_scalar("train/loss", avg_loss, optimizer_step)
                    tb_writer.add_scalar("train/video_loss", avg_vloss, optimizer_step)
                    tb_writer.add_scalar("train/audio_loss", avg_aloss, optimizer_step)
                    tb_writer.add_scalar("train/anchor_loss", avg_anchor_loss, optimizer_step)
                    tb_writer.add_scalar("train/lr", lr, optimizer_step)

                running_loss = 0.0
                running_vloss = 0.0
                running_aloss = 0.0
                running_anchor_loss = 0.0
                running_micro_batches = 0
                step_time_start = time.time()

            # ---- Save checkpoint at regular intervals ----
            if is_main and optimizer_step % save_every == 0:
                save_path = os.path.join(cfg.output_dir, f"checkpoint_step{optimizer_step}.pt")
                try:
                    torch.save({
                        "step": optimizer_step,
                        "step_unit": "optimizer_step",
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
                except Exception as e:
                    logger.error(f"Failed to save checkpoint at step {optimizer_step}: {e}")
                    logger.info("Attempting to save model state only...")
                    try:
                        model_only_path = os.path.join(
                            cfg.output_dir,
                            f"checkpoint_step{optimizer_step}_model_only.pt",
                        )
                        torch.save(raw_model.state_dict(), model_only_path)
                        logger.info(f"Saved model-only checkpoint: {model_only_path}")
                    except Exception as e2:
                        logger.error(f"Failed to save model-only checkpoint: {e2}")
            
            # ---- Save checkpoint at specific steps from config ----
            if is_main and optimizer_step in checkpoint_steps:
                save_path = os.path.join(run_dir, f"checkpoint_step{optimizer_step}.pt")
                try:
                    torch.save({
                        "step": optimizer_step,
                        "step_unit": "optimizer_step",
                        "model": raw_model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(),
                        "config": OmegaConf.to_container(cfg),
                    }, save_path)
                    logger.info(f"Saved specific checkpoint: {save_path}")
                except Exception as e:
                    logger.error(f"Failed to save specific checkpoint at step {optimizer_step}: {e}")
                    logger.info("Attempting to save model state only...")
                    try:
                        model_only_path = os.path.join(run_dir, f"checkpoint_step{optimizer_step}_model_only.pt")
                        torch.save(raw_model.state_dict(), model_only_path)
                        logger.info(f"Saved model-only checkpoint: {model_only_path}")
                    except Exception as e2:
                        logger.error(f"Failed to save model-only checkpoint: {e2}")

            # ---- Validation at configured checkpoint steps ----
            if optimizer_step in validation_steps and validation_cfg.get("enabled", False):
                if distributed:
                    dist.barrier()
                if is_main and validation_runner is not None:
                    try:
                        validation_summary = validation_runner.run(optimizer_step)
                        if tb_writer and validation_summary["average_ib_score"] is not None:
                            tb_writer.add_scalar(
                                "validation/average_ib_score",
                                validation_summary["average_ib_score"],
                                optimizer_step,
                            )
                        current_ib = validation_summary["average_ib_score"]
                        if current_ib is not None and (best_ib_score is None or current_ib > best_ib_score):
                            best_ib_score = current_ib
                            best_path = os.path.join(run_dir, "checkpoint_best_ib.pt")
                            torch.save({
                                "step": optimizer_step,
                                "step_unit": "optimizer_step",
                                "best_ib_score": best_ib_score,
                                "model": raw_model.state_dict(),
                                "optimizer": optimizer.state_dict(),
                                "scheduler": scheduler.state_dict(),
                                "config": OmegaConf.to_container(cfg),
                            }, best_path)
                            logger.info(
                                "New best ImageBind score %.6f at step %s. Saved %s",
                                best_ib_score,
                                optimizer_step,
                                best_path,
                            )
                    except Exception as e:
                        logger.exception(f"Validation failed at step {optimizer_step}: {e}")
                if distributed:
                    dist.barrier()
            
            # ---- Save latents periodically to record folder ----
            if is_main and optimizer_step % log_every == 0:
                try:
                    # Save a sample of latents from the current batch
                    latent_save_path = os.path.join(record_dir, f"latents_step{optimizer_step}.pt")
                    latent_data = {
                        "step": optimizer_step,
                        "video_latent": batch["video_latent"][0].cpu() if isinstance(batch["video_latent"], torch.Tensor) else batch["video_latent"][0],
                        "audio_latent": batch["audio_latent"][0].cpu() if isinstance(batch["audio_latent"], torch.Tensor) else batch["audio_latent"][0],
                    }
                    torch.save(latent_data, latent_save_path)
                except Exception as e:
                    logger.warning(f"Failed to save latents at step {optimizer_step}: {e}")

        epoch += 1

    # ---- Final save ----
    if is_main:
        final_path = os.path.join(cfg.output_dir, "checkpoint_final.pt")
        try:
            torch.save({
                "step": optimizer_step,
                "step_unit": "optimizer_step",
                "model": raw_model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "config": OmegaConf.to_container(cfg),
            }, final_path)
            logger.info(f"Training complete at step {optimizer_step}. Final checkpoint: {final_path}")
        except Exception as e:
            logger.error(f"Failed to save final checkpoint: {e}")
            logger.info("Attempting to save model state only...")
            try:
                model_only_path = os.path.join(cfg.output_dir, "checkpoint_final_model_only.pt")
                torch.save(raw_model.state_dict(), model_only_path)
                logger.info(f"Saved model-only checkpoint: {model_only_path}")
            except Exception as e2:
                logger.error(f"Failed to save model-only checkpoint: {e2}")
        
        # ---- Generate and save loss plot ----
        if loss_history:
            try:
                steps, total_losses, video_losses, audio_losses = zip(*loss_history)
                
                fig, axes = plt.subplots(2, 2, figsize=(15, 10))
                fig.suptitle('Training Loss History', fontsize=16)
                
                # Total loss
                axes[0, 0].plot(steps, total_losses, 'b-', linewidth=2)
                axes[0, 0].set_xlabel('Training Step')
                axes[0, 0].set_ylabel('Total Loss')
                axes[0, 0].set_title('Total Loss')
                axes[0, 0].grid(True, alpha=0.3)
                
                # Video loss
                axes[0, 1].plot(steps, video_losses, 'r-', linewidth=2)
                axes[0, 1].set_xlabel('Training Step')
                axes[0, 1].set_ylabel('Video Loss')
                axes[0, 1].set_title('Video Loss')
                axes[0, 1].grid(True, alpha=0.3)
                
                # Audio loss
                axes[1, 0].plot(steps, audio_losses, 'g-', linewidth=2)
                axes[1, 0].set_xlabel('Training Step')
                axes[1, 0].set_ylabel('Audio Loss')
                axes[1, 0].set_title('Audio Loss')
                axes[1, 0].grid(True, alpha=0.3)
                
                # Combined view
                axes[1, 1].plot(steps, total_losses, 'b-', linewidth=2, label='Total Loss', alpha=0.7)
                axes[1, 1].plot(steps, video_losses, 'r-', linewidth=2, label='Video Loss', alpha=0.7)
                axes[1, 1].plot(steps, audio_losses, 'g-', linewidth=2, label='Audio Loss', alpha=0.7)
                axes[1, 1].set_xlabel('Training Step')
                axes[1, 1].set_ylabel('Loss')
                axes[1, 1].set_title('All Losses Combined')
                axes[1, 1].legend()
                axes[1, 1].grid(True, alpha=0.3)
                
                plt.tight_layout()
                
                # Save plot
                plot_path = os.path.join(run_dir, "loss_plot.png")
                plt.savefig(plot_path, dpi=150, bbox_inches='tight')
                plt.close()
                logger.info(f"Saved loss plot: {plot_path}")
                
                # Also save loss history as JSON for later analysis
                loss_json_path = os.path.join(run_dir, "loss_history.json")
                loss_data = {
                    "steps": list(steps),
                    "total_loss": list(total_losses),
                    "video_loss": list(video_losses),
                    "audio_loss": list(audio_losses),
                }
                with open(loss_json_path, 'w') as f:
                    json.dump(loss_data, f, indent=2)
                logger.info(f"Saved loss history: {loss_json_path}")
                
            except Exception as e:
                logger.warning(f"Failed to generate loss plot: {e}")

        if tb_writer:
            tb_writer.close()

    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
