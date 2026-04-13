# Fine-tuning Procedure of the Ovi FusionModel

> This document describes, step by step, how the codebase fine-tunes the **Ovi FusionModel** — a dual-branch Diffusion Transformer (DiT) that jointly generates synchronised video and audio from text (or image + text) prompts.  
> All code references are to files in this repository.

---

## Overview

The fine-tuning pipeline consists of three major phases:

1. **Data preparation** — raw video clips are encoded into compact latent representations offline.  
2. **Training** — the pre-trained FusionModel is optimised on those latents using a flow-matching objective.  
3. **Checkpoint saving and evaluation** — trained weights are saved and can be loaded directly by the inference engine.

---

## Phase 1 — Data Preparation (`prepare_training_data_v2.py`)

Before any gradient update is performed, every training video is pre-processed into latent tensors that the model can directly consume during training.  This one-time offline step prevents repeated encoding work inside the training loop.

### 1.1  Input Format

The user provides a raw manifest file (`raw_manifest.jsonl`).  Each line is a JSON object containing at minimum:

```jsonl
{"video_path": "clips/clip001.mp4", "text_prompt": "A dog running on a beach at sunset"}
{"video_path": "clips/clip002.mp4", "text_prompt": "Two people dancing", "image_path": "frames/clip002.png"}
```

The optional `image_path` field activates **image-to-video (I2V)** mode for that sample, where the first video frame is conditioned on a clean reference image.

### 1.2  Video Encoding

For each sample the following is done:

1. **Frame extraction** — `num_raw_frames = (video_latent_length − 1) × 4 + 1` frames are sampled evenly from the video using OpenCV.  For a 5-second model this is 121 frames; for a 10-second model it is 241 frames.
2. **Spatial normalisation** — frames are resized so that their total pixel area matches the model's target area (e.g. 921 600 pixels for the `960×960` variant) while preserving aspect ratio.  Height and width are snapped to multiples of 32.  Pixel values are normalised from `[0, 255]` to `[−1, 1]`.
3. **VAE encoding** — the resulting `[C, F, H, W]` video tensor is passed through the **Wan 2.2 Video VAE** (`init_wan_vae_2_2`).  This encoder applies a **16× spatial downsample** and a **4× temporal downsample**, compressing the video into a latent tensor of shape `[48, F_lat, H_lat, W_lat]` where 48 is the latent channel dimension.

### 1.3  Audio Encoding

1. The audio track is extracted from the MP4 file using `torchaudio` and resampled to 16 kHz mono.  Duration is trimmed or zero-padded to match the model's target duration (5 s or 10 s).
2. The waveform is then encoded by the **MMAudio VAE** (`init_mmaudio_vae`), producing an audio latent of shape `[L, 20]` where `L` is the audio sequence length (157 for 5-second models) and 20 is the audio latent channel dimension.

### 1.4  Text Encoding

The text prompt is tokenised and encoded by a **UMT5-XXL T5 encoder** (`init_text_model`) operating in `bfloat16`.  The result is a sequence of token embeddings with shape `[text_len, 4096]`.

### 1.5  First-Frame Latent (I2V only)

If `image_path` is provided, the reference image is preprocessed (resized to the same spatial area) and encoded by the Video VAE, yielding a single-frame latent of shape `[48, 1, H_lat, W_lat]`.  This will later replace the first noisy video frame during training, conditioning the model on the clean reference.

### 1.6  Output

Each sample is saved as a `.pt` file containing all four tensors:

```
training_data/
├── latents/
│   ├── 000000.pt   # {video_latent, audio_latent, text_embedding, [first_frame_latent]}
│   └── ...
└── manifest.jsonl  # index of all .pt files; consumed by train_v2.py
```

---

## Phase 2 — Model Fine-tuning (`train_v2.py`)

### 2.1  Model Initialisation

The **FusionModel** is built by calling Ovi's own factory function:

```python
model, video_config, audio_config = init_fusion_score_model_ovi(rank=0, meta_init=True)
```

`FusionModel` wraps two independent **WanModel** (DiT backbone) instances — one for video and one for audio — each with **30 transformer blocks**.  Into every block of both branches, the class `FusionModel.inject_cross_attention_kv_projections()` inserts four new learnable linear layers:

| Layer | Role |
|---|---|
| `k_fusion` | Projects the *other* modality's tokens into key vectors for cross-modal attention |
| `v_fusion` | Projects the *other* modality's tokens into value vectors |
| `pre_attn_norm_fusion` | Layer normalisation applied before the cross-modal keys/values |
| `norm_k_fusion` | RMS normalisation on the resulting key vectors |

These fusion projections are **the core mechanism** by which video and audio tokens attend to each other inside every transformer block.

Pre-trained weights are then loaded from the checkpoint directory:

```python
load_fusion_checkpoint(model, checkpoint_path=ckpt_path, from_meta=True)
```

The model is moved to the target device and cast to `bfloat16`.  `model.set_rope_params()` is called to initialise Rotary Position Embeddings (RoPE) that encode 3D spatial-temporal positions for video and 1D positions for audio.

### 2.2  Freeze Strategy

Two fine-tuning modes are supported (set via `finetune.mode` in the config):

| Mode | Parameters Updated | Typical VRAM |
|---|---|---|
| `fusion_only` | Only `k_fusion`, `v_fusion`, `pre_attn_norm_fusion`, `norm_k_fusion` in all 30×2 blocks | ~24 GB |
| `full` | All model parameters | ~40 GB |

In `fusion_only` mode, `freeze_non_fusion()` iterates over every named parameter and sets `requires_grad = False` for anything whose name does not contain the substring `"fusion"`.  This focuses the gradient signal entirely on the cross-modal alignment layers and avoids catastrophic forgetting of the large pre-trained video/audio encoders.

### 2.3  Gradient Checkpointing

When `finetune.gradient_checkpointing: true`, gradient checkpointing is enabled on both the video and audio WanModel branches:

```python
model.video_model.set_gradient_checkpointing(True)
model.audio_model.set_gradient_checkpointing(True)
```

Inside `FusionModel.forward()`, every call to `single_fusion_block_forward` is also wrapped by `gradient_checkpointing(enabled=self.training and self.gradient_checkpointing, ...)`.  This re-computes activations during the backward pass instead of storing them, reducing peak VRAM by approximately 40% at the cost of extra computation.

### 2.4  Distributed Training

If more than one GPU is available (`WORLD_SIZE > 1`), **PyTorch Distributed Data Parallel (DDP)** is used:

```python
torch.distributed.init_process_group("nccl")
model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])
```

The dataset is split across GPUs with `DistributedSampler`.  Each GPU processes a disjoint subset of the batch in every step and gradients are automatically averaged across all ranks before the optimiser update.

### 2.5  Dataset and DataLoader

`OviLatentDataset` reads the `manifest.jsonl` file produced by Phase 1.  For each index it loads the corresponding `.pt` file and returns the four tensors.  A custom `collate_fn` assembles them into **lists of tensors** rather than stacked batches, because the FusionModel's forward pass expects a list (one element per sample) to support variable-length video and audio sequences.

### 2.6  Timestep Sampling (Flow-Matching)

The training objective follows **Rectified Flow / Flow Matching** with a logit-normal timestep distribution, as used in the original Ovi and Wan models.

For each training step, a continuous timestep $t \in (0, 1)$ is sampled per sample using:

$$\sigma_{\text{raw}} = \text{sigmoid}(\mathcal{N}(0,1))$$

$$t = \frac{\text{shift} \cdot \sigma_{\text{raw}}}{1 + (\text{shift} - 1) \cdot \sigma_{\text{raw}}}$$

where `shift = 5.0` (from `flow.shift` in the config) biases the distribution towards the middle of the trajectory, matching the scheduler used during inference.  The result is scaled to the model's 0–1000 integer range: `t_1000 = t × 1000`.

### 2.7  Building Noisy Inputs

For each sample in the mini-batch, independent Gaussian noise $\epsilon \sim \mathcal{N}(0, I)$ is sampled for both video and audio latents.  The noisy input at time $t$ is computed via linear interpolation:

$$x_t = (1 - t)\,x_0 + t\,\epsilon$$

The velocity (the target the model must predict) is:

$$v = \epsilon - x_0$$

For I2V samples, the **first temporal slice of the noisy video latent is replaced with the clean first-frame latent**, anchoring the model's prediction to the provided reference image.

### 2.8  Classifier-Free Guidance (CFG) Dropout

With probability `cfg_drop_rate = 0.1`, the text embedding for a given sample is replaced with an all-zero tensor.  This trains the model in an unconditional mode for 10% of iterations, enabling classifier-free guidance to be applied at inference time to improve prompt adherence.

### 2.9  Forward Pass

The noisy latents and (possibly zeroed) text embeddings are fed through `FusionModel.forward()` using exactly the same call signature as the inference engine:

```python
pred_vid_list, pred_aud_list = model(
    vid=noisy_videos,           # List[Tensor[C, F, H, W]]
    audio=noisy_audios,         # List[Tensor[L, C_audio]]
    t=t_1000,                   # Tensor[B]
    vid_context=context_vid,    # List[Tensor[text_len, D]]
    audio_context=context_aud,  # List[Tensor[text_len, D]]
    vid_seq_len=vid_seq_len,
    audio_seq_len=aud_seq_len,
    first_frame_is_clean=any_i2v,
)
```

Inside `FusionModel.forward()`, the processing proceeds as follows for each of the 30 block pairs:

1. **Video self-attention** — video tokens attend to each other within `vid_block`, conditioned on the timestep via per-token modulation scalings derived from sinusoidal time embeddings.
2. **Audio self-attention** — audio tokens attend to each other within `audio_block` in the same way.
3. **Audio → Video cross-modal attention** — audio tokens (after `pre_attn_norm_fusion`) are projected to keys and values via `k_fusion`/`v_fusion`, and the video query tokens attend to them using FlashAttention with 3D-RoPE applied to the queries and keys.
4. **Video → Audio cross-modal attention** — the symmetric operation: video tokens are projected and attended to by the audio queries.

This bidirectional cross-modal attention in every block is what allows the two branches to co-condition on each other throughout the denoising trajectory.

### 2.10  Loss Computation

The loss is the **mean squared error (MSE) between the model's predicted velocity and the ground-truth velocity**, computed separately for video and audio and then combined:

$$\mathcal{L}_{\text{video}} = \frac{1}{B}\sum_{i=1}^{B} \|\hat{v}^{(i)}_{\text{vid}} - v^{(i)}_{\text{vid}}\|^2$$

$$\mathcal{L}_{\text{audio}} = \frac{1}{B}\sum_{i=1}^{B} \|\hat{v}^{(i)}_{\text{aud}} - v^{(i)}_{\text{aud}}\|^2$$

$$\mathcal{L} = w_v \cdot \mathcal{L}_{\text{video}} + w_a \cdot \mathcal{L}_{\text{audio}}$$

where $w_v = w_a = 1.0$ by default.  Both losses are cast to `float32` before computation to prevent underflow in `bfloat16`.

### 2.11  Gradient Accumulation and Optimiser Update

The loss is divided by `gradient_accumulation_steps` (default 1) before `loss.backward()` is called, so that the effective gradient is equivalent to a larger batch.  After every `gradient_accumulation_steps` backward passes:

1. **Gradient clipping** — `torch.nn.utils.clip_grad_norm_` clips the global gradient norm to `max_grad_norm = 1.0` to prevent training instability.
2. **Learning-rate schedule** — the learning rate follows a **linear warm-up** for the first `warmup_steps = 100` steps, then decays according to a **cosine schedule**:

$$\text{lr}(s) = \text{lr}_{\text{base}} \times \frac{1 + \cos\!\left(\pi \cdot \frac{s - s_{\text{warm}}}{s_{\text{max}} - s_{\text{warm}}}\right)}{2}$$

3. **AdamW update** — the optimiser (`AdamW`, $\beta_1=0.9$, $\beta_2=0.999$, weight decay $= 0.01$, peak lr $= 10^{-5}$) performs a single parameter update on all `requires_grad=True` parameters, then the gradient buffers are zeroed.

### 2.12  Logging and Checkpointing

On the main process (`rank 0`):

- **TensorBoard** scalars `train/loss`, `train/video_loss`, `train/audio_loss`, and `train/lr` are written every `log_every = 10` steps to `training_outputs/tb_logs/`.
- **Checkpoints** are saved every `save_every = 500` steps as PyTorch `.pt` files containing the full `model.state_dict()` and `optimizer.state_dict()`.  A `checkpoint_latest.pt` is always kept for easy resumption.
- After `max_steps = 1000` total steps, a final `checkpoint_final.pt` is saved and training terminates.

---

## Phase 3 — Using the Fine-tuned Model

Because the checkpoints store the standard `FusionModel.state_dict()` (identical key names to Ovi's original `.safetensors`), they can be loaded directly into the inference engine:

```python
load_fusion_checkpoint(engine.model, "training_outputs/checkpoint_final.pt", from_meta=False)
```

No conversion step is needed.  The inference pipeline (`OviFusionEngine`, `inference.py`) remains completely unchanged — only the weights differ.

---

## Summary of Key Design Decisions

| Decision | Rationale |
|---|---|
| **Offline latent pre-encoding** | Encoding with three large models (VAE×2, T5) is computationally expensive; doing it once avoids a bottleneck in the training loop. |
| **Flow-matching velocity objective** | Matches the training objective of the original Ovi / Wan models, ensuring the fine-tuned model stays on the same learned manifold. |
| **Logit-normal timestep sampling with shift** | Concentrates training effort on the perceptually most important mid-range timesteps, consistent with the Wan 2.x training recipe. |
| **`fusion_only` freeze mode** | Only the 4 injected cross-modal projection layers per block (out of ~5 billion total parameters) are updated, dramatically reducing VRAM and training time while specifically improving audio–video alignment. |
| **Same forward signature as inference** | Guarantees that no discrepancy exists between training and inference; the fine-tuned weights plug directly back into the existing pipeline. |
| **Gradient checkpointing** | Makes it feasible to fine-tune the 960×960 model on a single 40 GB GPU. |
| **CFG dropout (10%)** | Teaches the model an unconditional mode, enabling classifier-free guidance at inference time without any extra training. |
