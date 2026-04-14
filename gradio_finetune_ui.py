#!/usr/bin/env python3
"""
gradio_finetune_ui.py  –  Gradio UI for the Ovi fine-tuning pipeline.

Three tabs:
  📋 1 · Prepare Data  – build/edit a raw manifest then run prepare_training_data_v2.py
  🏋️  2 · Fine-tune    – configure hyper-parameters and launch train_v2.py
  🎬  3 · Inference    – load OviFusionEngine (+ optional fine-tuned checkpoint) and generate

This file does NOT modify any existing files.  It only adds new UI code.

Usage
-----
    python gradio_finetune_ui.py
    python gradio_finetune_ui.py --server_name 0.0.0.0 --server_port 7892 --share
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
import threading
from pathlib import Path

import gradio as gr
from omegaconf import OmegaConf


# ─── CLI ──────────────────────────────────────────────────────────────────────
def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Ovi Fine-tuning & Inference Gradio UI")
    p.add_argument("--server_name", default="127.0.0.1",
                   help="Bind address (use 0.0.0.0 for LAN access)")
    p.add_argument("--server_port", type=int, default=7892)
    p.add_argument("--share", default=1,action="store_true",
                   help="Create a public Gradio share link")
    return p


# ─── Constants ────────────────────────────────────────────────────────────────
MODEL_NAMES = ["720x720_5s", "960x960_5s", "960x960_10s"]

# ─── Global inference engine (lazy-loaded once per session) ──────────────────
_ovi_engine = None
_loaded_cfg_key: tuple | None = None

# ─── Single active subprocess (data-prep or training) ────────────────────────
_active_proc: subprocess.Popen | None = None
_proc_lock = threading.Lock()


# ──────────────────────────────────────────────────────────────────────────────
# Low-level subprocess helpers
# ──────────────────────────────────────────────────────────────────────────────

def _kill_active_proc() -> bool:
    """Terminate the currently running subprocess (if any). Returns True if killed."""
    global _active_proc
    with _proc_lock:
        p = _active_proc
    if p is not None and p.poll() is None:
        p.terminate()
        try:
            p.wait(timeout=5)
        except subprocess.TimeoutExpired:
            p.kill()
        return True
    return False


def _stream_cmd(cmd: list[str]):
    """
    Generator: launch *cmd* as a subprocess, yield the growing stdout+stderr
    log each time a new line arrives.  Suitable for binding to a gr.Textbox.
    """
    global _active_proc
    _kill_active_proc()

    with _proc_lock:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            cwd=os.getcwd(),
            env=os.environ.copy(),
        )
        _active_proc = proc

    banner = f"▶  {' '.join(cmd)}\n{'─' * 60}\n"
    lines: list[str] = [banner]
    yield banner

    for line in iter(proc.stdout.readline, ""):
        lines.append(line)
        yield "".join(lines)

    proc.wait()
    with _proc_lock:
        _active_proc = None

    suffix = (
        "\n✅  Finished successfully.\n"
        if proc.returncode == 0
        else f"\n❌  Process exited with code {proc.returncode}.\n"
    )
    lines.append(suffix)
    yield "".join(lines)


# ──────────────────────────────────────────────────────────────────────────────
# Tab 1 – Prepare Training Data
# ──────────────────────────────────────────────────────────────────────────────

def _save_manifest(df, manifest_path: str) -> str:
    """Serialize the DataTable rows to a .jsonl raw manifest file."""
    if df is None:
        return "❌  Table is empty – add at least one row first."

    # Gradio 4 returns a pandas DataFrame; older versions return list-of-lists.
    try:
        import pandas as pd
        if isinstance(df, pd.DataFrame):
            iter_rows = [tuple(row) for _, row in df.iterrows()]
        else:
            iter_rows = [tuple(r) for r in df]
    except ImportError:
        iter_rows = [tuple(r) for r in df]

    records: list[dict] = []
    for row in iter_rows:
        vp = str(row[0]).strip() if len(row) > 0 and row[0] else ""
        tp = str(row[1]).strip() if len(row) > 1 and row[1] else ""
        ip = str(row[2]).strip() if len(row) > 2 and row[2] else ""
        if not vp or not tp:
            continue
        rec: dict = {"video_path": vp, "text_prompt": tp}
        if ip:
            rec["image_path"] = ip
        records.append(rec)

    if not records:
        return "❌  No valid rows (video_path + text_prompt are both required)."

    out = Path(manifest_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as fh:
        for r in records:
            fh.write(json.dumps(r) + "\n")

    return f"✅  Saved {len(records)} entries → {out}"


def _load_manifest(file_path) -> tuple:
    """Load a .jsonl manifest file into the DataTable."""
    if file_path is None:
        return gr.update(), "⚠️  No file selected."

    fp = file_path if isinstance(file_path, str) else file_path.name
    rows: list[list] = []
    with open(fp) as fh:
        for line in fh:
            line = line.strip()
            if line:
                r = json.loads(line)
                rows.append([
                    r.get("video_path",  ""),
                    r.get("text_prompt", ""),
                    r.get("image_path",  ""),
                ])

    return rows, f"✅  Loaded {len(rows)} entries from {fp}"


def _run_data_prep(
    input_manifest: str,
    output_dir: str,
    ckpt_dir: str,
    model_name: str,
    device: int,
):
    """Yield streaming logs from prepare_training_data_v2.py."""
    if not input_manifest or not os.path.isfile(input_manifest):
        yield f"❌  Manifest file not found: {input_manifest!r}\n"
        return

    cmd = [
        sys.executable, "prepare_training_data_v2.py",
        "--input_manifest", str(input_manifest),
        "--output_dir",     str(output_dir),
        "--ckpt_dir",       str(ckpt_dir),
        "--model_name",     str(model_name),
        "--device",         str(int(device)),
    ]
    yield from _stream_cmd(cmd)


# ──────────────────────────────────────────────────────────────────────────────
# Tab 2 – Fine-tune
# ──────────────────────────────────────────────────────────────────────────────

def _build_yaml_cfg(
    model_name, ckpt_dir, output_dir, data_manifest, mixed_precision,
    finetune_mode, gradient_ckpt,
    seed, batch_size, max_steps, lr, weight_decay,
    warmup_steps, lr_sched, grad_accum, max_grad_norm,
    flow_shift, cfg_drop, vid_w, aud_w, save_every, log_every,
) -> str:
    """Build a training YAML string from UI values."""
    cfg = OmegaConf.create({
        "model_name":      str(model_name),
        "ckpt_dir":        str(ckpt_dir),
        "output_dir":      str(output_dir),
        "data_manifest":   str(data_manifest),
        "mixed_precision": str(mixed_precision),
        "finetune": {
            "mode":                   str(finetune_mode),
            "gradient_checkpointing": bool(gradient_ckpt),
        },
        "training": {
            "seed":                        int(seed),
            "batch_size":                  int(batch_size),
            "max_steps":                   int(max_steps),
            "learning_rate":               float(lr),
            "weight_decay":                float(weight_decay),
            "warmup_steps":                int(warmup_steps),
            "lr_scheduler":                str(lr_sched),
            "gradient_accumulation_steps": int(grad_accum),
            "max_grad_norm":               float(max_grad_norm),
            "save_every":                  int(save_every),
            "log_every":                   int(log_every),
        },
        "data": {"num_workers": 4, "pin_memory": True},
        "flow": {
            "shift":         float(flow_shift),
            "cfg_drop_rate": float(cfg_drop),
        },
        "loss": {
            "video_weight": float(vid_w),
            "audio_weight": float(aud_w),
        },
    })
    return OmegaConf.to_yaml(cfg)


def _preview_config(*args) -> str:
    """Return the YAML string for the config preview box."""
    return _build_yaml_cfg(*args)


def _run_training(*args_plus_gpus):
    """Save the generated config and stream logs from train_v2.py."""
    *cfg_args, num_gpus = args_plus_gpus
    output_dir = cfg_args[2]

    os.makedirs(output_dir, exist_ok=True)
    yaml_text = _build_yaml_cfg(*cfg_args)
    cfg_path = os.path.join(output_dir, "training_config_ui.yaml")
    with open(cfg_path, "w") as fh:
        fh.write(yaml_text)

    n = int(num_gpus)
    cmd = (
        ["torchrun", f"--nproc_per_node={n}", "train_v2.py", "--config", cfg_path]
        if n > 1
        else [sys.executable, "train_v2.py", "--config", cfg_path]
    )
    yield from _stream_cmd(cmd)


def _stop_any_process() -> str:
    killed = _kill_active_proc()
    return "⏹️  Process stopped." if killed else "ℹ️  No active process to stop."


# ──────────────────────────────────────────────────────────────────────────────
# Tab 3 – Inference
# ──────────────────────────────────────────────────────────────────────────────

def _load_model(
    ckpt_dir: str,
    model_name: str,
    finetuned_ckpt: str,
    mode: str,
    cpu_offload: bool,
    fp8: bool,
    qint8: bool,
) -> str:
    """Load (or reuse) an OviFusionEngine instance with the given settings."""
    global _ovi_engine, _loaded_cfg_key

    import torch
    from ovi.ovi_fusion_engine import OviFusionEngine, DEFAULT_CONFIG

    key = (ckpt_dir, model_name, finetuned_ckpt or "", mode, cpu_offload, fp8, qint8)
    if _ovi_engine is not None and _loaded_cfg_key == key:
        return "✅  Model already loaded (settings unchanged)."

    # Free any previously loaded model to reclaim VRAM.
    if _ovi_engine is not None:
        del _ovi_engine
        _ovi_engine = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Build config from DEFAULT_CONFIG, overriding with UI values.
    base = dict(DEFAULT_CONFIG)
    base.update({
        "ckpt_dir":    str(ckpt_dir),
        "model_name":  str(model_name),
        "cpu_offload": bool(cpu_offload),
        "fp8":         bool(fp8),
        "qint8":       bool(qint8),
        "mode":        str(mode),
    })
    if finetuned_ckpt and os.path.isfile(finetuned_ckpt):
        base["finetuned_checkpoint"] = str(finetuned_ckpt)

    cfg = OmegaConf.create(base)

    try:
        device = 0 if torch.cuda.is_available() else "cpu"
        _ovi_engine = OviFusionEngine(
            config=cfg,
            device=device,
            target_dtype=torch.bfloat16,
        )
        _loaded_cfg_key = key
        ft_note = (
            f" + fine-tuned: {os.path.basename(finetuned_ckpt)}"
            if (finetuned_ckpt and os.path.isfile(finetuned_ckpt))
            else " (base weights)"
        )
        return f"✅  Loaded {model_name}{ft_note}"
    except Exception as exc:
        _ovi_engine = None
        return f"❌  Failed to load model:\n{exc}"


def _run_inference(
    text_prompt: str,
    image,
    video_height: int,
    video_width: int,
    seed: int,
    solver_name: str,
    sample_steps: int,
    shift: float,
    video_gs: float,
    audio_gs: float,
    slg_layer: int,
    video_neg: str,
    audio_neg: str,
):
    """Run inference using the already-loaded OviFusionEngine."""
    global _ovi_engine
    if _ovi_engine is None:
        raise gr.Error(
            "Model is not loaded — go to 'Model Settings' and click "
            "'Load / Reload Model' first."
        )

    from ovi.utils.io_utils import save_video

    image_path = str(image) if image and os.path.isfile(str(image)) else None

    try:
        gen_video, gen_audio, _ = _ovi_engine.generate(
            text_prompt=text_prompt,
            image_path=image_path,
            video_frame_height_width=[int(video_height), int(video_width)],
            seed=int(seed),
            solver_name=solver_name,
            sample_steps=int(sample_steps),
            shift=float(shift),
            video_guidance_scale=float(video_gs),
            audio_guidance_scale=float(audio_gs),
            slg_layer=int(slg_layer),
            video_negative_prompt=video_neg or "",
            audio_negative_prompt=audio_neg or "",
        )
        tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
        save_video(tmp.name, gen_video, gen_audio, fps=24, sample_rate=16000)
        return tmp.name, "✅  Video generated successfully."
    except Exception as exc:
        raise gr.Error(str(exc))


# ──────────────────────────────────────────────────────────────────────────────
# Build the Gradio UI
# ──────────────────────────────────────────────────────────────────────────────

def build_ui() -> gr.Blocks:
    with gr.Blocks(
        title="Ovi Fine-tuning & Inference UI",
    ) as demo:

        gr.Markdown(
            """# 🎬 Ovi Fine-tuning & Inference UI
A three-step workflow: **📋 Prepare Data → 🏋️ Fine-tune → 🎬 Infer**.  """
        )

        with gr.Tabs():

            # ══════════════════════════════════════════════════════════════════
            # TAB 1 – Prepare Training Data
            # ══════════════════════════════════════════════════════════════════
            with gr.TabItem("📋 1 · Prepare Data"):
                gr.Markdown(
                    """### Prepare Training Data
Build or load a raw manifest (JSONL), then encode videos + audio into latents using
`prepare_training_data_v2.py`.

**Workflow:**
1. Fill the table (or load an existing manifest) with *(video_path, text_prompt, optional image_path)*.
2. Click **💾 Save Manifest** to write the JSONL file.
3. Set encoding settings on the right.
4. Click **🚀 Run Encoding** — the log streams in real time below.
"""
                )

                with gr.Row():
                    # ── left: manifest editor ─────────────────────────────────
                    with gr.Column(scale=3):
                        gr.Markdown("#### 📝 Sample Manifest Editor")
                        sample_table = gr.Dataframe(
                            headers=["video_path", "text_prompt", "image_path (optional)"],
                            datatype=["str", "str", "str"],
                            column_count=3,
                            label="Training Samples",
                            interactive=True,
                        )

                        with gr.Row():
                            load_file = gr.File(
                                label="Load existing manifest (.jsonl)",
                                file_types=[".jsonl"],
                            )
                            load_manifest_btn = gr.Button("⬆️ Load from File")

                        with gr.Row():
                            manifest_path_inp = gr.Textbox(
                                value="training_data/raw_manifest.jsonl",
                                label="Save manifest to (path)",
                            )
                            save_manifest_btn = gr.Button("💾 Save Manifest")

                        save_status = gr.Textbox(
                            label="Status", interactive=False, lines=1,
                        )

                    # ── right: settings ────────────────────────────────────────
                    with gr.Column(scale=2):
                        gr.Markdown("#### ⚙️ Encoding Settings")
                        dp_output_dir = gr.Textbox(
                            value="training_data", label="Output Directory",
                            info="Latents will be saved under <output_dir>/latents/"
                        )
                        dp_ckpt_dir = gr.Textbox(
                            value="./ckpts", label="Checkpoint Directory",
                            info="Root folder containing downloaded model weights"
                        )
                        dp_model_name = gr.Dropdown(
                            choices=MODEL_NAMES, value="960x960_5s",
                            label="Model Name",
                        )
                        dp_device = gr.Number(
                            value=0, precision=0, minimum=0,
                            label="CUDA Device Index",
                        )
                        with gr.Row():
                            dp_run_btn  = gr.Button("🚀 Run Encoding", variant="primary")
                            dp_stop_btn = gr.Button("⏹️ Stop", variant="stop")

                dp_log = gr.Textbox(
                    label="Encoding Log",
                    lines=18, max_lines=50,
                    interactive=False,
                    placeholder="Logs will stream here…",
                )

                # ── wiring ────────────────────────────────────────────────────
                load_manifest_btn.click(
                    fn=_load_manifest,
                    inputs=[load_file],
                    outputs=[sample_table, save_status],
                )
                save_manifest_btn.click(
                    fn=_save_manifest,
                    inputs=[sample_table, manifest_path_inp],
                    outputs=[save_status],
                )
                dp_run_btn.click(
                    fn=_run_data_prep,
                    inputs=[manifest_path_inp, dp_output_dir, dp_ckpt_dir,
                            dp_model_name, dp_device],
                    outputs=[dp_log],
                )
                dp_stop_btn.click(fn=_stop_any_process, outputs=[dp_log])

            # ══════════════════════════════════════════════════════════════════
            # TAB 2 – Fine-tune
            # ══════════════════════════════════════════════════════════════════
            with gr.TabItem("🏋️ 2 · Fine-tune"):
                gr.Markdown(
                    """### Fine-tune the Model
Configure all hyper-parameters below.  Click **👁️ Preview YAML** to inspect the
generated config, then **🚀 Start Training** to launch `train_v2.py`.

The config is auto-saved to `<output_dir>/training_config_ui.yaml` before training starts.
Use **⏹️ Stop Training** to terminate the process at any time.
"""
                )

                with gr.Row():
                    # ── column A: paths & strategy ────────────────────────────
                    with gr.Column():
                        gr.Markdown("#### 📁 Paths & Model")
                        tr_model    = gr.Dropdown(
                            choices=MODEL_NAMES, value="960x960_5s", label="Model Name",
                        )
                        tr_ckpt_dir = gr.Textbox(value="./ckpts",              label="Pretrained Checkpoint Dir")
                        tr_out_dir  = gr.Textbox(value="./training_outputs",   label="Output Directory")
                        tr_manifest = gr.Textbox(
                            value="training_data/manifest.jsonl",
                            label="Encoded Manifest",
                            info="Output manifest from Tab 1 (training_data/manifest.jsonl)",
                        )
                        tr_prec = gr.Dropdown(
                            choices=["bf16", "fp32"], value="bf16", label="Mixed Precision",
                        )

                        gr.Markdown("#### 🎛️ Fine-tune Strategy")
                        tr_mode = gr.Dropdown(
                            choices=["fusion_only", "full"], value="fusion_only",
                            label="Fine-tune Mode",
                            info=(
                                "fusion_only → only cross-modal fusion layers (recommended, fast).\n"
                                "full → all parameters (slow, requires more VRAM)."
                            ),
                        )
                        tr_gc = gr.Checkbox(
                            value=True, label="Gradient Checkpointing",
                            info="Saves VRAM at the cost of slightly slower training.",
                        )

                    # ── column B: hyper-params ────────────────────────────────
                    with gr.Column():
                        gr.Markdown("#### 📊 Hyper-parameters")
                        tr_seed      = gr.Number(value=42,    precision=0, label="Seed")
                        tr_batch     = gr.Number(value=1,     precision=0, minimum=1, label="Batch Size")
                        tr_steps     = gr.Number(value=1000,  precision=0, minimum=1, label="Max Steps")
                        tr_lr        = gr.Number(value=1e-5,               label="Learning Rate")
                        tr_wd        = gr.Number(value=0.01,               label="Weight Decay")
                        tr_warmup    = gr.Number(value=100,   precision=0, label="Warmup Steps")
                        tr_sched     = gr.Dropdown(
                            choices=["cosine", "constant"], value="cosine", label="LR Scheduler",
                        )
                        tr_accum     = gr.Number(
                            value=1, precision=0, minimum=1,
                            label="Gradient Accumulation Steps",
                            info="Effective batch = batch_size × grad_accum.",
                        )
                        tr_gnorm     = gr.Number(value=1.0,               label="Max Grad Norm")
                        tr_save_ev   = gr.Number(value=500,  precision=0, label="Save Every N Steps")
                        tr_log_ev    = gr.Number(value=10,   precision=0, label="Log Every N Steps")

                    # ── column C: flow, loss, GPU, preview ────────────────────
                    with gr.Column():
                        gr.Markdown("#### 🌊 Flow-matching & Loss")
                        tr_shift    = gr.Slider(
                            0.0, 20.0, value=5.0, step=0.5, label="Flow Shift",
                        )
                        tr_cfg_drop = gr.Slider(
                            0.0, 0.5, value=0.1, step=0.01,
                            label="CFG Drop Rate",
                            info="Probability of dropping text conditioning (classifier-free guidance).",
                        )
                        tr_vid_w    = gr.Slider(
                            0.0, 5.0, value=1.0, step=0.1, label="Video Loss Weight",
                        )
                        tr_aud_w    = gr.Slider(
                            0.0, 5.0, value=1.0, step=0.1, label="Audio Loss Weight",
                        )

                        gr.Markdown("#### 🖥️ Multi-GPU")
                        tr_gpus = gr.Number(
                            value=1, precision=0, minimum=1, maximum=16,
                            label="Number of GPUs",
                            info="If > 1, uses torchrun --nproc_per_node=N.",
                        )

                        gr.Markdown("#### 📄 Config Preview")
                        tr_preview_btn = gr.Button("👁️ Preview YAML Config")
                        tr_yaml_box    = gr.Code(
                            language="yaml", label="Generated Config YAML", lines=22,
                        )

                with gr.Row():
                    tr_start_btn = gr.Button("🚀 Start Training", variant="primary")
                    tr_stop_btn  = gr.Button("⏹️ Stop Training",  variant="stop")

                tr_log = gr.Textbox(
                    label="Training Log",
                    lines=22, max_lines=80,
                    interactive=False,
                    placeholder="Logs will stream here…",
                )

                # ── helpers ────────────────────────────────────────────────────
                _cfg_inputs = [
                    tr_model, tr_ckpt_dir, tr_out_dir, tr_manifest, tr_prec,
                    tr_mode, tr_gc,
                    tr_seed, tr_batch, tr_steps, tr_lr, tr_wd,
                    tr_warmup, tr_sched, tr_accum, tr_gnorm,
                    tr_shift, tr_cfg_drop, tr_vid_w, tr_aud_w,
                    tr_save_ev, tr_log_ev,
                ]

                tr_preview_btn.click(
                    fn=_preview_config,
                    inputs=_cfg_inputs,
                    outputs=[tr_yaml_box],
                )
                tr_start_btn.click(
                    fn=_run_training,
                    inputs=_cfg_inputs + [tr_gpus],
                    outputs=[tr_log],
                )
                tr_stop_btn.click(fn=_stop_any_process, outputs=[tr_log])

            # ══════════════════════════════════════════════════════════════════
            # TAB 3 – Inference
            # ══════════════════════════════════════════════════════════════════
            with gr.TabItem("🎬 3 · Inference"):
                gr.Markdown(
                    """### Inference
Load the model (base or fine-tuned), configure generation parameters, then generate a video.

**Tips:**
- Changing any **Model Settings** requires re-clicking **🔃 Load / Reload Model**.
- Leave *Fine-tuned Checkpoint* blank to use the unmodified base weights.
- For `i2v` / `t2i2v` modes, upload a first-frame image.
- Wrap spoken text in `<S>...<E>` and optional audio captions in `<AUDCAP>...<ENDAUDCAP>`.
"""
                )

                with gr.Row():
                    # ── left: settings + prompt ────────────────────────────────
                    with gr.Column(scale=2):
                        with gr.Accordion("🔧 Model Settings", open=True):
                            inf_ckpt_dir    = gr.Textbox(value="./ckpts", label="Checkpoint Directory")
                            inf_model       = gr.Dropdown(
                                choices=MODEL_NAMES, value="960x960_5s", label="Model Name",
                            )
                            inf_ft_ckpt     = gr.Textbox(
                                value="",
                                placeholder="./training_outputs/checkpoint_final.pt  "
                                            "(leave blank to use the base model)",
                                label="Fine-tuned Checkpoint  (optional)",
                            )
                            inf_mode        = gr.Dropdown(
                                choices=["t2v", "i2v", "t2i2v"], value="t2v",
                                label="Mode",
                                info=(
                                    "t2v = text-to-video  |  "
                                    "i2v = image+text-to-video  |  "
                                    "t2i2v = text→image→video"
                                ),
                            )
                            with gr.Row():
                                inf_cpu_offload = gr.Checkbox(value=False, label="CPU Offload")
                                inf_fp8         = gr.Checkbox(value=False, label="FP8")
                                inf_qint8       = gr.Checkbox(value=False, label="QInt8")

                            load_model_btn  = gr.Button("🔃 Load / Reload Model", variant="primary")
                            load_status     = gr.Textbox(
                                value="⚠️  Model not loaded.", interactive=False,
                                label="Model Status",
                            )

                        with gr.Accordion("🎬 Generation Settings", open=True):
                            inf_prompt = gr.Textbox(
                                label="Text Prompt",
                                placeholder=(
                                    'A woman delivers a speech on stage. '
                                    '<S>Hello everyone, thank you for coming!<E> '
                                    '<AUDCAP>applause, crowd noise<ENDAUDCAP>'
                                ),
                                lines=3,
                            )
                            inf_image = gr.Image(
                                type="filepath",
                                label="First Frame Image  (required for i2v / t2i2v)",
                            )

                            with gr.Row():
                                inf_h = gr.Number(
                                    value=704, precision=0, minimum=128, maximum=1280,
                                    label="Height",
                                    info="Recommended: 704 / 960 / 1280",
                                )
                                inf_w = gr.Number(
                                    value=1280, precision=0, minimum=128, maximum=1280,
                                    label="Width",
                                    info="Recommended: 1280 / 960 / 704",
                                )

                            with gr.Row():
                                inf_seed   = gr.Number(value=100, precision=0, label="Seed")
                                inf_solver = gr.Dropdown(
                                    choices=["unipc", "euler", "dpm++"],
                                    value="unipc", label="Solver",
                                )
                                inf_steps  = gr.Number(
                                    value=50, precision=0, minimum=10, maximum=100,
                                    label="Sample Steps",
                                )

                            with gr.Row():
                                inf_shift = gr.Slider(
                                    0.0, 20.0, value=5.0, step=1.0, label="Shift",
                                )
                                inf_vgs   = gr.Slider(
                                    0.0, 10.0, value=4.0, step=0.5,
                                    label="Video Guidance Scale",
                                )
                                inf_ags   = gr.Slider(
                                    0.0, 10.0, value=3.0, step=0.5,
                                    label="Audio Guidance Scale",
                                )

                            inf_slg   = gr.Number(
                                value=11, precision=0, minimum=-1, maximum=30,
                                label="SLG Layer",
                                info="Skip-layer guidance layer index (-1 to disable).",
                            )
                            inf_vneg  = gr.Textbox(
                                value="jitter, bad hands, blur, distortion",
                                label="Video Negative Prompt",
                            )
                            inf_aneg  = gr.Textbox(
                                value="robotic, muffled, echo, distorted",
                                label="Audio Negative Prompt",
                            )

                            inf_gen_btn = gr.Button("🚀 Generate Video", variant="primary")

                    # ── right: output ──────────────────────────────────────────
                    with gr.Column(scale=1):
                        inf_video_out  = gr.Video(label="Generated Video")
                        inf_gen_status = gr.Textbox(
                            label="Status", interactive=False, lines=2,
                        )

                # ── wiring ────────────────────────────────────────────────────
                load_model_btn.click(
                    fn=_load_model,
                    inputs=[
                        inf_ckpt_dir, inf_model, inf_ft_ckpt,
                        inf_mode, inf_cpu_offload, inf_fp8, inf_qint8,
                    ],
                    outputs=[load_status],
                )
                inf_gen_btn.click(
                    fn=_run_inference,
                    inputs=[
                        inf_prompt, inf_image,
                        inf_h, inf_w,
                        inf_seed, inf_solver, inf_steps, inf_shift,
                        inf_vgs, inf_ags, inf_slg,
                        inf_vneg, inf_aneg,
                    ],
                    outputs=[inf_video_out, inf_gen_status],
                )

    return demo


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    args = _build_arg_parser().parse_args()
    ui = build_ui()
    ui.launch(
        server_name=args.server_name,
        server_port=args.server_port,
        share=args.share,
        inbrowser=True,
        theme=gr.themes.Soft(),
    )
