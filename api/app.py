#!/usr/bin/env python3
"""
Flask API for Ovi Video Generation Studio
=========================================
Endpoints:
  POST /api/prepare     – Encode raw videos into latents (prepare_training_data_v2.py)
  POST /api/train       – Fine-tune model (train_v2.py)
  POST /api/inference   – Generate videos (inference.py)
  POST /api/upload      – Upload an image for I2V mode
  GET  /api/status/<id> – Poll task status + streamed logs
  GET  /api/outputs     – List generated .mp4 files
  GET  /api/video/<..>  – Serve a video file
"""

import os
import sys
import uuid
import json
import threading
import subprocess
import tempfile
from pathlib import Path

import yaml
from flask import Flask, request, jsonify, send_file, abort
from flask_cors import CORS
from werkzeug.utils import secure_filename

# ── Project root (one level above api/) ─────────────────────────────────────
ROOT = Path(__file__).parent.parent.resolve()

app = Flask(__name__, static_folder="static")
CORS(app)

# ── Upload folder ────────────────────────────────────────────────────────────
UPLOAD_FOLDER = Path(__file__).parent / "uploads"
UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)

CONFIG_FOLDER = Path(__file__).parent / "configs"
CONFIG_FOLDER.mkdir(parents=True, exist_ok=True)

# ── In-memory task store ─────────────────────────────────────────────────────
tasks: dict = {}
tasks_lock = threading.Lock()


def _new_task() -> str:
    task_id = uuid.uuid4().hex[:10]
    with tasks_lock:
        tasks[task_id] = {"status": "running", "logs": [], "returncode": None}
    return task_id


def _run_subprocess(task_id: str, cmd: list, cwd: str = None):
    """Run *cmd* in a background thread; stream its stdout/stderr to the task log."""

    def _worker():
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            cwd=cwd or str(ROOT),
            env=env,
            text=True,
            bufsize=1,
        )
        with tasks_lock:
            tasks[task_id]["pid"] = proc.pid

        for line in proc.stdout:
            stripped = line.rstrip()
            if stripped:
                with tasks_lock:
                    tasks[task_id]["logs"].append(stripped)

        proc.wait()
        with tasks_lock:
            tasks[task_id]["returncode"] = proc.returncode
            tasks[task_id]["status"] = "done" if proc.returncode == 0 else "error"

    t = threading.Thread(target=_worker, daemon=True)
    t.start()


# ═══════════════════════════════════════════════════════════════════════════════
# Static / root
# ═══════════════════════════════════════════════════════════════════════════════

@app.route("/")
def index():
    return app.send_static_file("index.html")


# ═══════════════════════════════════════════════════════════════════════════════
# File upload (images for I2V)
# ═══════════════════════════════════════════════════════════════════════════════

@app.route("/api/upload", methods=["POST"])
def upload_file():
    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400
    f = request.files["file"]
    if f.filename == "":
        return jsonify({"error": "Empty filename"}), 400
    filename = secure_filename(f.filename)
    save_path = UPLOAD_FOLDER / filename
    f.save(str(save_path))
    return jsonify({"path": str(save_path), "name": filename})


# ═══════════════════════════════════════════════════════════════════════════════
# Prepare training data
# ═══════════════════════════════════════════════════════════════════════════════

@app.route("/api/prepare", methods=["POST"])
def prepare():
    data = request.get_json(force=True) or {}

    input_manifest = data.get("input_manifest", "").strip()
    if not input_manifest:
        return jsonify({"error": "input_manifest is required"}), 400

    output_dir  = data.get("output_dir",  "training_data").strip()
    ckpt_dir    = data.get("ckpt_dir",    "./ckpts").strip()
    model_name  = data.get("model_name",  "960x960_5s")
    device      = int(data.get("device", 0))

    cmd = [
        sys.executable, "prepare_training_data_v2.py",
        "--input_manifest", input_manifest,
        "--output_dir",     output_dir,
        "--ckpt_dir",       ckpt_dir,
        "--model_name",     model_name,
        "--device",         str(device),
    ]

    task_id = _new_task()
    _run_subprocess(task_id, cmd)
    return jsonify({"task_id": task_id})


# ═══════════════════════════════════════════════════════════════════════════════
# Training
# ═══════════════════════════════════════════════════════════════════════════════

@app.route("/api/train", methods=["POST"])
def train():
    data = request.get_json(force=True) or {}

    cfg = {
        "model_name":      data.get("model_name",     "960x960_5s"),
        "ckpt_dir":        data.get("ckpt_dir",       "./ckpts"),
        "data_manifest":   data.get("data_manifest",  "training_data/manifest.jsonl"),
        "output_dir":      data.get("output_dir",     "./training_outputs"),
        "mixed_precision": data.get("mixed_precision","bf16"),
        "finetune": {
            "mode":                   data.get("finetune_mode",           "fusion_only"),
            "gradient_checkpointing": bool(data.get("gradient_checkpointing", True)),
        },
        "training": {
            "seed":                         int(data.get("seed",              42)),
            "batch_size":                   int(data.get("batch_size",         1)),
            "max_steps":                    int(data.get("max_steps",       1000)),
            "learning_rate":              float(data.get("learning_rate",    1e-5)),
            "weight_decay":               float(data.get("weight_decay",    0.01)),
            "warmup_steps":                 int(data.get("warmup_steps",      100)),
            "lr_scheduler":                     data.get("lr_scheduler",   "cosine"),
            "gradient_accumulation_steps":  int(data.get("gradient_accumulation_steps", 1)),
            "max_grad_norm":              float(data.get("max_grad_norm",    1.0)),
            "save_every":                   int(data.get("save_every",       500)),
            "log_every":                    int(data.get("log_every",         10)),
        },
        "data": {
            "num_workers": int(data.get("num_workers", 4)),
            "pin_memory":  True,
        },
        "flow": {
            "shift":         float(data.get("shift",         5.0)),
            "cfg_drop_rate": float(data.get("cfg_drop_rate", 0.1)),
        },
        "loss": {
            "video_weight": float(data.get("video_weight", 1.0)),
            "audio_weight": float(data.get("audio_weight", 1.0)),
        },
    }

    cfg_path = CONFIG_FOLDER / f"train_{uuid.uuid4().hex[:8]}.yaml"
    with open(cfg_path, "w") as fh:
        yaml.dump(cfg, fh, default_flow_style=False)

    cmd = [sys.executable, "train_v2.py", "--config", str(cfg_path)]
    task_id = _new_task()
    _run_subprocess(task_id, cmd)
    return jsonify({"task_id": task_id})


# ═══════════════════════════════════════════════════════════════════════════════
# Inference
# ═══════════════════════════════════════════════════════════════════════════════

@app.route("/api/inference", methods=["POST"])
def inference():
    data = request.get_json(force=True) or {}

    output_dir = data.get("output_dir", "./outputs").strip()
    abs_output = (ROOT / output_dir).resolve()
    abs_output.mkdir(parents=True, exist_ok=True)

    res = data.get("video_frame_height_width", [704, 1280])
    if isinstance(res, str):
        try:
            res = json.loads(res)
        except Exception:
            res = [704, 1280]

    cfg = {
        "ckpt_dir":                 data.get("ckpt_dir",              "./ckpts"),
        "output_dir":               output_dir,
        "sample_steps":             int(data.get("sample_steps",           50)),
        "solver_name":              data.get("solver_name",           "unipc"),
        "model_name":               data.get("model_name",         "720x720_5s"),
        "shift":                  float(data.get("shift",                  5.0)),
        "sp_size":                  1,
        "audio_guidance_scale":   float(data.get("audio_guidance_scale",   3.0)),
        "video_guidance_scale":   float(data.get("video_guidance_scale",   4.0)),
        "mode":                     data.get("mode",                    "t2v"),
        "fp8":                     bool(data.get("fp8",                 False)),
        "cpu_offload":             bool(data.get("cpu_offload",         False)),
        "seed":                     int(data.get("seed",                  103)),
        "video_negative_prompt":    data.get("video_negative_prompt",
                                             "jitter, bad hands, blur, distortion"),
        "audio_negative_prompt":    data.get("audio_negative_prompt",
                                             "robotic, muffled, echo, distorted"),
        "video_frame_height_width": res,
        "text_prompt":              data.get("text_prompt",               ""),
        "slg_layer":                int(data.get("slg_layer",             11)),
        "each_example_n_times":     int(data.get("each_example_n_times",   1)),
    }

    ft_ckpt = data.get("finetuned_checkpoint", "").strip()
    if ft_ckpt:
        cfg["finetuned_checkpoint"] = ft_ckpt

    img_path = data.get("image_path", "").strip()
    if img_path:
        cfg["image_path"] = img_path

    cfg_path = CONFIG_FOLDER / f"infer_{uuid.uuid4().hex[:8]}.yaml"
    with open(cfg_path, "w") as fh:
        yaml.dump(cfg, fh, default_flow_style=False)

    cmd = [sys.executable, "inference.py", "--config_file", str(cfg_path)]
    task_id = _new_task()
    _run_subprocess(task_id, cmd)
    return jsonify({"task_id": task_id})


# ═══════════════════════════════════════════════════════════════════════════════
# Task status (polling)
# ═══════════════════════════════════════════════════════════════════════════════

@app.route("/api/status/<task_id>")
def task_status(task_id):
    with tasks_lock:
        task = tasks.get(task_id)
    if task is None:
        return jsonify({"error": "Task not found"}), 404
    return jsonify({
        "status":     task["status"],
        "logs":       task["logs"],
        "returncode": task.get("returncode"),
    })


# ═══════════════════════════════════════════════════════════════════════════════
# Output gallery
# ═══════════════════════════════════════════════════════════════════════════════

@app.route("/api/outputs")
def list_outputs():
    rel_dir = request.args.get("dir", "outputs").lstrip("./")
    full_path = ROOT / rel_dir
    if not full_path.exists():
        return jsonify([])

    videos = []
    for f in sorted(full_path.glob("**/*.mp4")):
        try:
            st = f.stat()
            videos.append({
                "name":  f.name,
                "path":  str(f.relative_to(ROOT)),
                "size":  st.st_size,
                "mtime": st.st_mtime,
            })
        except Exception:
            pass
    return jsonify(videos)


@app.route("/api/video/<path:filepath>")
def serve_video(filepath):
    full_path = ROOT / filepath
    if not full_path.exists() or not full_path.is_file():
        abort(404)
    return send_file(str(full_path), mimetype="video/mp4")


# ═══════════════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    print(f"  Ovi Studio → http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=args.debug, threaded=True)
