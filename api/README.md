# Ovi Studio — Flask API

A modern web interface for the Ovi twin-branch video generation pipeline.

---

## Structure

```
api/
├── app.py            ← Flask backend (all endpoints)
├── README.md
├── configs/          ← Auto-generated YAML configs per job
├── uploads/          ← Uploaded reference images (I2V)
└── static/
    ├── index.html    ← Single-page application
    ├── style.css     ← Design system (dark glassmorphism)
    └── app.js        ← Frontend logic
```

---

## Install dependencies

```bash
pip install flask flask-cors pyyaml
```

---

## Start the server

```bash
# From the project root:
python api/app.py

# Custom port:
python api/app.py --port 8080

# Debug mode:
python api/app.py --debug
```

Then open **http://localhost:5000** in your browser.

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/prepare` | Run `prepare_training_data_v2.py` |
| `POST` | `/api/train` | Run `train_v2.py` |
| `POST` | `/api/inference` | Run `inference.py` |
| `POST` | `/api/upload` | Upload a reference image |
| `GET`  | `/api/status/<task_id>` | Poll task status + logs |
| `GET`  | `/api/outputs?dir=<path>` | List generated videos |
| `GET`  | `/api/video/<path>` | Serve a video file |

---

## Workflow

1. **Prepare Data** tab → point at your raw `.jsonl` manifest, click *Start Preparation*. Latent `.pt` files are written to the output directory.
2. **Train** tab → configure hyperparameters, point at the manifest produced in step 1, click *Start Training*.
3. **Inference** tab → enter a prompt, choose T2V / I2V / T2I2V mode, optionally point at your fine-tuned checkpoint, click *Generate Video*.
4. **Gallery** tab → preview all generated videos.

---

## Notes

- All jobs run as background subprocesses; logs are streamed in real-time via polling.
- The backend writes temporary YAML configs to `api/configs/` for each job.
- For multi-GPU training use `torchrun` manually (not via the UI).
