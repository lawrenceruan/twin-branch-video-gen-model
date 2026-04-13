/* ════════════════════════════════════════════════════════════
   Ovi Studio — Frontend Application Logic
   ════════════════════════════════════════════════════════════ */

const API = '';          // same origin
const POLL_MS = 1500;    // log polling interval

// ── Active polling handles ────────────────────────────────
const pollers = {};   // { tabName: intervalId }

// ══════════════════════════════════════════════════════════
// Tab navigation
// ══════════════════════════════════════════════════════════

function switchTab(name) {
  document.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));
  document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));

  const panel = document.getElementById(`tab-${name}`);
  const btn   = document.querySelector(`.tab-btn[data-tab="${name}"]`);
  if (panel) panel.classList.add('active');
  if (btn)   btn.classList.add('active');

  if (name === 'gallery') loadGallery();
}

// Wire up tab buttons
document.querySelectorAll('.tab-btn').forEach(btn => {
  btn.addEventListener('click', () => switchTab(btn.dataset.tab));
});


// ══════════════════════════════════════════════════════════
// Toast notifications
// ══════════════════════════════════════════════════════════

function showToast(message, type = 'info', duration = 4000) {
  const container = document.getElementById('toast-container');
  const toast = document.createElement('div');
  toast.className = `toast ${type}`;
  toast.textContent = message;
  container.appendChild(toast);
  setTimeout(() => {
    toast.style.transition = 'opacity .35s, transform .35s';
    toast.style.opacity = '0';
    toast.style.transform = 'translateX(24px)';
    setTimeout(() => toast.remove(), 400);
  }, duration);
}


// ══════════════════════════════════════════════════════════
// Log display + polling
// ══════════════════════════════════════════════════════════

function clearLog(tabName) {
  const logEl = document.getElementById(`${tabName}-log`);
  if (logEl) logEl.textContent = '';
}

function appendLog(tabName, lines) {
  const logEl = document.getElementById(`${tabName}-log`);
  if (!logEl) return;
  const wasAtBottom = logEl.scrollHeight - logEl.scrollTop <= logEl.clientHeight + 40;
  lines.forEach(line => {
    const span = document.createElement('span');
    const l = line.toLowerCase();
    if (l.includes('error') || l.includes('traceback') || l.includes('exception')) {
      span.className = 'log-error';
    } else if (l.includes('warning') || l.includes('warn')) {
      span.className = 'log-warn';
    }
    span.textContent = line + '\n';
    logEl.appendChild(span);
  });
  if (wasAtBottom) logEl.scrollTop = logEl.scrollHeight;
}

function setStatusBadge(tabName, status) {
  const badge = document.getElementById(`${tabName}-status-badge`);
  if (!badge) return;
  badge.className = `status-badge ${status}`;
  badge.textContent = status === 'running' ? 'Running' : status === 'done' ? 'Done ✓' : 'Error ✕';
}

function startPolling(tabName, taskId) {
  // Stop any existing poller for this tab
  if (pollers[tabName]) clearInterval(pollers[tabName]);

  let lastLogCount = 0;

  const section = document.getElementById(`${tabName}-log-section`);
  if (section) section.classList.remove('hidden');
  setStatusBadge(tabName, 'running');

  pollers[tabName] = setInterval(async () => {
    try {
      const res  = await fetch(`${API}/api/status/${taskId}`);
      const data = await res.json();

      const newLines = data.logs.slice(lastLogCount);
      if (newLines.length) {
        appendLog(tabName, newLines);
        lastLogCount = data.logs.length;
      }

      if (data.status !== 'running') {
        clearInterval(pollers[tabName]);
        delete pollers[tabName];
        setStatusBadge(tabName, data.status);

        if (data.status === 'done') {
          showToast(`✓ ${capitalize(tabName)} completed successfully!`, 'success');
          if (tabName === 'inference') loadGallery();
        } else {
          showToast(`✕ ${capitalize(tabName)} failed — check logs for details.`, 'error', 6000);
        }
      }
    } catch (err) {
      console.error('Polling error:', err);
    }
  }, POLL_MS);
}

function capitalize(s) { return s.charAt(0).toUpperCase() + s.slice(1); }


// ══════════════════════════════════════════════════════════
// Helpers
// ══════════════════════════════════════════════════════════

function val(id, fallback = '') {
  const el = document.getElementById(id);
  if (!el) return fallback;
  if (el.type === 'checkbox') return el.checked;
  return el.value.trim() || fallback;
}

function numVal(id, def) {
  const v = parseFloat(val(id));
  return isNaN(v) ? def : v;
}

async function postJSON(url, body) {
  const res = await fetch(url, {
    method:  'POST',
    headers: { 'Content-Type': 'application/json' },
    body:    JSON.stringify(body),
  });
  return res.json();
}


// ══════════════════════════════════════════════════════════
// PREPARE
// ══════════════════════════════════════════════════════════

async function runPrepare() {
  const manifest = val('prep-manifest');
  if (!manifest) { showToast('Input manifest path is required.', 'error'); return; }

  clearLog('prepare');
  showToast('Starting data preparation…', 'info');

  const body = {
    input_manifest: manifest,
    output_dir:     val('prep-output', 'training_data'),
    ckpt_dir:       val('prep-ckpt',   './ckpts'),
    model_name:     val('prep-model',  '960x960_5s'),
    device:         numVal('prep-device', 0),
  };

  try {
    const data = await postJSON(`${API}/api/prepare`, body);
    if (data.error) { showToast(data.error, 'error'); return; }
    startPolling('prepare', data.task_id);
  } catch (err) {
    showToast('Request failed: ' + err.message, 'error');
  }
}


// ══════════════════════════════════════════════════════════
// TRAIN
// ══════════════════════════════════════════════════════════

async function runTrain() {
  clearLog('train');
  showToast('Starting training job…', 'info');

  const body = {
    data_manifest:   val('train-manifest', 'training_data/manifest.jsonl'),
    ckpt_dir:        val('train-ckpt',     './ckpts'),
    output_dir:      val('train-output',   './training_outputs'),
    model_name:      val('train-model',    '960x960_5s'),
    max_steps:       numVal('train-steps',   1000),
    batch_size:      numVal('train-batch',      1),
    learning_rate:   parseFloat(val('train-lr', '1e-5')),
    warmup_steps:    numVal('train-warmup',   100),
    lr_scheduler:    val('train-scheduler', 'cosine'),
    gradient_accumulation_steps: numVal('train-accum', 1),
    finetune_mode:   val('train-mode',      'fusion_only'),
    mixed_precision: val('train-precision', 'bf16'),
    save_every:      numVal('train-save',   500),
    log_every:       numVal('train-log',     10),
    gradient_checkpointing: val('train-gc', true),
    shift:           numVal('train-shift',  5.0),
  };

  try {
    const data = await postJSON(`${API}/api/train`, body);
    if (data.error) { showToast(data.error, 'error'); return; }
    startPolling('train', data.task_id);
  } catch (err) {
    showToast('Request failed: ' + err.message, 'error');
  }
}


// ══════════════════════════════════════════════════════════
// INFERENCE
// ══════════════════════════════════════════════════════════

let _uploadedImagePath = '';

function setMode(btn) {
  document.querySelectorAll('#infer-mode-group .seg-btn').forEach(b => b.classList.remove('active'));
  btn.classList.add('active');
  const mode = btn.dataset.value;
  const uploadField = document.getElementById('image-upload-field');
  if (uploadField) {
    uploadField.style.display = (mode === 'i2v') ? '' : 'none';
  }
}

// Initialise: hide upload field on load
document.addEventListener('DOMContentLoaded', () => {
  const uploadField = document.getElementById('image-upload-field');
  if (uploadField) uploadField.style.display = 'none';
  loadGallery();
});

function getSelectedMode() {
  const active = document.querySelector('#infer-mode-group .seg-btn.active');
  return active ? active.dataset.value : 't2v';
}

async function runInference() {
  const prompt = val('infer-prompt');
  if (!prompt) { showToast('Text prompt is required.', 'error'); return; }

  const mode = getSelectedMode();
  if (mode === 'i2v' && !_uploadedImagePath) {
    showToast('Please upload a reference image for I2V mode.', 'error');
    return;
  }

  clearLog('inference');
  showToast('Submitting inference job…', 'info');

  const resStr = val('infer-res', '704,1280');
  const resParts = resStr.split(',').map(Number);

  const body = {
    text_prompt:              prompt,
    mode:                     mode,
    model_name:               val('infer-model',    '720x720_5s'),
    video_frame_height_width: resParts,
    sample_steps:             numVal('infer-steps', 50),
    video_guidance_scale:     numVal('infer-vid-cfg', 4),
    audio_guidance_scale:     numVal('infer-aud-cfg', 3),
    seed:                     numVal('infer-seed', 103),
    ckpt_dir:                 val('infer-ckpt',     './ckpts'),
    finetuned_checkpoint:     val('infer-ft-ckpt',  ''),
    output_dir:               val('infer-output',   './outputs'),
    video_negative_prompt:    val('infer-neg-video', 'jitter, bad hands, blur, distortion'),
    audio_negative_prompt:    val('infer-neg-audio', 'robotic, muffled, echo, distorted'),
    cpu_offload:              val('infer-cpu-offload', false),
  };

  if (mode === 'i2v' && _uploadedImagePath) {
    body.image_path = _uploadedImagePath;
  }

  try {
    const data = await postJSON(`${API}/api/inference`, body);
    if (data.error) { showToast(data.error, 'error'); return; }
    startPolling('inference', data.task_id);
  } catch (err) {
    showToast('Request failed: ' + err.message, 'error');
  }
}


// ══════════════════════════════════════════════════════════
// Image upload
// ══════════════════════════════════════════════════════════

function handleFileSelect(event) {
  const file = event.target.files[0];
  if (file) uploadImage(file);
}

function handleDrop(event) {
  event.preventDefault();
  const zone = document.getElementById('upload-zone');
  zone.classList.remove('dragover');
  const file = event.dataTransfer.files[0];
  if (file && file.type.startsWith('image/')) uploadImage(file);
}

document.getElementById('upload-zone')?.addEventListener('dragover', e => {
  e.preventDefault();
  document.getElementById('upload-zone').classList.add('dragover');
});
document.getElementById('upload-zone')?.addEventListener('dragleave', () => {
  document.getElementById('upload-zone').classList.remove('dragover');
});

async function uploadImage(file) {
  const formData = new FormData();
  formData.append('file', file);

  try {
    showToast('Uploading image…', 'info', 2000);
    const res = await fetch(`${API}/api/upload`, { method: 'POST', body: formData });
    const data = await res.json();

    if (data.error) { showToast(data.error, 'error'); return; }

    _uploadedImagePath = data.path;

    // Preview
    const preview = document.getElementById('upload-preview');
    preview.innerHTML = `<img src="${URL.createObjectURL(file)}" alt="preview" />`;
    preview.classList.remove('hidden');

    const pathEl = document.getElementById('uploaded-path');
    pathEl.textContent = '✓ ' + data.path;
    pathEl.classList.remove('hidden');

    showToast('Image uploaded ✓', 'success', 2500);
  } catch (err) {
    showToast('Upload failed: ' + err.message, 'error');
  }
}


// ══════════════════════════════════════════════════════════
// Gallery
// ══════════════════════════════════════════════════════════

async function loadGallery() {
  const dir = val('gallery-dir', 'outputs');
  const grid  = document.getElementById('gallery-grid');
  const empty = document.getElementById('gallery-empty');
  grid.innerHTML  = '';
  empty.classList.add('hidden');

  try {
    const res   = await fetch(`${API}/api/outputs?dir=${encodeURIComponent(dir)}`);
    const videos = await res.json();

    if (!videos.length) {
      empty.classList.remove('hidden');
      return;
    }

    videos.forEach(v => {
      const card = createGalleryCard(v);
      grid.appendChild(card);
    });
  } catch (err) {
    showToast('Failed to load gallery: ' + err.message, 'error');
  }
}

function createGalleryCard(video) {
  const card = document.createElement('div');
  card.className = 'gallery-card';
  card.onclick = () => openLightbox(video);

  const sizeMB = (video.size / 1024 / 1024).toFixed(1);
  const date   = new Date(video.mtime * 1000).toLocaleString();

  card.innerHTML = `
    <div class="gallery-thumb">
      <video src="${API}/api/video/${video.path}" muted preload="metadata"></video>
      <div class="gallery-play-overlay">
        <div class="gallery-play-btn">
          <svg viewBox="0 0 20 20" fill="currentColor"><path fill-rule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM9.555 7.168A1 1 0 008 8v4a1 1 0 001.555.832l3-2a1 1 0 000-1.664l-3-2z" clip-rule="evenodd"/></svg>
        </div>
      </div>
    </div>
    <div class="gallery-info">
      <div class="gallery-name" title="${video.name}">${video.name}</div>
      <div class="gallery-meta">${sizeMB} MB · ${date}</div>
    </div>
  `;

  // Seek video to 0.5s for a thumbnail frame
  const vid = card.querySelector('video');
  vid.addEventListener('loadedmetadata', () => { vid.currentTime = 0.5; });

  return card;
}

function openLightbox(video) {
  const lb     = document.getElementById('lightbox');
  const lbVid  = document.getElementById('lightbox-video');
  const lbTitle = document.getElementById('lightbox-title');

  lbVid.src = `${API}/api/video/${video.path}`;
  lbTitle.textContent = video.path;
  lb.classList.remove('hidden');
  document.body.style.overflow = 'hidden';
}

function closeLightbox(event) {
  if (event && event.target !== document.getElementById('lightbox') && !event.target.closest('.lightbox-close')) return;
  const lb    = document.getElementById('lightbox');
  const lbVid = document.getElementById('lightbox-video');
  lb.classList.add('hidden');
  lbVid.pause();
  lbVid.src = '';
  document.body.style.overflow = '';
}

// Close lightbox on Escape
document.addEventListener('keydown', e => {
  if (e.key === 'Escape') closeLightbox({ target: document.getElementById('lightbox') });
});
