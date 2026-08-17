/* llama-autoloader WebUI */
const $ = (id) => document.getElementById(id);
let modelsCache = [];
let statusCache = null;
let editingId = null;
let stateManagingId = null;

// Track in-flight async actions per model to prevent WS re-renders from blowing away button state
const inFlightActions = new Map();

function esc(str) {
  const div = document.createElement('div');
  div.textContent = str || '';
  return div.innerHTML;
}

function showToast(msg, type = 'info') {
  const container = $('toast-container');
  const toast = document.createElement('div');
  toast.className = `toast ${type}`;
  toast.textContent = msg;
  container.appendChild(toast);
  setTimeout(() => {
    toast.style.opacity = '0';
    toast.style.transform = 'translateX(100%)';
    toast.style.transition = 'all 0.3s ease';
    setTimeout(() => toast.remove(), 300);
  }, 3500);
}

async function api(path, opts={}) {
  const r = await fetch(path, opts);
  if (!r.ok) {
    const t = await r.text();
    throw new Error(`${r.status}: ${t}`);
  }
  return r;
}

async function apiJSON(path, opts={}) {
  const r = await api(path, opts);
  return await r.json();
}

function fmtBytes(mb) {
  if (!mb) return '—';
  if (mb < 1024) return mb.toFixed(0) + ' MB';
  return (mb/1024).toFixed(2) + ' GB';
}

function statusBadge(loaded, ready) {
  if (loaded && ready) return '<span class="status-badge status-loaded">Ready</span>';
  if (loaded) return '<span class="status-badge status-loading">Loading...</span>';
  return '<span class="status-badge status-unloaded">○ Offline</span>';
}

function renderGPUs(gpus) {
  const grid = $('gpu-grid');
  if (!gpus || gpus.length === 0) {
    grid.innerHTML = '<div class="empty">No NVIDIA GPUs detected (nvidia-smi missing). CPU-only mode.</div>';
    return;
  }
  grid.innerHTML = gpus.map(g => {
    const pct = g.total_mb ? (g.used_mb / g.total_mb * 100) : 0;
    return `
      <div class="gpu-card">
        <div class="name"><span>${esc(g.name)}</span><span class="idx">GPU ${g.index}</span></div>
        <div class="bar"><div style="width:${pct.toFixed(1)}%"></div></div>
        <div class="gpu-stats">
          <span>${fmtBytes(g.used_mb)} / ${fmtBytes(g.total_mb)}</span>
          <span>util ${g.utilization_pct.toFixed(0)}%</span>
        </div>
      </div>`;
  }).join('');
}

function renderRAM(ram) {
  const pct = ram.pct;
  $('ram-bar').style.width = pct.toFixed(1) + '%';
  $('ram-used').textContent = fmtBytes(ram.used_mb) + ' used';
  $('ram-total').textContent = fmtBytes(ram.total_mb) + ' total';
}

function buildButtonRowHTML(m) {
  const act = inFlightActions.get(m.id);
  const isLoadInFlight = act === 'load';
  const isUnloadInFlight = act === 'unload';
  const isVisionInFlight = act === 'toggle-vision';

  const visionBtn = m.has_mmproj ? (
    m.use_mmproj
      ? `<button class="primary" data-act="toggle-vision" ${isVisionInFlight ? 'disabled' : ''}>${isVisionInFlight ? 'Updating...' : '👁 Vision: ON'}</button>`
      : `<button data-act="toggle-vision" ${isVisionInFlight ? 'disabled' : ''}>${isVisionInFlight ? 'Updating...' : '👁 Vision: OFF'}</button>`
  ) : '';

  let loadBtn = '';
  if (m.loaded) {
    loadBtn = `<button class="danger" data-act="unload" ${isUnloadInFlight ? 'disabled' : ''}>${isUnloadInFlight ? 'Unloading...' : 'Unload'}</button>`;
  } else {
    loadBtn = `<button class="primary" data-act="load" ${isLoadInFlight ? 'disabled' : ''}>${isLoadInFlight ? 'Loading...' : 'Load'}</button>`;
  }

  return `
    ${loadBtn}
    <button data-act="edit" ${act ? 'disabled' : ''}>Edit</button>
    ${visionBtn}
    <button data-act="copy-curl">cURL</button>
    <button data-act="states">States</button>
  `;
}

function updateButtonRow(card, m) {
  // Update buttons in-place to avoid flickering from innerHTML replacement.
  // Only called for existing cards during incremental updates.
  const btnRow = card.querySelector('.btn-row');
  if (!btnRow) return;

  const act = inFlightActions.get(m.id);
  const isLoadInFlight = act === 'load';
  const isUnloadInFlight = act === 'unload';
  const isVisionInFlight = act === 'toggle-vision';

  // Update or toggle load/unload button based on model state
  let loadBtn = btnRow.querySelector('button[data-act="load"], button[data-act="unload"]');
  if (!loadBtn) {
    // Button doesn't exist yet (e.g., loaded state changed), rebuild row once
    btnRow.innerHTML = buildButtonRowHTML(m);
    return;
  }

  if (m.loaded && loadBtn.dataset.act !== 'unload') {
    // Model is now loaded but we have a "Load" button — swap to Unload
    loadBtn.dataset.act = 'unload';
    loadBtn.className = 'danger';
    loadBtn.textContent = isUnloadInFlight ? 'Unloading...' : 'Unload';
    loadBtn.disabled = isUnloadInFlight;
  } else if (!m.loaded && loadBtn.dataset.act !== 'load') {
    // Model is now unloaded but we have an "Unload" button — swap to Load
    loadBtn.dataset.act = 'load';
    loadBtn.className = 'primary';
    loadBtn.textContent = isLoadInFlight ? 'Loading...' : 'Load';
    loadBtn.disabled = isLoadInFlight;
  } else {
    // Same action type, just update text/disabled state for in-flight
    if (m.loaded) {
      loadBtn.textContent = isUnloadInFlight ? 'Unloading...' : 'Unload';
      loadBtn.disabled = isUnloadInFlight;
    } else {
      loadBtn.textContent = isLoadInFlight ? 'Loading...' : 'Load';
      loadBtn.disabled = isLoadInFlight;
    }
  }

  // Update vision button if present
  const visionBtn = btnRow.querySelector('button[data-act="toggle-vision"]');
  if (visionBtn && m.has_mmproj) {
    if (m.use_mmproj) {
      visionBtn.className = 'primary';
      visionBtn.textContent = isVisionInFlight ? 'Updating...' : '👁 Vision: ON';
    } else {
      visionBtn.className = '';
      visionBtn.textContent = isVisionInFlight ? 'Updating...' : '👁 Vision: OFF';
    }
    visionBtn.disabled = isVisionInFlight;
  }

  // Update Edit button disabled state during any in-flight action
  const editBtn = btnRow.querySelector('button[data-act="edit"]');
  if (editBtn) {
    editBtn.disabled = !!act;
  }
}

function renderModels() {
  const filter = $('filter').value.toLowerCase();
  const statusFilter = $('status-filter').value;
  const grid = $('models-grid');
  let list = modelsCache;

  if (filter) {
    list = list.filter(m =>
      m.id.toLowerCase().includes(filter) ||
      (m.name||'').toLowerCase().includes(filter) ||
      (m.tags||[]).some(t => t.toLowerCase().includes(filter))
    );
  }

  if (statusFilter === 'loaded') {
    list = list.filter(m => m.loaded);
  } else if (statusFilter === 'offline') {
    list = list.filter(m => !m.loaded);
  }

  if (list.length === 0) {
    grid.innerHTML = '<div class="empty">No models match your filter.</div>';
    return;
  }

  const perModel = (statusCache && statusCache.per_model) || {};
  const idsNeeded = new Set(list.map(m => m.id));

  // Collect existing card nodes by data-id
  const existingCards = new Map();
  grid.querySelectorAll('.model-card').forEach(c => existingCards.set(c.dataset.id, c));

  // Remove cards no longer present in filtered list
  existingCards.forEach((card, id) => {
    if (!idsNeeded.has(id)) card.remove();
  });

  list.forEach(m => {
    const pm = perModel[m.id] || {};
    let card = existingCards.get(m.id);

    if (!card) {
      // Build brand new card element
      const tags = (m.tags||[]).map(t => `<span class="tag">${esc(t)}</span>`).join('');
      const argsDisplay = `--ctx-size ${m.ctx_size} --n-gpu-layers ${m.n_gpu_layers} ${m.args||''}`.trim();
      const visionTag = m.has_mmproj && m.use_mmproj ? '<span class="tag">vision</span>' : '';
      const name = esc(m.name || m.id);
      const id = esc(m.id);

      card = document.createElement('div');
      card.className = 'model-card';
      card.dataset.id = m.id;
      card.innerHTML = `
        <div class="model-head">
          <div>
            <div class="model-name">${name}</div>
            <div class="model-meta">${id} · ${fmtBytes(m.size_mb)}${m.port ? ' · :'+esc(m.port) : ''}${m.pid ? ' · pid '+esc(m.pid) : ''}</div>
            <div style="margin-top:6px;">${tags}${m.default?'<span class="tag">default</span>':''}${m.pinned?'<span class="tag">pinned</span>':''}${m.auto_save_state?'<span class="tag">autosave</span>':''}${visionTag}</div>
          </div>
          <div class="status-wrap">${statusBadge(m.loaded, m.ready)}</div>
        </div>

        <div class="args-box">${esc(argsDisplay) || '(no extra args)'}</div>

        <div class="model-stats">
          <div>VRAM: <b>${pm.vram_mb ? fmtBytes(pm.vram_mb) : '—'}</b></div>
          <div>RAM:  <b>${pm.ram_mb ? fmtBytes(pm.ram_mb) : '—'}</b></div>
          <div>ctx / max: <b>${m.ctx_size}${m.max_ctx_size ? ' / ' + m.max_ctx_size.toLocaleString() : ''}</b></div>
          <div>uptime: <b>${m.loaded && statusCache ? formatUptime(m.id) : '—'}</b></div>
        </div>

        <div class="btn-row">${buildButtonRowHTML(m)}</div>`;

      grid.appendChild(card);
    } else {
      // In-place DOM update without tearing down node
      const metaEl = card.querySelector('.model-meta');
      if (metaEl) {
        metaEl.innerHTML = `${esc(m.id)} · ${fmtBytes(m.size_mb)}${m.port ? ' · :'+esc(m.port) : ''}${m.pid ? ' · pid '+esc(m.pid) : ''}`;
      }

      const statusWrap = card.querySelector('.status-wrap');
      if (statusWrap) {
        statusWrap.innerHTML = statusBadge(m.loaded, m.ready);
      }

      const statsDiv = card.querySelector('.model-stats');
      if (statsDiv && statsDiv.children.length >= 4) {
        statsDiv.children[0].innerHTML = `VRAM: <b>${pm.vram_mb ? fmtBytes(pm.vram_mb) : '—'}</b>`;
        statsDiv.children[1].innerHTML = `RAM:  <b>${pm.ram_mb ? fmtBytes(pm.ram_mb) : '—'}</b>`;
        statsDiv.children[2].innerHTML = `ctx / max: <b>${m.ctx_size}${m.max_ctx_size ? ' / ' + m.max_ctx_size.toLocaleString() : ''}</b>`;
        statsDiv.children[3].innerHTML = `uptime: <b>${m.loaded && statusCache ? formatUptime(m.id) : '—'}</b>`;
      }

      // Update button row in-place to avoid flickering from innerHTML rebuild
      updateButtonRow(card, m);
    }

    // Attach/update button click listeners (idempotent)
    card.querySelectorAll('button[data-act]').forEach(btn => {
      btn.onclick = () => handleAction(m.id, btn.dataset.act, btn);
    });
  });
}

function formatUptime(id) {
  if (!statusCache || !Array.isArray(statusCache.uptime_models)) return '—';
  const m = statusCache.uptime_models.find(x => x.id === id);
  if (!m) return '—';
  const s = m.uptime_s;
  if (s < 60) return s.toFixed(0)+'s';
  if (s < 3600) return Math.floor(s/60)+'m';
  return Math.floor(s/3600)+'h ' + Math.floor((s%3600)/60)+'m';
}

async function handleAction(id, act, btn) {
  inFlightActions.set(id, act);
  renderModels(); // refresh UI immediately to disable button

  try {
    if (act === 'load') {
      await apiJSON(`/v1/models/${encodeURIComponent(id)}/load`, {method:'POST'});
      showToast(`Model '${id}' loaded successfully`, 'success');
    }
    if (act === 'unload') {
      await apiJSON(`/v1/models/${encodeURIComponent(id)}/unload`, {method:'POST'});
      showToast(`Model '${id}' unloaded`, 'info');
    }
    if (act === 'edit') {
      openEditModal(id);
    }
    if (act === 'states') {
      openStateModal(id);
    }
    if (act === 'toggle-vision') {
      const res = await apiJSON(`/v1/models/${encodeURIComponent(id)}/vision/toggle`, {method:'POST'});
      showToast(`Vision ${res.use_mmproj ? 'enabled' : 'disabled'} for '${id}'`, 'info');
    }
    if (act === 'copy-curl') {
      const host = statusCache ? statusCache.launcher.host : 'localhost';
      const port = statusCache ? statusCache.launcher.port : '9123';
      const cmd = `curl http://${host}:${port}/v1/chat/completions \\\n  -H 'Content-Type: application/json' \\\n  -d '{"model":"${id}", "messages":[{"role":"user","content":"Hello!"}]}'`;
      await navigator.clipboard.writeText(cmd);
      showToast('cURL command copied to clipboard!', 'success');
    }
  } catch (e) {
    showToast('Error: ' + e.message, 'error');
  } finally {
    inFlightActions.delete(id);
    await refreshAll();
  }
}

function renderBackendSelectors(st) {
  const backends = (st.launcher && st.launcher.backends) || [];
  const selected = (st.launcher && st.launcher.selected_backend) || "";

  const gSel = $('global-backend-select');
  const fSel = $('f-backend');

  const fVal = fSel.value;

  let gHtml = '<option value="">System Default</option>';
  let fHtml = '<option value="">(Use Global Default)</option>';

  backends.forEach(b => {
    gHtml += `<option value="${esc(b.id)}">${esc(b.name)}</option>`;
    fHtml += `<option value="${esc(b.id)}">${esc(b.name)}</option>`;
  });

  gSel.innerHTML = gHtml;
  fSel.innerHTML = fHtml;

  gSel.value = selected;
  if (fVal) fSel.value = fVal;
}

$('global-backend-select').onchange = async (ev) => {
  const val = ev.target.value;
  try {
    await apiJSON('/v1/backend/select', {
      method: 'POST',
      headers: {'Content-Type':'application/json'},
      body: JSON.stringify({backend: val})
    });
    showToast(`Global backend set to '${val || "Default"}'`, 'success');
    await refreshAll();
  } catch (e) {
    showToast('Failed to select backend: ' + e.message, 'error');
  }
};

function openEditModal(id) {
  const m = modelsCache.find(x => x.id === id);
  if (!m) return;
  editingId = id;
  $('f-name').value = m.name || '';
  $('f-desc').value = m.description || '';
  $('f-backend').value = m.backend || '';
  $('f-args').value = m.args || '';
  $('f-ctx').value = m.ctx_size !== undefined && m.ctx_size !== null ? m.ctx_size : '';
  const ctxLbl = $('f-ctx-label');
  if (ctxLbl) {
    ctxLbl.textContent = m.max_ctx_size ? `ctx_size (max: ${m.max_ctx_size.toLocaleString()})` : 'ctx_size';
  }
  const ctxInp = $('f-ctx');
  if (m.max_ctx_size) {
    ctxInp.max = m.max_ctx_size;
    ctxInp.placeholder = `Max ${m.max_ctx_size}`;
  } else {
    ctxInp.removeAttribute('max');
    ctxInp.placeholder = '8192';
  }
  $('f-ngl').value = m.n_gpu_layers !== undefined && m.n_gpu_layers !== null ? m.n_gpu_layers : '';
  $('f-vram').value = m.estimated_vram_mb || 0;
  $('f-tags').value = (m.tags||[]).join(', ');
  $('f-mmproj-file').value = m.mmproj_file || '';
  $('f-default').checked = !!m.default;
  $('f-pinned').checked = !!m.pinned;
  $('f-autosave').checked = !!m.auto_save_state;
  $('f-use-mmproj').checked = !!m.use_mmproj;
  $('modal-overlay').classList.add('active');
}

$('modal-cancel').onclick = () => $('modal-overlay').classList.remove('active');
$('modal-save').onclick = async () => {
  const ctxVal = $('f-ctx').value.trim();
  const nglVal = $('f-ngl').value.trim();
  const vramVal = $('f-vram').value.trim();

  const body = {
    name: $('f-name').value,
    description: $('f-desc').value,
    backend: $('f-backend').value,
    args: $('f-args').value,
    ctx_size: ctxVal ? parseInt(ctxVal, 10) : undefined,
    n_gpu_layers: nglVal ? parseInt(nglVal, 10) : undefined,
    estimated_vram_mb: vramVal ? parseInt(vramVal, 10) : 0,
    tags: $('f-tags').value.split(',').map(s=>s.trim()).filter(Boolean),
    mmproj_file: $('f-mmproj-file').value,
    default: $('f-default').checked,
    pinned: $('f-pinned').checked,
    auto_save_state: $('f-autosave').checked,
    use_mmproj: $('f-use-mmproj').checked,
  };

  const currentModel = modelsCache.find(x => x.id === editingId);
  if (currentModel && currentModel.max_ctx_size && body.ctx_size > currentModel.max_ctx_size) {
    body.ctx_size = currentModel.max_ctx_size;
    showToast(`Capped ctx_size to model max (${currentModel.max_ctx_size.toLocaleString()})`, 'info');
  }

  const btn = $('modal-save');
  btn.disabled = true;
  const originalText = btn.textContent;
  btn.textContent = 'Saving...';

  try {
    await apiJSON(`/v1/models/${encodeURIComponent(editingId)}/config`, {
      method: 'PUT', headers:{'Content-Type':'application/json'},
      body: JSON.stringify(body)
    });
    $('modal-overlay').classList.remove('active');
    showToast('Model config saved', 'success');
    await refreshAll();
  } catch (e) {
    showToast('Save failed: ' + e.message, 'error');
  } finally {
    btn.disabled = false;
    btn.textContent = originalText;
  }
};

// ---------- State Management Modal ----------
async function openStateModal(id) {
  stateManagingId = id;
  $('sm-model-id').textContent = id;
  $('sm-states-list').innerHTML = '<div class="empty" style="padding:12px;">Loading states…</div>';
  $('state-overlay').classList.add('active');
  await refreshStateList();
}

async function refreshStateList() {
  if (!stateManagingId) return;
  try {
    const res = await apiJSON(`/v1/models/${encodeURIComponent(stateManagingId)}/state`);
    const labels = res.labels || [];
    const listEl = $('sm-states-list');

    if (labels.length === 0) {
      listEl.innerHTML = '<div class="empty" style="padding:12px;">No saved states found for this model.</div>';
      return;
    }

    listEl.innerHTML = labels.map(lbl => `
      <div class="state-item">
        <span><b>${esc(lbl)}</b></span>
        <div class="actions">
          <button class="primary" data-act="restore-state" data-lbl="${esc(lbl)}">Restore</button>
          <button class="danger" data-act="delete-state" data-lbl="${esc(lbl)}">Delete</button>
        </div>
      </div>
    `).join('');

    listEl.querySelectorAll('button[data-act]').forEach(btn => {
      btn.onclick = () => handleStateAction(btn.dataset.act, btn.dataset.lbl);
    });
  } catch (e) {
    $('sm-states-list').innerHTML = `<div class="empty" style="padding:12px;color:var(--red);">Failed to load states: ${esc(e.message)}</div>`;
  }
}

async function handleStateAction(act, label) {
  if (!stateManagingId || !label) return;
  try {
    if (act === 'restore-state') {
      await apiJSON(`/v1/models/${encodeURIComponent(stateManagingId)}/state/load`, {
        method: 'POST', headers: {'Content-Type':'application/json'},
        body: JSON.stringify({label})
      });
      showToast(`Restored state '${label}' for '${stateManagingId}'`, 'success');
    }
    if (act === 'delete-state') {
      if (!confirm(`Delete state '${label}'?`)) return;
      await apiJSON(`/v1/models/${encodeURIComponent(stateManagingId)}/state/${encodeURIComponent(label)}`, {
        method: 'DELETE'
      });
      showToast(`Deleted state '${label}'`, 'info');
      await refreshStateList();
    }
  } catch (e) {
    showToast('State action failed: ' + e.message, 'error');
  }
}

$('sm-btn-save').onclick = async () => {
  const label = ($('sm-new-label').value || 'default').trim();
  if (!stateManagingId || !label) return;
  try {
    await apiJSON(`/v1/models/${encodeURIComponent(stateManagingId)}/state/save`, {
      method: 'POST', headers: {'Content-Type':'application/json'},
      body: JSON.stringify({label})
    });
    showToast(`Saved state '${label}' for '${stateManagingId}'`, 'success');
    await refreshStateList();
  } catch (e) {
    showToast('Failed to save state: ' + e.message, 'error');
  }
};

$('sm-cancel').onclick = () => $('state-overlay').classList.remove('active');

async function refreshAll() {
  try {
    const [models, st] = await Promise.all([
      apiJSON('/v1/models'),
      apiJSON('/v1/status'),
    ]);
    modelsCache = (models.data||[]).map(d => ({
      id: d.id,
      name: d.name || d.id,
      description: d.description || '',
      tags: d.tags || [],
      size_mb: d.size_mb || 0,
      loaded: d.loaded,
      ready: d.ready,
      port: d.port,
      pid: d.pid,
      default: d.default,
      pinned: d.pinned,
      auto_save_state: d.auto_save_state,
      backend: d.backend,
      use_mmproj: d.use_mmproj,
      mmproj_file: d.mmproj_file,
      has_mmproj: d.has_mmproj,
      ctx_size: d.ctx_size,
      max_ctx_size: d.max_ctx_size,
      n_gpu_layers: d.n_gpu_layers,
      estimated_vram_mb: d.estimated_vram_mb,
      args: d.args,
      gguf_name: d.gguf_name,
    }));
    statusCache = st;
    renderBackendSelectors(st);
    renderGPUs(st.gpus);
    renderRAM(st.ram);
    $('launcher-addr').textContent = `${st.launcher.host}:${st.launcher.port}`;
    $('loaded-count').textContent  = `${st.models_loaded}/${st.models_total}`;
    renderModels();
  } catch (e) {
    console.error(e);
  }
}

$('btn-rescan').onclick = async () => {
  await apiJSON('/v1/scan', {method:'POST'});
  modelsCache = [];
  showToast('Rescanned models directory', 'info');
  await refreshAll();
};

$('btn-unload-all').onclick = async () => {
  if (!confirm('Unload all models?')) return;
  await apiJSON('/v1/unload_all', {method:'POST'});
  showToast('All models unloaded', 'info');
  await refreshAll();
};

$('filter').oninput = renderModels;
$('status-filter').onchange = renderModels;

function wsLoop() {
  let ws;
  let retryDelay = 1000;
  function connect() {
    const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
    ws = new WebSocket(`${proto}//${location.host}/ws`);
    ws.onopen = () => { retryDelay = 1000; };
    ws.onmessage = async (ev) => {
      try {
        const st = JSON.parse(ev.data);
        statusCache = st;
        renderGPUs(st.gpus);
        renderRAM(st.ram);
        $('loaded-count').textContent = `${st.models_loaded}/${st.models_total}`;

        if (Array.isArray(st.uptime_models)) {
          const umMap = new Map(st.uptime_models.map(um => [um.id, um]));
          modelsCache.forEach(m => {
            const um = umMap.get(m.id);
            if (um) {
              m.loaded = true;
              m.ready = um.ready;
              m.port = um.port;
              m.pid = um.pid;
            } else {
              m.loaded = false;
              m.ready = false;
              m.port = null;
              m.pid = null;
            }
          });
        }
        renderModels();
      } catch (e) { console.error('WS update error:', e); }
    };
    ws.onclose = () => {
      setTimeout(connect, retryDelay);
      retryDelay = Math.min(retryDelay * 1.5, 10000);
    };
    ws.onerror = () => ws.close();
  }
  connect();
}

// ---------- Settings ----------
async function loadSettings() {
  try {
    const s = await apiJSON('/v1/settings');
    $('s-idle').value = s.idle_timeout_seconds;
    $('s-max-loaded').value = s.max_loaded_models;
    $('s-poll').value = s.poll_interval_seconds;
    $('s-base-port').value = s.base_port;
    $('s-default-args').value = s.default_args || '';
    $('s-selected-backend').value = s.selected_backend || '';
    $('s-auto-save').checked = !!s.auto_save_state;
    $('s-host').value = s.host || '';
    $('s-port').value = s.port || '';
  } catch (e) {
    console.error('Failed to load settings:', e);
    showToast('Failed to load settings: ' + e.message, 'error');
  }
}

async function saveSettings() {
  const idle = Number($('s-idle').value);
  const maxLoaded = Number($('s-max-loaded').value);
  const poll = Number($('s-poll').value);
  const basePort = Number($('s-base-port').value);

  if (isNaN(idle) || idle <= 0) {
    showToast('Idle timeout must be a positive number', 'error');
    return;
  }
  if (isNaN(maxLoaded) || maxLoaded < 1 || !Number.isInteger(maxLoaded)) {
    showToast('Max loaded models must be an integer ≥ 1', 'error');
    return;
  }
  if (isNaN(poll) || poll <= 0) {
    showToast('Poll interval must be a positive number', 'error');
    return;
  }
  if (isNaN(basePort) || basePort < 1024 || basePort > 65535 || !Number.isInteger(basePort)) {
    showToast('Base port must be an integer between 1024 and 65535', 'error');
    return;
  }

  const body = {
    idle_timeout_seconds: idle,
    max_loaded_models: maxLoaded,
    poll_interval_seconds: poll,
    base_port: basePort,
    default_args: $('s-default-args').value,
    selected_backend: $('s-selected-backend').value,
    auto_save_state: $('s-auto-save').checked,
  };
  try {
    await apiJSON('/v1/settings', {
      method: 'PUT',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(body),
    });
    $('settings-overlay').classList.remove('active');
    showToast('Global settings saved', 'success');
  } catch (e) {
    showToast('Settings save failed: ' + e.message, 'error');
  }
}

$('btn-settings').onclick = () => {
  loadSettings().then(() => $('settings-overlay').classList.add('active'));
};
$('settings-cancel').onclick = () => $('settings-overlay').classList.remove('active');
$('settings-save').onclick = saveSettings;

// Load settings on startup
loadSettings();
refreshAll().then(wsLoop);
