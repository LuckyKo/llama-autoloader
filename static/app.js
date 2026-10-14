/* llama-autoloader WebUI */
const $ = (id) => document.getElementById(id);
let modelsCache = [];
let statusCache = null;
let editingId = null;
let stateManagingId = null;

// Toast timing (ms)
const TOAST_MS = 3500;
const TOAST_FADE_MS = 300;

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
    setTimeout(() => toast.remove(), TOAST_FADE_MS);
  }, TOAST_MS);
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
  if (!ram) {
    $('ram-bar').style.width = '0%';
    $('ram-used').textContent = '—';
    $('ram-total').textContent = '—';
    return;
  }
  const pct = ram.pct;
  $('ram-bar').style.width = pct.toFixed(1) + '%';
  $('ram-used').textContent = fmtBytes(ram.used_mb) + ' used';
  $('ram-total').textContent = fmtBytes(ram.total_mb) + ' total';
}

function getInFlightState(id) {
  const act = inFlightActions.get(id);
  return {
    act,
    isLoadInFlight: act === 'load',
    isUnloadInFlight: act === 'unload',
    isVisionInFlight: act === 'toggle-vision',
    isDeleteInFlight: act === 'delete',
  };
}

function buildButtonRowHTML(m) {
  const { act, isLoadInFlight, isUnloadInFlight, isVisionInFlight, isDeleteInFlight } = getInFlightState(m.id);

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
    <button data-act="copy-id">Copy ID</button>
    ${m.engine === 'llama' ? '<button data-act="states">States</button>' : ''}
    ${m.engine !== 'llama' ? '<button class="danger" data-act="delete" ' + (act ? 'disabled' : '') + '>' + (isDeleteInFlight ? 'Deleting...' : 'Delete') + '</button>' : ''}
  `;
}

function updateButtonRow(card, m) {
  // Update buttons in-place to avoid flickering from innerHTML replacement.
  // Only called for existing cards during incremental updates.
  const btnRow = card.querySelector('.btn-row');
  if (!btnRow) return;

  const { act, isLoadInFlight, isUnloadInFlight, isVisionInFlight } = getInFlightState(m.id);

  // Structure of the row is determined by load-state (Load vs Unload button) and
  // whether a Vision button exists. If either changed, rebuild in one shot — this
  // covers load/unload swaps AND vision-button create/remove cleanly.
  const currentLoaded = !!btnRow.querySelector('button[data-act="unload"]');
  const currentHasVision = !!btnRow.querySelector('button[data-act="toggle-vision"]');
  const currentHasStates = !!btnRow.querySelector('button[data-act="states"]');
  if (currentLoaded !== !!m.loaded || currentHasVision !== !!m.has_mmproj ||
      currentHasStates !== (m.engine === 'llama')) {
    btnRow.innerHTML = buildButtonRowHTML(m);
    return;
  }

  // Structure unchanged — fine-grained text/disabled updates only.
  const loadBtn = m.loaded
    ? btnRow.querySelector('button[data-act="unload"]')
    : btnRow.querySelector('button[data-act="load"]');
  if (loadBtn) {
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

function tagsRowHTML(m) {
  const tags = (m.tags||[]).map(t => `<span class="tag">${esc(t)}</span>`).join('');
  return tags
    + (m.default ? '<span class="tag">default</span>' : '')
    + (m.pinned ? '<span class="tag">pinned</span>' : '')
    + (m.auto_save_state ? '<span class="tag">autosave</span>' : '')
    + (m.has_mmproj && m.use_mmproj ? '<span class="tag">vision</span>' : '')
    // Bridge cards carry an engine badge; llama is the default and gets none.
    + (m.engine && m.engine !== 'llama' ? `<span class="tag">${esc(m.engine)}</span>` : '');
}

function updateLoadedBar() {
  const bar = $('loaded-bar');
  if (!bar) return;
  if (!modelsCache || modelsCache.length === 0) { bar.textContent = 'Loading…'; return; }
  const loaded = modelsCache.filter(m => m.loaded);
  if (loaded.length === 0) { bar.textContent = 'No model currently loaded.'; return; }
  bar.textContent = 'Loaded: ' + loaded.map(m => `${m.name || m.id} (port ${m.port || '?'})`).join(' · ');
}
function renderModels() {
  updateLoadedBar();
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
    const hasFilter = filter || statusFilter !== 'all';
    grid.innerHTML = `<div class="empty">${hasFilter ? 'No models match your filter.' : 'No models found.'}</div>`;
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

    // Extra-args display, computed once and shared by both render paths.
    // Bridge engines get min-free-vram + free-form extra_args; llama gets its flags.
    const parts = [];
    if (m.engine !== 'llama') {
      if (m.min_free_vram_mib > 0) parts.push(`--min-free-vram-mib ${m.min_free_vram_mib}`);
      if (m.extra_args && m.extra_args.length) parts.push(m.extra_args.join(' '));
    } else {
      if (m.ctx_size != null) parts.push(`--ctx-size ${m.ctx_size}`);
      if (m.n_gpu_layers != null) parts.push(`--n-gpu-layers ${m.n_gpu_layers}`);
      if (m.args) parts.push(m.args);
    }
    const argsDisplay = parts.join(' ');

    // Meta line: bridge cards show the (truncated) engine-config path instead of size_mb.
    const metaMid = m.engine !== 'llama' && m.engine_config
      ? (m.engine_config.length > 60 ? '…' + m.engine_config.slice(-59) : m.engine_config)
      : fmtBytes(m.size_mb);

    if (!card) {
      // Build brand new card element
      const name = esc(m.name || m.id);
      const id = esc(m.id);

      card = document.createElement('div');
      card.className = 'model-card';
      card.dataset.id = m.id;
      card.innerHTML = `
        <div class="model-head">
          <div>
            <div class="model-name">${name}</div>
            <div class="model-meta">${id} · ${esc(metaMid)}${m.port ? ' · :'+esc(m.port) : ''}${m.pid ? ' · pid '+esc(m.pid) : ''}</div>
            <div class="tags-row" style="margin-top:6px;">${tagsRowHTML(m)}</div>
          </div>
          <div class="status-wrap">${statusBadge(m.loaded, m.ready)}</div>
        </div>

        <div class="args-box">${esc(argsDisplay) || '(no extra args)'}</div>

        <div class="model-stats">
          <div data-stat="vram">VRAM: <b>${pm.vram_mb ? fmtBytes(pm.vram_mb) : '—'}</b></div>
          <div data-stat="ram">RAM:  <b>${pm.ram_mb ? fmtBytes(pm.ram_mb) : '—'}</b></div>
          <div data-stat="ctx">ctx / max: <b>${m.ctx_size != null ? m.ctx_size : '—'}${m.max_ctx_size ? ' / ' + m.max_ctx_size.toLocaleString() : ''}</b></div>
          <div data-stat="uptime">uptime: <b>${m.loaded && statusCache ? formatUptime(m.id) : '—'}</b></div>
        </div>

        <div class="btn-row">${buildButtonRowHTML(m)}</div>`;

      // Only llama reports ctx; the row is meaningless for a bridge engine.
      if (m.engine !== 'llama') card.querySelector('[data-stat="ctx"]').remove();
      grid.appendChild(card);
    } else {
      // In-place DOM update without tearing down node
      const nameEl = card.querySelector('.model-name');
      if (nameEl) {
        nameEl.textContent = m.name || m.id;
      }

      const metaEl = card.querySelector('.model-meta');
      if (metaEl) {
        metaEl.innerHTML = `${esc(m.id)} · ${esc(metaMid)}${m.port ? ' · :'+esc(m.port) : ''}${m.pid ? ' · pid '+esc(m.pid) : ''}`;
      }

      const tagsRow = card.querySelector('.tags-row');
      if (tagsRow) {
        tagsRow.innerHTML = tagsRowHTML(m);
      }

      const argsBox = card.querySelector('.args-box');
      if (argsBox) {
        argsBox.textContent = argsDisplay || '(no extra args)';
      }

      const statusWrap = card.querySelector('.status-wrap');
      if (statusWrap) {
        statusWrap.innerHTML = statusBadge(m.loaded, m.ready);
      }

      const statsDiv = card.querySelector('.model-stats');
      if (statsDiv) {
        const vramEl = statsDiv.querySelector('[data-stat="vram"]');
        const ramEl = statsDiv.querySelector('[data-stat="ram"]');
        const ctxEl = statsDiv.querySelector('[data-stat="ctx"]');
        const uptimeEl = statsDiv.querySelector('[data-stat="uptime"]');
        if (vramEl) vramEl.innerHTML = `VRAM: <b>${pm.vram_mb ? fmtBytes(pm.vram_mb) : '—'}</b>`;
        if (ramEl) ramEl.innerHTML = `RAM:  <b>${pm.ram_mb ? fmtBytes(pm.ram_mb) : '—'}</b>`;
        if (ctxEl) ctxEl.innerHTML = `ctx / max: <b>${m.ctx_size != null ? m.ctx_size : '—'}${m.max_ctx_size ? ' / ' + m.max_ctx_size.toLocaleString() : ''}</b>`;
        if (uptimeEl) uptimeEl.innerHTML = `uptime: <b>${m.loaded && statusCache ? formatUptime(m.id) : '—'}</b>`;
        // Bridge engines have no ctx. ctxEl is null once removed → guarded (repeat renders).
        if (m.engine !== 'llama' && ctxEl) ctxEl.remove();
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
      showToast(`Model ${id} loaded successfully`, 'success');
    }
    if (act === 'unload') {
      await apiJSON(`/v1/models/${encodeURIComponent(id)}/unload`, {method:'POST'});
      showToast(`Model ${id} unloaded`, 'info');
    }
    if (act === 'delete') {
      if (!confirm(`Delete engine card ${id}?`)) return;
      await apiJSON(`/v1/models/${encodeURIComponent(id)}`, {method:'DELETE'});
      showToast(`Engine card ${id} deleted`, 'info');
    }
    if (act === 'edit') {
      openEditModal(id);
    }
    if (act === 'states') {
      openStateModal(id);
    }
    if (act === 'toggle-vision') {
      const res = await apiJSON(`/v1/models/${encodeURIComponent(id)}/vision/toggle`, {method:'POST'});
      showToast(`Vision ${res.use_mmproj ? 'enabled' : 'disabled'} for ${id}`, 'info');
    }
    if (act === 'copy-curl') {
      const host = statusCache ? statusCache.launcher.host : 'localhost';
      const port = statusCache ? statusCache.launcher.port : '9123';
      const cmd = `curl http://${host}:${port}/v1/chat/completions \\\n  -H 'Content-Type: application/json' \\\n  -d '{"model":"${id}", "messages":[{"role":"user","content":"Hello!"}]}'`;
      await navigator.clipboard.writeText(cmd);
      showToast('cURL command copied to clipboard!', 'success');
    }
    if (act === 'copy-id') {
      await navigator.clipboard.writeText(id);
      showToast(`Copied model id: ${id}`, 'success');
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

  let gHtml = '<option value="">System Default</option>';
  let fHtml = '<option value="">(Use Global Default)</option>';

  backends.forEach(b => {
    gHtml += `<option value="${esc(b.id)}">${esc(b.name)}</option>`;
    fHtml += `<option value="${esc(b.id)}">${esc(b.name)}</option>`;
  });

  gSel.innerHTML = gHtml;
  gSel.value = selected;

  // Always rebuild the edit-modal backend select so its options stay fresh
  // (openEditModal only sets .value on these options and never rebuilds them).
  const fVal = fSel.value;
  fSel.innerHTML = fHtml;
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
  // Bridge cards: gate the form (CSS via .bridge) and populate the engine-only inputs.
  // NOTE: the engine-config path is editable and IS read back on save — it used to be
  // displayed-only, so the path could never be corrected from the WebUI (E27).
  $('modal-overlay').classList.toggle('bridge', m.engine !== 'llama');
  if (m.engine !== 'llama') {
    $('f-engine-config').value = m.engine_config || '';
    $('f-min-free-vram-mib').value = m.min_free_vram_mib || 0;
    $('f-extra-args').value = (m.extra_args || []).join(' ');
  }
  $('modal-overlay').classList.add('active');
}

$('modal-cancel').onclick = () => { editingId = null; $('modal-overlay').classList.remove('active'); };
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
  const isBridge = !!(currentModel && currentModel.engine && currentModel.engine !== 'llama');
  if (isBridge) {
    // The engine-config path is now writable (E27): it used to have an input but no
    // reader, so edits were silently discarded and a wrong path was unfixable in the UI.
    body.engine_config = $('f-engine-config').value.trim();
    body.min_free_vram_mib = parseInt($('f-min-free-vram-mib').value, 10) || 0;
    body.extra_args = $('f-extra-args').value.split(/\s+/).filter(Boolean);
    // Llama-only keys must not leak into a bridge card file.
    ['ctx_size', 'n_gpu_layers', 'estimated_vram_mb', 'args', 'mmproj_file',
     'use_mmproj', 'backend', 'auto_save_state'].forEach(k => delete body[k]);
  } else if (currentModel && currentModel.max_ctx_size && body.ctx_size > currentModel.max_ctx_size) {
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
      showToast(`Restored state ${label} for ${stateManagingId}`, 'success');
    }
    if (act === 'delete-state') {
      if (!confirm(`Delete state '${label}'?`)) return;
      await apiJSON(`/v1/models/${encodeURIComponent(stateManagingId)}/state/${encodeURIComponent(label)}`, {
        method: 'DELETE'
      });
      showToast(`Deleted state ${label}`, 'info');
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
    showToast(`Saved state ${label} for ${stateManagingId}`, 'success');
    await refreshStateList();
  } catch (e) {
    showToast('Failed to save state: ' + e.message, 'error');
  }
};

$('sm-cancel').onclick = () => { stateManagingId = null; $('state-overlay').classList.remove('active'); };

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
      // Bridge-card fields (defaulted for llama cards).
      engine: d.engine || 'llama',
      engine_config: d.engine_config || '',
      resolved_binary: d.resolved_binary || '',
      min_free_vram_mib: d.min_free_vram_mib || 0,
      extra_args: d.extra_args || [],
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
    showToast('Refresh failed: ' + e.message, 'error');
  }
}

// ---------- Add engine bridge card ----------
// The engine dropdown is fed by GET /v1/engines, so a second engine needs no UI change.
// Engines that failed boot validation are listed but disabled, with the reason inline —
// hiding them would make a config typo look like "no engines configured".
async function loadEngineOptions() {
  const sel = $('ea-engine');
  $('ea-error-row').style.display = 'none';  // cleared up front; set below if needed
  let engines = [];
  try {
    engines = await apiJSON('/v1/engines');
  } catch (e) {
    sel.innerHTML = '<option value="">(could not load engines)</option>';
    $('ea-error-row').style.display = 'block';
    $('ea-error').textContent = 'Could not load engines: ' + e.message;
    return;
  }
  if (!engines.length) {
    sel.innerHTML = '<option value="">(no engines configured)</option>';
    return;
  }
  sel.innerHTML = engines.map(en => {
    const label = en.error ? `${en.label} — ${en.error}` : en.label;
    // ready=false means the server would reject the add with a 400; disable it so the
    // user reads why here instead of hitting an opaque failure on submit.
    return `<option value="${esc(en.name)}"${en.ready ? '' : ' disabled'}>${esc(label)}</option>`;
  }).join('');
  const firstReady = engines.find(en => en.ready);
  if (firstReady) sel.value = firstReady.name;
  // Spell out the problems below the dropdown. Disabled options cannot be clicked, so
  // without this the reason would only exist inside an unselectable <option> label.
  const problems = engines.filter(en => !en.ready);
  $('ea-error-row').style.display = problems.length ? 'block' : 'none';
  $('ea-error').textContent = problems.map(en => `${en.name}: ${en.error}`).join(' · ');
}

$('btn-add-engine').onclick = () => {
  $('ea-config').value = '';
  $('ea-name').value = '';
  $('engine-add-overlay').classList.add('active');
  loadEngineOptions();
};

$('ea-cancel').onclick = () => $('engine-add-overlay').classList.remove('active');

$('ea-add').onclick = async () => {
  const engine = $('ea-engine').value;
  const cfgPath = $('ea-config').value.trim();
  if (!engine) {
    showToast('Pick an engine first', 'error');
    return;
  }
  if (!cfgPath) {
    showToast('Engine config path is required', 'error');
    return;
  }
  const btn = $('ea-add');
  btn.disabled = true;
  try {
    const res = await apiJSON('/v1/models/engine', {
      method: 'POST', headers: {'Content-Type':'application/json'},
      body: JSON.stringify({engine, engine_config: cfgPath, name: $('ea-name').value.trim() || undefined})
    });
    $('engine-add-overlay').classList.remove('active');
    showToast(`Added ${engine} card ${res.id}`, 'success');
    await refreshAll();
  } catch (e) {
    // 400 (unknown/misconfigured engine, invalid config) / 409 (id collision) surface
    // here via e.message
    showToast('Add failed: ' + e.message, 'error');
  } finally {
    btn.disabled = false;
  }
};

$('btn-rescan').onclick = async () => {
  try {
    await apiJSON('/v1/scan', {method:'POST'});
    modelsCache = [];
    showToast('Rescanned models directory', 'info');
    await refreshAll();
  } catch (e) {
    showToast('Rescan failed: ' + e.message, 'error');
  }
};

$('btn-unload-all').onclick = async () => {
  if (!confirm('Unload all models?')) return;
  try {
    await apiJSON('/v1/unload_all', {method:'POST'});
    showToast('All models unloaded', 'info');
    await refreshAll();
  } catch (e) {
    showToast('Unload all failed: ' + e.message, 'error');
  }
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
    $('s-idle').value = s.idle_timeout_seconds ?? '';
    $('s-max-loaded').value = s.max_loaded_models ?? '';
    $('s-save-state-max-keep').value = s.save_state_max_keep ?? '';
    $('s-poll').value = s.poll_interval_seconds ?? '';
    $('s-base-port').value = s.base_port ?? '';
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
  const saveStateMaxKeep = Number($('s-save-state-max-keep').value);
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
  if (isNaN(saveStateMaxKeep) || saveStateMaxKeep < 1 || !Number.isInteger(saveStateMaxKeep)) {
    showToast('Max state files to keep must be an integer ≥ 1', 'error');
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
    save_state_max_keep: saveStateMaxKeep,
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
