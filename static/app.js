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

// ---------------------------------------------------------------------------
// Model types
// ---------------------------------------------------------------------------
// Mirrors MODEL_TYPE_* in server.py. The server is the source of truth for what a card
// IS (model_type on /v1/models); this table only supplies the presentation — colour,
// group description and the advisory hint — and the canonical group order.
const TYPE_META = {
  text:      { label: 'Text Generation',    color: '#7c5cff', desc: 'Chat / completion models' },
  vision:    { label: 'Vision',             color: '#4ade80', desc: 'Multimodal — vision built into the weights' },
  audio:     { label: 'Audio Generation',   color: '#fbbf24', desc: 'Text to speech and other audio output models' },
  asr:       { label: 'Speech to Text',     color: '#f59e0b', desc: 'Audio transcription' },
  embedding: { label: 'Embeddings',         color: '#38bdf8', desc: 'Serve /v1/embeddings, not chat' },
  draft:     { label: 'Draft / Speculative',color: '#a78bfa', desc: 'Speculative-decoding draft, used via --model-draft' },
  codec:     { label: 'Audio Codec',        color: '#fb923c', desc: 'Codec / VAE consumed by an audio model' },
  mmproj:    { label: 'Vision Projector',   color: '#94a3b8', desc: 'Vision projector, attached to a host model via --mmproj' },
};
const TYPE_ORDER = ['text','vision','audio','asr','embedding','draft','codec','mmproj'];
const DEFAULT_TYPE = 'text';

// Component types: loaded by a host model rather than served on their own. These get an
// explanatory notice on the card. NOT a block - the Load button stays enabled.
const TYPE_HINTS = {
  mmproj:    'Vision projector. Attached to a host model with --mmproj — not served on its own.',
  codec:     'Audio codec / VAE. Consumed by an audio model — not served on its own.',
  draft:     'Speculative-decoding draft. Referenced with --model-draft, not served on its own.',
  embedding: 'Encoder-only model. Serves /v1/embeddings, not /v1/chat/completions.',
};

function typeMeta(t) { return TYPE_META[t] || TYPE_META[DEFAULT_TYPE]; }
function typeOrderIndex(t) {
  const i = TYPE_ORDER.indexOf(t);
  return i === -1 ? TYPE_ORDER.length : i;
}

// Active type filter. '' means "All" (cards render under group headers).
let typeFilter = '';
// engine name -> {name, label, ready, error, color}, from GET /v1/engines. Fetched once
// at startup and refreshed whenever the engine list can have changed (add/delete card),
// NOT on every WebSocket tick — it is effectively static config.
let enginesCache = {};
// Distinct palette for engine badges. An engine's colour is derived from its name, so
// Strata always looks the same across reloads and instances.
const ENGINE_COLORS = ['#f472b6', '#22d3ee', '#fb923c', '#a3e635', '#c084fc', '#facc15'];
function engineColor(name) {
  let h = 0;
  for (let i = 0; i < name.length; i++) h = (h * 31 + name.charCodeAt(i)) >>> 0;
  return ENGINE_COLORS[h % ENGINE_COLORS.length];
}

// id -> card element, and type -> {section, grid, countEl, descEl}. Module-level so
// renderModels() can do in-place updates across renders instead of rebuilding the DOM
// on every 1 Hz WebSocket tick.
const cardNodes = new Map();
const groupNodes = new Map();

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
    ${(m.engine === 'llama' || m.engine === 'strata' || m.engine === 'ninfer') ? '<button data-act="states">States</button>' : ''}
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
      currentHasStates !== (m.engine === 'llama' || m.engine === 'strata' || m.engine === 'ninfer')) {
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

function typeBadgeHTML(m) {
  const t = m.model_type || DEFAULT_TYPE;
  const meta = typeMeta(t);
  // A dashed border marks a type the user pinned in the sidecar, so an auto-detected
  // classification is never mistaken for a deliberate choice.
  const pinned = !!m.model_type_override;
  return `<span class="type-badge${pinned ? ' override' : ''}" style="--type-color:${meta.color}"`
       + ` title="${esc(meta.label)}${pinned ? ' (pinned via sidecar override)' : ' (auto-detected)'}">`
       + `${esc(meta.label)}${pinned ? '<span class="badge-override-mark">✎</span>' : ''}</span>`;
}

// Bridge-card provenance: which configured engine serves this card, and whether that
// engine is actually usable. Dashed (like a pinned type) to read as "structural", and
// coloured per engine so Strata looks the same everywhere. A misconfigured engine is
// surfaced here rather than being discovered as an opaque 400 on Load — the WebUI already
// knows the reason from GET /v1/engines.
function engineBadgeHTML(m) {
  const name = m.engine;
  if (!name || name === 'llama') return '';
  const info = enginesCache[name];
  const label = info && info.label ? info.label : name;
  const broken = info && info.ready === false;
  const tip = !info
    ? `${name} (not present in the engine registry — loading will fail with a 400)`
    : broken ? `${label}: ${info.error}` : `${label} engine`;
  return `<span class="engine-badge${broken ? ' broken' : ''}" style="--engine-color:${engineColor(name)}"`
       + ` title="${esc(tip)}">⚙ ${esc(label)}${broken ? '<span class="engine-warn">!</span>' : ''}</span>`;
}

function typeHintHTML(m) {
  const t = m.model_type || DEFAULT_TYPE;
  let hint = TYPE_HINTS[t] || '';
  // Surfaces a bridge engine's boot-time validation error on the card itself.
  const info = enginesCache[m.engine];
  if (info && info.ready === false) {
    hint = `${info.label} engine is misconfigured: ${info.error}` + (hint ? ' — ' + hint : '');
  }
  if (!hint) return '';
  return `<div class="type-hint"><span class="icon">⚠</span><span>${esc(hint)}</span></div>`;
}

// Vision provenance: the bare "vision" tag used to hide the file that provides the
// capability, and whether WE inferred that link or the user set it. A wrong pairing was
// therefore undetectable from the card - which is exactly how one survived an audit.
function mmprojTagHTML(m) {
  if (!m.has_mmproj) return '';
  const file = m.mmproj_resolved || m.mmproj_file || 'unknown';
  const on = !!m.use_mmproj;
  const auto = m.mmproj_source === 'auto';
  const known = m.mmproj_source === 'auto' || m.mmproj_source === 'explicit';
  const src = !known ? 'unknown source'
             : auto ? 'inferred at scan — edit to pin it'
             : 'set in your sidecar';
  const short = file.length > 26 ? '…' + file.slice(-25) : file;
  // Single line on purpose: a newline inside a title attribute is not rendered as a break.
  const tip = `Vision projector: ${file} · Source: ${src} · `
            + `--mmproj is ${on ? 'ON for this model' : 'OFF (projector found, but the toggle is off)'}`;
  return `<span class="tag tag-proj${on ? '' : ' off'}" title="${esc(tip)}">`
       + `vision ← ${esc(short)}${auto ? '<span class="proj-mark">auto</span>' : ''}`
       + `${on ? '' : '<span class="proj-mark">off</span>'}</span>`;
}

// ---------- Vision projector inventory ----------
// Projectors are deliberately not model cards, so GET /v1/mmproj is the only place they
// appear at all. Without it a projector attached to nothing is simply invisible — which
// is the exact failure mode the provenance fields were added to expose. Fetched on load
// and after a rescan, NOT per tick: the set only changes when the tree does.
let projectorsCache = [];

async function loadProjectors() {
  const el = $('proj-list');
  try {
    const body = await apiJSON('/v1/mmproj');
    projectorsCache = (body && body.projectors) || [];
    renderProjectors();
  } catch (e) {
    if (el) el.innerHTML = `<div class="empty">Could not load projectors: ${esc(e.message)}</div>`;
  }
}

function renderProjectors() {
  const el = $('proj-list');
  if (!el) return;
  const rows = projectorsCache;
  $('proj-count').textContent = rows.length
    ? ` (${rows.length}, ${rows.filter(p => !p.owner).length} unpaired)`
    : '';
  if (!rows.length) {
    el.innerHTML = '<div class="empty">No vision projectors found under the models root.</div>';
    return;
  }
  const nameOf = (id) => (modelsCache.find(m => m.id === id) || {}).name || id;
  el.innerHTML = rows.map(p => {
    const owner = p.owner
      ? `<span class="proj-owner" title="${esc(p.owner)}">${esc(nameOf(p.owner))}</span>`
      : '<span class="proj-owner none">unpaired</span>';
    const src = p.source === 'auto' ? '<span class="proj-src auto">inferred</span>'
              : p.source === 'explicit' ? '<span class="proj-src">sidecar</span>' : '';
    const on = p.owner
      ? (p.enabled ? '<span class="proj-on">on</span>' : '<span class="proj-off2">off</span>')
      : '';
    const dir = p.dir || '';
    return `<div class="proj-row" title="${esc(dir)}">`
      + `<span class="proj-file">${esc(p.name)}</span>`
      + `<span class="proj-size">${esc(fmtBytes(p.size_mb))}</span>`
      + owner + src + on
      + `</div>`;
  }).join('');
}

function tagsRowHTML(m) {
  // Type badge leads the row — it is the primary classification, ahead of user tags.
  // The engine badge sits next to it: both describe what the card IS rather than
  // metadata the user attached.
  return typeBadgeHTML(m)
    + engineBadgeHTML(m)
    + (m.tags||[]).map(t => `<span class="tag">${esc(t)}</span>`).join('')
    + (m.default ? '<span class="tag">default</span>' : '')
    + (m.pinned ? '<span class="tag">pinned</span>' : '')
    + (m.auto_save_state ? '<span class="tag">autosave</span>' : '')
    + mmprojTagHTML(m);
}

function updateLoadedBar() {
  const bar = $('loaded-bar');
  if (!bar) return;
  if (!modelsCache || modelsCache.length === 0) { bar.textContent = 'Loading…'; return; }
  const loaded = modelsCache.filter(m => m.loaded);
  if (loaded.length === 0) { bar.textContent = 'No model currently loaded.'; return; }
  bar.textContent = 'Loaded: ' + loaded.map(m => `${m.name || m.id} (port ${m.port || '?'})`).join(' · ');
}
function renderTypeChips() {
  const row = $('type-chips');
  if (!row) return;
  // Counts come from the FULL cache, not the filtered list, so a chip never vanishes
  // just because the text or status filter is currently hiding its models.
  const counts = new Map();
  for (const m of modelsCache) {
    const t = m.model_type || DEFAULT_TYPE;
    counts.set(t, (counts.get(t) || 0) + 1);
  }
  // Known types first in canonical order, then any type the server reports that this
  // build has no metadata for — a new backend type still gets a usable chip.
  const present = TYPE_ORDER.filter(t => counts.has(t));
  for (const t of [...counts.keys()].sort()) {
    if (TYPE_ORDER.indexOf(t) === -1) present.push(t);
  }

  const parts = [`<button class="type-chip${typeFilter === '' ? ' active' : ''}" data-type="" style="--type-color:#7c5cff">All<span class="count">${modelsCache.length}</span></button>`];
  for (const t of present) {
    const meta = typeMeta(t);
    parts.push(`<button class="type-chip${typeFilter === t ? ' active' : ''}" data-type="${esc(t)}" style="--type-color:${meta.color}" title="${esc(meta.desc || meta.label)}">`
      + `<span class="dot"></span>${esc(meta.label)}<span class="count">${counts.get(t)}</span></button>`);
  }
  row.innerHTML = parts.join('');
  row.querySelectorAll('.type-chip').forEach(btn => {
    btn.onclick = () => {
      const t = btn.dataset.type;
      typeFilter = (t === typeFilter) ? '' : t;   // clicking the active chip returns to All
      renderModels();
    };
  });
}

// Build or update a single card in place; returns the card element.
//
// This used to be two branches of a loop (build-new vs in-place-update), which is how
// the two drifted apart. Now there is ONE path: the card is created with empty
// placeholders once, and every field is filled on every render. A field can therefore
// never be added to one path and forgotten in the other.
function renderCard(m, pm) {
  // Extra-args display, computed once and shared by both render paths.
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
  const metaLine = `${esc(m.id)} · ${esc(metaMid)}${m.port ? ' · :' + esc(m.port) : ''}${m.pid ? ' · pid ' + esc(m.pid) : ''}`;

  let card = cardNodes.get(m.id);
  const isNew = !card;
  if (isNew) {
    card = document.createElement('div');
    card.className = 'model-card';
    card.dataset.id = m.id;
    card.innerHTML = `
      <div class="model-head">
        <div>
          <div class="model-name"></div>
          <div class="model-meta"></div>
          <div class="tags-row" style="margin-top:6px;"></div>
        </div>
        <div class="status-wrap"></div>
      </div>
      <div class="model-hint-slot"></div>
      <div class="args-box"></div>
      <div class="model-stats">
        <div data-stat="vram">VRAM: <b></b></div>
        <div data-stat="ram">RAM:  <b></b></div>
        <div data-stat="ctx">ctx / max: <b></b></div>
        <div data-stat="uptime">uptime: <b></b></div>
      </div>
      <div class="btn-row"></div>`;
    cardNodes.set(m.id, card);
    // Seed the button row so updateButtonRow's structure comparison matches on the
    // first pass (an empty row would compare as "no vision / no states / not loaded"
    // and skip the rebuild, leaving the card with no buttons at all).
    card.querySelector('.btn-row').innerHTML = buildButtonRowHTML(m);
  }

  const nameEl = card.querySelector('.model-name');
  if (nameEl) nameEl.textContent = m.name || m.id;
  const metaEl = card.querySelector('.model-meta');
  if (metaEl) metaEl.innerHTML = metaLine;
  const tagsRow = card.querySelector('.tags-row');
  if (tagsRow) tagsRow.innerHTML = tagsRowHTML(m);
  const statusWrap = card.querySelector('.status-wrap');
  if (statusWrap) statusWrap.innerHTML = statusBadge(m.loaded, m.ready);
  const hintSlot = card.querySelector('.model-hint-slot');
  if (hintSlot) hintSlot.innerHTML = typeHintHTML(m);
  const argsBox = card.querySelector('.args-box');
  if (argsBox) argsBox.textContent = argsDisplay || '(no extra args)';

  const statsDiv = card.querySelector('.model-stats');
  if (statsDiv) {
    const vramEl = statsDiv.querySelector('[data-stat="vram"]');
    if (vramEl) vramEl.innerHTML = `VRAM: <b>${pm.vram_mb ? fmtBytes(pm.vram_mb) : '—'}</b>`;
    const ramEl = statsDiv.querySelector('[data-stat="ram"]');
    if (ramEl) ramEl.innerHTML = `RAM:  <b>${pm.ram_mb ? fmtBytes(pm.ram_mb) : '—'}</b>`;
    // Only llama reports ctx; the row is meaningless for a bridge engine, and once
    // removed ctxEl is null on later renders — so the removal is idempotent.
    const ctxEl = statsDiv.querySelector('[data-stat="ctx"]');
    if (m.engine !== 'llama') { if (ctxEl) ctxEl.remove(); }
    else if (ctxEl) ctxEl.innerHTML = `ctx / max: <b>${m.ctx_size != null ? m.ctx_size : '—'}${m.max_ctx_size ? ' / ' + m.max_ctx_size.toLocaleString() : ''}</b>`;
    const uptimeEl = statsDiv.querySelector('[data-stat="uptime"]');
    if (uptimeEl) uptimeEl.innerHTML = `uptime: <b>${m.loaded && statusCache ? formatUptime(m.id) : '—'}</b>`;
  }

  updateButtonRow(card, m);

  // Attach/update button click listeners (idempotent)
  card.querySelectorAll('button[data-act]').forEach(btn => {
    btn.onclick = () => handleAction(m.id, btn.dataset.act, btn);
  });
  return card;
}

function renderModels() {
  updateLoadedBar();
  renderTypeChips();
  const container = $('models-container');
  const emptySlot = $('models-grid');   // holds the "nothing matches" message
  if (!container || !emptySlot) return;

  const filter = ($('filter') ? $('filter').value : '').toLowerCase();
  const statusFilter = $('status-filter') ? $('status-filter').value : 'all';
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

  if (typeFilter) {
    list = list.filter(m => (m.model_type || DEFAULT_TYPE) === typeFilter);
  }

  // Retire cards that are no longer visible before deciding the empty state.
  const idsNeeded = new Set(list.map(m => m.id));
  cardNodes.forEach((node, id) => {
    if (!idsNeeded.has(id)) { node.remove(); cardNodes.delete(id); }
  });

  if (list.length === 0) {
    groupNodes.forEach(g => g.section.remove());
    groupNodes.clear();
    const hasFilter = filter || statusFilter !== 'all' || typeFilter;
    emptySlot.innerHTML = `<div class="empty">${hasFilter ? 'No models match your filter.' : 'No models found.'}</div>`;
    return;
  }
  emptySlot.innerHTML = '';

  // Group by type; canonical order first so sections never jump around between renders.
  const byType = new Map();
  for (const m of list) {
    const t = m.model_type || DEFAULT_TYPE;
    if (!byType.has(t)) byType.set(t, []);
    byType.get(t).push(m);
  }
  const types = [...byType.keys()].sort((a, b) =>
    typeOrderIndex(a) - typeOrderIndex(b) || a.localeCompare(b));

  // Retire group sections for types with nothing left to show.
  groupNodes.forEach((g, t) => {
    if (!byType.has(t)) { g.section.remove(); groupNodes.delete(t); }
  });

  const perModel = (statusCache && statusCache.per_model) || {};

  for (const t of types) {
    let g = groupNodes.get(t);
    if (!g) {
      const section = document.createElement('section');
      section.className = 'model-group';
      section.dataset.type = t;
      section.innerHTML = `<div class="group-head"><span class="dot"></span>`
        + `<span class="group-title"></span><span class="group-count"></span>`
        + `<span class="group-desc"></span></div><div class="group-grid"></div>`;
      section.querySelector('.group-title').textContent = typeMeta(t).label;
      g = {
        section,
        grid: section.querySelector('.group-grid'),
        countEl: section.querySelector('.group-count'),
        descEl: section.querySelector('.group-desc'),
      };
      groupNodes.set(t, g);
    }
    const meta = typeMeta(t);
    g.section.style.setProperty('--type-color', meta.color);
    g.descEl.textContent = meta.desc || '';
    g.countEl.textContent = `${byType.get(t).length}`;

    for (const m of byType.get(t)) {
      // appendChild MOVES an existing node, so this inserts new cards and re-orders the
      // existing ones without rebuilding anything (important: the WS pushes ~1 Hz).
      g.grid.appendChild(renderCard(m, perModel[m.id] || {}));
    }
    container.appendChild(g.section);
  }
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
      await loadEngines();   // a deleted card may have been an engine's last user
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

// Explain the current classification under the model-type select: what was detected, from
// what, and what picking a value will do. An empty note leaves the row collapsed.
function updateModelTypeNote(m) {
  const note = $('f-model-type-note');
  if (!note) return;
  const detected = m.model_type || DEFAULT_TYPE;
  const dLabel = TYPE_META[detected] ? TYPE_META[detected].label : detected;
  if (m.model_type_override) {
    note.textContent = `Pinned to ${TYPE_META[m.model_type_override] ? TYPE_META[m.model_type_override].label : m.model_type_override}`
      + ` (auto-detection said ${dLabel}). Choose "Auto-detect" to go back.`;
  } else {
    const src = m.arch ? `from architecture "${m.arch}"` : 'from the filename';
    note.textContent = `Auto-detected: ${dLabel} ${src}. Pick a value to override.`;
  }
}

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
  // Model type: "Auto-detect" (empty value) plus every known type. The option list is
  // rebuilt from the same TYPE_ORDER the chips use so the two can never disagree.
  const typeSel = $('f-model-type');
  if (typeSel) {
    typeSel.innerHTML = '<option value="">Auto-detect</option>'
      + TYPE_ORDER.map(t => `<option value="${esc(t)}">${esc(TYPE_META[t] ? TYPE_META[t].label : t)}</option>`).join('');
    typeSel.value = m.model_type_override || '';
    updateModelTypeNote(m);
  }
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
  // "" means "reset to auto-detection", which the backend re-derives immediately so the
  // card lands back in its detected group on the very next render.
  const mtSel = $('f-model-type');
  if (mtSel) body.model_type = mtSel.value || '';

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
      // Vision provenance: mmproj_file is only the CONFIGURED name (empty when the link
      // was inferred); mmproj_resolved is the file that actually reaches --mmproj, and
      // mmproj_source says whether we inferred it or the user set it.
      mmproj_resolved: d.mmproj_resolved || '',
      mmproj_source: d.mmproj_source || '',
      ctx_size: d.ctx_size,
      max_ctx_size: d.max_ctx_size,
      n_gpu_layers: d.n_gpu_layers,
      estimated_vram_mb: d.estimated_vram_mb,
      args: d.args,
      gguf_name: d.gguf_name,
      // Classification (backend-sourced). model_type drives grouping/badge/filter chips,
      // model_type_override is the raw sidecar value (empty = auto-detected), arch is the
      // raw general.architecture shown in the edit modal.
      model_type: d.model_type || 'text',
      model_type_override: d.model_type_override || '',
      model_type_label: d.model_type_label || '',
      arch: d.arch || '',
      auxiliary: !!d.auxiliary,
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
// Engine registry metadata for the card badges. Deliberately NOT part of refreshAll():
// it is static config, and refreshAll runs on every WebSocket tick.
// Returns the engine list, or null when the fetch itself failed (callers must tell that
// apart from "no engines configured" — the two need different messages).
async function loadEngines() {
  try {
    const engines = await apiJSON('/v1/engines');
    const next = {};
    (engines || []).forEach(en => { next[en.name] = en; });
    enginesCache = next;
    return engines || [];
  } catch (e) {
    // Keep the previous cache: a transient failure should not blank the badges on every
    // card. The badge falls back to the raw engine name meanwhile.
    return null;
  }
}

async function loadEngineOptions() {
  const sel = $('ea-engine');
  $('ea-error-row').style.display = 'none';  // cleared up front; set below if needed
  const engines = await loadEngines();
  if (engines === null) {
    sel.innerHTML = '<option value="">(could not load engines)</option>';
    $('ea-error-row').style.display = 'block';
    $('ea-error').textContent = 'Could not load engines — is the server still running?';
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
    // Re-read engine metadata too: a newly added card may be the first user of an engine
    // that was missing from the registry (badge would otherwise warn until next reload).
    await loadEngines();
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
    await loadProjectors();
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
// Engine metadata first, so the very first card paint already has correct engine badges
// instead of briefly showing raw engine names.
// Projector names are resolved server-side; the inventory panel links them back to the
// card they belong to, so a pairing is navigable in both directions.
$('proj-panel').addEventListener('toggle', (ev) => {
  if (ev.target.open && !projectorsCache.length) loadProjectors();
});
loadEngines().then(refreshAll).then(loadProjectors).then(wsLoop);
