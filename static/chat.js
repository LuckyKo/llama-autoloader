/* Omni chat panel — a testing tool for poking loaded llama-server models.
 *
 * Bound to one model at a time (the card the Chat button was clicked on).
 * Loads before app.js; the functions here only run on click, by which time
 * app.js has populated the shared global scope (modelsCache, apiJSON,
 * refreshAll, showToast).
 *
 * Capability gating is driven ONLY by the backend flags on the model card:
 *   m.multimodal -> show the file-attach row
 *   m.audio      -> show a "Speak" (TTS) button on each assistant turn
 *
 * No framework, no build step, no new dependencies. Plain classic script.
 */

// Per-open conversation state. Cleared on close; nothing is persisted.
let chatModelId = null;
let chatMultimodal = false;   // show attach row
let chatAudioCapable = false; // show Speak buttons
let chatMessages = [];        // {role, parts, media?} — parts are the API content-part shape
let pendingFile = null;       // {kind:'image'|'audio'|'video', dataUrl, name}
let chatAudio = null;         // current <Audio> element (so close() can stop it)
let chatTtsInFlight = false;  // guards against overlapping Speak clicks
// Source of truth for an in-flight send; the disabled button is only a visual consequence.
let chatSendInFlight = false;
// Set by closeChatModal(); aborts an in-flight reader and suppresses a late assistant push.
let chatClosed = false;

const $c = (id) => document.getElementById(id);

// ---- open / close -------------------------------------------------------

function openChatModal(id) {
  const m = modelsCache.find(x => x.id === id);
  if (!m) return;

  chatModelId = id;
  chatMultimodal = !!m.multimodal;
  chatAudioCapable = !!m.audio;
  chatMessages = [];
  pendingFile = null;
  chatSendInFlight = false;
  chatClosed = false;
  stopChatAudio();

  $c('chat-model-id').textContent = id;
  $c('chat-file').value = '';
  $c('chat-file-name').textContent = '';
  $c('chat-input').value = '';
  $c('chat-send').disabled = false;

  // Capability gates + loaded state.
  $c('chat-attach-row').style.display = chatMultimodal ? 'flex' : 'none';
  const loaded = !!m.loaded;
  $c('chat-load-row').style.display = loaded ? 'none' : 'block';
  $c('chat-body').style.display = loaded ? 'block' : 'none';

  renderChatMessages();
  $c('chat-overlay').classList.add('active');
}

function closeChatModal() {
  chatClosed = true;       // signal any in-flight send to stop reading and skip its late push
  chatModelId = null;
  chatMessages = [];
  pendingFile = null;
  stopChatAudio();
  $c('chat-overlay').classList.remove('active');
}

// Stop playback and release the object URL so a big WAV does not stay resident.
function stopChatAudio() {
  if (chatAudio) {
    chatAudio.pause();
    if (chatAudio.src && chatAudio.src.startsWith('blob:')) URL.revokeObjectURL(chatAudio.src);
    chatAudio = null;
  }
}

// ---- message rendering --------------------------------------------------

function renderChatMessages() {
  const list = $c('chat-messages');
  list.innerHTML = '';
  for (let i = 0; i < chatMessages.length; i++) list.appendChild(renderBubble(i));
  list.scrollTop = list.scrollHeight;
}

function renderBubble(i) {
  const msg = chatMessages[i];
  const wrap = document.createElement('div');
  wrap.className = 'chat-msg ' + msg.role;

  const role = document.createElement('div');
  role.className = 'chat-msg-role';
  role.textContent = msg.role === 'user' ? 'You' : 'Assistant';
  wrap.appendChild(role);

  const body = document.createElement('div');
  body.className = 'chat-msg-body';
  if (msg.role === 'user') {
    for (const p of msg.parts) {
      if (p.type === 'text') {
        const t = document.createElement('span');
        t.textContent = p.text;
        body.appendChild(t);
      } else if (p.type === 'image_url') {
        const img = document.createElement('img');
        img.className = 'chat-thumb';
        img.src = p.image_url.url;
        img.alt = 'attachment';
        body.appendChild(img);
      } else {
        const chip = document.createElement('span');
        chip.className = 'chat-chip';
        chip.textContent = (p.type === 'input_audio' ? '🎵 ' : '🎬 ') +
          (msg.media ? msg.media.name : p.type);
        body.appendChild(chip);
      }
    }
  } else {
    const t = document.createElement('span');
    t.textContent = msg.parts.map(p => p.text).join('');
    body.appendChild(t);
  }
  wrap.appendChild(body);

  // Speak toggle: assistant turns only, and only when the model is flagged audio.
  if (msg.role === 'assistant' && chatAudioCapable) {
    const btn = document.createElement('button');
    btn.className = 'chat-speak';
    btn.textContent = '🔊 Speak';
    const text = msg.parts.map(p => p.text).join('');
    if (text) btn.onclick = () => speakAssistant(text);
    wrap.appendChild(btn);
  }
  return wrap;
}

function addErrorBubble(msg) {
  const list = $c('chat-messages');
  const el = document.createElement('div');
  el.className = 'chat-msg chat-error';
  el.textContent = '⚠ ' + msg;
  list.appendChild(el);
  list.scrollTop = list.scrollHeight;
}

// ---- file attach --------------------------------------------------------

function onFileChange(e) {
  const f = e.target.files && e.target.files[0];
  if (!f) return;
  const reader = new FileReader();
  reader.onload = () => {
    pendingFile = { kind: kindFromMime(f.type), dataUrl: reader.result, name: f.name };
    $c('chat-file-name').textContent = f.name;
  };
  reader.readAsDataURL(f);
}

function kindFromMime(mime) {
  if (mime.startsWith('audio/')) return 'audio';
  if (mime.startsWith('video/')) return 'video';
  return 'image';
}

// input_audio / input_video take raw base64 (no "data:<mime>;base64," prefix);
// image_url takes the full data-URI. Strip the prefix for the audio/video .data field.
function stripDataPrefix(dataUrl) {
  const i = dataUrl.indexOf(',');
  return i === -1 ? dataUrl : dataUrl.slice(i + 1);
}

// ---- send (streamed) ----------------------------------------------------

function buildUserParts(text) {
  const parts = [];
  if (text) parts.push({ type: 'text', text });
  if (pendingFile) {
    if (pendingFile.kind === 'image') {
      parts.push({ type: 'image_url', image_url: { url: pendingFile.dataUrl } });
    } else if (pendingFile.kind === 'audio') {
      parts.push({ type: 'input_audio', input_audio: { data: stripDataPrefix(pendingFile.dataUrl) } });
    } else {
      parts.push({ type: 'input_video', input_video: { data: stripDataPrefix(pendingFile.dataUrl) } });
    }
  }
  return parts;
}

function toApiMsg(msg) {
  if (msg.role === 'assistant') return { role: 'assistant', content: msg.parts.map(p => p.text).join('') };
  // Text-only user turn -> send plain string content (max compatibility with non-llama
  // engines like strata/ninfer that may not accept the typed content-parts array).
  // Use the typed parts array only when real media is attached (only for m.multimodal models).
  const media = msg.parts.filter(p => p.type !== 'text');
  if (media.length === 0) {
    const text = msg.parts.filter(p => p.type === 'text').map(p => p.text).join('');
    return { role: 'user', content: text };
  }
  return { role: 'user', content: msg.parts };
}

async function sendMessage() {
  if (!chatModelId) return;
  if (chatSendInFlight) return;  // a stream is already in flight
  const text = $c('chat-input').value.trim();
  if (!text && !pendingFile) return;

  chatMessages.push({
    role: 'user',
    parts: buildUserParts(text),
    media: pendingFile ? { kind: pendingFile.kind, name: pendingFile.name } : null,
  });
  $c('chat-input').value = '';
  $c('chat-file').value = '';
  $c('chat-file-name').textContent = '';
  pendingFile = null;
  renderChatMessages();

  // Live streaming bubble (replaced by the final rendered bubble on completion).
  const list = $c('chat-messages');
  const streamEl = document.createElement('div');
  streamEl.className = 'chat-msg assistant';
  const streamBody = document.createElement('div');
  streamBody.className = 'chat-msg-body';
  streamEl.appendChild(streamBody);
  list.appendChild(streamEl);
  list.scrollTop = list.scrollHeight;

  let acc = '';
  try {
    chatSendInFlight = true;
    $c('chat-send').disabled = true;  // visual consequence of the in-flight flag
    const res = await fetch('/v1/chat/completions', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        model: chatModelId,
        messages: chatMessages.map(toApiMsg),
        stream: true,
        max_tokens: 512,  // deliberate cap: this is a poke-tool, not a full generation client
      }),
    });
    if (!res.ok) {
      streamEl.remove();
      addErrorBubble(await parseHttpError(res));
      return;
    }
    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buf = '';
    streamLoop:
    while (true) {
      if (chatClosed) { await reader.cancel(); break; }  // modal closed mid-stream: stop reading
      const { done, value } = await reader.read();
      if (done) break;
      buf += decoder.decode(value, { stream: true });
      // SSE frames are delimited by a blank line; a frame may split across reads,
      // so only consume complete "\n\n" blocks and keep the residual in buf.
      let nl;
      while ((nl = buf.indexOf('\n\n')) !== -1) {
        const frame = buf.slice(0, nl);
        buf = buf.slice(nl + 2);
        if (!frame.startsWith('data:')) continue;
        const data = frame.slice(5).trim();
        if (data === '[DONE]') break streamLoop;
        let j;
        try { j = JSON.parse(data); } catch (e) { continue; }
        const delta = j.choices && j.choices[0] && j.choices[0].delta;
        if (delta && typeof delta.content === 'string') {
          acc += delta.content;
          streamBody.textContent = acc;
          list.scrollTop = list.scrollHeight;
        }
      }
    }
    streamEl.remove();
    if (acc && !chatClosed) chatMessages.push({ role: 'assistant', parts: [{ type: 'text', text: acc }] });
    renderChatMessages();
  } catch (e) {
    streamEl.remove();
    addErrorBubble(e.message);
  } finally {
    chatSendInFlight = false;
    $c('chat-send').disabled = false;
  }
}

// Both the forwarder and llama.cpp return errors as JSON {"error": ...}, not plain text.
async function parseHttpError(res) {
  const t = await res.text();
  try {
    const j = JSON.parse(t);
    if (j && j.error) return String(j.error);
  } catch (e) { /* fall through to the raw body */ }
  return `HTTP ${res.status}` + (t ? ': ' + t.slice(0, 200) : '');
}

// ---- TTS ("Speak") ------------------------------------------------------

async function speakAssistant(text) {
  if (!chatModelId || !text || chatTtsInFlight) return;
  chatTtsInFlight = true;
  try {
    const res = await fetch('/tts', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      // "input" is the utterance key (NOT "text"); "model" is required for the
      // forwarder to route to the right llama-server port.
      body: JSON.stringify({ model: chatModelId, input: text }),
    });
    if (!res.ok) {
      showToast('Speak failed: ' + await parseHttpError(res), 'error');
      return;
    }
    const blob = await res.blob();
    if (!chatModelId) return;  // modal closed while the fetch was in flight
    stopChatAudio();
    chatAudio = new Audio(URL.createObjectURL(blob));
    chatAudio.play().catch(() => {});
  } catch (e) {
    showToast('Speak failed: ' + e.message, 'error');
  } finally {
    chatTtsInFlight = false;
  }
}

// ---- wiring (elements exist in index.html before this script runs) ------

$c('chat-close').onclick = closeChatModal;
$c('chat-send').onclick = sendMessage;
$c('chat-file').onchange = onFileChange;
$c('chat-load-btn').onclick = async () => {
  const btn = $c('chat-load-btn');
  btn.disabled = true;
  btn.textContent = 'Loading…';
  try {
    await apiJSON(`/v1/models/${encodeURIComponent(chatModelId)}/load`, { method: 'POST' });
    await refreshAll();
    const m = modelsCache.find(x => x.id === chatModelId);
    if (m && m.loaded) {
      $c('chat-load-row').style.display = 'none';
      $c('chat-body').style.display = 'block';
    } else {
      showToast('Model not ready yet — try again', 'info');
    }
  } catch (e) {
    showToast('Load failed: ' + e.message, 'error');
  } finally {
    btn.disabled = false;
    btn.textContent = 'Load model';
  }
};
$c('chat-input').addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    sendMessage();
  }
});
document.addEventListener('keydown', (e) => {
  if (e.key === 'Escape' && $c('chat-overlay').classList.contains('active')) closeChatModal();
});
