---
tags: [webui, omni-chat, streaming-sse, tts, capability-gating]
aliases: [omni-chat-webui-panel, chatjs-panel]
related: [[capability-flags-omni-derivation]], [[autoloader-webui-models-whitelist-drops-unknown-fields]], [[llamacpp-multimodal-content-part-keys]], [[llamacpp-tts-body-key-is-input-not-text]]
confidence: verified
author: coder_omni-chat-build_20261029_113936.jsonl
contributors:
  - coder_omni-chat-build_20261029_123833.jsonl
---

# Omni Chat WebUI panel (static/chat.js)

A minimal "Chat" panel in the llama-autoloader WebUI for poking loaded llama-server models.
Frontend-only, vanilla JS, no deps, no build step. Files: new `static/chat.js`; small edits to
`static/app.js` (Chat button + handleAction case), `static/index.html` (`#chat-overlay` modal +
`<script src="/static/chat.js">` BEFORE app.js), `static/styles.css` (`.chat-*` block).

## Integration (why it is cheap)
- Chat button is `data-act="chat"`, gated on `m.engine === 'llama'` in `buildButtonRowHTML`;
  `handleAction` adds `if (act === 'chat') openChatModal(id);`. Click dispatch is already generic
  (`app.js` `button[data-act]` → `handleAction`), so no new listener wiring.
- Capability gating reads ONLY the backend flags `m.multimodal` (attach row) / `m.audio` (Speak
  button). These reach the browser because the `/v1/models` whitelist in `app.js` already maps them
  (see [[autoloader-webui-models-whitelist-drops-unknown-fields]] — that gap was closed earlier).
- `chat.js` loads BEFORE `app.js` but references app.js globals (`modelsCache`, `apiJSON`,
  `refreshAll`, `showToast`) only INSIDE function bodies (run on click), never at load time —
  top-level `const $` in app.js is in TDZ when chat.js runs, so chat.js defines its own `$c`.

## Verified request shapes (live-tested against LFM3-Omni on :1234)
- Chat: `POST /v1/chat/completions` `{model, messages, stream:true, max_tokens:512}`. Content parts:
  `{type:'text',text}`, `{type:'image_url',image_url:{url:'data:<mime>;base64,...'}}`,
  `{type:'input_audio'|'input_video', input_audio/input_video:{data:'<raw base64, no data: prefix>'}}`.
  Response is SSE `text/event-stream`, frames `data: <json>\n\n`, terminator `data: [DONE]\n\n`.
  chat.js parses via `fetch → res.body.getReader() → TextDecoder → split on "\n\n"` keeping a residual
  buffer across reads (a frame can split across TCP reads). Result: streamed answer `'OK'`; image described
  as `'Red'`.
- TTS: `POST /tts` `{model, input}` — utterance key is `input` (NOT `text`); `model` is required for the
  forwarder to pick the port. Response is raw `audio/wav` (verified RIFF, 38444 bytes). Play via
  `URL.createObjectURL(blob)` → `new Audio`. See [[llamacpp-tts-body-key-is-input-not-text]].
- Errors from the forwarder/llama.cpp are JSON `{"error": ...}` — the UI parses `.error`, falls back to raw body.

## Gotchas
- The running :1234 process reported `multimodal=False, audio=False` for LFM3 (stale vs the capability-derivation
  fix in server.py source) — so the attach/Speak gates were correctly HIDDEN for LFM3 in the live test. The
  positive gate (attach/Speak showing) was NOT exercised because the only `multimodal=True` models (Qwen3-VL-4B)
  failed to start (VRAM). Gate logic is trivially correct (reads the plumbed flag); see
  [[capability-flags-omni-derivation]] addendum.
- **Superseded:** `sendMessage` used to guard re-entrancy with `if ($c('chat-send').disabled) return;` (the button
  as the in-flight marker). See the addendum below — this is now a dedicated boolean.
  `speakAssistant` uses a `chatTtsInFlight` flag + re-checks `chatModelId` after the await so closing the modal
  mid-fetch does not play audio or leak an object URL.
- Model-not-loaded: the modal shows a "Load model" button (reuses `POST /v1/models/{id}/load`), reveals the chat
  body only after a confirmed `m.loaded`. Capability flags are immutable per card, so they are not re-read after load.

## Addendum — chat.js state model (reviewed PASS-with-minors, applied) @ b81c6b52

The DOM control is no longer the in-flight flag. `chat.js` now keeps two module-level booleans as the
**source of truth** (the disabled button is only a visual consequence):

- `chatSendInFlight` — set `true` as the first statement inside `sendMessage`'s `try` (still before the
  `await fetch`, and there is no `await` between the re-entry guard and the set, so the guard stays race-free
  under JS single-threading). Re-entry guard is `if (chatSendInFlight) return;`. Cleared in `finally`, which
  covers every exit path (the `!res.ok` early `return` inside the try, and fetch/reader exceptions).
- `chatClosed` — set `true` as the FIRST statement of `closeChatModal()`. The read loop checks it at the top
  (`if (chatClosed) { await reader.cancel(); break; }`) to abort the streaming reader, and the final
  `chatMessages.push({role:'assistant',...})` is guarded with `!chatClosed` so a closed conversation never
  resurfaces a late message on re-open. `openChatModal()` resets both flags to `false` for a fresh conversation.

SSE loop changes (verified by simulating chunk boundaries — content frame always precedes `[DONE]`):
- `[DONE]` now does `break streamLoop;` (a label on the outer `while(true)`) instead of `continue` — the stream
  ends at `[DONE]` instead of waiting for the server to close. This exits BOTH the inner frame loop and the outer
  read loop (labeled break).
- There is NO standalone `buf += decoder.decode();` flush after the loop (it was inert — never re-scanned).
- Parsing order proof: the frame-split loop consumes complete `\n\n` blocks in order, so the last content frame
  is sliced and appended to `acc` in an earlier inner-loop iteration than the `[DONE]` frame, even when both
  arrive in the same `read()` chunk or the `\n\n` delimiter straddles a chunk boundary. Break-on-`[DONE]` drops
  nothing.

**Known non-blocking limitation (documented, NOT fixed):** a close→reopen within a single `await reader.read()`
cycle flips `chatClosed` back to `false` before the running stream observes it, so the old stream can still push
its late `acc` into the freshly-cleared `chatMessages`. Fixing this needs a per-send token captured in the
closure rather than the shared `chatClosed` flag; deemed out of scope for a testing poke-tool. Related:
[[attribute-cited-line-to-production-tree-not-scratch-copy]] (verify line numbers against the live tree — the
reviewer's refs 249/260/191/216/268 had already drifted; the inert-flush line was already absent).
