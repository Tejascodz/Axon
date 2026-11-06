"""A small web page for asking the index questions: GET / on the server (AXON_UI=0 turns it off).

The page holds no data. The user pastes their API key, which stays in this browser tab only
(sessionStorage, gone when the tab closes) and is sent as a Bearer token to /ask like any other
client. Answers and sources are shown as text, never as HTML, and the page is served with a strict
Content-Security-Policy (no inline script, no third-party content, no framing).
"""

CSP = ("default-src 'none'; script-src 'self'; style-src 'self'; connect-src 'self'; img-src 'self'; "
       "base-uri 'none'; form-action 'none'; frame-ancestors 'none'")

HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="referrer" content="no-referrer">
<title>Axon</title>
<link rel="stylesheet" href="/ui.css">
</head>
<body>
<header><h1>Axon</h1><p>Answers from your documents, with the passages they come from.</p></header>
<main>
  <section id="keybox">
    <label for="key">API key</label>
    <div class="row"><input id="key" type="password" autocomplete="off" spellcheck="false">
    <button id="savekey" type="button">Use key</button></div>
    <p class="hint">Kept in this tab only and sent to this server only.</p>
  </section>
  <section id="askbox" hidden>
    <label for="q">Question</label>
    <textarea id="q" rows="3" maxlength="2000" placeholder="Ask about your documents"></textarea>
    <div class="row"><button id="ask" type="button">Ask</button>
    <button id="forget" type="button" class="quiet">Forget key</button><span id="status" role="status"></span></div>
  </section>
  <section id="result" hidden>
    <h2>Answer</h2>
    <p id="answer"></p>
    <p id="abstained" class="note" hidden>No answer: the documents found do not support one.</p>
    <h2 id="srchead">Sources</h2>
    <ol id="sources"></ol>
    <details id="withheldbox" hidden><summary>Passages withheld</summary><ul id="withheld"></ul></details>
    <p id="rid" class="hint"></p>
  </section>
</main>
<script src="/ui.js"></script>
</body>
</html>
"""

CSS = """:root{--bg:#fafaf8;--fg:#1d1d1b;--muted:#6b6b66;--line:#deded8;--accent:#2d5be3;--on-accent:#fff;--card:#fff}
@media (prefers-color-scheme:dark){:root{--bg:#161615;--fg:#ecece8;--muted:#a0a09a;--line:#34342f;--accent:#7c9bff;--on-accent:#0d1020;--card:#1f1f1d}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:16px/1.5 system-ui,sans-serif}
header,main{max-width:760px;margin:0 auto;padding:16px}h1{margin:16px 0 0;font-size:24px}h2{font-size:17px;margin:24px 0 8px}
header p,.hint,.note{color:var(--muted)}.hint{font-size:13px}label{display:block;font-weight:600;margin:16px 0 6px}
input,textarea{width:100%;padding:10px;border:1px solid var(--line);border-radius:8px;background:var(--card);color:var(--fg);font:inherit}
.row{display:flex;gap:8px;align-items:center;margin-top:8px;flex-wrap:wrap}.row input{flex:1}
button{padding:9px 16px;border:0;border-radius:8px;background:var(--accent);color:var(--on-accent);font:inherit;cursor:pointer}
button.quiet{background:transparent;color:var(--muted);border:1px solid var(--line)}button:disabled{opacity:.5;cursor:wait}
#answer{white-space:pre-wrap;font-size:17px}ol{padding-left:22px}li{margin-bottom:12px}
.src{font-weight:600}.snip{color:var(--muted);font-size:14px;white-space:pre-wrap;overflow-wrap:anywhere}
#status{color:var(--muted);font-size:14px}details{margin-top:16px;color:var(--muted)}
"""

JS = """'use strict';
const $ = (id) => document.getElementById(id);
const store = { get() { try { return sessionStorage.getItem('axon_key') || ''; } catch (e) { return mem; } },
                set(v) { mem = v; try { v ? sessionStorage.setItem('axon_key', v) : sessionStorage.removeItem('axon_key'); } catch (e) {} } };
let mem = '';
function show() { const has = !!store.get(); $('keybox').hidden = has; $('askbox').hidden = !has; if (has) $('q').focus(); }
function text(el, s) { el.textContent = s == null ? '' : String(s); return el; }
function item(tag, cls, s) { const e = document.createElement(tag); if (cls) e.className = cls; return text(e, s); }
$('savekey').addEventListener('click', () => { const v = $('key').value.trim(); if (v) { store.set(v); $('key').value = ''; show(); } });
$('key').addEventListener('keydown', (e) => { if (e.key === 'Enter') $('savekey').click(); });
$('forget').addEventListener('click', () => { store.set(''); $('result').hidden = true; show(); });
$('q').addEventListener('keydown', (e) => { if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) $('ask').click(); });
$('ask').addEventListener('click', async () => {
  const q = $('q').value.trim(); if (!q) return;
  $('ask').disabled = true; text($('status'), 'Searching and answering...');
  try {
    const r = await fetch('/ask', { method: 'POST', credentials: 'omit', cache: 'no-store',
      headers: { 'Content-Type': 'application/json', 'Authorization': 'Bearer ' + store.get() },
      body: JSON.stringify({ question: q }) });
    let out = {}; try { out = await r.json(); } catch (e) {}
    if (r.status === 401) { store.set(''); show(); text($('status'), 'That key was not accepted.'); return; }
    if (!r.ok) { text($('status'), (out.detail || ('Error ' + r.status)) + (r.status === 503 || r.status === 429 ? ' Try again shortly.' : '')); return; }
    render(out); text($('status'), '');
  } catch (e) { text($('status'), 'Could not reach the server.'); }
  finally { $('ask').disabled = false; }
});
function render(out) {
  $('result').hidden = false;
  text($('answer'), out.abstained ? '' : out.answer);
  $('abstained').hidden = !out.abstained;
  const ol = $('sources'); ol.replaceChildren();
  (out.sources || []).forEach((s) => {
    const li = document.createElement('li');
    li.append(item('div', 'src', (s.source || s.id) + '  [' + s.id + ']'));
    const t = String(s.text || ''); li.append(item('div', 'snip', t.length > 600 ? t.slice(0, 600) + '...' : t));
    ol.append(li);
  });
  $('srchead').hidden = !(out.sources || []).length;
  const w = out.withheld || []; $('withheldbox').hidden = !w.length; const ul = $('withheld'); ul.replaceChildren();
  w.forEach((x) => ul.append(item('li', '', (x.title || x.id) + ': ' + (x.flags || []).join(', '))));
  text($('rid'), out.request_id ? 'Request ' + out.request_id : '');
}
show();
"""
