"use strict";
const $ = (id) => document.getElementById(id);
const chunks = new Map();
let docId = null;

function el(tag, text, cls) {
  const n = document.createElement(tag);
  if (text) n.textContent = text; // textContent only: document text is never parsed as HTML
  if (cls) n.className = cls;
  return n;
}

function cites(ids) {
  const box = el("span");
  for (const id of ids) {
    const c = chunks.get(id);
    if (!c) continue;
    const b = el("button", `${c.label}, page ${c.page}`, "chip");
    b.type = "button";
    b.setAttribute("aria-label", `Show source text for ${c.label}, page ${c.page}`);
    b.addEventListener("click", () => showSource(id, true));
    box.append(b);
  }
  return box;
}

function showSource(id, focus) {
  const c = chunks.get(id);
  $("source-meta").textContent = `${c.label}, page ${c.page}`;
  $("source-text").replaceChildren(el("mark", c.text));
  if (focus) $("source").focus();
}

function li(text, ids = []) {
  const n = el("li", text);
  n.append(" ", cites(ids));
  return n;
}

function fill(id, items, build) {
  const box = $(id);
  box.replaceChildren(...items.map(build));
  box.parentElement.hidden = items.length === 0;
}

function render(d) {
  const a = d.analysis;
  chunks.clear();
  d.chunks.forEach((c) => chunks.set(c.id, c));
  $("doc-type").textContent = a.document_type || "Document";
  $("summary").textContent = a.summary;
  fill("parties", a.parties, (p) => li(p.role ? `${p.name}, ${p.role}` : p.name));
  fill("facts", a.key_facts, (f) => li(`${f.label}: ${f.detail}`, f.sources));
  fill("obligations", a.obligations, (o) => li(o.detail, o.sources));
  fill("risks", a.risks, (r) => li(`${r.issue}. ${r.why}`, r.sources));
  $("answer").replaceChildren();
  $("results").hidden = false;
  $("doc-type").focus();
}

async function call(url, opts) {
  const r = await fetch(url, opts);
  const body = await r.json().catch(() => ({}));
  if (!r.ok) {
    throw new Error(typeof body.detail === "string" ? body.detail : "Something went wrong. Check your input and try again.");
  }
  return body;
}

$("upload-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const btn = $("upload-btn");
  const form = new FormData();
  form.append("file", $("file").files[0]);
  form.append("language", $("lang").value);
  btn.disabled = true;
  $("error").textContent = "";
  $("status").textContent = "Reading your document. This can take up to a minute.";
  try {
    const d = await call("/api/documents", { method: "POST", body: form });
    docId = d.doc_id;
    render(d);
    $("results").lang = $("lang").value;
    $("status").textContent = `Done: ${d.filename}`;
  } catch (err) {
    $("status").textContent = "";
    $("error").textContent = err.message;
  } finally {
    btn.disabled = false;
  }
});

$("ask-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const btn = $("ask-btn"), out = $("answer");
  btn.disabled = true;
  out.textContent = "Looking through the document.";
  try {
    const r = await call(`/api/documents/${docId}/ask`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question: $("question").value, language: $("lang").value }),
    });
    out.className = r.found ? "" : "not-found";
    out.replaceChildren(el("p", r.answer), cites(r.citations.map((c) => c.id)), el("br"), el("small", r.disclaimer));
    if (r.found) showSource(r.citations[0].id, false);
  } catch (err) {
    out.className = "not-found";
    out.textContent = err.message;
  } finally {
    btn.disabled = false;
  }
});
