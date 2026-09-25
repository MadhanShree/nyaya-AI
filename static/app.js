let documentId = null;
const $ = (id) => document.getElementById(id);

function setStatus(message, error = false) {
  $("uploadStatus").textContent = message;
  $("uploadStatus").style.color = error ? "#b42318" : "";
}

async function upload() {
  const file = $("file").files[0];
  if (!file) return setStatus("Please choose a document first.", true);
  const form = new FormData();
  form.append("file", file);
  form.append("language", $("language").value);
  setStatus("Uploading and extracting text...");
  $("uploadBtn").disabled = true;
  try {
    const res = await fetch("/api/upload", { method: "POST", body: form });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || "Upload failed.");
    documentId = data.document_id;
    setStatus(`Uploaded ${data.filename} — ${data.clause_count} clauses found.`);
    $("analysis").classList.remove("hidden");
    $("questions").classList.remove("hidden");
    $("summary").textContent = "Analyzing document...";
    $("risks").innerHTML = "<p class='muted'>Analyzing...</p>";
    await analyze();
  } catch (e) {
    setStatus(e.message, true);
  } finally {
    $("uploadBtn").disabled = false;
  }
}

async function analyze() {
  const res = await fetch(`/api/analyze/${encodeURIComponent(documentId)}`, { method: "POST" });
  const data = await res.json();
  if (!res.ok) throw new Error(data.detail || "Analysis failed.");
  $("summary").textContent = data.summary || "No summary was returned.";
  const risks = Array.isArray(data.risks) ? data.risks : [];
  $("risks").innerHTML = risks.length ? risks.map(r => `<div class="risk"><strong>${escapeHtml(r.text || "Review this clause")}</strong><div class="citation">${escapeHtml(r.citation || "")}</div></div>`).join("") : "<p class='muted'>No specific risks were returned. Read the document carefully.</p>";
}

$("uploadBtn").addEventListener("click", upload);
$("askForm").addEventListener("submit", async (event) => {
  event.preventDefault();
  const question = $("question").value.trim();
  if (!documentId || !question) return;
  $("answer").textContent = "Thinking...";
  try {
    const res = await fetch("/api/ask", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ document_id: documentId, question, language: $("language").value }) });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || "Question failed.");
    const cites = (data.citations || []).map(c => `<div class="citation">${escapeHtml(c)}</div>`).join("");
    $("answer").innerHTML = `<div>${escapeHtml(data.answer || "No answer returned.")}</div>${cites}`;
  } catch (e) {
    $("answer").textContent = e.message;
  }
});

function escapeHtml(value) {
  return String(value).replace(/[&<>'"]/g, ch => ({"&":"&amp;","<":"&lt;",">":"&gt;","'":"&#39;","\"":"&quot;"}[ch]));
}
