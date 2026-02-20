/* =====================================================
   LegalAid — Frontend Logic
   ===================================================== */

const API_BASE = "";

let activeFilename = null;
let uploadedDocs = [];
let isAsking = false;

// ── DOM refs ──────────────────────────────────────────
const messagesContainer = document.getElementById("messagesContainer");
const questionInput = document.getElementById("questionInput");
const askBtn = document.getElementById("askBtn");
const uploadZone = document.getElementById("uploadZone");
const fileInput = document.getElementById("fileInput");
const browseBtn = document.getElementById("browseBtn");
const uploadProgress = document.getElementById("uploadProgress");
const progressFill = document.getElementById("progressFill");
const progressLabel = document.getElementById("progressLabel");
const docList = document.getElementById("docList");
const docEmptyMsg = document.getElementById("docEmptyMsg");
const activeContextEl = document.getElementById("activeContext");
const clearCtxBtn = document.getElementById("clearCtxBtn");
const statusDot = document.querySelector(".status-dot");
const statusText = document.getElementById("statusText");

// ── Health check ──────────────────────────────────────
async function checkHealth() {
    try {
        const res = await fetch(`${API_BASE}/health`);
        if (res.ok) {
            statusDot.classList.add("online");
            statusText.textContent = "API Online";
        } else throw new Error();
    } catch {
        statusDot.classList.add("error");
        statusText.textContent = "API Offline";
    }
}

// ── Upload ────────────────────────────────────────────
browseBtn.addEventListener("click", () => fileInput.click());
uploadZone.addEventListener("click", (e) => { if (e.target !== browseBtn) fileInput.click(); });
fileInput.addEventListener("change", () => { if (fileInput.files[0]) uploadFile(fileInput.files[0]); });

uploadZone.addEventListener("dragover", (e) => { e.preventDefault(); uploadZone.classList.add("dragover"); });
uploadZone.addEventListener("dragleave", () => uploadZone.classList.remove("dragover"));
uploadZone.addEventListener("drop", (e) => {
    e.preventDefault();
    uploadZone.classList.remove("dragover");
    const file = e.dataTransfer.files[0];
    if (file) uploadFile(file);
});

async function uploadFile(file) {
    if (!file.name.toLowerCase().endsWith(".pdf")) {
        showToast("Only PDF files are supported.", "error"); return;
    }
    uploadProgress.classList.remove("hidden");
    progressFill.style.width = "0%";
    progressLabel.textContent = `Uploading ${file.name}…`;
    browseBtn.disabled = true;

    let fake = 0;
    const timer = setInterval(() => { fake = Math.min(fake + 7, 85); progressFill.style.width = fake + "%"; }, 250);

    try {
        const fd = new FormData();
        fd.append("file", file);
        const res = await fetch(`${API_BASE}/upload`, { method: "POST", body: fd });
        const data = await res.json();
        clearInterval(timer);
        progressFill.style.width = "100%";
        progressLabel.textContent = "Processing complete!";
        if (data.status === "success") {
            uploadedDocs.push({ filename: data.filename, chunks: data.chunks_added });
            renderDocList();
            showToast(`✅ Indexed ${data.chunks_added} chunks from "${data.filename}"`, "success");
            setActiveDoc(data.filename);
        } else {
            showToast(`❌ Upload failed: ${data.error || data.detail}`, "error");
        }
    } catch (err) {
        clearInterval(timer);
        showToast(`❌ Network error: ${err.message}`, "error");
    } finally {
        setTimeout(() => { uploadProgress.classList.add("hidden"); browseBtn.disabled = false; fileInput.value = ""; }, 1500);
    }
}

// ── Document list ─────────────────────────────────────
function renderDocList() {
    docEmptyMsg.classList.toggle("hidden", uploadedDocs.length > 0);
    document.querySelectorAll(".doc-item").forEach(el => el.remove());
    uploadedDocs.forEach(doc => {
        const item = document.createElement("div");
        item.className = "doc-item" + (activeFilename === doc.filename ? " active" : "");
        item.dataset.filename = doc.filename;
        item.innerHTML = `
      <span class="doc-item-icon">📄</span>
      <span class="doc-item-name" title="${doc.filename}">${doc.filename}</span>
      <span class="doc-item-chunks">${doc.chunks}c</span>`;
        item.addEventListener("click", () => setActiveDoc(doc.filename));
        docList.insertBefore(item, docEmptyMsg);
    });
}

function setActiveDoc(filename) {
    activeFilename = filename;
    activeContextEl.textContent = filename;
    clearCtxBtn.classList.remove("hidden");
    document.querySelectorAll(".doc-item").forEach(el => {
        el.classList.toggle("active", el.dataset.filename === filename);
    });
}

function clearContext() {
    activeFilename = null;
    activeContextEl.textContent = "All Documents";
    clearCtxBtn.classList.add("hidden");
    document.querySelectorAll(".doc-item").forEach(el => el.classList.remove("active"));
}

// ── General / greeting question patterns ────────────────
const GENERAL_PATTERNS = [
    /^(hi+|hello|hey+|hii+|helo|hai|yo|namaste|good\s*(morning|evening|afternoon|day))\W*$/i,
    /how (can|do|will|could) you help/i,
    /what can you do/i,
    /who are you/i,
    /what (are|is) (you|your|this|legalaid)/i,
    /tell me about yourself/i,
    /^(help|help me)\W*$/i,
    /^(thanks?|thank you|thx|ty)\W*$/i,
    /^(ok|okay|got it|alright|sure|cool|great|nice|awesome)\W*$/i,
    /^(what|how).*\?*$/.source && /^(what|how) (do|does|can|should) (i|we) (do|start|begin|use)/i,
];

const GENERAL_REPLY = [
    "**Hello! I'm LegalAid AI** \u2014 your intelligent legal document assistant.",
    "",
    "**What I can do for you:**",
    "- \uD83D\uDCC4 **Analyse any legal PDF** \u2014 Upload FIRs, acts, judgements, contracts, notices, or any legal document.",
    "- \uD83D\uDCAC **Answer legal questions** \u2014 Ask me anything about the content and I'll give a structured, cited answer.",
    "- \uD83D\uDD17 **Cite exact sources** \u2014 Every answer references the exact section or clause from your document.",
    "- \uD83D\uDDC2\uFE0F **Multiple documents** \u2014 Upload several files and filter questions to a specific one.",
    "",
    "**To get started:**",
    "1. Click **Choose File** (or drag-and-drop a PDF) in the left panel.",
    "2. Wait a few seconds for indexing to complete.",
    "3. Type your legal question below \u2014 for example: *\"What are the punishments for theft?\"*",
    "",
    "*I specialise in Indian law \u2014 IPC, BNS, CrPC, Labour Acts, and more.*"
].join("\n");

// ── Chat ──────────────────────────────────────────────
questionInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && e.ctrlKey) { e.preventDefault(); askQuestion(); }
    setTimeout(() => {
        questionInput.style.height = "auto";
        questionInput.style.height = Math.min(questionInput.scrollHeight, 140) + "px";
    }, 0);
});

async function askQuestion() {
    const q = questionInput.value.trim();
    if (!q || isAsking) return;

    // ── Hard-coded greetings only (instant client-side, no API needed) ──
    if (GENERAL_PATTERNS.some(p => p instanceof RegExp && p.test(q))) {
        appendUserMessage(q);
        setTimeout(() => { appendAIMessage(GENERAL_REPLY, [], null); }, 300);
        return;
    }

    isAsking = true;
    askBtn.disabled = true;
    questionInput.value = "";
    questionInput.style.height = "auto";

    appendUserMessage(q);

    // ── Route: no docs uploaded → /chat (Sarvam direct, no RAG)
    //           docs uploaded   → /ask  (full RAG pipeline)        ──
    const hasDocuments = uploadedDocs.length > 0;
    const endpoint = hasDocuments ? "/ask" : "/chat";
    const thinkLabel = hasDocuments ? "Analysing documents…" : "Thinking…";
    const thinkId = appendThinking(thinkLabel);

    try {
        const payload = hasDocuments
            ? { question: q, ...(activeFilename ? { filter_filename: activeFilename } : {}) }
            : { question: q };

        const res = await fetch(`${API_BASE}${endpoint}`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload),
        });
        const data = await res.json();
        removeThinking(thinkId);

        if (res.ok) {
            appendAIMessage(data.answer || "", data.citations || [], data.disclaimer);
        } else {
            appendAIMessage(`❌ Error: ${data.detail || "Unknown error"}`, []);
        }
    } catch (err) {
        removeThinking(thinkId);
        appendAIMessage(`❌ Network error: ${err.message}`, []);
    } finally {
        isAsking = false;
        askBtn.disabled = false;
        questionInput.focus();
    }
}

// ── Message rendering ─────────────────────────────────
function appendUserMessage(text) {
    // Hide the welcome card when the first message is sent
    if (welcomeCard && !welcomeCard.classList.contains("hidden")) {
        welcomeCard.classList.add("hidden");
    }
    const div = document.createElement("div");
    div.className = "message message-user";
    div.innerHTML = `
    <div class="message-avatar">YOU</div>
    <div class="message-body">
      <div class="message-bubble">${escHtml(text)}</div>
    </div>`;
    messagesContainer.appendChild(div);
    scrollBottom();
}

function appendAIMessage(rawText, citations, disclaimer) {
    const div = document.createElement("div");
    div.className = "message message-ai";

    const rendered = renderMarkdown(rawText);

    let citHtml = "";
    if (citations && citations.length > 0) {
        // De-duplicate by filename
        const seen = new Set();
        const unique = citations.filter(c => {
            const key = c.where;
            if (seen.has(key)) return false;
            seen.add(key); return true;
        });
        const items = unique.map(c => `
      <div class="cite-item">
        <span class="cite-ref">${escHtml(c.ref)}</span>
        <span><strong>${escHtml(c.title || "Section")}</strong>
          <span class="cite-where"> — ${escHtml(c.where || "")}</span>
        </span>
      </div>`).join("");
        citHtml = `<div class="citations-block">
      <div class="citations-heading">📎 Sources</div>${items}</div>`;
    }

    // Show disclaimer only for actual legal answers (not general chat)
    const dis = disclaimer
        ? `<div class="disclaimer-badge">\u26a0\uFE0F ${escHtml(disclaimer)}</div>`
        : (citations && citations.length > 0)
            ? `<div class="disclaimer-badge">\u26a0\uFE0F General legal information only \u2014 not legal advice. Consult a qualified professional.</div>`
            : "";

    div.innerHTML = `
    <div class="message-avatar">⚖️</div>
    <div class="message-body">
      <div class="message-bubble md-body">${rendered}</div>
      ${citHtml}
      ${dis}
    </div>`;
    messagesContainer.appendChild(div);
    scrollBottom();
}

function appendThinking(label = "Analysing documents…") {
    const id = "think-" + Date.now();
    const div = document.createElement("div");
    div.id = id;
    div.className = "message message-ai";
    div.innerHTML = `
    <div class="message-avatar">⚖️</div>
    <div class="message-body">
      <div class="thinking-bubble">
        <div class="dot"></div><div class="dot"></div><div class="dot"></div>
        <span style="font-size:0.78rem;color:var(--text-muted);margin-left:4px;">${escHtml(label)}</span>
      </div>
    </div>`;
    messagesContainer.appendChild(div);
    scrollBottom();
    return id;
}

function removeThinking(id) {
    const el = document.getElementById(id);
    if (el) el.remove();
}

// ── Markdown renderer ─────────────────────────────────
// Handles: ## headings, **bold**, *italic*, - bullets, numbered lists, blank lines
function renderMarkdown(text) {
    if (!text) return "";

    const lines = text.split("\n");
    let html = "";
    let inUl = false;
    let inOl = false;

    const closeList = () => {
        if (inUl) { html += "</ul>"; inUl = false; }
        if (inOl) { html += "</ol>"; inOl = false; }
    };

    for (let raw of lines) {
        const line = raw.trimEnd();

        // Skip leftover think tags if any slip through
        if (/<\/?think>/i.test(line)) continue;

        // ## Heading 2
        if (/^##\s+/.test(line)) {
            closeList();
            html += `<h3>${inlineMd(line.replace(/^##\s+/, ""))}</h3>`;
            continue;
        }
        // ### Heading 3
        if (/^###\s+/.test(line)) {
            closeList();
            html += `<h4>${inlineMd(line.replace(/^###\s+/, ""))}</h4>`;
            continue;
        }
        // **Bold heading** alone on a line (the prompt uses **Section Name**)
        if (/^\*\*[^*]+\*\*$/.test(line.trim())) {
            closeList();
            html += `<h4>${inlineMd(line.trim())}</h4>`;
            continue;
        }
        // Unordered list: - item
        if (/^- /.test(line)) {
            if (inOl) { html += "</ol>"; inOl = false; }
            if (!inUl) { html += "<ul>"; inUl = true; }
            html += `<li>${inlineMd(line.replace(/^- /, ""))}</li>`;
            continue;
        }
        // Ordered list: 1. item
        if (/^\d+\.\s/.test(line)) {
            if (inUl) { html += "</ul>"; inUl = false; }
            if (!inOl) { html += "<ol>"; inOl = true; }
            html += `<li>${inlineMd(line.replace(/^\d+\.\s/, ""))}</li>`;
            continue;
        }
        // Empty line → paragraph break
        if (line.trim() === "" || line === "---") {
            closeList();
            html += "<br>";
            continue;
        }
        // Normal paragraph line
        closeList();
        html += `<p>${inlineMd(line)}</p>`;
    }
    closeList();
    return html;
}

// Inline: **bold**, *italic*, `code`, [ref] citation tags
function inlineMd(text) {
    return escHtml(text)
        .replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")
        .replace(/\*(.+?)\*/g, "<em>$1</em>")
        .replace(/`(.+?)`/g, "<code>$1</code>")
        .replace(/\[(\d+)\]/g, '<span class="cite-ref">[$1]</span>');
}

// ── Utilities ─────────────────────────────────────────
function escHtml(str) {
    if (!str) return "";
    return String(str)
        .replace(/&/g, "&amp;").replace(/</g, "&lt;")
        .replace(/>/g, "&gt;").replace(/"/g, "&quot;");
}

function scrollBottom() {
    messagesContainer.scrollTo({ top: messagesContainer.scrollHeight, behavior: "smooth" });
}

function showToast(msg, type = "info") {
    const t = document.createElement("div");
    t.className = `toast ${type}`;
    t.textContent = msg;
    document.body.appendChild(t);
    setTimeout(() => t.remove(), 4000);
}

// ── Init ──────────────────────────────────────────────
checkHealth();
questionInput.focus();

// ── FAQ chip click handlers ─────────────────────────
const welcomeCard = document.getElementById("welcomeCard");

document.querySelectorAll(".faq-chip").forEach(chip => {
    chip.addEventListener("click", () => {
        const q = chip.dataset.q;
        if (!q) return;
        questionInput.value = q;
        questionInput.dispatchEvent(new Event("input"));
        askQuestion();
    });
});
