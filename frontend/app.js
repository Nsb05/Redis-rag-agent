/* ---- app.js: AI Agent Long-Term Memory (RAG) UI Logic ---- */

const API = 'http://localhost:8000';
let sessionId = 'session-' + Math.random().toString(36).slice(2, 9);
let memories = [];
let totalMemories = 0;
let lastLatency = 0;
let statsInterval = null;

// ── DOM refs ────────────────────────────────────────────────
const chatMessages    = document.getElementById('chat-messages');
const chatTextarea    = document.getElementById('chat-input');
const sendBtn         = document.getElementById('send-btn');
const sessionIdInput  = document.getElementById('session-id');
const memoryList      = document.getElementById('memory-list');
const memoryCountEl   = document.getElementById('memory-count');
const searchInput     = document.getElementById('memory-search');
const clearAllBtn     = document.getElementById('clear-all-btn');
const refreshBtn      = document.getElementById('refresh-memories-btn');
const redisStatus     = document.getElementById('redis-status');
const headerMemCount  = document.getElementById('header-mem-count');
const headerLatency   = document.getElementById('header-latency');
const statMemCount    = document.getElementById('stat-mem-count');
const statModel       = document.getElementById('stat-model');
const statDim         = document.getElementById('stat-dim');
const statIndex       = document.getElementById('stat-index');

// ── Neural network canvas animation ─────────────────────────
(function initNeural() {
  const canvas = document.getElementById('neural-canvas');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');

  let W, H, nodes = [], animId;
  const NODE_COUNT = 60;
  const MAX_DIST   = 160;

  function resize() {
    W = canvas.width  = window.innerWidth;
    H = canvas.height = window.innerHeight;
  }

  function initNodes() {
    nodes = Array.from({ length: NODE_COUNT }, () => ({
      x: Math.random() * W,
      y: Math.random() * H,
      vx: (Math.random() - 0.5) * 0.4,
      vy: (Math.random() - 0.5) * 0.4,
      r: Math.random() * 2 + 1,
      hue: 210 + Math.random() * 60,
    }));
  }

  function draw() {
    ctx.clearRect(0, 0, W, H);

    // Draw connections
    for (let i = 0; i < nodes.length; i++) {
      for (let j = i + 1; j < nodes.length; j++) {
        const dx = nodes[i].x - nodes[j].x;
        const dy = nodes[i].y - nodes[j].y;
        const dist = Math.sqrt(dx * dx + dy * dy);
        if (dist < MAX_DIST) {
          const alpha = (1 - dist / MAX_DIST) * 0.4;
          ctx.beginPath();
          ctx.moveTo(nodes[i].x, nodes[i].y);
          ctx.lineTo(nodes[j].x, nodes[j].y);
          ctx.strokeStyle = `hsla(${nodes[i].hue}, 80%, 65%, ${alpha})`;
          ctx.lineWidth = 0.6;
          ctx.stroke();
        }
      }
    }

    // Draw nodes
    for (const n of nodes) {
      ctx.beginPath();
      ctx.arc(n.x, n.y, n.r, 0, Math.PI * 2);
      ctx.fillStyle = `hsla(${n.hue}, 90%, 70%, 0.8)`;
      ctx.fill();

      // Move
      n.x += n.vx;
      n.y += n.vy;
      if (n.x < 0 || n.x > W) n.vx *= -1;
      if (n.y < 0 || n.y > H) n.vy *= -1;
    }

    animId = requestAnimationFrame(draw);
  }

  window.addEventListener('resize', () => { resize(); initNodes(); });
  resize();
  initNodes();
  draw();
})();

// ── Tab switching ────────────────────────────────────────────
document.querySelectorAll('.tab-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
    document.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));
    btn.classList.add('active');
    document.getElementById(btn.dataset.tab).classList.add('active');
  });
});

// ── Health check ─────────────────────────────────────────────
async function checkHealth() {
  const dot = document.getElementById('redis-dot');
  const txt = document.getElementById('redis-text');
  if (dot) dot.className = 'status-dot loading';
  try {
    const res = await fetch(`${API}/health`);
    const data = await res.json();
    const ok = data.redis === 'connected';
    if (dot) { dot.className = 'status-dot ' + (ok ? 'online' : 'offline'); }
    if (txt) txt.textContent = ok ? 'Redis Online' : 'Redis Offline';

    // Update stats info card
    updateStatsCard(data);

    if (ok) loadMemories();
  } catch {
    if (dot) dot.className = 'status-dot offline';
    if (txt) txt.textContent = 'API Offline';
  }
}

async function loadStats() {
  try {
    const res  = await fetch(`${API}/stats`);
    const data = await res.json();
    if (data.memory_count !== undefined) {
      totalMemories = data.memory_count;
      updateMemoryCount(totalMemories);
    }
    if (statModel) statModel.textContent = data.embedding_model || '—';
    if (statDim)   statDim.textContent   = data.embedding_dim   || '—';
    if (statIndex) statIndex.textContent = 'HNSW / COSINE';
  } catch { /* ignore */ }
}

function updateStatsCard(data) {
  const card = document.getElementById('redis-info-card');
  if (!card) return;
  card.innerHTML = `
    <h4>🔧 System Info</h4>
    <div class="info-row"><span class="info-key">Redis</span><span class="info-val">${data.redis}</span></div>
    <div class="info-row"><span class="info-key">Embedding</span><span class="info-val">${data.embedding_model || 'all-MiniLM-L6-v2'}</span></div>
    <div class="info-row"><span class="info-key">LLM</span><span class="info-val">${data.llm || 'llama-3.3-70b'}</span></div>
    <div class="info-row"><span class="info-key">Groq Key</span><span class="info-val">${data.groq_key_set ? '✓ set' : '✗ missing'}</span></div>
  `;
}

function updateMemoryCount(count) {
  totalMemories = count;
  if (memoryCountEl)  memoryCountEl.textContent  = count;
  if (headerMemCount) headerMemCount.textContent  = count;
  if (statMemCount)   statMemCount.textContent    = count;
}

function updateLatency(ms) {
  lastLatency = ms;
  if (headerLatency) headerLatency.textContent = ms.toFixed(1) + 'ms';
}

// ── Memories ─────────────────────────────────────────────────
async function loadMemories() {
  try {
    const res  = await fetch(`${API}/memories?limit=200`);
    const data = await res.json();
    memories = data.memories || [];
    updateMemoryCount(memories.length);
    renderMemories(memories);
  } catch {
    renderMemoryError();
  }
}

function renderMemories(list) {
  if (!memoryList) return;

  if (list.length === 0) {
    memoryList.innerHTML = `
      <div class="empty-state">
        <div class="empty-icon">🧠</div>
        <div class="empty-title">No memories yet</div>
        <div class="empty-sub">Upload a document or add memories manually to get started.</div>
      </div>`;
    return;
  }

  memoryList.innerHTML = list.map(m => {
    const date = m.created_at ? new Date(m.created_at).toLocaleString(undefined, { dateStyle: 'short', timeStyle: 'short' }) : '—';
    const preview = m.text.length > 120 ? m.text.slice(0, 120) + '…' : m.text;
    return `
      <div class="memory-card" data-id="${m.id}">
        <div class="memory-card-header">
          <span class="memory-source">${escapeHtml(m.source || 'manual')}</span>
          ${m.tags ? `<span class="memory-score low">${escapeHtml(m.tags)}</span>` : ''}
        </div>
        <div class="memory-text" id="mt-${m.id}">${escapeHtml(preview)}</div>
        <div class="memory-card-footer">
          <span class="memory-date">${date}</span>
          <div class="memory-actions">
            <button class="btn-icon expand" title="Expand" onclick="toggleExpand('${m.id}', ${JSON.stringify(m.text)})">⤢</button>
            <button class="btn-icon delete" title="Delete" onclick="deleteMemory('${m.id}')">✕</button>
          </div>
        </div>
      </div>`;
  }).join('');
}

function renderMemoriesWithScores(list) {
  // Used after semantic search to show scores
  if (!memoryList) return;

  if (list.length === 0) {
    memoryList.innerHTML = `<div class="empty-state"><div class="empty-icon">🔍</div><div class="empty-title">No results</div><div class="empty-sub">Try a different search query.</div></div>`;
    return;
  }

  memoryList.innerHTML = list.map(m => {
    const date = m.created_at ? new Date(m.created_at).toLocaleString(undefined, { dateStyle: 'short', timeStyle: 'short' }) : '—';
    const scoreClass = m.score >= 0.7 ? 'high' : m.score >= 0.4 ? 'medium' : 'low';
    const preview = m.text.length > 120 ? m.text.slice(0, 120) + '…' : m.text;
    return `
      <div class="memory-card" data-id="${m.id}">
        <div class="memory-card-header">
          <span class="memory-source">${escapeHtml(m.source || 'search')}</span>
          <span class="memory-score ${scoreClass}">⬡ ${(m.score * 100).toFixed(0)}%</span>
        </div>
        <div class="memory-text" id="mt-${m.id}">${escapeHtml(preview)}</div>
        <div class="memory-card-footer">
          <span class="memory-date">${date}</span>
          <div class="memory-actions">
            <button class="btn-icon expand" title="Expand" onclick="toggleExpand('${m.id}', ${JSON.stringify(m.text)})">⤢</button>
            <button class="btn-icon delete" title="Delete" onclick="deleteMemory('${m.id}')">✕</button>
          </div>
        </div>
      </div>`;
  }).join('');
}

function renderMemoryError() {
  if (memoryList) memoryList.innerHTML = `<div class="empty-state"><div class="empty-icon">⚠️</div><div class="empty-title">Cannot reach API</div><div class="empty-sub">Make sure the backend is running on port 8000.</div></div>`;
}

function toggleExpand(id, fullText) {
  const el = document.getElementById(`mt-${id}`);
  if (!el) return;
  if (el.classList.contains('expanded')) {
    el.classList.remove('expanded');
    el.textContent = fullText.length > 120 ? fullText.slice(0, 120) + '…' : fullText;
  } else {
    el.classList.add('expanded');
    el.textContent = fullText;
  }
}

async function deleteMemory(id) {
  try {
    const res = await fetch(`${API}/memories/${id}`, { method: 'DELETE' });
    if (res.ok) {
      showToast('Memory deleted', 'success');
      loadMemories();
    } else {
      showToast('Delete failed', 'error');
    }
  } catch {
    showToast('Network error', 'error');
  }
}

// ── Semantic search in sidebar ─────────────────────────────
let searchTimeout = null;
if (searchInput) {
  searchInput.addEventListener('input', () => {
    clearTimeout(searchTimeout);
    const q = searchInput.value.trim();
    const info = document.getElementById('search-info');

    if (!q) {
      if (info) info.textContent = `${memories.length} memories stored`;
      renderMemories(memories);
      return;
    }

    searchTimeout = setTimeout(async () => {
      try {
        const res  = await fetch(`${API}/search?q=${encodeURIComponent(q)}&top_k=10`);
        const data = await res.json();
        updateLatency(data.latency_ms || 0);
        if (info) info.innerHTML = `${data.count} results &bull; <span style="color:var(--success);font-family:'JetBrains Mono',monospace">${(data.latency_ms || 0).toFixed(2)}ms</span>`;
        renderMemoriesWithScores(data.results || []);
      } catch {
        /* ignore */
      }
    }, 350);
  });
}

// ── Clear all memories ─────────────────────────────────────
if (clearAllBtn) {
  clearAllBtn.addEventListener('click', async () => {
    if (!confirm('Delete ALL memories from Redis? This cannot be undone.')) return;
    try {
      const res  = await fetch(`${API}/memories`, { method: 'DELETE' });
      const data = await res.json();
      showToast(`Cleared ${data.deleted_count} memories`, 'success');
      memories = [];
      updateMemoryCount(0);
      renderMemories([]);
    } catch {
      showToast('Error clearing memories', 'error');
    }
  });
}

if (refreshBtn) {
  refreshBtn.addEventListener('click', () => { loadMemories(); loadStats(); });
}

// ── Chat ────────────────────────────────────────────────────
if (sessionIdInput) {
  sessionIdInput.value = sessionId;
  sessionIdInput.addEventListener('change', () => {
    sessionId = sessionIdInput.value.trim() || sessionId;
  });
}

if (chatTextarea) {
  chatTextarea.addEventListener('keydown', e => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  });

  // Auto-resize
  chatTextarea.addEventListener('input', () => {
    chatTextarea.style.height = 'auto';
    chatTextarea.style.height = Math.min(chatTextarea.scrollHeight, 140) + 'px';
  });
}

if (sendBtn) sendBtn.addEventListener('click', sendMessage);

async function sendMessage() {
  const text = chatTextarea?.value?.trim();
  if (!text) return;

  // Remove welcome card if present
  const welcome = document.getElementById('welcome-card');
  if (welcome) welcome.remove();

  appendMessage('user', text);
  chatTextarea.value = '';
  chatTextarea.style.height = 'auto';
  sendBtn.disabled = true;

  const typingEl = showTyping();

  try {
    const res = await fetch(`${API}/chat`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message: text, session_id: sessionId }),
    });

    typingEl.remove();

    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: 'Unknown error' }));
      appendMessage('assistant', `⚠️ Error: ${err.detail}`, [], 0);
    } else {
      const data = await res.json();
      appendMessage('assistant', data.response, data.memories_used || [], data.retrieval_latency_ms);
      updateLatency(data.retrieval_latency_ms || 0);

      // Refresh memory count
      loadStats();
    }
  } catch (err) {
    typingEl.remove();
    appendMessage('assistant', '⚠️ Cannot reach the backend. Make sure `uvicorn backend.main:app --reload` is running on port 8000.');
  }

  sendBtn.disabled = false;
  chatTextarea.focus();
  scrollToBottom();
}

function appendMessage(role, text, memoriesUsed = [], latencyMs = 0) {
  const div = document.createElement('div');
  div.className = `message ${role}`;

  const avatar = role === 'user' ? '👤' : '🤖';
  const time = new Date().toLocaleTimeString(undefined, { hour: '2-digit', minute: '2-digit' });

  let memoriesHtml = '';
  if (memoriesUsed && memoriesUsed.length > 0) {
    const pills = memoriesUsed.map(m => `
      <span class="memory-pill" title="${escapeHtml(m.text)}">
        🧠 ${escapeHtml(m.source || 'memory')}
        <span class="score-badge">${(m.score * 100).toFixed(0)}%</span>
      </span>
    `).join('');
    memoriesHtml = `<div class="memory-context">${pills}</div>`;
  }

  const latencyHtml = latencyMs > 0
    ? `<span class="retrieval-info" style="margin-left:6px">⚡ <span class="latency-val">${latencyMs.toFixed(2)}ms</span></span>`
    : '';

  div.innerHTML = `
    <div class="message-avatar">${avatar}</div>
    <div class="message-body">
      <div class="message-bubble">${formatText(text)}</div>
      ${memoriesHtml}
      <div class="message-meta">${time}${latencyHtml}</div>
    </div>
  `;

  chatMessages.appendChild(div);
  scrollToBottom();
}

function showTyping() {
  const div = document.createElement('div');
  div.className = 'typing-indicator';
  div.innerHTML = `
    <div class="message-avatar" style="background:linear-gradient(135deg,var(--accent-2),var(--accent-3))">🤖</div>
    <div class="typing-dots"><span></span><span></span><span></span></div>
  `;
  chatMessages.appendChild(div);
  scrollToBottom();
  return div;
}

function scrollToBottom() {
  if (chatMessages) chatMessages.scrollTop = chatMessages.scrollHeight;
}

function formatText(text) {
  // Basic markdown-like formatting
  return escapeHtml(text)
    .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
    .replace(/\*(.*?)\*/g, '<em>$1</em>')
    .replace(/`(.*?)`/g, `<code style="background:rgba(125,211,252,0.1);color:var(--text-code);padding:1px 5px;border-radius:4px;font-family:'JetBrains Mono',monospace;font-size:12px">$1</code>`)
    .replace(/\n/g, '<br>');
}

// ── Upload ──────────────────────────────────────────────────
const uploadForm = document.getElementById('upload-form');
if (uploadForm) {
  uploadForm.addEventListener('submit', async e => {
    e.preventDefault();
    const text   = document.getElementById('doc-text')?.value?.trim();
    const source = document.getElementById('doc-source')?.value?.trim() || 'upload';
    const tags   = document.getElementById('doc-tags')?.value?.trim()   || '';
    const file   = document.getElementById('doc-file')?.files?.[0];
    const resultEl = document.getElementById('upload-result');
    const submitBtn = uploadForm.querySelector('button[type="submit"]');

    if (!text && !file) {
      showResult(resultEl, 'error', 'Please provide text content or upload a file.');
      return;
    }

    submitBtn.disabled = true;
    submitBtn.textContent = 'Storing…';

    try {
      const formData = new FormData();
      if (text)   formData.append('text',   text);
      if (source) formData.append('source', source);
      if (tags)   formData.append('tags',   tags);
      if (file)   formData.append('file',   file);

      const res  = await fetch(`${API}/upload`, { method: 'POST', body: formData });
      const data = await res.json();

      if (res.ok) {
        showResult(resultEl, 'success', `✓ Stored ${data.chunks_stored} memory chunk(s) from "${data.source}"`);
        uploadForm.reset();
        loadMemories();
        loadStats();
        showToast(`${data.chunks_stored} memories added!`, 'success');
      } else {
        showResult(resultEl, 'error', `Error: ${data.detail}`);
      }
    } catch {
      showResult(resultEl, 'error', 'Network error — is the backend running?');
    }

    submitBtn.disabled = false;
    submitBtn.textContent = '⬆ Store in Memory';
  });
}

// ── Manual add memory ───────────────────────────────────────
const addMemoryForm = document.getElementById('add-memory-form');
if (addMemoryForm) {
  addMemoryForm.addEventListener('submit', async e => {
    e.preventDefault();
    const text   = document.getElementById('manual-text')?.value?.trim();
    const source = document.getElementById('manual-source')?.value?.trim() || 'manual';
    const tags   = document.getElementById('manual-tags')?.value?.trim()   || '';
    const resultEl = document.getElementById('add-memory-result');

    if (!text) return;

    try {
      const res  = await fetch(`${API}/memories`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text, source, tags }),
      });
      const data = await res.json();

      if (res.ok) {
        showResult(resultEl, 'success', `✓ Memory stored (ID: ${data.id.slice(0, 8)}…)`);
        addMemoryForm.reset();
        loadMemories();
        loadStats();
        showToast('Memory added!', 'success');
      } else {
        showResult(resultEl, 'error', `Error: ${data.detail}`);
      }
    } catch {
      showResult(resultEl, 'error', 'Network error.');
    }
  });
}

// ── Drop zone ───────────────────────────────────────────────
const dropZone = document.getElementById('drop-zone');
if (dropZone) {
  dropZone.addEventListener('dragover', e => { e.preventDefault(); dropZone.classList.add('drag-over'); });
  dropZone.addEventListener('dragleave', () => dropZone.classList.remove('drag-over'));
  dropZone.addEventListener('drop', e => {
    e.preventDefault();
    dropZone.classList.remove('drag-over');
    const file = e.dataTransfer.files[0];
    if (file) {
      const input = document.getElementById('doc-file');
      if (input) {
        const dt = new DataTransfer();
        dt.items.add(file);
        input.files = dt.files;
        dropZone.querySelector('.drop-zone-text').textContent = `📄 ${file.name}`;
      }
    }
  });
}

// ── Utility ─────────────────────────────────────────────────
function escapeHtml(str) {
  if (!str) return '';
  return String(str)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}

function showResult(el, type, msg) {
  if (!el) return;
  el.className = `upload-result ${type}`;
  el.textContent = msg;
  el.style.display = 'block';
  setTimeout(() => { el.style.display = 'none'; }, 5000);
}

function showToast(msg, type = 'info') {
  const container = document.getElementById('toast-container');
  if (!container) return;

  const icons = { success: '✓', error: '✕', info: 'ℹ' };
  const toast = document.createElement('div');
  toast.className = `toast ${type}`;
  toast.innerHTML = `<span>${icons[type] || 'ℹ'}</span> ${escapeHtml(msg)}`;
  container.appendChild(toast);

  setTimeout(() => {
    toast.style.animation = 'toastOut 0.3s ease forwards';
    setTimeout(() => toast.remove(), 300);
  }, 3000);
}

// ── Init ─────────────────────────────────────────────────────
window.addEventListener('DOMContentLoaded', () => {
  checkHealth();
  loadStats();
  statsInterval = setInterval(() => { loadStats(); }, 15000);
});
