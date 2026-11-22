/* LoRA node UI: polls /api/state and renders queue, budget, slots, events. */

const $ = (id) => document.getElementById(id);
const state = { presets: null, latest: null, pollTimer: null, detailTimer: null };

const STATE_LABEL = {
  queued: "排队",
  loading: "加载中",
  ready: "待计算",
  running: "计算中",
  done: "完成",
  error: "失败",
};

function fmtBytes(n) {
  if (n === null || n === undefined) return "—";
  return `${n.toLocaleString()} B`;
}

function esc(text) {
  return String(text ?? "").replace(/[&<>"']/g, (c) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
  }[c]));
}

async function api(path, options = {}) {
  const res = await fetch(path, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  let body = null;
  try { body = await res.json(); } catch { /* empty body */ }
  if (!res.ok) {
    const detail = body && body.detail
      ? (typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail))
      : `HTTP ${res.status}`;
    throw new Error(detail);
  }
  return body;
}

/* ------------------------------------------------------------- presets */

async function loadPresets() {
  const data = await api("/api/presets");
  state.presets = data;
  $("dim").textContent = data.input_dim;
  const select = $("adapter");
  select.innerHTML = data.presets
    .map((p) => `<option value="${esc(p.key)}">${esc(p.name)} · rank ${p.rank}</option>`)
    .join("");
  renderAdapterMeta();
  select.addEventListener("change", renderAdapterMeta);
}

function renderAdapterMeta() {
  if (!state.presets) return;
  const key = $("adapter").value;
  const preset = state.presets.presets.find((p) => p.key === key);
  if (!preset) return;
  const budget = state.presets.budget_bytes;
  const share = ((preset.bytes / budget) * 100).toFixed(0);
  $("adapter-meta").innerHTML =
    `驻留占用 <b>${fmtBytes(preset.bytes)}</b>（预算的 ${share}%）· ` +
    `模拟加载 ${Math.round(preset.load_delay * 1000)} ms · ` +
    `输出 <b>${state.presets.output_dim}</b> 维`;
}

/* --------------------------------------------------------------- submit */

async function submitRequest(event) {
  event.preventDefault();
  const msg = $("form-msg");
  msg.textContent = "";
  msg.className = "msg";
  const raw = $("vector").value.trim();
  let vector = null;
  if (raw) {
    vector = raw.split(/[\s,;]+/).filter(Boolean).map(Number);
    if (vector.some((n) => !Number.isFinite(n))) {
      msg.textContent = "向量包含非数字内容。";
      msg.className = "msg err";
      return;
    }
  }
  try {
    const body = {
      adapter: $("adapter").value,
      compute_ms: Number($("compute-ms").value) || 0,
      simulate_load_failure: $("fail-load").checked,
    };
    if (vector) body.vector = vector;
    const res = await api("/api/requests", { method: "POST", body: JSON.stringify(body) });
    msg.textContent = `已提交请求 ${res.id}${res.position ? `，队列第 ${res.position} 位` : ""}。`;
    msg.className = "msg ok";
    $("fail-load").checked = false;
    await refresh();
  } catch (err) {
    msg.textContent = `提交失败：${err.message}`;
    msg.className = "msg err";
  }
}

function fillSample() {
  const dim = state.presets ? state.presets.input_dim : 128;
  const parts = [];
  let seed = 42;
  for (let i = 0; i < dim; i += 1) {
    seed = (seed * 1103515245 + 12345) % 2147483648;
    parts.push(((seed / 2147483648) * 2 - 1).toFixed(4));
  }
  $("vector").value = parts.join(", ");
}

/* -------------------------------------------------------------- render */

function tagFor(item) {
  if (item.is_head && item.blocked_on_memory) {
    return `<span class="tag blocked">队首等内存</span>`;
  }
  if (item.is_head) return `<span class="tag head">队首</span>`;
  if (item.skipped_head) return `<span class="tag blocked">已让路</span>`;
  const cls = { running: "running", done: "done", error: "error" }[item.state] || "";
  return `<span class="tag ${cls}">${esc(STATE_LABEL[item.state] || item.state)}</span>`;
}

function renderBudget(snap) {
  const used = snap.used_bytes;
  const total = snap.budget_bytes;
  const pct = Math.min(100, (used / total) * 100);
  const fill = $("meter-fill");
  fill.style.width = `${pct}%`;
  fill.classList.toggle("high", pct > 85);
  $("budget-label").textContent = `${snap.max_concurrent} 个计算位`;
  $("used-label").textContent = `已用 ${fmtBytes(used)}（${pct.toFixed(0)}%）`;
  $("free-label").textContent = `剩余 ${fmtBytes(snap.free_bytes)}`;

  const body = $("resident-body");
  if (!snap.resident.length) {
    body.innerHTML = `<tr><td colspan="4" class="empty">暂无驻留权重</td></tr>`;
    return;
  }
  body.innerHTML = snap.resident
    .map((r) => {
      const status = r.loading
        ? `<span class="tag">加载中</span>`
        : r.in_use
          ? `<span class="tag running">在用</span>`
          : `<span class="tag">空闲 ${r.idle_seconds ?? 0}s</span>`;
      return `<tr>
        <td>${esc(r.name)} <span class="muted mono">${esc(r.key)}</span></td>
        <td class="mono">${fmtBytes(r.bytes)}</td>
        <td class="mono">${r.refcount}</td>
        <td>${status}</td>
      </tr>`;
    })
    .join("");
}

function renderSlots(snap) {
  const list = $("running-list");
  const slots = [];
  for (let i = 0; i < snap.max_concurrent; i += 1) {
    const item = snap.running[i];
    if (!item) {
      slots.push(`<div class="slot empty">计算位 ${i + 1} · 空闲</div>`);
      continue;
    }
    const elapsed = item.run_seconds ?? 0;
    const pct = Math.min(100, (elapsed * 1000 / Math.max(1, item.compute_ms)) * 100);
    slots.push(`<div class="slot">
      <div class="line1">
        <span class="mono">${esc(item.id)} · ${esc(item.adapter_name)}</span>
        <span class="muted">${elapsed.toFixed(1)}s / ${(item.compute_ms / 1000).toFixed(1)}s</span>
      </div>
      <div class="muted small">共享权重引用 ${item.shared_with} · ${item.source === "demo" ? "样例" : "手动"}</div>
      <div class="bar"><i style="width:${pct}%"></i></div>
    </div>`);
  }
  list.innerHTML = slots.join("");
  $("slots-label").textContent = `${snap.running.length}/${snap.max_concurrent} 占用`;
}

function renderQueue(snap) {
  const body = $("queue-body");
  $("queue-label").textContent = snap.queue.length ? `${snap.queue.length} 单等待` : "";
  if (!snap.queue.length) {
    body.innerHTML = `<tr><td colspan="6" class="empty">队列为空</td></tr>`;
    return;
  }
  body.innerHTML = snap.queue
    .map((item) => {
      const notes = [];
      if (item.position === 1 && item.blocked_on_memory) {
        notes.push("需要新的权重，但在用权重尚未释放");
      }
      if (item.position === 1) notes.push("若能用空闲权重腾出空间，将优先启动");
      if (item.skipped_head) notes.push("已让后到请求越过一次，不再放行");
      if (item.state === "loading") notes.push("正在加载权重，后到请求继续等待");
      if (item.state === "ready") notes.push("等待计算位");
      if (item.source === "demo") notes.push("样例请求");
      return `<tr class="clickable ${item.position === 1 ? "head" : ""}"
                  data-id="${esc(item.id)}">
        <td class="mono">${item.position}</td>
        <td class="mono">${esc(item.id)}</td>
        <td>${esc(item.adapter_name)}</td>
        <td>${tagFor(item)} ${esc(STATE_LABEL[item.state] || item.state)}</td>
        <td class="mono">${item.queue_seconds.toFixed(1)}s</td>
        <td class="muted small">${esc(notes.join("；"))}</td>
      </tr>`;
    })
    .join("");
  body.querySelectorAll("tr[data-id]").forEach((row) => {
    row.addEventListener("click", () => showDetail(row.dataset.id));
  });
}

function renderFinished(snap) {
  const body = $("finished-body");
  $("finished-label").textContent = snap.finished_total
    ? `共 ${snap.finished_total} 单，显示最近 ${Math.min(snap.finished_total, snap.finished_window)} 单`
    : "";
  if (!snap.finished.length) {
    body.innerHTML = `<tr><td colspan="5" class="empty">暂无完成请求</td></tr>`;
    return;
  }
  body.innerHTML = snap.finished
    .slice()
    .reverse()
    .map((item) => {
      const preview = item.error
        ? `<span class="tag error">${esc(item.error)}</span>`
        : `<span class="mono">[${(item.result_preview || []).slice(0, 4).join(", ")} …]</span>`;
      const took = item.run_seconds != null ? `${item.run_seconds.toFixed(1)}s` : "—";
      return `<tr class="clickable" data-id="${esc(item.id)}">
        <td class="mono">${esc(item.id)}</td>
        <td>${esc(item.adapter_name)}</td>
        <td>${tagFor(item)} ${esc(STATE_LABEL[item.state] || item.state)}</td>
        <td class="mono">${took}</td>
        <td>${preview}</td>
      </tr>`;
    })
    .join("");
  body.querySelectorAll("tr[data-id]").forEach((row) => {
    row.addEventListener("click", () => showDetail(row.dataset.id));
  });
}

const HIGHLIGHT = new Set(["head_bypassed", "bypass_dispatched", "waiting_memory", "load_failed"]);
const DONE_KINDS = new Set(["compute_done", "load_failed"]);

function renderEvents(snap) {
  const box = $("events");
  const items = snap.events.slice(-80);
  box.innerHTML = items
    .map((e) => {
      const time = new Date(e.ts * 1000).toLocaleTimeString("zh-CN", { hour12: false });
      const cls = HIGHLIGHT.has(e.kind) ? "bypass" : DONE_KINDS.has(e.kind) ? "done" : "";
      return `<div class="ev ${cls}">
        <span class="k">${time}</span>
        <span class="k ${HIGHLIGHT.has(e.kind) ? "warn" : ""}">${esc(e.kind)}</span>
        <span>${esc(e.message)}</span>
      </div>`;
    })
    .join("");
  box.scrollTop = box.scrollHeight;
}

function renderDemo(demo) {
  const btn = $("demo-btn");
  btn.disabled = Boolean(demo.running);
  btn.textContent = demo.running ? `样例运行中 · ${demo.phase}` : "运行样例";
  if (demo.error) {
    const msg = $("form-msg");
    msg.textContent = `样例出错：${demo.error}`;
    msg.className = "msg err";
  }
}

function render(snap) {
  state.latest = snap;
  renderBudget(snap);
  renderSlots(snap);
  renderQueue(snap);
  renderFinished(snap);
  renderEvents(snap);
  renderDemo(snap.demo);
}

/* -------------------------------------------------------------- polling */

async function refresh() {
  try {
    const snap = await api("/api/state");
    render(snap);
    setConn(true);
  } catch (err) {
    setConn(false, err.message);
  }
}

function setConn(ok, message) {
  const pill = $("conn");
  pill.className = `pill ${ok ? "live" : "down"}`;
  pill.textContent = ok ? "在线 · 1s 刷新" : `离线 · ${message || "无响应"}`;
}

function startPolling() {
  if (state.pollTimer) clearInterval(state.pollTimer);
  state.pollTimer = setInterval(refresh, 1000);
}

/* --------------------------------------------------------------- detail */

async function showDetail(id) {
  const dialog = $("detail-dialog");
  $("detail-title").textContent = `请求 ${id}`;
  $("detail-body").textContent = "加载中…";
  if (typeof dialog.showModal === "function") dialog.showModal();
  try {
    const detail = await api(`/api/requests/${id}`);
    $("detail-body").textContent = JSON.stringify(detail, null, 2);
  } catch (err) {
    $("detail-body").textContent = `加载失败：${err.message}`;
  }
}

/* ---------------------------------------------------------------- wiring */

async function runDemo() {
  const msg = $("form-msg");
  msg.textContent = "";
  try {
    await api("/api/demo", { method: "POST" });
    msg.textContent = "样例已启动：观察队首等内存时，后到请求先行派发一次。";
    msg.className = "msg ok";
    await refresh();
  } catch (err) {
    msg.textContent = `无法启动样例：${err.message}`;
    msg.className = "msg err";
  }
}

async function resetNode() {
  const msg = $("form-msg");
  try {
    await api("/api/reset", { method: "POST" });
    msg.textContent = "节点已重置。";
    msg.className = "msg ok";
    await refresh();
  } catch (err) {
    msg.textContent = `重置失败：${err.message}`;
    msg.className = "msg err";
  }
}

async function init() {
  try {
    await loadPresets();
  } catch (err) {
    setConn(false, err.message);
  }
  $("submit-form").addEventListener("submit", submitRequest);
  $("sample-btn").addEventListener("click", fillSample);
  $("demo-btn").addEventListener("click", runDemo);
  $("reset-btn").addEventListener("click", resetNode);
  $("detail-close").addEventListener("click", () => $("detail-dialog").close());
  $("detail-dialog").addEventListener("click", (event) => {
    if (event.target === $("detail-dialog")) $("detail-dialog").close();
  });
  await refresh();
  startPolling();
}

init();
