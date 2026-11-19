// Real-Time Traffic Analytics dashboard (block 10). Plain JS + Leaflet + Chart.js.
"use strict";

const CFG = window.APP_CONFIG || {};
const API = (CFG.apiBase || "").replace(/\/$/, "");
const LEVELS = ["free", "moderate", "heavy", "severe", "unknown"];
const LEVEL_LABEL = { free: "Free", moderate: "Moderate", heavy: "Heavy", severe: "Severe", unknown: "No data" };
const css = (v) => getComputedStyle(document.documentElement).getPropertyValue(v).trim();
const COLOR = () => ({ free: css("--good"), moderate: css("--warning"), heavy: css("--serious"),
                       severe: css("--critical"), unknown: css("--unknown") });
// slider positions: -12..0 = past 5-min slots, 1..3 = forecast horizons
const FUTURE = { 1: 15, 2: 30, 3: 60 };
const MPS = 0.44704;

const state = { meta: {}, latest: null, view: [], viewKind: "now", selected: null, markers: {}, tick: 300,
                timer: null, chat: [] };

async function api(path, opts) {
  const r = await fetch(API + path, opts);
  if (!r.ok) throw new Error(`${path}: HTTP ${r.status}`);
  return r.json();
}

const fmtTime = (ts) => (ts || "").slice(11, 16);
const fmtS = (s) => (s == null ? "–" : `${Math.round(s)} s`);
const addMin = (ts, m) => {
  const d = new Date(ts + "Z"); d.setUTCMinutes(d.getUTCMinutes() + m);
  return d.toISOString().slice(0, 19);
};
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

// ------------------------------------------------------------------ map
const map = L.map("map", { zoomControl: true }).setView([34.1, -118.35], 11);
L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
  maxZoom: 18, attribution: "&copy; OpenStreetMap contributors",
}).addTo(map);

function drawLegend() {
  const c = COLOR();
  document.getElementById("map-legend").innerHTML = LEVELS.map((l) =>
    `<span><i class="dot ${l}"></i>${LEVEL_LABEL[l]}</span>`).join("") +
    `<span>⚠ synthetic incident</span>`;
  return c;
}

function renderMap() {
  const c = COLOR();
  const fut = state.viewKind === "forecast";
  document.querySelector(".map-card").classList.toggle("forecast", fut);
  const byId = Object.fromEntries(state.view.map((v) => [v.sensor_id, v]));
  for (const [sid, m] of Object.entries(state.meta)) {
    const v = byId[sid];
    const lvl = v ? v.level || "unknown" : "unknown";
    const style = { radius: v && (lvl === "heavy" || lvl === "severe") ? 9 : 7, weight: 2,
                    color: v && v.incident_flag ? "#6250d6" : css("--surface"),
                    fillColor: c[lvl], fillOpacity: 0.95, dashArray: fut ? "3 3" : null };
    let mk = state.markers[sid];
    if (!mk) {
      mk = L.circleMarker([m.lat, m.lon], style).addTo(map);
      mk.on("click", () => selectSensor(sid));
      state.markers[sid] = mk;
    } else mk.setStyle(style);
    const speed = v && v.speed_mph != null ? `${v.speed_mph.toFixed(1)} mph` : "no reading";
    mk.bindTooltip(`<b>Sensor ${sid}</b>${fut ? " · forecast" : ""}<br>${LEVEL_LABEL[lvl]} · ${speed}` +
      `<br>free-flow ${Number(m.free_flow_mph).toFixed(1)} mph${v && v.delay_s != null ? ` · delay ${v.delay_s > 0 ? "+" : ""}${Math.round(v.delay_s)} s` : ""}` +
      `${v && v.incident_flag ? "<br>⚠ synthetic incident" : ""}`);
  }
}

// ------------------------------------------------------------------ summary + table
function renderSummary(summary) {
  const s = summary || { by_level: {}, total_km: 0, pct_congested: 0, incidents: 0 };
  document.getElementById("pct").textContent = `${(s.pct_congested || 0).toFixed(1)}%`;
  document.getElementById("km-total").textContent = `${(s.congested_km || 0).toFixed(1)} of ${s.total_km} km monitored`;
  document.getElementById("incidents").textContent = s.incidents ?? 0;
  document.getElementById("level-list").innerHTML = LEVELS.filter((l) => s.by_level[l]).map((l) =>
    `<li><i class="dot ${l}"></i>${LEVEL_LABEL[l]}<span class="km">${s.by_level[l].count} · ${s.by_level[l].km.toFixed(1)} km</span></li>`).join("");
}

function freeFlowTT(m) { return m.segment_length_m / (m.free_flow_mph * MPS); }

function renderTable() {
  const c = COLOR();
  const showAll = document.getElementById("show-all").checked;
  const q = document.getElementById("search").value.trim();
  const rows = state.view
    .filter((v) => state.meta[v.sensor_id])
    .filter((v) => showAll || v.level === "heavy" || v.level === "severe")
    .filter((v) => !q || v.sensor_id.includes(q))
    .sort((a, b) => (b.delay_s ?? -1e9) - (a.delay_s ?? -1e9));
  const tb = document.querySelector("#roads tbody");
  if (!rows.length) {
    tb.innerHTML = `<tr><td colspan="6" class="muted">No ${showAll ? "" : "heavy or severe "}segments ${state.viewKind === "forecast" ? "forecast" : "right now"}.</td></tr>`;
    return;
  }
  tb.innerHTML = rows.map((v) => {
    const m = state.meta[v.sensor_id];
    const ff = freeFlowTT(m);
    const max = Math.max(ff, v.expected_tt_s || 0, v.travel_time_s || 0) * 1.05 || 1;
    const bar = (val, lvl) => `<div class="bar"><i style="width:${Math.min(100, (val || 0) / max * 100)}%;background:${c[lvl]}"></i></div>`;
    const lvl = v.level || "unknown";
    const delay = v.delay_s == null ? "–" : `${v.delay_s > 0 ? "+" : ""}${Math.round(v.delay_s)} s`;
    return `<tr data-sid="${esc(v.sensor_id)}" class="${v.sensor_id === state.selected ? "selected" : ""}">
      <td><div class="seg-name">Sensor ${esc(v.sensor_id)} ${v.incident_flag ? '<span class="tag synthetic">⚠ synthetic incident</span>' : ""}</div>
          <div class="seg-sub">${m.lat.toFixed(4)}, ${m.lon.toFixed(4)} · ${LEVEL_LABEL[lvl]}</div></td>
      <td class="num">${Math.round(m.segment_length_m)} m</td>
      <td class="num">${fmtS(ff)}<div class="seg-sub">${Number(m.free_flow_mph).toFixed(1)} mph</div></td>
      <td class="num">${fmtS(v.expected_tt_s)}${bar(v.expected_tt_s, "free")}</td>
      <td class="num">${fmtS(v.travel_time_s)} <span class="seg-sub">${v.speed_mph != null ? v.speed_mph.toFixed(0) + " mph" : ""}</span>${bar(v.travel_time_s, lvl)}</td>
      <td class="num"><span class="delay ${lvl}">${delay}</span></td></tr>`;
  }).join("");
  tb.querySelectorAll("tr[data-sid]").forEach((tr) => tr.addEventListener("click", () => selectSensor(tr.dataset.sid)));
}

// ------------------------------------------------------------------ chart
let chart;
async function renderChart() {
  const sid = state.selected;
  if (!sid) return;
  document.getElementById("chart-title").textContent = `Sensor ${sid}`;
  const h = await api(`/history?sensor_id=${encodeURIComponent(sid)}`);
  const m = state.meta[sid];
  const pts = h.points || [];
  const fc = h.forecast || [];
  const labels = pts.map((p) => p.ts).concat(fc.map((f) => f.target_ts));
  const actual = pts.map((p) => p.travel_time_s).concat(fc.map(() => null));
  const expected = pts.map((p) => p.expected_tt_s).concat(fc.map((f) => f.expected_tt_s));
  const freeflow = labels.map(() => (m ? freeFlowTT(m) : null));
  const forecast = pts.map((p, i) => (i === pts.length - 1 ? p.travel_time_s : null)).concat(fc.map((f) => f.travel_time_s));
  const levelsByIdx = pts.map((p) => p.level).concat(fc.map((f) => f.level));
  const ink2 = css("--ink-2"), grid = css("--grid"), muted = css("--muted");
  const s1 = css("--series-1"), s2 = css("--series-2");
  const data = {
    labels,
    datasets: [
      { label: "Actual", data: actual, borderColor: s1, backgroundColor: s1, borderWidth: 2, pointRadius: 2.5, pointHoverRadius: 5, tension: 0.2 },
      { label: "Forecast", data: forecast, borderColor: s1, backgroundColor: css("--surface"), borderDash: [6, 4], borderWidth: 2,
        pointRadius: 4, pointBorderWidth: 2, spanGaps: true },
      { label: "Expected (usual)", data: expected, borderColor: s2, borderWidth: 2, pointRadius: 0, tension: 0.2 },
      { label: "Free-flow", data: freeflow, borderColor: muted, borderWidth: 1.5, borderDash: [2, 3], pointRadius: 0 },
    ],
  };
  const options = {
    responsive: true, maintainAspectRatio: false, animation: false,
    interaction: { mode: "index", intersect: false },
    plugins: {
      legend: { position: "bottom", labels: { color: ink2, boxWidth: 18, usePointStyle: false } },
      tooltip: { callbacks: {
        title: (items) => { const i = items[0].dataIndex; return `${labels[i].replace("T", " ")}${i >= pts.length ? " (forecast)" : ""}`; },
        afterTitle: (items) => LEVEL_LABEL[levelsByIdx[items[0].dataIndex] || "unknown"],
        label: (it) => (it.raw == null ? null : `${it.dataset.label}: ${Math.round(it.raw)} s`),
      } },
    },
    scales: {
      x: { ticks: { color: muted, maxTicksLimit: 8, callback: (v) => fmtTime(labels[v]) }, grid: { display: false } },
      y: { beginAtZero: true, title: { display: true, text: "seconds", color: muted }, ticks: { color: muted }, grid: { color: grid } },
    },
  };
  if (chart) { chart.data = data; chart.options = options; chart.update(); }
  else chart = new Chart(document.getElementById("chart"), { type: "line", data, options });
}

function selectSensor(sid) {
  state.selected = sid;
  renderTable();
  renderChart().catch(console.error);
}

// ------------------------------------------------------------------ timeline
function sliderMinutes() {
  const v = Number(document.getElementById("timeline").value);
  return v > 0 ? FUTURE[v] : v * 5;
}

function drawTicks() {
  const el = document.getElementById("ticks");
  const input = document.getElementById("timeline");
  const min = Number(input.min), max = Number(input.max);
  const marks = [[-12, "-60"], [-9, "-45"], [-6, "-30"], [-3, "-15"], [0, "Now"], [1, "+15"], [2, "+30"], [3, "+60"]];
  el.style.position = "relative"; el.style.height = "16px";
  el.innerHTML = marks.map(([v, t]) =>
    `<span class="${v > 0 ? "future" : ""}" style="position:absolute;left:${(v - min) / (max - min) * 100}%;transform:translateX(${v === min ? "0" : v === max ? "-100%" : "-50%"})">${t}</span>`).join("");
}

async function loadView() {
  const mins = sliderMinutes();
  const badge = document.getElementById("mode-badge");
  const asOf = state.latest?.as_of;
  let view, summary, label;
  if (mins === 0 || !asOf) {
    view = state.latest?.sensors || []; summary = state.latest?.summary; state.viewKind = "now";
    badge.className = "badge live"; badge.textContent = "LIVE"; label = `Now (${fmtTime(asOf)})`;
  } else if (mins < 0) {
    const ts = addMin(asOf, mins);
    const h = await api(`/history?ts=${ts}`);
    view = h.sensors; summary = h.summary; state.viewKind = "past";
    badge.className = "badge past"; badge.textContent = "HISTORY"; label = `${mins} min (${fmtTime(ts)})`;
  } else {
    const f = await api(`/forecast?horizon=${mins}`);
    view = f.items || []; summary = f.summary; state.viewKind = "forecast";
    badge.className = "badge forecast"; badge.textContent = `FORECAST +${mins} min`;
    label = f.available ? `+${mins} min (${fmtTime(f.target_ts)}) · ${f.model}` : "forecast not available yet";
  }
  state.view = view;
  document.getElementById("view-label").textContent = label;
  document.getElementById("table-when").textContent = state.viewKind === "now" ? "· live" : `· ${label}`;
  renderSummary(summary);
  renderMap();
  renderTable();
}

// ------------------------------------------------------------------ alerts + chat
async function loadAlerts() {
  const a = await api("/alerts?limit=30");
  const ul = document.getElementById("alerts");
  if (!a.alerts.length) { ul.innerHTML = `<li class="muted">No alerts yet.</li>`; return; }
  ul.innerHTML = a.alerts.map((x) => {
    const when = new Date(Number(String(x.sk).split("#")[0])).toLocaleTimeString();
    return `<li class="${esc(x.severity)}"><div class="when">${esc(when)} · ${esc((x.severity || "").toUpperCase())}${x.sent ? " · emailed" : ""}</div>${esc(x.message)}</li>`;
  }).join("");
}

function addMsg(role, text, extra) {
  const div = document.createElement("div");
  div.className = `msg ${role}`;
  div.textContent = text;
  if (extra) { const s = document.createElement("span"); s.className = "tools"; s.textContent = extra; div.appendChild(s); }
  const log = document.getElementById("chat-log");
  log.appendChild(div); log.scrollTop = log.scrollHeight;
}

document.getElementById("chat-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const input = document.getElementById("chat-input");
  const msg = input.value.trim() || input.placeholder;
  input.value = "";
  addMsg("user", msg);
  const btn = e.target.querySelector("button"); btn.disabled = true;
  try {
    const r = await api("/agent/chat", { method: "POST", headers: { "Content-Type": "application/json" },
                                          body: JSON.stringify({ message: msg, history: state.chat.slice(-6) }) });
    const tools = (r.tool_calls || []).map((t) => `${t.name}(${JSON.stringify(t.args)})`).join(", ");
    addMsg("bot", r.answer || r.error || "(no answer)", tools ? `Tool calls: ${tools}` : "No tool calls");
    document.getElementById("chat-provider").textContent = r.provider ? `LLM: ${r.provider}` : "";
    state.chat.push({ role: "user", content: msg }, { role: "assistant", content: r.answer || "" });
  } catch (err) { addMsg("bot", `Error: ${err.message}`); }
  finally { btn.disabled = false; }
});

// ------------------------------------------------------------------ refresh loop
async function refresh() {
  try {
    state.latest = await api("/latest");
    state.tick = state.latest.tick_seconds || CFG.tickSeconds || 300;
    document.getElementById("as-of").textContent = (state.latest.as_of || "–").replace("T", " ");
    await loadView();
    if (!state.selected && state.view.length) {
      const worst = [...state.view].sort((a, b) => (b.delay_s ?? -1e9) - (a.delay_s ?? -1e9))[0];
      state.selected = worst.sensor_id;
    }
    await Promise.all([renderChart(), loadAlerts()]);
    document.getElementById("refresh-note").textContent = `auto-refresh every ${Math.max(5, state.tick)} s`;
  } catch (err) {
    console.error(err);
    document.getElementById("refresh-note").textContent = `API error: ${err.message}`;
  }
  clearTimeout(state.timer);
  state.timer = setTimeout(refresh, Math.max(5, state.tick) * 1000);
}

async function init() {
  drawLegend();
  drawTicks();
  if (!API && CFG.mode !== "local") {
    document.getElementById("refresh-note").textContent = "config.js has no apiBase: run `python tasks.py deploy`";
  }
  const s = await api("/sensors");
  state.meta = Object.fromEntries(s.sensors.map((m) => [m.sensor_id, m]));
  const pts = s.sensors.map((m) => [m.lat, m.lon]);
  if (pts.length) map.fitBounds(pts, { padding: [20, 20] });
  const t0 = new URLSearchParams(location.search).get("t");   // e.g. ?t=2 opens on the +30 min forecast
  if (t0 !== null) document.getElementById("timeline").value = t0;
  document.getElementById("timeline").addEventListener("input", () => loadView().catch(console.error));
  document.getElementById("show-all").addEventListener("change", renderTable);
  document.getElementById("search").addEventListener("input", renderTable);
  api("/model").then((r) => {
    const m = r.model;
    document.getElementById("model-info").textContent = m
      ? `XGBoost (${m.trained_on}) · MAE ${Object.entries(m.mae).map(([h, v]) => `+${h}′ ${v}`).join(" · ")} mph`
      : "no model published: historical-average baseline";
  }).catch(() => {});
  refresh();
}

init().catch((err) => { document.getElementById("refresh-note").textContent = `API error: ${err.message}`; });
