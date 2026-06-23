/* AI DJ Set Planner — frontend logic (vanilla JS, no build step).
 *
 * Talks to the Flask JSON API in app.py via fetch. Keeps a small in-memory
 * mirror of the library (tracks + features + constraints) so the central table
 * and the energy curve can render without re-fetching on every interaction.
 */
"use strict";

const DEFAULT_PRESET = "Sofia Day Party Restaurant";

// Constraint options for the per-row dropdown (value -> label).
const CONSTRAINT_OPTIONS = [
  ["", "—"],
  ["MUST_PLAY", "Must play"],
  ["AVOID", "Avoid"],
  ["PREFERRED_INTRO", "Preferred intro"],
  ["PREFERRED_OUTRO", "Preferred outro"],
  ["PREFERRED_PEAK", "Preferred peak"],
];

// One-line character hint per venue (mirrors planning/venue_profiles.py labels).
const VENUE_HINTS = {
  RESTAURANT: "melodic · lounge · gentle",
  BAR: "percussion house · groovy · relaxed",
  BEACH: "sunny · organic · percussive",
  CLUB: "strong · danceable · peak-driven",
  AFTERPARTY: "hypnotic · low-vocal · keep-moving",
  OUTDOOR_DAY_PARTY: "sunny blend · melodic→percussive",
};

function updateVenueHint() {
  const el = $("venue-hint");
  if (el) el.textContent = VENUE_HINTS[$("f-venue").value] || "";
}

// In-memory state.
const state = {
  presets: [],           // [EventProfile-as-dict]
  tracks: [],            // [Track]
  features: {},          // {track_id: TrackFeatures}
  overrides: {},         // {track_id: energy}  (manual energy overrides)
  constraints: {},       // {track_id: constraint_type}  (one per track in UI)
  selected: new Set(),   // selected track ids
  lastPlan: null,        // last generated plan json
};

/* ---------------------------------------------------------------- helpers */
const $ = (id) => document.getElementById(id);

function fmtDuration(seconds) {
  if (seconds == null || isNaN(seconds)) return "—";
  const s = Math.round(seconds);
  const m = Math.floor(s / 60);
  const r = String(s % 60).padStart(2, "0");
  return `${m}:${r}`;
}

function fmtPct(v) {
  if (v == null || isNaN(v)) return "—";
  return v.toFixed(2);
}

function energyColor(v) {
  // Green (calm) -> amber (build) -> red (peak). v in 0..1.
  if (v == null || isNaN(v)) return "#3a4152";
  const hue = Math.max(0, Math.min(140, 140 - v * 140)); // 140=green, 0=red
  return `hsl(${hue}, 62%, 48%)`;
}

function log(message, level = "info") {
  const ul = $("status-log");
  const li = document.createElement("li");
  li.className = `lvl-${level}`;
  const ts = new Date().toLocaleTimeString();
  li.innerHTML = `<span class="ts">${ts}</span><span class="msg"></span>`;
  li.querySelector(".msg").textContent = message;
  ul.insertBefore(li, ul.firstChild);
}

function busy(on, text = "Working…") {
  $("busy-text").textContent = text;
  $("busy").classList.toggle("hidden", !on);
}

async function api(path, { method = "GET", body } = {}) {
  const opts = { method, headers: { "Content-Type": "application/json" } };
  if (body !== undefined) opts.body = JSON.stringify(body);
  const res = await fetch(path, opts);
  let data = {};
  try { data = await res.json(); } catch (_) { /* non-JSON */ }
  if (!res.ok) {
    const msg = (data && data.error) ? data.error : `HTTP ${res.status}`;
    throw new Error(msg);
  }
  return data;
}

/* ---------------------------------------------------------------- presets */
async function loadPresets() {
  const data = await api("/api/presets");
  state.presets = data.presets || [];
  const sel = $("f-preset");
  sel.innerHTML = "";
  for (const p of state.presets) {
    const opt = document.createElement("option");
    opt.value = p.name;
    opt.textContent = p.name;
    sel.appendChild(opt);
  }
  // Default to the Sofia preset and populate the form from it.
  const def = state.presets.find((p) => p.name === DEFAULT_PRESET) || state.presets[0];
  if (def) {
    sel.value = def.name;
    applyPresetToForm(def);
  }
}

function applyPresetToForm(p) {
  $("f-duration").value = p.target_duration_minutes;
  $("f-venue").value = p.venue_type;
  $("f-time").value = p.time_of_day;
  $("f-peak").value = p.peak_strategy;
  updateVenueHint();
  setSlider("f-maxenergy", p.max_energy);
  setSlider("f-minenergy", p.min_energy);
}

function setSlider(id, value) {
  $(id).value = value;
  $(`${id}-val`).textContent = Number(value).toFixed(2);
}

function readProfileFromForm() {
  return {
    preset: $("f-preset").value,            // base preset; fields below override
    target_duration_minutes: Number($("f-duration").value),
    venue_type: $("f-venue").value,
    time_of_day: $("f-time").value,
    peak_strategy: $("f-peak").value,
    max_energy: Number($("f-maxenergy").value),
    min_energy: Number($("f-minenergy").value),
    strict: $("f-strict").checked,
  };
}

/* ---------------------------------------------------------------- library */
async function loadLibrary() {
  const data = await api("/api/tracks");
  state.tracks = data.tracks || [];
  state.features = data.features || {};
  state.overrides = data.overrides || {};
  // Collapse the flat constraint list into one type per track for the UI.
  state.constraints = {};
  for (const c of data.constraints || []) {
    // The row dropdown shows a single selectable constraint; prefer the first.
    if (!(c.track_id in state.constraints)) {
      state.constraints[c.track_id] = c.constraint_type;
    }
  }
  renderLibrary();
}

function renderLibrary() {
  const body = $("library-body");
  $("lib-count").textContent = state.tracks.length
    ? `(${state.tracks.length})` : "";

  if (!state.tracks.length) {
    body.innerHTML =
      `<tr class="empty-row"><td colspan="10">No tracks yet — set a folder and click <strong>Scan</strong>.</td></tr>`;
    return;
  }

  const frag = document.createDocumentFragment();
  for (const t of state.tracks) {
    const f = state.features[t.id] || {};
    const tr = document.createElement("tr");
    if (state.selected.has(t.id)) tr.classList.add("selected-row");

    tr.appendChild(cellCheckbox(t));
    tr.appendChild(cellText(t.title, "cell-title", t.title));
    tr.appendChild(cellText(t.artist || "—"));
    tr.appendChild(cellNum(fmtDuration(t.duration_seconds)));
    tr.appendChild(cellNum(t.bpm != null ? Math.round(t.bpm) : "—"));
    tr.appendChild(cellKey(t.camelot_key || t.musical_key));
    tr.appendChild(cellEnergy(t, f));
    tr.appendChild(cellBar(f.danceability_score));
    tr.appendChild(cellRoleHint(f));
    tr.appendChild(cellConstraint(t));
    frag.appendChild(tr);
  }
  body.innerHTML = "";
  body.appendChild(frag);
}

function cellCheckbox(t) {
  const td = document.createElement("td");
  td.className = "col-sel";
  const cb = document.createElement("input");
  cb.type = "checkbox";
  cb.checked = state.selected.has(t.id);
  cb.addEventListener("change", () => {
    if (cb.checked) state.selected.add(t.id);
    else state.selected.delete(t.id);
    cb.closest("tr").classList.toggle("selected-row", cb.checked);
  });
  td.appendChild(cb);
  return td;
}

function cellText(text, cls = "", title = "") {
  const td = document.createElement("td");
  if (cls) td.className = cls;
  td.textContent = text || "—";
  if (title) td.title = title;
  return td;
}

function cellNum(text) {
  const td = document.createElement("td");
  td.className = "col-num";
  td.textContent = text;
  return td;
}

function cellKey(key) {
  const td = document.createElement("td");
  td.className = "col-key";
  if (key) {
    const span = document.createElement("span");
    span.className = "key-chip";
    span.textContent = key;
    td.appendChild(span);
  } else {
    td.textContent = "—";
  }
  return td;
}

function cellBar(value) {
  const td = document.createElement("td");
  td.className = "col-num";
  if (value == null || isNaN(value)) { td.textContent = "—"; return td; }
  const bar = document.createElement("div");
  bar.className = "bar";
  const fill = document.createElement("i");
  fill.style.width = `${Math.round(value * 100)}%`;
  fill.style.background = energyColor(value);
  const label = document.createElement("b");
  label.textContent = value.toFixed(2);
  bar.appendChild(fill);
  bar.appendChild(label);
  td.appendChild(bar);
  return td;
}

// Compact editable energy cell: a single number input whose left portion is
// tinted to act as the energy bar. Type 0–1 to override; a reset ✕ appears on
// hover only (when overridden) so it never widens the column.
function cellEnergy(t, f) {
  const td = document.createElement("td");
  td.className = "col-num col-energy";
  const v = f.energy_score;
  const overridden = String(t.id) in state.overrides;

  const cell = document.createElement("div");
  cell.className = "energy-cell";

  const inp = document.createElement("input");
  inp.type = "number";
  inp.min = "0"; inp.max = "1"; inp.step = "0.05";
  inp.className = "energy-input" + (overridden ? " overridden" : "");
  inp.value = (v != null && !isNaN(v)) ? Number(v).toFixed(2) : "";
  if (v != null && !isNaN(v)) {
    const pct = Math.round(v * 100);
    inp.style.background =
      `linear-gradient(90deg, ${energyColor(v)}55 ${pct}%, var(--bg-input) ${pct}%)`;
  }
  inp.title = overridden
    ? "Manual energy override — hover the ✕ to reset to the analyzed value"
    : "Type a value 0–1 to override the analyzed energy";
  inp.addEventListener("change", () => onEnergyOverride(t.id, inp.value));
  cell.appendChild(inp);

  if (overridden) {
    const x = document.createElement("button");
    x.type = "button";
    x.className = "energy-reset";
    x.textContent = "✕";
    x.title = "Reset to analyzed value";
    x.addEventListener("click", () => onEnergyClear(t.id));
    cell.appendChild(x);
  }

  td.appendChild(cell);
  return td;
}

async function onEnergyOverride(trackId, raw) {
  const e = Number(raw);
  if (isNaN(e) || e < 0 || e > 1) { log("Energy must be between 0.00 and 1.00.", "error"); return; }
  try {
    const data = await api("/api/features/override", {
      method: "POST", body: { track_id: trackId, energy_score: e },
    });
    state.features[trackId] = data.features;
    state.overrides[String(trackId)] = e;
    log(`Energy of track #${trackId} set to ${e.toFixed(2)} (manual).`, "ok");
    renderLibrary();
  } catch (err) { log(`Energy override failed: ${err.message}`, "error"); }
}

async function onEnergyClear(trackId) {
  try {
    const data = await api("/api/features/override", {
      method: "DELETE", body: { track_id: trackId },
    });
    if (data.features) state.features[trackId] = data.features;
    delete state.overrides[String(trackId)];
    log(`Energy of track #${trackId} reset to the analyzed value.`, "ok");
    renderLibrary();
  } catch (err) { log(`Energy reset failed: ${err.message}`, "error"); }
}

// A cheap "role hint" derived from features (intro/outro/peak suitability).
function cellRoleHint(f) {
  const td = document.createElement("td");
  const hints = [
    ["intro_suitability", "INTRO"],
    ["outro_suitability", "OUTRO"],
    ["peak_potential", "MAIN_PEAK"],
  ];
  let best = "WARM_GROOVE", bestVal = (f.energy_score ?? 0.5);
  for (const [k, role] of hints) {
    if ((f[k] ?? 0) > bestVal) { bestVal = f[k]; best = role; }
  }
  td.appendChild(rolePill(best));
  return td;
}

function rolePill(role) {
  const span = document.createElement("span");
  span.className = `role-pill role-${role}`;
  span.textContent = role.replace(/_/g, " ").toLowerCase();
  return span;
}

function cellConstraint(t) {
  const td = document.createElement("td");
  td.className = "col-constraint";
  const sel = document.createElement("select");
  sel.className = "row-constraint";
  for (const [value, label] of CONSTRAINT_OPTIONS) {
    const opt = document.createElement("option");
    opt.value = value;
    opt.textContent = label;
    sel.appendChild(opt);
  }
  sel.value = state.constraints[t.id] || "";
  sel.addEventListener("change", () => onConstraintChange(t.id, sel.value));
  td.appendChild(sel);
  return td;
}

async function onConstraintChange(trackId, value) {
  try {
    // Clear any prior constraint on this track (UI keeps one at a time), then
    // set the new one (unless cleared to "—").
    await api("/api/constraints", {
      method: "DELETE",
      body: { track_id: trackId },
    });
    if (value) {
      await api("/api/constraints", {
        method: "POST",
        body: { track_id: trackId, constraint_type: value },
      });
      state.constraints[trackId] = value;
      log(`Constraint ${value.toLowerCase()} set on track #${trackId}.`, "ok");
    } else {
      delete state.constraints[trackId];
      log(`Constraint cleared on track #${trackId}.`);
    }
  } catch (err) {
    log(`Constraint update failed: ${err.message}`, "error");
  }
}

/* ---------------------------------------------------------------- set plan */
function renderSet(plan) {
  state.lastPlan = plan;
  const body = $("set-body");

  if (!plan || !plan.tracks || !plan.tracks.length) {
    body.innerHTML =
      `<tr class="empty-row"><td colspan="10">No set yet — click <strong>Generate Set</strong>.</td></tr>`;
    $("set-summary").textContent = "";
    return;
  }

  const total = fmtDuration(plan.total_duration_seconds);
  const target = fmtDuration(plan.target_duration_seconds);
  $("set-summary").textContent =
    `${plan.tracks.length} tracks · ${total} / ${target} · score ${fmtPct(plan.total_score)}`;

  const frag = document.createDocumentFragment();
  for (const r of plan.tracks) {
    const tr = document.createElement("tr");
    tr.title = r.explanation || "";   // full reasoning on hover (was a column)
    tr.appendChild(cellNum(r.position + 1));
    tr.appendChild(cellText(r.title, "cell-title", r.title));
    tr.appendChild(cellText(r.artist || "—"));
    const roleTd = document.createElement("td");
    roleTd.appendChild(rolePill(r.role));
    if (r.is_locked) {
      const lock = document.createElement("span");
      lock.className = "lock-badge";
      lock.textContent = "🔒";
      roleTd.appendChild(lock);
    }
    tr.appendChild(roleTd);
    tr.appendChild(cellNum(fmtDuration(r.duration_seconds)));
    tr.appendChild(cellNum(r.bpm != null ? Math.round(r.bpm) : "—"));
    tr.appendChild(cellKey(r.key));
    tr.appendChild(cellBar(r.energy_score));
    tr.appendChild(cellNum(fmtPct(r.transition_score)));
    frag.appendChild(tr);
  }
  body.innerHTML = "";
  body.appendChild(frag);

  drawCurve(plan);
}

/* ---------------------------------------------------------------- curve */
function drawCurve(plan) {
  const host = $("curve-host");
  const pts = plan.energy_points || [];
  const segments = plan.segments || [];

  if (!pts.length) {
    host.innerHTML = `<p class="muted curve-empty">No energy data to plot.</p>`;
    return;
  }

  // Viewbox geometry. We draw target-energy bands (per segment, spanning their
  // start_pct..end_pct horizontally and min..max energy vertically) first, then
  // the actual energy line on top, with energy 0 at the bottom and 1 at the top.
  const W = 1000, H = 220;
  const padL = 34, padR = 12, padT = 12, padB = 22;
  const innerW = W - padL - padR;
  const innerH = H - padT - padB;

  const x = (frac) => padL + frac * innerW;
  const y = (energy) => padT + (1 - energy) * innerH;

  const parts = [];
  parts.push(`<svg viewBox="0 0 ${W} ${H}" preserveAspectRatio="none" xmlns="http://www.w3.org/2000/svg">`);

  // Horizontal gridlines + y-axis labels at 0, 0.5, 1.
  for (const g of [0, 0.25, 0.5, 0.75, 1]) {
    parts.push(`<line class="curve-axis" x1="${padL}" y1="${y(g)}" x2="${W - padR}" y2="${y(g)}" opacity="0.35" />`);
    parts.push(`<text class="axis-label" x="4" y="${y(g) + 3}">${g.toFixed(2)}</text>`);
  }

  // Target-energy bands per segment.
  for (const s of segments) {
    const sx = x(s.start_pct);
    const sw = (s.end_pct - s.start_pct) * innerW;
    const top = y(s.max_energy);
    const bandH = (s.max_energy - s.min_energy) * innerH;
    parts.push(`<rect class="seg-band" x="${sx.toFixed(1)}" y="${top.toFixed(1)}" width="${sw.toFixed(1)}" height="${Math.max(1, bandH).toFixed(1)}" />`);
    parts.push(`<line class="seg-divider" x1="${sx.toFixed(1)}" y1="${padT}" x2="${sx.toFixed(1)}" y2="${H - padB}" />`);
    // Segment role label near the top of the band.
    const label = (s.role || s.name || "").replace(/_/g, " ");
    parts.push(`<text class="seg-label" x="${(sx + 3).toFixed(1)}" y="${(padT + 9)}">${escapeXml(label)}</text>`);
  }

  // Actual energy polyline across the ordered tracks.
  const n = pts.length;
  const linePts = pts.map((e, i) => {
    const frac = n === 1 ? 0 : i / (n - 1);
    return `${x(frac).toFixed(1)},${y(e).toFixed(1)}`;
  });
  parts.push(`<polyline class="curve-line" points="${linePts.join(" ")}" />`);
  // Dots at each track.
  pts.forEach((e, i) => {
    const frac = n === 1 ? 0 : i / (n - 1);
    parts.push(`<circle class="curve-dot" cx="${x(frac).toFixed(1)}" cy="${y(e).toFixed(1)}" r="2.4" />`);
  });

  parts.push(`</svg>`);
  host.innerHTML = parts.join("");
}

function escapeXml(s) {
  return String(s).replace(/[<>&]/g, (c) => ({ "<": "&lt;", ">": "&gt;", "&": "&amp;" }[c]));
}

/* ---------------------------------------------------------------- actions */
async function doScan() {
  const folder = $("folder-input").value.trim();
  if (!folder) { log("Enter a music folder path first.", "error"); return; }
  busy(true, "Scanning folder…");
  try {
    const data = await api("/api/scan", { method: "POST", body: { folder } });
    log(`Scanned ${data.tracks.length} track(s) from ${folder}.`, "ok");
    await loadLibrary();
  } catch (err) {
    log(`Scan failed: ${err.message}`, "error");
  } finally { busy(false); }
}

async function doAnalyze() {
  if (!state.tracks.length) { log("Nothing to analyze — scan a folder first.", "error"); return; }
  const ids = state.selected.size ? [...state.selected] : null;
  busy(true, "Analyzing tracks…");
  try {
    const data = await api("/api/analyze", { method: "POST", body: { track_ids: ids } });
    const count = Object.keys(data.features || {}).length;
    log(`Analyzed ${count} track(s) using ${data.engine}.`, "ok");
    await loadLibrary();
  } catch (err) {
    log(`Analyze failed: ${err.message}`, "error");
  } finally { busy(false); }
}

async function doImport() {
  const xmlPath = prompt("Path to Rekordbox collection XML:", "");
  if (!xmlPath) return;
  busy(true, "Importing Rekordbox XML…");
  try {
    const data = await api("/api/rekordbox-import", { method: "POST", body: { xml_path: xmlPath.trim() } });
    log(`Rekordbox: matched ${data.matched} track(s).`, "ok");
    await loadLibrary();
  } catch (err) {
    log(`Rekordbox import failed: ${err.message}`, "error");
  } finally { busy(false); }
}

async function doGenerate() {
  if (!state.tracks.length) { log("Scan a folder before generating a set.", "error"); return; }
  busy(true, "Generating set…");
  try {
    const profile = readProfileFromForm();
    const data = await api("/api/generate", { method: "POST", body: { profile } });
    renderSet(data.plan);
    log(`Generated a ${data.plan.tracks.length}-track set (score ${fmtPct(data.plan.total_score)}).`, "ok");
  } catch (err) {
    log(`Generate failed: ${err.message}`, "error");
  } finally { busy(false); }
}

async function doExport(fmt) {
  if (!state.lastPlan) { log("Generate a set before exporting.", "error"); return; }
  busy(true, `Exporting ${fmt.toUpperCase()}…`);
  try {
    const data = await api(`/api/export/${fmt}`, { method: "POST", body: {} });
    log(`Exported ${fmt.toUpperCase()} → ${data.path}`, "ok");
  } catch (err) {
    log(`Export failed: ${err.message}`, "error");
  } finally { busy(false); }
}

/* ---------------------------------------------------------------- wiring */
function wire() {
  $("btn-scan").addEventListener("click", doScan);
  $("btn-analyze").addEventListener("click", doAnalyze);
  $("btn-import").addEventListener("click", doImport);
  $("btn-generate").addEventListener("click", doGenerate);
  $("btn-export-m3u").addEventListener("click", () => doExport("m3u"));
  $("btn-export-csv").addEventListener("click", () => doExport("csv"));
  $("btn-clear-log").addEventListener("click", () => { $("status-log").innerHTML = ""; });

  // Preset dropdown repopulates the form.
  $("f-preset").addEventListener("change", () => {
    const p = state.presets.find((x) => x.name === $("f-preset").value);
    if (p) applyPresetToForm(p);
  });

  // Venue dropdown updates the character hint.
  $("f-venue").addEventListener("change", updateVenueHint);

  // Live slider value readouts.
  for (const id of ["f-maxenergy", "f-minenergy"]) {
    $(id).addEventListener("input", () => {
      $(`${id}-val`).textContent = Number($(id).value).toFixed(2);
    });
  }
}

async function init() {
  wire();
  try {
    await loadPresets();
    await loadLibrary();
    log("Ready. Set a music folder and click Scan to begin.", "ok");
  } catch (err) {
    log(`Startup error: ${err.message}`, "error");
  }
}

document.addEventListener("DOMContentLoaded", init);
