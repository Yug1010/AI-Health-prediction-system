/* ═══════════════════════════════════════════════════════════════════════════
   HealthMind AI – Frontend JS (v2)
   Handles: patient profile, vitals live feedback, symptom selection, results
   ═══════════════════════════════════════════════════════════════════════════ */

"use strict";

// ── State ─────────────────────────────────────────────────────────────────────
const state = {
  allSymptoms: [],
  allCategories: {},
  preexistingOpts: [],
  familyHistoryOpts: [],
  selected: new Set(),
  selectedPreex: new Set(),
  selectedFamilyHist: new Set(),
  reportFindings: [],      // extracted findings from uploaded report
  gaugeChart: null,
  explainChart: null,
  vitalsRanges: null,
};

// Vitals ranges (mirrored client-side for instant feedback)
const VITALS_RANGES_CLIENT = {
  heart_rate: [[0, 39, "critical", "#E63946"], [40, 59, "warning", "#F7B731"], [60, 100, "normal", "#02C39A"], [101, 120, "warning", "#F7B731"], [121, 9999, "critical", "#E63946"]],
  systolic_bp: [[0, 89, "warning", "#F7B731"], [90, 119, "normal", "#02C39A"], [120, 129, "warning", "#F7B731"], [130, 139, "warning", "#E67E22"], [140, 179, "danger", "#E63946"], [180, 9999, "critical", "#E63946"]],
  diastolic_bp: [[0, 59, "warning", "#F7B731"], [60, 79, "normal", "#02C39A"], [80, 89, "warning", "#E67E22"], [90, 119, "danger", "#E63946"], [120, 9999, "critical", "#E63946"]],
  temperature: [[0, 95.9, "critical", "#E63946"], [96, 97.9, "warning", "#F7B731"], [98, 99, "normal", "#02C39A"], [99.1, 100.3, "warning", "#F7B731"], [100.4, 103, "danger", "#E67E22"], [103.1, 9999, "critical", "#E63946"]],
  spo2: [[0, 89, "critical", "#E63946"], [90, 93, "danger", "#E67E22"], [94, 95, "warning", "#F7B731"], [96, 100, "normal", "#02C39A"]],
  respiratory_rate: [[0, 11, "warning", "#F7B731"], [12, 20, "normal", "#02C39A"], [21, 24, "warning", "#E67E22"], [25, 9999, "critical", "#E63946"]],
};
const VITALS_LABELS = {
  heart_rate: "Normal: 60–100 bpm", systolic_bp: "Normal: 90–119 mmHg",
  diastolic_bp: "Normal: 60–79 mmHg", temperature: "Normal: 98–99°F",
  spo2: "Normal: 96–100%", respiratory_rate: "Normal: 12–20 br/min",
};

// ── DOM refs ──────────────────────────────────────────────────────────────────
const $ = id => document.getElementById(id);

const searchInput = $("symptom-search");
const searchResults = $("search-results");
const selectedTags = $("selected-tags");
const selectedCount = $("selected-count");
const clearBtn = $("clear-btn");
const categoriesPanel = $("categories-panel");
const toggleCats = $("toggle-categories");
const analyseBtn = $("analyse-btn");
const emptyState = $("empty-state");
const loadingState = $("loading-state");
const resultsContent = $("results-content");
const emergencyAlert = $("emergency-alert");
const riskNumber = $("risk-number");
const riskLabelBadge = $("risk-label-badge");
const confidenceLbl = $("confidence-label");

// ── Boot ──────────────────────────────────────────────────────────────────────
async function init() {
  const res = await fetch("/api/meta");
  const data = await res.json();
  state.allSymptoms = data.all_symptoms;
  state.allCategories = data.categories;
  state.preexistingOpts = data.preexisting;
  state.familyHistoryOpts = data.family_history;
  buildPreexisting();
  buildFamilyHistory();
  buildCategories();
  setupVitalsListeners();
  setupValidation();
}

// ── Patient profile validation ─────────────────────────────────────────────
function setupValidation() {
  $("input-age").addEventListener("input", updateAnalyseBtn);
}
function updateAnalyseBtn() {
  const age = parseInt($("input-age").value);
  const ok = age >= 1 && age <= 120 && state.selected.size >= 2;
  analyseBtn.disabled = !ok;
}

// ── Pre-existing condition chips ───────────────────────────────────────────
function buildPreexisting() {
  const wrap = $("preexisting-chips");
  wrap.innerHTML = "";
  state.preexistingOpts.forEach(c => {
    const chip = document.createElement("button");
    chip.className = "cond-chip";
    chip.dataset.id = c.id;
    chip.innerHTML = `<span class="chip-icon">${c.icon}</span>${c.label}`;
    chip.addEventListener("click", () => {
      if (state.selectedPreex.has(c.id)) state.selectedPreex.delete(c.id);
      else state.selectedPreex.add(c.id);
      chip.classList.toggle("selected", state.selectedPreex.has(c.id));
    });
    wrap.appendChild(chip);
  });
}

// ── Family history chips ───────────────────────────────────────────────────
function buildFamilyHistory() {
  const wrap = $("family-history-chips");
  if (!wrap) return;
  wrap.innerHTML = "";
  state.familyHistoryOpts.forEach(c => {
    const chip = document.createElement("button");
    chip.className = "cond-chip fh-chip";
    chip.dataset.id = c.id;
    chip.innerHTML = `<span class="chip-icon">${c.icon}</span>${c.label}`;
    chip.addEventListener("click", () => {
      if (state.selectedFamilyHist.has(c.id)) state.selectedFamilyHist.delete(c.id);
      else state.selectedFamilyHist.add(c.id);
      chip.classList.toggle("selected", state.selectedFamilyHist.has(c.id));
    });
    wrap.appendChild(chip);
  });
}

// ── Vitals live feedback ───────────────────────────────────────────────────
function setupVitalsListeners() {
  const map = {
    "v-hr": { key: "heart_rate", pill: "v-hr-status" },
    "v-sys": { key: "systolic_bp", pill: "v-bp-status" },
    "v-dia": { key: "diastolic_bp", pill: "v-bp-status" },
    "v-temp": { key: "temperature", pill: "v-temp-status" },
    "v-spo2": { key: "spo2", pill: "v-spo2-status" },
    "v-rr": { key: "respiratory_rate", pill: "v-rr-status" },
  };
  Object.entries(map).forEach(([inputId, { key, pill }]) => {
    $(inputId).addEventListener("input", () => {
      const val = parseFloat($(inputId).value);
      if (!val) { $(pill).textContent = ""; $(pill).style.background = ""; return; }
      const status = getVitalStatus(key, val);
      const pillEl = $(pill);
      pillEl.textContent = status.label;
      pillEl.style.background = status.color + "20";
      pillEl.style.color = status.color;
      pillEl.style.border = `1px solid ${status.color}50`;
      pillEl.style.padding = ".15rem .5rem";
      pillEl.style.borderRadius = "50px";
    });
  });
}

function getVitalStatus(key, value) {
  const ranges = VITALS_RANGES_CLIENT[key] || [];
  for (const [min, max, status, color] of ranges) {
    if (value >= min && value <= max) {
      const labels = {
        normal: "Normal ✓", warning: "Abnormal ⚠", danger: "High Risk !", critical: "Critical 🚨"
      };
      return { status, color, label: labels[status] || status };
    }
  }
  return { status: "unknown", color: "#64748B", label: "—" };
}

// ── Category panel ─────────────────────────────────────────────────────────
function buildCategories() {
  categoriesPanel.innerHTML = "";
  for (const [catName, symptoms] of Object.entries(state.allCategories)) {
    const group = document.createElement("div");
    group.className = "category-group";
    group.innerHTML = `<span class="category-label">${catName}</span>
      <div class="category-chips"></div>`;
    const chips = group.querySelector(".category-chips");
    symptoms.forEach(s => {
      const chip = document.createElement("button");
      chip.className = "cat-chip";
      chip.dataset.id = s.id;
      chip.textContent = s.label;
      if (state.selected.has(s.id)) chip.classList.add("selected");
      chip.addEventListener("click", () => toggleSymptom(s.id));
      chips.appendChild(chip);
    });
    categoriesPanel.appendChild(group);
  }
}

toggleCats.addEventListener("click", () => {
  const hidden = categoriesPanel.classList.toggle("hidden");
  toggleCats.textContent = hidden ? "Show ▼" : "Hide ▲";
});

// ── Symptom search ─────────────────────────────────────────────────────────
searchInput.addEventListener("input", () => {
  const q = searchInput.value.trim().toLowerCase();
  if (!q) { searchResults.classList.add("hidden"); return; }
  const matches = state.allSymptoms.filter(s =>
    s.label.toLowerCase().includes(q) || s.id.includes(q)
  );
  if (!matches.length) {
    searchResults.innerHTML = `<div class="no-results">No matching symptoms found</div>`;
  } else {
    searchResults.innerHTML = matches.slice(0, 8).map(s => {
      const already = state.selected.has(s.id);
      return `<div class="search-result-item ${already ? "already-selected" : ""}"
                   data-id="${s.id}" data-label="${s.label}">
        <span>${s.label}</span>
        ${already
          ? '<span class="already-badge">✓ Added</span>'
          : '<span style="color:var(--teal);font-size:.8rem">+ Add</span>'}
      </div>`;
    }).join("");
    searchResults.querySelectorAll(".search-result-item:not(.already-selected)").forEach(el => {
      el.addEventListener("click", () => {
        toggleSymptom(el.dataset.id);
        searchInput.value = "";
        searchResults.classList.add("hidden");
      });
    });
  }
  searchResults.classList.remove("hidden");
});

document.addEventListener("click", e => {
  if (!e.target.closest(".search-wrap") && !e.target.closest(".search-results"))
    searchResults.classList.add("hidden");
});

// ── Symptom toggle ─────────────────────────────────────────────────────────
function toggleSymptom(id) {
  if (state.selected.has(id)) state.selected.delete(id);
  else state.selected.add(id);
  updateTagsUI();
  updateCategoryChips();
  updateAnalyseBtn();
}

function updateTagsUI() {
  const count = state.selected.size;
  selectedCount.textContent = count;
  clearBtn.classList.toggle("hidden", count === 0);
  if (count === 0) {
    selectedTags.innerHTML = '<span class="tag-placeholder">No symptoms selected yet</span>';
    return;
  }
  selectedTags.innerHTML = "";
  state.selected.forEach(id => {
    const sym = state.allSymptoms.find(s => s.id === id);
    const label = sym ? sym.label : id;
    const tag = document.createElement("span");
    tag.className = "symptom-tag";
    tag.innerHTML = `${label} <span class="tag-remove" data-id="${id}">✕</span>`;
    tag.querySelector(".tag-remove").addEventListener("click", () => {
      state.selected.delete(id);
      updateTagsUI(); updateCategoryChips(); updateAnalyseBtn();
    });
    selectedTags.appendChild(tag);
  });
}

function updateCategoryChips() {
  document.querySelectorAll(".cat-chip").forEach(chip => {
    chip.classList.toggle("selected", state.selected.has(chip.dataset.id));
  });
}

clearBtn.addEventListener("click", () => {
  state.selected.clear(); updateTagsUI(); updateCategoryChips(); updateAnalyseBtn();
});

// ── Analyse ────────────────────────────────────────────────────────────────
analyseBtn.addEventListener("click", runAnalysis);

async function runAnalysis() {
  emptyState.classList.add("hidden");
  resultsContent.classList.add("hidden");
  loadingState.classList.remove("hidden");
  $("results-panel").scrollIntoView({ behavior: "smooth", block: "start" });

  // Gather vitals
  const vitals = {};
  const vmap = {
    "v-hr": "heart_rate", "v-sys": "systolic_bp", "v-dia": "diastolic_bp",
    "v-temp": "temperature", "v-spo2": "spo2", "v-rr": "respiratory_rate",
  };
  for (const [inputId, key] of Object.entries(vmap)) {
    const val = $(`${inputId}`).value;
    if (val !== "") vitals[key] = parseFloat(val);
  }

  const payload = {
    symptoms: [...state.selected],
    age: parseInt($("input-age").value) || null,
    gender: $("input-gender").value || null,
    preexisting: [...state.selectedPreex],
    family_history: [...state.selectedFamilyHist],
    report_findings: state.reportFindings,
    vitals,
  };

  try {
    const res = await fetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const data = await res.json();
    if (!res.ok) { alert(data.error || "Analysis failed."); showEmptyState(); return; }
    renderResults(data);
  } catch (err) {
    console.error(err);
    alert("Network error. Is the Flask server running?");
    showEmptyState();
  }
}

function showEmptyState() {
  loadingState.classList.add("hidden"); emptyState.classList.remove("hidden");
}

// ── Render results ─────────────────────────────────────────────────────────
function renderResults(data) {
  loadingState.classList.add("hidden");
  resultsContent.classList.remove("hidden");

  // Patient summary bar
  renderPatientSummary(data.patient_profile);

  // Emergency
  if (data.emergency && data.emergency_messages.length) {
    emergencyAlert.classList.remove("hidden");
    $("emergency-messages").innerHTML =
      data.emergency_messages.map(m => `<p>• ${m}</p>`).join("");
  } else {
    emergencyAlert.classList.add("hidden");
  }

  // Risk score
  animateNumber(riskNumber, 0, data.risk_score, 900);
  riskLabelBadge.textContent = data.risk_label;
  riskLabelBadge.style.background = riskColor(data.risk_label);
  riskLabelBadge.style.color = "#fff";
  confidenceLbl.textContent = data.confidence_label;
  confidenceLbl.style.color = data.confidence_color;
  drawGauge(data.risk_score, data.risk_label);

  // Vitals results
  renderVitalsResults(data.vitals_summary);

  // Preexisting impact
  renderPreexistingImpact(data.preexisting_impact);

  // Family history impact
  renderFamilyHistoryImpact(data.family_history_impact);

  // Report findings impact
  renderReportImpact(data.report_impact);

  // Predictions
  renderPredictions(data.predictions);

  // Explain chart
  renderExplainChart(data.contributions);
}

// ── Patient summary bar ────────────────────────────────────────────────────
function renderPatientSummary(profile) {
  const bar = $("patient-summary-bar");
  const pills = [];
  if (profile.age) {
    const grpLabel = { "child": "Child (≤17)", "adult": "Adult (18–59)", "senior": "Senior (60+)" }[profile.age_group] || "";
    pills.push(`<span class="profile-pill">👤 Age ${profile.age} · ${grpLabel}</span>`);
  }
  if (profile.gender && profile.gender !== "prefer_not")
    pills.push(`<span class="profile-pill">${profile.gender}</span>`);
  profile.preexisting.forEach(p =>
    pills.push(`<span class="profile-pill">${p.icon} ${p.label}</span>`)
  );
  (profile.family_history || []).forEach(f =>
    pills.push(`<span class="profile-pill" style="background:rgba(108,92,231,.12);color:#6C5CE7;border:1px solid rgba(108,92,231,.25)">🧬 Fam: ${f.label}</span>`)
  );
  bar.innerHTML = pills.length
    ? `<strong style="color:var(--teal);margin-right:.4rem">Profile:</strong>${pills.join("")}`
    : "";
}

// ── Vitals results ─────────────────────────────────────────────────────────
function renderVitalsResults(summary) {
  const panel = $("vitals-panel");
  const cards = $("vitals-cards");
  if (!summary || Object.keys(summary).length === 0) {
    panel.classList.add("hidden"); return;
  }
  panel.classList.remove("hidden");
  cards.innerHTML = "";
  for (const [key, info] of Object.entries(summary)) {
    const card = document.createElement("div");
    card.className = "vital-result-card";
    card.style.borderTopColor = info.color;
    card.innerHTML = `
      <div class="vrc-label">${info.label}</div>
      <div class="vrc-value" style="color:${info.color}">${info.value}</div>
      <div class="vrc-unit">${info.unit}</div>
      <div class="vrc-status" style="color:${info.color}">${info.label_status || info.label}</div>`;
    // Fix label_status
    card.querySelector(".vrc-status").textContent = info.label;
    cards.appendChild(card);
  }
}

// ── Preexisting impact ─────────────────────────────────────────────────────
function renderPreexistingImpact(notes) {
  const panel = $("preexisting-panel");
  const list = $("preexisting-impact-list");
  if (!notes || notes.length === 0) { panel.classList.add("hidden"); return; }
  panel.classList.remove("hidden");
  list.innerHTML = notes.map(n =>
    `<div class="impact-note">⚠️ ${n}</div>`
  ).join("");
}

// ── Family history impact ──────────────────────────────────────────────────
function renderFamilyHistoryImpact(notes) {
  const panel = $("family-history-panel");
  const list = $("family-history-impact-list");
  if (!panel || !notes || notes.length === 0) {
    if (panel) panel.classList.add("hidden");
    return;
  }
  panel.classList.remove("hidden");
  list.innerHTML = notes.map(n =>
    `<div class="impact-note fh-impact-note">🧬 ${n}</div>`
  ).join("");
}

// ── Predictions ────────────────────────────────────────────────────────────
function renderPredictions(predictions) {
  const list = $("predictions-list");
  list.innerHTML = "";
  predictions.forEach((p, i) => {
    const card = document.createElement("div");
    card.className = `prediction-card ${i === 0 ? "top-card" : ""}`;
    const badges = [];
    if (p.age_modifier && Math.abs(p.age_modifier - 1) > 0.05)
      badges.push(`<span class="mod-badge">${p.age_modifier_label}</span>`);
    if (p.preexisting_boost && p.preexisting_boost > 1.05)
      badges.push(`<span class="mod-badge boost">⚕️ Boosted by preexisting conditions (×${p.preexisting_boost})</span>`);
    if (p.family_history_boost && p.family_history_boost > 1.05)
      badges.push(`<span class="mod-badge fh-boost">🧬 Elevated by family history (×${p.family_history_boost})</span>`);
    if (p.report_boost && p.report_boost > 1.05)
      badges.push(`<span class="mod-badge report-boost">📋 Confirmed by report findings (×${p.report_boost})</span>`);

    card.innerHTML = `
      <div class="pred-accent-bar" style="background:${p.risk_color}"></div>
      <div class="prediction-top-row">
        <span class="prediction-name">
          <span class="prediction-icon">${p.icon}</span>
          ${p.condition}
          ${i === 0 ? '<span class="top-badge">Top Match</span>' : ""}
        </span>
        <span class="prediction-prob">${p.probability}%</span>
      </div>
      <div class="prob-bar-bg">
        <div class="prob-bar-fill" style="width:0%;background:${p.risk_color}" data-w="${p.probability}"></div>
      </div>
      <span class="risk-pill" style="background:${p.risk_color}22;color:${p.risk_color};border:1px solid ${p.risk_color}44">
        ${p.risk_level.charAt(0).toUpperCase() + p.risk_level.slice(1)} Risk
      </span>
      <p class="prediction-desc">${p.description}</p>
      <p class="prediction-rec"><strong>Recommendation:</strong> ${p.recommendation}</p>
      ${badges.length ? `<div class="modifier-badges">${badges.join("")}</div>` : ""}
    `;
    list.appendChild(card);
    requestAnimationFrame(() => setTimeout(() => {
      const bar = card.querySelector(".prob-bar-fill");
      if (bar) bar.style.width = bar.dataset.w + "%";
    }, 100));
  });
}

// ── Gauge chart ────────────────────────────────────────────────────────────
function drawGauge(score, label) {
  const ctx = $("gauge-chart").getContext("2d");
  if (state.gaugeChart) state.gaugeChart.destroy();
  const color = riskColor(label);
  state.gaugeChart = new Chart(ctx, {
    type: "doughnut",
    data: {
      datasets: [{
        data: [score, 100 - score],
        backgroundColor: [color, "rgba(255,255,255,.12)"],
        borderWidth: 0, circumference: 270, rotation: -135
      }]
    },
    options: {
      responsive: false, cutout: "72%",
      plugins: { legend: { display: false }, tooltip: { enabled: false } },
      animation: { duration: 1000, easing: "easeOutQuart" }
    },
  });
  $("gauge-label").textContent = label;
}

// ── Explain chart ──────────────────────────────────────────────────────────
function renderExplainChart(contributions) {
  const ctx = $("explain-chart").getContext("2d");
  if (state.explainChart) state.explainChart.destroy();
  const labels = contributions.map(c => c.symptom);
  const values = contributions.map(c => c.contribution);
  const max = Math.max(...values) || 1;
  const colors = values.map(v => `rgba(2,128,144,${0.3 + (v / max) * 0.7})`);
  state.explainChart = new Chart(ctx, {
    type: "bar",
    data: {
      labels, datasets: [{
        label: "Influence (%)", data: values,
        backgroundColor: colors, borderRadius: 6, borderWidth: 0,
      }]
    },
    options: {
      indexAxis: "y", responsive: true,
      plugins: {
        legend: { display: false },
        tooltip: { callbacks: { label: c => ` Influence: ${c.parsed.x.toFixed(2)}%` } }
      },
      scales: {
        x: { beginAtZero: true, grid: { color: "rgba(0,0,0,.05)" }, ticks: { font: { size: 11 } } },
        y: { grid: { display: false }, ticks: { font: { size: 12 } } },
      },
      animation: { duration: 800, easing: "easeOutQuart" },
    },
  });
}

// ── Report impact panel ────────────────────────────────────────────────────
function renderReportImpact(notes) {
  const panel = $("report-impact-panel");
  const list = $("report-impact-list");
  if (!panel || !notes || notes.length === 0) {
    if (panel) panel.classList.add("hidden");
    return;
  }
  panel.classList.remove("hidden");
  list.innerHTML = notes.map(n =>
    `<div class="impact-note report-impact-note">📋 ${n}</div>`
  ).join("");
}

// ── Helpers ────────────────────────────────────────────────────────────────
function riskColor(label) {
  return { Critical: "#E63946", High: "#E67E22", Medium: "#F7B731", Low: "#02C39A" }[label] || "#64748B";
}
function animateNumber(el, from, to, duration) {
  const start = performance.now();
  (function step(now) {
    const t = Math.min((now - start) / duration, 1);
    el.textContent = Math.round(from + (to - from) * easeOut(t));
    if (t < 1) requestAnimationFrame(step);
  })(performance.now());
}
function easeOut(t) { return 1 - Math.pow(1 - t, 4); }

// ══════════════════════════════════════════════════════════════════════════
//  REPORT MODULE — Upload PDF tab + Manual Entry tab
// ══════════════════════════════════════════════════════════════════════════

const REPORT_TYPE_LABELS = {
  blood_test: "🩸 Blood Test",
  radiology: "🫁 Radiology / Imaging",
  urine_test: "🧪 Urine Test",
  other: "📋 Medical Report",
};

// ── Tab switcher ──────────────────────────────────────────────────────────
document.querySelectorAll(".report-tab").forEach(btn => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".report-tab").forEach(b => b.classList.remove("active"));
    btn.classList.add("active");
    const tab = btn.dataset.tab;
    $("tab-content-upload").classList.toggle("hidden", tab !== "upload");
    $("tab-content-manual").classList.toggle("hidden", tab !== "manual");
    // Clear findings whenever tab changes
    resetFindings();
  });
});

// ── PDF Upload ────────────────────────────────────────────────────────────
const uploadZone = $("upload-zone");
const fileInput = $("report-file-input");

uploadZone.addEventListener("click", () => fileInput.click());
uploadZone.addEventListener("dragover", e => { e.preventDefault(); uploadZone.classList.add("drag-over"); });
uploadZone.addEventListener("dragleave", () => uploadZone.classList.remove("drag-over"));
uploadZone.addEventListener("drop", e => {
  e.preventDefault();
  uploadZone.classList.remove("drag-over");
  if (e.dataTransfer.files[0]) handleFileChosen(e.dataTransfer.files[0]);
});
fileInput.addEventListener("change", () => {
  if (fileInput.files[0]) handleFileChosen(fileInput.files[0]);
});

function handleFileChosen(file) {
  if (file.type !== "application/pdf") {
    alert("Only PDF files are supported for auto-extraction.\n\nFor image/photo reports, please use the ✏️ Enter Values tab to input your lab values manually.");
    return;
  }
  if (file.size > 10 * 1024 * 1024) {
    alert("File is too large. Maximum size is 10 MB.");
    return;
  }
  $("upload-zone").classList.add("hidden");
  $("file-chosen").classList.remove("hidden");
  $("file-name").textContent = file.name;
  $("extract-btn")._file = file;
}

$("extract-btn").addEventListener("click", async function () {
  const file = this._file;
  if (!file) return;

  $("file-chosen").classList.add("hidden");
  $("extract-loading").classList.remove("hidden");
  resetFindings();

  const formData = new FormData();
  formData.append("report", file);

  try {
    const res = await fetch("/api/analyze-report", { method: "POST", body: formData });
    const data = await res.json();

    if (!res.ok) {
      alert(data.error || "Could not extract findings from this PDF. Try entering values manually in the ✏️ Enter Values tab.");
      $("upload-zone").classList.remove("hidden");
      $("file-chosen").classList.add("hidden");
      return;
    }

    // If zero findings extracted, suggest manual tab
    if (!data.findings || data.findings.length === 0) {
      alert("No recognisable lab values were found in this PDF.\n\nPlease use the ✏️ Enter Values tab to input your values manually.");
      $("upload-zone").classList.remove("hidden");
      return;
    }

    state.reportFindings = data.findings;
    renderExtractedFindings(data);

  } catch (err) {
    console.error(err);
    alert("Network error while reading PDF.");
    $("upload-zone").classList.remove("hidden");
  } finally {
    $("extract-loading").classList.add("hidden");
  }
});

$("remove-file-btn").addEventListener("click", () => {
  fileInput.value = "";
  $("file-chosen").classList.add("hidden");
  $("upload-zone").classList.remove("hidden");
  resetFindings();
});

// ── Manual entry ──────────────────────────────────────────────────────────
$("apply-manual-btn").addEventListener("click", async () => {
  const MANUAL_IDS = ["hb", "wbc", "plt", "fbs", "hba1c", "chol", "ldl", "trig", "creat", "sgpt", "tsh", "ua"];
  const values = {};
  MANUAL_IDS.forEach(id => {
    const el = $(`m-${id}`);
    if (el && el.value !== "") values[id] = parseFloat(el.value);
  });

  if (Object.keys(values).length === 0) {
    alert("Please enter at least one lab value before applying.");
    return;
  }

  try {
    const res = await fetch("/api/process-manual", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ values }),
    });
    const data = await res.json();

    if (!res.ok) {
      alert(data.error || "Could not process the values.");
      return;
    }

    state.reportFindings = data.findings;
    renderExtractedFindings(data);

  } catch (err) {
    console.error(err);
    alert("Network error. Is the Flask server running?");
  }
});

// ── Render findings (shared by both tabs) ─────────────────────────────────
function renderExtractedFindings(data) {
  const findings = data.findings || [];
  $("findings-section").classList.remove("hidden");

  const abnormal = findings.filter(f => f.flag !== "normal_finding" && f.flag !== "other").length;
  const method = { ai: "🤖 AI", rule_based: "🔍 Auto", manual: "✏️ Manual" }[data.method] || "";

  $("findings-count-label").textContent =
    `${method}  ·  ${findings.length} finding${findings.length !== 1 ? "s" : ""}` +
    (abnormal ? `  ·  ⚠️ ${abnormal} abnormal` : "  ·  ✅ All normal");

  const rtBadge = $("report-type-badge");
  rtBadge.textContent = REPORT_TYPE_LABELS[data.report_type] || "📋 Medical Report";
  rtBadge.className = "report-type-badge";

  const chipsEl = $("findings-chips");
  chipsEl.innerHTML = "";

  // Only show abnormal ones prominently; normal ones dimmed
  const sorted = [...findings].sort((a, b) => {
    const isAbnA = a.flag !== "normal_finding" && a.flag !== "other";
    const isAbnB = b.flag !== "normal_finding" && b.flag !== "other";
    return isAbnB - isAbnA;
  });

  sorted.forEach(f => {
    const isAbnormal = f.flag !== "normal_finding" && f.flag !== "other";
    const chip = document.createElement("div");
    chip.className = `finding-chip ${isAbnormal ? "" : "finding-chip-normal"}`;
    chip.style.borderColor = f.color + (isAbnormal ? "70" : "30");
    chip.style.background = f.color + (isAbnormal ? "14" : "06");

    const statusIcon = { normal: "✅", low: "⬇️", high: "⬆️", abnormal: "⚠️" }[f.status] || "📋";
    chip.innerHTML = `
      <span class="finding-icon">${f.icon}</span>
      <div class="finding-info">
        <span class="finding-name">${f.parameter}</span>
        <span class="finding-val" style="color:${f.color}">${statusIcon} ${f.value} ${f.unit || ""}</span>
      </div>
      <span class="finding-flag" style="color:${f.color}">${f.flag_label}</span>`;
    chipsEl.appendChild(chip);
  });

  const notesEl = $("clinical-notes-list");
  notesEl.innerHTML = (data.clinical_notes || []).map(n =>
    `<div class="clinical-note">💬 ${n}</div>`
  ).join("");
}

// ── Clear findings ─────────────────────────────────────────────────────────
$("clear-findings-btn").addEventListener("click", () => {
  resetFindings();
  // Also clear manual inputs
  ["hb", "wbc", "plt", "fbs", "hba1c", "chol", "ldl", "trig", "creat", "sgpt", "tsh", "ua"]
    .forEach(id => { const el = $(`m-${id}`); if (el) el.value = ""; });
});

function resetFindings() {
  state.reportFindings = [];
  $("findings-section").classList.add("hidden");
  $("findings-chips").innerHTML = "";
  $("clinical-notes-list").innerHTML = "";
}

// ── Start ──────────────────────────────────────────────────────────────────
init();
