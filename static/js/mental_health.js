/* ═══════════════════════════════════════════════════════════════════════════
   HealthMind AI – Mental Health Module JS
   Completely separate from main.js — does not modify any existing state
   ═══════════════════════════════════════════════════════════════════════════ */

"use strict";

// ── State ─────────────────────────────────────────────────────────────────────
const mhState = {
  answers: {},   // { mh_01: 2, mh_02: 0, … }
  totalQ: 20,
  gaugeChart: null,
  radarChart: null,
};

const ANSWER_LABELS = ["Not at all", "Several days", "More than half the days", "Nearly every day"];

const DOMAIN_COLORS = {
  Mood: "#028090",
  Anxiety: "#E67E22",
  Sleep: "#6C5CE7",
  Stress: "#E63946",
  Social: "#02C39A",
  Physical: "#F7B731",
};

// ── Answer selection via event delegation (no inline onclick) ─────────────────
document.addEventListener("click", function (e) {
  const btn = e.target.closest(".ans-btn");
  if (!btn) return;
  const qid = btn.dataset.qid;
  const val = parseInt(btn.dataset.val, 10);
  selectAnswer(qid, val, btn);
});

function selectAnswer(qid, val, btn) {
  mhState.answers[qid] = val;

  // Clear all buttons for this question, highlight selected
  const group = btn.closest(".answer-buttons");
  group.querySelectorAll(".ans-btn").forEach(b => {
    b.className = "ans-btn";
  });
  btn.classList.add(`selected-${val}`);

  // Update label
  document.getElementById(`lbl-${qid}`).textContent = ANSWER_LABELS[val];

  // Update progress
  updateProgress();
}

function updateProgress() {
  const answered = Object.keys(mhState.answers).length;
  const total = mhState.totalQ;
  const pct = Math.round((answered / total) * 100);

  document.getElementById("progress-text").textContent = `${answered} of ${total} answered`;
  document.getElementById("progress-pct").textContent = `${pct}%`;
  document.getElementById("progress-fill").style.width = `${pct}%`;

  document.getElementById("mh-analyse-btn").disabled = answered < total;
}

// ── Analyse ────────────────────────────────────────────────────────────────────
document.getElementById("mh-analyse-btn").addEventListener("click", async () => {
  // Show loading
  document.getElementById("mh-empty-state").classList.add("hidden");
  document.getElementById("mh-results-content").classList.add("hidden");
  document.getElementById("mh-loading").classList.remove("hidden");
  document.getElementById("mh-results-panel").scrollIntoView({ behavior: "smooth", block: "start" });

  const payload = {
    answers: mhState.answers,
    age: document.getElementById("mh-age").value || null,
    gender: document.getElementById("mh-gender").value || null,
  };

  try {
    const res = await fetch("/predict-mental", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const data = await res.json();

    if (!res.ok) {
      alert(data.error || "Analysis failed. Please try again.");
      document.getElementById("mh-loading").classList.add("hidden");
      document.getElementById("mh-empty-state").classList.remove("hidden");
      return;
    }

    renderMHResults(data);
  } catch (err) {
    console.error(err);
    alert("Network error — is the Flask server running?");
    document.getElementById("mh-loading").classList.add("hidden");
    document.getElementById("mh-empty-state").classList.remove("hidden");
  }
});

// ── Render results ─────────────────────────────────────────────────────────────
function renderMHResults(data) {
  document.getElementById("mh-loading").classList.add("hidden");
  document.getElementById("mh-results-content").classList.remove("hidden");

  // Crisis alert
  const crisisEl = document.getElementById("mh-crisis-alert");
  if (data.crisis_flag) {
    crisisEl.classList.remove("hidden");
  } else {
    crisisEl.classList.add("hidden");
  }

  // Score number (animated)
  animateNumber(document.getElementById("mh-score-num"), 0, data.score, 1000);

  // Score colour
  const scoreNumEl = document.getElementById("mh-score-num");
  setTimeout(() => { scoreNumEl.style.color = data.color; }, 100);

  // Level badge
  const badge = document.getElementById("mh-level-badge");
  badge.textContent = `${data.emoji} ${data.level}`;
  badge.style.background = data.color + "33";
  badge.style.color = data.color;
  badge.style.border = `1px solid ${data.color}66`;

  // Gauge
  drawMHGauge(data.score, data.color, data.level);

  // Score fill bar
  setTimeout(() => {
    document.getElementById("mh-score-fill").style.width = `${data.score}%`;
  }, 200);

  // Summary
  const summaryBox = document.getElementById("mh-summary-box");
  summaryBox.style.borderLeftColor = data.color;
  summaryBox.style.background = data.color + "12";
  document.getElementById("mh-summary-text").textContent = data.summary;

  // Domain breakdown
  renderDomainBars(data.domain_scores);
  drawRadarChart(data.domain_scores);

  // Recommendations
  renderRecommendations(data.recommendations);
}

// ── Gauge ─────────────────────────────────────────────────────────────────────
function drawMHGauge(score, color, label) {
  const ctx = document.getElementById("mh-gauge").getContext("2d");
  if (mhState.gaugeChart) mhState.gaugeChart.destroy();

  mhState.gaugeChart = new Chart(ctx, {
    type: "doughnut",
    data: {
      datasets: [{
        data: [score, 100 - score],
        backgroundColor: [color, "rgba(255,255,255,.12)"],
        borderWidth: 0,
        circumference: 270,
        rotation: -135,
      }],
    },
    options: {
      responsive: false,
      cutout: "72%",
      plugins: { legend: { display: false }, tooltip: { enabled: false } },
      animation: { duration: 1100, easing: "easeOutQuart" },
    },
  });

  document.getElementById("mh-gauge-lbl").textContent = label;
}

// ── Radar chart ───────────────────────────────────────────────────────────────
function drawRadarChart(domainScores) {
  const ctx = document.getElementById("mh-radar-chart").getContext("2d");
  if (mhState.radarChart) mhState.radarChart.destroy();

  // Exclude Crisis domain from radar
  const entries = Object.entries(domainScores).filter(([d]) => d !== "Crisis");
  const labels = entries.map(([d]) => d);
  const values = entries.map(([, v]) => v);

  mhState.radarChart = new Chart(ctx, {
    type: "radar",
    data: {
      labels,
      datasets: [{
        label: "Your Score",
        data: values,
        backgroundColor: "rgba(2,128,144,.15)",
        borderColor: "#028090",
        borderWidth: 2,
        pointBackgroundColor: "#028090",
        pointRadius: 5,
      }],
    },
    options: {
      responsive: true,
      scales: {
        r: {
          min: 0, max: 100,
          ticks: { stepSize: 25, font: { size: 10 }, color: "#64748B" },
          grid: { color: "rgba(0,0,0,.08)" },
          angleLines: { color: "rgba(0,0,0,.08)" },
          pointLabels: { font: { size: 12, weight: "600" }, color: "#0A2540" },
        },
      },
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: { label: c => ` ${c.label}: ${c.raw}%` },
        },
      },
      animation: { duration: 900, easing: "easeOutQuart" },
    },
  });
}

// ── Domain progress bars ──────────────────────────────────────────────────────
function renderDomainBars(domainScores) {
  const container = document.getElementById("domain-bars-list");
  container.innerHTML = "";

  const entries = Object.entries(domainScores)
    .filter(([d]) => d !== "Crisis")
    .sort(([, a], [, b]) => b - a);

  entries.forEach(([domain, score]) => {
    const color = DOMAIN_COLORS[domain] || "#028090";
    const row = document.createElement("div");
    row.className = "domain-bar-row";
    row.innerHTML = `
      <div class="domain-bar-header">
        <span class="domain-bar-name">${domain}</span>
        <span class="domain-bar-score" style="color:${color}">${score}%</span>
      </div>
      <div class="domain-bar-track">
        <div class="domain-bar-fill" style="width:0%;background:${color}" data-w="${score}"></div>
      </div>`;
    container.appendChild(row);

    // Animate bar
    requestAnimationFrame(() => setTimeout(() => {
      row.querySelector(".domain-bar-fill").style.width = score + "%";
    }, 150));
  });
}

// ── Recommendations ───────────────────────────────────────────────────────────
function renderRecommendations(recs) {
  const list = document.getElementById("mh-recs-list");
  list.innerHTML = "";
  recs.forEach((rec, i) => {
    const item = document.createElement("div");
    item.className = "rec-item";
    item.innerHTML = `<span class="rec-num">${i + 1}</span><span>${rec}</span>`;
    list.appendChild(item);
  });
}

// ── Helpers ───────────────────────────────────────────────────────────────────
function animateNumber(el, from, to, duration) {
  const start = performance.now();
  (function step(now) {
    const t = Math.min((now - start) / duration, 1);
    el.textContent = Math.round(from + (to - from) * easeOut(t));
    if (t < 1) requestAnimationFrame(step);
  })(performance.now());
}

function easeOut(t) { return 1 - Math.pow(1 - t, 4); }
