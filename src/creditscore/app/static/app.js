// Real-time credit scoring UI logic.

const form = document.getElementById("score-form");
const resultBox = document.getElementById("result");
const decisionEl = document.getElementById("decision");
const gaugeFill = document.getElementById("gauge-fill");
const probEl = document.getElementById("prob");
const bandEl = document.getElementById("band");
const thresholdEl = document.getElementById("threshold");
const badge = document.getElementById("model-badge");

// A representative low-risk applicant, for the "Fill sample" button.
const SAMPLE = {
  age: 41, monthly_income: 4200, employment_length: 9, loan_amount: 12000,
  loan_term_months: 36, debt_to_income: 0.22, credit_history_length: 18,
  num_past_defaults: 0, credit_utilization: 0.24, num_open_accounts: 5,
  num_recent_inquiries: 1, home_ownership: "own", purpose: "car",
};

async function loadMetadata() {
  try {
    const r = await fetch("/api/metadata");
    const m = await r.json();
    if (m.champion) {
      const auc = m.test_metrics ? ` · AUC ${m.test_metrics.roc_auc.toFixed(3)}` : "";
      badge.textContent = `champion: ${m.champion}${auc} · threshold ${m.threshold?.toFixed(3)}`;
    } else {
      badge.textContent = "no trained model — run the pipeline";
    }
  } catch { badge.textContent = "metadata unavailable"; }
}

document.getElementById("sample-btn").addEventListener("click", () => {
  for (const [k, v] of Object.entries(SAMPLE)) {
    const el = form.elements[k];
    if (el) el.value = v;
  }
});

form.addEventListener("submit", async (e) => {
  e.preventDefault();
  const data = {};
  for (const el of form.elements) {
    if (el.name && el.value !== "") data[el.name] = el.value;
  }
  const btn = document.getElementById("score-btn");
  btn.disabled = true; btn.textContent = "Scoring…";
  try {
    const r = await fetch("/api/score", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(data),
    });
    const res = await r.json();
    if (!r.ok) throw new Error(res.error || "Scoring failed");
    render(res);
  } catch (err) {
    alert(err.message);
  } finally {
    btn.disabled = false; btn.textContent = "Score applicant";
  }
});

function render(res) {
  resultBox.classList.remove("hidden");
  const approve = res.decision === "APPROVE";
  decisionEl.textContent = res.decision;
  decisionEl.className = "decision " + (approve ? "approve" : "decline");
  const pct = (res.probability_default * 100).toFixed(1);
  gaugeFill.style.width = pct + "%";
  probEl.textContent = `Probability of default: ${pct}%`;
  bandEl.textContent = `Risk band: ${res.risk_band.replace("_", " ")}`;
  thresholdEl.textContent = `Decision threshold: ${res.threshold} (approve if P(default) < threshold)`;
}

loadMetadata();
