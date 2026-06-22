/* main.js — Loan Approval Predictor */

async function predict() {
  const btn   = document.getElementById("btnPredict");
  const label = document.getElementById("btnLabel");
  const errEl = document.getElementById("formError");

  // ── Gather inputs 
  const payload = {
    gender:             get("gender"),
    married:            get("married"),
    dependents:         get("dependents"),
    education:          get("education"),
    self_employed:      get("self_employed"),
    applicant_income:   parseFloat(get("applicant_income"))   || 0,
    coapplicant_income: parseFloat(get("coapplicant_income")) || 0,
    loan_amount:        parseFloat(get("loan_amount"))         || 0,
    loan_term:          parseFloat(get("loan_term"))           || 360,
    credit_history:     parseFloat(get("credit_history"))      || 0,
    property_area:      get("property_area"),
  };

  // ── Basic validation 
  errEl.classList.add("hidden");
  if (payload.applicant_income <= 0) {
    showError("Applicant income must be greater than 0."); return;
  }
  if (payload.loan_amount <= 0) {
    showError("Loan amount must be greater than 0."); return;
  }

  // ── Loading state ──────────────────────────────────────────────
  btn.disabled = true;
  label.textContent = "Predicting…";

  try {
    const res  = await fetch("/predict", {
      method:  "POST",
      headers: { "Content-Type": "application/json" },
      body:    JSON.stringify(payload),
    });

    if (!res.ok) throw new Error("Server error: " + res.status);
    const data = await res.json();
    showResult(data);
  } catch (err) {
    showError("Something went wrong. Is the Flask server running?");
    console.error(err);
  } finally {
    btn.disabled = false;
    label.textContent = "Predict Approval";
  }
}

function showResult(data) {
  const card       = document.getElementById("resultCard");
  const placeholder= document.getElementById("resultPlaceholder");
  const content    = document.getElementById("resultContent");
  const badge      = document.getElementById("verdictBadge");
  const msg        = document.getElementById("resultMessage");
  const probVal    = document.getElementById("probValue");
  const probFill   = document.getElementById("probFill");
  const confVal    = document.getElementById("confValue");

  placeholder.classList.add("hidden");
  content.classList.remove("hidden");

  const cls = data.approved ? "approved" : "rejected";

  badge.textContent = data.verdict;
  badge.className   = "verdict-badge " + cls;
  msg.textContent   = data.message;
  probVal.textContent = data.probability + "%";
  confVal.textContent = data.confidence  + "%";

  probFill.style.width = "0%";
  probFill.className   = "prob-fill " + cls;
  setTimeout(() => { probFill.style.width = data.probability + "%"; }, 50);

  card.scrollIntoView({ behavior: "smooth", block: "center" });
}

function resetForm() {
  document.getElementById("resultPlaceholder").classList.remove("hidden");
  document.getElementById("resultContent").classList.add("hidden");
}

function showError(msg) {
  const el = document.getElementById("formError");
  el.textContent = "⚠️  " + msg;
  el.classList.remove("hidden");
}

function get(id) {
  return document.getElementById(id).value;
}