// Heart Disease Prediction UI – minimal vanilla JS (no build step)

const PRESETS = {
  high: { age: 67, sex: 1, cp: 3, trestbps: 160, chol: 286, fbs: 0, restecg: 2,
          thalach: 108, exang: 1, oldpeak: 1.5, slope: 2, ca: 3, thal: 3 },
  low:  { age: 41, sex: 0, cp: 1, trestbps: 130, chol: 204, fbs: 0, restecg: 2,
          thalach: 172, exang: 0, oldpeak: 1.4, slope: 1, ca: 0, thal: 2 },
  reset:{ age: 63, sex: 1, cp: 3, trestbps: 145, chol: 233, fbs: 1, restecg: 0,
          thalach: 150, exang: 0, oldpeak: 2.3, slope: 0, ca: 0, thal: 1 },
};

const NUMERIC_FIELDS = ["age", "trestbps", "chol", "thalach", "oldpeak"];

const $ = (sel) => document.querySelector(sel);
const form = $("#predict-form");
const submitBtn = $("#submit-btn");
const errEl = $("#form-error");
const resultEl = $("#result");

function applyPreset(name) {
  const p = PRESETS[name];
  if (!p) return;
  for (const [k, v] of Object.entries(p)) {
    const el = form.elements[k];
    if (el) el.value = String(v);
  }
}

document.querySelectorAll("[data-preset]").forEach((btn) => {
  btn.addEventListener("click", () => applyPreset(btn.dataset.preset));
});

function readPayload() {
  const data = {};
  for (const el of form.elements) {
    if (!el.name) continue;
    const raw = el.value;
    data[el.name] = NUMERIC_FIELDS.includes(el.name) ? parseFloat(raw) : parseInt(raw, 10);
  }
  return data;
}

function setLoading(on) {
  submitBtn.disabled = on;
  submitBtn.querySelector(".spinner").hidden = !on;
  submitBtn.querySelector(".btn-label").textContent = on ? "Predicting…" : "Predict risk";
}

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;",
    ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

function renderResult(body) {
  const cls = body.label === "disease" ? "disease" : "no_disease";
  const verdictText = body.label === "disease" ? "Likely heart disease" : "No heart disease likely";
  const confPct = (body.confidence * 100).toFixed(1);
  const bars = Object.entries(body.probabilities)
    .sort(([, a], [, b]) => b - a)
    .map(([cls2, p]) => {
      const pct = (p * 100).toFixed(1);
      return `<div class="bar ${cls2}">
        <span>${cls2.replace("_", " ")}</span>
        <div class="track"><div class="fill" style="width:${pct}%"></div></div>
        <span class="pct">${pct}%</span>
      </div>`;
    }).join("");

  resultEl.classList.remove("empty");
  resultEl.innerHTML = `
    <div class="verdict ${cls}">
      <div class="badge">${cls === "disease" ? "!" : "✓"}</div>
      <div>
        <h3>${escapeHtml(verdictText)}</h3>
        <p>Model confidence: <strong>${confPct}%</strong></p>
      </div>
    </div>
    <div class="bars">${bars}</div>
  `;
}

function renderError(message) {
  resultEl.classList.remove("empty");
  resultEl.innerHTML = `<div class="verdict disease">
    <div class="badge">×</div>
    <div><h3>Prediction failed</h3><p>${escapeHtml(message)}</p></div>
  </div>`;
}

form.addEventListener("submit", async (e) => {
  e.preventDefault();
  errEl.textContent = "";
  setLoading(true);
  try {
    const payload = readPayload();
    const res = await fetch("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const body = await res.json().catch(() => ({}));
    if (!res.ok) {
      const detail = Array.isArray(body.detail)
        ? body.detail.map((d) => `${d.loc?.slice(-1)[0] || "field"}: ${d.msg}`).join("; ")
        : body.detail || res.statusText;
      throw new Error(detail);
    }
    renderResult(body);
  } catch (err) {
    renderError(err.message || String(err));
  } finally {
    setLoading(false);
  }
});

async function pollHealth() {
  const pill = $("#health-pill");
  try {
    const res = await fetch("/health");
    const body = await res.json();
    if (res.ok && body.model_loaded) {
      pill.textContent = `model ready · v${body.api_version}`;
      pill.className = "pill pill-ok";
    } else {
      pill.textContent = "model unavailable";
      pill.className = "pill pill-down";
    }
  } catch {
    pill.textContent = "API unreachable";
    pill.className = "pill pill-down";
  }
}
pollHealth();
