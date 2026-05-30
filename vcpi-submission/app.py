#!/usr/bin/env python
"""FastAPI demo for the VCPI virtual-cell predictor.

Loads the Ridge model persisted by ``train_pipeline.py`` and serves
per-gene expression predictions for an arbitrary SMILES.

Run
---
    .venv/bin/uvicorn app:app --reload --port 8000

Then:
    curl -s localhost:8000/predict -H 'content-type: application/json' \
         -d '{"smiles": "CCOc1ccccc1", "top_k": 10}' | python -m json.tool

Endpoints
---------
GET  /health   -> model status + gene counts
POST /predict  -> {"smiles": "...", "top_k": 20, "full": false}
    Returns top changed genes (largest |pred - training mean|), nearest
    training compounds (Tanimoto), and optionally the full predicted
    expression vector over every scored gene.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

MODEL_PATH = Path(__file__).resolve().parent / "artifacts" / "model.joblib"

app = FastAPI(title="VCPI Virtual-Cell Predictor", version="0.1.0")

# Loaded once at startup (see _load).
_A: dict | None = None
_GEN = None


def _load() -> dict:
    global _A, _GEN  # noqa: PLW0603
    if _A is None:
        import joblib
        from rdkit.Chem import rdFingerprintGenerator

        if not MODEL_PATH.exists():
            msg = (
                f"No model at {MODEL_PATH}. Train first: "
                "`.venv/bin/python train_pipeline.py --demo`  (offline) or "
                "`... train_pipeline.py --job tvc-bhr-009`  (real, needs TVC_TOKEN)."
            )
            raise RuntimeError(msg)
        _A = joblib.load(MODEL_PATH)
        _GEN = rdFingerprintGenerator.GetMorganGenerator(
            radius=_A["radius"], fpSize=_A["n_bits"]
        )
    return _A


def _featurize(smiles: str) -> np.ndarray:
    from rdkit import Chem

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise HTTPException(status_code=400, detail=f"Unparseable SMILES: {smiles!r}")
    return _GEN.GetFingerprintAsNumPy(mol).astype(np.float32)


class PredictRequest(BaseModel):
    smiles: str = Field(..., examples=["CCOc1ccccc1"])
    top_k: int = Field(20, ge=1, le=200, description="how many top-changed genes to return")
    full: bool = Field(False, description="include the full per-gene expression vector")


@app.get("/", response_class=HTMLResponse)
def home() -> str:
    """Human-friendly local demo page."""
    return """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>VCPI Virtual Cell</title>
  <style>
    :root {
      --ink: #17202a;
      --muted: #657180;
      --line: #d8dee7;
      --surface: #ffffff;
      --soft: #f6f8fb;
      --accent: #0f766e;
      --accent-dark: #115e59;
      --warm: #9a3412;
      --blue: #1d4ed8;
      --good: #166534;
      --bad: #991b1b;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      color: var(--ink);
      background: #f4f7fa;
      line-height: 1.45;
    }
    header {
      background: #ffffff;
      border-bottom: 1px solid var(--line);
    }
    .wrap {
      max-width: 1120px;
      margin: 0 auto;
      padding: 24px;
    }
    .topbar {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 16px;
    }
    .brand {
      display: flex;
      align-items: center;
      gap: 12px;
      min-width: 0;
    }
    .mark {
      width: 42px;
      height: 42px;
      border-radius: 8px;
      display: grid;
      place-items: center;
      background: #e0f2f1;
      color: var(--accent-dark);
      font-weight: 800;
      letter-spacing: 0;
      flex: 0 0 auto;
    }
    h1 {
      margin: 0;
      font-size: 26px;
      line-height: 1.1;
      letter-spacing: 0;
    }
    .subtitle {
      margin: 6px 0 0;
      color: var(--muted);
      font-size: 14px;
    }
    .status {
      display: flex;
      gap: 8px;
      align-items: center;
      color: var(--good);
      font-size: 14px;
      white-space: nowrap;
    }
    .dot {
      width: 9px;
      height: 9px;
      border-radius: 50%;
      background: var(--good);
    }
    main .wrap {
      display: grid;
      grid-template-columns: minmax(0, 1.1fr) minmax(320px, 0.9fr);
      gap: 18px;
      align-items: start;
    }
    section, .panel {
      background: var(--surface);
      border: 1px solid var(--line);
      border-radius: 8px;
    }
    .panel {
      padding: 20px;
    }
    h2 {
      margin: 0 0 12px;
      font-size: 17px;
      letter-spacing: 0;
    }
    .explain p, .explain li {
      color: var(--muted);
      font-size: 14px;
    }
    .explain ul {
      margin: 10px 0 0;
      padding-left: 18px;
    }
    label {
      display: block;
      font-weight: 700;
      margin-bottom: 8px;
      font-size: 14px;
    }
    textarea {
      width: 100%;
      min-height: 96px;
      resize: vertical;
      border: 1px solid #bcc7d5;
      border-radius: 8px;
      padding: 12px;
      font: 15px/1.35 ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      color: var(--ink);
      background: #fbfcfe;
    }
    textarea:focus, select:focus {
      outline: 3px solid #bfe7e3;
      border-color: var(--accent);
    }
    .controls {
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      align-items: end;
      margin-top: 14px;
    }
    .field {
      min-width: 130px;
    }
    select {
      width: 100%;
      border: 1px solid #bcc7d5;
      border-radius: 8px;
      padding: 10px;
      background: #fff;
      color: var(--ink);
    }
    button {
      border: 0;
      border-radius: 8px;
      background: var(--accent);
      color: white;
      padding: 11px 16px;
      font-weight: 800;
      cursor: pointer;
      min-height: 42px;
    }
    button:hover { background: var(--accent-dark); }
    button.secondary {
      background: #e8edf3;
      color: var(--ink);
    }
    button.secondary:hover { background: #dbe3ec; }
    .examples {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin-top: 12px;
    }
    .chip {
      border: 1px solid var(--line);
      background: #fff;
      color: var(--ink);
      padding: 8px 10px;
      border-radius: 999px;
      font-size: 13px;
      cursor: pointer;
    }
    .chip:hover { border-color: var(--accent); color: var(--accent-dark); }
    .metrics {
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 10px;
      margin-bottom: 14px;
    }
    .metric {
      background: var(--soft);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 12px;
    }
    .metric strong {
      display: block;
      font-size: 20px;
      margin-bottom: 2px;
    }
    .metric span {
      color: var(--muted);
      font-size: 12px;
    }
    .result {
      min-height: 180px;
    }
    .placeholder {
      color: var(--muted);
      background: var(--soft);
      border: 1px dashed #b8c4d2;
      border-radius: 8px;
      padding: 18px;
      font-size: 14px;
    }
    .notice {
      background: #fff7ed;
      border: 1px solid #fed7aa;
      color: #7c2d12;
      border-radius: 8px;
      padding: 12px;
      font-size: 13px;
      margin-top: 12px;
    }
    table {
      width: 100%;
      border-collapse: collapse;
      font-size: 13px;
    }
    th, td {
      text-align: left;
      padding: 9px 8px;
      border-bottom: 1px solid var(--line);
      vertical-align: middle;
    }
    th {
      color: var(--muted);
      font-size: 12px;
      background: #f8fafc;
    }
    .up { color: var(--good); font-weight: 700; }
    .down { color: var(--bad); font-weight: 700; }
    .small {
      color: var(--muted);
      font-size: 12px;
    }
    .two-col {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 14px;
      margin-top: 14px;
    }
    .loading {
      color: var(--blue);
      font-weight: 700;
    }
    .error {
      color: var(--bad);
      font-weight: 700;
      background: #fef2f2;
      border: 1px solid #fecaca;
      border-radius: 8px;
      padding: 12px;
    }
    footer {
      color: var(--muted);
      font-size: 12px;
      padding: 0 24px 24px;
      max-width: 1120px;
      margin: 0 auto;
    }
    @media (max-width: 860px) {
      main .wrap, .two-col { grid-template-columns: 1fr; }
      .metrics { grid-template-columns: 1fr; }
      .topbar { align-items: flex-start; flex-direction: column; }
      h1 { font-size: 23px; }
    }
  </style>
</head>
<body>
  <header>
    <div class="wrap topbar">
      <div class="brand">
        <div class="mark">VC</div>
        <div>
          <h1>VCPI Virtual Cell</h1>
          <p class="subtitle">Predict THP-1 gene expression response from a compound SMILES.</p>
        </div>
      </div>
      <div class="status"><span class="dot"></span><span id="healthText">Checking model...</span></div>
    </div>
  </header>

  <main>
    <div class="wrap">
      <section class="panel">
        <h2>Predict A Cell Response</h2>
        <label for="smiles">Compound SMILES</label>
        <textarea id="smiles" spellcheck="false">CC(=O)Oc1ccccc1C(=O)O</textarea>
        <div class="controls">
          <div class="field">
            <label for="topK">Genes shown</label>
            <select id="topK">
              <option value="5">5</option>
              <option value="10" selected>10</option>
              <option value="20">20</option>
              <option value="50">50</option>
            </select>
          </div>
          <button id="predictBtn">Predict Response</button>
          <button class="secondary" id="clearBtn">Clear</button>
        </div>
        <div class="examples" aria-label="Example compounds">
          <button class="chip" data-smiles="CC(=O)Oc1ccccc1C(=O)O">Aspirin-like</button>
          <button class="chip" data-smiles="CC(C)Cc1ccc(cc1)[C@@H](C)C(=O)O">Ibuprofen-like</button>
          <button class="chip" data-smiles="CN1C=NC2=C1C(=O)N(C(=O)N2C)C">Caffeine</button>
          <button class="chip" data-smiles="CC(C)(C)NCC(O)c1ccc(O)c(CO)c1">Albuterol-like</button>
        </div>
        <div class="notice">
          This demo returns predicted expression and chemical neighbors. The contest submission file uses the stronger validated blend model in <code>artifacts/predictions.parquet</code>.
        </div>
      </section>

      <aside class="panel explain">
        <h2>What This Means</h2>
        <p>A THP-1 cell is treated with a compound for 24 hours. The model estimates the final activity level of each scored gene.</p>
        <ul>
          <li><strong>Predicted expression</strong> is on the <code>log2(CPM + 1)</code> RNA-seq scale.</li>
          <li><strong>Delta vs mean</strong> shows how far a gene is from the average training response.</li>
          <li><strong>Nearest compounds</strong> are chemically similar training examples by Morgan fingerprint similarity.</li>
        </ul>
      </aside>

      <section class="panel result" style="grid-column: 1 / -1;">
        <div class="metrics">
          <div class="metric"><strong id="geneCount">12,995</strong><span>modeled genes</span></div>
          <div class="metric"><strong id="trainCount">10,261</strong><span>training compounds</span></div>
          <div class="metric"><strong id="modelName">Ridge</strong><span>API demo model</span></div>
        </div>
        <div id="result">
          <div class="placeholder">
            Paste a SMILES string and run a prediction. Results will show the genes with the largest predicted change and the closest training compounds.
          </div>
        </div>
      </section>
    </div>
  </main>

  <footer>
    Local demo for VCPI hackathon. Use <code>/docs</code> for raw API testing and <code>/health</code> for model status.
  </footer>

  <script>
    const smilesEl = document.getElementById("smiles");
    const resultEl = document.getElementById("result");
    const topKEl = document.getElementById("topK");
    const btn = document.getElementById("predictBtn");

    function esc(value) {
      return String(value).replace(/[&<>"']/g, ch => ({
        "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#039;"
      }[ch]));
    }

    function direction(delta) {
      if (delta > 0) return '<span class="up">up</span>';
      if (delta < 0) return '<span class="down">down</span>';
      return '<span class="small">same</span>';
    }

    async function loadHealth() {
      try {
        const res = await fetch("/health");
        const data = await res.json();
        document.getElementById("healthText").textContent = "Model ready";
        document.getElementById("geneCount").textContent = Number(data.n_scored_genes).toLocaleString();
        document.getElementById("trainCount").textContent = Number(data.n_train_compounds).toLocaleString();
        document.getElementById("modelName").textContent = data.model.split("(")[0];
      } catch {
        document.getElementById("healthText").textContent = "Model unavailable";
      }
    }

    async function predict() {
      const smiles = smilesEl.value.trim();
      if (!smiles) {
        resultEl.innerHTML = '<div class="error">Paste a compound SMILES first.</div>';
        return;
      }
      btn.disabled = true;
      resultEl.innerHTML = '<div class="placeholder loading">Predicting cell response...</div>';
      try {
        const res = await fetch("/predict", {
          method: "POST",
          headers: {"content-type": "application/json"},
          body: JSON.stringify({smiles, top_k: Number(topKEl.value)})
        });
        const data = await res.json();
        if (!res.ok) throw new Error(data.detail || "Prediction failed");

        const genes = data.top_changed_genes.map(g => `
          <tr>
            <td><code>${esc(g.gene_id)}</code></td>
            <td>${Number(g.predicted_expression).toFixed(4)}</td>
            <td>${Number(g.delta_vs_mean).toFixed(4)}</td>
            <td>${direction(Number(g.delta_vs_mean))}</td>
          </tr>
        `).join("");
        const nearest = data.nearest_training_compounds.map(c => `
          <tr>
            <td><code>${esc(c.compound)}</code></td>
            <td>${Number(c.tanimoto).toFixed(4)}</td>
          </tr>
        `).join("");

        resultEl.innerHTML = `
          <h2>Predicted Response</h2>
          <p class="small">Input SMILES: <code>${esc(data.smiles)}</code></p>
          <div class="two-col">
            <div>
              <h2>Top Changed Genes</h2>
              <table>
                <thead><tr><th>Gene</th><th>Expression</th><th>Delta</th><th>Direction</th></tr></thead>
                <tbody>${genes}</tbody>
              </table>
            </div>
            <div>
              <h2>Nearest Training Compounds</h2>
              <table>
                <thead><tr><th>Compound</th><th>Tanimoto</th></tr></thead>
                <tbody>${nearest}</tbody>
              </table>
              <div class="notice">
                A higher Tanimoto score means the query compound is more chemically similar to a training compound.
              </div>
            </div>
          </div>
        `;
      } catch (err) {
        resultEl.innerHTML = `<div class="error">${esc(err.message)}</div>`;
      } finally {
        btn.disabled = false;
      }
    }

    document.getElementById("predictBtn").addEventListener("click", predict);
    document.getElementById("clearBtn").addEventListener("click", () => {
      smilesEl.value = "";
      resultEl.innerHTML = '<div class="placeholder">Paste a SMILES string and run a prediction.</div>';
    });
    document.querySelectorAll(".chip").forEach(chip => {
      chip.addEventListener("click", () => {
        smilesEl.value = chip.dataset.smiles;
        predict();
      });
    });
    smilesEl.addEventListener("keydown", ev => {
      if ((ev.metaKey || ev.ctrlKey) && ev.key === "Enter") predict();
    });

    loadHealth();
  </script>
</body>
</html>
    """


@app.get("/health")
def health() -> dict:
    a = _load()
    return {
        "status": "ok",
        "model": f"{a.get('model_kind', 'ridge')}(Morgan ECFP4)",
        "n_train_genes": len(a["train_genes"]),
        "n_scored_genes": len(a["scored_genes"]),
        "n_train_compounds": int(len(a["train_compounds"])),
    }


@app.post("/predict")
def predict(req: PredictRequest) -> dict:
    a = _load()
    fp = _featurize(req.smiles)

    train_genes: list[str] = a["train_genes"]
    pred = np.clip(a["model"].predict(fp[None, :])[0], 0.0, None).astype(np.float32)
    gene_mean = a["gene_mean"]
    delta = pred - gene_mean  # change vs the average training compound

    order = np.argsort(-np.abs(delta))[: req.top_k]
    top_changed = [
        {
            "gene_id": train_genes[i],
            "predicted_expression": round(float(pred[i]), 4),
            "delta_vs_mean": round(float(delta[i]), 4),
        }
        for i in order
    ]

    # Nearest training compounds by Tanimoto on Morgan bits.
    train_fps = np.unpackbits(a["train_fps_packed"], axis=1)[:, : a["n_bits"]].astype(bool)
    q = fp.astype(bool)
    inter = (train_fps & q).sum(axis=1)
    union = (train_fps | q).sum(axis=1)
    tani = np.where(union > 0, inter / union, 0.0)
    nn = np.argsort(-tani)[:5]
    nearest = [
        {"compound": str(a["train_compounds"][i]), "tanimoto": round(float(tani[i]), 4)}
        for i in nn
    ]

    out: dict = {
        "smiles": req.smiles,
        "n_genes_modeled": len(train_genes),
        "top_changed_genes": top_changed,
        "nearest_training_compounds": nearest,
    }
    if req.full:
        # Full scored-gene vector: modeled genes from Ridge, the rest at
        # the global training mean (same fallback as the submission).
        pos = {g: j for j, g in enumerate(train_genes)}
        full = {
            g: round(float(pred[pos[g]]) if g in pos else a["global_mean"], 4)
            for g in a["scored_genes"]
        }
        out["predicted_expression"] = full
    return out
