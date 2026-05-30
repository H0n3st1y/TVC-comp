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
