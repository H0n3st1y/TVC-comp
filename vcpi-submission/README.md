# Virtual Cell Predictor — VCPI Hackathon Submission

**Team:** Anghelo  
**Contest:** AIxBio Virtual Cell Predictor (VCPI)  
**Date:** May 30, 2026

---

## What It Does

Given a compound's chemical structure (SMILES string), this model predicts how every scored gene in a THP-1 immune cell will respond after 24 hours of treatment at 10 µM.

**Input:** A compound SMILES string  
**Output:** Predicted `log2(CPM + 1)` expression level for 12,995 genes

---

## How It Works

```
Compound SMILES
     ↓
RDKit Morgan Fingerprint (ECFP4 — 2048 bits, radius 2)
     ↓
Ridge Regression (α = 10,000)
     ↓
12,995-gene expression vector (log2 CPM + 1)
```

**Why Ridge?** Ridge regression on binary chemical fingerprints is fast, interpretable, and competitive. A scaffold-split grid search over α ∈ {10, 100, 1000, 10000, 100000} found α=10,000 as the optimum — enough regularization to generalize to novel chemotypes without collapsing to the mean.

**Why ECFP4?** Morgan fingerprints (radius 2, 2048 bits) are the standard for compound-activity modeling. They encode circular substructures around each atom, capturing both local chemistry and broader scaffold features.

---

## Training Data

| Dataset | Job ID | Compounds | Samples |
|---|---|---|---|
| vcpi-0003 | tvc-qnu-012 | 10,261 | 21.9K |

- **Cell line:** THP-1 (human monocyte)
- **Condition:** 24h treatment at 10 µM + DMSO controls
- **Target:** Per-compound mean `log2(CPM + 1)` across replicates

---

## Validation Results (Scaffold Split, 200 held-out compounds)

| Model | wMSE | vs Baseline |
|---|---|---|
| **Ridge α=10,000 (submitted)** | **0.2287** | **−0.0004 ✅ beats baseline** |
| Per-gene mean baseline | 0.2291 | — |
| Ridge α=10 (default) | 0.2568 | +0.0277 worse |

Validation uses a **Bemis–Murcko scaffold split** — held-out compounds carry scaffolds not seen during training, which is a more honest estimate of generalization to the contest's novel test compounds.

---

## Submission File

| File | Rows | Compounds | Genes |
|---|---|---|---|
| `predictions.parquet` | 13,826,680 | 1,064 | 12,995 |

Columns: `compound` (user_compound_id), `gene_id`, `predicted_expression`  
All values are non-negative `log2(CPM + 1)`.

---

## API Demo

The model is served as a FastAPI app. Start with:

```bash
pip install fastapi "uvicorn[standard]" rdkit scikit-learn joblib
uvicorn app:app --port 8000
```

**GET `/health`** — model status  
**POST `/predict`** — predict expression for a SMILES

```bash
curl -s localhost:8000/predict \
  -H 'content-type: application/json' \
  -d '{"smiles": "CC(=O)Oc1ccccc1C(=O)O", "top_k": 5}'
```

Returns:
- `top_changed_genes` — genes with largest predicted deviation from training mean
- `nearest_training_compounds` — most structurally similar training compounds (Tanimoto)
- `full=true` — full 12,995-gene expression vector

---

## Files

| File | Purpose |
|---|---|
| `predictions.parquet` | Contest submission |
| `train_pipeline.py` | Full training pipeline (fetch → filter → featurize → Ridge → validate → export) |
| `models.py` | MLP architecture (PyTorch) — available as `--model mlp` |
| `app.py` | FastAPI demo server |

---

## Next Steps (if more time)

1. **Add tvc-kdl-010 + tvc-bhr-009** — 3,780 more compounds from different chemotypes; more diversity helps Ridge generalize further
2. **MLP** — already implemented in `models.py`; needs more epochs on real data to compete
3. **Target PCA** — compress 12,995 genes → 256 components before MLP to reduce overfitting
4. **ChemBERTa or GNN encoder** — richer molecular representations beyond binary fingerprints
