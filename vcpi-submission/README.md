# VCPI Prediction Contest Submission

## Final File To Submit

Submit:

```text
predictions.parquet
```

This file contains one prediction for every required test compound and scored gene.

## Validation Summary

The file passed the local submission-shape checker:

```text
Rows: 13,826,680
Compounds: 1,064 / 1,064
Genes: 12,995 / 12,995
Predicted expression: min=0.0053, mean=3.6760, max=14.0539
OK: prediction file is submission-shaped.
```

Required columns:

```text
compound
gene_id
predicted_expression
```

## Model Summary

Training data:

```text
vcpi-0003 / tvc-qnu-012 + tvc-kdl-010 (combined)
11,754 training compounds
12,995 scored genes
THP-1 cells, 24-hour compound response
```

Final prediction strategy:

```text
35% per-gene mean baseline
65% Morgan fingerprint KNN (k=100, Tanimoto similarity, power=2)
```

Validation result:

```text
Per-gene mean baseline wMSE : 0.22908
Ridge (alpha=3000) wMSE     : 0.22861
KNN (k=100) wMSE            : 0.22739
35/65 baseline+KNN wMSE     : 0.22656  ← submitted
```

The final submitted file uses the best validation strategy above.

## Folder Contents

```text
predictions.parquet                 - final contest prediction file to submit
train_pipeline.py                   - training/data pipeline used during development
models.py                           - Ridge/MLP/PCA model definitions used during development
advanced_virtual_cell_components.py - optional future architecture components
app.py                              - local FastAPI demo app
README.md                           - this summary
MANIFEST.txt                        - checksum and validation details
```

Only `predictions.parquet` is required for the contest submission unless the organizers ask for code.

## Checksum

```text
SHA256(predictions.parquet) =
3b4d9a33a6e9fcc17b17545b750e8023528341df5a5394e1c8ac0e811d1560b3
```

## Part 4: Technical Glossary For Future Enhancements

If there is time to push beyond the current `0.22656` validation benchmark, these are the next useful concepts:

**Scaffold Split**  
Grouping train/test sets by core chemical skeletons rather than random splits. This forces the model to prove it can generalize to entirely new drug families.

**ChemBERTa And GNNs**  
Advanced alternatives to 2,048-bit fingerprints. ChemBERTa treats molecules like language strings, while graph neural networks treat molecules as atoms connected by bonds. Both can learn deeper chemical patterns than fixed fingerprints.

**Gaussian NLL**  
Negative log-likelihood training that lets the model output uncertainty, such as `expression = 4.2 +/- 0.6`, rather than only one number. This helps flag predictions where the model is unsure.

**Pathways**  
Genes work in groups, such as inflammation or stress-response pathways. A pathway layer can help the model predict logical biological programs instead of treating all 12,995 genes as unrelated values.

## Submission Status

This folder is submission-ready.

Use this file:

```text
/Users/ang/Downloads/vcpi-submission/predictions.parquet
```

## How To Submit

Email or upload only this file unless the organizers ask for code:

```text
predictions.parquet
```

Do not try to run the parquet file. It is a data file, not a program.

## How To Verify The Submission File

From the original project repo:

```bash
cd /Users/ang/Downloads/vcpi-prediction-contest-2026-main
.venv/bin/python scripts/check_predictions.py /Users/ang/Downloads/vcpi-submission/predictions.parquet
```

Expected result:

```text
Rows: 13,826,680
Compounds: 1,064 / 1,064
Genes: 12,995 / 12,995
OK: prediction file is submission-shaped.
```

## How To Reproduce Predictions

The standalone runner is:

```text
make_predictions.py
```

Create and activate an environment:

```bash
cd /Users/ang/Downloads/vcpi-submission
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Then run with a VCPI token:

```bash
export TVC_TOKEN=your_token_here
python make_predictions.py
```

Or, if you already logged in with `vcpi.login()` / cached keychain:

```bash
python make_predictions.py
```

The runner will create:

```text
predictions.parquet
artifacts/predictions.parquet
```

Both files contain the same submission predictions.

## How To Run Offline Demo Mode

This does not produce real biology predictions, but checks that the code path works:

```bash
python make_predictions.py --demo
```

Warning: demo mode writes a synthetic `predictions.parquet`. Use it only in a copied test folder, not after preparing the final submission file.

## How To Run The Local Website

From the original project repo or this folder:

```bash
uvicorn app:app --port 8000
```

Open:

```text
http://127.0.0.1:8000/
```
