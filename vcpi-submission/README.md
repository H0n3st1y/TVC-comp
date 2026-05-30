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
Predicted expression: min=0.0058, mean=3.6581, max=14.0582
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
vcpi-0003 / tvc-qnu-012
10,261 training compounds
12,995 scored genes
THP-1 cells, 24-hour compound response
```

Final prediction strategy:

```text
40% per-gene mean baseline
60% Morgan fingerprint KNN chemical-neighbor signal
```

Validation result:

```text
Per-gene mean baseline wMSE: 0.2291
Ridge alpha=10000 wMSE:      0.2287
40/60 baseline+KNN wMSE:     0.2266
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
574b47b79deccd5f09d3aae1242a23d07369984ba924b4514d25870286107a01
```

## Part 4: Technical Glossary For Future Enhancements

If there is time to push beyond the current `0.2277` validation benchmark, these are the next useful concepts:

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
