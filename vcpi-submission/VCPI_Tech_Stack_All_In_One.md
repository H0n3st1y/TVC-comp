# VCPI Virtual Cell Tech Stack All-In-One

## 1. What This Project Does

This project predicts how THP-1 cells respond to chemical compounds.

Input:

```text
compound SMILES
```

Output:

```text
predicted gene expression for 12,995 genes
```

Final submission file:

```text
/Users/ang/Downloads/vcpi-submission/predictions.parquet
```

## 2. Biology Goal

When a compound is added to a cell, the cell changes gene activity.

Some genes go up.

Some genes go down.

Some genes stay near the same level.

The model predicts the final gene expression state after treatment.

Simple version:

```text
compound -> THP-1 cell response -> gene expression prediction
```

## 3. Dataset Used

Main training dataset:

```text
vcpi-0003 = tvc-qnu-012
```

Training size:

```text
10,261 compounds
12,995 scored genes
```

Test set:

```text
1,064 compounds
12,995 scored genes
13,826,680 final prediction rows
```

## 4. Main Data Pipeline

Main file:

```text
train_pipeline.py
```

What it does:

```text
1. Downloads VCPI data using vcpi-client.
2. Loads raw count data, metadata, and compound chemistry.
3. Converts counts into log2(CPM + 1).
4. Aligns compounds with SMILES strings.
5. Builds Morgan chemical fingerprints.
6. Trains/evaluates models.
7. Writes predictions.parquet.
```

Important output:

```text
artifacts/train_expression_wide.parquet
```

This is the compact training matrix:

```text
compound x gene expression
```

## 5. Final Submission Model

The final model is a validation-optimized blend.

It uses:

```text
40% per-gene mean baseline
60% KNN chemical-neighbor prediction
```

Validation score:

```text
wMSE = 0.2266
```

Previous best:

```text
wMSE = 0.2277
```

So the latest final file improved the benchmark.

## 6. Per-Gene Mean Baseline

This is the simplest model.

For each gene, it predicts the average expression value seen in the training data.

It ignores the compound.

Why it matters:

```text
It is a strong baseline because many genes do not change dramatically.
```

## 7. Morgan Fingerprints

A Morgan fingerprint turns a compound structure into numbers.

In this project:

```text
SMILES -> 2,048-bit Morgan fingerprint
```

Simple explanation:

```text
Each bit answers whether a certain chemical pattern exists in the molecule.
```

Example:

```text
ring pattern present -> 1
specific atom neighborhood absent -> 0
```

The fingerprint is used to compare compounds and find chemical neighbors.

## 8. KNN Chemical-Neighbor Model

KNN means:

```text
K nearest neighbors
```

For a test compound, the model finds similar training compounds using Morgan fingerprint similarity.

Similarity metric:

```text
Tanimoto similarity
```

Then it predicts gene expression by averaging the expression profiles of chemically similar compounds.

Final best setting:

```text
k = 100
```

## 9. Blend Optimizer

Main file:

```text
scripts/optimize_submission_blend.py
```

What it does:

```text
1. Tests baseline.
2. Tests Ridge models.
3. Tests KNN models.
4. Tests blends of baseline + Ridge + KNN.
5. Keeps the best validation result.
6. Only promotes a new prediction file if it beats the benchmark.
```

Best result found:

```text
40% baseline
0% Ridge
60% KNN
wMSE = 0.2266
```

Why this is useful:

```text
It improves the final file without risking the existing submission.
```

## 10. Ridge Regression

Ridge is a regularized linear model.

Input:

```text
Morgan fingerprint
```

Output:

```text
gene expression vector
```

Best tested Ridge:

```text
alpha = 10000
wMSE around 0.2287
```

Ridge helped, but the final blend performed better.

## 11. MLP And PCA Support

Main file:

```text
models.py
```

Includes:

```text
TorchMLPRegressor
PCATargetRegressor
```

MLP means:

```text
multi-layer perceptron neural network
```

Target PCA means:

```text
compress 12,995 gene outputs into fewer expression programs
```

Example:

```text
12,995 genes -> 256 PCA components -> predict -> convert back to genes
```

This is future-facing model support. It is not the current final submission model.

## 12. Advanced Architecture Components

Main file:

```text
advanced_virtual_cell_components.py
```

This contains optional future model upgrades:

```text
Scaffold split
ChemBERTa dataset/model wrapper
PyTorch Geometric graph conversion
GCN/GAT graph neural network model
Gaussian NLL uncertainty loss
Pathway-to-gene decoder
GNN + pathway virtual cell model
```

These are not needed to submit the current file.

They are included as next-step engineering support.

## 13. Scaffold Split

Scaffold split groups compounds by their core chemical skeleton.

This makes validation harder and more realistic.

Instead of random splitting:

```text
similar compounds can leak between train and validation
```

Scaffold split does:

```text
train on one chemical family
validate on different chemical families
```

This better tests whether the model generalizes to new compounds.

## 14. ChemBERTa

ChemBERTa is a transformer model for SMILES strings.

It treats chemical strings like language.

Possible future use:

```text
SMILES -> ChemBERTa embedding -> gene expression model
```

Why it may help:

```text
It can learn deeper chemical context than fixed fingerprints.
```

## 15. Graph Neural Networks

Molecules are naturally graphs.

Atoms are nodes.

Bonds are edges.

GNN input:

```text
molecular graph
```

GNN output:

```text
learned molecular embedding
```

Possible future use:

```text
molecule graph -> GNN embedding -> gene expression prediction
```

## 16. Gaussian NLL Uncertainty

Normal models output one number.

Gaussian NLL lets the model output:

```text
mean prediction
uncertainty / variance
```

Example:

```text
expression = 4.2 +/- 0.6
```

Why it matters:

```text
It helps flag predictions where the model is unsure.
```

## 17. Pathway Layer

Genes do not work alone.

They work in biological groups called pathways.

Examples:

```text
inflammation pathway
stress-response pathway
cell-cycle pathway
metabolism pathway
```

The pathway decoder works like:

```text
compound embedding -> pathway activations -> gene expression
```

Why it may help:

```text
It makes predictions more biologically structured.
```

## 18. Local Website And API

Main file:

```text
app.py
```

Run:

```bash
.venv/bin/uvicorn app:app --port 8000
```

Open:

```text
http://127.0.0.1:8000/
```

The website lets a user:

```text
paste a SMILES string
run prediction
see top changed genes
see nearest training compounds
read a simple biology explanation
```

API docs:

```text
http://127.0.0.1:8000/docs
```

Health endpoint:

```text
http://127.0.0.1:8000/health
```

## 19. Final Submission Folder

Folder:

```text
/Users/ang/Downloads/vcpi-submission
```

Final file:

```text
/Users/ang/Downloads/vcpi-submission/predictions.parquet
```

Zip:

```text
/Users/ang/Downloads/vcpi-submission-final.zip
```

Unless organizers ask for code, submit only:

```text
predictions.parquet
```

## 20. Final Validation

The final file passed:

```text
Rows: 13,826,680
Compounds: 1,064 / 1,064
Genes: 12,995 / 12,995
Predicted expression: min=0.0058, mean=3.6581, max=14.0582
OK: prediction file is submission-shaped
```

## 21. Current Best Summary

Current best model:

```text
40% per-gene mean baseline
60% KNN chemical-neighbor signal
```

Current best validation:

```text
wMSE = 0.2266
```

Main reason it works:

```text
It keeps the strong stable baseline, but adds chemical information from similar training compounds.
```

## 22. One-Minute Pitch

We built a virtual THP-1 cell response predictor.

It uses real VCPI compound-response data from `vcpi-0003`.

For each test compound, it predicts expression values for all 12,995 scored genes.

The final model combines a strong per-gene expression baseline with a chemical-neighbor signal from Morgan fingerprint similarity.

The local website lets a user paste a compound SMILES and view predicted gene changes plus chemically similar training compounds.

