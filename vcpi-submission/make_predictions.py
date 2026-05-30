#!/usr/bin/env python3
"""Reproduce the VCPI contest submission predictions.parquet.

Strategy: 35% per-gene mean baseline + 65% Tanimoto KNN (k=100, power=2).
Training data: tvc-qnu-012 + tvc-kdl-010 combined (11,754 compounds).
Validated wMSE: 0.22656 (scaffold-split, lower is better).

Usage
-----
First run (fetches training data from the API):

    python make_predictions.py

Re-run without re-fetching (uses cached artifacts/):

    python make_predictions.py --skip-fetch

Offline demo mode (no TVC_TOKEN required, synthetic data):

    python make_predictions.py --demo

Output
------
    predictions.parquet             — final file to submit
    artifacts/predictions.parquet   — same file, kept with generated artifacts

Requirements
------------
See requirements.txt. Activate your venv first:

    source .venv/bin/activate       # macOS/Linux
    .venv\\Scripts\\activate         # Windows

Token
-----
Set the TVC_TOKEN environment variable, or run `vcpi login` once to cache
it in the system keychain:

    export TVC_TOKEN=your_token_here
    python make_predictions.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
ARTIFACTS = ROOT / "artifacts"
EXPRESSION_WIDE_PATH = ARTIFACTS / "train_expression_wide.parquet"
CHEMISTRY_PATH = ARTIFACTS / "train_chemistry.parquet"
PREDICTIONS_PATH = ARTIFACTS / "predictions.parquet"
SUBMISSION_PATH = ROOT / "predictions.parquet"

JOBS = ["tvc-qnu-012", "tvc-kdl-010"]   # training releases to combine

K = 100           # KNN neighbors
POWER = 2.0       # Tanimoto similarity weight exponent
BASELINE_W = 0.35 # weight on per-gene mean baseline
KNN_W = 0.65      # weight on KNN signal

RADIUS = 2
N_BITS = 2048


# ---------------------------------------------------------------------------
# Fingerprints
# ---------------------------------------------------------------------------

def build_fingerprints(smiles: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """SMILES list -> (N × N_BITS float32 matrix, bool validity mask)."""
    from rdkit import Chem
    from rdkit.Chem import rdFingerprintGenerator

    gen = rdFingerprintGenerator.GetMorganGenerator(radius=RADIUS, fpSize=N_BITS)
    x = np.zeros((len(smiles), N_BITS), dtype=np.float32)
    valid = np.zeros(len(smiles), dtype=bool)
    for i, smi in enumerate(smiles):
        if not isinstance(smi, str) or not smi:
            continue
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        x[i] = gen.GetFingerprintAsNumPy(mol).astype(np.float32)
        valid[i] = True
    print(f"[fp] {int(valid.sum())}/{len(smiles)} SMILES featurized ({N_BITS} bits)")
    return x, valid


# ---------------------------------------------------------------------------
# KNN predictor
# ---------------------------------------------------------------------------

def _tanimoto_batch(
    query: np.ndarray, ref: np.ndarray, ref_sum: np.ndarray
) -> np.ndarray:
    inter = query @ ref.T
    denom = query.sum(axis=1, keepdims=True) + ref_sum[None, :] - inter
    return np.divide(
        inter, denom,
        out=np.zeros_like(inter, dtype=np.float32),
        where=denom > 0,
    )


def knn_predict(
    x_query: np.ndarray,
    x_ref: np.ndarray,
    y_ref: np.ndarray,
    *,
    k: int,
    power: float,
    batch_size: int = 128,
) -> np.ndarray:
    """Tanimoto-weighted KNN expression prediction."""
    x_ref = x_ref.astype(np.float32)
    x_query = x_query.astype(np.float32)
    ref_sum = x_ref.sum(axis=1).astype(np.float32)
    out = np.empty((x_query.shape[0], y_ref.shape[1]), dtype=np.float32)
    n = x_query.shape[0]

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        sims = _tanimoto_batch(x_query[start:end], x_ref, ref_sum)
        kk = min(k, x_ref.shape[0])
        idx = np.argpartition(-sims, kth=kk - 1, axis=1)[:, :kk]
        top = np.take_along_axis(sims, idx, axis=1)
        w = np.power(np.maximum(top, 0), power).astype(np.float32)
        ws = w.sum(axis=1, keepdims=True)
        w = np.divide(w, ws, out=np.full_like(w, 1.0 / kk), where=ws > 0)
        for row, ri in enumerate(idx):
            out[start + row] = w[row] @ y_ref[ri]
        print(f"  {end}/{n}", end="\r")

    print()
    return np.clip(out, 0.0, None)


# ---------------------------------------------------------------------------
# Fetch + normalize training data
# ---------------------------------------------------------------------------

def fetch_and_cache(jobs: list[str]) -> None:
    """Download training data from the VCPI API and cache as parquet."""
    import os
    import getpass
    import polars as pl
    import vcpi
    from vcpi_prediction_contest import load_gene_filter

    # --- auth ---
    token = os.environ.get("TVC_TOKEN")
    if not token and sys.stdin.isatty():
        token = getpass.getpass("Paste your TVC API token (input hidden): ").strip()
    if token:
        os.environ["TVC_TOKEN"] = token
        vcpi.login(token)
        print("[auth] logged in with TVC_TOKEN")
    else:
        print("[auth] no token provided — relying on cached keychain token")

    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    gene_filter = load_gene_filter()
    expr_parts: list[pd.DataFrame] = []
    chem_parts: list[pd.DataFrame] = []

    for job in jobs:
        print(f"\n[fetch] {job} ...")
        exp = vcpi.load_experiment(job)

        meta = exp["metadata"].filter(
            (pl.col("cell_line") == "THP-1")
            & (pl.col("timepoint") == "24h")
            & (
                (
                    (pl.col("compound_concentration") == 10_000)
                    & (pl.col("compound_concentration_unit") == "nM")
                )
                | (pl.col("user_compound_id") == "DMSO")
            )
        )
        keep = set(meta["sequenced_id"].cast(pl.Utf8).to_list())
        data = exp["data"].select(
            ["gene_id", *[c for c in exp["data"].columns if c != "gene_id" and c in keep]]
        )

        counts_df = data.to_pandas()
        meta_df = meta.to_pandas()
        chem_df = (
            exp["chemistry"].to_pandas()
            .drop_duplicates(subset=["compound"])
            .reset_index(drop=True)
        )
        chem_df["source_job"] = job
        chem_parts.append(chem_df)

        # log2(CPM + 1) → compound × gene mean
        counts_df["gene_id"] = counts_df["gene_id"].astype(str)
        gene_set = set(map(str, gene_filter))
        counts_df = counts_df[counts_df["gene_id"].isin(gene_set)].set_index("gene_id")
        counts_df.columns = counts_df.columns.astype(str)

        meta_df["sequenced_id"] = meta_df["sequenced_id"].astype(str)
        meta_df["user_compound_id"] = meta_df["user_compound_id"].astype(str)
        s2c = pd.Series(
            meta_df["user_compound_id"].to_numpy(),
            index=meta_df["sequenced_id"].to_numpy(),
        )

        shared = [s for s in counts_df.columns if s in s2c.index]
        counts_df = counts_df[shared]
        s2c = s2c.loc[shared]

        lib = counts_df.sum(axis=0).replace(0, np.nan)
        log_cpm = np.log2(counts_df.div(lib, axis=1) * 1e6 + 1.0).astype("float32")
        log_cpm.columns = pd.Index(s2c.to_numpy(), name="compound")
        wide = log_cpm.T.groupby(level="compound", sort=True).mean().astype("float32")
        wide = wide.reindex(columns=gene_filter)
        wide.index.name = "compound"
        expr_parts.append(wide)
        print(f"[fetch] {job}: {wide.shape[0]:,} compounds × {wide.shape[1]:,} genes")

    # --- combine jobs ---
    combined = pd.concat(expr_parts, axis=0)
    if combined.index.has_duplicates:
        n = int(combined.index.duplicated().sum())
        print(f"[data] averaging {n:,} duplicate compound IDs across jobs")
        combined = combined.groupby(level=0, sort=True).mean()
    combined = combined.sort_index().astype("float32")

    chemistry = (
        pd.concat(chem_parts, ignore_index=True)
        .drop_duplicates(subset=["compound"])
        .reset_index(drop=True)
    )

    combined.to_parquet(EXPRESSION_WIDE_PATH)
    chemistry.to_parquet(CHEMISTRY_PATH)
    print(f"\n[save] {EXPRESSION_WIDE_PATH.name}: {combined.shape[0]:,} × {combined.shape[1]:,}")
    print(f"[save] {CHEMISTRY_PATH.name}: {len(chemistry):,} compounds with chemistry")


# ---------------------------------------------------------------------------
# Demo mode (offline, synthetic)
# ---------------------------------------------------------------------------

def build_demo_artifacts() -> None:
    """Create synthetic training artifacts so the pipeline runs offline."""
    from vcpi_prediction_contest import load_gene_filter, load_test_compounds

    rng = np.random.default_rng(42)
    genes = load_gene_filter()
    test_df = load_test_compounds()

    # Fabricate a small synthetic training set from test compound SMILES.
    n_train = min(500, len(test_df))
    sample = test_df.sample(n_train, random_state=42).reset_index(drop=True)

    x, valid = build_fingerprints(sample["smiles"].tolist())
    sample = sample.loc[valid].reset_index(drop=True)
    x = x[valid]

    proj = rng.normal(0, 0.04, size=(N_BITS, len(genes))).astype(np.float32)
    base = rng.uniform(1.0, 4.0, size=len(genes)).astype(np.float32)
    noise = rng.normal(0, 0.3, size=(len(sample), len(genes))).astype(np.float32)
    expr_vals = np.clip(base + x @ proj + noise, 0.0, None)

    ARTIFACTS.mkdir(parents=True, exist_ok=True)

    wide = pd.DataFrame(expr_vals, index=sample["compound"].astype(str), columns=genes)
    wide.to_parquet(EXPRESSION_WIDE_PATH)

    chem = sample[["compound", "smiles"]].rename(columns={"compound": "compound"})
    chem.to_parquet(CHEMISTRY_PATH)

    print(f"[demo] synthetic artifacts: {len(sample)} compounds × {len(genes)} genes")


# ---------------------------------------------------------------------------
# Main prediction pipeline
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--skip-fetch", action="store_true",
        help="skip API fetch; reuse existing artifacts/ parquet files",
    )
    parser.add_argument(
        "--demo", action="store_true",
        help="offline demo mode — builds synthetic training data (no token needed)",
    )
    parser.add_argument(
        "--jobs", default=",".join(JOBS),
        help=f"comma-separated VCPI job IDs to combine (default: {','.join(JOBS)})",
    )
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Step 1 — get training data
    # ------------------------------------------------------------------
    if args.demo:
        print("\n[demo] building synthetic training artifacts (offline mode)")
        build_demo_artifacts()
    elif not args.skip_fetch:
        jobs = [j.strip() for j in args.jobs.split(",") if j.strip()]
        print(f"\n[fetch] fetching training data for: {jobs}")
        fetch_and_cache(jobs)
    else:
        if not EXPRESSION_WIDE_PATH.exists():
            print(
                f"ERROR: --skip-fetch set but {EXPRESSION_WIDE_PATH} not found.\n"
                "Run without --skip-fetch to download training data first.",
                file=sys.stderr,
            )
            return 1
        print(f"[skip-fetch] using cached {EXPRESSION_WIDE_PATH.name}")

    # ------------------------------------------------------------------
    # Step 2 — load training matrix
    # ------------------------------------------------------------------
    from vcpi_prediction_contest import load_gene_filter, load_test_compounds

    print("\n[load] training expression matrix ...")
    wide = pd.read_parquet(EXPRESSION_WIDE_PATH)
    wide.index = wide.index.astype(str)
    wide.columns = wide.columns.astype(str)
    wide = wide.fillna(wide.mean(axis=0))

    chem = pd.read_parquet(CHEMISTRY_PATH)
    # Real VCPI chemistry has both `compound` (internal UUID) and
    # `user_compound_id` (contest join key). Prefer the contest key.
    compound_col = "user_compound_id" if "user_compound_id" in chem.columns else "compound"
    chem[compound_col] = chem[compound_col].astype(str)
    smap = (
        chem.dropna(subset=["smiles"])
        .drop_duplicates(compound_col)
        .set_index(compound_col)["smiles"]
    )
    common = [c for c in wide.index if c in smap.index]
    wide = wide.loc[common]
    smiles_train = smap.loc[common]
    train_genes = list(wide.columns)
    y = wide.to_numpy(dtype=np.float32)
    print(f"[data] {wide.shape[0]:,} training compounds × {wide.shape[1]:,} genes")

    # ------------------------------------------------------------------
    # Step 3 — fingerprints
    # ------------------------------------------------------------------
    print("\n[fp] building fingerprints for training set ...")
    x_tr, valid_tr = build_fingerprints(smiles_train.tolist())
    if (~valid_tr).any():
        keep = np.where(valid_tr)[0]
        y, x_tr = y[keep], x_tr[keep]
        print(f"[fp] dropped {(~valid_tr).sum()} unparseable train SMILES")

    gene_mean = y.mean(axis=0)   # per-gene baseline

    print("[fp] building fingerprints for test set ...")
    test_df = load_test_compounds()
    test_df["compound"] = test_df["compound"].astype(str)
    x_test, valid_test = build_fingerprints(test_df["smiles"].tolist())
    n_invalid = int((~valid_test).sum())
    if n_invalid:
        print(f"[fp] {n_invalid} test SMILES unparseable — will use mean fallback")

    # ------------------------------------------------------------------
    # Step 4 — KNN predictions (k=100)
    # ------------------------------------------------------------------
    print(f"\n[knn] Tanimoto KNN k={K}, power={POWER} ...")
    knn_raw = knn_predict(x_test, x_tr, y, k=K, power=POWER)
    if n_invalid:
        knn_raw[~valid_test] = gene_mean

    # ------------------------------------------------------------------
    # Step 5 — align to full scored gene set
    # ------------------------------------------------------------------
    scored_genes = load_gene_filter()
    gene_pos = {g: i for i, g in enumerate(scored_genes)}
    train_col = {g: j for j, g in enumerate(train_genes)}
    shared = [g for g in train_genes if g in gene_pos]
    g_pos = [gene_pos[g] for g in shared]
    t_col = [train_col[g] for g in shared]

    global_mean = float(y.mean())
    n_test = len(test_df)
    n_scored = len(scored_genes)

    knn_mat = np.full((n_test, n_scored), global_mean, dtype=np.float32)
    knn_mat[:, g_pos] = knn_raw[:, t_col]

    baseline_mat = np.full((n_test, n_scored), global_mean, dtype=np.float32)
    baseline_mat[:, g_pos] = gene_mean[t_col]

    # ------------------------------------------------------------------
    # Step 6 — blend
    # ------------------------------------------------------------------
    print(f"\n[blend] {int(BASELINE_W*100)}% baseline + {int(KNN_W*100)}% KNN")
    blend_mat = np.clip(BASELINE_W * baseline_mat + KNN_W * knn_mat, 0.0, None).astype(np.float32)

    # ------------------------------------------------------------------
    # Step 7 — save
    # ------------------------------------------------------------------
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    long = (
        pd.DataFrame(blend_mat, index=test_df["compound"].astype(str), columns=scored_genes)
        .reset_index(names="compound")
        .melt(id_vars="compound", var_name="gene_id", value_name="predicted_expression")
    )
    long["predicted_expression"] = long["predicted_expression"].astype("float32")
    long = long[["compound", "gene_id", "predicted_expression"]]
    long.to_parquet(PREDICTIONS_PATH, index=False)
    long.to_parquet(SUBMISSION_PATH, index=False)

    expr = long["predicted_expression"]
    expected = n_test * n_scored
    print(f"\n[save] {PREDICTIONS_PATH}: {len(long):,} rows")
    print(f"[save] {SUBMISSION_PATH}: {len(long):,} rows")
    print(f"[stat] min={expr.min():.4f}  mean={expr.mean():.4f}  max={expr.max():.4f}")

    # ------------------------------------------------------------------
    # Step 8 — quick validation
    # ------------------------------------------------------------------
    assert len(long) == expected, f"row count {len(long)} != {expected}"
    assert long["compound"].nunique() == n_test, "missing test compounds"
    assert long["gene_id"].nunique() == n_scored, "missing scored genes"
    assert expr.isna().sum() == 0, "NaN predictions found"
    assert (expr < 0).sum() == 0, "negative predictions found"
    print("[check] all assertions passed — predictions.parquet is submission-ready.\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
