#!/usr/bin/env python
"""Validate predictions.parquet against test_compounds.csv.

Checks:
  1. All 1,064 test compounds are covered (by compound ID).
  2. All 12,995 scored genes are present for every compound.
  3. No missing (NaN) predictions.
  4. No negative predictions (log2(CPM+1) is non-negative).
  5. No duplicate (compound, gene_id) pairs.
  6. Row count is exactly 13,826,680.
  7. Required columns exist: compound, gene_id, predicted_expression.

Usage (run from vcpi-submission/ folder):
    python check_files/validate_submission.py

Or point to custom paths:
    python check_files/validate_submission.py \
        --predictions ../artifacts/predictions.parquet \
        --test-compounds check_files/test_compounds.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


REQUIRED_COLUMNS = {"compound", "gene_id", "predicted_expression"}
EXPECTED_ROWS = 13_826_680
EXPECTED_COMPOUNDS = 1_064
EXPECTED_GENES = 12_995


def run_checks(predictions_path: Path, test_compounds_path: Path) -> bool:
    import pandas as pd

    ok = True

    def fail(msg: str) -> None:
        nonlocal ok
        print(f"  FAIL  {msg}")
        ok = False

    def check(label: str, passed: bool, detail: str = "") -> None:
        nonlocal ok
        if passed:
            print(f"  ok    {label}" + (f"  ({detail})" if detail else ""))
        else:
            fail(label + (f"  ({detail})" if detail else ""))

    # ---- load files -------------------------------------------------------
    print(f"\nLoading predictions: {predictions_path}")
    df = pd.read_parquet(predictions_path)
    print(f"  {len(df):,} rows, columns: {list(df.columns)}")

    print(f"\nLoading test compounds: {test_compounds_path}")
    tc = pd.read_csv(test_compounds_path, dtype={"compound": str})
    expected_cpds = set(tc["compound"].astype(str))
    print(f"  {len(expected_cpds):,} unique compounds")

    print("\n--- Checks ---")

    # 1. Required columns
    missing_cols = REQUIRED_COLUMNS - set(df.columns)
    check("required columns present", not missing_cols,
          f"missing: {missing_cols}" if missing_cols else "compound, gene_id, predicted_expression")

    if "compound" not in df.columns or "gene_id" not in df.columns or "predicted_expression" not in df.columns:
        print("\nCannot continue — required columns missing.")
        return False

    # Cast compound to str for comparison
    df["compound"] = df["compound"].astype(str)

    # 2. Row count
    check("row count", len(df) == EXPECTED_ROWS,
          f"{len(df):,} (expected {EXPECTED_ROWS:,})")

    # 3. Compound coverage
    pred_cpds = set(df["compound"].unique())
    n_covered = len(pred_cpds & expected_cpds)
    missing_cpds = expected_cpds - pred_cpds
    extra_cpds = pred_cpds - expected_cpds
    check(
        f"all {EXPECTED_COMPOUNDS} test compounds covered",
        n_covered == EXPECTED_COMPOUNDS and not missing_cpds,
        f"{n_covered}/{EXPECTED_COMPOUNDS} covered"
        + (f", missing {len(missing_cpds)}: {sorted(missing_cpds)[:5]}..." if missing_cpds else "")
        + (f", {len(extra_cpds)} extra compounds in predictions" if extra_cpds else ""),
    )

    # 4. Gene coverage
    n_genes = df["gene_id"].nunique()
    check(f"all {EXPECTED_GENES} genes present", n_genes == EXPECTED_GENES,
          f"{n_genes:,} unique genes found")

    # 5. Per-compound gene count (sample first 50 compounds for speed)
    sample_cpds = sorted(pred_cpds & expected_cpds)[:50]
    sub = df[df["compound"].isin(sample_cpds)]
    gene_counts = sub.groupby("compound")["gene_id"].nunique()
    wrong = gene_counts[gene_counts != EXPECTED_GENES]
    check(
        f"each compound has exactly {EXPECTED_GENES} gene rows (spot-check 50 compounds)",
        len(wrong) == 0,
        f"{len(wrong)} compounds with wrong gene count" if len(wrong) else "all good",
    )

    # 6. No NaN
    n_nan = df["predicted_expression"].isna().sum()
    check("no NaN predictions", n_nan == 0, f"{n_nan:,} NaN values" if n_nan else "clean")

    # 7. No negatives
    n_neg = (df["predicted_expression"] < 0).sum()
    check("no negative predictions", n_neg == 0,
          f"{n_neg:,} negative values" if n_neg else "clean")

    # 8. No duplicates
    n_dup = df.duplicated(subset=["compound", "gene_id"]).sum()
    check("no duplicate (compound, gene_id) pairs", n_dup == 0,
          f"{n_dup:,} duplicates" if n_dup else "clean")

    # ---- summary stats ----------------------------------------------------
    expr = df["predicted_expression"]
    print(f"\n--- Prediction stats ---")
    print(f"  min  : {expr.min():.4f}")
    print(f"  mean : {expr.mean():.4f}")
    print(f"  max  : {expr.max():.4f}")
    print(f"  dtype: {expr.dtype}")

    # ---- final verdict ----------------------------------------------------
    print()
    if ok:
        print("RESULT: ALL CHECKS PASSED — file is submission-ready.")
    else:
        print("RESULT: ONE OR MORE CHECKS FAILED — see FAIL lines above.")
    print()
    return ok


def main() -> int:
    here = Path(__file__).resolve().parent
    submission_dir = here.parent

    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--predictions",
        default=str(submission_dir / "predictions.parquet"),
        help="Path to predictions.parquet (default: ../predictions.parquet relative to this script)",
    )
    parser.add_argument(
        "--test-compounds",
        default=str(here / "test_compounds.csv"),
        help="Path to test_compounds.csv (default: check_files/test_compounds.csv)",
    )
    args = parser.parse_args()

    pred_path = Path(args.predictions)
    tc_path = Path(args.test_compounds)

    if not pred_path.exists():
        print(f"ERROR: predictions file not found: {pred_path}", file=sys.stderr)
        return 1
    if not tc_path.exists():
        print(f"ERROR: test_compounds file not found: {tc_path}", file=sys.stderr)
        return 1

    passed = run_checks(pred_path, tc_path)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
