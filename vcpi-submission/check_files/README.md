# check_files/

Validation folder for the VCPI submission.

## Contents

```
test_compounds.csv      — the 1,064 test compounds from the contest
validate_submission.py  — checker script (run to verify predictions.parquet)
VALIDATION_REPORT.txt   — pre-run validation results (all checks passed)
README.md               — this file
```

## How to run the checker

From the `vcpi-submission/` folder:

```bash
python3 check_files/validate_submission.py
```

Expected output:

```
RESULT: ALL CHECKS PASSED — file is submission-ready.
```

To point at a different predictions file:

```bash
python3 check_files/validate_submission.py \
    --predictions /path/to/predictions.parquet \
    --test-compounds check_files/test_compounds.csv
```

## What it checks

1. Required columns exist (`compound`, `gene_id`, `predicted_expression`)
2. Exactly 13,826,680 rows (1,064 compounds × 12,995 genes)
3. All 1,064 test compounds are covered
4. All 12,995 scored genes are present
5. Each compound has exactly 12,995 gene rows (spot-check)
6. No NaN predictions
7. No negative predictions
8. No duplicate (compound, gene_id) pairs
