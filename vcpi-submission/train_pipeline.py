#!/usr/bin/env python
"""VCPI first training pipeline.

SMILES -> RDKit Morgan fingerprint -> Ridge regression -> per-gene
log2(CPM + 1) expression vector, for the THP-1 / 24h / 10 uM contest
condition.

Stages
------
1.  Authenticate to vcpi-client (TVC_TOKEN).
2.  ``vcpi.load_experiment(JOB)`` for the configured release.
3.  Filter metadata to THP-1, 24h, 10000 nM + DMSO controls.
4.  Select the matching count columns via ``metadata.sequenced_id``.
5.  Save train_counts / train_metadata / train_chemistry parquet.
6.  ``counts_to_expression`` -> train_expression.parquet.
7.  Load bundled test_compounds + gene_filter.
8.  Build Morgan fingerprints for train + test SMILES.
9.  Train Ridge (input = fingerprint, target = expression vector).
10. Predict every test compound x every scored gene.
11. Save predictions.parquet (compound, gene_id, predicted_expression).
12. Held-out-compound validation: Ridge vs per-gene-mean baseline (wMSE).

The featurization, model, validation, and submission stages run without
the network as long as the parquet artifacts from stages 2-6 exist; only
stages 2-6 require TVC_TOKEN. Use ``--skip-fetch`` to reuse artifacts.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
JOB = "tvc-bhr-009"  # primary training release (override with --job)
RADIUS = 2
N_BITS = 2048
RIDGE_ALPHA = 10_000.0  # tuned via scaffold-split grid search on tvc-qnu-012
N_VAL_COMPOUNDS = 200
RANDOM_STATE = 42

# MLP defaults (adapted from the Colab STEP 7 cell). Overridable via CLI.
MLP_EPOCHS = 50
MLP_BATCH_SIZE = 256
MLP_LR = 1e-3

HERE = Path(__file__).resolve().parent
ARTIFACTS = HERE / "artifacts"

COUNTS_PATH = ARTIFACTS / "train_counts.parquet"
METADATA_PATH = ARTIFACTS / "train_metadata.parquet"
CHEMISTRY_PATH = ARTIFACTS / "train_chemistry.parquet"
EXPRESSION_PATH = ARTIFACTS / "train_expression.parquet"
EXPRESSION_WIDE_PATH = ARTIFACTS / "train_expression_wide.parquet"
PREDICTIONS_PATH = ARTIFACTS / "predictions.parquet"
MODEL_PATH = ARTIFACTS / "model.joblib"

# Contest join key (numeric LIMS id as a string).
COMPOUND_KEY = "user_compound_id"

# Local chemistry CSVs (chemistry only — no counts) used by --demo to
# fabricate a fingerprint-driven training target so the full stack
# (featurize -> Ridge -> scaffold validation -> serve) is testable
# offline before a TVC_TOKEN is available.
DOWNLOADS = Path("/Users/ang/Downloads")
LOCAL_COMPOUNDS = {
    "tvc-bhr-009": DOWNLOADS / "compounds-tvc-bhr-009-2026-05-30.csv",
    "tvc-qnu-012": DOWNLOADS / "datasets" / "compounds-tvc-qnu-012-2026-05-30.csv",
    "tvc-kdl-010": DOWNLOADS / "datasets" / "compounds-tvc-kdl-010-2026-05-30.csv",
}
DEMO_N_GENES = 3000  # synthetic scored-gene subset size for --demo


# --------------------------------------------------------------------------- #
# Stage 1-6: fetch + persist training artifacts
# --------------------------------------------------------------------------- #
def authenticate() -> None:
    """Validate / persist the TVC token so vcpi-client can fetch data.

    Resolution order (mirrors the Colab notebook's STEP 2b):
    1. ``TVC_TOKEN`` env var, if set.
    2. Interactive ``getpass`` prompt, if running on a TTY.
    3. Otherwise rely on a token cached in the system keychain from a
       prior ``vcpi login``; ``load_experiment`` raises clearly if absent.
    """
    import getpass

    import vcpi

    token = os.environ.get("TVC_TOKEN")
    if not token and sys.stdin.isatty():
        # Paste your API key from thevirtualcell.com/dashboard -> Settings.
        token = getpass.getpass("Paste your TVC API token (input hidden): ").strip()

    if token:
        os.environ["TVC_TOKEN"] = token
        vcpi.login(token)  # validates + persists to the system keychain
        print("[auth] logged in with TVC_TOKEN")
    else:
        print("[auth] no TVC_TOKEN provided; relying on cached keychain token")


def fetch_training_data(job: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Replicate the official recipe for a single VCPI release.

    Returns ``(counts, metadata, chemistry)`` as pandas frames, filtered
    to the contest condition (THP-1 / 24h / 10000 nM + DMSO controls).
    """
    import polars as pl
    import vcpi

    print(f"[fetch] vcpi.load_experiment({job!r}) ...")
    exp = vcpi.load_experiment(job)

    meta = exp["metadata"].filter(
        (pl.col("cell_line") == "THP-1")
        & (pl.col("timepoint") == "24h")
        & (
            # vcpi stores 10 uM as 10000 nM
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

    counts = data.to_pandas()
    metadata = meta.to_pandas()
    chemistry = (
        exp["chemistry"]
        .to_pandas()
        .drop_duplicates(subset=["compound"])
        .reset_index(drop=True)
    )
    print(
        f"[fetch] {job}: counts {counts.shape}, metadata {metadata.shape}, "
        f"chemistry {chemistry.shape}"
    )
    return counts, metadata, chemistry


def build_artifacts(job: str) -> None:
    """Stages 2-6: fetch, save raw artifacts, and build expression."""
    from vcpi_prediction_contest import load_gene_filter

    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    authenticate()
    counts, metadata, chemistry = fetch_training_data(job)

    # The raw counts matrix for qnu-012 is large enough to exhaust local
    # disk on small laptops. Keep only the compact artifacts needed for
    # training and submission.
    metadata.to_parquet(METADATA_PATH)
    chemistry.to_parquet(CHEMISTRY_PATH)
    print(f"[save] {METADATA_PATH.name}, {CHEMISTRY_PATH.name}")

    # Convert raw UMI counts -> compact compound x scored-gene matrix in
    # log2(CPM + 1). Filtering to the official scored genes before the
    # conversion avoids writing an enormous long table.
    expression_wide = counts_to_expression_wide(
        counts,
        metadata,
        gene_filter=load_gene_filter(),
        compound_col=COMPOUND_KEY,
    )
    expression_wide.to_parquet(EXPRESSION_WIDE_PATH)
    print(
        f"[save] {EXPRESSION_WIDE_PATH.name}: "
        f"{expression_wide.shape[0]:,} compounds x {expression_wide.shape[1]:,} genes"
    )


def build_demo_artifacts(job: str) -> None:
    """Offline stand-in for stages 2-6 (no network / no token).

    Reads the local chemistry CSV for ``job`` and fabricates a
    *fingerprint-driven* synthetic expression target: a random linear
    projection of each compound's Morgan fingerprint plus noise. Because
    the signal is a real function of structure, Ridge genuinely beats the
    per-gene-mean baseline and the served API returns molecule-specific
    predictions — exercising the whole product end-to-end. Values are NOT
    biologically meaningful; rerun without --demo (with a token) for real
    predictions.
    """
    from vcpi_prediction_contest import load_gene_filter

    csv = LOCAL_COMPOUNDS.get(job)
    if csv is None or not csv.exists():
        msg = f"--demo needs a local chemistry CSV for {job}; none found at {csv}"
        raise FileNotFoundError(msg)

    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    print(f"[demo] fabricating training target from {csv.name} (SYNTHETIC)")
    chem = pd.read_csv(csv, dtype={COMPOUND_KEY: str})
    chem = chem.dropna(subset=["smiles"]).drop_duplicates(COMPOUND_KEY).reset_index(drop=True)
    chem.to_parquet(CHEMISTRY_PATH)

    rng = np.random.default_rng(RANDOM_STATE)
    genes = sorted(rng.choice(load_gene_filter(), size=DEMO_N_GENES, replace=False).tolist())

    x, valid = build_fingerprints(chem["smiles"].tolist())
    chem, x = chem.loc[valid].reset_index(drop=True), x[valid]
    proj = rng.normal(0.0, 1.0, size=(N_BITS, len(genes))).astype(np.float32) * 0.04
    signal = x @ proj  # n_compounds x n_genes, structure-dependent
    base = rng.uniform(1.0, 4.0, size=len(genes)).astype(np.float32)  # per-gene level
    noise = rng.normal(0.0, 0.3, size=signal.shape).astype(np.float32)
    expr_vals = np.clip(base + signal + noise, 0.0, None)

    comps = chem[COMPOUND_KEY].to_numpy()
    expr = pd.DataFrame(
        {
            "compound": np.repeat(comps, len(genes)),
            "gene_id": np.tile(genes, len(comps)),
            "expression": expr_vals.reshape(-1),
        }
    )
    expr.to_parquet(EXPRESSION_PATH)
    print(
        f"[demo] {EXPRESSION_PATH.name}: {len(expr):,} rows, "
        f"{len(comps)} compounds x {len(genes)} synthetic genes"
    )


def counts_to_expression_wide(
    counts: pd.DataFrame,
    metadata: pd.DataFrame,
    *,
    gene_filter: list[str],
    sample_col: str = "sequenced_id",
    compound_col: str = COMPOUND_KEY,
    gene_col: str = "gene_id",
) -> pd.DataFrame:
    """Convert raw counts to a compact compound x scored-gene matrix.

    This is the same normalization as ``counts_to_expression``:
    ``log2(CPM + 1)`` per sample, then mean across replicate samples
    for each compound. It returns a wide matrix to avoid materializing
    hundreds of millions of long-format rows.
    """
    if gene_col not in counts.columns:
        msg = f"counts is missing `{gene_col}`"
        raise ValueError(msg)
    for col in (sample_col, compound_col):
        if col not in metadata.columns:
            msg = f"metadata is missing `{col}`"
            raise ValueError(msg)

    counts = counts.copy()
    counts[gene_col] = counts[gene_col].astype(str)
    keep_genes = set(map(str, gene_filter))
    counts = counts[counts[gene_col].isin(keep_genes)].set_index(gene_col)
    counts.columns = counts.columns.astype(str)

    metadata = metadata.copy()
    metadata[sample_col] = metadata[sample_col].astype(str)
    metadata[compound_col] = metadata[compound_col].astype(str)
    sample_to_compound = pd.Series(
        metadata[compound_col].to_numpy(),
        index=metadata[sample_col].to_numpy(),
        name=compound_col,
    )

    shared_samples = [s for s in counts.columns if s in sample_to_compound.index]
    if not shared_samples:
        msg = "No overlapping samples between counts columns and metadata"
        raise ValueError(msg)
    dropped = counts.shape[1] - len(shared_samples)
    if dropped:
        print(f"[expr] dropping {dropped:,} count columns absent from metadata")

    counts = counts[shared_samples]
    sample_to_compound = sample_to_compound.loc[shared_samples]

    library_size = counts.sum(axis=0).replace(0, np.nan)
    log_cpm = np.log2(counts.div(library_size, axis=1) * 1e6 + 1.0)
    log_cpm = log_cpm.astype("float32")

    sample_genes = log_cpm.T
    sample_genes.index = pd.Index(sample_to_compound.to_numpy(), name="compound")
    wide = sample_genes.groupby(level="compound", sort=True).mean().astype("float32")
    wide = wide.reindex(columns=sorted(keep_genes & set(wide.columns)))
    return wide


# --------------------------------------------------------------------------- #
# Stage 8: Morgan fingerprints
# --------------------------------------------------------------------------- #
def build_fingerprints(smiles: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """SMILES list -> (N x N_BITS float32 matrix, bool validity mask).

    Invalid / unparseable SMILES yield an all-zero row and ``False`` in
    the mask so callers can decide whether to drop or fall back.
    """
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
    print(f"[fp] featurized {int(valid.sum())}/{len(smiles)} SMILES ({N_BITS} bits)")
    return x, valid


# --------------------------------------------------------------------------- #
# Stage 7 + 9-12: features, model, predict, validate
# --------------------------------------------------------------------------- #
def load_training_matrix() -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """Build the (compound x scored-gene) expression matrix + SMILES map.

    Returns
    -------
    wide
        Expression matrix indexed by ``user_compound_id``; columns are the
        scored genes present in training. NaNs (compound missing a gene)
        are filled with the per-gene training mean.
    smiles
        ``user_compound_id`` -> SMILES Series aligned to ``wide.index``.
    train_genes
        Ordered list of the scored-gene columns of ``wide``.
    """
    from vcpi_prediction_contest import load_gene_filter

    scored = set(load_gene_filter())
    if EXPRESSION_WIDE_PATH.exists():
        wide = pd.read_parquet(EXPRESSION_WIDE_PATH)
        wide.index = wide.index.astype(str)
        wide.columns = wide.columns.astype(str)
    else:
        expr = pd.read_parquet(EXPRESSION_PATH)  # compound, gene_id, expression
        expr["compound"] = expr["compound"].astype(str)

        expr = expr[expr["gene_id"].isin(scored)]

        wide = expr.pivot(index="compound", columns="gene_id", values="expression")
        wide = wide.fillna(wide.mean(axis=0))  # per-gene mean fill for gaps

    chem = pd.read_parquet(CHEMISTRY_PATH)
    chem[COMPOUND_KEY] = chem[COMPOUND_KEY].astype(str)

    smap = (
        chem.dropna(subset=["smiles"])
        .drop_duplicates(COMPOUND_KEY)
        .set_index(COMPOUND_KEY)["smiles"]
    )
    common = [c for c in wide.index if c in smap.index]
    dropped = len(wide.index) - len(common)
    if dropped:
        print(f"[data] dropped {dropped} compounds with no chemistry SMILES (e.g. DMSO)")
    wide = wide.loc[common]
    smiles = smap.loc[common]
    train_genes = list(wide.columns)
    print(
        f"[data] training matrix: {wide.shape[0]} compounds x {wide.shape[1]} "
        f"scored genes present (of {len(scored)} scored)"
    )
    return wide, smiles, train_genes


def fit_ridge(x: np.ndarray, y: np.ndarray, alpha: float = RIDGE_ALPHA):
    from sklearn.linear_model import Ridge

    model = Ridge(alpha=alpha)
    model.fit(x, y)
    return model


def fit_mlp(
    x: np.ndarray,
    y: np.ndarray,
    *,
    epochs: int = MLP_EPOCHS,
    batch_size: int = MLP_BATCH_SIZE,
    lr: float = MLP_LR,
    scale: bool = True,
    verbose: bool = True,
):
    from models import TorchMLPRegressor

    reg = TorchMLPRegressor(
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        scale=scale,
        seed=RANDOM_STATE,
        verbose=verbose,
    )
    reg.fit(x, y)
    return reg


def build_model(kind: str, x: np.ndarray, y: np.ndarray, *, verbose: bool = True, **mlp_kw):
    """Fit and return the requested model (sklearn-style fit/predict)."""
    if kind == "ridge":
        return fit_ridge(x, y)
    if kind == "mlp":
        return fit_mlp(x, y, verbose=verbose, **mlp_kw)
    msg = f"unknown model kind: {kind!r}"
    raise ValueError(msg)


def scaffold_split(
    smiles: list[str], n_val: int, seed: int = RANDOM_STATE
) -> tuple[np.ndarray, np.ndarray]:
    """Bemis-Murcko scaffold split -> (train_idx, val_idx).

    Groups compounds by Murcko scaffold, then places whole scaffold
    groups into train (largest first) until the train quota is met; the
    remaining (smaller, distinct) scaffolds form the validation set. This
    holds out novel chemotypes, a fairer test of generalization than a
    random split (which leaks scaffolds across the split).
    """
    from rdkit import Chem
    from rdkit.Chem.Scaffolds import MurckoScaffold

    groups: dict[str, list[int]] = {}
    for i, smi in enumerate(smiles):
        mol = Chem.MolFromSmiles(smi) if isinstance(smi, str) else None
        try:
            scaf = MurckoScaffold.MurckoScaffoldSmiles(mol=mol) if mol else f"_none_{i}"
        except Exception:  # noqa: BLE001 - degenerate molecules
            scaf = f"_err_{i}"
        groups.setdefault(scaf or f"_empty_{i}", []).append(i)

    rng = np.random.default_rng(seed)
    ordered = sorted(groups.values(), key=len, reverse=True)
    n_train_target = len(smiles) - n_val
    train_idx: list[int] = []
    val_idx: list[int] = []
    for grp in ordered:
        if len(train_idx) + len(grp) <= n_train_target:
            train_idx.extend(grp)
        else:
            val_idx.extend(grp)
    # Guard against degenerate cases (e.g. one giant scaffold group).
    if not val_idx or not train_idx:
        perm = rng.permutation(len(smiles))
        val_idx, train_idx = perm[:n_val].tolist(), perm[n_val:].tolist()
        print("[val] scaffold split degenerate; fell back to random split")
    n_scaf = len({s for s, g in groups.items() for _ in g})
    print(f"[val] scaffold split: {n_scaf} scaffolds -> {len(train_idx)} train / {len(val_idx)} val")
    return np.array(train_idx), np.array(val_idx)


def validate(
    wide: pd.DataFrame,
    x: np.ndarray,
    smiles: pd.Series,
    train_genes: list[str],
    *,
    model_kind: str = "ridge",
    mlp_kw: dict | None = None,
) -> dict[str, float]:
    """Held-out-compound validation in leaderboard wMSE.

    Always scores the per-gene-mean baseline and Ridge; additionally
    scores the MLP when ``model_kind == "mlp"``. Uses a Bemis-Murcko
    scaffold split so validation compounds carry scaffolds unseen in
    training. Returns ``{name: wmse_mean}``.
    """
    from vcpi_prediction_contest import aggregate_leaderboards, score_compounds

    n = wide.shape[0]
    n_val = min(N_VAL_COMPOUNDS, max(1, n // 5))
    tr_idx, val_idx = scaffold_split(smiles.tolist(), n_val)

    y = wide.to_numpy(dtype=np.float32)
    val_compounds = wide.index[val_idx]

    def to_long(matrix: np.ndarray, value_col: str) -> pd.DataFrame:
        frame = pd.DataFrame(matrix, index=val_compounds, columns=train_genes)
        return frame.reset_index(names="compound").melt(
            id_vars="compound", var_name="gene_id", value_name=value_col
        )

    truth = to_long(y[val_idx], "expression")

    # Candidate predictions on the held-out scaffolds.
    preds: dict[str, np.ndarray] = {
        "per-gene-mean baseline": np.tile(y[tr_idx].mean(axis=0), (len(val_idx), 1)),
        "Ridge (Morgan fp)": np.clip(
            fit_ridge(x[tr_idx], y[tr_idx]).predict(x[val_idx]), 0.0, None
        ),
    }
    if model_kind == "mlp":
        mlp = fit_mlp(x[tr_idx], y[tr_idx], **(mlp_kw or {}))
        preds["MLP (Morgan fp)"] = np.clip(mlp.predict(x[val_idx]), 0.0, None)

    scores = {
        name: aggregate_leaderboards(
            score_compounds(truth, to_long(p, "predicted_expression"), gene_filter=train_genes)
        )["wmse_mean"]
        for name, p in preds.items()
    }

    baseline = scores["per-gene-mean baseline"]
    print("\n==================== VALIDATION (wMSE, lower is better) ===========")
    for name in sorted(scores, key=scores.get):
        tag = "  <-- best" if scores[name] == min(scores.values()) else ""
        vs = "" if name == "per-gene-mean baseline" else f"  (delta vs baseline {baseline - scores[name]:+.4f})"
        print(f"  {name:<24}: {scores[name]:.4f}{vs}{tag}")
    print("===================================================================\n")
    return scores


def save_model(
    model,
    wide: pd.DataFrame,
    x: np.ndarray,
    train_genes: list[str],
    scored_genes: list[str],
    model_kind: str,
) -> None:
    """Persist the trained model + everything the API needs to serve.

    Stores the fitted model (Ridge or the MLP regressor — both expose
    ``.predict``), the gene ordering, per-gene training mean (reference
    for "top changed genes"), the global mean fallback, the feature
    config, and the training fingerprints + compound ids (packed to bits)
    for nearest-training-compound interpretability.
    """
    import joblib

    y = wide.to_numpy(dtype=np.float32)
    artifact = {
        "model": model,
        "model_kind": model_kind,
        "train_genes": list(train_genes),
        "scored_genes": list(scored_genes),
        "gene_mean": y.mean(axis=0).astype(np.float32),  # aligned to train_genes
        "global_mean": float(y.mean()),
        "radius": RADIUS,
        "n_bits": N_BITS,
        "train_compounds": wide.index.to_numpy().astype(str),
        "train_fps_packed": np.packbits(x.astype(bool), axis=1),  # compact
    }
    joblib.dump(artifact, MODEL_PATH)
    size_mb = MODEL_PATH.stat().st_size / 1e6
    print(f"[save] {MODEL_PATH.name}: {model_kind} + {len(train_genes)} genes ({size_mb:.1f} MB)")


def predict_test(
    wide: pd.DataFrame,
    x: np.ndarray,
    train_genes: list[str],
    *,
    model_kind: str = "ridge",
    mlp_kw: dict | None = None,
) -> None:
    """Train on all data, predict every test compound x every scored gene."""
    from vcpi_prediction_contest import load_gene_filter, load_test_compounds

    scored_genes = load_gene_filter()  # full official scored set (sorted)
    test_df = load_test_compounds()
    test_df["compound"] = test_df["compound"].astype(str)

    y = wide.to_numpy(dtype=np.float32)
    print(f"[train] fitting final {model_kind} on all {wide.shape[0]} compounds")
    model = build_model(model_kind, x, y, **(mlp_kw or {}))
    save_model(model, wide, x, train_genes, scored_genes, model_kind)

    x_test, valid_test = build_fingerprints(test_df["smiles"].tolist())
    pred = np.clip(model.predict(x_test), 0.0, None)  # n_test x len(train_genes)

    # Fallback for unparseable test SMILES: per-gene training mean.
    gene_mean = y.mean(axis=0)
    if (~valid_test).any():
        pred[~valid_test] = gene_mean
        print(f"[predict] {int((~valid_test).sum())} test SMILES used mean fallback")

    # Assemble predictions over the FULL scored gene set. Scored genes
    # absent from training get the global training mean (non-negative).
    # Build with numpy to avoid fragmented column-by-column assignment.
    global_mean = float(y.mean())
    gene_pos = {g: i for i, g in enumerate(scored_genes)}
    train_col = {g: j for j, g in enumerate(train_genes)}
    shared = [g for g in train_genes if g in gene_pos]
    mat = np.full((len(test_df), len(scored_genes)), global_mean, dtype=np.float32)
    mat[:, [gene_pos[g] for g in shared]] = pred[:, [train_col[g] for g in shared]]

    full = pd.DataFrame(mat, index=test_df["compound"], columns=scored_genes)
    long = (
        full.reset_index(names="compound")
        .melt(id_vars="compound", var_name="gene_id", value_name="predicted_expression")
    )
    long["predicted_expression"] = long["predicted_expression"].clip(lower=0.0).astype("float32")
    long = long[["compound", "gene_id", "predicted_expression"]]

    long.to_parquet(PREDICTIONS_PATH, index=False)
    expected = len(test_df) * len(scored_genes)
    print(
        f"[save] {PREDICTIONS_PATH.name}: {len(long):,} rows "
        f"(expected {len(test_df)} x {len(scored_genes)} = {expected:,})"
    )
    assert len(long) == expected, "row count != test_compounds x scored_genes"
    assert list(long.columns) == ["compound", "gene_id", "predicted_expression"]


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", default=JOB, help=f"VCPI release (default {JOB})")
    parser.add_argument(
        "--skip-fetch",
        action="store_true",
        help="reuse existing parquet artifacts; skip network fetch (stages 2-6)",
    )
    parser.add_argument(
        "--no-validate", action="store_true", help="skip held-out validation"
    )
    parser.add_argument(
        "--model",
        choices=["ridge", "mlp"],
        default="ridge",
        help="final/served model (default ridge). validation always shows both.",
    )
    parser.add_argument("--ridge-alpha", type=float, default=RIDGE_ALPHA, help="Ridge regularization (default 10000)")
    parser.add_argument("--epochs", type=int, default=MLP_EPOCHS, help="MLP epochs")
    parser.add_argument("--batch-size", type=int, default=MLP_BATCH_SIZE, help="MLP batch size")
    parser.add_argument("--lr", type=float, default=MLP_LR, help="MLP learning rate")
    parser.add_argument(
        "--no-scale",
        action="store_true",
        help="MLP: skip input standardization (feed raw Morgan bits)",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="no token: fabricate fingerprint-driven SYNTHETIC labels so the "
        "full stack (train -> validate -> serve) is testable offline",
    )
    args = parser.parse_args()

    mlp_kw = {
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "scale": not args.no_scale,
    }

    if args.demo:
        build_demo_artifacts(args.job)
    elif not args.skip_fetch:
        build_artifacts(args.job)
    elif not EXPRESSION_PATH.exists():
        print(
            f"[error] --skip-fetch set but {EXPRESSION_PATH} is missing. "
            "Run once without --skip-fetch (needs TVC_TOKEN), or use --demo.",
            file=sys.stderr,
        )
        return 1

    wide, smiles, train_genes = load_training_matrix()
    x, valid = build_fingerprints(smiles.tolist())
    if (~valid).any():
        keep = np.where(valid)[0]
        wide = wide.iloc[keep]
        x = x[keep]
        smiles = smiles.iloc[keep]
        print(f"[data] dropped {int((~valid).sum())} compounds with invalid SMILES")

    if not args.no_validate:
        validate(wide, x, smiles, train_genes, model_kind=args.model, mlp_kw=mlp_kw)

    predict_test(wide, x, train_genes, model_kind=args.model, mlp_kw=mlp_kw)
    print("[done] pipeline complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
