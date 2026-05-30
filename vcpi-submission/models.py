"""Model definitions for the VCPI pipeline.

Holds the PyTorch MLP (adapted from the Colab STEP 7 code) wrapped in a
scikit-learn-style ``fit`` / ``predict`` regressor so it slots into the
same training / validation / prediction / serving path as the Ridge
baseline in ``train_pipeline.py``.

Design notes vs. the original Colab cell
----------------------------------------
- **Same target space as the pipeline**: trains on per-(compound, gene)
  ``log2(CPM + 1)`` mean expression (non-negative), not log2 fold-change,
  so predictions can be written straight to the submission. The caller
  clips at 0.
- **Honest early stopping**: carves an *internal* validation split out of
  the training rows for checkpoint selection, so it never peeks at the
  scaffold-held-out compounds the pipeline scores on.
- **Picklable**: after training the network is moved to CPU and the whole
  regressor (net + scaler stats + dims) is joblib-dumped, so ``app.py``
  serves it via the same ``model.predict(...)`` call as Ridge.
- Optional input standardization (``scale=True`` mirrors the Colab
  ``StandardScaler``; binary Morgan bits often train fine without it, so
  it can be disabled).
"""

from __future__ import annotations

import numpy as np

# torch is imported lazily inside methods so that loading a *Ridge*
# artifact (which never references this module) doesn't drag torch in.


def _select_device() -> str:
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _build_net(n_in: int, n_out: int):
    """The Colab PerturbMLP architecture: 1024 -> 1024 -> 512 -> n_out."""
    import torch.nn as nn

    return nn.Sequential(
        nn.Linear(n_in, 1024),
        nn.LayerNorm(1024),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(1024, 1024),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(1024, 512),
        nn.ReLU(),
        nn.Linear(512, n_out),
    )


class TorchMLPRegressor:
    """Multi-output MLP regressor with a sklearn-like API.

    Parameters
    ----------
    epochs, batch_size, lr, weight_decay
        Optimisation hyperparameters (Adam + ReduceLROnPlateau), matching
        the Colab defaults.
    val_frac
        Fraction of the *training* rows held out internally for best-
        checkpoint selection / LR scheduling.
    scale
        If True, standardize inputs (Colab StandardScaler behaviour).
    seed
        Reproducibility seed for torch + the internal split.
    """

    def __init__(
        self,
        *,
        epochs: int = 50,
        batch_size: int = 256,
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
        val_frac: float = 0.1,
        scale: bool = True,
        seed: int = 42,
        verbose: bool = True,
    ) -> None:
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.val_frac = val_frac
        self.scale = scale
        self.seed = seed
        self.verbose = verbose
        # Learned state (populated by fit; all CPU / numpy so it pickles).
        self.net = None
        self.n_in_: int | None = None
        self.n_out_: int | None = None
        self.mean_: np.ndarray | None = None
        self.scale_: np.ndarray | None = None
        self.best_val_mse_: float | None = None

    # ------------------------------------------------------------------ #
    def _standardize(self, x: np.ndarray) -> np.ndarray:
        if not self.scale:
            return x.astype(np.float32)
        return ((x - self.mean_) / self.scale_).astype(np.float32)

    def fit(self, x: np.ndarray, y: np.ndarray) -> TorchMLPRegressor:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset

        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)
        device = _select_device()

        x = np.asarray(x, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
        self.n_in_, self.n_out_ = x.shape[1], y.shape[1]

        # Standardization stats (guard zero-variance / all-zero bits).
        self.mean_ = x.mean(axis=0)
        std = x.std(axis=0)
        self.scale_ = np.where(std < 1e-8, 1.0, std).astype(np.float32)
        xs = self._standardize(x)

        # Internal train/val split for checkpoint selection.
        n = xs.shape[0]
        perm = rng.permutation(n)
        n_val = max(1, int(round(self.val_frac * n)))
        vi, ti = perm[:n_val], perm[n_val:]

        net = _build_net(self.n_in_, self.n_out_).to(device)
        opt = torch.optim.Adam(net.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=3, factor=0.5)
        loss_fn = nn.MSELoss()

        x_tr = torch.tensor(xs[ti], device=device)
        y_tr = torch.tensor(y[ti], device=device)
        x_vl = torch.tensor(xs[vi], device=device)
        y_vl = torch.tensor(y[vi], device=device)
        loader = DataLoader(
            TensorDataset(x_tr, y_tr), batch_size=self.batch_size, shuffle=True
        )

        if self.verbose:
            print(
                f"[mlp] {self.n_in_} -> 1024 -> 1024 -> 512 -> {self.n_out_} "
                f"on {device} | {len(ti)} train / {len(vi)} internal-val | "
                f"{self.epochs} epochs | scale={self.scale}"
            )

        best_val = float("inf")
        best_state = None
        for epoch in range(self.epochs):
            net.train()
            for xb, yb in loader:
                opt.zero_grad()
                loss_fn(net(xb), yb).backward()
                opt.step()
            net.eval()
            with torch.no_grad():
                val_mse = loss_fn(net(x_vl), y_vl).item()
            sched.step(val_mse)
            if val_mse < best_val:
                best_val = val_mse
                best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
            if self.verbose and (epoch + 1) % 10 == 0:
                print(f"[mlp]   epoch {epoch + 1:3d}/{self.epochs} | internal val MSE {val_mse:.4f}")

        net.load_state_dict(best_state)
        net.to("cpu").eval()  # CPU for portable pickling / serving
        self.net = net
        self.best_val_mse_ = best_val
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        import torch

        if self.net is None:
            msg = "TorchMLPRegressor is not fitted"
            raise RuntimeError(msg)
        xs = self._standardize(np.asarray(x, dtype=np.float32))
        self.net.eval()
        out = np.empty((xs.shape[0], self.n_out_), dtype=np.float32)
        bs = 4096
        with torch.no_grad():
            for i in range(0, xs.shape[0], bs):
                xb = torch.from_numpy(xs[i : i + bs])
                out[i : i + bs] = self.net(xb).numpy()
        return out
