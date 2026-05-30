"""Advanced model components for the VCPI virtual-cell predictor.

These are optional upgrade modules, not the current validated submission
path. They are designed to be imported into the existing training pipeline
when you want to experiment beyond Morgan-fingerprint Ridge/KNN.

Implemented components:

1. Bemis-Murcko scaffold split.
2. ChemBERTa SMILES dataset boilerplate.
3. PyTorch Geometric molecule graph conversion and GNN encoders.
4. Gaussian NLL output head for uncertainty-aware regression.
5. Pathway-to-gene hierarchical decoder.

The expected target remains absolute gene expression on the log2(CPM + 1)
scale, shaped as:

    y: (n_compounds, n_genes)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal

import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from torch import nn
from torch.utils.data import Dataset


# ---------------------------------------------------------------------------
# 1. Scaffold split
# ---------------------------------------------------------------------------


def murcko_scaffold(smiles: str) -> str:
    """Return the Bemis-Murcko scaffold for a SMILES string.

    Invalid or scaffoldless molecules receive a deterministic fallback
    scaffold string so they still belong to a split group.
    """
    mol = Chem.MolFromSmiles(smiles) if isinstance(smiles, str) else None
    if mol is None:
        return "__invalid__"
    try:
        scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol)
    except Exception:  # noqa: BLE001 - RDKit can throw on unusual molecules
        return "__scaffold_error__"
    return scaffold or "__empty_scaffold__"


def scaffold_split_indices(
    smiles: list[str],
    *,
    train_frac: float = 0.8,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Split molecule indices by scaffold group.

    No scaffold appears in both train and validation. Groups are shuffled
    and then greedily assigned until the target train fraction is reached.
    This is stricter and more realistic than a random compound split.
    """
    if not 0.0 < train_frac < 1.0:
        msg = "train_frac must be between 0 and 1"
        raise ValueError(msg)

    groups: dict[str, list[int]] = {}
    for i, smi in enumerate(smiles):
        groups.setdefault(murcko_scaffold(smi), []).append(i)

    rng = np.random.default_rng(seed)
    scaffold_groups = list(groups.values())
    rng.shuffle(scaffold_groups)
    scaffold_groups.sort(key=len, reverse=True)

    n_train_target = int(round(train_frac * len(smiles)))
    train_idx: list[int] = []
    val_idx: list[int] = []

    for group in scaffold_groups:
        if len(train_idx) + len(group) <= n_train_target:
            train_idx.extend(group)
        else:
            val_idx.extend(group)

    if not train_idx or not val_idx:
        # Degenerate case: one giant scaffold. Fall back to random split.
        perm = rng.permutation(len(smiles))
        n_train = max(1, min(len(smiles) - 1, n_train_target))
        train_idx = perm[:n_train].tolist()
        val_idx = perm[n_train:].tolist()

    return np.array(train_idx, dtype=int), np.array(val_idx, dtype=int)


# ---------------------------------------------------------------------------
# 2A. ChemBERTa dataset boilerplate
# ---------------------------------------------------------------------------


class ChemBERTaExpressionDataset(Dataset):
    """Dataset that tokenizes SMILES and returns expression targets.

    Example:

        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            "seyonec/ChemBERTa-zinc-base-v1"
        )
        ds = ChemBERTaExpressionDataset(smiles, y, tokenizer)
    """

    def __init__(
        self,
        smiles: list[str],
        y: np.ndarray | torch.Tensor,
        tokenizer,
        *,
        max_length: int = 160,
    ) -> None:
        self.smiles = list(smiles)
        self.y = torch.as_tensor(y, dtype=torch.float32)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.smiles)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        enc = self.tokenizer(
            self.smiles[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        item = {k: v.squeeze(0) for k, v in enc.items()}
        item["y"] = self.y[idx]
        return item


class ChemBERTaRegressor(nn.Module):
    """ChemBERTa encoder plus regression head.

    Set ``uncertainty=True`` to return ``(mean, variance)`` for Gaussian
    NLL training.
    """

    def __init__(
        self,
        transformer,
        *,
        n_genes: int,
        hidden_dim: int = 768,
        dropout: float = 0.1,
        uncertainty: bool = False,
    ) -> None:
        super().__init__()
        self.transformer = transformer
        self.uncertainty = uncertainty
        out_dim = n_genes * 2 if uncertainty else n_genes
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )
        self.var_activation = nn.Softplus()

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        out = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        # ChemBERTa/RoBERTa usually has CLS-like token at position 0.
        pooled = out.last_hidden_state[:, 0]
        raw = self.head(pooled)
        if not self.uncertainty:
            return raw
        mean, raw_var = raw.chunk(2, dim=-1)
        var = self.var_activation(raw_var) + 1e-6
        return mean, var


# ---------------------------------------------------------------------------
# 2B. PyTorch Geometric molecule graphs and GNN encoders
# ---------------------------------------------------------------------------


ATOM_TYPES = [
    "C",
    "N",
    "O",
    "S",
    "F",
    "Cl",
    "Br",
    "I",
    "P",
    "B",
    "Si",
    "other",
]


def atom_features(atom: Chem.Atom) -> list[float]:
    symbol = atom.GetSymbol()
    atom_type = [float(symbol == t) for t in ATOM_TYPES[:-1]]
    atom_type.append(float(symbol not in ATOM_TYPES[:-1]))
    return [
        *atom_type,
        float(atom.GetAtomicNum()) / 100.0,
        float(atom.GetTotalDegree()) / 8.0,
        float(atom.GetFormalCharge()),
        float(atom.GetTotalNumHs()) / 8.0,
        float(atom.GetIsAromatic()),
        float(atom.IsInRing()),
    ]


def smiles_to_pyg_data(smiles: str, y: np.ndarray | torch.Tensor | None = None):
    """Convert SMILES to a PyTorch Geometric Data object.

    Requires ``torch_geometric`` to be installed.
    """
    from torch_geometric.data import Data

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        msg = f"invalid SMILES: {smiles!r}"
        raise ValueError(msg)

    x = torch.tensor([atom_features(a) for a in mol.GetAtoms()], dtype=torch.float32)
    edges: list[tuple[int, int]] = []
    edge_attr: list[list[float]] = []

    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        bond_type = bond.GetBondType()
        feat = [
            float(bond_type == Chem.rdchem.BondType.SINGLE),
            float(bond_type == Chem.rdchem.BondType.DOUBLE),
            float(bond_type == Chem.rdchem.BondType.TRIPLE),
            float(bond_type == Chem.rdchem.BondType.AROMATIC),
            float(bond.IsInRing()),
        ]
        edges.extend([(i, j), (j, i)])
        edge_attr.extend([feat, feat])

    if edges:
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        edge_attr_t = torch.tensor(edge_attr, dtype=torch.float32)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr_t = torch.empty((0, 5), dtype=torch.float32)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr_t)
    if y is not None:
        data.y = torch.as_tensor(y, dtype=torch.float32)
    data.smiles = smiles
    return data


class MoleculeGraphDataset(Dataset):
    """SMILES + expression target dataset for PyG graph loaders."""

    def __init__(self, smiles: list[str], y: np.ndarray | torch.Tensor) -> None:
        self.smiles = list(smiles)
        self.y = torch.as_tensor(y, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.smiles)

    def __getitem__(self, idx: int):
        return smiles_to_pyg_data(self.smiles[idx], self.y[idx])


class GNNExpressionRegressor(nn.Module):
    """GCN/GAT molecular encoder with optional uncertainty output."""

    def __init__(
        self,
        *,
        node_dim: int,
        n_genes: int,
        hidden_dim: int = 256,
        num_layers: int = 4,
        conv: Literal["gcn", "gat"] = "gcn",
        dropout: float = 0.15,
        uncertainty: bool = False,
    ) -> None:
        super().__init__()
        from torch_geometric.nn import GATConv, GCNConv

        self.uncertainty = uncertainty
        self.dropout = nn.Dropout(dropout)
        conv_cls = GATConv if conv == "gat" else GCNConv

        layers = []
        in_dim = node_dim
        for _ in range(num_layers):
            if conv == "gat":
                layer = conv_cls(in_dim, hidden_dim // 4, heads=4, concat=True)
            else:
                layer = conv_cls(in_dim, hidden_dim)
            layers.append(layer)
            in_dim = hidden_dim
        self.layers = nn.ModuleList(layers)
        self.norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in layers])

        out_dim = n_genes * 2 if uncertainty else n_genes
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )
        self.var_activation = nn.Softplus()

    def forward(self, data):
        from torch_geometric.nn import global_mean_pool

        h = data.x
        for conv, norm in zip(self.layers, self.norms, strict=True):
            h = conv(h, data.edge_index)
            h = norm(h)
            h = torch.relu(h)
            h = self.dropout(h)
        pooled = global_mean_pool(h, data.batch)
        raw = self.head(pooled)
        if not self.uncertainty:
            return raw
        mean, raw_var = raw.chunk(2, dim=-1)
        var = self.var_activation(raw_var) + 1e-6
        return mean, var


# ---------------------------------------------------------------------------
# 3. Gaussian NLL training helpers
# ---------------------------------------------------------------------------


@dataclass
class TrainBatchOutput:
    loss: torch.Tensor
    mean: torch.Tensor
    variance: torch.Tensor | None = None


def gaussian_nll_step(model: nn.Module, batch, *, graph_batch: bool = False) -> TrainBatchOutput:
    """Compute Gaussian NLL loss for either transformer or graph batches."""
    loss_fn = nn.GaussianNLLLoss(full=False, reduction="mean")

    if graph_batch:
        y = batch.y
        mean, var = model(batch)
    else:
        y = batch["y"]
        mean, var = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
        )
    loss = loss_fn(mean, y, var)
    return TrainBatchOutput(loss=loss, mean=mean, variance=var)


def mse_step(model: nn.Module, batch, *, graph_batch: bool = False) -> TrainBatchOutput:
    """Plain MSE version for direct comparison with Gaussian NLL."""
    loss_fn = nn.MSELoss()
    if graph_batch:
        y = batch.y
        mean = model(batch)
    else:
        y = batch["y"]
        mean = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
        )
    return TrainBatchOutput(loss=loss_fn(mean, y), mean=mean, variance=None)


def train_epoch(
    model: nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    *,
    device: torch.device,
    use_gaussian_nll: bool = False,
    graph_batch: bool = False,
) -> float:
    """Generic one-epoch trainer for the modules in this file."""
    model.train()
    losses = []
    for batch in loader:
        batch = batch.to(device) if graph_batch else {k: v.to(device) for k, v in batch.items()}
        optimizer.zero_grad()
        out = (
            gaussian_nll_step(model, batch, graph_batch=graph_batch)
            if use_gaussian_nll
            else mse_step(model, batch, graph_batch=graph_batch)
        )
        out.loss.backward()
        optimizer.step()
        losses.append(float(out.loss.detach().cpu()))
    return float(np.mean(losses))


# ---------------------------------------------------------------------------
# 4. Biological pathway hierarchical decoder
# ---------------------------------------------------------------------------


class PathwayGeneDecoder(nn.Module):
    """Decode molecular embeddings through pathway activations to genes.

    Concept:

        molecule embedding -> pathway scores -> gene expression

    If a pathway-to-gene matrix is provided, it is used as an initialized
    biological prior. Set ``train_mapping=False`` to keep the mapping fixed.
    """

    def __init__(
        self,
        *,
        embedding_dim: int,
        n_pathways: int,
        n_genes: int,
        pathway_gene_matrix: torch.Tensor | np.ndarray | None = None,
        train_mapping: bool = True,
        uncertainty: bool = False,
    ) -> None:
        super().__init__()
        self.uncertainty = uncertainty
        self.pathway_head = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.LayerNorm(embedding_dim),
            nn.Linear(embedding_dim, n_pathways),
        )

        if pathway_gene_matrix is None:
            init = torch.empty(n_pathways, n_genes)
            nn.init.xavier_uniform_(init)
        else:
            init = torch.as_tensor(pathway_gene_matrix, dtype=torch.float32)
            if init.shape != (n_pathways, n_genes):
                msg = (
                    f"pathway_gene_matrix must have shape {(n_pathways, n_genes)}, "
                    f"got {tuple(init.shape)}"
                )
                raise ValueError(msg)

        self.pathway_to_gene = nn.Parameter(init, requires_grad=train_mapping)
        self.gene_bias = nn.Parameter(torch.zeros(n_genes))

        if uncertainty:
            self.log_var_head = nn.Sequential(
                nn.Linear(n_pathways, n_pathways),
                nn.ReLU(),
                nn.Linear(n_pathways, n_genes),
            )
            self.var_activation = nn.Softplus()

    def forward(self, embedding: torch.Tensor):
        pathway_scores = self.pathway_head(embedding)
        mean = pathway_scores @ self.pathway_to_gene + self.gene_bias
        if not self.uncertainty:
            return mean, pathway_scores
        raw_var = self.log_var_head(pathway_scores)
        var = self.var_activation(raw_var) + 1e-6
        return mean, var, pathway_scores


class GNNPathwayVirtualCell(nn.Module):
    """End-to-end graph model with pathway decoder."""

    def __init__(
        self,
        *,
        node_dim: int,
        n_genes: int,
        n_pathways: int = 256,
        hidden_dim: int = 256,
        pathway_gene_matrix: torch.Tensor | np.ndarray | None = None,
        uncertainty: bool = False,
    ) -> None:
        super().__init__()
        from torch_geometric.nn import GCNConv

        self.uncertainty = uncertainty
        self.conv1 = GCNConv(node_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.decoder = PathwayGeneDecoder(
            embedding_dim=hidden_dim,
            n_pathways=n_pathways,
            n_genes=n_genes,
            pathway_gene_matrix=pathway_gene_matrix,
            uncertainty=uncertainty,
        )

    def forward(self, data):
        from torch_geometric.nn import global_mean_pool

        h = torch.relu(self.conv1(data.x, data.edge_index))
        h = torch.relu(self.conv2(h, data.edge_index))
        h = self.norm(h)
        embedding = global_mean_pool(h, data.batch)
        return self.decoder(embedding)


# ---------------------------------------------------------------------------
# Pathway matrix helper
# ---------------------------------------------------------------------------


def build_pathway_gene_matrix(
    gene_ids: list[str],
    pathway_to_genes: dict[str, Iterable[str]],
    *,
    normalize: bool = True,
) -> tuple[np.ndarray, list[str]]:
    """Create a pathway x gene binary membership matrix.

    ``pathway_to_genes`` can come from MSigDB, Reactome, KEGG, GO, or a
    custom dictionary. Gene identifiers must match the model's gene_id
    order.
    """
    gene_pos = {g: i for i, g in enumerate(gene_ids)}
    pathway_names = list(pathway_to_genes)
    mat = np.zeros((len(pathway_names), len(gene_ids)), dtype=np.float32)

    for p_i, pathway in enumerate(pathway_names):
        for gene in pathway_to_genes[pathway]:
            if gene in gene_pos:
                mat[p_i, gene_pos[gene]] = 1.0

    if normalize:
        row_sum = mat.sum(axis=1, keepdims=True)
        mat = np.divide(mat, row_sum, out=mat, where=row_sum > 0)

    return mat, pathway_names

