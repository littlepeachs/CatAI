#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from rdkit import Chem, RDLogger
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from full_retrain.data import ROOT, load_task
from full_retrain.metrics import best_threshold_accuracy, classification_metrics


RDLogger.DisableLog("rdApp.*")


TOKEN_TYPE_IDS = {
    "global": 0,
    "reactant_site": 1,
    "reactant_neighbor1": 2,
    "reactant_neighbor2": 3,
    "reactant_atom": 4,
    "solvent_atom": 5,
    "ligand_donor": 6,
    "ligand_chiral": 7,
    "ligand_atom": 8,
    "metal_atom": 9,
    "missing": 10,
    "imine_n": 11,
    "imine_atom": 12,
    "thiol_s": 13,
    "thiol_atom": 14,
}

COORDINATION_METAL_ATOMIC_NUMBERS = {
    *range(21, 31),
    *range(39, 49),
    *range(57, 81),
}


@dataclass
class AtomTokenStore:
    node_features: np.ndarray
    atom_mask: np.ndarray
    conformer_mask: np.ndarray
    atomic_numbers: np.ndarray
    atom_descriptors: np.ndarray
    positions: np.ndarray
    mapping: pd.DataFrame
    metadata: pd.DataFrame


@dataclass
class TensorBundle:
    x: torch.Tensor
    numeric: torch.Tensor
    token_mask: torch.Tensor
    role_ids: torch.Tensor
    token_type_ids: torch.Tensor
    atomic_numbers: torch.Tensor
    y: torch.Tensor
    row_index: np.ndarray


def load_atom_token_store(atom_token_dir: Path) -> AtomTokenStore:
    data = np.load(atom_token_dir / "component_atom_tokens.npz")
    return AtomTokenStore(
        node_features=data["node_features"],
        atom_mask=data["atom_mask"].astype(bool),
        conformer_mask=data["conformer_mask"].astype(bool),
        atomic_numbers=data["atomic_numbers"].astype(np.int64),
        atom_descriptors=data["atom_descriptors"].astype(np.float32),
        positions=data["positions"].astype(np.float32),
        mapping=pd.read_csv(atom_token_dir / "reaction_component_map.csv"),
        metadata=pd.read_csv(atom_token_dir / "component_metadata.csv"),
    )


def split_frames(bundle) -> dict[str, pd.DataFrame]:
    frames = {}
    for part, frame in [("train", bundle.train), ("val", bundle.val), ("test", bundle.test)]:
        if frame is None:
            continue
        tmp = frame.copy()
        tmp["_split_row_pos"] = np.arange(len(tmp), dtype=int)
        tmp["_source_index"] = tmp.index.to_numpy(dtype=int)
        frames[part] = tmp
    return frames


def parse_site_atom_maps(value: object) -> list[int]:
    if pd.isna(value):
        return []
    return [int(token) for token in re.split(r"[^0-9]+", str(value)) if token]


def clear_atom_maps(mol: Chem.Mol) -> Chem.Mol:
    out = Chem.Mol(mol)
    for atom in out.GetAtoms():
        atom.SetAtomMapNum(0)
    return out


def site_indices_for_reactant(row: pd.Series, canonical_smiles: str) -> set[int]:
    site_maps = parse_site_atom_maps(row.get("reaction_site_pairs", ""))
    if not site_maps:
        return set()

    atom_map = str(row.get("atom_map", ""))
    lhs = atom_map.split(">>", 1)[0] if ">>" in atom_map else atom_map
    mapped = Chem.MolFromSmiles(lhs)
    target = Chem.MolFromSmiles(canonical_smiles)
    if mapped is None or target is None:
        return set()

    map_to_query_idx = {
        int(atom.GetAtomMapNum()): int(atom.GetIdx())
        for atom in mapped.GetAtoms()
        if int(atom.GetAtomMapNum()) > 0
    }
    query_site_indices = [map_to_query_idx[num] for num in site_maps if num in map_to_query_idx]
    if not query_site_indices:
        return set()

    query = clear_atom_maps(mapped)
    match = target.GetSubstructMatch(query)
    if match and len(match) == query.GetNumAtoms():
        return {int(match[idx]) for idx in query_site_indices if idx < len(match)}

    if target.GetNumAtoms() == query.GetNumAtoms():
        return {idx for idx in query_site_indices if idx < target.GetNumAtoms()}
    return set()


def neighbor_shells(mol: Chem.Mol | None, seeds: set[int], max_radius: int = 2) -> dict[int, int]:
    if mol is None or not seeds:
        return {}
    distances = {idx: 0 for idx in seeds if idx < mol.GetNumAtoms()}
    frontier = set(distances)
    for radius in range(1, max_radius + 1):
        next_frontier = set()
        for idx in frontier:
            atom = mol.GetAtomWithIdx(idx)
            for neighbor in atom.GetNeighbors():
                n_idx = int(neighbor.GetIdx())
                if n_idx not in distances:
                    distances[n_idx] = radius
                    next_frontier.add(n_idx)
        frontier = next_frontier
        if not frontier:
            break
    return distances


def pooled_conformer_features(store: AtomTokenStore, component_id: int) -> tuple[np.ndarray, np.ndarray, bool]:
    atom_mask = store.atom_mask[component_id]
    conf_mask = store.conformer_mask[component_id]
    if not conf_mask.any() or not atom_mask.any():
        n_atoms = int(atom_mask.sum())
        return (
            np.zeros((n_atoms, store.node_features.shape[-1]), dtype=np.float32),
            np.zeros((n_atoms, 3), dtype=np.float32),
            False,
        )
    weights = conf_mask.astype(np.float32)[:, None, None]
    denom = weights.sum(axis=0).clip(min=1.0)
    features = (store.node_features[component_id].astype(np.float32) * weights).sum(axis=0) / denom
    xyz = (store.positions[component_id] * weights).sum(axis=0) / denom
    return features[atom_mask], xyz[atom_mask], True


def normalized_positions(xyz: np.ndarray) -> np.ndarray:
    if len(xyz) == 0:
        return xyz.astype(np.float32)
    centered = xyz - xyz.mean(axis=0, keepdims=True)
    scale = float(np.sqrt((centered**2).sum(axis=1).mean()))
    if scale < 1e-6:
        scale = 1.0
    return (centered / scale).astype(np.float32)


def token_type_for_atom(role: str, atom_idx: int, atomic_num: int, descriptors: np.ndarray, shells: dict[int, int]) -> int:
    if role == "reactant":
        radius = shells.get(atom_idx)
        if radius == 0:
            return TOKEN_TYPE_IDS["reactant_site"]
        if radius == 1:
            return TOKEN_TYPE_IDS["reactant_neighbor1"]
        if radius == 2:
            return TOKEN_TYPE_IDS["reactant_neighbor2"]
        return TOKEN_TYPE_IDS["reactant_atom"]
    if role == "solvent":
        return TOKEN_TYPE_IDS["solvent_atom"]
    if role == "metal":
        return TOKEN_TYPE_IDS["metal_atom"]
    if role == "ligand":
        if descriptors[2] > 0:
            return TOKEN_TYPE_IDS["ligand_chiral"]
        if descriptors[3] > 0:
            return TOKEN_TYPE_IDS["ligand_donor"]
        return TOKEN_TYPE_IDS["ligand_atom"]
    if role == "catalyst":
        if atomic_num in COORDINATION_METAL_ATOMIC_NUMBERS:
            return TOKEN_TYPE_IDS["metal_atom"]
        if descriptors[2] > 0:
            return TOKEN_TYPE_IDS["ligand_chiral"]
        if descriptors[3] > 0:
            return TOKEN_TYPE_IDS["ligand_donor"]
        return TOKEN_TYPE_IDS["ligand_atom"]
    if role == "imine":
        if atomic_num == 7:
            return TOKEN_TYPE_IDS["imine_n"]
        return TOKEN_TYPE_IDS["imine_atom"]
    if role == "thiol":
        if atomic_num == 16:
            return TOKEN_TYPE_IDS["thiol_s"]
        return TOKEN_TYPE_IDS["thiol_atom"]
    return TOKEN_TYPE_IDS["reactant_atom"]


def build_tensor_bundle(
    *,
    store: AtomTokenStore,
    mapping_task: str = "aho_major",
    part: str,
    target_frame: pd.DataFrame,
    target_column: str,
    max_rows: int,
    max_tokens: int,
) -> TensorBundle:
    mapping = store.mapping[(store.mapping["task"].astype(str) == mapping_task) & (store.mapping["split_part"] == part)].copy()
    if max_rows > 0:
        keep_rows = set(range(min(max_rows, len(target_frame))))
        target_frame = target_frame.iloc[:max_rows].copy()
        mapping = mapping[mapping["split_row_pos"].isin(keep_rows)].copy()

    metadata = store.metadata.set_index("component_id")
    map_by_row = {
        int(row_pos): frame.sort_values("slot")
        for row_pos, frame in mapping.groupby("split_row_pos", sort=False)
    }
    orb_dim = int(store.node_features.shape[-1])
    numeric_dim = 8

    x_rows = []
    numeric_rows = []
    mask_rows = []
    role_rows = []
    type_rows = []
    atomic_rows = []
    y_rows = []
    row_indices = []
    token_counts = []

    for row_pos, (_, row) in enumerate(target_frame.iterrows()):
        y_value = pd.to_numeric(row.get(target_column), errors="coerce")
        if not np.isfinite(y_value):
            continue

        x = np.zeros((max_tokens, orb_dim), dtype=np.float16)
        numeric = np.zeros((max_tokens, numeric_dim), dtype=np.float32)
        token_mask = np.zeros((max_tokens,), dtype=bool)
        role_ids = np.zeros((max_tokens,), dtype=np.int64)
        token_type_ids = np.full((max_tokens,), TOKEN_TYPE_IDS["missing"], dtype=np.int64)
        atomic_numbers = np.zeros((max_tokens,), dtype=np.int64)
        cursor = 0

        def add_token(
            feature: np.ndarray,
            role_id: int,
            token_type_id: int,
            atomic_num: int,
            numeric_values: np.ndarray,
        ) -> None:
            nonlocal cursor
            if cursor >= max_tokens:
                return
            x[cursor] = feature.astype(np.float16)
            numeric[cursor] = numeric_values.astype(np.float32)
            token_mask[cursor] = True
            role_ids[cursor] = int(role_id)
            token_type_ids[cursor] = int(token_type_id)
            atomic_numbers[cursor] = int(max(0, min(118, atomic_num)))
            cursor += 1

        for map_row in map_by_row.get(row_pos, pd.DataFrame()).itertuples(index=False):
            component_id = int(map_row.component_id)
            role = str(map_row.role)
            role_id = int(map_row.role_id)
            if component_id < 0 or component_id not in metadata.index:
                add_token(
                    np.zeros((orb_dim,), dtype=np.float32),
                    role_id,
                    TOKEN_TYPE_IDS["missing"],
                    0,
                    np.zeros((numeric_dim,), dtype=np.float32),
                )
                continue

            component_meta = metadata.loc[component_id]
            canonical_smiles = str(component_meta["canonical_smiles"])
            mol = Chem.MolFromSmiles(canonical_smiles)
            features, xyz, has_3d = pooled_conformer_features(store, component_id)
            atom_nums = store.atomic_numbers[component_id][store.atom_mask[component_id]]
            descriptors = store.atom_descriptors[component_id][store.atom_mask[component_id]]
            xyz_norm = normalized_positions(xyz)

            if len(features) == 0 or not has_3d:
                add_token(
                    np.zeros((orb_dim,), dtype=np.float32),
                    role_id,
                    TOKEN_TYPE_IDS["missing"],
                    0,
                    np.zeros((numeric_dim,), dtype=np.float32),
                )
                continue

            global_feature = features.mean(axis=0)
            add_token(
                global_feature,
                role_id,
                TOKEN_TYPE_IDS["global"],
                0,
                np.array([0, 0, 0, 0, 0, 0, 0, 1], dtype=np.float32),
            )

            site_indices = site_indices_for_reactant(row, canonical_smiles) if role == "reactant" else set()
            shells = neighbor_shells(mol, site_indices, max_radius=2)

            for atom_idx in range(len(features)):
                desc = descriptors[atom_idx] if atom_idx < len(descriptors) else np.zeros((4,), dtype=np.float32)
                atom_num = int(atom_nums[atom_idx]) if atom_idx < len(atom_nums) else 0
                token_type = token_type_for_atom(role, atom_idx, atom_num, desc, shells)
                pos = xyz_norm[atom_idx] if atom_idx < len(xyz_norm) else np.zeros((3,), dtype=np.float32)
                numeric_values = np.concatenate([pos.astype(np.float32), desc.astype(np.float32), np.array([1.0], dtype=np.float32)])
                add_token(features[atom_idx], role_id, token_type, atom_num, numeric_values)

        x_rows.append(x)
        numeric_rows.append(numeric)
        mask_rows.append(token_mask)
        role_rows.append(role_ids)
        type_rows.append(token_type_ids)
        atomic_rows.append(atomic_numbers)
        y_rows.append(float(y_value))
        row_indices.append(int(row["_source_index"]))
        token_counts.append(int(cursor))

    if not x_rows:
        raise ValueError(f"No valid rows for {part}")

    out = TensorBundle(
        x=torch.from_numpy(np.stack(x_rows, axis=0)),
        numeric=torch.from_numpy(np.stack(numeric_rows, axis=0)),
        token_mask=torch.from_numpy(np.stack(mask_rows, axis=0)),
        role_ids=torch.from_numpy(np.stack(role_rows, axis=0)),
        token_type_ids=torch.from_numpy(np.stack(type_rows, axis=0)),
        atomic_numbers=torch.from_numpy(np.stack(atomic_rows, axis=0)),
        y=torch.from_numpy(np.asarray(y_rows, dtype=np.float32)),
        row_index=np.asarray(row_indices, dtype=int),
    )
    print(
        f"[tokens] part={part} rows={len(y_rows)} "
        f"tokens_mean={np.mean(token_counts):.1f} tokens_max={np.max(token_counts)}",
        flush=True,
    )
    return out


class Orb3DSiteAttentionModel(nn.Module):
    """Standalone site-aware reaction classifier over frozen Orb/OrbMol atom tokens."""

    def __init__(
        self,
        orb_dim: int,
        numeric_dim: int,
        hidden_dim: int,
        num_roles: int,
        num_token_types: int,
        num_layers: int,
        num_heads: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.input_norm = nn.LayerNorm(orb_dim)
        self.input_proj = nn.Linear(orb_dim, hidden_dim)
        self.numeric_proj = nn.Linear(numeric_dim, hidden_dim)
        self.role_embedding = nn.Embedding(num_roles, hidden_dim)
        self.token_type_embedding = nn.Embedding(num_token_types, hidden_dim)
        self.atomic_number_embedding = nn.Embedding(119, hidden_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.pool_score = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        x: torch.Tensor,
        numeric: torch.Tensor,
        token_mask: torch.Tensor,
        role_ids: torch.Tensor,
        token_type_ids: torch.Tensor,
        atomic_numbers: torch.Tensor,
    ) -> torch.Tensor:
        x = x.float()
        token_mask = token_mask.bool()
        h = (
            self.input_proj(self.input_norm(x))
            + self.numeric_proj(numeric.float())
            + self.role_embedding(role_ids)
            + self.token_type_embedding(token_type_ids)
            + self.atomic_number_embedding(atomic_numbers.clamp(min=0, max=118))
        )
        h = h * token_mask.unsqueeze(-1).to(h.dtype)
        cls = self.cls_token.expand(h.size(0), -1, -1)
        tokens = torch.cat([cls, h], dim=1)
        cls_mask = torch.ones((h.size(0), 1), dtype=torch.bool, device=h.device)
        full_mask = torch.cat([cls_mask, token_mask], dim=1)
        encoded = self.encoder(tokens, src_key_padding_mask=~full_mask)
        body = encoded[:, 1:]
        logits = self.pool_score(body).squeeze(-1)
        logits = logits.masked_fill(~token_mask, torch.finfo(logits.dtype).min)
        weights = torch.softmax(logits, dim=1).unsqueeze(-1)
        pooled = (body * weights).sum(dim=1)
        return self.head(torch.cat([encoded[:, 0], pooled], dim=-1)).squeeze(-1)


def make_loader(bundle: TensorBundle, batch_size: int, shuffle: bool) -> DataLoader:
    ds = TensorDataset(
        bundle.x,
        bundle.numeric,
        bundle.token_mask,
        bundle.role_ids,
        bundle.token_type_ids,
        bundle.atomic_numbers,
        bundle.y,
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def train_epoch(model: nn.Module, loader: DataLoader, optimizer, device: str, pos_weight: torch.Tensor) -> float:
    model.train()
    losses = []
    for x, numeric, token_mask, role_ids, token_type_ids, atomic_numbers, y in loader:
        pred = model(
            x.to(device),
            numeric.to(device),
            token_mask.to(device),
            role_ids.to(device),
            token_type_ids.to(device),
            atomic_numbers.to(device),
        )
        loss = F.binary_cross_entropy_with_logits(pred, y.to(device), pos_weight=pos_weight)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses)) if losses else float("nan")


@torch.no_grad()
def predict(model: nn.Module, bundle: TensorBundle, batch_size: int, device: str) -> np.ndarray:
    model.eval()
    preds = []
    for x, numeric, token_mask, role_ids, token_type_ids, atomic_numbers, _ in make_loader(bundle, batch_size, shuffle=False):
        raw = model(
            x.to(device),
            numeric.to(device),
            token_mask.to(device),
            role_ids.to(device),
            token_type_ids.to(device),
            atomic_numbers.to(device),
        )
        preds.append(torch.sigmoid(raw).detach().cpu().numpy())
    return np.concatenate(preds, axis=0) if preds else np.array([], dtype=float)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train standalone OrbMol-v2 site-aware atom-token attention on AHO major.")
    parser.add_argument("--split", default="paper_exact_test", choices=["paper_exact_test", "paper_random42", "example"])
    parser.add_argument(
        "--atom-token-dir",
        type=Path,
        default=ROOT / "reproduction_outputs" / "full_retrain" / "orb3d_atom_tokens_aho_major",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "reproduction_outputs" / "full_retrain" / "orb3d_site_attention_aho_major",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--hidden-dim", type=int, default=192)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--lr", type=float, default=6e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--max-tokens", type=int, default=176)
    parser.add_argument("--max-train-rows", type=int, default=0)
    parser.add_argument("--max-eval-rows", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260722)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    store = load_atom_token_store(args.atom_token_dir)
    bundle = load_task("aho_major", split=args.split)
    frames = split_frames(bundle)
    train_bundle = build_tensor_bundle(
        store=store,
        part="train",
        target_frame=frames["train"],
        target_column=bundle.target_column,
        max_rows=args.max_train_rows,
        max_tokens=args.max_tokens,
    )
    val_bundle = build_tensor_bundle(
        store=store,
        part="val",
        target_frame=frames["val"],
        target_column=bundle.target_column,
        max_rows=args.max_eval_rows,
        max_tokens=args.max_tokens,
    )
    test_bundle = build_tensor_bundle(
        store=store,
        part="test",
        target_frame=frames["test"],
        target_column=bundle.target_column,
        max_rows=args.max_eval_rows,
        max_tokens=args.max_tokens,
    )

    y_train = train_bundle.y.numpy()
    positives = float(y_train.sum())
    negatives = float(len(y_train) - positives)
    pos_weight = torch.tensor([negatives / max(positives, 1.0)], dtype=torch.float32, device=args.device)

    model = Orb3DSiteAttentionModel(
        orb_dim=int(store.node_features.shape[-1]),
        numeric_dim=8,
        hidden_dim=args.hidden_dim,
        num_roles=16,
        num_token_types=max(TOKEN_TYPE_IDS.values()) + 1,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        dropout=args.dropout,
    ).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loader = make_loader(train_bundle, args.batch_size, shuffle=True)

    best_state = None
    best_eval = float("inf")
    history = []
    for epoch in range(args.epochs):
        loss = train_epoch(model, loader, optimizer, args.device, pos_weight)
        val_pred = predict(model, val_bundle, args.batch_size, args.device)
        val_true = val_bundle.y.numpy().astype(float)
        val_best = best_threshold_accuracy(val_true, val_pred)
        eval_score = 1.0 - val_best["best_accuracy"]
        history.append(
            {
                "epoch": epoch,
                "train_loss": loss,
                "eval_score": float(eval_score),
                "eval_best_accuracy": val_best["best_accuracy"],
                "eval_best_threshold": val_best["best_threshold"],
            }
        )
        if eval_score < best_eval:
            best_eval = float(eval_score)
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        print(
            f"[epoch] {epoch + 1}/{args.epochs} train_loss={loss:.6f} "
            f"val_best_acc={val_best['best_accuracy']:.6f} threshold={val_best['best_threshold']:.2f}",
            flush=True,
        )

    if best_state is not None:
        model.load_state_dict(best_state)
    val_pred = predict(model, val_bundle, args.batch_size, args.device)
    val_true = val_bundle.y.numpy().astype(float)
    threshold = best_threshold_accuracy(val_true, val_pred)["best_threshold"]
    test_pred = predict(model, test_bundle, args.batch_size, args.device)
    test_true = test_bundle.y.numpy().astype(float)
    metrics = classification_metrics(test_true, test_pred, threshold=threshold)
    metrics.update(best_threshold_accuracy(test_true, test_pred))

    pred_df = pd.DataFrame(
        {
            "row_index": test_bundle.row_index,
            "y_true": test_true,
            "y_prob": test_pred,
            "y_pred": (test_pred >= threshold).astype(int),
        }
    )
    pred_path = args.out_dir / "aho_major__orbmol_v2_site_attention__predictions.csv"
    pred_df.to_csv(pred_path, index=False)
    metrics.update(
        {
            "task": "aho_major",
            "split": args.split,
            "model": "orbmol_v2_site_attention",
            "feature": "atom_site_tokens_orbmol_v2",
            "n_train": int(len(train_bundle.y)),
            "n_val": int(len(val_bundle.y)),
            "atom_token_dir": str(args.atom_token_dir),
            "prediction_path": str(pred_path),
        }
    )
    pd.DataFrame([metrics]).to_csv(args.out_dir / "metrics.csv", index=False)
    pd.DataFrame(history).to_csv(args.out_dir / "training_history.csv", index=False)
    torch.save({"model_state_dict": model.state_dict(), "args": vars(args)}, args.out_dir / "model.pt")
    config = vars(args).copy()
    for key, value in list(config.items()):
        if isinstance(value, Path):
            config[key] = str(value)
    with open(args.out_dir / "run_config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    print(pd.DataFrame([metrics]).to_string(index=False))
    print(f"Wrote {args.out_dir}")


if __name__ == "__main__":
    main()
