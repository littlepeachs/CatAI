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
from full_retrain.train_orb3d_reaction_complex_heads import ROLE_IDS
from full_retrain.train_orb3d_site_attention import (
    TOKEN_TYPE_IDS,
    clear_atom_maps,
    load_atom_token_store,
    neighbor_shells,
    normalized_positions,
    parse_site_atom_maps,
    split_frames,
    token_type_for_atom,
)
from full_retrain.train_orb3d_site_classical_heads import conformer_feature_dim, conformer_pooled_atom_features
from full_retrain.train_orb3d_site_geometry_heads import mean_positions


RDLogger.DisableLog("rdApp.*")


@dataclass
class AtomRelationTensorBundle:
    x: torch.Tensor
    numeric: torch.Tensor
    token_mask: torch.Tensor
    role_ids: torch.Tensor
    token_type_ids: torch.Tensor
    atomic_numbers: torch.Tensor
    component_ids: torch.Tensor
    y: torch.Tensor
    row_index: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train standalone OrbMol-v2 atom-token attention with same-component 3D distance bias."
    )
    parser.add_argument("--split", default="paper_exact_test", choices=["paper_exact_test", "paper_random42", "example"])
    parser.add_argument(
        "--atom-token-dir",
        type=Path,
        default=ROOT / "reproduction_outputs" / "full_retrain" / "orb3d_atom_tokens_aho_major_ligand_metal_h100_c3",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "reproduction_outputs" / "full_retrain" / "orb3d_atom_relation_attention_aho_major",
    )
    parser.add_argument("--conformer-feature-mode", default="mean_std_max", choices=["mean", "mean_std", "mean_std_max"])
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--hidden-dim", type=int, default=192)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--patience", type=int, default=28)
    parser.add_argument("--lr", type=float, default=6e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--max-train-rows", type=int, default=0)
    parser.add_argument("--max-eval-rows", type=int, default=0)
    parser.add_argument(
        "--relation-mode",
        default="same_component",
        choices=[
            "same_component",
            "same_component_role",
            "site_donor_constrained",
            "site_donor_orientation",
        ],
        help=(
            "same_component keeps the original within-component distance/RBF bias. "
            "same_component_role adds cross-component role/type pair flags while "
            "keeping distances/RBFs restricted to same-component atom pairs. "
            "site_donor_constrained also restricts cross-component attention to "
            "reactant-site/donor/metal/solvent role pairs and pools from those "
            "chemically focused tokens. site_donor_orientation adds invariant "
            "component-local radial and focus-distance geometry to that constrained "
            "attention view."
        ),
    )
    parser.add_argument("--seed", type=int, default=20260728)
    return parser.parse_args()


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


def component_entry(
    store,
    metadata: pd.DataFrame,
    component_id: int,
    cache: dict[tuple[int, str], dict],
    conformer_feature_mode: str,
) -> dict:
    key = (int(component_id), conformer_feature_mode)
    if key in cache:
        return cache[key]
    dim = conformer_feature_dim(store, conformer_feature_mode)
    if component_id < 0 or component_id not in metadata.index:
        entry = {
            "mol": None,
            "features": np.zeros((0, dim), dtype=np.float32),
            "xyz": np.zeros((0, 3), dtype=np.float32),
            "descriptors": np.zeros((0, 4), dtype=np.float32),
            "atomic_numbers": np.zeros((0,), dtype=np.int64),
            "has_3d": False,
            "canonical_smiles": "",
        }
        cache[key] = entry
        return entry
    meta = metadata.loc[component_id]
    canonical_smiles = str(meta["canonical_smiles"])
    features, has_features = conformer_pooled_atom_features(store, component_id, conformer_feature_mode)
    xyz, has_xyz = mean_positions(store, component_id)
    atom_mask = store.atom_mask[component_id]
    entry = {
        "mol": Chem.MolFromSmiles(canonical_smiles),
        "features": features.astype(np.float32),
        "xyz": xyz.astype(np.float32),
        "descriptors": store.atom_descriptors[component_id][atom_mask].astype(np.float32),
        "atomic_numbers": store.atomic_numbers[component_id][atom_mask].astype(np.int64),
        "has_3d": bool(has_features and has_xyz and len(features) == len(xyz)),
        "canonical_smiles": canonical_smiles,
    }
    cache[key] = entry
    return entry


def build_tensor_bundle(
    *,
    store,
    mapping_task: str = "aho_major",
    part: str,
    target_frame: pd.DataFrame,
    target_column: str,
    max_rows: int,
    max_tokens: int,
    conformer_feature_mode: str,
) -> AtomRelationTensorBundle:
    mapping = store.mapping[
        (store.mapping["task"].astype(str) == mapping_task)
        & (store.mapping["split_part"] == part)
    ].copy()
    if max_rows > 0:
        keep_rows = set(range(min(max_rows, len(target_frame))))
        target_frame = target_frame.iloc[:max_rows].copy()
        mapping = mapping[mapping["split_row_pos"].isin(keep_rows)].copy()

    metadata = store.metadata.set_index("component_id")
    map_by_row = {
        int(row_pos): frame.sort_values("slot")
        for row_pos, frame in mapping.groupby("split_row_pos", sort=False)
    }
    orb_dim = conformer_feature_dim(store, conformer_feature_mode)
    numeric_dim = 8
    cache: dict[tuple[int, str], dict] = {}

    x_rows = []
    numeric_rows = []
    mask_rows = []
    role_rows = []
    type_rows = []
    atomic_rows = []
    component_rows = []
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
        component_ids = np.full((max_tokens,), -1, dtype=np.int64)
        cursor = 0

        def add_token(
            feature: np.ndarray,
            role_id: int,
            token_type_id: int,
            atomic_num: int,
            component_local_id: int,
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
            component_ids[cursor] = int(component_local_id)
            cursor += 1

        for component_local_id, map_row in enumerate(map_by_row.get(row_pos, pd.DataFrame()).itertuples(index=False)):
            component_id = int(map_row.component_id)
            role = str(map_row.role)
            role_id = int(map_row.role_id)
            entry = component_entry(store, metadata, component_id, cache, conformer_feature_mode)
            features = entry["features"]
            xyz = entry["xyz"]
            descriptors = entry["descriptors"]
            atom_nums = entry["atomic_numbers"]

            if len(features) == 0 or not entry["has_3d"]:
                add_token(
                    np.zeros((orb_dim,), dtype=np.float32),
                    role_id,
                    TOKEN_TYPE_IDS["missing"],
                    0,
                    component_local_id,
                    np.zeros((numeric_dim,), dtype=np.float32),
                )
                continue

            xyz_norm = normalized_positions(xyz)
            global_feature = features.mean(axis=0)
            add_token(
                global_feature,
                role_id,
                TOKEN_TYPE_IDS["global"],
                0,
                component_local_id,
                np.array([0, 0, 0, 0, 0, 0, 0, 1], dtype=np.float32),
            )

            site_indices = site_indices_for_reactant(row, entry["canonical_smiles"]) if role == "reactant" else set()
            mol = entry["mol"] if isinstance(entry["mol"], Chem.Mol) else None
            shells = neighbor_shells(mol, site_indices, max_radius=2)

            for atom_idx in range(len(features)):
                desc = descriptors[atom_idx] if atom_idx < len(descriptors) else np.zeros((4,), dtype=np.float32)
                atom_num = int(atom_nums[atom_idx]) if atom_idx < len(atom_nums) else 0
                token_type = token_type_for_atom(role, atom_idx, atom_num, desc, shells)
                pos = xyz_norm[atom_idx] if atom_idx < len(xyz_norm) else np.zeros((3,), dtype=np.float32)
                numeric_values = np.concatenate(
                    [pos.astype(np.float32), desc.astype(np.float32), np.array([1.0], dtype=np.float32)]
                )
                add_token(features[atom_idx], role_id, token_type, atom_num, component_local_id, numeric_values)

        x_rows.append(x)
        numeric_rows.append(numeric)
        mask_rows.append(token_mask)
        role_rows.append(role_ids)
        type_rows.append(token_type_ids)
        atomic_rows.append(atomic_numbers)
        component_rows.append(component_ids)
        y_rows.append(float(y_value))
        row_indices.append(int(row["_source_index"]))
        token_counts.append(int(cursor))

    if not x_rows:
        raise ValueError(f"No valid rows for {part}")

    out = AtomRelationTensorBundle(
        x=torch.from_numpy(np.stack(x_rows, axis=0)),
        numeric=torch.from_numpy(np.stack(numeric_rows, axis=0)),
        token_mask=torch.from_numpy(np.stack(mask_rows, axis=0)),
        role_ids=torch.from_numpy(np.stack(role_rows, axis=0)),
        token_type_ids=torch.from_numpy(np.stack(type_rows, axis=0)),
        atomic_numbers=torch.from_numpy(np.stack(atomic_rows, axis=0)),
        component_ids=torch.from_numpy(np.stack(component_rows, axis=0)),
        y=torch.from_numpy(np.asarray(y_rows, dtype=np.float32)),
        row_index=np.asarray(row_indices, dtype=int),
    )
    print(
        f"[tokens] part={part} rows={len(y_rows)} tokens_mean={np.mean(token_counts):.1f} "
        f"tokens_max={np.max(token_counts)} orb_dim={orb_dim}",
        flush=True,
    )
    return out


class AtomRelationEncoderLayer(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, relation_dim: int, dropout: float) -> None:
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.qkv = nn.Linear(hidden_dim, hidden_dim * 3)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.relation_bias = nn.Sequential(
            nn.LayerNorm(relation_dim),
            nn.Linear(relation_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, num_heads),
        )
        self.attn_dropout = nn.Dropout(dropout)
        self.dropout = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, relation: torch.Tensor, full_mask: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        h = self.norm1(x)
        qkv = self.qkv(h).view(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim**0.5)
        bias = self.relation_bias(relation).permute(0, 3, 1, 2)
        scores = scores + bias
        if full_mask.dim() == 2:
            attention_mask = full_mask[:, None, None, :]
        elif full_mask.dim() == 3:
            attention_mask = full_mask[:, None, :, :]
        else:
            raise ValueError(f"full_mask must have rank 2 or 3, got {full_mask.dim()}")
        scores = scores.masked_fill(~attention_mask, torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores, dim=-1)
        weights = self.attn_dropout(weights)
        attended = torch.matmul(weights, v).transpose(1, 2).reshape(batch_size, seq_len, self.hidden_dim)
        x = x + self.dropout(self.out_proj(attended))
        x = x + self.ffn(self.norm2(x))
        return x


class Orb3DAtomRelationAttentionModel(nn.Module):
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
        relation_mode: str,
    ) -> None:
        super().__init__()
        self.relation_mode = relation_mode
        self.register_buffer("rbf_centers", torch.linspace(0.0, 3.0, 8), persistent=False)
        self.rbf_width = 0.45
        if relation_mode == "same_component":
            relation_dim = 11
        elif relation_mode == "same_component_role":
            relation_dim = 27
        elif relation_mode == "site_donor_constrained":
            relation_dim = 33
        elif relation_mode == "site_donor_orientation":
            relation_dim = 43
        else:
            raise ValueError(f"Unsupported relation_mode: {relation_mode}")
        self.input_norm = nn.LayerNorm(orb_dim)
        self.input_proj = nn.Linear(orb_dim, hidden_dim)
        self.numeric_proj = nn.Linear(numeric_dim, hidden_dim)
        self.role_embedding = nn.Embedding(num_roles, hidden_dim)
        self.token_type_embedding = nn.Embedding(num_token_types, hidden_dim)
        self.atomic_number_embedding = nn.Embedding(119, hidden_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.layers = nn.ModuleList(
            [AtomRelationEncoderLayer(hidden_dim, num_heads, relation_dim, dropout) for _ in range(num_layers)]
        )
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

    def relation_features(
        self,
        numeric: torch.Tensor,
        token_mask: torch.Tensor,
        component_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        pos = numeric[..., :3].float()
        token_mask = token_mask.bool()
        component_ids = component_ids.long()
        same_component = (
            (component_ids[:, :, None] == component_ids[:, None, :])
            & (component_ids[:, :, None] >= 0)
            & token_mask[:, :, None]
            & token_mask[:, None, :]
        )
        delta = pos[:, :, None, :] - pos[:, None, :, :]
        dist = torch.sqrt(torch.clamp((delta * delta).sum(dim=-1), min=0.0) + 1e-8)
        dist = dist * same_component.to(dist.dtype)
        rbf = torch.exp(-0.5 * ((dist[..., None] - self.rbf_centers.to(dist.device)) / self.rbf_width) ** 2)
        rbf = rbf * same_component[..., None].to(rbf.dtype)
        eye = torch.eye(token_mask.size(1), dtype=torch.bool, device=token_mask.device).unsqueeze(0)
        relation_body = torch.cat(
            [
                same_component[..., None].float(),
                eye.expand_as(same_component)[..., None].float(),
                dist[..., None],
                rbf,
            ],
            dim=-1,
        )
        batch_size, n_tokens, _, relation_dim = relation_body.shape
        full_relation = torch.zeros(
            batch_size,
            n_tokens + 1,
            n_tokens + 1,
            relation_dim,
            dtype=relation_body.dtype,
            device=relation_body.device,
        )
        full_relation[:, 1:, 1:, :] = relation_body
        cls_mask = torch.ones((batch_size, 1), dtype=torch.bool, device=token_mask.device)
        full_mask = torch.cat([cls_mask, token_mask], dim=1)
        return full_relation, full_mask

    def chemically_focused_token_mask(
        self,
        token_mask: torch.Tensor,
        role_ids: torch.Tensor,
        token_type_ids: torch.Tensor,
    ) -> torch.Tensor:
        token_type_ids = token_type_ids.long()
        role_ids = role_ids.long()
        reactant = role_ids == ROLE_IDS["reactant"]
        ligand = role_ids == ROLE_IDS["ligand"]
        metal = role_ids == ROLE_IDS["metal"]
        solvent = role_ids == ROLE_IDS["solvent"]
        global_token = token_type_ids == TOKEN_TYPE_IDS["global"]
        reactant_site_shell = (
            (token_type_ids == TOKEN_TYPE_IDS["reactant_site"])
            | (token_type_ids == TOKEN_TYPE_IDS["reactant_neighbor1"])
            | (token_type_ids == TOKEN_TYPE_IDS["reactant_neighbor2"])
        )
        ligand_donor_context = (
            (token_type_ids == TOKEN_TYPE_IDS["ligand_donor"])
            | (token_type_ids == TOKEN_TYPE_IDS["ligand_chiral"])
            | (ligand & global_token)
        )
        reactant_context = reactant_site_shell | (reactant & global_token)
        focus = reactant_context | ligand_donor_context | metal | solvent
        return focus & token_mask.bool()

    def role_pair_relation_features(
        self,
        numeric: torch.Tensor,
        token_mask: torch.Tensor,
        role_ids: torch.Tensor,
        token_type_ids: torch.Tensor,
        component_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        pos = numeric[..., :3].float()
        token_mask = token_mask.bool()
        valid_pair = token_mask[:, :, None] & token_mask[:, None, :]
        component_ids = component_ids.long()
        same_component = (
            (component_ids[:, :, None] == component_ids[:, None, :])
            & (component_ids[:, :, None] >= 0)
            & valid_pair
        )
        cross_component = valid_pair & ~same_component
        same_role = (role_ids[:, :, None] == role_ids[:, None, :]) & valid_pair
        same_token_type = (token_type_ids[:, :, None] == token_type_ids[:, None, :]) & valid_pair

        token_type_ids = token_type_ids.long()
        role_ids = role_ids.long()
        site = token_type_ids == TOKEN_TYPE_IDS["reactant_site"]
        shell2 = (
            site
            | (token_type_ids == TOKEN_TYPE_IDS["reactant_neighbor1"])
            | (token_type_ids == TOKEN_TYPE_IDS["reactant_neighbor2"])
        )
        ligand = role_ids == ROLE_IDS["ligand"]
        metal = role_ids == ROLE_IDS["metal"]
        solvent = role_ids == ROLE_IDS["solvent"]
        reactant = role_ids == ROLE_IDS["reactant"]
        donor = token_type_ids == TOKEN_TYPE_IDS["ligand_donor"]
        chiral = token_type_ids == TOKEN_TYPE_IDS["ligand_chiral"]

        def unordered_pair(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
            return ((left[:, :, None] & right[:, None, :]) | (right[:, :, None] & left[:, None, :])) & valid_pair

        role_pair_flags = [
            unordered_pair(site, donor),
            unordered_pair(shell2, donor),
            unordered_pair(site, ligand),
            unordered_pair(shell2, ligand),
            unordered_pair(site, metal),
            unordered_pair(shell2, metal),
            unordered_pair(donor, metal),
            unordered_pair(chiral, site),
            unordered_pair(solvent, ligand),
            unordered_pair(solvent, metal),
            unordered_pair(reactant, ligand),
            unordered_pair(reactant, metal),
        ]

        delta = pos[:, :, None, :] - pos[:, None, :, :]
        dist = torch.sqrt(torch.clamp((delta * delta).sum(dim=-1), min=0.0) + 1e-8)
        dist = dist * same_component.to(dist.dtype)
        rbf = torch.exp(-0.5 * ((dist[..., None] - self.rbf_centers.to(dist.device)) / self.rbf_width) ** 2)
        rbf = rbf * same_component[..., None].to(rbf.dtype)
        eye = torch.eye(token_mask.size(1), dtype=torch.bool, device=token_mask.device).unsqueeze(0)
        relation_body = torch.cat(
            [
                valid_pair[..., None].float(),
                same_component[..., None].float(),
                cross_component[..., None].float(),
                same_role[..., None].float(),
                same_token_type[..., None].float(),
                eye.expand_as(valid_pair)[..., None].float(),
                *[flag[..., None].float() for flag in role_pair_flags],
                dist[..., None],
                rbf,
            ],
            dim=-1,
        )
        batch_size, n_tokens, _, relation_dim = relation_body.shape
        full_relation = torch.zeros(
            batch_size,
            n_tokens + 1,
            n_tokens + 1,
            relation_dim,
            dtype=relation_body.dtype,
            device=relation_body.device,
        )
        full_relation[:, 1:, 1:, :] = relation_body
        cls_mask = torch.ones((batch_size, 1), dtype=torch.bool, device=token_mask.device)
        full_mask = torch.cat([cls_mask, token_mask], dim=1)
        return full_relation, full_mask

    def site_donor_constrained_relation_features(
        self,
        numeric: torch.Tensor,
        token_mask: torch.Tensor,
        role_ids: torch.Tensor,
        token_type_ids: torch.Tensor,
        component_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        pos = numeric[..., :3].float()
        token_mask = token_mask.bool()
        valid_pair = token_mask[:, :, None] & token_mask[:, None, :]
        component_ids = component_ids.long()
        same_component = (
            (component_ids[:, :, None] == component_ids[:, None, :])
            & (component_ids[:, :, None] >= 0)
            & valid_pair
        )
        cross_component = valid_pair & ~same_component
        same_role = (role_ids[:, :, None] == role_ids[:, None, :]) & valid_pair
        same_token_type = (token_type_ids[:, :, None] == token_type_ids[:, None, :]) & valid_pair

        token_type_ids = token_type_ids.long()
        role_ids = role_ids.long()
        site = token_type_ids == TOKEN_TYPE_IDS["reactant_site"]
        shell2 = (
            site
            | (token_type_ids == TOKEN_TYPE_IDS["reactant_neighbor1"])
            | (token_type_ids == TOKEN_TYPE_IDS["reactant_neighbor2"])
        )
        ligand = role_ids == ROLE_IDS["ligand"]
        metal = role_ids == ROLE_IDS["metal"]
        solvent = role_ids == ROLE_IDS["solvent"]
        reactant = role_ids == ROLE_IDS["reactant"]
        global_token = token_type_ids == TOKEN_TYPE_IDS["global"]
        donor = token_type_ids == TOKEN_TYPE_IDS["ligand_donor"]
        chiral = token_type_ids == TOKEN_TYPE_IDS["ligand_chiral"]
        reactant_context = shell2 | (reactant & global_token)
        ligand_donor_context = donor | chiral | (ligand & global_token)
        metal_context = metal
        solvent_context = solvent
        focus = reactant_context | ligand_donor_context | metal_context | solvent_context

        def unordered_pair(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
            return ((left[:, :, None] & right[:, None, :]) | (right[:, :, None] & left[:, None, :])) & valid_pair

        role_pair_flags = [
            unordered_pair(site, donor),
            unordered_pair(shell2, donor),
            unordered_pair(site, ligand),
            unordered_pair(shell2, ligand),
            unordered_pair(site, metal),
            unordered_pair(shell2, metal),
            unordered_pair(donor, metal),
            unordered_pair(chiral, site),
            unordered_pair(solvent, ligand),
            unordered_pair(solvent, metal),
            unordered_pair(reactant, ligand),
            unordered_pair(reactant, metal),
        ]
        constrained_cross = (
            unordered_pair(reactant_context, ligand_donor_context)
            | unordered_pair(reactant_context, metal_context)
            | unordered_pair(reactant_context, solvent_context)
            | unordered_pair(ligand_donor_context, metal_context)
            | unordered_pair(ligand_donor_context, solvent_context)
            | unordered_pair(metal_context, solvent_context)
        ) & cross_component
        allowed_pair = same_component | constrained_cross

        delta = pos[:, :, None, :] - pos[:, None, :, :]
        dist = torch.sqrt(torch.clamp((delta * delta).sum(dim=-1), min=0.0) + 1e-8)
        dist = dist * same_component.to(dist.dtype)
        rbf = torch.exp(-0.5 * ((dist[..., None] - self.rbf_centers.to(dist.device)) / self.rbf_width) ** 2)
        rbf = rbf * same_component[..., None].to(rbf.dtype)
        eye = torch.eye(token_mask.size(1), dtype=torch.bool, device=token_mask.device).unsqueeze(0)
        relation_body = torch.cat(
            [
                valid_pair[..., None].float(),
                same_component[..., None].float(),
                cross_component[..., None].float(),
                allowed_pair[..., None].float(),
                same_role[..., None].float(),
                same_token_type[..., None].float(),
                eye.expand_as(valid_pair)[..., None].float(),
                unordered_pair(focus, focus)[..., None].float(),
                unordered_pair(reactant_context, ligand_donor_context)[..., None].float(),
                unordered_pair(ligand_donor_context, metal_context)[..., None].float(),
                unordered_pair(reactant_context, metal_context)[..., None].float(),
                unordered_pair(solvent_context, focus)[..., None].float(),
                *[flag[..., None].float() for flag in role_pair_flags],
                dist[..., None],
                rbf,
            ],
            dim=-1,
        )
        batch_size, n_tokens, _, relation_dim = relation_body.shape
        full_relation = torch.zeros(
            batch_size,
            n_tokens + 1,
            n_tokens + 1,
            relation_dim,
            dtype=relation_body.dtype,
            device=relation_body.device,
        )
        full_relation[:, 1:, 1:, :] = relation_body

        key_mask = torch.cat(
            [torch.ones((batch_size, 1), dtype=torch.bool, device=token_mask.device), token_mask],
            dim=1,
        )
        full_pair_mask = torch.zeros(
            batch_size,
            n_tokens + 1,
            n_tokens + 1,
            dtype=torch.bool,
            device=token_mask.device,
        )
        full_pair_mask[:, 0, :] = key_mask
        full_pair_mask[:, :, 0] = True
        full_pair_mask[:, 1:, 1:] = allowed_pair
        return full_relation, full_pair_mask

    def site_donor_orientation_relation_features(
        self,
        numeric: torch.Tensor,
        token_mask: torch.Tensor,
        role_ids: torch.Tensor,
        token_type_ids: torch.Tensor,
        component_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        pos = numeric[..., :3].float()
        token_mask = token_mask.bool()
        valid_pair = token_mask[:, :, None] & token_mask[:, None, :]
        component_ids = component_ids.long()
        same_component = (
            (component_ids[:, :, None] == component_ids[:, None, :])
            & (component_ids[:, :, None] >= 0)
            & valid_pair
        )
        cross_component = valid_pair & ~same_component
        same_role = (role_ids[:, :, None] == role_ids[:, None, :]) & valid_pair
        same_token_type = (token_type_ids[:, :, None] == token_type_ids[:, None, :]) & valid_pair

        token_type_ids = token_type_ids.long()
        role_ids = role_ids.long()
        site = token_type_ids == TOKEN_TYPE_IDS["reactant_site"]
        shell2 = (
            site
            | (token_type_ids == TOKEN_TYPE_IDS["reactant_neighbor1"])
            | (token_type_ids == TOKEN_TYPE_IDS["reactant_neighbor2"])
        )
        ligand = role_ids == ROLE_IDS["ligand"]
        metal = role_ids == ROLE_IDS["metal"]
        solvent = role_ids == ROLE_IDS["solvent"]
        reactant = role_ids == ROLE_IDS["reactant"]
        global_token = token_type_ids == TOKEN_TYPE_IDS["global"]
        donor = token_type_ids == TOKEN_TYPE_IDS["ligand_donor"]
        chiral = token_type_ids == TOKEN_TYPE_IDS["ligand_chiral"]
        reactant_context = shell2 | (reactant & global_token)
        ligand_donor_context = donor | chiral | (ligand & global_token)
        metal_context = metal
        solvent_context = solvent
        focus = reactant_context | ligand_donor_context | metal_context | solvent_context

        def unordered_pair(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
            return ((left[:, :, None] & right[:, None, :]) | (right[:, :, None] & left[:, None, :])) & valid_pair

        role_pair_flags = [
            unordered_pair(site, donor),
            unordered_pair(shell2, donor),
            unordered_pair(site, ligand),
            unordered_pair(shell2, ligand),
            unordered_pair(site, metal),
            unordered_pair(shell2, metal),
            unordered_pair(donor, metal),
            unordered_pair(chiral, site),
            unordered_pair(solvent, ligand),
            unordered_pair(solvent, metal),
            unordered_pair(reactant, ligand),
            unordered_pair(reactant, metal),
        ]
        constrained_cross = (
            unordered_pair(reactant_context, ligand_donor_context)
            | unordered_pair(reactant_context, metal_context)
            | unordered_pair(reactant_context, solvent_context)
            | unordered_pair(ligand_donor_context, metal_context)
            | unordered_pair(ligand_donor_context, solvent_context)
            | unordered_pair(metal_context, solvent_context)
        ) & cross_component
        allowed_pair = same_component | constrained_cross

        delta = pos[:, :, None, :] - pos[:, None, :, :]
        dist = torch.sqrt(torch.clamp((delta * delta).sum(dim=-1), min=0.0) + 1e-8)
        dist = dist * same_component.to(dist.dtype)
        rbf = torch.exp(-0.5 * ((dist[..., None] - self.rbf_centers.to(dist.device)) / self.rbf_width) ** 2)
        rbf = rbf * same_component[..., None].to(rbf.dtype)

        radial = torch.sqrt(torch.clamp((pos * pos).sum(dim=-1), min=0.0) + 1e-8)
        unit = pos / radial.clamp(min=1e-6).unsqueeze(-1)
        centroid_cos = (unit[:, :, None, :] * unit[:, None, :, :]).sum(dim=-1)
        radial_i = radial[:, :, None]
        radial_j = radial[:, None, :]

        def masked_center(mask: torch.Tensor) -> torch.Tensor:
            weights = (mask & token_mask).float()
            denom = weights.sum(dim=1, keepdim=True).clamp(min=1.0)
            return (pos * weights[..., None]).sum(dim=1) / denom

        reactant_center = masked_center(reactant_context)
        ligand_center = masked_center(ligand_donor_context)
        metal_center = masked_center(metal_context)
        solvent_center = masked_center(solvent_context)
        token_center = torch.zeros_like(pos)
        token_center = torch.where(reactant[..., None], reactant_center[:, None, :], token_center)
        token_center = torch.where(ligand[..., None], ligand_center[:, None, :], token_center)
        token_center = torch.where(metal[..., None], metal_center[:, None, :], token_center)
        token_center = torch.where(solvent[..., None], solvent_center[:, None, :], token_center)
        rel_focus = pos - token_center
        focus_radius = torch.sqrt(torch.clamp((rel_focus * rel_focus).sum(dim=-1), min=0.0) + 1e-8)
        focus_unit = rel_focus / focus_radius.clamp(min=1e-6).unsqueeze(-1)
        focus_cos = (focus_unit[:, :, None, :] * focus_unit[:, None, :, :]).sum(dim=-1)
        focus_i = focus_radius[:, :, None]
        focus_j = focus_radius[:, None, :]

        eye = torch.eye(token_mask.size(1), dtype=torch.bool, device=token_mask.device).unsqueeze(0)
        relation_body = torch.cat(
            [
                valid_pair[..., None].float(),
                same_component[..., None].float(),
                cross_component[..., None].float(),
                allowed_pair[..., None].float(),
                same_role[..., None].float(),
                same_token_type[..., None].float(),
                eye.expand_as(valid_pair)[..., None].float(),
                unordered_pair(focus, focus)[..., None].float(),
                unordered_pair(reactant_context, ligand_donor_context)[..., None].float(),
                unordered_pair(ligand_donor_context, metal_context)[..., None].float(),
                unordered_pair(reactant_context, metal_context)[..., None].float(),
                unordered_pair(solvent_context, focus)[..., None].float(),
                *[flag[..., None].float() for flag in role_pair_flags],
                dist[..., None],
                rbf,
                (centroid_cos * same_component.float())[..., None],
                (centroid_cos.abs() * same_component.float())[..., None],
                ((radial_i - radial_j).abs() * same_component.float())[..., None],
                ((radial_i + radial_j) * same_component.float())[..., None],
                (focus_i * allowed_pair.float())[..., None],
                (focus_j * allowed_pair.float())[..., None],
                ((focus_i - focus_j).abs() * allowed_pair.float())[..., None],
                ((focus_i + focus_j) * allowed_pair.float())[..., None],
                (focus_cos * same_component.float())[..., None],
                (focus_cos.abs() * same_component.float())[..., None],
            ],
            dim=-1,
        )
        batch_size, n_tokens, _, relation_dim = relation_body.shape
        full_relation = torch.zeros(
            batch_size,
            n_tokens + 1,
            n_tokens + 1,
            relation_dim,
            dtype=relation_body.dtype,
            device=relation_body.device,
        )
        full_relation[:, 1:, 1:, :] = relation_body

        key_mask = torch.cat(
            [torch.ones((batch_size, 1), dtype=torch.bool, device=token_mask.device), token_mask],
            dim=1,
        )
        full_pair_mask = torch.zeros(
            batch_size,
            n_tokens + 1,
            n_tokens + 1,
            dtype=torch.bool,
            device=token_mask.device,
        )
        full_pair_mask[:, 0, :] = key_mask
        full_pair_mask[:, :, 0] = True
        full_pair_mask[:, 1:, 1:] = allowed_pair
        return full_relation, full_pair_mask

    def forward(
        self,
        x: torch.Tensor,
        numeric: torch.Tensor,
        token_mask: torch.Tensor,
        role_ids: torch.Tensor,
        token_type_ids: torch.Tensor,
        atomic_numbers: torch.Tensor,
        component_ids: torch.Tensor,
    ) -> torch.Tensor:
        token_mask = token_mask.bool()
        h = (
            self.input_proj(self.input_norm(x.float()))
            + self.numeric_proj(numeric.float())
            + self.role_embedding(role_ids)
            + self.token_type_embedding(token_type_ids)
            + self.atomic_number_embedding(atomic_numbers.clamp(min=0, max=118))
        )
        h = h * token_mask.unsqueeze(-1).to(h.dtype)
        cls = self.cls_token.expand(h.size(0), -1, -1)
        encoded = torch.cat([cls, h], dim=1)
        if self.relation_mode == "same_component":
            relation, full_mask = self.relation_features(numeric, token_mask, component_ids)
        elif self.relation_mode == "same_component_role":
            relation, full_mask = self.role_pair_relation_features(
                numeric, token_mask, role_ids, token_type_ids, component_ids
            )
        elif self.relation_mode == "site_donor_constrained":
            relation, full_mask = self.site_donor_constrained_relation_features(
                numeric, token_mask, role_ids, token_type_ids, component_ids
            )
        elif self.relation_mode == "site_donor_orientation":
            relation, full_mask = self.site_donor_orientation_relation_features(
                numeric, token_mask, role_ids, token_type_ids, component_ids
            )
        else:
            raise ValueError(f"Unsupported relation_mode: {self.relation_mode}")
        for layer in self.layers:
            encoded = layer(encoded, relation, full_mask)
        body = encoded[:, 1:]
        pool_mask = token_mask
        if self.relation_mode in {"site_donor_constrained", "site_donor_orientation"}:
            focus_mask = self.chemically_focused_token_mask(token_mask, role_ids, token_type_ids)
            has_focus = focus_mask.any(dim=1, keepdim=True)
            pool_mask = torch.where(has_focus, focus_mask, token_mask)
        logits = self.pool_score(body).squeeze(-1)
        logits = logits.masked_fill(~pool_mask, torch.finfo(logits.dtype).min)
        weights = torch.softmax(logits, dim=1).unsqueeze(-1)
        pooled = (body * weights).sum(dim=1)
        return self.head(torch.cat([encoded[:, 0], pooled], dim=-1)).squeeze(-1)


def make_loader(bundle: AtomRelationTensorBundle, batch_size: int, shuffle: bool) -> DataLoader:
    ds = TensorDataset(
        bundle.x,
        bundle.numeric,
        bundle.token_mask,
        bundle.role_ids,
        bundle.token_type_ids,
        bundle.atomic_numbers,
        bundle.component_ids,
        bundle.y,
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def train_epoch(model: nn.Module, loader: DataLoader, optimizer, device: str, pos_weight: torch.Tensor) -> float:
    model.train()
    losses = []
    for x, numeric, token_mask, role_ids, token_type_ids, atomic_numbers, component_ids, y in loader:
        pred = model(
            x.to(device),
            numeric.to(device),
            token_mask.to(device),
            role_ids.to(device),
            token_type_ids.to(device),
            atomic_numbers.to(device),
            component_ids.to(device),
        )
        loss = F.binary_cross_entropy_with_logits(pred, y.to(device), pos_weight=pos_weight)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses)) if losses else float("nan")


@torch.no_grad()
def predict(model: nn.Module, bundle: AtomRelationTensorBundle, batch_size: int, device: str) -> np.ndarray:
    model.eval()
    preds = []
    for x, numeric, token_mask, role_ids, token_type_ids, atomic_numbers, component_ids, _ in make_loader(
        bundle, batch_size, shuffle=False
    ):
        raw = model(
            x.to(device),
            numeric.to(device),
            token_mask.to(device),
            role_ids.to(device),
            token_type_ids.to(device),
            atomic_numbers.to(device),
            component_ids.to(device),
        )
        preds.append(torch.sigmoid(raw).detach().cpu().numpy())
    return np.concatenate(preds, axis=0) if preds else np.array([], dtype=float)


def model_tag(conformer_feature_mode: str, relation_mode: str) -> str:
    if relation_mode == "same_component":
        suffix = "relation_attention"
    elif relation_mode == "same_component_role":
        suffix = "role_relation_attention"
    elif relation_mode == "site_donor_constrained":
        suffix = "site_donor_constrained_attention"
    elif relation_mode == "site_donor_orientation":
        suffix = "site_donor_orientation_attention"
    else:
        raise ValueError(f"Unsupported relation_mode: {relation_mode}")
    if conformer_feature_mode == "mean":
        return f"orbmol_v2_atom_{suffix}"
    return f"orbmol_v2_atom_{conformer_feature_mode}_{suffix}"


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
        conformer_feature_mode=args.conformer_feature_mode,
    )
    val_bundle = build_tensor_bundle(
        store=store,
        part="val",
        target_frame=frames["val"],
        target_column=bundle.target_column,
        max_rows=args.max_eval_rows,
        max_tokens=args.max_tokens,
        conformer_feature_mode=args.conformer_feature_mode,
    )
    test_bundle = build_tensor_bundle(
        store=store,
        part="test",
        target_frame=frames["test"],
        target_column=bundle.target_column,
        max_rows=args.max_eval_rows,
        max_tokens=args.max_tokens,
        conformer_feature_mode=args.conformer_feature_mode,
    )

    y_train = train_bundle.y.numpy()
    positives = float(y_train.sum())
    negatives = float(len(y_train) - positives)
    pos_weight = torch.tensor([negatives / max(positives, 1.0)], dtype=torch.float32, device=args.device)

    model = Orb3DAtomRelationAttentionModel(
        orb_dim=int(train_bundle.x.shape[-1]),
        numeric_dim=int(train_bundle.numeric.shape[-1]),
        hidden_dim=args.hidden_dim,
        num_roles=16,
        num_token_types=max(TOKEN_TYPE_IDS.values()) + 1,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        dropout=args.dropout,
        relation_mode=args.relation_mode,
    ).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loader = make_loader(train_bundle, args.batch_size, shuffle=True)

    best_state = None
    best_eval = -np.inf
    bad_epochs = 0
    history = []
    for epoch in range(args.epochs):
        loss = train_epoch(model, loader, optimizer, args.device, pos_weight)
        val_pred = predict(model, val_bundle, args.batch_size, args.device)
        val_true = val_bundle.y.numpy().astype(float)
        val_best = best_threshold_accuracy(val_true, val_pred)
        val_metrics = classification_metrics(val_true, val_pred, threshold=val_best["best_threshold"])
        eval_score = float(val_best["best_accuracy"])
        history.append(
            {
                "epoch": epoch,
                "train_loss": loss,
                "eval_best_accuracy": val_best["best_accuracy"],
                "eval_best_threshold": val_best["best_threshold"],
                "eval_auroc": val_metrics["auroc"],
                "eval_auprc": val_metrics["auprc"],
            }
        )
        if eval_score > best_eval + 1e-7:
            best_eval = eval_score
            bad_epochs = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad_epochs += 1
        print(
            f"[epoch] {epoch + 1}/{args.epochs} loss={loss:.6f} "
            f"val_best_acc={val_best['best_accuracy']:.6f} "
            f"val_auroc={val_metrics['auroc']:.6f} threshold={val_best['best_threshold']:.2f}",
            flush=True,
        )
        if epoch + 1 >= 20 and bad_epochs >= args.patience:
            print(f"[early_stop] epoch={epoch + 1} best_val_acc={best_eval:.6f}", flush=True)
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    val_pred = predict(model, val_bundle, args.batch_size, args.device)
    val_true = val_bundle.y.numpy().astype(float)
    threshold = best_threshold_accuracy(val_true, val_pred)["best_threshold"]
    test_pred = predict(model, test_bundle, args.batch_size, args.device)
    test_true = test_bundle.y.numpy().astype(float)
    metrics = classification_metrics(test_true, test_pred, threshold=threshold)
    metrics.update(best_threshold_accuracy(test_true, test_pred))

    tag = model_tag(args.conformer_feature_mode, args.relation_mode)
    val_pred_path = args.out_dir / f"aho_major__{tag}__validation_predictions.csv"
    pd.DataFrame(
        {
            "row_index": val_bundle.row_index,
            "y_true": val_true,
            "y_prob": val_pred,
            "y_pred": (val_pred >= threshold).astype(int),
        }
    ).to_csv(val_pred_path, index=False)
    pred_path = args.out_dir / f"aho_major__{tag}__predictions.csv"
    pd.DataFrame(
        {
            "row_index": test_bundle.row_index,
            "y_true": test_true,
            "y_prob": test_pred,
            "y_pred": (test_pred >= threshold).astype(int),
        }
    ).to_csv(pred_path, index=False)
    metrics.update(
        {
            "task": "aho_major",
            "split": args.split,
            "model": tag,
            "feature": f"atom_token_attention_with_{args.relation_mode}_distance_bias_{args.conformer_feature_mode}_orbmol_v2",
            "conformer_feature_mode": args.conformer_feature_mode,
            "relation_mode": args.relation_mode,
            "n_train": int(len(train_bundle.y)),
            "n_val": int(len(val_bundle.y)),
            "atom_token_dir": str(args.atom_token_dir),
            "validation_prediction_path": str(val_pred_path),
            "prediction_path": str(pred_path),
            "notes": (
                "Standalone OrbMol-v2 atom-token attention with "
                f"{args.relation_mode} atom-pair 3D relation bias computed inside "
                "self-attention; no ChemAHNet features, logits, labels, residuals, "
                "or predictions."
            ),
        }
    )
    pd.DataFrame([metrics]).to_csv(args.out_dir / "metrics.csv", index=False)
    pd.DataFrame(history).to_csv(args.out_dir / "history.csv", index=False)
    torch.save({"model_state_dict": model.state_dict(), "config": vars(args)}, args.out_dir / f"aho_major__{tag}.pt")
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
