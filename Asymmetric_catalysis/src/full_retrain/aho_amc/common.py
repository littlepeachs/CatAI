from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping

import torch


AHO_ROOT = Path(os.environ.get("CATAI_WORKSPACE_ROOT", "/root/autodl-tmp/AHO"))
MOMA_ROOT = Path(os.environ.get("CATAI_MOMA_ROOT", "/root/autodl-tmp/MoMa"))
ORB_CHECKPOINT_ROOT = Path(
    os.environ.get("CATAI_ORB_CHECKPOINT_ROOT", "/root/autodl-tmp/checkpoints_orb")
)
AMC_OUTPUT_ROOT = AHO_ROOT / "reproduction_outputs" / "aho_amc"
RAW_ORB_CHECKPOINT = ORB_CHECKPOINT_ROOT / "orb-v2-20241011.ckpt"
MOMA_BASE_CHECKPOINT = ORB_CHECKPOINT_ROOT / "orb_v2_base_moma_checkpoint.pt"
MODULE_ROOT = ORB_CHECKPOINT_ROOT / "modules"
TOKEN_ROOT = (
    AHO_ROOT
    / "reproduction_outputs"
    / "full_retrain"
    / "orb3d_atom_tokens_aho_major_ligand_metal_h100_c3"
)
CONFORMER_CACHE = TOKEN_ROOT / "orb_finetune_conformers_v2_three.joblib"

MODULE_NAMES = (
    "mp_eform",
    "mp_bandgap",
    "mp_energy_above_hull",
    "mp_efermi",
    "n_th_cond",
    "p_e_cond",
    "n_e_cond",
    "p_th_cond",
    "mp_gvrh",
    "mp_kvrh",
    "castelli_eform",
    "jarvis_gvrh",
    "jarvis_kvrh",
    "jarvis_bandgap",
    "jarvis_eform",
    "jarvis_total_energy",
    "adsorption_energy",
    "surface_energy",
    "Ea_log_wo_cell",
    "gibbs_free_energy",
    "Electronic_E",
    "Metal_q",
    "HL_Gap",
    "dos_fermi_level",
    "oqmd_delta_e",
    "oqmd_stability",
    "oqmd_band_gap",
    "qmof_bg",
    "qmof_e",
    "cepdb_pce",
)


def ensure_moma_import_path() -> None:
    moma = str(MOMA_ROOT)
    if moma not in sys.path:
        sys.path.insert(0, moma)


def sha256_file(path: Path, *, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_schema(state_dict: Mapping[str, torch.Tensor]) -> dict[str, dict[str, Any]]:
    return {
        key: {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
        }
        for key, value in state_dict.items()
    }


def load_moma_checkpoint(path: Path) -> dict[str, Any]:
    ensure_moma_import_path()
    return torch.load(path, map_location="cpu", weights_only=False)


def map_moma_backbone_state(
    checkpoint_or_state: Mapping[str, Any],
) -> dict[str, torch.Tensor]:
    state = checkpoint_or_state.get("state_dict", checkpoint_or_state)
    result: dict[str, torch.Tensor] = {}
    for original_key, value in state.items():
        key = original_key.removeprefix("module.")
        if not key.startswith("backbone."):
            continue
        result[key.removeprefix("backbone.")] = value
    return result


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, ensure_ascii=False)
        handle.write("\n")
