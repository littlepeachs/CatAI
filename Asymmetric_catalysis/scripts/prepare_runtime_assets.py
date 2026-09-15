#!/usr/bin/env python3
"""Materialize the six fused JMP backbones and audited token-cache views.

The archive deliberately does not duplicate the 24 GB Pi-CFM checkpoint pool
or the existing conformation caches.  This command reconstructs one task-local
parameter-space AMC checkpoint from the sealed coefficients and links the
immutable three-conformer atom-token payloads only after hash validation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


HERE = Path(__file__).resolve().parent
ARCHIVE = HERE.parent
DEFAULT_WORKSPACE = HERE.parents[2]
TASKS = (
    "aho_ddg",
    "cpa_main",
    "cpa_transformation",
    "cpa_imine",
    "cpa_thiol",
    "cpa_catalyst",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace-root", type=Path, default=DEFAULT_WORKSPACE)
    parser.add_argument("--runtime-root", type=Path)
    parser.add_argument("--picfm-root", type=Path, default=Path("/root/autodl-tmp/Pi-CFM"))
    parser.add_argument("--moma-root", type=Path, default=Path("/root/autodl-tmp/MoMa"))
    parser.add_argument("--freeze-root", type=Path)
    parser.add_argument("--cpa-conformer-root", type=Path)
    parser.add_argument("--aho-conformer-cache", type=Path)
    parser.add_argument("--catalyst-token-source", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--cpu-threads", type=int, default=8)
    return parser.parse_args()


def execute(command: list[str], *, env: dict[str, str]) -> None:
    print("+ " + " ".join(command), flush=True)
    subprocess.run(command, check=True, env=env, cwd=ARCHIVE)


def main() -> None:
    args = parse_args()
    workspace = args.workspace_root.resolve()
    runtime = (
        args.runtime_root.resolve()
        if args.runtime_root is not None
        else workspace / "outputs/catai_jmp_amc_runtime_assets"
    )
    if runtime.exists():
        raise FileExistsError(
            f"{runtime} already exists; choose a new --runtime-root so an audited "
            "runtime is never silently overwritten"
        )
    if not args.picfm_root.is_dir() or not args.moma_root.is_dir():
        raise FileNotFoundError("Pi-CFM checkpoint pool or MoMa runtime is missing")
    selection_dir = ARCHIVE / "configs/amc_selections"
    cpa_conformers = (
        args.cpa_conformer_root.resolve()
        if args.cpa_conformer_root is not None
        else workspace / "conformer_caches"
    )
    aho_conformers = (
        args.aho_conformer_cache.resolve()
        if args.aho_conformer_cache is not None
        else cpa_conformers / "aho_ddg.joblib"
    )
    old_freeze = (
        args.freeze_root.resolve()
        if args.freeze_root is not None
        else workspace / "freeze"
    )
    catalyst_assets = (
        args.catalyst_token_source.resolve()
        if args.catalyst_token_source is not None
        else workspace / "catalyst_k2/fused_atom_tokens/cpa_catalyst"
    )
    env = os.environ.copy()
    python_paths = [str(ARCHIVE / "src"), str(workspace)]
    if env.get("PYTHONPATH"):
        python_paths.append(env["PYTHONPATH"])
    env.update(
        {
            "PYTHONPATH": os.pathsep.join(python_paths),
            "CATAI_WORKSPACE_ROOT": str(workspace),
            "CATAI_MOMA_ROOT": str(args.moma_root.resolve()),
            "CATAI_PICFM_ROOT": str(args.picfm_root.resolve()),
        }
    )
    records: list[dict[str, object]] = []
    for task in TASKS:
        if task == "aho_ddg":
            checkpoint_dir = runtime / "aho_materialized/jmp/checkpoints"
            conformer_cache = aho_conformers
            token_source = old_freeze / "features/aho_ddg"
            token_target = runtime / "token_views/aho_ddg"
        elif task == "cpa_catalyst":
            checkpoint_dir = runtime / "catalyst_k2_materialized/jmp/checkpoints"
            conformer_cache = cpa_conformers / f"{task}.joblib"
            token_source = catalyst_assets
            token_target = (
                runtime
                / "catalyst_k2_materialized/jmp/fused_atom_tokens/cpa_catalyst"
            )
        else:
            checkpoint_dir = runtime / "cpa_materialized/jmp/checkpoints"
            conformer_cache = cpa_conformers / f"{task}.joblib"
            token_source = old_freeze / f"features/cpa/{task}"
            token_target = runtime / f"token_views/{task}"
        for required in (selection_dir / f"{task}.json", conformer_cache, token_source):
            if not required.exists():
                raise FileNotFoundError(required)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        execute(
            [
                sys.executable,
                "-u",
                "-m",
                "full_retrain.aho_amc.fuse_jmp_checkpoints",
                "--task",
                task,
                "--selection-dir",
                str(selection_dir),
                "--picfm-root",
                str(args.picfm_root.resolve()),
                "--output-dir",
                str(checkpoint_dir),
                "--conformer-cache",
                str(conformer_cache),
                "--device",
                args.device,
                "--cpu-threads",
                str(args.cpu_threads),
            ],
            env=env,
        )
        checkpoint = checkpoint_dir / f"{task}_jmp_amc_fused.pt"
        execute(
            [
                sys.executable,
                str(HERE / "prepare_compatible_token_cache.py"),
                "--source",
                str(token_source),
                "--checkpoint",
                str(checkpoint),
                "--output",
                str(token_target),
            ],
            env=env,
        )
        records.append(
            {
                "task": task,
                "selection": str(selection_dir / f"{task}.json"),
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": sha256(checkpoint),
                "token_view": str(token_target),
                "token_manifest_sha256": sha256(token_target / "manifest.json"),
            }
        )
    payload = {
        "schema_version": 1,
        "tasks": list(TASKS),
        "parameter_space_amc": True,
        "prediction_ensemble": False,
        "three_conformer_token_views": True,
        "records": records,
    }
    (runtime / "runtime_manifest.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"Prepared audited runtime assets at {runtime}", flush=True)


if __name__ == "__main__":
    main()
