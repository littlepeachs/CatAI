#!/usr/bin/env python3
"""Run one CatAI historical-recipe point over the six frozen tasks.

Each run keeps the task-specific AMC backbone/token representation and final
J185/A* topology fixed.  The five non-Catalyst tasks start from the sealed
historical readout and repeat its final trainable scope with the configured
seed. Catalyst is rebuilt with the archived two-stage K2/H8 recipe because
its historical binary checkpoint was not retained.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd


HERE = Path(__file__).resolve().parent
ARCHIVE = HERE.parent
DEFAULT_WORKSPACE = HERE.parents[2]
PROJECT = DEFAULT_WORKSPACE
FREEZE = PROJECT / "reproduction_outputs/catai_attention_6task_freeze_20260818_v2"
RUNTIME = PROJECT / "outputs/catai_jmp_amc_runtime_assets"
VENDOR: Path | None = None
OUT_ROOT = PROJECT / "outputs/catai_jmp_amc_historical"
CONFORMERS = (
    PROJECT
    / "reproduction_outputs/cpa_amc_fuse_locked_orra_20260728/conformer_caches"
)
AHO_CONFORMERS = (
    PROJECT
    / "reproduction_outputs/full_retrain/"
    "orb3d_atom_tokens_aho_major_ligand_metal_h100_c3/"
    "orb_finetune_conformers_v2_three.joblib"
)
HISTORICAL_SEED = 20260725
DISPLAY = {
    "aho_ddg": "AHO ΔΔG‡",
    "cpa_main": "CPA Main",
    "cpa_transformation": "CPA Transformation",
    "cpa_imine": "CPA Imine",
    "cpa_thiol": "CPA Thiol",
    "cpa_catalyst": "CPA Catalyst",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one fixed historical CatAI reaction-selectivity point."
    )
    parser.add_argument("--seed", type=int, default=HISTORICAL_SEED)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument(
        "--task",
        action="append",
        choices=tuple(DISPLAY),
        default=[],
        help="Optional task to run; repeat for multiple tasks. Default: all six.",
    )
    parser.add_argument("--workspace-root", type=Path, default=DEFAULT_WORKSPACE)
    parser.add_argument("--runtime-root", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--freeze-root", type=Path)
    parser.add_argument("--vendor-root", type=Path)
    parser.add_argument(
        "--moma-root", type=Path, default=Path("/root/autodl-tmp/MoMa")
    )
    parser.add_argument(
        "--picfm-root", type=Path, default=Path("/root/autodl-tmp/Pi-CFM")
    )
    parser.add_argument("--cpa-conformer-root", type=Path)
    parser.add_argument("--aho-conformer-cache", type=Path)
    return parser.parse_args()


def display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT.resolve()))
    except ValueError:
        return str(path.resolve())


def run(command: list[str], output: Path, env: dict[str, str]) -> None:
    output.mkdir(parents=True, exist_ok=True)
    log_path = output / "train.log"
    with log_path.open("w", encoding="utf-8") as log:
        subprocess.run(
            command,
            cwd=PROJECT,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
        )


def base_command(
    *,
    task: str,
    token_dir: Path,
    checkpoint: Path,
    conformer_cache: Path,
    output: Path,
    seed: int,
) -> list[str]:
    return [
        sys.executable,
        "-u",
        "-m",
        "full_retrain.train_unified_orra_orb_finetune",
        "--task",
        task,
        "--mapping-task",
        "aho_major" if task == "aho_ddg" else task,
        "--atom-token-dir",
        str(token_dir),
        "--orb-checkpoint",
        str(checkpoint),
        "--backbone-family",
        "jmp",
        "--backbone-feature-source",
        "token_cache",
        "--conformer-cache",
        str(conformer_cache),
        "--out-dir",
        str(output),
        "--device",
        "cuda:0",
        "--conformer-feature-mode",
        "mean_std_max",
        "--conformer-aggregation",
        "fixed",
        "--num-heads",
        "8",
        "--max-tokens",
        "128",
        "--orb-lr",
        "0",
        "--orb-warmup-epochs",
        "0",
        "--orb-weight-decay",
        "0",
        "--orb-gradient-clip-norm",
        "0",
        "--orb-train-scope",
        "frozen",
        "--orb-module-mode",
        "eval",
        "--readout-module-mode",
        "train",
        "--readout-gradient-clip-norm",
        "5",
        "--lr-factor",
        "0.5",
        "--min-lr",
        "1e-8",
        "--density-quantile",
        "0.95",
        "--unseen-density-floor",
        "0",
        "--ood-gate-mode",
        "exponential",
        "--gate-residual-mode",
        "centered_ood",
        "--ood-shift-scale",
        "1",
        "--gate-regularization",
        "0",
        "--adam-beta1",
        "0.9",
        "--adam-beta2",
        "0.999",
        "--adam-eps",
        "1e-8",
        "--amp",
        "--seed",
        str(seed),
    ]


def cpa_architecture() -> list[str]:
    return [
        "--hidden-dim",
        "128",
        "--num-self-layers",
        "1",
        "--num-role-layers",
        "1",
        "--cross-attention-view",
        "roles_types",
        "--dropout",
        "0.10",
        "--regression-head-hidden-dims",
        "256,128,64",
        "--regression-head-dropout",
        "0.05",
        "--smiles-sequence-adapter",
        "--smiles-sequence-hidden-dim",
        "64",
        "--smiles-sequence-num-layers",
        "1",
        "--smiles-sequence-num-heads",
        "4",
        "--smiles-sequence-dropout",
        "0.10",
        "--smiles-sequence-residual-multiplier",
        "3",
    ]


def neutral_gate() -> list[str]:
    return [
        "--density-ratio-cap",
        "100",
        "--density-train-mode",
        "exact_seen",
        "--ood-threshold",
        "1000000",
        "--ood-scale-init",
        "0.2",
        "--ood-base-shift",
        "0",
        "--residual-global-center-mix",
        "0",
        "--freeze-gate",
        "--sample-prior-mode",
        "global",
    ]


def run_standard_task(task: str, seed: int, seed_root: Path, env: dict[str, str]) -> Path:
    if task == "aho_ddg":
        token_dir = RUNTIME / "token_views/aho_ddg"
        backbone = RUNTIME / "aho_materialized/jmp/checkpoints/aho_ddg_jmp_amc_fused.pt"
        init = FREEZE / "checkpoints/aho_ddg_j185/model.pt"
        output = seed_root / task
        command = base_command(
            task=task,
            token_dir=token_dir,
            checkpoint=backbone,
            conformer_cache=AHO_CONFORMERS,
            output=output,
            seed=seed,
        )
        command += [
            "--init-readout-checkpoint",
            str(init),
            "--init-readout-mode",
            "full",
            "--hidden-dim",
            "128",
            "--num-self-layers",
            "2",
            "--num-role-layers",
            "1",
            "--cross-attention-view",
            "roles_types",
            "--dropout",
            "0.12",
            "--ddg-sign-magnitude",
            "--ddg-sign-magnitude-hidden-dim",
            "128",
            "--ddg-sign-magnitude-weight",
            "0",
            "--ddg-sign-magnitude-huber-delta",
            "0.75",
            "--ddg-sign-magnitude-residual-multiplier",
            "1.75",
            "--ddg-sign-bce-magnitude-strength",
            "0.5",
            "--ddg-sign-bce-magnitude-scale",
            "1",
            "--ddg-residual-calibrator",
            "--readout-train-scope",
            "ddg_sign_magnitude_and_residual_calibrator",
            "--batch-size",
            "128",
            "--epochs",
            "100",
            "--patience",
            "20",
            "--lr",
            "3e-4",
            "--weight-decay",
            "1e-4",
            "--lr-patience",
            "7",
            "--density-train-mode",
            "exact_seen",
            "--ood-threshold",
            "1",
            "--ood-scale-init",
            "0.177463099360466",
            "--ood-base-shift",
            "1",
            "--residual-global-center-mix",
            "0",
            "--sample-prior-mode",
            "global",
            "--loss",
            "mse",
            "--huber-delta",
            "0.75",
            "--selection-metric",
            "ddg_gate_balance",
            "--scheduler-metric",
            "ddg_gate_balance",
            "--ddg-r2-gate",
            "0.58",
            "--ddg-mae-gate",
            "0.80",
        ]
        run(command, output, env)
        return output

    token_dir = RUNTIME / f"token_views/{task}"
    backbone = RUNTIME / f"cpa_materialized/jmp/checkpoints/{task}_jmp_amc_fused.pt"
    init = FREEZE / f"checkpoints/cpa/{task}.pt"
    output = seed_root / task
    command = base_command(
        task=task,
        token_dir=token_dir,
        checkpoint=backbone,
        conformer_cache=CONFORMERS / f"{task}.joblib",
        output=output,
        seed=seed,
    )
    command += cpa_architecture()
    command += ["--init-readout-checkpoint", str(init), "--init-readout-mode", "full"]

    if task == "cpa_main":
        command += neutral_gate() + [
            "--readout-train-scope",
            "all",
            "--batch-size",
            "256",
            "--epochs",
            "59",
            "--patience",
            "1000",
            "--epoch-ema-decay",
            "0.90",
            "--lr",
            "6e-4",
            "--weight-decay",
            "1e-4",
            "--lr-patience",
            "1000",
            "--loss",
            "huber",
            "--huber-delta",
            "0.75",
            "--selection-metric",
            "r2",
            "--scheduler-metric",
            "r2",
            "--final-retrain-train-plus-val",
        ]
    elif task == "cpa_transformation":
        command += neutral_gate() + [
            "--readout-train-scope",
            "regression_output",
            "--batch-size",
            "128",
            "--epochs",
            "40",
            "--patience",
            "12",
            "--lr",
            "4e-5",
            "--weight-decay",
            "0",
            "--lr-patience",
            "5",
            "--loss",
            "mse",
            "--huber-delta",
            "0.75",
            "--selection-metric",
            "r2",
            "--scheduler-metric",
            "r2",
        ]
    elif task == "cpa_imine":
        command += neutral_gate() + [
            "--readout-train-scope",
            "regression_output",
            "--batch-size",
            "128",
            "--epochs",
            "40",
            "--patience",
            "12",
            "--lr",
            "1e-5",
            "--weight-decay",
            "0",
            "--lr-patience",
            "5",
            "--loss",
            "mse",
            "--huber-delta",
            "0.75",
            "--selection-metric",
            "r2",
            "--scheduler-metric",
            "r2",
            "--validation-mode",
            "group_column",
            "--group-val-column",
            "Imine",
            "--group-val-fraction",
            "0.25",
            "--group-val-strategy",
            "random",
        ]
    elif task == "cpa_thiol":
        command += [
            "--density-ratio-cap",
            "100",
            "--density-train-mode",
            "exact_seen",
            "--ood-threshold",
            "15.50552",
            "--ood-scale-init",
            "1",
            "--ood-base-shift",
            "0",
            "--residual-global-center-mix",
            "1",
            "--freeze-gate",
            "--sample-prior-mode",
            "train_quantile",
            "--prior-quantile",
            "0.20",
            "--readout-train-scope",
            "smiles_sequence_adapter",
            "--batch-size",
            "256",
            "--epochs",
            "28",
            "--patience",
            "15",
            "--lr",
            "6e-4",
            "--weight-decay",
            "1e-4",
            "--lr-patience",
            "7",
            "--loss",
            "huber",
            "--huber-delta",
            "0.75",
            "--selection-metric",
            "r2",
            "--scheduler-metric",
            "r2",
            "--validation-mode",
            "group_column",
            "--group-val-column",
            "Thiol",
            "--group-val-fraction",
            "0.34",
            "--group-val-strategy",
            "random",
        ]
    else:
        raise ValueError(task)
    run(command, output, env)
    return output


def run_catalyst(seed: int, seed_root: Path, env: dict[str, str]) -> Path:
    task = "cpa_catalyst"
    token_dir = RUNTIME / "catalyst_k2_materialized/jmp/fused_atom_tokens/cpa_catalyst"
    backbone = RUNTIME / "catalyst_k2_materialized/jmp/checkpoints/cpa_catalyst_jmp_amc_fused.pt"
    stage1 = seed_root / "cpa_catalyst_stage1"
    stage2 = seed_root / "cpa_catalyst_stage2"
    test = seed_root / task
    common = base_command(
        task=task,
        token_dir=token_dir,
        checkpoint=backbone,
        conformer_cache=CONFORMERS / "cpa_catalyst.joblib",
        output=stage1,
        seed=seed,
    )
    common += [
        "--hidden-dim",
        "128",
        "--num-self-layers",
        "1",
        "--num-role-layers",
        "1",
        "--cross-attention-view",
        "roles_types",
        "--dropout",
        "0.10",
        "--regression-head-hidden-dims",
        "256,128,64",
        "--regression-head-dropout",
        "0.05",
        "--batch-size",
        "256",
        "--patience",
        "1000",
        "--lr",
        "6e-4",
        "--weight-decay",
        "1e-4",
        "--lr-patience",
        "1000",
        "--density-ratio-cap",
        "10",
        "--unseen-density-floor",
        "0",
        "--ood-threshold",
        "1000000",
        "--ood-scale-init",
        "0.2",
        "--ood-base-shift",
        "0",
        "--freeze-gate",
        "--sample-prior-mode",
        "global",
        "--loss",
        "huber",
        "--huber-delta",
        "0.75",
        "--selection-metric",
        "r2",
        "--scheduler-metric",
        "r2",
        "--validation-mode",
        "group_column",
        "--group-val-column",
        "Catalyst",
        "--group-val-fraction",
        "0.20",
        "--group-val-strategy",
        "random",
        "--final-retrain-train-plus-val",
    ]
    stage1_command = common + [
        "--epochs",
        "6",
        "--readout-train-scope",
        "all",
        "--density-train-mode",
        "leave_one_component_out",
        "--residual-global-center-mix",
        "1",
        "--skip-test-evaluation",
    ]
    run(stage1_command, stage1, env)

    stage2_command = base_command(
        task=task,
        token_dir=token_dir,
        checkpoint=backbone,
        conformer_cache=CONFORMERS / "cpa_catalyst.joblib",
        output=stage2,
        seed=seed,
    )
    stage2_command += cpa_architecture() + [
        "--init-readout-checkpoint",
        str(stage1 / "model.pt"),
        "--init-readout-mode",
        "full",
        "--batch-size",
        "256",
        "--epochs",
        "11",
        "--patience",
        "1000",
        "--lr",
        "6e-4",
        "--weight-decay",
        "1e-4",
        "--lr-patience",
        "1000",
        "--density-ratio-cap",
        "10",
        "--density-train-mode",
        "exact_seen",
        "--ood-threshold",
        "1000000",
        "--ood-scale-init",
        "0.2",
        "--ood-base-shift",
        "0",
        "--residual-global-center-mix",
        "0",
        "--freeze-gate",
        "--sample-prior-mode",
        "global",
        "--loss",
        "huber",
        "--huber-delta",
        "0.75",
        "--selection-metric",
        "r2",
        "--scheduler-metric",
        "r2",
        "--validation-mode",
        "group_column",
        "--group-val-column",
        "Catalyst",
        "--group-val-fraction",
        "0.20",
        "--group-val-strategy",
        "random",
        "--final-retrain-train-plus-val",
        "--readout-train-scope",
        "smiles_sequence_adapter",
        "--smiles-sequence-catalyst-h8-augmentation",
        "--smiles-sequence-catalyst-h8-target-mode",
        "copy_target",
        "--smiles-sequence-catalyst-h8-target-quantile",
        "0.10",
        "--smiles-sequence-catalyst-h8-target-condition-columns",
        "Imine,Thiol",
        "--skip-test-evaluation",
    ]
    run(stage2_command, stage2, env)

    test_command = base_command(
        task=task,
        token_dir=token_dir,
        checkpoint=backbone,
        conformer_cache=CONFORMERS / "cpa_catalyst.joblib",
        output=test,
        seed=seed,
    )
    test_command += cpa_architecture() + [
        "--init-readout-checkpoint",
        str(stage2 / "model.pt"),
        "--init-readout-mode",
        "full",
        "--batch-size",
        "256",
        "--epochs",
        "0",
        "--patience",
        "1",
        "--lr",
        "0",
        "--weight-decay",
        "0",
        "--readout-gradient-clip-norm",
        "0",
        "--lr-patience",
        "1",
        "--density-ratio-cap",
        "10",
        "--density-train-mode",
        "exact_seen",
        "--ood-threshold",
        "1000000",
        "--ood-scale-init",
        "0.2",
        "--ood-base-shift",
        "0",
        "--residual-global-center-mix",
        "0",
        "--freeze-gate",
        "--sample-prior-mode",
        "global",
        "--loss",
        "huber",
        "--huber-delta",
        "0.75",
        "--selection-metric",
        "r2",
        "--scheduler-metric",
        "r2",
        "--validation-mode",
        "group_column",
        "--group-val-column",
        "Catalyst",
        "--group-val-fraction",
        "0.20",
        "--group-val-strategy",
        "random",
        "--final-retrain-train-plus-val",
        "--readout-train-scope",
        "smiles_sequence_adapter",
        "--readout-module-mode",
        "eval",
    ]
    run(test_command, test, env)
    return test


def completed(output: Path) -> bool:
    if not (output / "metrics.csv").is_file():
        return False
    frame = pd.read_csv(output / "metrics.csv")
    return len(frame) == 1 and bool(frame.iloc[0].get("test_evaluated", False))


def main() -> None:
    global PROJECT, FREEZE, RUNTIME, VENDOR, OUT_ROOT, CONFORMERS, AHO_CONFORMERS
    args = parse_args()
    PROJECT = args.workspace_root.resolve()
    FREEZE = (
        args.freeze_root.resolve()
        if args.freeze_root is not None
        else PROJECT
        / "reproduction_outputs/catai_attention_6task_freeze_20260818_v2"
    )
    RUNTIME = (
        args.runtime_root.resolve()
        if args.runtime_root is not None
        else PROJECT / "outputs/catai_jmp_amc_runtime_assets"
    )
    VENDOR = args.vendor_root.resolve() if args.vendor_root is not None else None
    OUT_ROOT = (
        args.output_root.resolve()
        if args.output_root is not None
        else PROJECT / "outputs/catai_jmp_amc_historical"
    )
    CONFORMERS = (
        args.cpa_conformer_root.resolve()
        if args.cpa_conformer_root is not None
        else PROJECT
        / "reproduction_outputs/cpa_amc_fuse_locked_orra_20260728/conformer_caches"
    )
    AHO_CONFORMERS = (
        args.aho_conformer_cache.resolve()
        if args.aho_conformer_cache is not None
        else PROJECT
        / "reproduction_outputs/full_retrain/"
        "orb3d_atom_tokens_aho_major_ligand_metal_h100_c3/"
        "orb_finetune_conformers_v2_three.joblib"
    )
    seed_root = OUT_ROOT / f"historical_seed_{args.seed}"
    seed_root.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    python_paths = [str(ARCHIVE / "src"), str(PROJECT)]
    if VENDOR is not None:
        python_paths.insert(0, str(VENDOR))
    if env.get("PYTHONPATH"):
        python_paths.append(env["PYTHONPATH"])
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": str(args.gpu),
            "PYTHONHASHSEED": str(args.seed),
            "PYTHONPATH": os.pathsep.join(python_paths),
            "CATAI_WORKSPACE_ROOT": str(PROJECT),
            "CATAI_MOMA_ROOT": str(args.moma_root.resolve()),
            "CATAI_PICFM_ROOT": str(args.picfm_root.resolve()),
            "OMP_NUM_THREADS": "2",
            "MKL_NUM_THREADS": "2",
            "OPENBLAS_NUM_THREADS": "2",
            "NUMEXPR_NUM_THREADS": "2",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        }
    )
    rows: list[dict[str, object]] = []
    selected_tasks = args.task or list(DISPLAY)
    for task in selected_tasks:
        output = seed_root / task
        if not completed(output):
            print(f"[CatAI] seed={args.seed} task={task}", flush=True)
            if task == "cpa_catalyst":
                output = run_catalyst(args.seed, seed_root, env)
            else:
                output = run_standard_task(task, args.seed, seed_root, env)
        metric = pd.read_csv(output / "metrics.csv").iloc[0]
        rows.append(
            {
                "task": DISPLAY[task],
                "internal_task": task,
                "method": "CatAI+Attention",
                "seed": args.seed,
                "n": int(metric["n"]),
                "r2": float(metric["r2"]),
                "rmse": float(metric["rmse"]),
                "mae": float(metric["mae"]),
                "checkpoint": display_path(output / "model.pt"),
                "prediction_path": display_path(output / "test_predictions.csv"),
            }
        )
        pd.DataFrame(rows).to_csv(
            seed_root / "metrics.partial.csv", index=False, float_format="%.15g"
        )
        print(f"[done] {task} MAE={float(metric['mae']):.6f}", flush=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(seed_root / "metrics.csv", index=False, float_format="%.15g")
    (seed_root / "protocol.json").write_text(
        json.dumps(
            {
                "seed": args.seed,
                "gpu": args.gpu,
                "tasks": selected_tasks,
                "fixed_test_partitions": True,
                "fixed_task_specific_amc_weights": True,
                "fixed_j185_or_a_star_topology": True,
                "shared_locked_initialization": True,
                "protocol": (
                    "single historical-recipe readout continuation; Catalyst archived "
                    "two-stage K2/H8 replay"
                ),
                "replicate_aggregation": False,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(frame.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
