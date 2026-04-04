#!/usr/bin/env python3
"""Active learning utilities extracted from the notebook for batch execution."""

from __future__ import annotations
import pickle
import argparse
import json
import os
import random
import sys
import warnings
from typing import Any, Dict, List, Optional

BENCHMARK_ROOT = os.path.dirname(os.path.abspath(__file__))
if BENCHMARK_ROOT not in sys.path:
    sys.path.insert(0, BENCHMARK_ROOT)

import matplotlib.pyplot as plt
import pandas as pd
import torch

from cgcnn_model.cgcnn.data_process import CIFData

from al_pipeline.config import ALConfig
from al_pipeline.models.cgcnn import CGCNNSubsetTrainer
from al_pipeline.models.dpp import DPPSubsetTrainer
from al_pipeline.models.escn import ESCNSubsetTrainer, EscnCIFDataset
from al_pipeline.models.schnet import SchnetSubsetTrainer
from al_pipeline.models.gemnet_oc import GemnetOCSubsetTrainer
from al_pipeline.models.et import ETSubsetTrainer
from al_pipeline.models.catai import CatAISubsetTrainer

import time

warnings.filterwarnings("ignore", category=UserWarning)

SELECTED_SEEDS_PATH = os.path.join(BENCHMARK_ROOT, "selected_40_random_seeds.pkl")


def _load_selected_trial_seeds() -> List[int]:
    with open(SELECTED_SEEDS_PATH, "rb") as seed_file:
        seeds = pickle.load(seed_file)

    if not isinstance(seeds, list) or not all(isinstance(seed, int) for seed in seeds):
        raise ValueError(f"Invalid seed list in {SELECTED_SEEDS_PATH}")

    return seeds


def _prepare_backend(cfg: ALConfig):
    model_type = (cfg.model_type or "cgcnn").lower()
    task_name = cfg.label_dir.split('/')[-1]
    process_dir = None
    if task_name == '[C]':
        process_dir = './processed_c'
    elif task_name == '[O]':
        process_dir = './processed_o'
    else:
        raise ValueError(f"Unsupported active-learning task: {task_name}. Only [C] and [O] are supported.")

    if model_type == "cgcnn":
        dataset = CIFData(
            label_dir=cfg.label_dir,
            properties=cfg.properties,
            struct_dir=cfg.structure_dir,
            aug=False,
        )
        trainer_cls = CGCNNSubsetTrainer
    elif model_type == "escn":
        dataset = EscnCIFDataset(
            label_dir=cfg.label_dir,
            structure_dir=cfg.structure_dir,
            properties=cfg.properties,
            aug=False,
        )
        trainer_cls = ESCNSubsetTrainer
    elif model_type == "dpp":
        dataset = EscnCIFDataset(
            label_dir=cfg.label_dir,
            structure_dir=cfg.structure_dir,
            properties=cfg.properties,
            aug=False,
        )
        trainer_cls = DPPSubsetTrainer
    elif model_type == "schnet":
        dataset = EscnCIFDataset(
            label_dir=cfg.label_dir,
            structure_dir=cfg.structure_dir,
            properties=cfg.properties,
            aug=False,
        )
        trainer_cls = SchnetSubsetTrainer
    elif model_type == "gemnet_oc":
        dataset = EscnCIFDataset(
            label_dir=cfg.label_dir,
            structure_dir=cfg.structure_dir,
            properties=cfg.properties,
            aug=False,
        )
        trainer_cls = GemnetOCSubsetTrainer
    elif model_type == "et":
        dataset = EscnCIFDataset(
            label_dir=cfg.label_dir,
            structure_dir=cfg.structure_dir,
            properties=cfg.properties,
            aug=False,
        )
        trainer_cls = ETSubsetTrainer
    elif model_type == "catai":
        dataset = EscnCIFDataset(
            label_dir=cfg.label_dir,
            structure_dir=cfg.structure_dir,
            properties=cfg.properties,
            aug=False,
            process_dir = process_dir,
        )
        trainer_cls = CatAISubsetTrainer
    else:
        raise ValueError(f"Unsupported model_type: {cfg.model_type}")
    return dataset, trainer_cls


def active_learning_loop(cfg: ALConfig):
    os.makedirs(cfg.save_dir, exist_ok=True)
    cfg.out_size = len(cfg.properties)

    dataset, trainer_cls = _prepare_backend(cfg)

    total_samples = len(dataset)
    if cfg.val_size < 0:
        raise ValueError("val_size must be non-negative")
    if cfg.val_size > 0 and total_samples <= cfg.val_size + cfg.init_size:
        raise ValueError(
            f"Too few samples: {total_samples} <= val_size({cfg.val_size}) + init_size({cfg.init_size})"
        )

    df_all = dataset.id_prop_data
    df_all.to_csv(os.path.join(cfg.save_dir, "df_all.csv"), index=False)
    
    # Sampling 40 indices fixed
    sorted_indices = df_all.sort_values(by='y_relax').index.tolist()
    
    length_of_sorted_indices = len(sorted_indices)
    # Fixed sampling of 40 indices from the interval between 0.4 and 0.6 of the sorted indices length
    start_idx = int(length_of_sorted_indices * 0.3)
    end_idx = int(length_of_sorted_indices * 0.7)
    interval_length = end_idx - start_idx
    if interval_length < 40:
        raise ValueError(f"Interval between 0.4 and 0.6 of sorted indices is too small ({interval_length}) to sample 40 indices.")
    step = interval_length // 40
    sampled_indices = sorted_indices[start_idx:start_idx + step * 40:step][:40]

    rng = random.Random(cfg.random_seed)
    all_indices = list(range(total_samples))
    # Remove sampled_indices from all_indices
    all_indices = [idx for idx in all_indices if idx not in sampled_indices]
    
    rng.shuffle(all_indices)
    val_indices = sorted(all_indices[: cfg.val_size]) if cfg.val_size > 0 else []
    pool_indices = all_indices[cfg.val_size:]

    # Save sampled_indices to a text file in cfg.save_dir
    sampled_indices_path = os.path.join(cfg.save_dir, "sampled_indices.txt")
    with open(sampled_indices_path, 'w') as f:
        f.write(','.join(map(str, sampled_indices)))
    print(f"Sampled indices saved to {sampled_indices_path}")
    
    sampled_indices.extend(sorted(pool_indices[: cfg.init_size]))
    labeled = sampled_indices
    
    unlabeled = sorted(pool_indices[cfg.init_size:])
    
    prop = cfg.properties[0]
    pool_df = df_all.iloc[pool_indices]
    true_best_value = float(pool_df[prop].max() if cfg.maximize_best else pool_df[prop].min())
    true_best_indices = set(pool_df.index[pool_df[prop] == true_best_value].tolist())

    print(f"Validation set size = {len(val_indices)}")

    vals0 = df_all[prop].iloc[labeled].to_numpy()
    best0 = float(vals0.max() if cfg.maximize_best else vals0.min())
    history_best = [best0]

    discovered_round = 0 if any(i in true_best_indices for i in labeled) else None
    discovered_found = discovered_round is not None
    round_summaries = []

    
    for r in range(1, cfg.rounds + 1):
        print(f"=== Round {r}/{cfg.rounds} | Labeled={len(labeled)} | Unlabeled={len(unlabeled)} ===")
        trainer = trainer_cls(cfg, dataset, labeled, val_indices)

        round_log_path = os.path.join(cfg.save_dir, f"train_log_round_{r}.txt")
        trainer.fit(log_path=round_log_path)

        if not unlabeled:
            history_best.append(history_best[-1])
            break

        preds = trainer.predict_subset_mc(unlabeled)
        if not preds.empty:
            idxs = preds["dataset_idx"].astype(int).tolist()
            for prop_name in cfg.properties:
                true_vals = df_all.iloc[idxs][prop_name].to_numpy()
                preds[f"v_{prop_name}_true"] = true_vals
        mean_col = f"v_{prop}_mean"
        std_col = f"v_{prop}_std"

        if cfg.maximize_best:
            best_idxs = preds.nlargest(cfg.per_round_best_k, mean_col)["dataset_idx"].tolist()
        else:
            best_idxs = preds.nsmallest(cfg.per_round_best_k, mean_col)["dataset_idx"].tolist()
        uncert_idxs = preds.nlargest(cfg.per_round_uncert_k, std_col)["dataset_idx"].tolist()

        candidate_indices_for_scoring = labeled + best_idxs + uncert_idxs
        vals_r = df_all[prop].iloc[candidate_indices_for_scoring].to_numpy()
        best_r = float(vals_r.max() if cfg.maximize_best else vals_r.min())
        history_best.append(best_r)

        if (not discovered_found) and any(i in true_best_indices for i in (best_idxs + uncert_idxs)):
            discovered_round = r
            discovered_found = True

        selected, seen = [], set()
        for idx in best_idxs + uncert_idxs:
            if idx not in seen:
                selected.append(idx)
                seen.add(idx)
        labeled = sorted(labeled + selected)
        unlabeled = sorted([i for i in unlabeled if i not in seen])

        preds_path = os.path.join(cfg.save_dir, f"preds_round_{r}.csv")
        preds.to_csv(preds_path, index=False)

        round_summaries.append({
            "round": r,
            "labeled_size_end": len(labeled),
            "selected_best": best_idxs,
            "selected_uncert": uncert_idxs,
            "best_round_value": float(best_r),
            "validation_indices": val_indices,
            "train_log_path": os.path.relpath(round_log_path, start=cfg.save_dir),
            "preds_file": os.path.relpath(preds_path, start=cfg.save_dir),
        })

    if not discovered_found:
        discovered_round = len(history_best) - 1

    summary_df = pd.DataFrame(round_summaries)
    summary_df["true_best_value"] = true_best_value
    summary_df["true_best_found"] = discovered_found
    summary_df["true_best_mark_round_on_plot"] = discovered_round
    summary_df["validation_indices"] = "|".join(map(str, val_indices)) if val_indices else ""
    summary_df["validation_size"] = len(val_indices)
    summary_csv = os.path.join(cfg.save_dir, "round_summaries.csv")
    summary_df.to_csv(summary_csv, index=False)

    plt.rcParams["axes.unicode_minus"] = False
    xs = list(range(len(history_best)))
    ys = history_best
    plt.figure()
    plt.plot(xs, ys, marker="o")
    plt.xlabel("Round")
    plt.ylabel(f"Best {prop} (labeled union topK-best this round)")
    plt.title("Best value per round (round0 = initial selection)")

    star_x = int(discovered_round)
    if discovered_found:
        star_y = ys[star_x]
        note = f"Found true best = {true_best_value:.4f} at round {star_x}"
    else:
        star_y = true_best_value
        note = f"Global best = {true_best_value:.4f} (never discovered; marked at final round)"
    ymin = min(min(ys), star_y)
    ymax = max(max(ys), star_y)
    pad = 0.05 * (ymax - ymin + 1e-12)
    plt.ylim(ymin - pad, ymax + pad)
    plt.plot([star_x], [star_y], marker="*", markersize=16, color="red", zorder=5)
    to_right = star_x < (xs[0] + xs[-1]) / 2 if xs else True
    x_off = 18 if to_right else -18
    ha = "left" if to_right else "right"
    plt.annotate(
        note,
        (star_x, star_y),
        textcoords="offset points",
        xytext=(x_off, 12),
        ha=ha,
        va="bottom",
        color="red",
        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="red", alpha=0.8),
    )
    plot_path = os.path.join(cfg.save_dir, "best_per_round.png")
    plt.tight_layout()
    plt.savefig(plot_path, bbox_inches="tight", dpi=150)
    plt.close()

    print(f"[OK] {summary_csv}")
    print(f"[OK] {plot_path}")
    return summary_df, plot_path


# Remaining CLI helpers stay the same as before.

def run_active_learning_per_adsorbate(
    export_root: str,
    property_name: str = "y_relax",
    categories: Optional[List[str]] = None,
    init_size: int = 20,
    rounds: int = 5,
    val_size: int = 20,
    per_round_best_k: int = 10,
    per_round_uncert_k: int = 10,
    epochs: int = 100,
    batch_size: int = 16,
    eval_times: int = 30,
    maximize_best: bool = False,
    seed: int = 23,
    model_type: str = "cgcnn",
    model_kwargs: Optional[Dict[str, Any]] = None,
    early_stop_patience: Optional[int] = 10,
) -> pd.DataFrame:
    export_root = os.path.abspath(export_root)
    assert os.path.isdir(export_root), f"Directory not found: {export_root}"

    subdirs = []
    for name in sorted(os.listdir(export_root)):
        path = os.path.join(export_root, name)
        if not os.path.isdir(path):
            continue
        if categories and (name not in categories):
            continue
        if os.path.isfile(os.path.join(path, "id_prop.csv")) and os.path.isdir(os.path.join(path, "structures")):
            subdirs.append((name, path))
    
    print(f"Found {len(subdirs)} categories.")
    results = []

    for name, root in subdirs:
        print("\n" + "=" * 80)
        print(f"[Category] {name}")

        model_dir = (model_type or "cgcnn").lower()
        save_dir = os.path.join(root, "al_runs", model_dir)

        patience = None if (early_stop_patience is not None and early_stop_patience <= 0) else early_stop_patience

        cfg = ALConfig(
            label_dir=root,
            structure_dir=os.path.join(root, "structures"),
            properties=[property_name],
            init_size=init_size,
            rounds=rounds,
            per_round_best_k=per_round_best_k,
            per_round_uncert_k=per_round_uncert_k,
            epochs=epochs,
            batch_size=batch_size,
            eval_times=eval_times,
            cuda=torch.cuda.is_available(),
            save_dir=save_dir,
            lr_milestones=None,
            weight_decay=0.0,
            dropout=0.1,
            maximize_best=maximize_best,
            random_seed=seed,
            model_type=model_type,
            model_kwargs=model_kwargs,
            val_size=val_size,
            early_stop_patience=patience,
        )

        try:
            df = pd.read_csv(os.path.join(root, "id_prop.csv"))
            n_rows = len(df)
            if n_rows <= cfg.init_size + 1:
                cfg.init_size = max(1, min(5, n_rows // 2))
                print(f"[{name}] dataset is small, adjusting init_size to {cfg.init_size}")
        except Exception:
            n_rows = None

        try:
            summary_df, plot_path = active_learning_loop(cfg)
            results.append({"ads_smi": name, "n": n_rows, "save_dir": cfg.save_dir, "plot": plot_path})
        except Exception as exc:
            print(f"[{name}] run failed: {exc}")
            results.append({"ads_smi": name, "n": n_rows, "save_dir": cfg.save_dir, "error": str(exc)})

    return pd.DataFrame(results)


def run_multi_trial_experiment(
    export_root: str,
    property_name: str = "y_relax",
    categories: Optional[List[str]] = None,
    init_size: int = 20,
    rounds: int = 5,
    val_size: int = 20,
    per_round_best_k: int = 10,
    per_round_uncert_k: int = 10,
    epochs: int = 100,
    batch_size: int = 16,
    eval_times: int = 30,
    maximize_best: bool = False,
    base_seed: int = 23,
    trials: int = 40,
    model_type: str = "cgcnn",
    task_name: str = "[C]",
    model_kwargs: Optional[Dict[str, Any]] = None,
    early_stop_patience: Optional[int] = 10,
) -> pd.DataFrame:
    export_root = os.path.abspath(export_root)
    assert os.path.isdir(export_root), f"Directory not found: {export_root}"

    subdirs = []
    for name in sorted(os.listdir(export_root)):
        path = os.path.join(export_root, name)
        if not os.path.isdir(path):
            continue
        if categories and (name not in categories):
            continue
        if os.path.isfile(os.path.join(path, "id_prop.csv")) and os.path.isdir(os.path.join(path, "structures")):
            if name == task_name:
                subdirs.append((name, path))
    print(f"Found {len(subdirs)} categories.")
    records_per_category = []

    random_seeds = _load_selected_trial_seeds()
    if trials > len(random_seeds):
        raise ValueError(
            f"Requested {trials} trials, but only {len(random_seeds)} ordered seeds are available in {SELECTED_SEEDS_PATH}"
        )
    
    for name, root in subdirs:
        print("\n" + "=" * 80)
        print(f"[Category] {name}")

        model_dir = (model_type or "cgcnn").lower()
        category_records = []
        patience = None if (early_stop_patience is not None and early_stop_patience <= 0) else early_stop_patience

        for trial_idx, seed in enumerate(random_seeds[:trials]):
            start_time = time.time()
            # torch.manual_seed(seed)
            # torch.cuda.manual_seed(seed)
            # torch.cuda.manual_seed_all(seed)
            save_dir = os.path.join(root, "al_runs", model_dir, f"trial_{trial_idx:02d}")
            os.makedirs(save_dir, exist_ok=True)

            cfg = ALConfig(
                label_dir=root,
                structure_dir=os.path.join(root, "structures"),
                properties=[property_name],
                init_size=init_size,
                rounds=rounds,
                per_round_best_k=per_round_best_k,
                per_round_uncert_k=per_round_uncert_k,
                epochs=epochs,
                batch_size=batch_size,
                eval_times=eval_times,
                cuda=torch.cuda.is_available(),
                save_dir=save_dir,
                lr_milestones=None,
                weight_decay=0.0,
                dropout=0.1,
                maximize_best=maximize_best,
                random_seed=seed,
                model_type=model_type,
                model_kwargs=model_kwargs,
                val_size=val_size,
                early_stop_patience=patience,
            )

            summary_df, plot_path = active_learning_loop(cfg)
            found_flag = bool(summary_df["true_best_found"].iloc[-1]) if not summary_df.empty else False
            found_round = int(summary_df["true_best_mark_round_on_plot"].iloc[-1]) if found_flag else None
            category_records.append({
                "category": name,
                "trial": trial_idx,
                "seed": seed,
                "save_dir": save_dir,
                "plot": plot_path,
                "found": found_flag,
                "found_round": found_round,
            })
            print(f"[{name}][Trial {trial_idx}] completed in {time.time() - start_time:.2f} seconds.")

        category_df = pd.DataFrame(category_records)
        if not category_df.empty:
            found_rounds = category_df.loc[category_df["found"], "found_round"]
            mean_round = found_rounds.mean() if not found_rounds.empty else None
            category_df["mean_found_round"] = mean_round
            category_df["found_count"] = found_rounds.count()
        records_per_category.append(category_df)

    if records_per_category:
        return pd.concat(records_per_category, ignore_index=True)
    return pd.DataFrame()


def _parse_categories(value: Optional[str]) -> Optional[List[str]]:
    if value is None:
        return None
    if not value.strip():
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def _parse_model_kwargs(value: Optional[str]) -> Optional[Dict[str, Any]]:
    if value is None:
        return None
    if not value.strip():
        return None
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Failed to parse model_kwargs JSON: {exc}") from exc
    if not isinstance(parsed, dict):
        raise ValueError("model_kwargs must decode to a JSON object")
    return parsed


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Active learning pipeline utilities.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_common_arguments(subparser: argparse.ArgumentParser):
        subparser.add_argument("--export-root", default="exp_data", help="Root directory containing adsorbate folders.")
        subparser.add_argument("--property-name", default="y_relax", help="Target property column name.")
        subparser.add_argument("--categories", default=None, help="Comma-separated subset of categories to process.")
        subparser.add_argument("--init-size", type=int, default=20, help="Initial labeled size per category.")
        subparser.add_argument("--rounds", type=int, default=10, help="Number of active learning rounds.")
        subparser.add_argument("--val-size", type=int, default=40, help="Validation set size held out before AL.")
        subparser.add_argument("--per-round-best-k", type=int, default=10, help="Top-k candidates selected by mean.")
        subparser.add_argument("--per-round-uncert-k", type=int, default=10, help="Top-k candidates selected by std.")
        subparser.add_argument("--epochs", type=int, default=100, help="Training epochs per round.")
        subparser.add_argument("--batch-size", type=int, default=16, help="Batch size for training and inference.")
        subparser.add_argument("--eval-times", type=int, default=10, help="MC dropout evaluation passes.")
        subparser.add_argument("--task-name", default="[C]", help="Task type: [C] or [O].")
        subparser.add_argument(
            "--early-stop-patience",
            type=int,
            default=10,
            help="Stop training early if validation does not improve for this many rounds (<=0 disables).",
        )
        subparser.add_argument("--maximize-best", action="store_true", help="Flip to maximize the property value.")
        subparser.add_argument("--model-type", default="dpp", help="Backend: cgcnn, escn, dpp, schnet, gemnet_oc, et, or catai.")
        subparser.add_argument(
            "--model-kwargs",
            default=None,
            help="JSON string with additional encoder kwargs.",
        )

    run_parser = subparsers.add_parser("per-adsorbate", help="Run active learning once per category.")
    add_common_arguments(run_parser)
    run_parser.add_argument("--seed", type=int, default=23, help="Random seed for data splits.")

    multi_parser = subparsers.add_parser("multi-trial", help="Run repeated experiments per category.")
    add_common_arguments(multi_parser)
    multi_parser.add_argument("--base-seed", type=int, default=23, help="Unused legacy option kept for compatibility.")
    multi_parser.add_argument("--trials", type=int, default=40, help="Number of ordered seed runs per category.")

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    categories = _parse_categories(args.categories)
    model_kwargs = _parse_model_kwargs(args.model_kwargs)

    if args.command == "per-adsorbate":
        df = run_active_learning_per_adsorbate(
            export_root=args.export_root,
            property_name=args.property_name,
            categories=categories,
            init_size=args.init_size,
            rounds=args.rounds,
            per_round_best_k=args.per_round_best_k,
            per_round_uncert_k=args.per_round_uncert_k,
            val_size=args.val_size,
            epochs=args.epochs,
            batch_size=args.batch_size,
            eval_times=args.eval_times,
            early_stop_patience=args.early_stop_patience,
            maximize_best=args.maximize_best,
            seed=args.seed,
            model_type=args.model_type,
            model_kwargs=model_kwargs,
        )
    elif args.command == "multi-trial":
        df = run_multi_trial_experiment(
            export_root=args.export_root,
            property_name=args.property_name,
            categories=categories,
            init_size=args.init_size,
            rounds=args.rounds,
            per_round_best_k=args.per_round_best_k,
            per_round_uncert_k=args.per_round_uncert_k,
            val_size=args.val_size,
            epochs=args.epochs,
            batch_size=args.batch_size,
            eval_times=args.eval_times,
            early_stop_patience=args.early_stop_patience,
            maximize_best=args.maximize_best,
            base_seed=args.base_seed,
            trials=args.trials,
            task_name=args.task_name,
            model_type=args.model_type,
            model_kwargs=model_kwargs,
        )
    else:
        parser.error(f"Unknown command: {args.command}")
        return 2

    if not df.empty:
        print(df)
    else:
        print("No results generated.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
