from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


DEFAULT_RUN_GLOB = "*-catai_screening_infer+POE_lmdb+infer+{checkpoint_id}"
DEFAULT_RESULTS_DIR = "results"
DEFAULT_NPZ_NAME = "test_predictions_graph_embedding.npz"
DEFAULT_CHECKPOINT = 0
DEFAULT_TOP_K = 30
DEFAULT_SMILES_CSV = Path("POE_infer/mat_index_smiles.csv")


def normalize_id(value: object) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="ignore")
    if isinstance(value, np.generic):
        value = value.item()
    return str(value)


def find_first_key(keys: list[str], keyword: str) -> str:
    for key in keys:
        if keyword in key.lower():
            return key
    raise KeyError(f"没有找到包含 {keyword!r} 的 key，可用 keys: {keys}")


def find_optional_first_key(keys: list[str], keyword: str) -> str | None:
    for key in keys:
        if keyword in key.lower():
            return key
    return None


def resolve_run_dir(results_root: Path, run_glob: str, checkpoint_id: int) -> Path:
    matched_dirs = sorted(results_root.glob(run_glob.format(checkpoint_id=checkpoint_id)))
    if not matched_dirs:
        raise FileNotFoundError(
            f"在 {results_root} 下没有找到 checkpoint {checkpoint_id} 对应目录，"
            f"匹配模式: {run_glob}"
        )
    if len(matched_dirs) > 1:
        selected_dir = matched_dirs[-1]
        print(
            f"checkpoint {checkpoint_id} 匹配到多个目录，默认使用最新目录: {selected_dir}\n"
            f"全部匹配目录: {matched_dirs}"
        )
        return selected_dir

    return matched_dirs[0]


def load_ranked_samples(
    checkpoint_id: int,
    results_root: Path,
    run_glob: str,
    npz_name: str,
) -> list[tuple[str, float]]:
    run_dir = resolve_run_dir(results_root, run_glob, checkpoint_id)
    npz_path = run_dir / npz_name
    if not npz_path.exists():
        raise FileNotFoundError(f"文件不存在: {npz_path}")

    data = np.load(npz_path, allow_pickle=True)
    energy_key = find_first_key(list(data.files), "energy")
    id_key = find_first_key(list(data.files), "id")

    energies = np.asarray(data[energy_key], dtype=float)
    ids = [normalize_id(item) for item in data[id_key]]

    if len(ids) != len(energies):
        raise ValueError(
            f"checkpoint {checkpoint_id} 的 id 数量与 energy 数量不一致: "
            f"{len(ids)} vs {len(energies)}"
        )

    sorted_pairs = sorted(zip(ids, energies), key=lambda item: item[1], reverse=True)

    print(
        f"checkpoint {checkpoint_id}: 读取 {len(sorted_pairs)} 条样本，"
        f"文件路径: {npz_path}"
    )
    return [(sample_id, float(energy)) for sample_id, energy in sorted_pairs]


def load_smiles_mapping(screening_root: Path, smiles_csv: Path) -> dict[str, str]:
    smiles_csv_path = smiles_csv
    if not smiles_csv_path.is_absolute():
        smiles_csv_path = screening_root / smiles_csv_path

    with smiles_csv_path.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))

    mapping: dict[str, str] = {}
    for row in rows:
        mapping[row["ID"]] = row["SMILES"]

    print(f"读取 smiles 映射 {len(mapping)} 条，文件路径: {smiles_csv_path}")
    return mapping


def print_top_smiles(
    samples: list[tuple[str, float]],
    smiles_mapping: dict[str, str],
    top_k: int,
) -> None:
    print(f"\n=== Top {min(top_k, len(samples))} smiles ===")
    missing_ids: list[str] = []
    for rank, (sample_id, energy) in enumerate(samples[:top_k], start=1):
        smiles = smiles_mapping.get(sample_id)
        if smiles is None:
            missing_ids.append(sample_id)
            smiles = f"<missing-smiles:{sample_id}>"
        print(f"{rank}\t{smiles}\t{energy:.6f}")

    if missing_ids:
        print(f"\n警告: 以下 ID 未在 smiles 映射中找到: {missing_ids}")


def parse_args() -> argparse.Namespace:
    screening_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="对 POE 推理结果按能量排序并打印前若干个 smiles。")
    parser.add_argument(
        "--results-root",
        type=Path,
        default=screening_root / DEFAULT_RESULTS_DIR,
        help="推理结果目录，默认是 Large_scale_screening/results",
    )
    parser.add_argument(
        "--checkpoint",
        type=int,
        default=DEFAULT_CHECKPOINT,
        help="读取的 checkpoint 编号",
    )
    parser.add_argument(
        "--run-glob",
        default=DEFAULT_RUN_GLOB,
        help="结果目录匹配模式，需包含 {checkpoint_id} 占位符",
    )
    parser.add_argument(
        "--npz-name",
        default=DEFAULT_NPZ_NAME,
        help="推理输出 npz 文件名",
    )
    parser.add_argument(
        "--smiles-csv",
        type=Path,
        default=DEFAULT_SMILES_CSV,
        help="ID 到 smiles 的映射表，默认是 Large_scale_screening/POE_infer/mat_index_smiles.csv",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        help="打印前 k 个 smiles",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    screening_root = Path(__file__).resolve().parents[2]
    samples = load_ranked_samples(
        checkpoint_id=args.checkpoint,
        results_root=args.results_root,
        run_glob=args.run_glob,
        npz_name=args.npz_name,
    )
    smiles_mapping = load_smiles_mapping(screening_root, args.smiles_csv)
    print_top_smiles(samples, smiles_mapping, args.top_k)


if __name__ == "__main__":
    main()