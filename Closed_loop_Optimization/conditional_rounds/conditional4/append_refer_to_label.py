from __future__ import annotations

import csv
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
LABEL_PATH = BASE_DIR / "label_data.csv"
SOURCE_FILES = [
    BASE_DIR / "final_rec_yield_with_refer.csv",
    BASE_DIR / "final_rec_uncertainty_with_refer.csv",
]
MATCH_COLUMNS = ("base", "solvent", "nm", "cat")


def load_label_rows() -> list[dict[str, str]]:
    with LABEL_PATH.open("r", encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def normalize_source_row(row: dict[str, str]) -> dict[str, str] | None:
    refer_yield = row.get("Refer_Yield_Mean", "").strip()
    if not refer_yield:
        return None

    return {
        "ID": row["ID"].strip(),
        "base": row["Base"].strip(),
        "solvent": row["Solvent"].strip(),
        "nm": row["NM"].strip(),
        "cat": row["Cat"].strip(),
        "YIELD": refer_yield,
    }


def load_source_rows() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for source_path in SOURCE_FILES:
        with source_path.open("r", encoding="utf-8", newline="") as file:
            for row in csv.DictReader(file):
                normalized = normalize_source_row(row)
                if normalized is not None:
                    rows.append(normalized)
    return rows


def make_key(row: dict[str, str]) -> tuple[str, str, str, str]:
    return tuple(row[column] for column in MATCH_COLUMNS)


def main() -> None:
    label_rows = load_label_rows()
    source_rows = load_source_rows()

    existing_keys = {make_key(row) for row in label_rows}
    appended_rows: list[dict[str, str]] = []

    for row in source_rows:
        key = make_key(row)
        if key in existing_keys:
            continue
        label_rows.append(row)
        appended_rows.append(row)
        existing_keys.add(key)

    with LABEL_PATH.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["ID", "base", "solvent", "nm", "cat", "YIELD"])
        writer.writeheader()
        writer.writerows(label_rows)

    print(f"新增 {len(appended_rows)} 条记录到 {LABEL_PATH}")


if __name__ == "__main__":
    main()
