from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
REFER_PATH = BASE_DIR / "refer.csv"
INPUT_FILES = [
    BASE_DIR / "final_rec_yield.csv",
    BASE_DIR / "final_rec_uncertainty.csv",
]
OUTPUT_SUFFIX = "_with_refer"


def load_refer_map(refer_path: Path) -> dict[tuple[str, str, str, str], list[str]]:
    refer_map: dict[tuple[str, str, str, str], list[str]] = defaultdict(list)
    with refer_path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.reader(file, delimiter="\t")
        for row in reader:
            if not row:
                continue
            row = [item.strip() for item in row]
            if len(row) < 6:
                row += [""] * (6 - len(row))
            _, base, solvent, nm, cat, yield_value = row[:6]
            refer_map[(base, solvent, nm, cat)].append(yield_value)
    return dict(refer_map)


def summarize_yields(values: list[str]) -> tuple[str, str]:
    numeric_values = [float(value) for value in values if str(value).strip() != ""]
    if not numeric_values:
        return "", ""

    mean_value = sum(numeric_values) / len(numeric_values)
    if mean_value.is_integer():
        mean_text = str(int(mean_value))
    else:
        mean_text = f"{mean_value:.6f}".rstrip("0").rstrip(".")
    values_text = ";".join(
        str(int(value)) if float(value).is_integer() else str(value) for value in numeric_values
    )
    return mean_text, values_text


def enrich_file(input_path: Path, refer_map: dict[tuple[str, str, str, str], list[str]]) -> Path:
    output_path = input_path.with_name(f"{input_path.stem}{OUTPUT_SUFFIX}{input_path.suffix}")
    with input_path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        rows = list(reader)
        fieldnames = list(reader.fieldnames or [])

    for extra_field in ("Refer_Yield_Mean", "Refer_Yield_Values"):
        if extra_field not in fieldnames:
            fieldnames.append(extra_field)

    for row in rows:
        key = (
            row["Base"].strip(),
            row["Solvent"].strip(),
            row["NM"].strip(),
            row["Cat"].strip(),
        )
        refer_values = refer_map.get(key, [])
        mean_text, values_text = summarize_yields(refer_values)
        row["Refer_Yield_Mean"] = mean_text
        row["Refer_Yield_Values"] = values_text

    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return output_path


def main() -> None:
    refer_map = load_refer_map(REFER_PATH)
    for input_path in INPUT_FILES:
        output_path = enrich_file(input_path, refer_map)
        print(f"已生成: {output_path}")


if __name__ == "__main__":
    main()
