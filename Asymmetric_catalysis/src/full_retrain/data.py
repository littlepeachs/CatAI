from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, train_test_split


ROOT = Path(
    os.environ.get("CATAI_WORKSPACE_ROOT", Path(__file__).resolve().parents[1])
).resolve()
CHEMAHNET_ROOT = ROOT / "ChemAHNet"


@dataclass(frozen=True)
class DatasetBundle:
    name: str
    task_type: str
    train: pd.DataFrame | None
    val: pd.DataFrame | None
    test: pd.DataFrame | None
    cv_folds: list[tuple[np.ndarray, np.ndarray]] | None
    input_columns: tuple[str, ...]
    target_column: str
    source: str


def signed_ddg(df: pd.DataFrame) -> pd.Series:
    """ChemAHNet Fig. 4c uses signed ddG; the CSV stores positive ddG plus Label."""
    labels = pd.to_numeric(df["Label"], errors="coerce").fillna(0).astype(int)
    ddg = pd.to_numeric(df["ddG"], errors="coerce")
    sign = np.where(labels.to_numpy() == 1, 1.0, -1.0)
    return pd.Series(ddg.to_numpy() * sign, index=df.index, name="signed_ddG")


def _normalize_aho(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["signed_ddG"] = signed_ddg(out)
    out["seq2seq_source"] = (
        out["Reactant SMILES"].fillna("").astype(str)
        + "."
        + out["Solvent SMILES"].fillna("").astype(str)
        + "."
        + out["Catalyst SMILES"].fillna("").astype(str)
    )
    out["seq2seq_target"] = out["Product SMILES"].fillna("").astype(str)
    return out


def load_aho_full() -> pd.DataFrame:
    return _normalize_aho(pd.read_csv(CHEMAHNET_ROOT / "9478_file.csv"))


def load_aho_example() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = _normalize_aho(pd.read_csv(CHEMAHNET_ROOT / "data/example_train.csv"))
    val = _normalize_aho(pd.read_csv(CHEMAHNET_ROOT / "data/example_eval.csv"))
    test = _normalize_aho(pd.read_csv(CHEMAHNET_ROOT / "data/example_test.csv"))
    return train, val, test


def _aho_paper_exact_test(split: str = "classification") -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Use the released Fig. 3/Fig. 4 test rows exactly.

    The first 940 rows of 9478_file.csv match Fig. 3b target order exactly.
    The paper validation indices are not released, so validation is taken from
    the next 948 rows and the remainder is used for training.
    """
    df = load_aho_full()
    test = df.iloc[:940].copy()
    val = df.iloc[940:1888].copy()
    train = df.iloc[1888:].copy()
    return train, val, test


def _aho_random42(stratify: str | None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = load_aho_full()
    strat = df[stratify] if stratify else None
    pool, test = train_test_split(
        df,
        test_size=0.10,
        random_state=42,
        shuffle=True,
        stratify=strat,
    )
    strat_pool = pool[stratify] if stratify else None
    train, val = train_test_split(
        pool,
        test_size=1 / 9,
        random_state=42,
        shuffle=True,
        stratify=strat_pool,
    )
    return train.copy(), val.copy(), test.copy()


def aho_major(split: str) -> DatasetBundle:
    if split == "example":
        train, val, test = load_aho_example()
        source = "ChemAHNet/data/example_{train,eval,test}.csv"
    elif split == "paper_exact_test":
        train, val, test = _aho_paper_exact_test("classification")
        source = "9478_file.csv first 940 rows reproduce released Fig. 3b test order"
    elif split == "paper_random42":
        train, val, test = _aho_random42(stratify="Label")
        source = "9478_file.csv random 80/10/10 split, random_state=42"
    else:
        raise ValueError(f"Unsupported AHO split: {split}")
    return DatasetBundle(
        name="aho_major",
        task_type="classification",
        train=train,
        val=val,
        test=test,
        cv_folds=None,
        input_columns=("Reactant SMILES", "Solvent SMILES", "Catalyst SMILES"),
        target_column="Label",
        source=source,
    )


def aho_ddg(split: str) -> DatasetBundle:
    if split == "example":
        train, val, test = load_aho_example()
        source = "ChemAHNet/data/example_{train,eval,test}.csv"
    elif split == "paper_exact_test":
        train, val, test = _aho_paper_exact_test("regression")
        source = "9478_file.csv first 940 rows align with released Fig. 4c test targets after signing ddG"
    elif split == "paper_random42":
        train, val, test = _aho_random42(stratify=None)
        source = "9478_file.csv random 80/10/10 split, random_state=42"
    else:
        raise ValueError(f"Unsupported AHO split: {split}")
    return DatasetBundle(
        name="aho_ddg",
        task_type="regression",
        train=train,
        val=val,
        test=test,
        cv_folds=None,
        input_columns=("Reactant SMILES", "Solvent SMILES", "Catalyst SMILES"),
        target_column="signed_ddG",
        source=source,
    )


def aho_seq2seq(split: str) -> DatasetBundle:
    if split == "example":
        train, val, test = load_aho_example()
        source = "ChemAHNet/data/example_{train,eval,test}.csv"
    elif split == "paper_exact_test":
        train, val, test = _aho_paper_exact_test("seq2seq")
        source = "9478_file.csv first 940 rows reproduce released Fig. 3b target order"
    elif split == "paper_random42":
        train, val, test = _aho_random42(stratify="Label")
        source = "9478_file.csv random 80/10/10 split, random_state=42"
    else:
        raise ValueError(f"Unsupported AHO split: {split}")
    return DatasetBundle(
        name="aho_seq2seq_product",
        task_type="seq2seq",
        train=train,
        val=val,
        test=test,
        cv_folds=None,
        input_columns=("seq2seq_source",),
        target_column="seq2seq_target",
        source=source,
    )


def rh_binol() -> DatasetBundle:
    df = pd.read_csv(CHEMAHNET_ROOT / "data/data3.csv").copy()
    folds = list(KFold(n_splits=10, shuffle=True, random_state=42).split(df))
    return DatasetBundle(
        name="rh_binol",
        task_type="regression",
        train=df,
        val=None,
        test=None,
        cv_folds=folds,
        input_columns=("Reactant SMILES", "Solvent SMILES", "Catalyst SMILES(RDKit)"),
        target_column="ddG",
        source="ChemAHNet/data/data3.csv; 10-fold KFold random_state=42",
    )


def _load_cpa_frame(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path).copy()
    df["target"] = pd.to_numeric(df["Output"], errors="coerce")
    return df


def cpa_main() -> DatasetBundle:
    df = _load_cpa_frame(CHEMAHNET_ROOT / "data/output.csv")
    test_size = 475 / 1075
    train, test = train_test_split(df, test_size=test_size, random_state=42, shuffle=True)
    train, val = train_test_split(train, test_size=0.10, random_state=42, shuffle=True)
    return DatasetBundle(
        name="cpa_main",
        task_type="regression",
        train=train.copy(),
        val=val.copy(),
        test=test.copy(),
        cv_folds=None,
        input_columns=("Catalyst", "Imine", "Thiol"),
        target_column="target",
        source="ChemAHNet/data/output.csv; same random_state=42 split logic as main_ddG_thiol.py",
    )


def cpa_scaffold(split_name: str) -> DatasetBundle:
    mapping = {
        "cpa_imine": ("train_Imine.csv", "test_Imine.csv"),
        "cpa_thiol": ("train_thiol.csv", "test_thiol.csv"),
        "cpa_catalyst": ("train_Catalyst.csv", "test_Catalyst.csv"),
        "cpa_transformation": ("train_Transformation.csv", "test_Transformation.csv"),
    }
    if split_name not in mapping:
        raise ValueError(f"Unsupported CPA split: {split_name}")
    train_file, test_file = mapping[split_name]
    train = _load_cpa_frame(CHEMAHNET_ROOT / "data" / train_file)
    test = _load_cpa_frame(CHEMAHNET_ROOT / "data" / test_file)
    train, val = train_test_split(train, test_size=0.10, random_state=42, shuffle=True)
    return DatasetBundle(
        name=split_name,
        task_type="regression",
        train=train.copy(),
        val=val.copy(),
        test=test.copy(),
        cv_folds=None,
        input_columns=("Catalyst", "Imine", "Thiol"),
        target_column="target",
        source=f"ChemAHNet/data/{train_file} and {test_file}",
    )


def external_regression(name: str) -> DatasetBundle:
    files = {
        "oca": "OCA_output.csv",
        "pca": "PCA_output.csv",
        "oe": "OE_output.csv",
    }
    if name not in files:
        raise ValueError(f"Unsupported external dataset: {name}")
    df = pd.read_csv(CHEMAHNET_ROOT / "data" / files[name]).copy()
    target_col = "Output" if "Output" in df.columns else "ddG"
    if target_col not in df.columns:
        # These public CSVs store ee; ChemAHNet preprocess converts it to ddG-like targets.
        # Use the paired pickle/row data for final paper metrics, but keep this path trainable.
        target_col = "ee"
    df["target"] = pd.to_numeric(df[target_col], errors="coerce")
    cols = tuple(c for c in ("Catalyst", "reactant1", "reactant2", "solvent") if c in df.columns)
    folds = list(KFold(n_splits=10, shuffle=True, random_state=42).split(df))
    return DatasetBundle(
        name=name,
        task_type="regression",
        train=df,
        val=None,
        test=None,
        cv_folds=folds,
        input_columns=cols,
        target_column="target",
        source=f"ChemAHNet/data/{files[name]}; 10-fold KFold random_state=42",
    )


def load_task(task: str, split: str = "paper_exact_test") -> DatasetBundle:
    if task == "aho_major":
        return aho_major(split)
    if task == "aho_ddg":
        return aho_ddg(split)
    if task == "aho_seq2seq":
        return aho_seq2seq(split)
    if task == "rh_binol":
        return rh_binol()
    if task == "cpa_main":
        return cpa_main()
    if task in {"cpa_imine", "cpa_thiol", "cpa_catalyst", "cpa_transformation"}:
        return cpa_scaffold(task)
    if task in {"oca", "pca", "oe"}:
        return external_regression(task)
    raise ValueError(f"Unknown task: {task}")


def all_fingerprint_tasks() -> Iterable[str]:
    return (
        "aho_major",
        "aho_ddg",
        "rh_binol",
        "cpa_main",
        "cpa_imine",
        "cpa_thiol",
        "cpa_catalyst",
        "cpa_transformation",
        "oca",
        "pca",
        "oe",
    )
