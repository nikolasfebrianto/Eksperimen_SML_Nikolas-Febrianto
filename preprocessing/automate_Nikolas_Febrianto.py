"""Automated preprocessing for UCI Breast Cancer Wisconsin Diagnostic dataset.

Example:
    python preprocessing/automate_Nikolas_Febrianto.py \
        --raw-path breast_cancer_raw/breast_cancer_raw.csv \
        --output-dir preprocessing/breast_cancer_preprocessing
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

TARGET_COLUMN = "is_malignant"
RANDOM_STATE = 42


def clean_column_name(name: str) -> str:
    name = name.strip().lower()
    name = re.sub(r"[^a-z0-9]+", "_", name)
    name = re.sub(r"_+", "_", name).strip("_")
    return name


def preprocess(raw_path: str | Path, output_dir: str | Path) -> None:
    raw_path = Path(raw_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not raw_path.exists():
        raise FileNotFoundError(f"Dataset raw tidak ditemukan: {raw_path}")

    df = pd.read_csv(raw_path)
    initial_rows = len(df)
    df.columns = [clean_column_name(col) for col in df.columns]
    df = df.drop_duplicates()
    df = df.dropna(subset=["diagnosis"]).copy()
    df["diagnosis"] = df["diagnosis"].astype(str).str.upper().str.strip()
    df = df[df["diagnosis"].isin(["M", "B"])].copy()
    df[TARGET_COLUMN] = df["diagnosis"].map({"M": 1, "B": 0}).astype(int)

    drop_columns = ["sample_id", "diagnosis"]
    X = df.drop(columns=drop_columns + [TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors="coerce")

    missing_before = X.isna().sum()
    medians = X.median(numeric_only=True)
    X = X.fillna(medians)

    feature_columns = X.columns.tolist()
    cap_bounds = {}
    X_capped = X.copy()
    for col in feature_columns:
        q1 = X_capped[col].quantile(0.25)
        q3 = X_capped[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        cap_bounds[col] = {"lower": float(lower), "upper": float(upper)}
        X_capped[col] = X_capped[col].clip(lower=lower, upper=upper)

    full_data = X_capped.copy()
    full_data[TARGET_COLUMN] = y.values

    train_df, test_df = train_test_split(
        full_data,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=full_data[TARGET_COLUMN],
    )

    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(train_df[feature_columns]),
        columns=feature_columns,
        index=train_df.index,
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(test_df[feature_columns]),
        columns=feature_columns,
        index=test_df.index,
    )

    train_out = X_train_scaled.copy()
    train_out[TARGET_COLUMN] = train_df[TARGET_COLUMN].astype(int).values
    test_out = X_test_scaled.copy()
    test_out[TARGET_COLUMN] = test_df[TARGET_COLUMN].astype(int).values
    full_out = pd.concat([train_out, test_out], axis=0).sort_index()

    train_out.to_csv(output_dir / "train.csv", index=False)
    test_out.to_csv(output_dir / "test.csv", index=False)
    full_out.to_csv(output_dir / "full_preprocessed.csv", index=False)

    joblib.dump(
        {
            "scaler": scaler,
            "feature_columns": feature_columns,
            "target_column": TARGET_COLUMN,
            "target_mapping": {"M": 1, "B": 0},
            "median_values": medians.to_dict(),
            "iqr_cap_bounds": cap_bounds,
            "source_dataset": "UCI Breast Cancer Wisconsin (Diagnostic)",
        },
        output_dir / "preprocessor.joblib",
    )

    metadata = {
        "dataset_name": "Breast Cancer Wisconsin (Diagnostic)",
        "source": "UCI Machine Learning Repository",
        "url": "https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic",
        "target_column": TARGET_COLUMN,
        "target_definition": {"0": "benign", "1": "malignant"},
        "raw_rows": int(initial_rows),
        "clean_rows": int(len(full_data)),
        "train_rows": int(len(train_out)),
        "test_rows": int(len(test_out)),
        "feature_count": int(len(feature_columns)),
        "feature_columns": feature_columns,
        "random_state": RANDOM_STATE,
    }
    preprocessing_report = {
        "duplicates_removed": int(initial_rows - len(df)),
        "missing_values_before_imputation": {k: int(v) for k, v in missing_before.to_dict().items()},
        "numeric_features": feature_columns,
        "outlier_method": "IQR capping",
        "scaling_method": "StandardScaler fitted only on training split",
        "class_distribution_full": {str(k): int(v) for k, v in full_data[TARGET_COLUMN].value_counts().sort_index().to_dict().items()},
        "train_shape": list(train_out.shape),
        "test_shape": list(test_out.shape),
    }

    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    (output_dir / "feature_columns.json").write_text(json.dumps({"features": feature_columns, "target": TARGET_COLUMN}, indent=2), encoding="utf-8")
    (output_dir / "preprocessing_report.json").write_text(json.dumps(preprocessing_report, indent=2), encoding="utf-8")

    print(f"Dataset preprocessing telah disimpan di: {output_dir}")
    print(json.dumps(metadata, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Automate preprocessing for UCI Breast Cancer dataset")
    parser.add_argument("--raw-path", default="breast_cancer_raw/breast_cancer_raw.csv")
    parser.add_argument("--output-dir", default="preprocessing/breast_cancer_preprocessing")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    preprocess(args.raw_path, args.output_dir)
