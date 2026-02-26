from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


DEFAULT_DATA_PATH = "pricing_dataset.csv"
DEFAULT_TARGET = "Units Sold"
DEFAULT_MODEL_PATH = "artifacts/pricing_demand_model.joblib"
DEFAULT_METRICS_PATH = "artifacts/pricing_demand_metrics.json"
RANDOM_STATE = 42


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a pricing demand model with preprocessing pipeline."
    )
    parser.add_argument("--data-path", default=DEFAULT_DATA_PATH, help="Path to CSV dataset")
    parser.add_argument("--target", default=DEFAULT_TARGET, help="Target column to predict")
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Test split ratio (default: 0.2)",
    )
    parser.add_argument(
        "--model-out",
        default=DEFAULT_MODEL_PATH,
        help="Output path for the trained model pipeline (.joblib)",
    )
    parser.add_argument(
        "--metrics-out",
        default=DEFAULT_METRICS_PATH,
        help="Output path for evaluation metrics (.json)",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=RANDOM_STATE,
        help="Random seed",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Train/evaluate only, do not save artifacts",
    )
    return parser.parse_args()


def load_dataset(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    if "Date" in out.columns:
        dt = pd.to_datetime(out["Date"], errors="coerce")
        out["Date_year"] = dt.dt.year
        out["Date_month"] = dt.dt.month
        out["Date_day"] = dt.dt.day
        out["Date_dayofweek"] = dt.dt.dayofweek
        out["Date_is_month_start"] = dt.dt.is_month_start.astype("float")
        out["Date_is_month_end"] = dt.dt.is_month_end.astype("float")
        out = out.drop(columns=["Date"])

    return out


def leakage_columns_for_target(target: str) -> list[str]:
    # Revenue directly depends on price * units; drop it when predicting units.
    if target == "Units Sold":
        return ["Revenue ($)"]
    if target == "Revenue ($)":
        return ["Units Sold"]
    return []


def build_one_hot_encoder() -> OneHotEncoder:
    # sklearn compatibility (older vs newer versions)
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def build_pipeline(
    X: pd.DataFrame, random_state: int = RANDOM_STATE
) -> tuple[Pipeline, list[str], list[str]]:
    numeric_cols = list(X.select_dtypes(include=["number"]).columns)
    categorical_cols = list(
        X.select_dtypes(include=["object", "category", "bool"]).columns
    )

    numeric_transformer = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median"))]
    )
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", build_one_hot_encoder()),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols),
        ],
        remainder="drop",
    )

    model = RandomForestRegressor(
        n_estimators=300,
        min_samples_leaf=2,
        random_state=random_state,
        n_jobs=1,
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )
    return pipeline, numeric_cols, categorical_cols


def prepare_training_data(df: pd.DataFrame, target: str) -> tuple[pd.DataFrame, pd.Series]:
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found in dataset.")

    work = engineer_features(df)

    drop_cols = [target]
    for col in leakage_columns_for_target(target):
        if col in work.columns and col not in drop_cols:
            drop_cols.append(col)

    X = work.drop(columns=drop_cols, errors="ignore")
    y = pd.to_numeric(work[target], errors="coerce")

    valid_mask = y.notna()
    X = X.loc[valid_mask].copy()
    y = y.loc[valid_mask].copy()

    if X.empty:
        raise ValueError("No training rows available after preprocessing.")

    return X, y


def train_and_evaluate(
    df: pd.DataFrame,
    target: str = DEFAULT_TARGET,
    test_size: float = 0.2,
    random_state: int = RANDOM_STATE,
) -> tuple[Pipeline, dict, dict]:
    X, y = prepare_training_data(df, target)
    pipeline, numeric_cols, categorical_cols = build_pipeline(X, random_state=random_state)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    metrics = {
        "target": target,
        "rows_total": int(len(X)),
        "rows_train": int(len(X_train)),
        "rows_test": int(len(X_test)),
        "mae": float(mean_absolute_error(y_test, y_pred)),
        "rmse": rmse,
        "r2": float(r2_score(y_test, y_pred)),
    }

    metadata = {
        "feature_count_raw": int(X.shape[1]),
        "numeric_features": numeric_cols,
        "categorical_features": categorical_cols,
        "dropped_for_leakage": leakage_columns_for_target(target),
        "engineered_features": [
            "Date_year",
            "Date_month",
            "Date_day",
            "Date_dayofweek",
            "Date_is_month_start",
            "Date_is_month_end",
        ],
        "model_type": "RandomForestRegressor",
        "test_size": test_size,
        "random_state": random_state,
    }

    return pipeline, metrics, metadata


def save_artifacts(
    pipeline: Pipeline,
    metrics: dict,
    metadata: dict,
    model_out: str,
    metrics_out: str,
) -> None:
    model_path = Path(model_out)
    metrics_path = Path(metrics_out)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    dump(pipeline, model_path)
    payload = {"metrics": metrics, "metadata": metadata}
    metrics_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()

    df = load_dataset(args.data_path)
    pipeline, metrics, metadata = train_and_evaluate(
        df,
        target=args.target,
        test_size=args.test_size,
        random_state=args.random_state,
    )

    if not args.no_save:
        save_artifacts(
            pipeline,
            metrics,
            metadata,
            model_out=args.model_out,
            metrics_out=args.metrics_out,
        )

    print("\n=== Training Complete ===")
    print(f"Target       : {metrics['target']}")
    print(f"Rows         : {metrics['rows_total']} (train={metrics['rows_train']}, test={metrics['rows_test']})")
    print(f"MAE          : {metrics['mae']:.4f}")
    print(f"RMSE         : {metrics['rmse']:.4f}")
    print(f"R2           : {metrics['r2']:.4f}")
    print(f"Num features : {len(metadata['numeric_features'])}")
    print(f"Cat features : {len(metadata['categorical_features'])}")
    print(f"Dropped leak : {metadata['dropped_for_leakage']}")
    if not args.no_save:
        print(f"Model saved  : {args.model_out}")
        print(f"Metrics saved: {args.metrics_out}")


if __name__ == "__main__":
    main()
