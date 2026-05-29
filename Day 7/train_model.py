"""Train and save the passenger crowd prediction model."""

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "text_dataset" / "passenger_detail.csv"
MODEL_PATH = BASE_DIR / "random_forest_model.pkl"
WEB_MODEL_DATA_PATH = BASE_DIR / "web" / "model_data.json"

TARGET_COLUMN = "crowd_level"
NUMERIC_FEATURES = ["age", "fare"]
CATEGORICAL_FEATURES = [
    "gender",
    "booking_type",
    "source",
    "destination",
    "ticket_status",
]
FEATURE_COLUMNS = ["age", "gender", "booking_type", "source", "destination", "fare", "ticket_status"]
VALID_LABELS = ["low", "medium", "high"]


def load_dataset() -> tuple[pd.DataFrame, pd.Series]:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)
    required = FEATURE_COLUMNS + [TARGET_COLUMN]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df.dropna(subset=required).copy()
    if df.empty:
        raise ValueError("No usable rows remain after dropping missing values.")

    df[TARGET_COLUMN] = df[TARGET_COLUMN].astype(str).str.strip().str.lower()
    df = df[df[TARGET_COLUMN].isin(VALID_LABELS)].copy()
    if df.empty:
        raise ValueError(f"No valid target labels found. Expected: {VALID_LABELS}")

    for column in NUMERIC_FEATURES:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df = df.dropna(subset=NUMERIC_FEATURES).copy()
    if df.empty:
        raise ValueError("No usable rows remain after numeric conversion.")

    for column in CATEGORICAL_FEATURES:
        df[column] = df[column].astype(str).str.strip()

    return df[FEATURE_COLUMNS], df[TARGET_COLUMN]


def build_pipeline() -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", "passthrough", NUMERIC_FEATURES),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_FEATURES,
            ),
        ]
    )

    classifier = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        class_weight="balanced",
    )

    return Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", classifier),
        ]
    )


def get_options(X: pd.DataFrame) -> dict[str, list[str]]:
    return {
        column: sorted(X[column].dropna().astype(str).str.strip().unique().tolist())
        for column in CATEGORICAL_FEATURES
    }


def get_fare_ranges(X: pd.DataFrame, y: pd.Series) -> dict[str, dict[str, int]]:
    summary = (
        pd.DataFrame({"fare": X["fare"], TARGET_COLUMN: y})
        .groupby(TARGET_COLUMN)["fare"]
        .agg(["min", "max"])
        .to_dict("index")
    )
    return {
        label: {key: int(value) for key, value in ranges.items()}
        for label, ranges in summary.items()
    }


def export_web_model_data(bundle: dict) -> None:
    metrics = bundle["metrics"]
    WEB_MODEL_DATA_PATH.write_text(
        json.dumps(
            {
                "options": bundle["options"],
                "labels": [label.upper() for label in bundle["labels"]],
                "metrics": metrics,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def main() -> None:
    X, y = load_dataset()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"Rows used: {len(X)}")
    print(f"Features: {FEATURE_COLUMNS}")
    print(f"Model accuracy: {accuracy:.4f}")
    print("\nConfusion matrix:")
    print(confusion_matrix(y_test, y_pred, labels=VALID_LABELS))
    print("\nClassification report:")
    print(classification_report(y_test, y_pred, labels=VALID_LABELS))

    bundle = {
        "model": pipeline,
        "feature_columns": FEATURE_COLUMNS,
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "labels": VALID_LABELS,
        "options": get_options(X),
        "metrics": {
            "accuracy": round(float(accuracy), 4),
            "rows_used": int(len(X)),
            "target_counts": {label: int(count) for label, count in y.value_counts().items()},
            "fare_ranges": get_fare_ranges(X, y),
        },
    }

    joblib.dump(bundle, MODEL_PATH)
    export_web_model_data(bundle)
    print(f"\nModel bundle saved to: {MODEL_PATH}")
    print(f"Web model data saved to: {WEB_MODEL_DATA_PATH}")


if __name__ == "__main__":
    main()
