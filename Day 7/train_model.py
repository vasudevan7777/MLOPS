"""Train and save the railway crowd forecasting model."""

import json
from datetime import datetime, timedelta
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


BASE_DIR = Path(__file__).resolve().parent
SOURCE_DATA_PATH = BASE_DIR / "text_dataset" / "passenger_detail.csv"
DATASET_PATH = BASE_DIR / "text_dataset" / "railway_demand.csv"
MODEL_PATH = BASE_DIR / "random_forest_model.pkl"
WEB_MODEL_DATA_PATH = BASE_DIR / "web" / "model_data.json"

TARGET_COLUMN = "crowd_level"
NUMERIC_FEATURES = [
    "train_capacity",
    "booked_seats",
    "available_seats",
    "waiting_list_count",
]
CATEGORICAL_FEATURES = [
    "day_type",
    "festival_flag",
    "source_station",
    "destination_station",
]
FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES
VALID_LABELS = ["low", "medium", "high"]
OCCUPANCY_THRESHOLDS = {
    "low": (0.0, 0.40),
    "medium": (0.41, 0.75),
    "high": (0.76, 1.0),
}

STATION_FALLBACK = [
    "Chennai",
    "Bangalore",
    "Mumbai",
    "Delhi",
    "Hyderabad",
    "Kolkata",
    "Pune",
    "Jaipur",
    "Lucknow",
    "Kochi",
    "Madurai",
    "Trichy",
]


def occupancy_to_label(rate: float) -> str:
    if rate <= OCCUPANCY_THRESHOLDS["low"][1]:
        return "low"
    if rate <= OCCUPANCY_THRESHOLDS["medium"][1]:
        return "medium"
    return "high"


def normalize_route_frame(source_df: pd.DataFrame | None) -> pd.DataFrame:
    if source_df is None or source_df.empty:
        rng = np.random.default_rng(42)
        size = 4000
        sources = rng.choice(STATION_FALLBACK, size=size, replace=True)
        destinations = rng.choice(STATION_FALLBACK, size=size, replace=True)
        routes = pd.DataFrame({"source": sources, "destination": destinations})
        routes = routes[routes["source"] != routes["destination"]].copy()
        return routes.reset_index(drop=True)

    if "source" not in source_df.columns or "destination" not in source_df.columns:
        return normalize_route_frame(None)

    routes = source_df[["source", "destination"]].dropna().copy()
    routes = routes[routes["source"] != routes["destination"]].copy()
    if routes.empty:
        return normalize_route_frame(None)
    return routes.reset_index(drop=True)


def generate_travel_dates(rng: np.random.Generator, size: int) -> list[datetime]:
    start = datetime(2024, 1, 1)
    end = datetime(2025, 12, 31)
    delta_days = (end - start).days
    offsets = rng.integers(0, delta_days + 1, size=size)
    return [start + timedelta(days=int(offset)) for offset in offsets]


def create_demand_dataset() -> pd.DataFrame:
    source_df = pd.read_csv(SOURCE_DATA_PATH) if SOURCE_DATA_PATH.exists() else None
    routes = normalize_route_frame(source_df)

    rng = np.random.default_rng(42)
    size = len(routes)
    travel_dates = generate_travel_dates(rng, size)
    day_types = ["Weekend" if date.weekday() >= 5 else "Weekday" for date in travel_dates]

    route_tuples = list(zip(routes["source"], routes["destination"]))
    route_counts = pd.Series(route_tuples).value_counts()
    max_count = float(route_counts.max()) if not route_counts.empty else 1.0
    popularity = np.array([route_counts[route] / max_count for route in route_tuples])

    capacities = []
    for score in popularity:
        if score >= 0.7:
            capacities.append(1200)
        elif score >= 0.5:
            capacities.append(1000)
        elif score >= 0.3:
            capacities.append(800)
        else:
            capacities.append(600)

    capacities = np.array(capacities)
    occupancy_rates = 0.25 + 0.6 * popularity + rng.normal(0, 0.08, size=size)
    occupancy_rates = np.clip(occupancy_rates, 0.15, 1.0)
    booked_seats = np.round(capacities * occupancy_rates).astype(int)
    available_seats = np.maximum(capacities - booked_seats, 0)

    waiting_list = []
    for rate in occupancy_rates:
        if rate >= 0.85:
            waiting_list.append(int(rng.integers(40, 220)))
        elif rate >= 0.7:
            waiting_list.append(int(rng.integers(10, 120)))
        else:
            waiting_list.append(int(rng.integers(0, 60)))

    festival_flag = []
    for day_type in day_types:
        probability = 0.18 if day_type == "Weekend" else 0.08
        festival_flag.append("Yes" if rng.random() < probability else "No")

    route_ids = {route: idx + 1001 for idx, route in enumerate(route_counts.index)}
    train_ids = [f"TR{route_ids[route]:04d}" for route in route_tuples]

    crowd_levels = [occupancy_to_label(rate) for rate in occupancy_rates]

    dataset = pd.DataFrame(
        {
            "train_id": train_ids,
            "source_station": routes["source"].astype(str).str.strip(),
            "destination_station": routes["destination"].astype(str).str.strip(),
            "train_capacity": capacities,
            "booked_seats": booked_seats,
            "available_seats": available_seats,
            "waiting_list_count": waiting_list,
            "day_type": day_types,
            "festival_flag": festival_flag,
            "travel_date": [date.strftime("%Y-%m-%d") for date in travel_dates],
            "crowd_level": crowd_levels,
        }
    )

    DATASET_PATH.write_text(dataset.to_csv(index=False), encoding="utf-8")
    return dataset


def load_dataset() -> pd.DataFrame:
    if DATASET_PATH.exists():
        df = pd.read_csv(DATASET_PATH)
    else:
        df = create_demand_dataset()

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

    return df


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
    options = {
        column: sorted(X[column].dropna().astype(str).str.strip().unique().tolist())
        for column in CATEGORICAL_FEATURES
    }
    if "festival_flag" not in options:
        options["festival_flag"] = ["No", "Yes"]
    return options


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
    df = load_dataset()
    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

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
            "occupancy_thresholds": {
                "low": {"min": 0, "max": 40},
                "medium": {"min": 41, "max": 75},
                "high": {"min": 76, "max": 100},
            },
        },
    }

    joblib.dump(bundle, MODEL_PATH)
    export_web_model_data(bundle)
    print(f"\nModel bundle saved to: {MODEL_PATH}")
    print(f"Web model data saved to: {WEB_MODEL_DATA_PATH}")


if __name__ == "__main__":
    main()
