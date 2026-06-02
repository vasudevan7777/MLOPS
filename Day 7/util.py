from pathlib import Path
import math

import joblib
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "random_forest_model.pkl"
DATA_PATH = BASE_DIR / "text_dataset" / "railway_demand.csv"

_model_bundle = None
_analytics_bundle = None


def get_model_bundle() -> dict:
    global _model_bundle
    if _model_bundle is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError("Model file not found. Run train_model.py first.")

        loaded = joblib.load(MODEL_PATH)
        if isinstance(loaded, dict) and "model" in loaded:
            _model_bundle = loaded
        else:
            _model_bundle = {
                "model": loaded,
                "feature_columns": [
                    "train_capacity",
                    "booked_seats",
                    "available_seats",
                    "waiting_list_count",
                    "day_type",
                    "festival_flag",
                    "source_station",
                    "destination_station",
                ],
                "numeric_features": [
                    "train_capacity",
                    "booked_seats",
                    "available_seats",
                    "waiting_list_count",
                ],
                "categorical_features": [
                    "day_type",
                    "festival_flag",
                    "source_station",
                    "destination_station",
                ],
                "labels": ["low", "medium", "high"],
                "options": {},
                "metrics": {},
            }
    return _model_bundle


def create_empty_analytics() -> dict:
    return {
        "dataset_summary": {},
        "route_stats": {"top_routes": []},
        "occupancy_distribution": {},
    }


def safe_mean(series: pd.Series, digits: int = 2) -> float | None:
    if series.empty:
        return None
    value = series.mean()
    if pd.isna(value):
        return None
    return round(float(value), digits)


def get_analytics_bundle() -> dict:
    global _analytics_bundle
    if _analytics_bundle is not None:
        return _analytics_bundle

    if not DATA_PATH.exists():
        _analytics_bundle = create_empty_analytics()
        return _analytics_bundle

    df = pd.read_csv(DATA_PATH)
    if df.empty:
        _analytics_bundle = create_empty_analytics()
        return _analytics_bundle

    summary = {"total_rows": int(len(df))}

    if "train_id" in df.columns:
        summary["total_trains"] = int(df["train_id"].dropna().nunique())
    if "source_station" in df.columns:
        summary["unique_sources"] = int(df["source_station"].dropna().nunique())
    if "destination_station" in df.columns:
        summary["unique_destinations"] = int(df["destination_station"].dropna().nunique())

    occupancy_rate = pd.Series(dtype=float)
    if "booked_seats" in df.columns and "train_capacity" in df.columns:
        booked = pd.to_numeric(df["booked_seats"], errors="coerce")
        capacity = pd.to_numeric(df["train_capacity"], errors="coerce")
        occupancy_rate = booked.divide(capacity.where(capacity > 0))
        summary["avg_occupancy"] = safe_mean(occupancy_rate.dropna() * 100, digits=2)
        summary["avg_capacity"] = safe_mean(capacity.dropna(), digits=0)

    if "waiting_list_count" in df.columns:
        waiting = pd.to_numeric(df["waiting_list_count"], errors="coerce").dropna()
        summary["avg_waiting_list"] = safe_mean(waiting, digits=1)

    top_routes = []
    if "source_station" in df.columns and "destination_station" in df.columns:
        route_summary = (
            df.groupby(["source_station", "destination_station"], dropna=True)
            .size()
            .sort_values(ascending=False)
            .head(5)
        )
        for (source, destination), count in route_summary.items():
            top_routes.append({"route": f"{source} -> {destination}", "count": int(count)})

    occupancy_distribution = {}
    if "crowd_level" in df.columns:
        counts = df["crowd_level"].astype(str).str.strip().str.lower().value_counts()
        occupancy_distribution = {label: int(count) for label, count in counts.items()}

    _analytics_bundle = {
        "dataset_summary": summary,
        "route_stats": {"top_routes": top_routes},
        "occupancy_distribution": occupancy_distribution,
    }
    return _analytics_bundle


def first_value(payload: dict, *names: str):
    for name in names:
        value = payload.get(name)
        if value not in (None, ""):
            return value
    return None


def parse_number(payload: dict, field: str) -> float:
    value = first_value(payload, field)
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be a number.") from exc
    if math.isnan(number) or math.isinf(number):
        raise ValueError(f"{field} must be a valid number.")
    return number


def parse_int(payload: dict, field: str) -> int:
    number = parse_number(payload, field)
    return int(round(number))


def parse_text(payload: dict, field: str, legacy_field: str | None = None) -> str:
    value = first_value(payload, field, legacy_field or field)
    if value is None:
        raise ValueError(f"{field} is required.")
    cleaned = str(value).strip()
    if not cleaned:
        raise ValueError(f"{field} is required.")
    return cleaned


def normalize_day_type(value: str) -> str:
    normalized = value.strip().lower()
    if normalized in {"weekday", "week day", "week-day"}:
        return "Weekday"
    if normalized in {"weekend", "week end", "week-end"}:
        return "Weekend"
    raise ValueError("day_type must be Weekday or Weekend.")


def normalize_festival_flag(value: str) -> str:
    normalized = value.strip().lower()
    if normalized in {"yes", "y", "true", "1"}:
        return "Yes"
    if normalized in {"no", "n", "false", "0"}:
        return "No"
    raise ValueError("festival_flag must be Yes or No.")


def validate_row(row: dict) -> None:
    if row["train_capacity"] <= 0:
        raise ValueError("train_capacity must be greater than zero.")
    if row["booked_seats"] < 0:
        raise ValueError("booked_seats cannot be negative.")
    if row["available_seats"] < 0:
        raise ValueError("available_seats cannot be negative.")
    if row["waiting_list_count"] < 0:
        raise ValueError("waiting_list_count cannot be negative.")
    if row["booked_seats"] > row["train_capacity"]:
        raise ValueError("booked_seats cannot exceed train_capacity.")
    if row["available_seats"] > row["train_capacity"]:
        raise ValueError("available_seats cannot exceed train_capacity.")
    if row["booked_seats"] + row["available_seats"] > row["train_capacity"]:
        raise ValueError("booked_seats + available_seats cannot exceed train_capacity.")
    if row["source_station"].strip().casefold() == row["destination_station"].strip().casefold():
        raise ValueError("source_station and destination_station must be different.")


def build_input_frame(payload: dict) -> pd.DataFrame:
    bundle = get_model_bundle()
    row = {
        "train_capacity": parse_int(payload, "train_capacity"),
        "booked_seats": parse_int(payload, "booked_seats"),
        "available_seats": parse_int(payload, "available_seats"),
        "waiting_list_count": parse_int(payload, "waiting_list_count"),
        "day_type": normalize_day_type(parse_text(payload, "day_type", "dayType")),
        "festival_flag": normalize_festival_flag(parse_text(payload, "festival_flag", "festivalFlag")),
        "source_station": parse_text(payload, "source_station", "sourceStation"),
        "destination_station": parse_text(payload, "destination_station", "destinationStation"),
    }

    validate_row(row)

    return pd.DataFrame([row], columns=bundle["feature_columns"])
