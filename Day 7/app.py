from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, jsonify, render_template, request


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "random_forest_model.pkl"
WEB_DIR = BASE_DIR / "web"

app = Flask(
    __name__,
    template_folder=str(WEB_DIR),
    static_folder=str(WEB_DIR),
    static_url_path="/static",
)

model_bundle = None


@app.after_request
def add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    return response


def get_model_bundle() -> dict:
    global model_bundle
    if model_bundle is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError("Model file not found. Run train_model.py first.")

        loaded = joblib.load(MODEL_PATH)
        if isinstance(loaded, dict) and "model" in loaded:
            model_bundle = loaded
        else:
            model_bundle = {
                "model": loaded,
                "feature_columns": [
                    "age",
                    "gender",
                    "booking_type",
                    "source",
                    "destination",
                    "fare",
                    "ticket_status",
                ],
                "numeric_features": ["age", "fare"],
                "categorical_features": [
                    "gender",
                    "booking_type",
                    "source",
                    "destination",
                    "ticket_status",
                ],
                "labels": ["low", "medium", "high"],
                "options": {},
                "metrics": {},
            }
    return model_bundle


def first_value(payload: dict, *names: str):
    for name in names:
        value = payload.get(name)
        if value not in (None, ""):
            return value
    return None


def parse_float(payload: dict, field: str) -> float:
    value = first_value(payload, field)
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be a number.") from exc


def parse_text(payload: dict, field: str, legacy_field: str | None = None) -> str:
    value = first_value(payload, field, legacy_field or field)
    if value is None:
        raise ValueError(f"{field} is required.")
    return str(value).strip()


def build_input_frame(payload: dict) -> pd.DataFrame:
    bundle = get_model_bundle()
    row = {
        "age": parse_float(payload, "age"),
        "gender": parse_text(payload, "gender"),
        "booking_type": parse_text(payload, "booking_type", "bookingType"),
        "source": parse_text(payload, "source"),
        "destination": parse_text(payload, "destination"),
        "fare": parse_float(payload, "fare"),
        "ticket_status": parse_text(payload, "ticket_status", "ticketStatus"),
    }

    return pd.DataFrame([row], columns=bundle["feature_columns"])


@app.get("/")
def home():
    return render_template("index.html")


@app.get("/api/options")
def options():
    bundle = get_model_bundle()
    return jsonify(
        {
            "options": bundle.get("options", {}),
            "features": bundle["feature_columns"],
            "labels": [label.upper() for label in bundle.get("labels", [])],
        }
    )


@app.get("/api/model-info")
def model_info():
    bundle = get_model_bundle()
    metrics = bundle.get("metrics", {})
    return jsonify(
        {
            "accuracy": metrics.get("accuracy"),
            "rows_used": metrics.get("rows_used"),
            "target_counts": metrics.get("target_counts", {}),
            "fare_ranges": metrics.get("fare_ranges", {}),
        }
    )


@app.post("/predict")
def predict():
    try:
        payload = request.get_json(silent=True) or {}
        input_df = build_input_frame(payload)
        bundle = get_model_bundle()
        model = bundle["model"]

        prediction = str(model.predict(input_df)[0]).strip().lower()
        probabilities = {}
        if hasattr(model, "predict_proba"):
            probabilities = {
                str(label).upper(): round(float(probability), 4)
                for label, probability in zip(model.classes_, model.predict_proba(input_df)[0])
            }

        return jsonify(
            {
                "prediction": prediction.upper(),
                "probabilities": probabilities,
            }
        )
    except Exception as exc:
        app.logger.exception("Prediction failed")
        return jsonify({"error": str(exc)}), 400


if __name__ == "__main__":
    app.run(debug=True, port=5002)
