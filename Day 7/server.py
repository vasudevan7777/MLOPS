from pathlib import Path

from flask import Flask, jsonify, render_template, request

from util import build_input_frame, get_analytics_bundle, get_model_bundle


BASE_DIR = Path(__file__).resolve().parent
WEB_DIR = BASE_DIR / "web"

app = Flask(
    __name__,
    template_folder=str(WEB_DIR),
    static_folder=str(WEB_DIR),
    static_url_path="",
)


@app.after_request
def add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    return response


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
            "occupancy_thresholds": metrics.get("occupancy_thresholds", {}),
        }
    )


@app.get("/api/analytics")
def analytics():
    return jsonify(get_analytics_bundle())


@app.post("/predict")
@app.post("/api/predict")
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
