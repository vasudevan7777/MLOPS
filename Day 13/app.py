import os
import time
import uuid
from pathlib import Path

import cv2
from flask import Flask, jsonify, render_template, request, url_for
from ultralytics import YOLO
from werkzeug.utils import secure_filename


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "best_80epoch.pt"
UPLOAD_DIR = BASE_DIR / "static" / "uploads"
PREDICTION_DIR = BASE_DIR / "static" / "predictions"

ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "mp4", "avi", "mov"}
IMAGE_EXTENSIONS = {"jpg", "jpeg", "png"}
VIDEO_EXTENSIONS = {"mp4", "avi", "mov"}

MAX_VIDEO_FRAMES = int(os.getenv("MAX_VIDEO_FRAMES", "600"))
VIDEO_FRAME_STRIDE = int(os.getenv("VIDEO_FRAME_STRIDE", "2"))
MODEL_IMAGE_SIZE = int(os.getenv("MODEL_IMAGE_SIZE", "640"))
MODEL_CONFIDENCE = float(os.getenv("MODEL_CONFIDENCE", "0.25"))

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = int(os.getenv("MAX_UPLOAD_MB", "100")) * 1024 * 1024

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
PREDICTION_DIR.mkdir(parents=True, exist_ok=True)

model = YOLO(str(MODEL_PATH))


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def file_kind(filename):
    extension = filename.rsplit(".", 1)[1].lower()
    if extension in IMAGE_EXTENSIONS:
        return "image"
    if extension in VIDEO_EXTENSIONS:
        return "video"
    return None


def crowd_level(person_count):
    if person_count <= 5:
        return {"label": "Low Crowd", "status": "low"}
    if person_count <= 15:
        return {"label": "Medium Crowd", "status": "medium"}
    return {"label": "High Crowd", "status": "high"}


def person_stats(result):
    person_count = 0
    confidences = []

    for box in result.boxes:
        cls = int(box.cls[0])
        if model.names[cls] == "person":
            person_count += 1
            confidences.append(float(box.conf[0]))

    average_confidence = sum(confidences) / len(confidences) if confidences else 0.0
    return person_count, average_confidence


def cleanup_old_files(max_age_seconds=60 * 60 * 2):
    now = time.time()
    for folder in (UPLOAD_DIR, PREDICTION_DIR):
        for item in folder.iterdir():
            if item.is_file() and now - item.stat().st_mtime > max_age_seconds:
                item.unlink(missing_ok=True)


def predict_image(input_path):
    result = model.predict(
        source=str(input_path),
        imgsz=MODEL_IMAGE_SIZE,
        conf=MODEL_CONFIDENCE,
        verbose=False,
    )[0]

    person_count, confidence = person_stats(result)
    output_name = f"{input_path.stem}_prediction.jpg"
    output_path = PREDICTION_DIR / output_name
    cv2.imwrite(str(output_path), result.plot())

    return {
        "media_type": "image",
        "media_url": url_for("static", filename=f"predictions/{output_name}"),
        "passenger_count": person_count,
        "confidence": round(confidence * 100, 2),
        "crowd": crowd_level(person_count),
    }


def predict_video(input_path):
    capture = cv2.VideoCapture(str(input_path))
    if not capture.isOpened():
        raise ValueError("Could not open the uploaded video.")

    fps = capture.get(cv2.CAP_PROP_FPS) or 24
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    output_name = f"{input_path.stem}_prediction.mp4"
    output_path = PREDICTION_DIR / output_name
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps / max(VIDEO_FRAME_STRIDE, 1),
        (width, height),
    )

    processed_frames = 0
    max_person_count = 0
    total_person_detections = 0
    confidence_sum = 0.0
    confidence_count = 0
    frame_index = 0

    while processed_frames < MAX_VIDEO_FRAMES:
        success, frame = capture.read()
        if not success:
            break

        if frame_index % max(VIDEO_FRAME_STRIDE, 1) == 0:
            result = model.predict(
                source=frame,
                imgsz=MODEL_IMAGE_SIZE,
                conf=MODEL_CONFIDENCE,
                verbose=False,
            )[0]
            person_count, confidence = person_stats(result)
            max_person_count = max(max_person_count, person_count)
            total_person_detections += person_count
            if person_count:
                confidence_sum += confidence
                confidence_count += 1
            writer.write(result.plot())
            processed_frames += 1

        frame_index += 1

    capture.release()
    writer.release()

    if processed_frames == 0:
        raise ValueError("No readable frames were found in the uploaded video.")

    average_confidence = confidence_sum / confidence_count if confidence_count else 0.0

    return {
        "media_type": "video",
        "media_url": url_for("static", filename=f"predictions/{output_name}"),
        "passenger_count": max_person_count,
        "total_detections": total_person_detections,
        "processed_frames": processed_frames,
        "confidence": round(average_confidence * 100, 2),
        "crowd": crowd_level(max_person_count),
    }


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    started_at = time.perf_counter()
    cleanup_old_files()

    if "file" not in request.files:
        return jsonify({"error": "Please choose an image or video file."}), 400

    uploaded_file = request.files["file"]
    if not uploaded_file.filename:
        return jsonify({"error": "No file was selected."}), 400

    if not allowed_file(uploaded_file.filename):
        return jsonify({"error": "Supported formats: jpg, jpeg, png, mp4, avi, mov."}), 400

    original_name = secure_filename(uploaded_file.filename)
    extension = original_name.rsplit(".", 1)[1].lower()
    unique_name = f"{uuid.uuid4().hex}_{Path(original_name).stem}.{extension}"
    input_path = UPLOAD_DIR / unique_name
    uploaded_file.save(input_path)

    try:
        kind = file_kind(original_name)
        result = predict_image(input_path) if kind == "image" else predict_video(input_path)
        result["processing_time"] = round(time.perf_counter() - started_at, 2)
        result["filename"] = original_name
        return jsonify(result)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.errorhandler(413)
def too_large(_error):
    return jsonify({"error": f"File is too large. Maximum upload size is {os.getenv('MAX_UPLOAD_MB', '100')} MB."}), 413


if __name__ == "__main__":
    port = int(os.getenv("PORT", "5000"))
    app.run(host="0.0.0.0", port=port, debug=False)
