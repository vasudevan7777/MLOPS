from ultralytics import YOLO

model = YOLO("best_80epoch.pt")

results = model.predict(
    source="test.jpg",
    save=True
)

print("Prediction Completed")