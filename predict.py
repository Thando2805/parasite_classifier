from ultralytics import YOLO

# 1. Load your custom model weights
model = YOLO("backend/models/best.pt")

# 2. Run prediction using your exact test images path
results = model.predict(
    source="dataset_pipeline/dataset/test/images",
    conf=0.25,
    save=True
)

print("Inference complete! Check runs/detect/predict/")