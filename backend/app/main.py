from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import cv2
import base64
import numpy as np
from ultralytics import YOLO
import os
from contextlib import asynccontextmanager

model = None

# Automatically find 'best.pt' in the exact same folder as this main.py file
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "best.pt")

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    if os.path.exists(MODEL_PATH):
        print(f"Loading YOLO model from {MODEL_PATH}...")
        model = YOLO(MODEL_PATH)
        print("Model loaded successfully!")
    else:
        print(f"WARNING: '{MODEL_PATH}' not found!")
    yield

app = FastAPI(lifespan=lifespan)

# Allow frontend requests from any origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Root endpoint (fixes 404 on base URL)
@app.get("/")
def read_root():
    return {"status": "online", "message": "Parasite Classifier API is running!"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    global model
    if model is None:
        if os.path.exists(MODEL_PATH):
            model = YOLO(MODEL_PATH)
        else:
            return {"success": False, "error": f"Model weights not found at {MODEL_PATH}."}

    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        return {"success": False, "error": "Could not decode image."}

    results = model(img)
    plotted_img = results[0].plot()
    
    _, buffer = cv2.imencode('.jpg', plotted_img)
    annotated_base64 = base64.b64encode(buffer).decode('utf-8')
    annotated_data_url = f"data:image/jpeg;base64,{annotated_base64}"

    detections = []
    for box in results[0].boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        cls_name = model.names[cls_id]
        detections.append({"class": cls_name, "confidence": conf})

    return {
        "success": True,
        "detections": detections,
        "annotated_image": annotated_data_url
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=True)