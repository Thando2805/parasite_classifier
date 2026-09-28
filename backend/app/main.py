from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import cv2
import base64
import numpy as np
from ultralytics import YOLO
import os
from contextlib import asynccontextmanager

model = None

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "best.pt")

# Flexible search paths for index.html depending on repo structure
POSSIBLE_INDEX_PATHS = [
    os.path.join(BASE_DIR, "..", "..", "frontend", "index.html"),
    os.path.join(BASE_DIR, "..", "frontend", "index.html"),
    os.path.join(BASE_DIR, "frontend", "index.html"),
    os.path.join(BASE_DIR, "index.html"),
]

def find_index_file():
    for path in POSSIBLE_INDEX_PATHS:
        abs_path = os.path.abspath(path)
        if os.path.exists(abs_path):
            return abs_path
    return None

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

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve the mobile web UI directly on the root URL
@app.get("/")
def read_root():
    index_file = find_index_file()
    if index_file:
        return FileResponse(index_file)
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
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)