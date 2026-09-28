import os
import base64
import cv2
import numpy as np
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from ultralytics import YOLO

model = None

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "best.pt")

def find_index_file():
    # 1. Direct path checks
    check_paths = [
        os.path.join(BASE_DIR, "index.html"),
        os.path.join(BASE_DIR, "frontend", "index.html"),
        os.path.join(BASE_DIR, "..", "frontend", "index.html"),
        os.path.join(BASE_DIR, "..", "..", "frontend", "index.html"),
        os.path.join(os.getcwd(), "index.html"),
        os.path.join(os.getcwd(), "frontend", "index.html"),
    ]
    for path in check_paths:
        abs_p = os.path.abspath(path)
        if os.path.exists(abs_p):
            return abs_p

    # 2. Dynamic directory search across container working directory
    search_root = os.path.abspath(os.path.join(BASE_DIR, "..", ".."))
    for root, dirs, files in os.walk(search_root):
        if "index.html" in files:
            return os.path.join(root, "index.html")
            
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