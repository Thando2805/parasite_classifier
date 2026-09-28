from ultralytics import YOLO
import cv2
import numpy as np

# Points to backend/models/best.pt relative to where uvicorn runs
model = YOLO("models/best.pt")

def run_inference(image_bytes: bytes, conf_threshold: float = 0.25):
    np_img = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
    
    results = model.predict(source=img, conf=conf_threshold)
    
    detections = []
    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            xyxy = box.xyxy[0].tolist()
            class_name = model.names[cls_id]
            
            detections.append({
                "class": class_name,
                "confidence": round(conf, 4),
                "bbox": [round(coord, 2) for coord in xyxy]
            })
            
    return detections