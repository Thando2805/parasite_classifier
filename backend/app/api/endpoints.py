from fastapi import APIRouter, UploadFile, File, HTTPException
from app.services.inference import run_inference

router = APIRouter()

@router.post("/predict")
async def predict_image(file: UploadFile = File(...), conf: float = 0.25):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image.")
    
    image_bytes = await file.read()
    detections = run_inference(image_bytes, conf_threshold=conf)
    
    return {
        "filename": file.filename,
        "success": True,
        "detections": detections
    }