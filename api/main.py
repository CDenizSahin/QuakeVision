from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
import torch
import numpy as np
from PIL import Image
import io
import os
from ultralytics import YOLO
from api.services.llm_service import generate_disaster_report, ReportRequest

app = FastAPI(title="QuakeVision Backend API")

# Load model globally
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "best.pt")
model = None

@app.on_event("startup")
def load_model():
    global model
    if os.path.exists(MODEL_PATH):
        model = YOLO(MODEL_PATH)
        print(f"Model successfully loaded from {MODEL_PATH}")
    else:
        print(f"Warning: Model not found at {MODEL_PATH}")

@app.get("/health")
def health_check():
    return {"status": "ok", "model_loaded": model is not None}

@app.post("/analyze/image")
async def analyze_image(
    file: UploadFile = File(...),
    conf_threshold: float = Form(0.15),
    iou_threshold: float = Form(0.45)
):
    if model is None:
        raise HTTPException(status_code=500, detail="Model is not loaded on the server.")
        
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image file: {e}")

    # Inference
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        
    try:
        # En Keskin (Ultra HD) Ayarlar ve 750 Donanım Limiti
        results = model.predict(
            source=image, 
            conf=conf_threshold, 
            iou=iou_threshold,
            imgsz=1536,
            max_det=750,
            half=True,
            retina_masks=False,
            device=0 if torch.cuda.is_available() else "cpu"
        )
    except Exception as e:
        if "out of memory" in str(e).lower():
            return JSONResponse(status_code=500, content={"message": "Ekran kartı belleği (VRAM) doldu! Lütfen güven eşiğini (Confidence) 0.15 veya daha yüksek bir değere çekip tekrar deneyin."})
        return JSONResponse(status_code=500, content={"message": f"Analiz hatası: {str(e)}"})
    
    # Process results
    results_data = []
    class_labels = {0: "Sağlam", 1: "Hafif Hasarlı", 2: "Ağır Hasarlı", 3: "Yıkılmış"}
    
    try:
        for r in results:
            if r.boxes is not None and r.masks is not None:
                classes = r.boxes.cls.cpu().numpy()
                confs = r.boxes.conf.cpu().numpy()
                masks = r.masks.xy
                
                for mask, cls, conf in zip(masks, classes, confs):
                    cls_id = int(cls)
                    
                    # Calculate area and center using Shoelace
                    x = mask[:, 0]
                    y = mask[:, 1]
                    area = 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
                    cx, cy = np.mean(x), np.mean(y)
                    
                    polygon = [[int(px), int(py)] for px, py in zip(x, y)]
                    
                    results_data.append({
                        "class_id": cls_id,
                        "class_name": class_labels.get(cls_id, "Bilinmiyor"),
                        "confidence": float(conf),
                        "area_px": float(area),
                        "cx": float(cx),
                        "cy": float(cy),
                        "polygon": polygon
                    })
    except RuntimeError as e:
        if "not enough memory" in str(e).lower() or "out of memory" in str(e).lower():
            return JSONResponse(status_code=500, content={"message": "Sistem belleği (RAM) yetersiz! Ekranda çok fazla bina bulundu. Lütfen Güven Eşiğini (Confidence) 0.25'e yükseltip tekrar deneyin."})
        return JSONResponse(status_code=500, content={"message": f"Sistem hatası: {str(e)}"})
    except Exception as e:
        return JSONResponse(status_code=500, content={"message": f"Beklenmeyen hata: {str(e)}"})
    # VRAM Sızıntısını Önlemek için Belleği Zorla Temizle
    import gc
    try:
        del results
    except:
        pass
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
                
    return JSONResponse(content={
        "width": image.width,
        "height": image.height,
        "detections": results_data
    })

@app.post("/generate_report")
async def create_report(request: ReportRequest):
    # This calls the local Ollama LLM to generate the text
    report_text = generate_disaster_report(request)
    return {"report": report_text}
