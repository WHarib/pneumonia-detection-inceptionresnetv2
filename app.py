import io, base64, os
import numpy as np
from PIL import Image
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import tensorflow as tf
from tensorflow import keras
import cv2

# ---------------- CONFIG ----------------
# Use the local file you already COPY into the image
MODEL_PATH = os.getenv("MODEL_PATH", "model.keras")

IMG_SIZE   = (200, 200)          # what you trained on
THRESHOLD  = float(os.getenv("THRESHOLD", 0.5))   # change if you want
# ----------------------------------------

app = FastAPI(title="Pneumonia Detection API (InceptionResNetV2)")

_model = None

def get_model():
    global _model
    if _model is None:
        # Just load locally; no hf_hub_download
        _model = keras.models.load_model(MODEL_PATH, compile=False)
    return _model

def preprocess(img: Image.Image) -> np.ndarray:
    arr = np.array(img.resize(IMG_SIZE)) / 255.0
    return np.expand_dims(arr.astype("float32"), 0)

def overlay(img: Image.Image, label: str) -> str:
    colour = (255, 0, 0) if label == "PNEUMONIA" else (0, 255, 0)
    buf = io.BytesIO()
    Image.blend(img, Image.new("RGB", img.size, colour), 0.25).save(buf, "PNG")
    return base64.b64encode(buf.getvalue()).decode()

@app.get("/")
def root():
    return {"ok": True, "message": "POST an X-ray to /predict"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        pil = Image.open(io.BytesIO(await file.read())).convert("RGB")
    except Exception as e:
        raise HTTPException(400, f"Bad image: {e}")

    model = get_model()
    pred = float(model.predict(preprocess(pil))[0][0])  # assuming index 0 is pneumonia prob as before
    label = "PNEUMONIA" if pred > THRESHOLD else "NORMAL"
    conf  = round(pred if label == "PNEUMONIA" else 1 - pred, 4)

    return JSONResponse({
        "diagnosis": label,
        "confidence": conf,
        "raw_score": pred,
        "threshold": THRESHOLD,
        "processed_image": overlay(pil, label)
    })