import io, base64
from typing import Optional
from pathlib import Path

import numpy as np
from PIL import Image
from fastapi import FastAPI, UploadFile, File, Header, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse
import tensorflow as tf
from tensorflow import keras
from huggingface_hub import hf_hub_download

# ---------------- CONFIG ----------------
REPO_ID   = "your-username/my-pneumonia-inceptionresnetv2"
FILENAME  = "model.keras"
IMG_SIZE  = (200, 200)          # change if you trained on another size
THRESHOLD = 0.5
EXPECT_KEY = None               # set to a string or env var if you want an API key
# ----------------------------------------

app = FastAPI(title="Pneumonia Detection API", docs_url="/docs", redoc_url="/redoc")
_model: Optional[keras.Model] = None

def ensure_auth(x_api_key: Optional[str]):
    if EXPECT_KEY and x_api_key != EXPECT_KEY:
        raise HTTPException(401, "Invalid or missing API key.")

def get_model() -> keras.Model:
    global _model
    if _model is None:
        local_path = hf_hub_download(repo_id=REPO_ID, filename=FILENAME)
        _model = keras.models.load_model(local_path, compile=False)
    return _model

def preprocess(img: Image.Image) -> np.ndarray:
    arr = np.array(img.resize(IMG_SIZE)) / 255.0
    return np.expand_dims(arr.astype("float32"), 0)

def overlay(img: Image.Image, label: str) -> str:
    colour = (255, 0, 0) if label == "PNEUMONIA" else (0, 255, 0)
    buf = io.BytesIO()
    Image.blend(img, Image.new("RGB", img.size, colour), 0.25).save(buf, "PNG")
    return base64.b64encode(buf.getvalue()).decode()

@app.get("/", response_class=HTMLResponse)
async def root():
    return (
        "<h1>Pneumonia Detection API – running</h1>"
        "<p>POST an X-ray to <code>/predict</code>.</p>"
    )

@app.get("/health")
async def health():
    return {"ok": True}

@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    x_api_key: Optional[str] = Header(None)
):
    ensure_auth(x_api_key)
    img = Image.open(io.BytesIO(await file.read())).convert("RGB")
    model = get_model()
    pred = float(model.predict(preprocess(img))[0][0])
    label = "PNEUMONIA" if pred > THRESHOLD else "NORMAL"
    conf  = round(pred if pred > THRESHOLD else 1 - pred, 4)

    return JSONResponse({
        "diagnosis": label,
        "confidence": conf,
        "raw_score": pred,
        "processed_image": overlay(img, label)
    })
