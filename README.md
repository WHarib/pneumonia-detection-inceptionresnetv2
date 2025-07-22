
---
title: Pneumonia Detection API (InceptionResNetV2)
emoji: 🫁
colorFrom: blue
colorTo: gray
sdk: docker
sdk_version: "1.0"
app_file: app.py
pinned: false
---

# 🫁 Pneumonia Detection API (InceptionResNetV2)

A Hugging Face Space that serves a chest X-ray classifier built on **InceptionResNetV2** via FastAPI.  
The model distinguishes **PNEUMONIA** from **NORMAL** radiographs and was trained on the _Chest X-ray Pneumonia_ dataset (Kaggle, Mooney 2017).

---

## 🧠 Background

This service reproduces the pneumonia pipeline described in the study:

> _Evaluating Explainable Artificial Intelligence (XAI) Techniques in Chest Radiology Imaging Through a Human-Centred Lens_  
> Izegbua E. Ihongbe **et al.** – *PLOS ONE* 19 (10), 2024, e0308758  
> DOI: 10.1371/journal.pone.0308758

Key training details  
| Setting | Value |
|---------|-------|
| Backbone | InceptionResNetV2 (conv base frozen) |
| Input size | 200 × 200 × 3 |
| Head | GAP → Dense 512 → Dense 128 → Softmax 2 |
| Optimiser / LR | Adam 1 × 10⁻³ |
| Epochs | 20 |
| Dataset split | Train 5216 • Val 16 • Test 624 |

Reported hold-out accuracy: **~90 %**

---

## 🚀 Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET`  | `/`          | Landing page |
| `GET`  | `/health`    | Liveness probe (`{"ok": true}`) |
| `POST` | `/predict`   | Classify a single X-ray |

### `/predict` request

- **Content-Type**: `multipart/form-data`
- **Field**: `file` – PNG/JPEG image

### Response JSON

```json
{
  "diagnosis": "PNEUMONIA",
  "confidence": 0.93,
  "raw_score": 0.9316,
  "processed_image": "<base64-encoded PNG with overlay>"
}
📝 Example usage
python
Copy
Edit
import requests
with open("patient_xray.png", "rb") as f:
    r = requests.post(
        "https://<username>-pneumonia-inceptionresnetv2.hf.space/predict",
        files={"file": f},
        headers={"x-api-key": "YOUR_API_KEY"}  # omit if not enabled
    )
print(r.json())
Or via curl:

bash
Copy
Edit
curl -F "file=@patient_xray.png" \
     https://<username>-pneumonia-inceptionresnetv2.hf.space/predict
⚙️ Dependencies
TensorFlow 2.17 CPU

FastAPI

Pillow

NumPy

HuggingFace Hub

All pinned in requirements.txt.

🔒 Security & rate-limiting
If you set the EXPECT_KEY environment variable, the API expects an
x-api-key header on every request and returns 401 Unauthorised otherwise.

📜 Licence
Model weights and code released under the Apache 2.0 licence.
Dataset licenced by the original Kaggle authors.

🙏 Acknowledgements
Original dataset: Chest X-ray Pneumonia (Paul Timothy Mooney, Kaggle).

TensorFlow and Keras teams for the deep-learning framework.

Hugging Face Spaces for the hosting infrastructure.