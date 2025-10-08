from typing import Union
from fastapi import FastAPI
from pydantic import BaseModel
from app.bigram_model import BigramModel

from app.embeddings import (
    EmbedRequest, EmbedResponse, SimilarityRequest, SimilarityResponse,
    embed_texts, cosine_similarity, MODEL_NAME
)

app = FastAPI()

corpus = [
    "The Count of Monte Cristo is a novel written by Alexandre Dumas. \
It tells the story of Edmond Dantès, who is falsely imprisoned and later seeks revenge.",
    "this is another example sentence",
    "we are generating text based on bigram probabilities",
    "bigram models are simple but effective"
]

bigram_model = BigramModel(corpus)

class TextGenerationRequest(BaseModel):
    start_word: str
    length: int

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.post("/generate")
def generate_text(request: TextGenerationRequest):
    generated_text = bigram_model.generate_text(request.start_word, request.length)
    return {"generated_text": generated_text}

@app.post("/embed", response_model=EmbedResponse)
def embed(req: EmbedRequest):
    return embed_texts(req.texts)

@app.post("/similarity", response_model=SimilarityResponse)
def similarity(req: SimilarityRequest):
    return cosine_similarity(req.a, req.b)

import io
import torch
from PIL import Image
from pathlib import Path
import torchvision.transforms as T
from fastapi import UploadFile, File
from CNN_Assignment2.model import CNN64 

CIFAR10_CLASSES = [
    "airplane","automobile","bird","cat","deer",
    "dog","frog","horse","ship","truck"
]

device = torch.device(
    "mps" if torch.backends.mps.is_available()
    else ("cuda" if torch.cuda.is_available() else "cpu")
)

ROOT = Path(__file__).resolve().parents[1]          
CKPT = ROOT / "CNN_Assignment2" / "cnn64_cifar10.pt"

_model: CNN64 | None = CNN64().to(device)
try:
    _state = torch.load(CKPT, map_location=device)
    _model.load_state_dict(_state["state_dict"] if isinstance(_state, dict) and "state_dict" in _state else _state)
    _model.eval()
    print(f"Loaded model from {CKPT}")
except FileNotFoundError:
    print(f"Checkpoint not found at {CKPT}. /predict will be disabled until you train.")
    _model = None
except Exception as e:
    print(f"Failed to load checkpoint at {CKPT}: {e}")
    _model = None

_pre = T.Compose([T.Resize((64, 64)), T.ToTensor()])

@app.get("/health")
def health():
    return {"status": "ok", "device": str(device), "model_loaded": _model is not None}

@app.get("/classes")
def classes():
    return CIFAR10_CLASSES

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if _model is None:
        return {"error": "model not loaded; train first or fix CKPT path"}
    byts = await file.read()
    img = Image.open(io.BytesIO(byts)).convert("RGB")
    x = _pre(img).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = torch.softmax(_model(x), dim=1).cpu().squeeze(0)
    top = torch.topk(probs, k=3)
    return {
        "top3": [
            {"class": CIFAR10_CLASSES[i], "prob": float(p)}
            for p, i in zip(top.values.tolist(), top.indices.tolist())
        ]
    }