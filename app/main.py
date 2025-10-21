from typing import Union
from fastapi import FastAPI
from pydantic import BaseModel
from app.bigram_model import BigramModel

#Assignment 1
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

@app.post("/bigram/generate")
def generate_text(request: TextGenerationRequest):
    generated_text = bigram_model.generate_text(request.start_word, request.length)
    return {"generated_text": generated_text}

@app.post("/embed", response_model=EmbedResponse)
def embed(req: EmbedRequest):
    return embed_texts(req.texts)

@app.post("/similarity", response_model=SimilarityResponse)
def similarity(req: SimilarityRequest):
    return cosine_similarity(req.a, req.b)

#Assignment 2 CNN 
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

#Assignment 3 GAN
import io
import base64
import torch
from fastapi import Query
from fastapi.responses import JSONResponse
from torchvision.utils import make_grid
from GAN_Assignment3.model import Generator

CKPT_GAN = ROOT / "GAN_Assignment3" / "checkpoints" / "G_epoch020.pt"
G = Generator(z_dim=100).to(device)
try:
    state = torch.load(CKPT_GAN, map_location=device)

    G.load_state_dict(state if isinstance(state, dict) and next(iter(state)).startswith("deconv") else (
        state["state_dict"] if isinstance(state, dict) and "state_dict" in state else state
    ))
    G.eval()
    print(f"Loaded GAN checkpoint from {CKPT_GAN}")
except FileNotFoundError:
    print(f"GAN checkpoint not found at {CKPT_GAN}. /gan/generate will 404.")
    G = None
except Exception as e:
    print(f"Failed to load GAN checkpoint at {CKPT_GAN}: {e}")
    G = None

@app.get("/gan/generate")
def gan_generate(n: int = Query(16, ge=1, le=64)):
    if G is None:
        return JSONResponse({"error": "GAN generator not loaded; check checkpoint path"}, status_code=503)
    with torch.no_grad():
        z = torch.randn(n, 100, device=device)
        imgs = G(z).cpu() 
        grid = make_grid(imgs, nrow=max(1, int(n**0.5)), normalize=True, value_range=(-1, 1))

        import numpy as np
        from PIL import Image
        nd = (grid.permute(1, 2, 0).numpy() * 255).clip(0, 255).astype("uint8")
        if nd.shape[-1] == 1:
            nd = nd.squeeze(-1)
        img = Image.fromarray(nd)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        return JSONResponse({"image_base64_png": b64})