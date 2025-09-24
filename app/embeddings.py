import os
from typing import List
from fastapi import HTTPException
from pydantic import BaseModel
import spacy
import numpy as np

MODEL_NAME = os.getenv("SPACY_MODEL", "en_core_web_md")

try:
    nlp = spacy.load(MODEL_NAME)
except OSError as e:
    raise RuntimeError(
        f"Failed to load spaCy model '{MODEL_NAME}'. "
        f"Install it via: python -m spacy download {MODEL_NAME}"
    ) from e

if nlp.vocab.vectors_length == 0:
    raise RuntimeError(
        f"spaCy model '{MODEL_NAME}' has no word vectors. "
        "Use en_core_web_md or en_core_web_lg."
    )

DIM = nlp.vocab.vectors_length

class EmbedRequest(BaseModel):
    texts: List[str]

class EmbedResponse(BaseModel):
    model: str
    dim: int
    embeddings: List[List[float]]

class SimilarityRequest(BaseModel):
    a: str
    b: str

class SimilarityResponse(BaseModel):
    model: str
    similarity: float

def embed_texts(texts: List[str]) -> EmbedResponse:
    if not texts:
        raise HTTPException(status_code=400, detail="texts cannot be empty")
    docs = list(nlp.pipe(texts, disable=["tagger", "parser", "ner", "lemmatizer"]))
    vecs = [doc.vector.tolist() for doc in docs]
    return EmbedResponse(model=MODEL_NAME, dim=DIM, embeddings=vecs)

def cosine_similarity(a: str, b: str) -> SimilarityResponse:
    d1, d2 = nlp(a), nlp(b)
    v1, v2 = d1.vector, d2.vector
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    sim = float(np.dot(v1, v2) / (n1 * n2)) if n1 and n2 else 0.0
    return SimilarityResponse(model=MODEL_NAME, similarity=sim)