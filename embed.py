import json
import os
import numpy as np
from sentence_transformers import SentenceTransformer

DATA_DIR = "data"
MODEL_NAME = "all-MiniLM-L6-v2"


def pick_device():
    import torch
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


with open(os.path.join(DATA_DIR, "passages.json")) as f:
    passages = json.load(f)


texts = [p["title"] + ". " + p["text"] for p in passages]

device = pick_device()
print(f"device: {device}")

model = SentenceTransformer(MODEL_NAME, device=device)
emb = model.encode(
    texts,
    batch_size=128,
    normalize_embeddings=True,
    show_progress_bar=True,
)

np.save(os.path.join(DATA_DIR, "emb.npy"), emb)
print(f"saved {emb.shape} to {DATA_DIR}/emb.npy")