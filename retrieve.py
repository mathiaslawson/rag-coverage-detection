"""Retrieval over the passage corpus, and a recall@k baseline.

Reads  data/passages.json, data/title_to_idx.json, data/emb.npy, data/questions
Writes data/retrieval_baseline.json  per-question top-k scores and hit status
"""

import json
import os
import numpy as np
from datasets import load_from_disk
from sentence_transformers import SentenceTransformer

DATA_DIR = "data"
MODEL_NAME = "all-MiniLM-L6-v2"
K = 5


def pick_device():
    import torch
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def retrieve(query, model, emb, k=K):
    """Return (indices, scores) of the k nearest passages."""
    qv = model.encode([query], normalize_embeddings=True)[0]
    scores = emb @ qv
    top = np.argsort(-scores)[:k]
    return top, scores[top]


def main():
    with open(os.path.join(DATA_DIR, "passages.json")) as f:
        passages = json.load(f)
    with open(os.path.join(DATA_DIR, "title_to_idx.json")) as f:
        title_to_idx = json.load(f)

    emb = np.load(os.path.join(DATA_DIR, "emb.npy"))
    questions = load_from_disk(os.path.join(DATA_DIR, "questions"))

    model = SentenceTransformer(MODEL_NAME, device=pick_device())

    records = []
    for q in questions:
        idxs, scores = retrieve(q["question"], model, emb)

        gold_titles = set(q["supporting_facts"]["title"])
        gold_idxs = {title_to_idx[t] for t in gold_titles if t in title_to_idx}

        retrieved = set(int(i) for i in idxs)
        n_gold_found = len(retrieved & gold_idxs)

        records.append({
            "question": q["question"],
            "answer": q["answer"],
            "top1_score": float(scores[0]),
            "top_idxs": [int(i) for i in idxs],
            "n_gold": len(gold_idxs),
            "n_gold_found": n_gold_found,
            "all_gold_found": n_gold_found == len(gold_idxs),
        })

    out = os.path.join(DATA_DIR, "retrieval_baseline.json")
    with open(out, "w") as f:
        json.dump(records, f)

    any_hit = np.mean([r["n_gold_found"] > 0 for r in records])
    all_hit = np.mean([r["all_gold_found"] for r in records])
    scores = np.array([r["top1_score"] for r in records])

    print(f"questions:                  {len(records)}")
    print(f"recall@{K} (any gold):       {any_hit:.3f}")
    print(f"recall@{K} (all gold):       {all_hit:.3f}")
    print(f"top-1 similarity mean:      {scores.mean():.3f}")
    print(f"top-1 similarity min/max:   {scores.min():.3f} / {scores.max():.3f}")


if __name__ == "__main__":
    main()