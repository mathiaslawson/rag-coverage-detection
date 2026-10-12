import json
import os
from datasets import load_dataset

OUT_DIR = "data"
N_QUESTIONS = 1000
SEED = 0

os.makedirs(OUT_DIR, exist_ok=True)

ds = load_dataset("framolfese/2WikiMultihopQA", split="validation")
questions = ds.shuffle(seed=SEED).select(range(N_QUESTIONS))

seen = set()
passages = []
total_slots = 0

for q in questions:
    titles = q["context"]["title"]
    all_sentences = q["context"]["sentences"]
    for title, sentences in zip(titles, all_sentences):
        total_slots += 1
        if title in seen:
            continue
        seen.add(title)
        passages.append({"title": title, "text": " ".join(sentences)})

title_to_idx = {p["title"]: i for i, p in enumerate(passages)}

with open(os.path.join(OUT_DIR, "passages.json"), "w") as f:
    json.dump(passages, f)

with open(os.path.join(OUT_DIR, "title_to_idx.json"), "w") as f:
    json.dump(title_to_idx, f)

questions.save_to_disk(os.path.join(OUT_DIR, "questions"))

print(f"context slots seen:     {total_slots}")
print(f"unique passages kept:   {len(passages)}")
print(f"duplicates discarded:   {total_slots - len(passages)}")
print(f"example: {passages[0]['title']}")