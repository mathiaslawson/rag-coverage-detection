# Coverage Detection in Retrieval-Augmented Generation

Does a retrieval-augmented system know when a question falls outside its knowledge base?

A vector retriever always returns its nearest matches. It has no representation of "nothing here" — only of "this was closest." A generator downstream writes fluently whether or not the passages it received support an answer. This project asks which signals, if any, reliably detect that a question has left the system's coverage.

## The observation that motivated this

Two queries run against a corpus that did not contain the relevant passages:

| Query | Top-1 cosine | Returned |
|---|---|---|
| "Who is the mother of the director of film Polish-Russian War?" | 0.62 | A Polish film director — adjacent, and wrong |
| "best recipe for jollof rice" | 0.32 | People surnamed Rice |

Both questions are out of coverage. A 0.5 threshold accepts the first and rejects the second. The second scores as high as it does partly because subword tokenisation shares the token "rice" — the score is tracking surface proximity in embedding space, not whether supporting evidence exists.

The quantity being measured is not the quantity being claimed.

## Method

**Construct known out-of-coverage conditions.** 2WikiMultiHopQA ships with Wikidata entity-relation paths, so the graph and the ground truth come for free. Remove a fraction of entities or relation types from the graph, and remove the matching passages from the retrieval index. This produces questions whose out-of-coverage status is known by construction rather than annotated. Repeat at several removal rates.

**Compare three abstention signals.**

1. **Similarity** — top-k cosine score from the vector retriever. The baseline this project argues against.
2. **Model confidence** — token likelihood, or agreement across sampled answers.
3. **Structure** — after linking the question's entities, does a path exist between them in the remaining graph, and how long is it?

**Evaluate honestly.** Risk–coverage curves (error on answered questions against the fraction answered), summarised by area under the curve, plus AUROC for separating covered from uncovered questions. A metric that rewards abstaining on everything is not a metric.

The question is whether explicit graph structure detects out-of-coverage queries more reliably than similarity or confidence, neither of which distinguishes "no supporting evidence" from "weak supporting evidence."

## Status

In progress. Started September 2026.

**Done**
- Corpus built from the 2WikiMultiHopQA validation split: 1,000 sampled questions, 6,250 unique passages after deduplication by title
- Dense embeddings via `all-MiniLM-L6-v2`, 384 dimensions, normalised
- Retrieval demonstrated by hand; the observation above

**Next**
- Retrieval over all 1,000 in-coverage questions: recall@5 and top-1 similarity per question
- Generation baseline with a prompt that permits "unknown"
- The removal procedure at several rates
- Three-signal comparison on risk–coverage curves and AUROC

## Reproducing

```
dataset:   framolfese/2WikiMultihopQA, validation split
embedder:  sentence-transformers/all-MiniLM-L6-v2
generator: google/flan-t5-base
index:     faiss IndexFlatIP (exact; approximate search would add recall loss
           that could be mistaken for a coverage effect)
```

```bash
pip install -r requirements.txt
python build_corpus.py     # writes passages.json
python embed.py            # writes emb.npy
```

A note on the data shape: in this HuggingFace mirror, `context` is not a list of `[title, sentences]` pairs. It is a dict of two parallel lists, `{'title': [...], 'sentences': [[...], ...]}`, and `supporting_facts` has the same shape. Use `zip(q["context"]["title"], q["context"]["sentences"])`.

## Licence

MIT. Dataset licences are the respective datasets'.