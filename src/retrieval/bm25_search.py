from rank_bm25 import BM25Okapi
import json
import os
import re

CHUNKS_PATH = "data/processed/chunks.json"


# =========================
# CLEAN TOKENIZER (IMPORTANT)
# =========================
def tokenize(text):
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)  # remove punctuation
    return text.split()


class BM25Search:
    def __init__(self):
        print("[BM25] Initializing...")

        if not os.path.exists(CHUNKS_PATH):
            print("[BM25] No chunks found")
            self.docs = []
            self.bm25 = None
            return

        with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
            chunks = json.load(f)

        self.docs = chunks

        # 🔥 Better tokenization
        tokenized_corpus = [tokenize(c["text"]) for c in chunks]

        self.bm25 = BM25Okapi(tokenized_corpus)

        print(f"[BM25] Loaded documents: {len(self.docs)}")

    def search(self, query, top_k=5):
        with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
            chunks = json.load(f)
        if not self.bm25:
            return []

        # 🔥 Consistent tokenization
        tokenized_query = tokenize(query)

        scores = self.bm25.get_scores(tokenized_query)

        ranked = sorted(list(enumerate(scores)), key=lambda x: x[1], reverse=True)

        results = []
        for idx, score in ranked[:top_k]:
            results.append(
                {
                    "score": float(score),
                    "text": self.docs[idx]["text"],
                    "metadata": self.docs[idx]["metadata"],
                }
            )

        return results
