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

    def search(self, query, top_k=5, filters=None):
        if not self.bm25:
            return []

        tokenized_query = query.split()
        scores = self.bm25.get_scores(tokenized_query)

        ranked = sorted(
            list(enumerate(scores)),
            key=lambda x: x[1],
            reverse=True
        )

        results = []

        for idx, score in ranked:
            doc = self.docs[idx]

            # -------------------------
            # APPLY METADATA FILTER
            # -------------------------
            if filters:
                skip = False
                for k, v in filters.items():
                    if doc["metadata"].get(k) != v:
                        skip = True
                        break
                if skip:
                    continue

            results.append({
                "text": doc["text"],
                "metadata": doc["metadata"],
                "score": float(score)
            })

            if len(results) >= top_k:
                break

        return results
