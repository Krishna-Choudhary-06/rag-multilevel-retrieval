import json
import os

CACHE_PATH = "cache/query_cache.json"


class QueryCache:
    def __init__(self):
        os.makedirs("cache", exist_ok=True)

        if os.path.exists(CACHE_PATH):
            with open(CACHE_PATH, "r") as f:
                self.cache = json.load(f)
        else:
            self.cache = {}

    def get(self, query):
        return self.cache.get(query)

    def set(self, query, result):
        self.cache[query] = result
        with open(CACHE_PATH, "w") as f:
            json.dump(self.cache, f, indent=2)
