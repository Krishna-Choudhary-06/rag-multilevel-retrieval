import json

METADATA_PATH = "embeddings/metadata_store/metadata.json"


class MetadataFilter:
    def __init__(self):
        with open(METADATA_PATH, "r", encoding="utf-8") as f:
            self.metadata = json.load(f)

    def filter(self, filters: dict):
        """
        filters example:
        {
            "doc_id": "rag_intro.txt"
        }
        """

        filtered = []

        for item in self.metadata:
            match = True

            for key, value in filters.items():
                if item["metadata"].get(key) != value:
                    match = False
                    break

            if match:
                filtered.append(item)

        return filtered
