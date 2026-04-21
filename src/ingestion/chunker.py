import uuid
from typing import List, Dict


class TextChunker:
    def __init__(self, chunk_size: int = 500, overlap: int = 80):
        self.chunk_size = chunk_size
        self.overlap = overlap

        if overlap >= chunk_size:
            raise ValueError("Overlap must be smaller than chunk size")

    def chunk_text(
        self,
        text: str,
        base_metadata: Dict
    ) -> List[Dict]:
        """
        Splits text into overlapping chunks and attaches metadata.
        """

        chunks = []
        start = 0
        text_length = len(text)

        while start < text_length:
            end = start + self.chunk_size
            chunk_text = text[start:end]

            if not chunk_text.strip():
                break

            chunk = {
                "chunk_id": str(uuid.uuid4()),
                "text": chunk_text,
                "metadata": base_metadata.copy()
            }

            chunks.append(chunk)

            start += self.chunk_size - self.overlap

        return chunks


def chunk_documents(
    documents: List[Dict],
    chunk_size: int,
    overlap: int
) -> List[Dict]:
    """
    Processes multiple documents into chunks.

    Each document must have:
    {
        "text": "...",
        "metadata": {...}
    }
    """

    chunker = TextChunker(chunk_size, overlap)
    all_chunks = []

    for doc in documents:
        text = doc["text"]
        metadata = doc.get("metadata", {})

        chunks = chunker.chunk_text(text, metadata)
        all_chunks.extend(chunks)

    return all_chunks