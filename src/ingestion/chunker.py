import uuid
import hashlib


# =========================
# GENERATE STABLE CHUNK ID
# =========================
def generate_chunk_id(text: str) -> str:
    """
    Generates a deterministic ID for a chunk based on its content.
    Ensures duplicate chunks are not re-embedded.
    """
    return hashlib.md5(text.strip().encode("utf-8")).hexdigest()


# =========================
# CHUNKING FUNCTION
def chunk_text(text, chunk_size=500, overlap=50):
    """
    Splits text into overlapping chunks

    Args:
        text (str): full document text
        chunk_size (int): size of each chunk
        overlap (int): overlap between chunks

    Returns:
        List[str]
    """

    chunks = []

    if not text or len(text.strip()) == 0:
        return chunks

    start = 0
    text_length = len(text)

    while start < text_length:
        end = start + chunk_size
        chunk = text[start:end]

        # clean chunk
        chunk = chunk.strip()
        if chunk:
            chunks.append(chunk)

        start += chunk_size - overlap

    print(f"[CHUNKER] Total chunks created: {len(chunks)}")

    return chunks
