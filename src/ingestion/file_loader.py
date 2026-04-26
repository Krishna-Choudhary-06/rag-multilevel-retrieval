import os
import pandas as pd
from PyPDF2 import PdfReader


# =========================
# TXT LOADER
# =========================
def load_txt(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


# =========================
# PDF LOADER
# =========================
def load_pdf(path):
    reader = PdfReader(path)
    text = ""

    for page in reader.pages:
        text += page.extract_text() or ""

    return text


# =========================
# CSV LOADER
# =========================
def load_csv(path):
    df = pd.read_csv(path)
    return df.to_string()


# =========================
# MAIN LOADER
# =========================
def load_file(path):
    ext = os.path.splitext(path)[1].lower()

    if ext in [".txt", ".md", ".json", ".log"]:
        return load_txt(path)

    elif ext == ".pdf":
        return load_pdf(path)

    elif ext == ".csv":
        return load_csv(path)

    else:
        print(f"[SKIP] Unsupported file: {path}")
        return None
