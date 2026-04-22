import os
from src.ingestion.ingest_uploaded import ingest_uploaded


def ingest_folder(folder_path):
    upload_dir = "data/uploaded"

    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith((".txt", ".pdf", ".csv")):
                src = os.path.join(root, file)
                dst = os.path.join(upload_dir, file)

                with open(src, "rb") as f_src:
                    with open(dst, "wb") as f_dst:
                        f_dst.write(f_src.read())

    return ingest_uploaded()
