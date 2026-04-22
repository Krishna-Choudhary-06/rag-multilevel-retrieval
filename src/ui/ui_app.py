import streamlit as st
import requests
import time

API_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="RAG System", layout="wide")

st.title("RAG System (Multi-Level Retrieval)")

# =========================
# QUERY SECTION
# =========================
user_id = st.text_input("User ID", value="user1")

query = st.text_input("Ask a question:")

if st.button("Search"):
    if query:
        start_time = time.time()

        try:
            res = requests.post(
                f"{API_URL}/query", json={"query": query, "user_id": user_id}
            )

            if res.status_code == 200:
                data = res.json()

                latency = round(time.time() - start_time, 2)
                st.write(f"⏱ Response Time: {latency} sec")

                # =========================
                # ANSWER
                # =========================
                st.subheader("Answer")
                st.write(data.get("answer", "No answer"))

                # =========================
                # SOURCES
                # =========================
                st.subheader("Sources")
                for s in data.get("sources", []):
                    st.write(f"- {s}")

                if "metadata" in data:
                    st.subheader("📁 Metadata")

                    for i, meta in enumerate(data["metadata"]):
                        st.write(f"Chunk {i+1}:")
                        st.json(meta)

                # =========================
                # RETRIEVED CHUNKS
                # =========================
                with st.expander("🔍 Retrieved Chunks"):
                    for r in data.get("reranked", []):
                        st.write(r.get("text", ""))

                # =========================
                # SCORES
                # =========================
                with st.expander("📊 Scores"):
                    st.write(data.get("scores", []))

                # =========================
                # RERANKED RESULTS
                # =========================
                with st.expander("⚖️ Reranked Results"):
                    for r in data.get("reranked", []):
                        st.write(r)

            else:
                st.error("Failed to connect to API")

        except Exception as e:
            st.error(f"Error: {str(e)}")

# =========================
# UPLOAD SECTION
# =========================
st.subheader("📁 Upload Documents")

uploaded_files = st.file_uploader(
    "Upload documents (TXT, PDF, CSV)", accept_multiple_files=True
)

if st.button("Upload"):
    if uploaded_files:
        files = [("files", (file.name, file, file.type)) for file in uploaded_files]

        try:
            res = requests.post(f"{API_URL}/upload-multiple", files=files)

            if res.status_code == 200:
                data = res.json()
                st.success("Upload successful!")
                st.write(data)

            else:
                st.error("Upload failed")

        except Exception as e:
            st.error(str(e))
