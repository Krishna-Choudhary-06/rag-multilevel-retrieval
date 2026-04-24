import streamlit as st
import requests

API_URL = "http://127.0.0.1:8000/query"

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="RAG System for docs",
    layout="wide"
)

# =========================
# HEADER
# =========================
st.title("🧠 Clever RAG System")
st.caption("Metadata → Semantic → Deep Research")

# =========================
# SIDEBAR (CONTROL PANEL)
# =========================
with st.sidebar:
    st.header("📂 Document Upload")
    uploaded_files = st.file_uploader(
        "Upload new files", 
        type=["pdf", "txt", "md"], 
        accept_multiple_files=True
    )
    
    if st.button("Upload & Index", use_container_width=True):
        if uploaded_files:
            with st.spinner("Uploading and indexing files..."):
                files_payload = [
                    ("files", (file.name, file.getvalue(), file.type))
                    for file in uploaded_files
                ]
                try:
                    upload_url = "http://127.0.0.1:8000/upload-multiple"
                    res = requests.post(upload_url, files=files_payload)
                    if res.status_code == 200:
                        st.success(f"Success! Indexed {res.json().get('chunks_created', 0)} chunks.")
                    else:
                        st.error(f"Upload failed: {res.text}")
                except Exception as e:
                    st.error(f"Error: {e}")
        else:
            st.warning("Select files first")

    st.markdown("---")

    st.header("⚙️ Controls")

    # MODE SELECTOR
    mode = st.selectbox(
        "Search Mode",
        ["Auto", "Metadata Only", "Semantic Only", "Deep"]
    )

    # FILE FILTER
    file_filter = st.selectbox(
        "Filter by File",
        ["All", "DBMS Notes.pdf", "SCSE-AIML-HACKATHON-PS.pdf"]
    )

    # USER ID (multi-user support)
    user_id = st.text_input("User ID", value="user1")

    st.markdown("---")

    st.caption("💡 Tips")
    st.caption("- Use Metadata mode for fast filtering")
    st.caption("- Use Deep mode for detailed answers")

# =========================
# MAIN INPUT
# =========================
query = st.text_area(
    "Ask your question",
    placeholder="e.g. explain normalization in DBMS...",
    height=120
)

# =========================
# BUILD FILTERS
# =========================
filters = {}

if file_filter != "All":
    filters["doc_id"] = file_filter

# =========================
# SEARCH BUTTON
# =========================
if st.button("🚀 Search", use_container_width=True):

    if not query.strip():
        st.warning("Please enter a query")
        st.stop()

    payload = {
        "query": query,
        "user_id": user_id,
        "filters": filters,
        "mode": mode
    }

    # =========================
    # API CALL
    # =========================
    with st.spinner("Searching..."):
        try:
            response = requests.post(API_URL, json=payload)

            if response.status_code != 200:
                st.error(f"API Error: {response.text}")
                st.stop()

            data = response.json()

        except Exception as e:
            st.error(f"Connection failed: {e}")
            st.stop()

    # =========================
    # OUTPUT UI
    # =========================
    st.caption(f"**Current Mode:** {data.get('mode', mode)}")

    # ANSWER
    st.subheader("💡 Answer")
    st.write(data.get("answer", "No answer"))

    st.markdown("---")

    tab_src, tab_meta, tab_sem, tab_deep, tab_dbg = st.tabs([
        "📚 Sources", "🗂️ Metadata", "🔍 Semantic", "🧠 Deep", "🐛 Debug"
    ])

    chunks_to_use = data.get("reranked", [])
    if not chunks_to_use:
        chunks_to_use = data.get("retrieved_chunks", [])

    # Group sources by document
    grouped_sources = {}
    for i, c in enumerate(chunks_to_use):
        meta = c.get("metadata", {})
        doc_id = meta.get("doc_id") or meta.get("source") or f"untracked_doc_{i}"
        
        if doc_id not in grouped_sources:
            grouped_sources[doc_id] = {
                "max_score": c.get("score", 0),
                "chunks": []
            }
        
        # Deduplication check
        is_dup = False
        for ex_c in grouped_sources[doc_id]["chunks"]:
            if ex_c["text"][:100] == c.get("text", "")[:100]:
                is_dup = True
                break
        
        if not is_dup:
            grouped_sources[doc_id]["chunks"].append(c)
            if c.get("score", 0) > grouped_sources[doc_id]["max_score"]:
                grouped_sources[doc_id]["max_score"] = c.get("score", 0)

    # Sort grouped sources by highest relevance score
    sorted_sources = sorted(grouped_sources.items(), key=lambda x: x[1]["max_score"], reverse=True)

    with tab_src:
        if sorted_sources:
            # Highlight tracking
            all_chunks = []
            for d in chunks_to_use:
                if d not in all_chunks:
                    all_chunks.append(d)
            top_3_texts = [c.get("text", "") for c in sorted(all_chunks, key=lambda x: x.get("score", 0), reverse=True)[:3]]

            for doc_id, info in sorted_sources:
                with st.expander(f"📄 {doc_id} (Relevance: {info['max_score']:.4f})", expanded=True):
                    for c in info["chunks"]:
                        text = c.get("text", "")
                        star = "⭐ " if text in top_3_texts else ""
                        st.markdown(f"**{star}Chunk Score: {c.get('score', 0):.4f}**")
                        st.write(text)
                        st.markdown("---")
        else:
            st.write("No sources used.")

    with tab_meta:
        st.write("### Metadata Filter Layer")
        st.write(f"**Mode Selected:** {mode}")
        st.write(f"**Applied Filters:** {filters}")
        if mode == "Metadata Only":
            for c in data.get("retrieved_chunks", []):
                doc_id = c.get('metadata', {}).get('doc_id') or "unknown"
                st.write(f"- `{doc_id}`: {c.get('text', '')[:150]}...")

    with tab_sem:
        st.write("### Base Semantic Retrieval (FAISS)")
        for c in data.get("retrieved_chunks", []):
            st.write(f"**Score {c.get('score', 0):.4f}** - {c.get('text', '')[:200]}...")
            st.markdown("---")

    with tab_deep:
        st.write("### Reranked Deep Context")
        reranked = data.get("reranked", [])
        if reranked:
            for c in reranked:
                st.write(f"**Score {c.get('score', 0):.4f}** - {c.get('text', '')[:200]}...")
                st.markdown("---")
        else:
            st.write("No reranking applied in this mode.")

    with tab_dbg:
        st.write("### Raw Pipeline Response")
        st.json(data)

    # LATENCY
    st.caption(f"⏱ Latency: {data.get('latency', 0)} sec")