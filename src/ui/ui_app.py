import streamlit as st
import requests

# UI Title
st.title("RAG System (Multi-Level Retrieval)")

# Input box
query = st.text_input("Ask a question:")

# Button trigger
if st.button("Search"):
    if query.strip() == "":
        st.warning("Please enter a query.")
    else:
        try:
            res = requests.post(
                "http://127.0.0.1:8000/query",
                json={"query": query}
            )

            data = res.json()

            st.subheader("Answer")
            st.write(data["answer"])

            st.subheader("Sources")
            for src in data["sources"]:
                st.write(f"- {src}")

        except Exception as e:
            st.error("Failed to connect to API. Is FastAPI running?")