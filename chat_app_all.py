#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Streamlit RAG Chatbot (FAISS version)
- OpenAI embeddings (for indexing/search)
- FAISS vector store (no Chroma, no pydantic v1)
- Anthropic Claude for answers
- Auto-builds FAISS index in /tmp if missing
"""

import os
from pathlib import Path
import sys
import subprocess
import pandas as pd
import streamlit as st

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_anthropic import ChatAnthropic
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# Paths (Cloud-safe defaults)
INDEX_DIR = Path(os.getenv("INDEX_DIR", "/tmp/index_all_faiss"))
CSV_PATH  = Path(os.getenv("CSV_PATH", "./all_brands_support_log_embedding_ready.csv"))

PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template=(
        "You are a bilingual (Japanese/English) support engineer. "
        "Answer ONLY using the information in the context below.\n"
        "If the answer cannot be found, say so and request escalation.\n\n"
        "Rules:\n"
        "- Include the source **brand name** in your answer.\n"
        "- Use numbered steps when appropriate.\n"
        "- Add notes for risks/version differences.\n"
        "- End with citations listing (brand, qa_id, resolved_at, ticket_number).\n\n"
        "—— Context ——\n{context}\n—— End Context ——\n\n"
        "Question: {question}\n"
        "Answer:"
    ),
)

@st.cache_resource(show_spinner=False)
def load_faiss() -> FAISS:
    """Load FAISS index from disk (cached across reruns)."""
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    # allow_dangerous_deserialization=True is required by langchain FAISS loader
    return FAISS.load_local(
        str(INDEX_DIR),
        embeddings,
        allow_dangerous_deserialization=True
    )

def build_index_if_missing():
    """Build FAISS index in /tmp if it doesn't exist."""
    if not INDEX_DIR.exists():
        if not CSV_PATH.exists():
            st.error(f"CSV not found at {CSV_PATH}. Upload/commit it.")
            st.stop()
        st.warning("Index not found. Building it now — this may take a few minutes...")
        with st.spinner("Creating embeddings & building FAISS index..."):
            env = os.environ.copy()
            env["INDEX_DIR"] = str(INDEX_DIR)
            env["CSV_PATH"] = str(CSV_PATH)
            proc = subprocess.run(
                [sys.executable, "index_all.py", "--csv", str(CSV_PATH), "--out", str(INDEX_DIR)],
                capture_output=True, text=True, env=env
            )
        if proc.returncode != 0:
            st.error("Index build failed.\n\nSTDERR:\n" + proc.stderr + "\n\nSTDOUT:\n" + proc.stdout)
            st.stop()

def main():
    st.set_page_config(page_title="All-brands RAG Chat (FAISS)", layout="wide")
    st.title("All-brands RAG Chat (Claude + FAISS)")

    # Secrets check
    if not os.getenv("OPENAI_API_KEY"):
        st.error("❌ Missing OPENAI_API_KEY (for embeddings). Add it to Streamlit secrets.")
        st.stop()
    if not os.getenv("ANTHROPIC_API_KEY"):
        st.error("❌ Missing ANTHROPIC_API_KEY (for answers). Add it to Streamlit secrets.")
        st.stop()

    # Build index if needed
    build_index_if_missing()

    # Brand list for filter (optional)
    brands = ["All"]
    try:
        df = pd.read_csv(CSV_PATH, usecols=["brand"])
        brands += sorted([b for b in df["brand"].dropna().unique() if str(b).strip()])
    except Exception:
        pass

    # Sidebar controls
    brand = st.sidebar.selectbox("Brand filter", options=brands, index=0)
    topk = st.sidebar.slider("Top-K documents", 2, 12, 5)
    temperature = st.sidebar.slider("Temperature", 0.0, 1.0, 0.0, 0.1)
    model = st.sidebar.selectbox(
        "Claude model",
        ["claude-3-7-sonnet-20250219", "claude-3-5-haiku-20241022", "claude-3-haiku-20240307"],
        index=0,
    )

    # Load FAISS retriever (cached)
    faiss_store = load_faiss()

    # NOTE: FAISS doesn't support metadata filters internally.
    # We filter results post-retrieval when brand != "All".
    def retrieve(query: str):
        docs = faiss_store.similarity_search(query, k=max(12, topk))  # overfetch
        if brand != "All":
            docs = [d for d in docs if (d.metadata or {}).get("brand") == brand]
        return docs[:topk]

    # UI
    query = st.text_input("Ask a question")
    if st.button("Ask") and query.strip():
        with st.spinner("Searching and composing…"):
            candidates = retrieve(query)

            st.subheader("Evidence")
            for i, d in enumerate(candidates, 1):
                m = d.metadata or {}
                st.markdown(
                    f"- **{i}.** brand={m.get('brand','')}, qa_id={m.get('qa_id','')}, "
                    f"resolved_at={m.get('resolved_at','')}, ticket={m.get('ticket_number','')}"
                )

            context = "\n\n---\n\n".join(
                [
                    f"Brand: { (d.metadata or {}).get('brand','') }\n"
                    f"qa_id: { (d.metadata or {}).get('qa_id','') }\n"
                    f"resolved_at: { (d.metadata or {}).get('resolved_at','') }\n"
                    f"ticket_number: { (d.metadata or {}).get('ticket_number','') }\n"
                    f"CONTENT:\n{d.page_content}"
                    for d in candidates
                ]
            )

            llm = ChatAnthropic(model=model, temperature=temperature)
            chain = LLMChain(llm=llm, prompt=PROMPT)
            answer = chain.run(context=context, question=query)

        st.subheader("Answer")
        st.write(answer)

    st.markdown("---")
    with st.sidebar.expander("Debug"):
        st.write("INDEX_DIR", str(INDEX_DIR), "exists:", INDEX_DIR.exists())
        st.write("CSV_PATH", str(CSV_PATH), "exists:", CSV_PATH.exists())
        st.write("OPENAI_API_KEY set:", bool(os.getenv("OPENAI_API_KEY")))
        st.write("ANTHROPIC_API_KEY set:", bool(os.getenv("ANTHROPIC_API_KEY")))

if __name__ == "__main__":
    main()
