#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Streamlit RAG Chat (Qdrant + OpenAI embeddings + Claude).
- Searches Qdrant with optional brand filter.
- Shows evidence with brand/qa_id/resolved_at/ticket.
- Answers via Anthropic Claude (bilingual JA/EN).

Env (Streamlit Secrets suggested):
  OPENAI_API_KEY
  ANTHROPIC_API_KEY
  QDRANT_URL
  QDRANT_API_KEY
  QDRANT_COLLECTION (optional, default: support_logs_all)
  CSV_PATH (optional for listing brands locally)
"""

import os
from pathlib import Path
import pandas as pd
import streamlit as st

from qdrant_client import QdrantClient
from langchain_openai import OpenAIEmbeddings
from langchain_anthropic import ChatAnthropic
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import Qdrant as LcQdrant

# ---------------- Prompt ----------------
PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template=(
        "You are a bilingual (JA/EN) support engineer. Answer ONLY using the context below.\n"
        "If you don't know, say so and request escalation.\n\n"
        "Rules:\n"
        "- Include the source **brand name**.\n"
        "- Use numbered steps when appropriate.\n"
        "- Add notes for risks/version differences.\n"
        "- End with citations listing (brand, qa_id, resolved_at, ticket_number).\n\n"
        "—— Context ——\n{context}\n—— End Context ——\n\n"
        "Question: {question}\n"
        "Answer:"
    )
)

def get_qdrant_client() -> QdrantClient:
    url = os.getenv("QDRANT_URL")
    key = os.getenv("QDRANT_API_KEY")
    if not url or not key:
        raise RuntimeError("Please set QDRANT_URL and QDRANT_API_KEY.")
    return QdrantClient(url=url, api_key=key)

def load_retriever(collection: str, k: int, brand_filter: str | None):
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    client = get_qdrant_client()

    vs = LcQdrant(
        client=client,
        collection_name=collection,
        embeddings=embeddings,
    )

    # Qdrant filter syntax via LangChain:
    if brand_filter and brand_filter != "All":
        filt = {"must": [{"key": "brand", "match": {"value": brand_filter}}]}
        return vs.as_retriever(search_kwargs={"k": k, "filter": filt})
    return vs.as_retriever(search_kwargs={"k": k})

def main():
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("Please set OPENAI_API_KEY.")
    if not os.getenv("ANTHROPIC_API_KEY"):
        raise RuntimeError("Please set ANTHROPIC_API_KEY.")

    st.set_page_config(page_title="All-brands RAG Chat (Qdrant)", layout="wide")
    st.title("All-brands RAG Chat (Qdrant + Claude)")

    qdrant_collection = os.getenv("QDRANT_COLLECTION", "support_logs_all")

    # Sidebar controls
    topk = st.sidebar.slider("Top-K documents", min_value=2, max_value=12, value=5, step=1)
    temperature = st.sidebar.slider("Temperature", 0.0, 1.0, 0.0, 0.1)
    # Use models you verified as available for your Anthropic account
    model = st.sidebar.selectbox(
        "Claude model",
        ["claude-3-7-sonnet-20250219", "claude-3-5-haiku-20241022"],
        index=0,
    )

    # Brand list from CSV (local, for UI only)
    brands = ["All"]
    csv_path = Path(os.getenv("CSV_PATH", "./all_brands_support_log_embedding_ready.csv"))
    if csv_path.exists():
        try:
            df = pd.read_csv(csv_path, usecols=["brand"])
            bset = sorted([b for b in df["brand"].dropna().unique() if str(b).strip()])
            brands += bset
        except Exception:
            pass

    brand = st.sidebar.selectbox("Brand filter", options=brands, index=0)

    # Buttons to check/initialize data
    col_a, col_b = st.sidebar.columns(2)
    with col_a:
        if st.button("Count Points"):
            try:
                client = get_qdrant_client()
                coll_info = client.get_collection(qdrant_collection)
                st.success(f"Collection '{qdrant_collection}' is present.")
            except Exception as e:
                st.error(f"Qdrant error: {e}")

    with col_b:
        if st.button("Build Index"):
            # Run index_all.py inside the same process (simple trigger)
            import subprocess, sys
            st.info("Building index…this may take a few minutes.")
            env = os.environ.copy()
            proc = subprocess.run([sys.executable, "index_all.py"], capture_output=True, text=True)
            if proc.returncode == 0:
                st.success("Index build finished.")
            else:
                st.error("Index build failed.")
                st.code(proc.stdout)
                st.code(proc.stderr)

    # Build retriever + chain
    retriever = load_retriever(qdrant_collection, k=topk, brand_filter=brand)
    llm = ChatAnthropic(model=model, temperature=temperature)

    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": PROMPT},
    )

    query = st.text_input("Ask a question")
    if st.button("Ask") and query.strip():
        with st.spinner("Searching and composing…"):
            # show candidates first
            candidates = retriever.get_relevant_documents(query)
            st.subheader("Evidence")
            for i, d in enumerate(candidates, 1):
                m = d.metadata or {}
                st.markdown(
                    f"- **{i}.** brand={m.get('brand','')}, "
                    f"qa_id={m.get('qa_id','')}, "
                    f"resolved_at={m.get('resolved_at','')}, "
                    f"ticket={m.get('ticket_number','')}"
                )
            answer = chain.run(query)

        st.subheader("Answer")
        st.write(answer)

    st.markdown("---")
    st.caption("Tip: choose 'All' for cross-brand search, or pick a single brand to restrict retrieval.")

if __name__ == "__main__":
    main()
