#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Streamlit RAG Chatbot for all brands.
Uses Chroma vectorstore + OpenAI embeddings + Anthropic Claude for answers.
Cloud-safe version: stores index under /tmp and uses duckdb+parquet backend.
"""

import os
import sys
import subprocess
from pathlib import Path
import pandas as pd
import streamlit as st

from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_anthropic import ChatAnthropic
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from chromadb.config import Settings

# -------------------------------------------------------------
# Environment defaults (safe for Streamlit Cloud)
# -------------------------------------------------------------
os.environ.setdefault("CHROMA_DB_IMPL", "duckdb+parquet")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

INDEX_DIR = Path(os.getenv("INDEX_DIR", "/tmp/index_all"))
CSV_PATH = Path(os.getenv("CSV_PATH", "./all_brands_support_log_embedding_ready.csv"))

# -------------------------------------------------------------
# Prompt Template
# -------------------------------------------------------------
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

# -------------------------------------------------------------
# Retriever
# -------------------------------------------------------------
def load_retriever(index_dir: Path, k: int = 4, brand_filter: str | None = None):
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    client_settings = Settings(
        is_persistent=True,
        persist_directory=str(index_dir),
        anonymized_telemetry=False,
    )

    vs = Chroma(
        persist_directory=str(index_dir),
        embedding_function=embeddings,
        client_settings=client_settings,
    )

    if brand_filter and brand_filter != "All":
        return vs.as_retriever(search_kwargs={"k": k, "filter": {"brand": brand_filter}})
    return vs.as_retriever(search_kwargs={"k": k})

# -------------------------------------------------------------
# Main App
# -------------------------------------------------------------
def main():
    st.set_page_config(page_title="All-brands RAG Chat", layout="wide")
    st.title("All-brands RAG Chat (Claude + Chroma)")

    index_dir = INDEX_DIR

    # Auto-build if missing
    if not index_dir.exists():
        if not CSV_PATH.exists():
            st.error(f"CSV not found at {CSV_PATH}. Upload or commit it.")
            st.stop()

        st.warning("Index not found. Building it now — please wait a few minutes...")
        with st.spinner("Creating embeddings & building Chroma index..."):
            build_env = {**os.environ, "INDEX_DIR": str(index_dir), "CSV_PATH": str(CSV_PATH)}
            proc = subprocess.run(
                [sys.executable, "index_all.py", "--csv", str(CSV_PATH), "--out", str(index_dir)],
                capture_output=True,
                text=True,
                env=build_env,
            )
        if proc.returncode != 0:
            st.error("Index build failed.\n\nSTDERR:\n" + proc.stderr + "\n\nSTDOUT:\n" + proc.stdout)
            st.stop()

    # Brand list
    brands = ["All"]
    if CSV_PATH.exists():
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

    retriever = load_retriever(index_dir, k=topk, brand_filter=brand)
    llm = ChatAnthropic(model=model, temperature=temperature)

    query = st.text_input("Ask a question")

    if st.button("Ask") and query.strip():
        with st.spinner("Searching and composing…"):
            candidates = retriever.get_relevant_documents(query)

            st.subheader("Evidence")
            for i, d in enumerate(candidates, 1):
                m = d.metadata
                st.markdown(
                    f"- **{i}.** brand={m.get('brand','')}, qa_id={m.get('qa_id','')}, "
                    f"resolved_at={m.get('resolved_at','')}, ticket={m.get('ticket_number','')}"
                )

            context = "\n\n---\n\n".join(
                [
                    f"Brand: {d.metadata.get('brand','')}\nqa_id: {d.metadata.get('qa_id','')}\n"
                    f"resolved_at: {d.metadata.get('resolved_at','')}\nticket_number: {d.metadata.get('ticket_number','')}\n"
                    f"CONTENT:\n{d.page_content}"
                    for d in candidates
                ]
            )

            llm_chain = LLMChain(llm=llm, prompt=PROMPT)
            answer = llm_chain.run(context=context, question=query)

        st.subheader("Answer")
        st.write(answer)

    st.markdown("---")
    st.caption("Tip: choose 'All' for cross-brand search, or select one brand to narrow results.")

# -------------------------------------------------------------
if __name__ == "__main__":
    if not os.getenv("ANTHROPIC_API_KEY"):
        raise RuntimeError("Please set ANTHROPIC_API_KEY first.")
    main()
