#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Streamlit RAG Chatbot for Support Logs
- Uses OpenAI embeddings + Anthropic Claude for answers
- Stores vectors in Chroma (duckdb+parquet backend)
- Auto-builds index if not found
"""

# -------------------- Environment Setup --------------------
import os
os.environ["CHROMA_DB_IMPL"] = "duckdb+parquet"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from pathlib import Path
import subprocess
import sys
import pandas as pd
import streamlit as st

from chromadb.config import Settings
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_anthropic import ChatAnthropic
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain


# -------------------- Paths --------------------
INDEX_DIR = Path(os.getenv("INDEX_DIR", "/tmp/index_all"))
CSV_PATH  = Path(os.getenv("CSV_PATH", "./all_brands_support_log_embedding_ready.csv"))


# -------------------- Global Chroma Settings --------------------
DEFAULT_CLIENT_SETTINGS = Settings(
    chroma_db_impl="duckdb+parquet",
    persist_directory=str(INDEX_DIR),
    anonymized_telemetry=False,
)


# -------------------- Cached Vector Store --------------------
@st.cache_resource(show_spinner=False)
def get_vectorstore() -> Chroma:
    """Initialize and cache Chroma instance safely."""
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    return Chroma(
        persist_directory=str(INDEX_DIR),
        embedding_function=embeddings,
        client_settings=DEFAULT_CLIENT_SETTINGS,
    )


# -------------------- Retriever Loader --------------------
def load_retriever(k: int = 4, brand_filter: str | None = None):
    vs = get_vectorstore()
    if brand_filter and brand_filter != "All":
        return vs.as_retriever(search_kwargs={"k": k, "filter": {"brand": brand_filter}})
    return vs.as_retriever(search_kwargs={"k": k})


# -------------------- Helper: Build Index --------------------
def build_index_if_missing():
    """Auto-runs index_all.py if the index folder doesn't exist."""
    if not INDEX_DIR.exists():
        st.warning("Index not found. Building it now — please wait a few minutes...")
        build_env = os.environ.copy()
        build_env["INDEX_DIR"] = str(INDEX_DIR)
        build_env["CSV_PATH"] = str(CSV_PATH)

        result = subprocess.run(
            [sys.executable, "index_all.py", "--csv", str(CSV_PATH), "--out", str(INDEX_DIR)],
            capture_output=True,
            text=True,
            env=build_env,
        )
        if result.returncode != 0:
            st.error(f"Index build failed.\n\nSTDERR:\n{result.stderr}\n\nSTDOUT:\n{result.stdout}")
            st.stop()
        else:
            st.success("Index built successfully! You can now chat.")


# -------------------- Chat Function --------------------
def run_chat(question: str, retriever, model="claude-3-5-sonnet-20240620", temperature=0.1):
    docs = retriever.get_relevant_documents(question)
    context = "\n\n".join([d.page_content for d in docs])

    prompt = PromptTemplate.from_template("""
You are a support engineer analyzing QA logs.
Use the context below to answer clearly and concisely.

Context:
{context}

Question: {question}

Answer:
""")

    llm = ChatAnthropic(model=model, temperature=temperature)
    chain = LLMChain(prompt=prompt, llm=llm)
    return chain.run(context=context, question=question)


# -------------------- Streamlit App --------------------
def main():
    st.set_page_config(page_title="All-Brands Support Log Chat", layout="wide")
    st.title("🧠 All-Brands RAG Chatbot")
    st.caption("Search and summarize support logs across multiple brands.")

    # Secrets validation
    if not os.getenv("ANTHROPIC_API_KEY"):
        st.error("❌ Missing Anthropic API key. Set `ANTHROPIC_API_KEY` in Streamlit Secrets.")
        st.stop()
    if not os.getenv("OPENAI_API_KEY"):
        st.error("❌ Missing OpenAI API key. Set `OPENAI_API_KEY` in Streamlit Secrets.")
        st.stop()

    # Index check
    build_index_if_missing()

    # Sidebar controls
    with st.sidebar:
        st.header("🔧 Settings")
        topk = st.slider("Results per query (k)", 2, 10, 4)
        brand = st.selectbox("Brand filter", ["All", "LeCreuset", "Herno", "Moncler"])
        st.markdown("---")
        st.caption("Data source: all_brands_support_log_embedding_ready.csv")

    # Input + response area
    user_q = st.text_input("Ask a question about support logs:")
    if st.button("Search", type="primary"):
        if not user_q.strip():
            st.warning("Please enter a question first.")
        else:
            retriever = load_retriever(k=topk, brand_filter=brand)
            with st.spinner("Thinking..."):
                answer = run_chat(user_q, retriever)
                st.markdown("### 💬 Answer")
                st.write(answer)


if __name__ == "__main__":
    main()
