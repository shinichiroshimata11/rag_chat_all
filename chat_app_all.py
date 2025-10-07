#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Streamlit RAG Chatbot for all brands.
Uses Chroma vectorstore + OpenAI embeddings + Anthropic Claude for answers.
"""

import os
from pathlib import Path
import pandas as pd
import streamlit as st

INDEX_DIR = Path(os.getenv("INDEX_DIR", "./index_all"))   # local default
CSV_PATH  = Path(os.getenv("CSV_PATH", "./all_brands_support_log_embedding_ready.csv"))

from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_anthropic import ChatAnthropic
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA, LLMChain

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
    )
)

# -------------------------------------------------------------
# Helper function
# -------------------------------------------------------------
def load_retriever(index_dir: Path, k: int = 4, brand_filter: str | None = None):
    """Load the Chroma vector retriever."""
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vs = Chroma(persist_directory=str(index_dir), embedding_function=embeddings)
    if brand_filter and brand_filter != "All":
        return vs.as_retriever(search_kwargs={"k": k, "filter": {"brand": brand_filter}})
    return vs.as_retriever(search_kwargs={"k": k})

# -------------------------------------------------------------
# Main App
# -------------------------------------------------------------
def main():
    st.set_page_config(page_title="All-brands RAG Chat", layout="wide")
    st.title("All-brands RAG Chat (Claude + Chroma)")

    # Sidebar controls
    index_dir = INDEX_DIR
    if not index_dir.exists():
        st.warning("Index not found. Run: python index_all.py --csv ./all_brands_support_log_embedding_ready.csv")
        st.stop()

    # Load brand list
    merged_csv = CSV_PATH
    brands = ["All"]
    if merged_csv.exists():
        try:
            df = pd.read_csv(merged_csv, usecols=["brand"])
            unique_brands = sorted([b for b in df["brand"].dropna().unique() if str(b).strip()])
            brands += unique_brands
        except Exception:
            pass

    brand = st.sidebar.selectbox("Brand filter", options=brands, index=0)
    topk = st.sidebar.slider("Top-K documents", min_value=2, max_value=12, value=5, step=1)
    temperature = st.sidebar.slider("Temperature", 0.0, 1.0, 0.0, 0.1)
    model = st.sidebar.selectbox(
        "Claude model",
        ["claude-3-7-sonnet-20250219", "claude-3-5-haiku-20241022", "claude-3-haiku-20240307"],
        index=0
    )

    retriever = load_retriever(index_dir, k=topk, brand_filter=brand)
    llm = ChatAnthropic(model=model, temperature=temperature)

    # Input box
    query = st.text_input("Ask a question")

    if st.button("Ask") and query.strip():
        with st.spinner("Searching and composing…"):
            # Retrieve documents
            candidates = retriever.get_relevant_documents(query)

            # Show evidence
            st.subheader("Evidence")
            for i, d in enumerate(candidates, 1):
                m = d.metadata
                st.markdown(
                    f"- **{i}.** brand={m.get('brand','')}, qa_id={m.get('qa_id','')}, "
                    f"resolved_at={m.get('resolved_at','')}, ticket={m.get('ticket_number','')}"
                )

            # Build context including brand info
            context_blocks = []
            for d in candidates:
                m = d.metadata
                block = (
                    f"Brand: {m.get('brand','')}\n"
                    f"qa_id: {m.get('qa_id','')}\n"
                    f"resolved_at: {m.get('resolved_at','')}\n"
                    f"ticket_number: {m.get('ticket_number','')}\n"
                    f"CONTENT:\n{d.page_content}"
                )
                context_blocks.append(block)
            context = "\n\n---\n\n".join(context_blocks)

            # Run Claude directly
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
