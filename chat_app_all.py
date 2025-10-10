#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Streamlit RAG Chatbot for all brands (Qdrant + OpenAI embeddings + Anthropic for answers).
- Assumes Qdrant payload has "page_content" with the answer text (see upload_to_qdrant.py).
"""

import os
from typing import List, Dict, Any

import streamlit as st
from langchain_openai import OpenAIEmbeddings
from langchain_anthropic import ChatAnthropic
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_community.vectorstores import Qdrant as QdrantVS
from qdrant_client import QdrantClient


# ---------- helpers ----------
def need_env(var: str):
    val = os.getenv(var)
    if not val:
        raise RuntimeError(f"Please set {var} first.")
    return val


def doc_text(d) -> str:
    """Robust text getter: use d.page_content; fallback to metadata."""
    return (getattr(d, "page_content", None)
            or d.metadata.get("page_content")
            or d.metadata.get("answer")
            or "")


def mget(md: Dict[str, Any], key: str, default: str = "") -> str:
    v = md.get(key)
    return "" if v is None else str(v)


def render_health(client: QdrantClient, collection: str):
    try:
        info = client.get_collection(collection)
        st.success(f"Connected to Qdrant. Collection **{collection}** exists (status: {info.status}).")
    except Exception as e:
        st.error(f"Qdrant check failed: {e}")


# ---------- prompt ----------
PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template=(
        "You are a bilingual (Japanese/English) support engineer.\n"
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


# ---------- app ----------
def main():
    st.set_page_config(page_title="All-Brands RAG", page_icon="💬", layout="wide")
    st.title("All-Brands Support RAG 💬")

    # sidebar env preview
    with st.sidebar:
        st.markdown("### 🔧 RAG Health Check")
        oa = os.getenv("OPENAI_API_KEY", "")
        an = os.getenv("ANTHROPIC_API_KEY", "")
        qurl = os.getenv("QDRANT_URL", "")
        qkey = os.getenv("QDRANT_API_KEY", "")
        qcol = os.getenv("QDRANT_COLLECTION", "support_logs_all")

        st.code(
            f"""OPENAI_API_KEY    = {oa[:4]}…{oa[-4:] if oa else ''}
ANTHROPIC_API_KEY = {an[:4]}…{an[-4:] if an else ''}
QDRANT_URL        = {qurl}
QDRANT_COLLECTION = {qcol}
""",
            language="bash",
        )

    # hard fail if API keys missing (so the app doesn't spin forever)
    openai_key = need_env("OPENAI_API_KEY")
    anthropic_key = need_env("ANTHROPIC_API_KEY")
    qdrant_url = need_env("QDRANT_URL")
    qdrant_key = need_env("QDRANT_API_KEY")
    collection = os.getenv("QDRANT_COLLECTION", "support_logs_all")

    # clients
    qclient = QdrantClient(url=qdrant_url, api_key=qdrant_key)
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=openai_key)

    # Vector store: we rely on default content_payload_key="page_content"
    vectordb = QdrantVS(
        client=qclient,
        collection_name=collection,
        embeddings=embeddings,
    )
    retriever = vectordb.as_retriever(search_kwargs={"k": 5})

    with st.sidebar:
        render_health(qclient, collection)

    # UI controls
    with st.expander("Options", expanded=False):
        k = st.slider("Top-K", 1, 10, 5)
        temperature = st.slider("Temperature", 0.0, 1.0, 0.2, 0.1)

    query = st.text_input("質問 / Question", placeholder="例: Cegidがフリーズした時の対処は？")
    ask = st.button("Ask")

    if not ask or not query.strip():
        return

    # retrieve (defensive against any weird documents)
    try:
        retriever.search_kwargs["k"] = k
        candidates = retriever.get_relevant_documents(query)
    except Exception as e:
        st.error(f"Error during retrieval.\n\n{e}")
        return

    # Build context and evidence safely
    usable = [
        d for d in candidates
        if doc_text(d)  # must have text
    ]
    if not usable:
        st.warning("No relevant documents found.")
        return

    # Evidence panel
    st.markdown("### Evidence")
    for i, d in enumerate(usable, 1):
        brand = mget(d.metadata, "brand", "N/A")
        qa_id = mget(d.metadata, "qa_id", "N/A")
        resolved = mget(d.metadata, "resolved_at", "N/A")
        ticket = mget(d.metadata, "ticket_number", "N/A")
        st.write(f"{i}. brand={brand}, qa_id={qa_id}, resolved_at={resolved}, ticket={ticket}")
        with st.expander(f"Snippet {i}", expanded=False):
            st.write(doc_text(d))

    context_blocks = []
    for d in usable:
        brand = mget(d.metadata, "brand")
        qa_id = mget(d.metadata, "qa_id")
        resolved = mget(d.metadata, "resolved_at")
        ticket = mget(d.metadata, "ticket_number")
        context_blocks.append(
            f"[brand={brand} qa_id={qa_id} resolved_at={resolved} ticket={ticket}]\n{doc_text(d)}"
        )
    context = "\n\n---\n\n".join(context_blocks)

    # LLM
    llm = ChatAnthropic(model="claude-3-5-haiku-20241022", temperature=temperature, api_key=anthropic_key)
    chain = LLMChain(llm=llm, prompt=PROMPT)

    with st.spinner("Thinking…"):
        answer = chain.run({"context": context, "question": query})

    st.markdown("### Answer")
    st.write(answer)


if __name__ == "__main__":
    main()
