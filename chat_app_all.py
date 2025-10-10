#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from typing import List
import streamlit as st
from langchain_openai import OpenAIEmbeddings
from langchain_anthropic import ChatAnthropic
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_community.vectorstores import Qdrant as QdrantVS
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue

# ---------------------------------
# Environment and helper functions
# ---------------------------------
def need_env(var: str) -> str:
    val = os.getenv(var)
    if not val:
        st.error(f"{var} が設定されていません。")
        st.stop()
    return val

def doc_text(doc):
    """Safely get text from Document"""
    if getattr(doc, "page_content", None):
        return doc.page_content
    md = doc.metadata or {}
    payload = md.get("payload", md)
    if isinstance(payload, dict):
        for key in ["answer", "question", "text"]:
            if payload.get(key):
                return str(payload[key])
    return ""

# ---------------------------------
# Prompt template
# ---------------------------------
PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template=(
        "あなたは日英バイリンガルのサポートエンジニアです。\n"
        "以下のコンテキストの情報のみに基づいて簡潔に回答してください。\n"
        "—— コンテキスト ——\n{context}\n——\n\n"
        "質問: {question}\n"
        "回答:"
    )
)

# ---------------------------------
# Main app
# ---------------------------------
def main():
    st.set_page_config(page_title="All-Brands RAG Chat", page_icon="💬", layout="wide")
    st.title("💬 全ブランドRAGチャット")

    # Environment setup
    openai_key = need_env("OPENAI_API_KEY")
    anthropic_key = need_env("ANTHROPIC_API_KEY")
    qdrant_url = need_env("QDRANT_URL")
    qdrant_key = need_env("QDRANT_API_KEY")
    collection = os.getenv("QDRANT_COLLECTION", "support_logs_all")

    # Qdrant client + embeddings
    qclient = QdrantClient(url=qdrant_url, api_key=qdrant_key)
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=openai_key)
    vectordb = QdrantVS(client=qclient, collection_name=collection, embeddings=embeddings)
    retriever = vectordb.as_retriever(search_kwargs={"k": 5})

    # Sidebar info
    with st.sidebar:
        st.markdown("### 環境設定")
        st.code(
            f"""OPENAI_API_KEY      = {openai_key[:5]}...{openai_key[-4:]}
ANTHROPIC_API_KEY   = {anthropic_key[:5]}...{anthropic_key[-4:]}
QDRANT_URL          = {qdrant_url}
QDRANT_COLLECTION   = {collection}""",
            language="bash"
        )

    # Brand filter
    brand_filter = st.text_input("ブランドで絞り込み (例: JINS, Moncler など)", "")
    k = st.slider("Top-K 検索件数", 1, 10, 5)
    show_debug = st.checkbox("メタデータを表示", value=False)

    if brand_filter:
        retriever.search_kwargs["filter"] = Filter(
            must=[FieldCondition(key="brand", match=MatchValue(value=brand_filter))]
        )
    else:
        retriever.search_kwargs.pop("filter", None)

    query = st.text_input("質問を入力してください")
    ask = st.button("送信")

    if not ask or not query.strip():
        return

    # Retrieval
    try:
        retriever.search_kwargs["k"] = k
        candidates = retriever.get_relevant_documents(query)
    except Exception as e:
        st.error(f"取得エラー: {e}")
        return

    usable = [d for d in candidates if doc_text(d)]
    if not usable:
        st.warning("該当データが見つかりません。")
        return

    # Evidence display
    st.markdown("### エビデンス（Evidence）")
    citations = []
    context_blocks = []

    for i, d in enumerate(usable, 1):
        md = d.metadata or {}
        payload = md.get("payload", md)
        brand = payload.get("brand", "N/A")
        qa_id = payload.get("qa_id", "N/A")
        resolved = payload.get("resolved_at", "N/A")
        ticket = payload.get("ticket_number", "N/A")

        citations.append(f"({brand}, {qa_id}, {resolved}, {ticket})")

        st.markdown(f"**{i}. brand={brand}, qa_id={qa_id}, resolved_at={resolved}, ticket={ticket}**")
        with st.expander(f"スニペット {i}", expanded=False):
            st.write(doc_text(d))
            if show_debug:
                st.json(md)

        context_blocks.append(f"[{brand}][{qa_id}][{resolved}][{ticket}]\n{doc_text(d)}")

    context = "\n\n---\n\n".join(context_blocks)

    # LLM
    llm = ChatAnthropic(model="claude-3-5-haiku-20241022", temperature=0.2, api_key=anthropic_key)
    chain = LLMChain(llm=llm, prompt=PROMPT)

    with st.spinner("生成中..."):
        answer = chain.run({"context": context, "question": query})

    st.markdown("### 回答（Answer）")
    st.write(answer)

    st.markdown("### 引用（Citations）")
    for c in citations:
        st.text(c)


if __name__ == "__main__":
    main()
