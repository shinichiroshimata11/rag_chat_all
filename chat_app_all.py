#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Streamlit RAG Chatbot (Qdrant + OpenAI Embeddings + Anthropic Claude)

- Brand filter (auto-fetched from Qdrant)
- JP/EN UI toggle (Japanese default)
- Top-K and Temperature controls
- Evidence + proper citations from Qdrant payload
"""

from __future__ import annotations

import os
from typing import List, Dict, Any, Optional

import streamlit as st
import pandas as pd

from qdrant_client import QdrantClient
from qdrant_client.http.models import Filter, FieldCondition, MatchValue

from langchain_community.vectorstores import Qdrant as LCQdrant
from langchain_openai import OpenAIEmbeddings
from langchain_anthropic import ChatAnthropic
from langchain.prompts import PromptTemplate
from langchain_core.documents import Document


# ---------------------------
# Helpers
# ---------------------------

def getenv_or_fail(name: str) -> str:
    val = os.getenv(name, "").strip()
    if not val:
        raise RuntimeError(f"Please set {name} in the environment.")
    return val


def get_payload(doc: Document) -> Dict[str, Any]:
    """
    Qdrant (via langchain_community.vectorstores.qdrant) returns Documents with:
      doc.page_content -> str | None
      doc.metadata -> {'payload': {...}}  OR sometimes flat dict
    This normalizes to a simple dict.
    """
    meta = doc.metadata or {}
    payload = meta.get("payload", meta)
    if not isinstance(payload, dict):
        payload = {}
    return payload


def doc_text(doc: Document) -> str:
    """
    Ensure we always have some content to show the user & model.
    Priority:
      1) doc.page_content
      2) payload['answer'] + payload['question'] if available
      3) join all str payload values as last resort
    """
    if doc.page_content:
        return str(doc.page_content)

    p = get_payload(doc)
    qa = []
    if p.get("question"):
        qa.append(f"Q: {p.get('question')}")
    if p.get("answer"):
        qa.append(f"A: {p.get('answer')}")
    if qa:
        return "\n".join(qa)

    # Fallback: join any string-like payload values
    parts = []
    for k, v in p.items():
        try:
            sv = str(v)
        except Exception:
            continue
        if sv and sv.lower() not in ("none", "nan", "null"):
            parts.append(f"{k}={sv}")
    return "\n".join(parts) if parts else ""


def build_prompt(lang: str = "ja") -> PromptTemplate:
    if lang == "en":
        template = (
            "You are a concise bilingual (Japanese/English) support engineer.\n"
            "Answer ONLY from the context below. If the answer is not present, say so and ask to escalate.\n\n"
            "Rules:\n"
            "- Include the source brand name in the answer if available.\n"
            "- Use numbered steps where helpful.\n"
            "- Add notes for risks/version differences if relevant.\n\n"
            "—— Context ——\n{context}\n—— End Context ——\n\n"
            "Question: {question}\n"
            "Answer:"
        )
    else:
        template = (
            "あなたは簡潔な日英バイリンガルのサポートエンジニアです。\n"
            "以下のコンテキストに含まれる情報のみで回答してください。見つからない場合はその旨を伝え、エスカレーションを促してください。\n\n"
            "ルール:\n"
            "- 可能であれば回答内に**ブランド名**を含める。\n"
            "- 必要に応じて番号付き手順を使う。\n"
            "- リスクやバージョン差異の注意点があれば追記する。\n\n"
            "—— コンテキスト ——\n{context}\n—— コンテキストここまで ——\n\n"
            "質問: {question}\n"
            "回答:"
        )
    return PromptTemplate(input_variables=["context", "question"], template=template)


def connect_qdrant() -> tuple[QdrantClient, str, OpenAIEmbeddings]:
    q_url = getenv_or_fail("QDRANT_URL")
    q_key = getenv_or_fail("QDRANT_API_KEY")
    collection = getenv_or_fail("QDRANT_COLLECTION")
    oai_key = getenv_or_fail("OPENAI_API_KEY")

    embeddings = OpenAIEmbeddings(api_key=oai_key, model="text-embedding-3-small")
    client = QdrantClient(url=q_url, api_key=q_key)
    return client, collection, embeddings


def make_vectorstore(client: QdrantClient, collection: str, embeddings: OpenAIEmbeddings) -> LCQdrant:
    return LCQdrant(
        client=client,
        collection_name=collection,
        embeddings=embeddings,
    )


def get_all_brands(client: QdrantClient, collection: str, sample: int = 1000) -> List[str]:
    """
    Pulls up to `sample` points and extracts distinct non-empty 'brand' payload values.
    """
    seen = set()
    next_offset = None
    fetched = 0

    while fetched < sample:
        points, next_offset = client.scroll(
            collection_name=collection,
            with_payload=True,
            limit=min(256, sample - fetched),
            offset=next_offset,
        )
        if not points:
            break
        for p in points:
            payload = p.payload or {}
            brand = payload.get("brand")
            if brand and isinstance(brand, str):
                seen.add(brand.strip())
        fetched += len(points)
        if next_offset is None:
            break

    return sorted(seen, key=lambda x: x.lower())


def build_filter(brand: Optional[str]) -> Optional[Filter]:
    if not brand or brand == "（すべてのブランド）" or brand == "(All brands)":
        return None
    # Match on payload.key == value
    return Filter(must=[FieldCondition(key="brand", match=MatchValue(value=brand))])


def retrieve(
    vs: LCQdrant,
    query: str,
    k: int,
    brand: Optional[str] = None,
) -> List[Document]:
    flt = build_filter(brand)
    # Use vectorstore API directly to keep control
    if flt is None:
        docs = vs.similarity_search(query, k=k)
    else:
        docs = vs.similarity_search(query, k=k, filter=flt)
    # Ensure page_content is never None to satisfy pydantic
    fixed: List[Document] = []
    for d in docs:
        text = doc_text(d)
        if not text:
            # force a minimal non-empty text to avoid validation error
            p = get_payload(d)
            text = p.get("answer") or p.get("question") or "（内容なし / no content）"
        fixed.append(Document(page_content=text, metadata=d.metadata))
    return fixed


def run_llm(
    question: str,
    context: str,
    temperature: float,
    lang: str = "ja",
) -> str:
    anthropic_key = getenv_or_fail("ANTHROPIC_API_KEY")
    llm = ChatAnthropic(api_key=anthropic_key, model="claude-3-5-sonnet-20241022", temperature=temperature)
    prompt = build_prompt(lang=lang)
    chain = prompt | llm
    out = chain.invoke({"context": context, "question": question})
    # langchain_anthropic outputs a BaseMessage; get .content (str | list[dict])
    if hasattr(out, "content"):
        return out.content if isinstance(out.content, str) else str(out.content)
    return str(out)


# ---------------------------
# Streamlit UI
# ---------------------------

def main():
    st.set_page_config(page_title="Support RAG Chat", page_icon="💬", layout="wide")

    # Language toggle
    lang = st.sidebar.radio("Language / 言語", options=["日本語 (JA)", "English (EN)"], index=0)
    is_ja = (lang == "日本語 (JA)")
    L = {
        "title": "サポートRAGチャット" if is_ja else "Support RAG Chat",
        "ask": "質問を入力してください…" if is_ja else "Type your question…",
        "brand": "ブランド" if is_ja else "Brand",
        "brand_all": "（すべてのブランド）" if is_ja else "(All brands)",
        "topk": "Top-K（取得件数）" if is_ja else "Top-K (results)",
        "temp": "温度（創造性）" if is_ja else "Temperature (creativity)",
        "btn": "送信" if is_ja else "Send",
        "evidence": "エビデンス（Evidence）" if is_ja else "Evidence",
        "snippet": "スニペット" if is_ja else "Snippet",
        "citations": "引用（Citations）" if is_ja else "Citations",
        "health": "接続ヘルスチェック" if is_ja else "Health Check",
        "connected": "Qdrantに接続しました。" if is_ja else "Connected to Qdrant.",
        "not_connected": "Qdrant接続に失敗しました。" if is_ja else "Failed to connect to Qdrant.",
        "answer": "回答（Answer）" if is_ja else "Answer",
        "debug": "デバッグ表示" if is_ja else "Show Debug",
    }

    st.title(f"💬 {L['title']}")

    # Controls
    with st.sidebar:
        show_debug = st.checkbox(L["debug"], value=False)
        topk = st.slider(L["topk"], min_value=1, max_value=8, value=5, step=1)
        temperature = st.slider(L["temp"], min_value=0.0, max_value=1.0, value=0.2, step=0.05)

    # Health check + client init
    try:
        client, collection, embeddings = connect_qdrant()
        st.sidebar.success(f"{L['connected']} ({collection})")
    except Exception as e:
        st.sidebar.error(f"{L['not_connected']} - {e}")
        st.stop()

    # Vector store
    vs = make_vectorstore(client, collection, embeddings)

    # Brand picker (from live collection)
    try:
        brands = get_all_brands(client, collection, sample=1000)
    except Exception:
        brands = []

    brands_opts = [L["brand_all"]] + brands if brands else [L["brand_all"]]
    brand = st.selectbox(L["brand"], options=brands_opts, index=0)

    # Query box
    query = st.text_input(L["ask"], value="")
    submit = st.button(L["btn"], type="primary")

    # Health check expander
    with st.expander(L["health"]):
        st.write("OPENAI_API_KEY =", "set" if os.getenv("OPENAI_API_KEY") else "missing")
        st.write("ANTHROPIC_API_KEY =", "set" if os.getenv("ANTHROPIC_API_KEY") else "missing")
        st.write("QDRANT_URL =", os.getenv("QDRANT_URL"))
        st.write("QDRANT_COLLECTION =", os.getenv("QDRANT_COLLECTION"))
        # quick collection probe
        try:
            info = client.get_collection(collection)
            st.write("Collection status:", getattr(info, "status", "ok"))
        except Exception as e:
            st.write("Collection check error:", str(e))

    if not submit or not query.strip():
        return

    # Retrieve documents
    try:
        candidates = retrieve(vs, query.strip(), k=topk, brand=None if brand == L["brand_all"] else brand)
    except Exception as e:
        st.error("Error during retrieval.")
        if show_debug:
            st.exception(e)
        return

    # Evidence + build context
    st.markdown(f"### {L['evidence']}")
    citations: List[str] = []
    context_blocks: List[str] = []

    for i, d in enumerate(candidates, 1):
        payload = get_payload(d)

        brand_val = payload.get("brand", "N/A")
        qa_id = payload.get("qa_id", "N/A")
        resolved = payload.get("resolved_at", "N/A")
        ticket = payload.get("ticket_number", "N/A")

        # Store citation tuple
        citations.append(f"({brand_val}, {qa_id}, {resolved}, {ticket})")

        st.markdown(f"**{i}. brand={brand_val}, qa_id={qa_id}, resolved_at={resolved}, ticket={ticket}**")
        with st.expander(f"{L['snippet']} {i}", expanded=False):
            st.write(doc_text(d))
            if show_debug:
                st.json(d.metadata)

        context_blocks.append(
            f"[brand={brand_val} qa_id={qa_id} resolved_at={resolved} ticket={ticket}]\n{doc_text(d)}"
        )

    # Compose context for LLM
    full_context = "\n\n----\n\n".join(context_blocks)

    # Run LLM
    try:
        answer = run_llm(
            question=query.strip(),
            context=full_context,
            temperature=temperature,
            lang="ja" if is_ja else "en",
        )
    except Exception as e:
        st.error("LLM 呼び出しでエラーが発生しました。" if is_ja else "Error while calling the LLM.")
        if show_debug:
            st.exception(e)
        return

    # Render answer
    st.markdown(f"### {L['answer']}")
    st.write(answer)

    # Citations
    st.markdown(f"### {L['citations']}")
    if citations:
        for c in citations:
            st.write(c)
    else:
        st.write("(なし)" if is_ja else "(none)")

    # Optional debug
    if show_debug:
        st.caption("— debug —")
        dbg = [{"payload": get_payload(d), "page_content_len": len(doc_text(d))} for d in candidates]
        st.json(dbg)


if __name__ == "__main__":
    main()
