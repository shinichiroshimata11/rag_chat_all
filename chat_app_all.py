#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
All-Brands RAG Chatbot (Qdrant + OpenAI embeddings + Anthropic)
- 日本語UI
- ブランド絞り込みドロップダウン
- Qdrantのpayloadには page_content（本文）/ brand / qa_id / resolved_at / ticket_number が入っている想定
"""

import os
from typing import List, Dict, Any, Set

import streamlit as st
from langchain_openai import OpenAIEmbeddings
from langchain_anthropic import ChatAnthropic
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_community.vectorstores import Qdrant as QdrantVS
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue


# -------------------- helpers --------------------
def need_env(var: str) -> str:
    val = os.getenv(var)
    if not val:
        raise RuntimeError(f"{var} が設定されていません（環境変数）。")
    return val


def doc_text(d) -> str:
    """安全にテキストを取得（page_content優先、無ければmetadata経由）"""
    return (getattr(d, "page_content", None)
            or d.metadata.get("page_content")
            or d.metadata.get("answer")
            or "")


def mget(md: Dict[str, Any], key: str, default: str = "") -> str:
    v = md.get(key)
    return default if v is None else str(v)


def render_health(client: QdrantClient, collection: str):
    try:
        info = client.get_collection(collection)
        st.success(f"Qdrant接続OK：コレクション **{collection}**（status: {info.status}）")
    except Exception as e:
        st.error(f"Qdrant接続/コレクション確認に失敗: {e}")


def fetch_brand_options(client: QdrantClient, collection: str, max_scan: int = 300) -> List[str]:
    """
    Qdrantには集計APIがないため、scrollで先頭 max_scan 件を覗いてブランド候補を作る。
    コストを抑えるため取り過ぎない設計。
    """
    uniq: Set[str] = set()
    next_offset = None
    remaining = max_scan
    while remaining > 0:
        limit = min(100, remaining)
        points, next_offset = client.scroll(
            collection_name=collection,
            limit=limit,
            with_payload=True,
            offset=next_offset,
        )
        for p in points:
            b = p.payload.get("brand")
            if b:
                uniq.add(str(b))
        remaining -= len(points)
        if not next_offset or len(points) == 0:
            break
    return sorted(uniq)


# -------------------- prompt --------------------
PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template=(
        "あなたは日英バイリンガルのサポートエンジニアです。\n"
        "以下のコンテキストの情報のみに基づいて、簡潔かつ正確に回答してください。\n"
        "もし答えが見つからない場合は、その旨を述べてエスカレーションを促してください。\n\n"
        "ルール:\n"
        "- 回答内に**ソースのブランド名**を入れること。\n"
        "- 適切な場合は番号付き手順を使うこと。\n"
        "- バージョン差異・リスクがあれば注意書きを入れること。\n"
        "- 最後に (brand, qa_id, resolved_at, ticket_number) を列挙すること。\n\n"
        "—— コンテキスト ——\n{context}\n—— コンテキストここまで ——\n\n"
        "質問: {question}\n"
        "回答:"
    )
)


# -------------------- app --------------------
def main():
    st.set_page_config(page_title="全ブランドRAGサポート", page_icon="💬", layout="wide")
    st.title("全ブランド サポートRAG 💬")

    # サイドバー（環境変数プレビュー & ヘルスチェック）
    with st.sidebar:
        st.markdown("### 🔧 接続ヘルスチェック")
        oa = os.getenv("OPENAI_API_KEY", "")
        an = os.getenv("ANTHROPIC_API_KEY", "")
        qurl = os.getenv("QDRANT_URL", "")
        qkey = os.getenv("QDRANT_API_KEY", "")
        qcol = os.getenv("QDRANT_COLLECTION", "support_logs_all")

        st.code(
            f"""OPENAI_API_KEY      = {oa[:4]}…{oa[-4:] if oa else ''}
ANTHROPIC_API_KEY   = {an[:4]}…{an[-4:] if an else ''}
QDRANT_URL          = {qurl}
QDRANT_COLLECTION   = {qcol}
""",
            language="bash",
        )

    # 必須環境変数
    openai_key = need_env("OPENAI_API_KEY")
    anthropic_key = need_env("ANTHROPIC_API_KEY")
    qdrant_url = need_env("QDRANT_URL")
    qdrant_key = need_env("QDRANT_API_KEY")
    collection = os.getenv("QDRANT_COLLECTION", "support_logs_all")

    # クライアント
    qclient = QdrantClient(url=qdrant_url, api_key=qdrant_key)
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=openai_key)

    # VectorStore（content_payload_key=page_content 前提）
    vectordb = QdrantVS(client=qclient, collection_name=collection, embeddings=embeddings)
    retriever = vectordb.as_retriever(search_kwargs={"k": 5})

    with st.sidebar:
        render_health(qclient, collection)

    # ========= オプション&ブランド絞り込み =========
    with st.expander("オプション / Options", expanded=True):
        # ブランド候補の取得（軽量スキャン）
        brand_options = ["（すべて）"] + fetch_brand_options(qclient, collection, max_scan=300)
        sel_brand = st.selectbox("ブランドで絞り込み", brand_options, index=0, help="Qdrantのpayload.brandでフィルタします")
        k = st.slider("Top-K（取得件数）", 1, 10, 5, help="上位何件の類似ドキュメントを使うか")
        temperature = st.slider("温度（多様性）", 0.0, 1.0, 0.2, 0.1, help="数値が高いほど出力が多様になります")

    # フィルタ設定（選択された場合のみ）
    if sel_brand and sel_brand != "（すべて）":
        retriever.search_kwargs["filter"] = Filter(
            must=[FieldCondition(key="brand", match=MatchValue(value=sel_brand))]
        )
    else:
        # 既存のfilterが残らないよう明示削除
        retriever.search_kwargs.pop("filter", None)

    # ========= 入力＆実行 =========
    query = st.text_input("質問", placeholder="例: Cegidがフリーズした時の対処は？ / What to do when Cegid freezes?")
    ask = st.button("実行（Ask）")

    if not ask or not query.strip():
        return

    # 取得
    try:
        retriever.search_kwargs["k"] = k
        candidates = retriever.get_relevant_documents(query)
    except Exception as e:
        st.error(f"検索中にエラーが発生しました。\n\n{e}")
        return

    usable = [d for d in candidates if doc_text(d)]
    if not usable:
        st.warning("該当ドキュメントが見つかりませんでした。条件（ブランド/Top-K）を調整して再試行してください。")
        return

    # 証拠（Evidence）
    st.markdown("### エビデンス（Evidence）")
    for i, d in enumerate(usable, 1):
        brand = mget(d.metadata, "brand", "N/A")
        qa_id = mget(d.metadata, "qa_id", "N/A")
        resolved = mget(d.metadata, "resolved_at", "N/A")
        ticket = mget(d.metadata, "ticket_number", "N/A")
        st.write(f"{i}. brand={brand}, qa_id={qa_id}, resolved_at={resolved}, ticket={ticket}")
        with st.expander(f"スニペット {i}", expanded=False):
            st.write(doc_text(d))

    # コンテキストを生成
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

    # モデル
    llm = ChatAnthropic(model="claude-3-5-haiku-20241022", temperature=temperature, api_key=anthropic_key)
    chain = LLMChain(llm=llm, prompt=PROMPT)

    with st.spinner("生成中… / Thinking…"):
        answer = chain.run({"context": context, "question": query})

    st.markdown("### 回答（Answer）")
    st.write(answer)


if __name__ == "__main__":
    main()
