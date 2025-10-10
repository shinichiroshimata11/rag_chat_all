#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
All-Brands RAG Chatbot (Qdrant + OpenAI embeddings + Anthropic)
- 日本語UI
- ブランド絞り込みドロップダウン
- 頑健なメタデータ抽出（brand / qa_id / resolved_at / ticket_number）
"""

import os
from typing import Any, Dict, Iterable, List, Set

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


def _flatten_dict(d: Dict[str, Any], depth: int = 2) -> Dict[str, Any]:
    """
    小さい深さでネスト辞書を平坦化する。キー衝突は上書きOK。
    例: {'payload': {...}, 'qdrant__payload': {...}} を一つにまとめる。
    """
    out: Dict[str, Any] = {}
    if not isinstance(d, dict):
        return out
    stack: List[tuple[Dict[str, Any], int]] = [(d, 0)]
    while stack:
        cur, lvl = stack.pop()
        for k, v in cur.items():
            if isinstance(v, dict) and lvl < depth:
                stack.append((v, lvl + 1))
            else:
                if v is not None:
                    out[k] = v
    return out


def collect_metadata(doc) -> Dict[str, Any]:
    """
    LangChainのDocument.metadataの中に、実際のQdrant payloadが
    いろいろなキーで入る場合に対応（'payload', 'qdrant__payload', 'document' など）。
    """
    md: Dict[str, Any] = {}

    # 1) まず doc.metadata 自体
    if hasattr(doc, "metadata") and isinstance(doc.metadata, dict):
        md.update(_flatten_dict(doc.metadata))

        # 2) よくあるネスト候補を順に統合
        for k in ("payload", "qdrant__payload", "document", "metadata", "data"):
            if k in doc.metadata and isinstance(doc.metadata[k], dict):
                md.update(_flatten_dict(doc.metadata[k]))

    # 3) 念のため doc.dict() 内も見る（実装差異への保険）
    try:
        as_dict = getattr(doc, "dict", None)
        if callable(as_dict):
            d_all = doc.dict()
            if "metadata" in d_all and isinstance(d_all["metadata"], dict):
                md.update(_flatten_dict(d_all["metadata"]))
    except Exception:
        pass

    return md


def doc_text(doc) -> str:
    """
    本文テキストを安全に抽出。
    優先: doc.page_content -> payload.page_content -> payload.answer -> "".
    """
    # 1) page_content が入っていれば最優先
    if getattr(doc, "page_content", None):
        return doc.page_content

    # 2) メタデータから拾う
    md = collect_metadata(doc)
    for key in ("page_content", "text", "answer", "content", "body"):
        if md.get(key):
            return str(md[key])

    # 3) 最後の保険：question + answer の連結（両方あれば）
    q = md.get("question")
    a = md.get("answer")
    if q or a:
        return f"Q: {q or ''}\nA: {a or ''}".strip()

    return ""


def meta_get(md: Dict[str, Any], candidates: Iterable[str], default: str = "N/A") -> str:
    for k in candidates:
        v = md.get(k)
        if v not in (None, "", "null", "None"):
            return str(v)
    return default


def render_health(client: QdrantClient, collection: str):
    try:
        info = client.get_collection(collection)
        st.success(f"Qdrant接続OK：コレクション **{collection}**（status: {info.status}）")
    except Exception as e:
        st.error(f"Qdrant接続/コレクション確認に失敗: {e}")


def fetch_brand_options(client: QdrantClient, collection: str, max_scan: int = 300) -> List[str]:
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

    # サイドバー（環境変数の簡易表示）
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

    # 必須ENV
    openai_key = need_env("OPENAI_API_KEY")
    anthropic_key = need_env("ANTHROPIC_API_KEY")
    qdrant_url = need_env("QDRANT_URL")
    qdrant_key = need_env("QDRANT_API_KEY")
    collection = os.getenv("QDRANT_COLLECTION", "support_logs_all")

    # クライアント・ベクターストア
    qclient = QdrantClient(url=qdrant_url, api_key=qdrant_key)
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=openai_key)
    vectordb = QdrantVS(client=qclient, collection_name=collection, embeddings=embeddings)
    retriever = vectordb.as_retriever(search_kwargs={"k": 5})

    with st.sidebar:
        render_health(qclient, collection)

    # オプション
    with st.expander("オプション / Options", expanded=True):
        brand_options = ["（すべて）"] + fetch_brand_options(qclient, collection, max_scan=300)
        sel_brand = st.selectbox("ブランドで絞り込み", brand_options, index=0)
        k = st.slider("Top-K（取得件数）", 1, 10, 5)
        temperature = st.slider("温度（多様性）", 0.0, 1.0, 0.2, 0.1)
        show_debug = st.checkbox("デバッグ: メタデータを表示", value=False)

    if sel_brand and sel_brand != "（すべて）":
        retriever.search_kwargs["filter"] = Filter(
            must=[FieldCondition(key="brand", match=MatchValue(value=sel_brand))]
        )
    else:
        retriever.search_kwargs.pop("filter", None)

    query = st.text_input("質問", placeholder="例: Cegidがフリーズした時の対処は？")
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
        st.warning("該当ドキュメントが見つかりません。")
        return

    # ========= エビデンス =========
    st.markdown("### エビデンス（Evidence）")
    citations: List[str] = []
    context_blocks: List[str] = []

    for i, d in enumerate(usable, 1):
        md_all = collect_metadata(d)

        brand = meta_get(md_all, ("brand",))
        qa_id = meta_get(md_all, ("qa_id", "qaid", "qaId"))
        resolved = meta_get(md_all, ("resolved_at", "resolvedAt", "date", "resolved"))
        ticket = meta_get(md_all, ("ticket_number", "ticket", "ticket_no"))

        # ここで N/A にならないように、question/answer から推測補完も可能（必要なら有効化）
        # if brand == "N/A" and md_all.get("question"):
        #     brand = "不明ブランド"

        citations.append(f"({brand}, {qa_id}, {resolved}, {ticket})")

        st.markdown(f"**{i}. brand={brand}, qa_id={qa_id}, resolved_at={resolved}, ticket={ticket}**")
        with st.expander(f"スニペット {i}", expanded=False):
            st.write(doc_text(d))
            if show_debug:
                st.caption("↓ メタデータ（フラット化後）")
                st.json(md_all)

        context_blocks.append(
            f"[brand={brand} qa_id={qa_id} resolved_at={resolved} ticket={ticket}]\n{doc_text(d)}"
        )

    context = "\n\n---\n\n".join(context_blocks)
    citations_text = "\n".join(citations)

    # モデル
    llm = ChatAnthropic(model="claude-3-5-haiku-20241022", temperature=temperature, api_key=anthropic_key)
    chain = LLMChain(llm=llm, prompt=PROMPT)

    with st.spinner("生成中… / Thinking…"):
        answer = chain.run({"context": context, "question": query})

    st.markdown("### 回答（Answer）")
    st.write(answer)

    st.markdown("### 引用（Citations）")
    st.text(citations_text)


if __name__ == "__main__":
    main()
