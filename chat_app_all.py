#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Streamlit RAG Chatbot (Qdrant + OpenAI + Anthropic)
- Remote Qdrant collection with metadata filter on "brand"
- OpenAI text-embedding-3-small
- Anthropic Claude for generation
- Shows evidence with brand + citations
"""

import os
from typing import Optional, Dict, Any, List

import pandas as pd
import streamlit as st

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from langchain_community.vectorstores import Qdrant as LCQdrant
from langchain_openai import OpenAIEmbeddings
from langchain_anthropic import ChatAnthropic
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA


# -----------------------
# Config & Constants
# -----------------------
APP_TITLE = "All-brands RAG Chat (Claude + Qdrant)"
DEFAULT_TOPK = 5
DEFAULT_TEMPERATURE = 0.0

# Default Anthropic models you likely have:
ANTHROPIC_MODELS = [
    "claude-3-7-sonnet-20250219",  # recommended if your key has access
    "claude-3-5-haiku-20241022",   # fast/cheap fallback
]

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


# -----------------------
# Helpers
# -----------------------
def mask(s: Optional[str], keep: int = 6) -> str:
    if not s:
        return "(not set)"
    if len(s) <= keep:
        return s
    return s[:keep] + "…"


def build_brand_filter(brand: Optional[str]) -> Optional[Dict[str, Any]]:
    """Qdrant filter dict for a single brand value (metadata key: 'brand')."""
    if brand and brand != "All":
        return {
            "must": [
                {"key": "brand", "match": {"value": brand}},
            ]
        }
    return None


def load_retriever(
    client: QdrantClient,
    collection: str,
    k: int = 5,
    brand: Optional[str] = None,
):
    """Create a LangChain retriever backed by Qdrant."""
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vs = LCQdrant(
        client=client,
        collection_name=collection,
        embeddings=embeddings,
    )

    flt = build_brand_filter(brand)
    if flt:
        return vs.as_retriever(search_kwargs={"k": k, "filter": flt})
    return vs.as_retriever(search_kwargs={"k": k})


def qdrant_health_panel(client: QdrantClient, coll: str):
    """Robust Qdrant health check that works across client versions."""
    st.subheader("🔧 RAG Health Check")

    # Show env summary (masked)
    env_rows = [
        ("OPENAI_API_KEY", mask(os.getenv("OPENAI_API_KEY"))),
        ("ANTHROPIC_API_KEY", mask(os.getenv("ANTHROPIC_API_KEY"))),
        ("QDRANT_URL", os.getenv("QDRANT_URL") or "(not set)"),
        ("QDRANT_API_KEY", mask(os.getenv("QDRANT_API_KEY"))),
        ("QDRANT_COLLECTION", coll),
    ]
    df = pd.DataFrame(env_rows, columns=["Variable", "Value"])
    st.dataframe(df, hide_index=True, use_container_width=True)

    st.markdown("**Qdrant Connectivity**")
    try:
        info = client.get_collection(coll)  # models.CollectionInfo
        st.success(f"Connected to Qdrant. Collection **{coll}** exists.")

        # points count (safe)
        try:
            cnt = client.count(coll).count
            st.caption(f"Points in collection: {cnt}")
        except Exception as e_cnt:
            st.warning(f"Could not read points count: {e_cnt}")

        # Robust vector params (single or multi vector)
        try:
            vp = info.config.params.vectors
            details = ""
            # Case 1: single VectorParams (has .size)
            if hasattr(vp, "size"):
                dist = getattr(vp, "distance", None)
                details = f"Vector size={vp.size}" + (f", distance={dist}" if dist else "")
            # Case 2: multi-vector dict
            elif isinstance(vp, dict) and len(vp) > 0:
                name, v = next(iter(vp.items()))
                dist = getattr(v, "distance", None)
                details = f"Vector '{name}': size={v.size}" + (f", distance={dist}" if dist else "")
            # Case 3: fallback .config list
            elif hasattr(vp, "config"):
                cfg = vp.config[0] if isinstance(vp.config, (list, tuple)) and vp.config else vp.config
                dist = getattr(cfg, "distance", None)
                size = getattr(cfg, "size", None)
                details = f"Vector size={size}" + (f", distance={dist}" if dist else "")

            if details:
                st.caption(details)
        except Exception as e_vec:
            st.warning(f"Could not parse vector params: {e_vec}")

    except Exception as e:
        st.error("❌ Qdrant connection/collection check failed.")
        st.exception(e)


# -----------------------
# Streamlit App
# -----------------------
def main():
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    st.title(APP_TITLE)

    # --- Env / Qdrant setup
    QDRANT_URL = os.getenv("QDRANT_URL")
    QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
    COLLECTION = os.getenv("QDRANT_COLLECTION", "support_logs_all")

    if not QDRANT_URL or not QDRANT_API_KEY:
        st.error("Please set QDRANT_URL and QDRANT_API_KEY.")
        st.stop()

    # Create Qdrant client (use HTTP; remote cluster)
    client = QdrantClient(
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY,
        prefer_grpc=False,  # keep False for Streamlit Cloud (HTTP works everywhere)
        timeout=60,
    )

    # --- Sidebar Controls
    with st.sidebar:
        st.header("Settings")
        # Brand list is dynamic: fetch distinct brands? Quick way is a free-text + 'All'.
        brand = st.selectbox("Brand filter", ["All"], index=0, help="If your data was uploaded with 'brand' payloads, this filter restricts retrieval.")
        topk = st.slider("Top-K documents", 2, 12, DEFAULT_TOPK, 1)
        temperature = st.slider("Temperature", 0.0, 1.0, DEFAULT_TEMPERATURE, 0.1)
        model = st.selectbox("Claude model", ANTHROPIC_MODELS, index=0)

        st.markdown("---")
        if st.checkbox("Show RAG health panel", value=True):
            qdrant_health_panel(client, COLLECTION)

    # --- LLM
    if not os.getenv("ANTHROPIC_API_KEY"):
        st.error("Please set ANTHROPIC_API_KEY.")
        st.stop()

    llm = ChatAnthropic(model=model, temperature=temperature)

    # --- Retriever
    try:
        retriever = load_retriever(client, collection=COLLECTION, k=topk, brand=brand)
    except Exception as e:
        st.error("Failed to initialize retriever from Qdrant.")
        st.exception(e)
        st.stop()

    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": PROMPT},
        return_source_documents=True,
    )

    # --- UI
    query = st.text_input("Ask a question")
    go = st.button("Ask")

    if go and query.strip():
        with st.spinner("Searching and composing…"):
            # Get candidates first (for visibility)
            try:
                candidates = retriever.get_relevant_documents(query)
            except Exception as e:
                st.error("Error during retrieval.")
                st.exception(e)
                st.stop()

            st.subheader("Evidence")
            if candidates:
                for i, d in enumerate(candidates, 1):
                    m = d.metadata or {}
                    st.markdown(
                        f"- **{i}.** brand=`{m.get('brand','')}`, "
                        f"qa_id=`{m.get('qa_id','')}`, "
                        f"resolved_at=`{m.get('resolved_at','')}`, "
                        f"ticket=`{m.get('ticket_number','')}`"
                    )
            else:
                st.caption("No candidate chunks returned.")

            # Run the QA chain
            try:
                result = chain.invoke({"query": query})
                answer = result["result"]
                docs: List[Any] = result.get("source_documents", [])
            except Exception as e:
                st.error("Generation failed.")
                st.exception(e)
                st.stop()

        st.subheader("Answer")
        st.write(answer)

        # Build citation footer (brand, qa_id, resolved_at, ticket_number)
        if docs:
            st.markdown("**Citations:**")
            lines = []
            for d in docs:
                m = d.metadata or {}
                lines.append(
                    f"- ({m.get('brand','')}, {m.get('qa_id','')}, "
                    f"{m.get('resolved_at','')}, {m.get('ticket_number','')})"
                )
            st.write("\n".join(lines))


if __name__ == "__main__":
    # Minimal guards for local runs
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("Please set OPENAI_API_KEY.")
    if not os.getenv("ANTHROPIC_API_KEY"):
        raise RuntimeError("Please set ANTHROPIC_API_KEY.")
    if not os.getenv("QDRANT_URL") or not os.getenv("QDRANT_API_KEY"):
        raise RuntimeError("Please set QDRANT_URL and QDRANT_API_KEY.")
    main()
