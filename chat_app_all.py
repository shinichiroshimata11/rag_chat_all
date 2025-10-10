# chat_app_all.py — TEMP HEALTH CHECK (safe to deploy)
import os, traceback, textwrap
import streamlit as st

st.set_page_config(page_title="RAG Health Check", layout="centered")
st.title("🔧 RAG Health Check")

def mask(s, keep=4):
    if not s: return "⟂ (missing)"
    if len(s) <= keep * 2: return s[0:2] + "…" + s[-2:]
    return s[:keep] + "…" + s[-keep:]

# --- 1) Read env/secrets
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "support_logs_all")

st.subheader("Environment")
st.code(
    f"""OPENAI_API_KEY      = {mask(OPENAI_API_KEY)}
ANTHROPIC_API_KEY   = {mask(ANTHROPIC_API_KEY)}
QDRANT_URL          = {QDRANT_URL or "⟂ (missing)"} 
QDRANT_API_KEY      = {mask(QDRANT_API_KEY)}
QDRANT_COLLECTION   = {QDRANT_COLLECTION}""",
    language="txt",
)

missing = []
if not QDRANT_URL: missing.append("QDRANT_URL")
if not QDRANT_API_KEY: missing.append("QDRANT_API_KEY")
if not QDRANT_COLLECTION: missing.append("QDRANT_COLLECTION")
if not OPENAI_API_KEY: missing.append("OPENAI_API_KEY (needed to create embeddings)")

if missing:
    st.error("Missing environment variables: " + ", ".join(missing))
    st.stop()

# --- 2) Qdrant connectivity + collection check
st.subheader("Qdrant Connectivity")
try:
    from qdrant_client import QdrantClient
    client = QdrantClient(
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY,
        timeout=10,  # be explicit so it won't hang forever
    )

    info = client.get_collection(QDRANT_COLLECTION)
    st.success(f"Connected to Qdrant. Collection **{QDRANT_COLLECTION}** exists.")
    st.write(f"Vectors size: {info.vectors_count}, distance: {info.config.params.distance}")

except Exception as e:
    st.error("❌ Qdrant connection/collection check failed.")
    st.code("".join(traceback.format_exception_only(type(e), e)))
    st.stop()

# --- 3) Tiny embed + search to verify end-to-end
st.subheader("Embedding + Vector Search Smoke Test")
try:
    # use OpenAI embeddings to create a 1536-dim vector
    from openai import OpenAI
    oai = OpenAI(api_key=OPENAI_API_KEY)
    test_text = "hello from health check"
    emb = oai.embeddings.create(model="text-embedding-3-small", input=[test_text])
    vec = emb.data[0].embedding  # 1536 floats

    # If the collection was created with 1536 dim, search should succeed.
    hits = client.search(
        collection_name=QDRANT_COLLECTION,
        query_vector=vec,
        limit=3,
    )

    st.success("End-to-end test OK: embedding created and search returned results.")
    if not hits:
        st.info("No hits returned (collection may be empty or not yet uploaded).")
    else:
        for i, h in enumerate(hits, 1):
            st.write(f"#{i} score={h.score:.4f} payload_keys={list((h.payload or {}).keys())}")

except Exception as e:
    st.error("❌ Embedding + search test failed.")
    st.code("".join(traceback.format_exception_only(type(e), e)))
    st.info("Common causes:\n"
            "- Collection vector size isn’t 1536 (but your embeddings are 1536)\n"
            "- Wrong collection name\n"
            "- QDRANT_URL missing :6333 for the REST port\n"
            "- OPENAI_API_KEY missing or invalid")
