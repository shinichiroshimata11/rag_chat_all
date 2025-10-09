import os
import pandas as pd
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Qdrant

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "support_logs_all")
CSV_PATH = os.getenv("CSV_PATH", "./all_brands_support_log_embedding_ready.csv")

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

def get_qdrant():
    return QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

def ensure_collection(client: QdrantClient, vector_size: int):
    existing = [c.name for c in client.get_collections().collections]
    if QDRANT_COLLECTION not in existing:
        client.create_collection(
            collection_name=QDRANT_COLLECTION,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )

def build_index_if_empty():
    client = get_qdrant()
    ensure_collection(client, vector_size=1536)  # embedding-3-small dim
    count = client.count(QDRANT_COLLECTION).count
    if count > 0:
        return  # already indexed

    df = pd.read_csv(CSV_PATH)
    # make the document text (adjust to your CSV schema)
    df["text"] = df.apply(
        lambda r: f"[{r.get('brand','')}] {r.get('question','')} {r.get('answer','')}",
        axis=1,
    )
    metadatas = df[["qa_id","brand","resolved_at","ticket_number"]].to_dict("records")
    Qdrant.from_texts(
        texts=df["text"].tolist(),
        embedding=embeddings,
        metadatas=metadatas,
        url=QDRANT_URL,
        prefer_grpc=False,
        api_key=QDRANT_API_KEY,
        collection_name=QDRANT_COLLECTION,
    )

def get_retriever(k=5, brand_filter: str | None = None):
    client = get_qdrant()
    vs = Qdrant(
        client=client,
        collection_name=QDRANT_COLLECTION,
        embeddings=embeddings,
    )
    if brand_filter and brand_filter != "All":
        return vs.as_retriever(search_kwargs={"k": k, "filter": {"must": [{"key":"brand","match":{"value": brand_filter}}]}})
    return vs.as_retriever(search_kwargs={"k": k})