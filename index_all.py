#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Build Qdrant index from your merged CSV.

Env vars:
  CSV_PATH                (default: ./all_brands_support_log_embedding_ready.csv)
  QDRANT_URL              (required, e.g. https://YOUR-CLUSTER.cloud.qdrant.io)
  QDRANT_API_KEY          (required)
  QDRANT_COLLECTION       (default: support_logs_all)
  OPENAI_API_KEY          (required for embeddings)

Run locally:
  python index_all.py
"""

import os
import math
import uuid
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any

from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, Distance, PointStruct
from langchain_openai import OpenAIEmbeddings

CSV_PATH = Path(os.getenv("CSV_PATH", "./all_brands_support_log_embedding_ready.csv"))
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "support_logs_all")

EMBED_MODEL = "text-embedding-3-small"  # 1536-dim

def load_rows(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    df = pd.read_csv(csv_path)
    # Normalize expected columns
    for col in ["question", "answer", "brand", "qa_id", "resolved_at", "ticket_number"]:
        if col not in df.columns:
            df[col] = ""
    # Derive a 'text' field if missing
    if "text" not in df.columns:
        df["text"] = df.apply(
            lambda r: f"Brand: {str(r['brand']).strip()}\nQA_ID: {str(r['qa_id']).strip()}\nResolved: {str(r['resolved_at']).strip()}\nTicket: {str(r['ticket_number']).strip()}\n\nQ: {str(r['question']).strip()}\nA: {str(r['answer']).strip()}",
            axis=1,
        )
    # Drop empty texts
    df["text"] = df["text"].astype(str).apply(lambda s: s.strip())
    df = df[df["text"] != ""].copy()
    df.reset_index(drop=True, inplace=True)
    return df

def main():
    if not QDRANT_URL or not QDRANT_API_KEY:
        raise RuntimeError("Please set QDRANT_URL and QDRANT_API_KEY for Qdrant.")

    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("Please set OPENAI_API_KEY for embeddings.")

    print(f"Loading: {CSV_PATH}")
    df = load_rows(CSV_PATH)
    print(f"Rows: {len(df)}")

    # Create client and collection
    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

    print(f"Preparing collection: {QDRANT_COLLECTION}")
    # Recreate or create-if-missing
    try:
        client.get_collection(QDRANT_COLLECTION)
        # If exists, we keep it and upsert (safe)
        print("Collection exists. Upserting new points…")
    except Exception:
        print("Collection not found. Creating…")
        client.recreate_collection(
            collection_name=QDRANT_COLLECTION,
            vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
        )

    embeddings = OpenAIEmbeddings(model=EMBED_MODEL)

    texts: List[str] = df["text"].tolist()
    print("Computing embeddings… (batching)")
    # Batch to avoid timeouts
    batch_size = 256
    num_batches = math.ceil(len(texts) / batch_size)
    inserted = 0

    for b in range(num_batches):
        s = b * batch_size
        e = min((b + 1) * batch_size, len(texts))
        batch_texts = texts[s:e]

        vecs = embeddings.embed_documents(batch_texts)

        points = []
        for i, v in enumerate(vecs):
            row = df.iloc[s + i]
            payload: Dict[str, Any] = {
                "text": row["text"],
                "brand": str(row["brand"]).strip(),
                "qa_id": str(row["qa_id"]).strip(),
                "resolved_at": str(row["resolved_at"]).strip(),
                "ticket_number": str(row["ticket_number"]).strip(),
                "question": str(row["question"]).strip(),
                "answer": str(row["answer"]).strip(),
            }
            points.append(
                PointStruct(
                    id=str(uuid.uuid4()),
                    vector=v,
                    payload=payload,
                )
            )

        client.upsert(collection_name=QDRANT_COLLECTION, points=points)
        inserted += len(points)
        print(f"Upserted {inserted}/{len(texts)}")

    print("Done.")

if __name__ == "__main__":
    main()
