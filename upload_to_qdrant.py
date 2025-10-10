#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Upload CSV rows to Qdrant.
- Embeds the `answer` column with OpenAI (text-embedding-3-small, 1536 dims)
- Stores the main text under payload key "page_content" (LangChain default)
- Keeps brand/qa_id/resolved_at/ticket_number/question as flat metadata
"""

import os
import uuid
import argparse
import pandas as pd
from tqdm import tqdm
from qdrant_client import QdrantClient
from qdrant_client.http import models as qm
from openai import OpenAI

EMBED_MODEL = "text-embedding-3-small"
EMBED_DIM = 1536


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--collection", required=True)
    ap.add_argument("--recreate", action="store_true")
    ap.add_argument("--batch", type=int, default=128)
    args = ap.parse_args()

    qdrant_url = os.environ["QDRANT_URL"]
    qdrant_key = os.environ["QDRANT_API_KEY"]
    oai_key = os.environ["OPENAI_API_KEY"]

    # 1) Load & clean
    df = pd.read_csv(args.csv)
    # ensure expected columns exist
    expected = ["qa_id", "brand", "question", "answer", "ticket_number", "resolved_at"]
    for col in expected:
        if col not in df.columns:
            df[col] = ""

    # keep only non-empty answer rows
    df = df.dropna(subset=["answer"])
    df = df[df["answer"].astype(str).str.strip() != ""]
    # fill NAs
    for col in expected:
        df[col] = df[col].fillna("").astype(str)

    # 2) Qdrant collection (create or recreate)
    client = QdrantClient(url=qdrant_url, api_key=qdrant_key)
    if args.recreate:
        client.recreate_collection(
            collection_name=args.collection,
            vectors_config=qm.VectorParams(size=EMBED_DIM, distance=qm.Distance.COSINE),
        )
    else:
        try:
            client.get_collection(args.collection)
        except Exception:
            client.recreate_collection(
                collection_name=args.collection,
                vectors_config=qm.VectorParams(size=EMBED_DIM, distance=qm.Distance.COSINE),
            )

    # 3) Embed & upsert
    oai = OpenAI(api_key=oai_key)
    ids = [str(uuid.uuid4()) for _ in range(len(df))]
    print(f"Uploading {len(df)} rows to '{args.collection}' ...")

    for i in tqdm(range(0, len(df), args.batch), desc="Upserting"):
        chunk = df.iloc[i : i + args.batch]
        texts = chunk["answer"].astype(str).tolist()

        emb = oai.embeddings.create(model=EMBED_MODEL, input=texts)
        vectors = [e.embedding for e in emb.data]

        payloads = []
        for (_, r), text in zip(chunk.iterrows(), texts):
            payloads.append(
                {
                    "page_content": text,  # <- LangChain reads this
                    "brand": r["brand"],
                    "qa_id": r["qa_id"],
                    "resolved_at": r["resolved_at"],
                    "ticket_number": r["ticket_number"],
                    "question": r["question"],
                }
            )

        client.upsert(
            collection_name=args.collection,
            points=[
                qm.PointStruct(id=pid, vector=vec, payload=pl)
                for pid, vec, pl in zip(ids[i : i + args.batch], vectors, payloads)
            ],
        )

    info = client.get_collection(args.collection)
    print("Done. Collection status:", info.status)


if __name__ == "__main__":
    main()
