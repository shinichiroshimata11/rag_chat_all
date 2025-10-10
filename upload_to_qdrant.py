import os
import pandas as pd
from qdrant_client import QdrantClient, models
from openai import OpenAI

QDRANT_URL = os.environ["QDRANT_URL"]
QDRANT_API_KEY = os.environ["QDRANT_API_KEY"]
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]

CSV_PATH = "all_brands_support_log_embedding_ready.csv"
COLLECTION = os.getenv("QDRANT_COLLECTION", "support_logs_all")
EMBED_MODEL = "text-embedding-3-small"
BATCH = 64  # conservative

# ---- Load & prepare data
df = pd.read_csv(CSV_PATH)

# ensure str and handle NaNs
for col in ["brand", "question", "answer", "qa_id", "ticket_number", "resolved_at"]:
    if col in df.columns:
        df[col] = df[col].astype(str).fillna("")

# build the text field
if "text" not in df.columns:
    df["text"] = (
        "[" + df["brand"].fillna("") + "] "
        + "Q: " + df["question"].fillna("") + "\nA: " + df["answer"].fillna("")
    )

# drop rows where text is empty/whitespace
df["text"] = df["text"].astype(str)
df = df[df["text"].str.strip() != ""].reset_index(drop=True)

# ---- Clients
qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, prefer_grpc=False)
oai = OpenAI(api_key=OPENAI_API_KEY)

# ---- Ensure collection exists
if not qdrant.collection_exists(COLLECTION):
    qdrant.recreate_collection(
        collection_name=COLLECTION,
        vectors_config=models.VectorParams(size=1536, distance=models.Distance.COSINE),
    )
else:
    print(f"Collection '{COLLECTION}' already exists.")

# ---- Start id from current count to avoid collisions
try:
    next_id = qdrant.count(COLLECTION, exact=True).count
except Exception:
    next_id = 0

total = len(df)
print(f"🚀 Uploading {total} rows to '{COLLECTION}' at {QDRANT_URL}")

offset = 0
while offset < total:
    # slice batch
    batch = df.iloc[offset : offset + BATCH]

    # filter empties in the batch (again, just in case)
    batch = batch[batch["text"].str.strip() != ""]
    if batch.empty:
        offset += BATCH
        continue

    texts = batch["text"].tolist()

    # create embeddings
    emb_resp = oai.embeddings.create(model=EMBED_MODEL, input=texts)
    vectors = [d.embedding for d in emb_resp.data]

    # If the API returned fewer vectors than inputs (rare, but protect against it)
    if len(vectors) != len(texts):
        print(
            f"⚠️  embeddings mismatch: got {len(vectors)} for {len(texts)} inputs; "
            "skipping this batch."
        )
        offset += BATCH
        continue

    # Build points aligned with *filtered* batch
    points = []
    for row, vec in zip(batch.itertuples(index=False), vectors):
        payload = {
            "text": row.text,
            "brand": getattr(row, "brand", ""),
            "qa_id": getattr(row, "qa_id", ""),
            "ticket_number": getattr(row, "ticket_number", ""),
            "resolved_at": getattr(row, "resolved_at", ""),
            "question": getattr(row, "question", ""),
            "answer": getattr(row, "answer", ""),
        }
        points.append(
            models.PointStruct(
                id=int(next_id),
                vector=vec,
                payload=payload,
            )
        )
        next_id += 1

    # Upsert
    qdrant.upsert(collection_name=COLLECTION, points=points)

    offset += BATCH
    print(f"  ✓ {min(offset, total)}/{total}")

count = qdrant.count(COLLECTION, exact=True).count
print(f"✅ Done. Points in collection: {count}")
