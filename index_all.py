#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build a SINGLE Chroma index from the merged CSV (all brands).
Usage:
  python index_all.py --csv ./all_brands_support_log_embedding_ready.csv --out ./index_all
Env:
  OPENAI_API_KEY
"""
import os
import argparse
from pathlib import Path
import pandas as pd

from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

def build_docs_from_csv(csv_path: Path):
    df = pd.read_csv(csv_path)
    docs = []
    for _, r in df.iterrows():
        q = str(r.get("question", "")).strip()
        a = str(r.get("answer", "")).strip()
        if q == "" and a == "":
            continue
        content = f"Q: {q}\nA: {a}"
        meta = {
            "qa_id": str(r.get("qa_id", "")),
            "brand": str(r.get("brand", "")),
            "ticket_number": str(r.get("ticket_number", "")),
            "resolved_at": str(r.get("resolved_at", "")),
            "source_file": Path(csv_path).name,
        }
        docs.append(Document(page_content=content, metadata=meta))
    return docs

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, required=True, help="Path to merged CSV (all brands)")
    ap.add_argument("--out", type=str, default="./index_all", help="Chroma persist directory")
    ap.add_argument("--chunk_size", type=int, default=600)
    ap.add_argument("--chunk_overlap", type=int, default=80)
    args = ap.parse_args()

    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("Please set OPENAI_API_KEY for embeddings.")

    csv_path = Path(args.csv)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[1/3] Loading CSV…")
    docs = build_docs_from_csv(csv_path)

    print("[2/3] Splitting…")
    splitter = RecursiveCharacterTextSplitter(chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap)
    chunks = splitter.split_documents(docs)
    print(f"  → {len(chunks)} chunks")

    print("[3/3] Writing Chroma index…")
    _ = Chroma.from_documents(
        chunks,
        embedding=OpenAIEmbeddings(model="text-embedding-3-small"),
        persist_directory=str(out_dir)
    )
    print(f"[OK] Single index written at: {out_dir}")

if __name__ == "__main__":
    main()
