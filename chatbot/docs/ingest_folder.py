# -*- coding: utf-8 -*-
import os, argparse, hashlib, time
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_community.vectorstores.pgvector import PGVector
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

def make_store():
    load_dotenv()
    conn = os.getenv("CONNECTION_STRING", "postgresql+psycopg2://admin:admin@postgres:5433/vectordb")
    coll = os.getenv("COLLECTION_NAME", "vectordb")
    embed_model = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")
    embeddings = OpenAIEmbeddings(model=embed_model)
    store = PGVector(collection_name=coll, connection_string=conn, embedding_function=embeddings)
    return store, coll

def sha1_id(source: str, content: str) -> str:
    return hashlib.sha1(f"{source}\n{content}".encode("utf-8")).hexdigest()

def main(path: str, pattern: str, chunk_size: int, chunk_overlap: int):
    store, coll = make_store()

    loader = DirectoryLoader(
        path,
        glob=pattern,
        loader_cls=TextLoader,
        loader_kwargs={"autodetect_encoding": True},
        show_progress=True,
        use_multithreading=True,
    )
    docs = loader.load()
    now = int(time.time())
    for d in docs:
        d.metadata["source"] = d.metadata.get("source") or d.metadata.get("file_path") or path
        d.metadata["indexed_at"] = now

    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.split_documents(docs)

    ids = [sha1_id(c.metadata.get("source",""), c.page_content) for c in chunks]
    store.add_documents(chunks, ids=ids)  # 동일 ID면 중복 방지(버전에 따라 upsert/skip)

    print(f"✅ Done: files={len(docs)} → chunks={len(chunks)} → collection='{coll}'")

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Ingest .txt files in a folder into PGVector")
    p.add_argument("--path", required=True)
    p.add_argument("--pattern", default="**/*.txt")
    p.add_argument("--chunk-size", type=int, default=800)
    p.add_argument("--chunk-overlap", type=int, default=120)
    args = p.parse_args()
    main(args.path, args.pattern, args.chunk_size, args.chunk_overlap)
