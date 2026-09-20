"""ChromaDB vector store with local SentenceTransformer embeddings."""

import json
from pathlib import Path

import chromadb
from chromadb.utils import embedding_functions

# Local embedding model — no API calls, assignment-compliant
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# Paths
BASE_DIR = Path(__file__).parent.parent
CORPUS_PATH = BASE_DIR / "corpus.jsonl"
CHROMA_DB_DIR = BASE_DIR / "chroma_db"
COLLECTION_NAME = "kestrel_corpus"


_cached_ef = None
_cached_client = None
_cached_collection = None


def get_embedding_function():
    """Returns the cached local SentenceTransformer embedding function."""
    global _cached_ef
    if _cached_ef is None:
        _cached_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=EMBEDDING_MODEL_NAME
        )
    return _cached_ef


def get_chroma_client():
    """Returns the cached persistent ChromaDB client."""
    global _cached_client
    if _cached_client is None:
        _cached_client = chromadb.PersistentClient(path=str(CHROMA_DB_DIR))
    return _cached_client


def get_collection():
    """Returns the cached Chroma collection, ingesting corpus if empty."""
    global _cached_collection
    if _cached_collection is not None:
        return _cached_collection

    client = get_chroma_client()
    ef = get_embedding_function()

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=ef,
    )

    if collection.count() == 0:
        print(f"[vectorstore] Collection empty — ingesting {CORPUS_PATH} ...")
        _ingest(collection)
        print(f"[vectorstore] Done. {collection.count()} chunks indexed.")

    _cached_collection = collection
    return _cached_collection


def _ingest(collection):
    """Reads corpus.jsonl and adds every chunk to ChromaDB."""
    if not CORPUS_PATH.exists():
        raise FileNotFoundError(f"corpus.jsonl not found at {CORPUS_PATH}")

    ids, docs, metas = [], [], []

    with open(CORPUS_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            chunk = json.loads(line)

            chunk_id = chunk["chunk_id"]
            title = chunk.get("title", "")
            text = chunk.get("text", "")

            # Prepend title for richer embedding context
            document = f"Title: {title}\n\n{text}"

            meta = {
                "chunk_id": chunk_id,
                "doc_id": chunk.get("doc_id", ""),
                "title": title,
                "category": chunk.get("category", ""),
                "published": chunk.get("published", ""),
                "version": chunk.get("version", ""),
                "owner": chunk.get("owner", ""),
                "source_url": chunk.get("source_url", ""),
            }
            # ChromaDB doesn't like empty-string metadata values
            meta = {k: v for k, v in meta.items() if v}

            ids.append(chunk_id)
            docs.append(document)
            metas.append(meta)

    collection.add(ids=ids, documents=docs, metadatas=metas)


def search_corpus(query: str, k: int = 3):
    """
    Semantic search over the Kestrel corpus.
    Returns a list of dicts with 'text' and 'metadata' keys.
    """
    collection = get_collection()
    results = collection.query(query_texts=[query], n_results=k)

    chunks = []
    if results and results["documents"]:
        for i in range(len(results["documents"][0])):
            chunks.append({
                "text": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
            })
    return chunks


if __name__ == "__main__":
    print("Testing vector store ...")
    col = get_collection()
    print(f"Total chunks: {col.count()}")
    hits = search_corpus("How do Beacons work?", k=2)
    for h in hits:
        print(f"  {h['metadata']['chunk_id']}  —  {h['metadata']['title']}")
