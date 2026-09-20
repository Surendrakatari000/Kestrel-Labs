"""Citation normalization and extraction utilities."""

import json
import re
from pathlib import Path
from typing import List, Dict, Optional

# Lazy-loaded cache for corpus chunk_id -> metadata
_CORPUS_MAP: Optional[Dict[str, dict]] = None

def _get_corpus_map() -> Dict[str, dict]:
    """Loads and caches chunk_id -> metadata from corpus.jsonl."""
    global _CORPUS_MAP
    if _CORPUS_MAP is None:
        _CORPUS_MAP = {}
        # Locate corpus.jsonl at repository root
        corpus_path = Path(__file__).resolve().parent.parent.parent / "corpus.jsonl"
        if corpus_path.exists():
            try:
                with open(corpus_path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            data = json.loads(line)
                            cid = data.get("chunk_id")
                            if cid:
                                _CORPUS_MAP[cid] = data
            except Exception:
                pass
    return _CORPUS_MAP


def normalize_citations(text: str, chunks: Optional[List[dict]] = None) -> str:
    """
    Ensures every cited chunk_id in the text consistently includes its document title
    in the standard format: [chunk_id: title].
    
    Handles:
    - [chunk_id] -> [chunk_id: title]
    - 【chunk_id】 -> [chunk_id: title]
    - [chunk_id: existing_or_partial_title] -> [chunk_id: canonical_title]
    - 【chunk_id: ...】 -> [chunk_id: canonical_title]
    """
    if not text:
        return text

    # Build title lookup from retrieved chunks
    title_lookup = {}
    if chunks:
        for c in chunks:
            m = c.get("metadata", {})
            cid = m.get("chunk_id")
            title = m.get("title")
            if cid and title:
                title_lookup[cid] = title

    # Fallback to corpus.jsonl for any chunk_id not in retrieved_chunks
    corpus_map = _get_corpus_map()

    def get_title(chunk_id: str) -> str:
        if chunk_id in title_lookup:
            return title_lookup[chunk_id]
        if chunk_id in corpus_map:
            return corpus_map[chunk_id].get("title", "")
        return ""

    # Pattern matches bracketed citations starting with chunk_id pattern (e.g. word-word:num)
    # [chunk_id ...] or 【chunk_id ...】
    citation_pattern = re.compile(r"[\[\【]\s*([a-zA-Z0-9\-]+:[0-9]+)(?:[^\]\】]*)[\]\】]")

    def replace_citation(match: re.Match) -> str:
        cid = match.group(1)
        canonical_title = get_title(cid)
        if canonical_title:
            return f"[{cid}: {canonical_title}]"
        # If title cannot be found, preserve standard [chunk_id] format
        return f"[{cid}]"

    normalized = citation_pattern.sub(replace_citation, text)
    return normalized


def extract_cited_sources(text: str, chunks: Optional[List[dict]] = None) -> List[dict]:
    """
    Extracts ordered list of unique cited sources with chunk_id, title, and metadata.
    """
    if not text:
        return []

    # Find all chunk_ids in order of appearance
    pattern = re.compile(r"[\[\【]\s*([a-zA-Z0-9\-]+:[0-9]+)")
    matches = pattern.findall(text)
    
    seen = set()
    ordered_ids = []
    for cid in matches:
        if cid not in seen:
            seen.add(cid)
            ordered_ids.append(cid)

    # Lookup details
    title_lookup = {}
    if chunks:
        for c in chunks:
            m = c.get("metadata", {})
            cid = m.get("chunk_id")
            if cid:
                title_lookup[cid] = m

    corpus_map = _get_corpus_map()

    sources = []
    for cid in ordered_ids:
        meta = title_lookup.get(cid) or corpus_map.get(cid) or {}
        sources.append({
            "chunk_id": cid,
            "title": meta.get("title", cid),
            "category": meta.get("category", ""),
            "version": meta.get("version", ""),
            "published": meta.get("published", ""),
        })

    return sources
