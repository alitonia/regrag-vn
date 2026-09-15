"""Corpus loading and hashing for the generation harness.

There is no chunk-dict -> ``LegalChunk`` loader anywhere else in the repo: the
writer side (``scripts/build_tier1_corpus.py``, ``scripts/build_corpus_chunks.py``)
serialises ``asdict(chunk)``, and ``scripts/regenerate.py`` reads the JSON as raw
dicts. This module supplies the reader side the campaign needs.

It lives in ``regrag/generation/`` rather than ``regrag/corpus/`` purely because
``regrag/corpus/`` is outside this harness's write scope; it has no generation
dependencies and should be relocated there once that is possible.

Loud-failure contract (docs/PROJECT_PLAN.md §5)
----------------------------------------------
* missing corpus file              -> FileNotFoundError
* corpus file with 0 chunks        -> ProvenanceError, naming the ingest script
* chunks with mixed corpus_source  -> ProvenanceError (a campaign must be bound
                                      to exactly one corpus tier, otherwise the
                                      provenance tag on the rows is a lie)
* chunk dicts missing required keys-> ProvenanceError naming the chunk_id
* corpus_source UNSET on any chunk -> ProvenanceError

An empty corpus is the live state of ``data/processed_chunks/corpus_chunks.json``
as of 2026-09-11, so this is not a hypothetical guard: without it the RAG modes
would retrieve nothing, the model would answer from parametric memory, and the
result would be indistinguishable from the closed-book column.
"""

from __future__ import annotations

import hashlib
import json
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

from regrag.corpus.canonical import normalize_ws
from regrag.models import LegalChunk
from regrag.provenance import CORPUS_UNSET, ProvenanceError

_CHUNK_REQUIRED_KEYS = (
    "chunk_id",
    "doc_id",
    "article_id",
    "text",
    "corpus_source",
)


def file_sha256(path: str) -> str:
    """Full-file sha256 hex digest. Used for the CSV hash, as regenerate.py does."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def chunk_from_dict(raw: Dict[str, Any]) -> LegalChunk:
    """Rebuild a LegalChunk from its asdict() serialisation."""
    missing = [k for k in _CHUNK_REQUIRED_KEYS if k not in raw]
    if missing:
        raise ProvenanceError(
            f"Chunk {raw.get('chunk_id', '<no chunk_id>')!r} is missing required "
            f"key(s) {missing}. Refusing to build a LegalChunk from a partial "
            "record - a chunk without text or corpus_source would silently "
            "degrade retrieval."
        )
    if not (raw.get("corpus_source") or "").strip() or raw["corpus_source"] == CORPUS_UNSET:
        raise ProvenanceError(
            f"Chunk {raw['chunk_id']!r} has corpus_source={raw.get('corpus_source')!r}. "
            "Every chunk must carry a real corpus tag before it can be retrieved "
            "over in a publishable campaign."
        )
    if not (raw.get("text") or "").strip():
        raise ProvenanceError(
            f"Chunk {raw['chunk_id']!r} has empty text. An empty chunk can still "
            "be returned by a retriever and would blank out the RAG context."
        )

    known = {f for f in LegalChunk.__dataclass_fields__}  # type: ignore[attr-defined]
    kwargs = {k: v for k, v in raw.items() if k in known}
    kwargs.setdefault("doc_title", "")
    kwargs.setdefault("chapter", None)
    kwargs.setdefault("article_title", "")
    kwargs.setdefault("clause_id", None)
    kwargs.setdefault("metadata", {})
    return LegalChunk(**kwargs)


def load_chunks(path: str) -> List[LegalChunk]:
    """Load and validate a processed-chunks JSON file.

    Returns chunks in file order (retrieval re-ranks them; BM25's fallback path
    is order-sensitive, so preserving a deterministic order matters).
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Corpus file not found: {path}. Build it with "
            "scripts/build_corpus_chunks.py (Tier 2, publishable) or "
            "scripts/build_tier1_corpus.py (Tier 1, development fixture only)."
        )
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    if not isinstance(raw, list):
        raise ProvenanceError(
            f"{path} must contain a JSON list of chunk objects, got {type(raw).__name__}."
        )
    if not raw:
        raise ProvenanceError(
            f"Corpus file {path} contains 0 chunks. The RAG modes cannot run "
            "against an empty corpus: retrieval would return nothing, the model "
            "would answer from parametric memory, and the RAG column would be "
            "indistinguishable from closed-book with no warning. Ingest the "
            "source documents first (scripts/build_corpus_chunks.py)."
        )

    chunks = [chunk_from_dict(r) for r in raw]

    sources = sorted({c.corpus_source for c in chunks})
    if len(sources) > 1:
        raise ProvenanceError(
            f"Corpus file {path} mixes corpus_source tags {sources}. A campaign "
            "is bound to exactly one corpus tier so that the tag stamped on every "
            "row is true. Split the file or re-ingest."
        )

    ids = [c.chunk_id for c in chunks]
    if len(set(ids)) != len(ids):
        dupes = sorted({i for i in ids if ids.count(i) > 1})[:10]
        raise ProvenanceError(
            f"Corpus file {path} has duplicate chunk_id(s): {dupes}. Duplicate "
            "ids make retrieved citations ambiguous."
        )
    return chunks


def corpus_source_of(chunks: Sequence[LegalChunk]) -> str:
    """The single corpus_source tag shared by every chunk."""
    if not chunks:
        raise ProvenanceError("Cannot derive corpus_source from an empty chunk list.")
    sources = {c.corpus_source for c in chunks}
    if len(sources) != 1:
        raise ProvenanceError(f"Mixed corpus_source tags in one corpus: {sorted(sources)}")
    return next(iter(sources))


def corpus_hash(chunks: Sequence[LegalChunk]) -> str:
    """Content hash of the corpus, stable across dict key order and whitespace.

    Binds cache keys to the corpus that was actually retrieved over, so that
    re-ingesting Tier 2 invalidates cached RAG generations instead of silently
    reusing answers that were conditioned on different context.

    Uses normalize_ws (NFC + whitespace collapse, diacritics preserved) so that
    cosmetic reformatting of an ingest does not invalidate the whole campaign,
    while any real text change does.
    """
    h = hashlib.sha256()
    for c in sorted(chunks, key=lambda x: x.chunk_id):
        payload = "\x1f".join(
            [
                c.chunk_id,
                c.doc_id or "",
                c.article_id or "",
                c.clause_id or "",
                c.corpus_source or "",
                normalize_ws(c.text or ""),
            ]
        )
        h.update(payload.encode("utf-8"))
        h.update(b"\x1e")
    return h.hexdigest()


def describe_corpus(chunks: Sequence[LegalChunk], path: str) -> Dict[str, Any]:
    """Manifest-ready description of the corpus a campaign is bound to."""
    from regrag.provenance import publishable_corpus

    src = corpus_source_of(chunks)
    by_doc: Dict[str, int] = {}
    for c in chunks:
        by_doc[c.doc_id] = by_doc.get(c.doc_id, 0) + 1
    return {
        "corpus_path": path,
        "corpus_source": src,
        "corpus_hash": corpus_hash(chunks),
        "chunk_count": len(chunks),
        "distinct_doc_ids": len(by_doc),
        "publishable_corpus": publishable_corpus(src),
    }
