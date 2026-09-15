"""Sparse BM25 Index with Vietnamese word segmentation support."""

import re
import sys
from typing import List, Tuple
from regrag.models import LegalChunk, RetrievedResult
from regrag.provenance import degraded, is_degraded

# Optional import of pyvi for Vietnamese word segmentation
try:
    from pyvi import ViTokenizer
    HAS_PYVI = True
except ImportError:
    ViTokenizer = None
    HAS_PYVI = False

try:
    from rank_bm25 import BM25Okapi
    HAS_RANK_BM25 = True
except ImportError:
    BM25Okapi = None
    HAS_RANK_BM25 = False

_DEGRADED_WARNED = False

# Instrument identifiers carry hyphens that pyvi treats as word separators, so
# "18/2024/TT-NHNN" arrives from ViTokenizer as ['18','2024','tt','nhnn'] and the
# citation can never be retrieved. They are masked out before segmentation and
# re-inserted afterwards with their hyphens intact.
_INSTRUMENT_ID_RE = re.compile(
    r"\b\d{1,3}\s*/\s*\d{4}\s*/\s*[A-ZĐ]{1,12}(?:-[A-ZĐ0-9]{1,12})*", re.IGNORECASE
)
_BARE_AGENCY_RE = re.compile(r"\b[A-ZĐ]{1,8}(?:-[A-ZĐ0-9]{1,12})+\b", re.IGNORECASE)


def _clean(token: str) -> str:
    return re.sub(r"[^\w\s\-]", " ", token.lower()).strip()


def tokenize_vietnamese(text: str) -> List[str]:
    """Tokenize Vietnamese text with compound word handling."""
    masked: List[str] = []

    def _stash(match: "re.Match[str]") -> str:
        masked.append(match.group(0))
        return f" zzid{len(masked) - 1}zz "

    protected = _INSTRUMENT_ID_RE.sub(_stash, text or "")
    protected = _BARE_AGENCY_RE.sub(_stash, protected)

    if HAS_PYVI and ViTokenizer is not None:
        protected = ViTokenizer.tokenize(protected)

    text_clean = re.sub(r"[^\w\s\-]", " ", protected.lower())
    for i, original in enumerate(masked):
        text_clean = text_clean.replace(f"zzid{i}zz", _clean(original))
    return [t for t in text_clean.split() if t.strip("-")]


class BM25Index:
    """BM25 index wrapper for legal chunks."""

    def __init__(self, chunks: List[LegalChunk]) -> None:
        global _DEGRADED_WARNED
        self.chunks = chunks

        if HAS_RANK_BM25 and HAS_PYVI:
            self.backend = "bm25-rank_bm25+pyvi"
        else:
            reasons = []
            if not HAS_RANK_BM25:
                reasons.append("rank_bm25 missing")
            if not HAS_PYVI:
                reasons.append("pyvi missing")
            self.backend = degraded(", ".join(reasons))

        if is_degraded(self.backend) and not _DEGRADED_WARNED:
            _DEGRADED_WARNED = True
            sys.stderr.write(
                f"[WARNING] BM25Index constructed with degraded backend ({self.backend}). "
                "Results cannot be used for paper reporting.\n"
            )
            sys.stderr.flush()

        self.corpus_tokens = [
            tokenize_vietnamese(chunk.formatted_context()) for chunk in chunks
        ]
        if HAS_RANK_BM25 and BM25Okapi is not None and self.corpus_tokens:
            self._bm25 = BM25Okapi(self.corpus_tokens)
        else:
            self._bm25 = None

    def search(self, query: str, top_k: int = 3) -> List[RetrievedResult]:
        """Retrieve top_k chunks matching the query using BM25."""
        if not self.chunks:
            return []

        query_tokens = tokenize_vietnamese(query)
        if self._bm25:
            scores = self._bm25.get_scores(query_tokens)
            ranked_indices = sorted(
                range(len(scores)), key=lambda i: scores[i], reverse=True
            )[:top_k]
            return [
                RetrievedResult(
                    chunk=self.chunks[idx],
                    score=float(scores[idx]),
                    rank=rank + 1,
                    retriever_backend=self.backend,
                )
                for rank, idx in enumerate(ranked_indices)
            ]

        # Simple term overlap fallback if rank_bm25 is not installed
        q_set = set(query_tokens)
        scored = []
        for idx, tokens in enumerate(self.corpus_tokens):
            overlap = len(q_set.intersection(set(tokens)))
            scored.append((idx, float(overlap)))
        scored.sort(key=lambda x: x[1], reverse=True)
        return [
            RetrievedResult(
                chunk=self.chunks[idx],
                score=score,
                rank=rank + 1,
                retriever_backend=self.backend,
            )
            for rank, (idx, score) in enumerate(scored[:top_k])
        ]
