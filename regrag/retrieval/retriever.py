"""Retrieval coordinator supporting BM25, Dense, and Closed-book modes."""

from typing import List, Optional
from regrag.models import LegalChunk, RetrievedResult
from regrag.indexing.bm25 import BM25Index
from regrag.indexing.dense import DenseIndex


class RegulatoryRetriever:
    """Coordinates retrieval over legal chunks using sparse and dense indexes."""

    def __init__(
        self,
        chunks: List[LegalChunk],
        dense_model_name: str = "BAAI/bge-m3",
    ) -> None:
        self.chunks = chunks
        self.bm25_index = BM25Index(chunks)
        self.dense_index = DenseIndex(chunks, model_name=dense_model_name)

    def retrieve(
        self,
        query: str,
        mode: str = "rag_bm25",
        top_k: int = 3,
    ) -> List[RetrievedResult]:
        """Retrieve top_k chunks based on the specified retrieval mode."""
        if mode == "closed_book":
            return []
        elif mode == "rag_bm25":
            return self.bm25_index.search(query, top_k=top_k)
        elif mode == "rag_dense":
            return self.dense_index.search(query, top_k=top_k)
        else:
            raise ValueError(f"Unknown retrieval mode: {mode}")
