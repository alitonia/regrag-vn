"""Dense vector index for legal chunks using multilingual embeddings."""

from typing import List, Optional
from regrag.models import LegalChunk, RetrievedResult
from regrag.provenance import CORPUS_UNSET, ProvenanceError, require

try:
    import torch
except ImportError:
    torch = None

try:
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    SentenceTransformer = None
    HAS_SENTENCE_TRANSFORMERS = False


class DenseIndex:
    """Dense embedding index supporting BGE-M3 and multilingual models."""

    def __init__(
        self,
        chunks: List[LegalChunk],
        model_name: str = "BAAI/bge-m3",
        device: Optional[str] = None,
    ) -> None:
        self.chunks = chunks
        self.model_name = model_name
        self.device = device or (
            "cuda"
            if HAS_SENTENCE_TRANSFORMERS and torch is not None and torch.cuda.is_available()
            else "cpu"
        )
        self._model = None
        self._embeddings = None
        self.backend: str = CORPUS_UNSET

    def build(self) -> None:
        """Encode all chunks into dense embeddings."""
        require(
            HAS_SENTENCE_TRANSFORMERS,
            "DenseIndex requires 'sentence_transformers' to be installed (pip install sentence-transformers).",
        )
        if not self.chunks:
            self._model = None
            self._embeddings = None
            self.backend = CORPUS_UNSET
            return

        self._model = SentenceTransformer(self.model_name, device=self.device)
        texts = [chunk.formatted_context() for chunk in self.chunks]
        self._embeddings = self._model.encode(
            texts,
            convert_to_tensor=True,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        self.backend = self.model_name
        self._release_gpu()

    def _release_gpu(self) -> None:
        """Return the GPU to the generation phase after index build.

        The 2026-09-12 campaign OOMed at row ~171 with BGE-M3 resident
        (~2.3 GiB fp32) beside a 4-bit 7B and its attention transients.
        Retrieval needs the encoder for one query at a time, which the CPU
        serves in ~100 ms - orders of magnitude below a generation step."""
        if self._model is not None:
            try:
                self._model.to("cpu")
            except Exception:
                pass
        emb = getattr(self, "_embeddings", None)
        if emb is not None and hasattr(emb, "is_cuda") and emb.is_cuda:
            self._embeddings = emb.cpu()
        try:
            if torch is not None and torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    def search(self, query: str, top_k: int = 3) -> List[RetrievedResult]:
        """Retrieve top_k chunks by cosine similarity."""
        require(
            HAS_SENTENCE_TRANSFORMERS,
            "DenseIndex requires 'sentence_transformers' to be installed (pip install sentence-transformers).",
        )
        require(
            self._model is not None and self._embeddings is not None,
            "DenseIndex has not been built or contains no chunks. Call build() on non-empty chunks before search().",
        )

        import torch
        query_emb = self._model.encode(
            query, convert_to_tensor=True, normalize_embeddings=True
        )
        scores = torch.matmul(self._embeddings, query_emb).cpu().tolist()
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
