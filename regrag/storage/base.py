"""Abstract repository interface for persisting benchmark results.

Fulfills the course requirement of having a unified business logic interface
supported by multiple swappable persistence layers.
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any, Tuple
from regrag.models import EvaluationRecord, GenerationResult
from regrag.provenance import CORPUS_UNSET, is_degraded, publishable_corpus


class BenchmarkResultRepository(ABC):
    """Abstract storage interface for RegRAG-VN benchmark results."""

    @abstractmethod
    def save_evaluation(self, record: EvaluationRecord) -> None:
        """Persist an evaluation record."""
        pass

    @abstractmethod
    def get_evaluation(
        self, question_id: str, model_name: str, retrieval_mode: str
    ) -> Optional[EvaluationRecord]:
        """Retrieve an evaluation record by composite key."""
        pass

    @abstractmethod
    def list_evaluations(
        self,
        model_name: Optional[str] = None,
        retrieval_mode: Optional[str] = None,
    ) -> List[EvaluationRecord]:
        """List evaluations matching the optional filter criteria."""
        pass

    @abstractmethod
    def save_generation(self, result: GenerationResult) -> None:
        """Persist a raw generation result."""
        pass

    @abstractmethod
    def list_generations(
        self,
        model_name: Optional[str] = None,
        retrieval_mode: Optional[str] = None,
    ) -> List[GenerationResult]:
        """List raw generations matching the optional filter criteria."""
        pass

    @abstractmethod
    def export_summary(self) -> Dict[str, Any]:
        """Compute and export high-level metric summaries across all conditions."""
        pass


def publishable_evaluations(
    records: List[EvaluationRecord],
) -> Tuple[List[EvaluationRecord], List[EvaluationRecord]]:
    """Split evaluation records into publishable and refused.

    export_summary() used to average every stored row regardless of corpus tier,
    retriever backend or scorer validation status - the most direct path by which an
    unpublishable number reaches a paper table, since scripts/run_eval.py prints
    exactly this. A row is publishable only when its corpus is Tier 2, its retriever
    backend is not DEGRADED/UNSET, and its scorer has been validated (metric_status
    == "final"). Refused rows are returned so callers can report the count; they must
    never be dropped silently.
    """
    kept: List[EvaluationRecord] = []
    refused: List[EvaluationRecord] = []
    for r in records:
        backend = getattr(r, "retriever_backend", CORPUS_UNSET)
        corpus = getattr(r, "corpus_source", CORPUS_UNSET)
        status = getattr(r, "metric_status", "unset")
        if is_degraded(backend) or not publishable_corpus(corpus) or status != "final":
            refused.append(r)
        else:
            kept.append(r)
    return kept, refused
