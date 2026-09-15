"""In-memory implementation of the benchmark persistence repository."""

from typing import List, Optional, Dict, Any, Tuple
from regrag.models import EvaluationRecord, GenerationResult
from regrag.storage.base import BenchmarkResultRepository, publishable_evaluations


class InMemoryResultRepository(BenchmarkResultRepository):
    """Stores benchmark evaluations and generations in memory."""

    def __init__(self) -> None:
        self._evaluations: Dict[Tuple[str, str, str], EvaluationRecord] = {}
        self._generations: List[GenerationResult] = []

    def save_evaluation(self, record: EvaluationRecord) -> None:
        key = (record.question_id, record.model_name, record.retrieval_mode)
        self._evaluations[key] = record

    def get_evaluation(
        self, question_id: str, model_name: str, retrieval_mode: str
    ) -> Optional[EvaluationRecord]:
        return self._evaluations.get((question_id, model_name, retrieval_mode))

    def list_evaluations(
        self,
        model_name: Optional[str] = None,
        retrieval_mode: Optional[str] = None,
    ) -> List[EvaluationRecord]:
        results = list(self._evaluations.values())
        if model_name:
            results = [r for r in results if r.model_name == model_name]
        if retrieval_mode:
            results = [r for r in results if r.retrieval_mode == retrieval_mode]
        return results

    def save_generation(self, result: GenerationResult) -> None:
        self._generations.append(result)

    def list_generations(
        self,
        model_name: Optional[str] = None,
        retrieval_mode: Optional[str] = None,
    ) -> List[GenerationResult]:
        results = self._generations
        if model_name:
            results = [g for g in results if g.model_name == model_name]
        if retrieval_mode:
            results = [g for g in results if g.retrieval_mode == retrieval_mode]
        return results

    def export_summary(self) -> Dict[str, Any]:
        evals = list(self._evaluations.values())
        kept, refused = publishable_evaluations(evals)
        if not kept:
            return {
                "total": 0,
                "groups": {},
                "refused_unpublishable": len(refused),
                "note": "no publishable rows; nothing averaged",
            }

        # Aggregate by (model, mode)
        groups: Dict[Tuple[str, str], List[EvaluationRecord]] = {}
        for e in kept:
            k = (e.model_name, e.retrieval_mode)
            groups.setdefault(k, []).append(e)

        summary = {"total_evaluations": len(kept), "groups": {}}
        for (m, r), recs in groups.items():
            n = len(recs)
            prec = sum(r.citation_precision for r in recs) / n
            rec = sum(r.citation_recall for r in recs) / n
            halluc = sum(1 for r in recs if r.hallucinated) / n
            abst_corr = sum(1 for r in recs if r.abstained_correctly) / n
            avg_correctness = sum(r.correctness_score for r in recs) / n
            summary["groups"][f"{m}::{r}"] = {
                "count": n,
                "citation_precision": round(prec, 4),
                "citation_recall": round(rec, 4),
                "hallucination_rate": round(halluc, 4),
                # Not an accuracy: see metrics.compute_abstention_metrics for the
                # real TP/FP/TN/FN figure over probes and answerable rows.
                "abstained_correctly_rate": round(abst_corr, 4),
                "avg_correctness": round(avg_correctness, 4),
            }
        summary["refused_unpublishable"] = len(refused)
        return summary
