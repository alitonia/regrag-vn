"""File-based implementation of the benchmark persistence repository.

Persists evaluation and generation records to JSON and CSV files on disk.
Shares identical interface with InMemoryResultRepository.
"""

import os
import csv
import json
from dataclasses import asdict
from typing import List, Optional, Dict, Any, Tuple
from regrag.models import EvaluationRecord, GenerationResult
from regrag.storage.base import BenchmarkResultRepository, publishable_evaluations


class FileResultRepository(BenchmarkResultRepository):
    """Stores benchmark evaluations and generations as JSON and CSV files."""

    def __init__(self, base_dir: str) -> None:
        self.base_dir = base_dir
        self.eval_json_path = os.path.join(base_dir, "evaluations.json")
        self.eval_csv_path = os.path.join(base_dir, "evaluations.csv")
        self.gen_json_path = os.path.join(base_dir, "generations.json")
        os.makedirs(base_dir, exist_ok=True)

        self._eval_cache: Dict[Tuple[str, str, str], EvaluationRecord] = {}
        self._gen_cache: List[GenerationResult] = []
        self._load_from_disk()

    def _load_from_disk(self) -> None:
        """Load existing evaluations from disk if available."""
        if os.path.exists(self.eval_json_path):
            try:
                with open(self.eval_json_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    for item in data:
                        rec = EvaluationRecord(**item)
                        self._eval_cache[(rec.question_id, rec.model_name, rec.retrieval_mode)] = rec
            except Exception as e:
                raise IOError(f"Failed to load evaluations from {self.eval_json_path}: {e}") from e

        if os.path.exists(self.gen_json_path):
            try:
                with open(self.gen_json_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    for item in data:
                        self._gen_cache.append(GenerationResult(**item))
            except Exception as e:
                raise IOError(f"Failed to load generations from {self.gen_json_path}: {e}") from e

    def _flush_evaluations(self) -> None:
        """Atomically persist evaluations to both JSON and CSV."""
        eval_list = [asdict(r) for r in self._eval_cache.values()]

        # Write JSON atomically
        tmp_json = self.eval_json_path + ".tmp"
        try:
            with open(tmp_json, "w", encoding="utf-8") as f:
                json.dump(eval_list, f, ensure_ascii=False, indent=2)
            os.replace(tmp_json, self.eval_json_path)

            # Write CSV atomically
            if eval_list:
                fieldnames = list(eval_list[0].keys())
                tmp_csv = self.eval_csv_path + ".tmp"
                with open(tmp_csv, "w", encoding="utf-8", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(eval_list)
                os.replace(tmp_csv, self.eval_csv_path)
        except Exception as e:
            raise IOError(f"Failed to persist evaluations to disk in {self.base_dir}: {e}") from e

    def _flush_generations(self) -> None:
        """Atomically persist generations to JSON."""
        gen_list = [asdict(g) for g in self._gen_cache]
        tmp_json = self.gen_json_path + ".tmp"
        try:
            with open(tmp_json, "w", encoding="utf-8") as f:
                json.dump(gen_list, f, ensure_ascii=False, indent=2)
            os.replace(tmp_json, self.gen_json_path)
        except Exception as e:
            raise IOError(f"Failed to persist generations to disk in {self.base_dir}: {e}") from e

    def save_evaluation(self, record: EvaluationRecord) -> None:
        key = (record.question_id, record.model_name, record.retrieval_mode)
        self._eval_cache[key] = record
        self._flush_evaluations()

    def get_evaluation(
        self, question_id: str, model_name: str, retrieval_mode: str
    ) -> Optional[EvaluationRecord]:
        return self._eval_cache.get((question_id, model_name, retrieval_mode))

    def list_evaluations(
        self,
        model_name: Optional[str] = None,
        retrieval_mode: Optional[str] = None,
    ) -> List[EvaluationRecord]:
        results = list(self._eval_cache.values())
        if model_name:
            results = [r for r in results if r.model_name == model_name]
        if retrieval_mode:
            results = [r for r in results if r.retrieval_mode == retrieval_mode]
        return results

    def save_generation(self, result: GenerationResult) -> None:
        self._gen_cache.append(result)
        self._flush_generations()

    def list_generations(
        self,
        model_name: Optional[str] = None,
        retrieval_mode: Optional[str] = None,
    ) -> List[GenerationResult]:
        results = self._gen_cache
        if model_name:
            results = [g for g in results if g.model_name == model_name]
        if retrieval_mode:
            results = [g for g in results if g.retrieval_mode == retrieval_mode]
        return results

    def export_summary(self) -> Dict[str, Any]:
        evals = list(self._eval_cache.values())
        kept, refused = publishable_evaluations(evals)
        if not kept:
            return {
                "total": 0,
                "groups": {},
                "refused_unpublishable": len(refused),
                "note": "no publishable rows; nothing averaged",
            }

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
                # Not an accuracy: this is the share of rows whose abstention decision
                # was correct. True abstention accuracy (TP/FP/TN/FN over probes and
                # answerable rows) comes from metrics.compute_abstention_metrics.
                "abstained_correctly_rate": round(abst_corr, 4),
                "avg_correctness": round(avg_correctness, 4),
            }
        summary["refused_unpublishable"] = len(refused)
        return summary
