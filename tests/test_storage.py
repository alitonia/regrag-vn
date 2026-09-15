"""Unit tests for BenchmarkResultRepository implementations.

Verifies that both InMemoryResultRepository and FileResultRepository satisfy
the exact same behavioral contract (two persistence layers with unchanged business logic).
"""

import os
import shutil
import tempfile
import unittest
from regrag.models import EvaluationRecord, GenerationResult
from regrag.provenance import CORPUS_TIER2
from regrag.storage.in_memory import InMemoryResultRepository
from regrag.storage.file_repo import FileResultRepository


class TestStorageImplementations(unittest.TestCase):

    def setUp(self) -> None:
        self.temp_dir = tempfile.mkdtemp()
        self.sample_eval = EvaluationRecord(
            question_id="Q001",
            model_name="qwen2.5-7b",
            retrieval_mode="rag_bm25",
            is_answerable=True,
            citation_precision=1.0,
            citation_recall=1.0,
            abstained=False,
            abstained_correctly=False,
            correctness_score=2.0,
            hallucinated=False,
            corpus_source=CORPUS_TIER2,
            retriever_backend="bm25-rank_bm25+pyvi",
            metric_status="final",
            notes="Test record",
        )
        self.sample_gen = GenerationResult(
            question_id="Q001",
            model_name="qwen2.5-7b",
            retrieval_mode="rag_bm25",
            prompt="Prompt...",
            raw_response="Answer...",
            answer_text="Answer...",
            abstained=False,
        )

    def tearDown(self) -> None:
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def _test_repository_contract(self, repo):
        # 1. Initially empty
        self.assertEqual(len(repo.list_evaluations()), 0)
        self.assertEqual(len(repo.list_generations()), 0)

        # 2. Save and retrieve evaluation
        repo.save_evaluation(self.sample_eval)
        retrieved = repo.get_evaluation("Q001", "qwen2.5-7b", "rag_bm25")
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved.question_id, "Q001")
        self.assertEqual(retrieved.citation_precision, 1.0)

        # 3. Filtering
        matches = repo.list_evaluations(model_name="qwen2.5-7b")
        self.assertEqual(len(matches), 1)
        no_matches = repo.list_evaluations(model_name="nonexistent")
        self.assertEqual(len(no_matches), 0)

        # 4. Save and list generations
        repo.save_generation(self.sample_gen)
        gens = repo.list_generations(retrieval_mode="rag_bm25")
        self.assertEqual(len(gens), 1)
        self.assertEqual(gens[0].question_id, "Q001")

        # 5. Export summary
        summary = repo.export_summary()
        self.assertEqual(summary["total_evaluations"], 1)
        group_key = "qwen2.5-7b::rag_bm25"
        self.assertIn(group_key, summary["groups"])
        self.assertEqual(summary["groups"][group_key]["citation_precision"], 1.0)
        self.assertEqual(summary["groups"][group_key]["hallucination_rate"], 0.0)

        # 6. An unpublishable row is refused by the summary, not averaged in
        bad = EvaluationRecord(
            question_id="Q002",
            model_name="qwen2.5-7b",
            retrieval_mode="rag_bm25",
            is_answerable=True,
            citation_precision=0.0,
            citation_recall=0.0,
            correctness_score=0.0,
        )
        repo.save_evaluation(bad)
        summary = repo.export_summary()
        self.assertEqual(summary["total_evaluations"], 1)
        self.assertEqual(summary["refused_unpublishable"], 1)

    def test_in_memory_repository(self):
        repo = InMemoryResultRepository()
        self._test_repository_contract(repo)

    def test_file_repository(self):
        repo = FileResultRepository(self.temp_dir)
        self._test_repository_contract(repo)

        # Verify physical files exist
        self.assertTrue(os.path.exists(os.path.join(self.temp_dir, "evaluations.json")))
        self.assertTrue(os.path.exists(os.path.join(self.temp_dir, "evaluations.csv")))
        self.assertTrue(os.path.exists(os.path.join(self.temp_dir, "generations.json")))

        # Test reload from disk in a fresh repository instance
        reloaded_repo = FileResultRepository(self.temp_dir)
        reloaded_eval = reloaded_repo.get_evaluation("Q001", "qwen2.5-7b", "rag_bm25")
        self.assertIsNotNone(reloaded_eval)
        self.assertEqual(reloaded_eval.question_id, "Q001")


if __name__ == "__main__":
    unittest.main()
