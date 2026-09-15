"""Unit tests for provenance guards, degraded index identification, and placeholder metrics."""

import os
import shutil
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import torch

from regrag.models import (
    LegalChunk,
    GenerationResult,
    GoldQuestion,
    EvaluationRecord,
)
from regrag.provenance import (
    ProvenanceError,
    assert_publishable,
    CORPUS_TIER1,
    CORPUS_TIER2,
    CORPUS_UNSET,
    is_degraded,
)
from regrag.indexing import dense, bm25
from regrag.indexing.dense import DenseIndex
from regrag.indexing.bm25 import BM25Index, tokenize_vietnamese
from regrag.evaluation.metrics import (
    METRIC_STATUS_FINAL,
    METRIC_STATUS_UNVALIDATED,
    evaluate_response,
)
from regrag.storage.file_repo import FileResultRepository


class TestProvenance(unittest.TestCase):

    def setUp(self) -> None:
        self.sample_chunk = LegalChunk(
            chunk_id="C001",
            doc_id="18/2024/TT-NHNN",
            doc_title="Quy định hoạt động thẻ",
            chapter="Chương II",
            article_id="14",
            article_title="Hạn mức giao dịch thẻ",
            clause_id="1",
            text="Tổ chức phát hành thẻ thỏa thuận với chủ thẻ về hạn mức thanh toán.",
            corpus_source=CORPUS_TIER2,
        )

    # 1. DenseIndex raises ProvenanceError when sentence_transformers is unavailable
    def test_dense_search_raises_when_backend_unavailable(self):
        with patch.object(dense, "HAS_SENTENCE_TRANSFORMERS", False):
            index = DenseIndex([self.sample_chunk])
            with self.assertRaises(ProvenanceError) as cm:
                index.search("hạn mức thẻ")
            self.assertIn("sentence_transformers", str(cm.exception))
            self.assertIn("pip install", str(cm.exception))

    def test_dense_build_raises_when_backend_unavailable(self):
        with patch.object(dense, "HAS_SENTENCE_TRANSFORMERS", False):
            index = DenseIndex([self.sample_chunk])
            with self.assertRaises(ProvenanceError) as cm:
                index.build()
            self.assertIn("sentence_transformers", str(cm.exception))

    def test_dense_search_raises_when_unbuilt(self):
        mock_st_cls = MagicMock()
        with patch.object(dense, "HAS_SENTENCE_TRANSFORMERS", True), \
             patch.object(dense, "SentenceTransformer", mock_st_cls):
            index = DenseIndex([self.sample_chunk])
            # search without calling build() must raise ProvenanceError
            with self.assertRaises(ProvenanceError) as cm:
                index.search("hạn mức thẻ")
            self.assertIn("not been built", str(cm.exception))

    def test_dense_search_raises_when_chunks_empty(self):
        mock_st_cls = MagicMock()
        with patch.object(dense, "HAS_SENTENCE_TRANSFORMERS", True), \
             patch.object(dense, "SentenceTransformer", mock_st_cls):
            index = DenseIndex([])
            index.build()
            with self.assertRaises(ProvenanceError) as cm:
                index.search("hạn mức thẻ")
            self.assertIn("not been built or contains no chunks", str(cm.exception))

    def test_dense_build_releases_gpu_after_indexing(self):
        """BGE-M3 must not stay resident beside a 4-bit 7B during generation
        (2026-09-12 OOM at row ~171: ~2.3 GiB of fp32 encoder held the VRAM
        that attention transients needed)."""
        mock_st_cls = MagicMock()
        mock_model = MagicMock()
        mock_model.encode.return_value = torch.tensor([[1.0, 0.0]])
        mock_st_cls.return_value = mock_model
        with patch.object(dense, "HAS_SENTENCE_TRANSFORMERS", True), \
             patch.object(dense, "SentenceTransformer", mock_st_cls):
            index = DenseIndex([self.sample_chunk])
            index.build()
            mock_model.to.assert_called_once_with("cpu")

    # 2. DenseIndex stamps non-UNSET, non-DEGRADED retriever_backend when available
    def test_dense_stamps_truthful_backend_when_available(self):
        mock_st_cls = MagicMock()
        mock_model = MagicMock()
        mock_model.encode.side_effect = [
            torch.tensor([[1.0, 0.0]]),  # chunk embeddings
            torch.tensor([1.0, 0.0]),    # query embedding
        ]
        mock_st_cls.return_value = mock_model

        with patch.object(dense, "HAS_SENTENCE_TRANSFORMERS", True), \
             patch.object(dense, "SentenceTransformer", mock_st_cls):
            model_name = "BAAI/bge-m3"
            index = DenseIndex([self.sample_chunk], model_name=model_name)
            index.build()

            self.assertEqual(index.backend, model_name)
            self.assertFalse(is_degraded(index.backend))

            results = index.search("hạn mức thẻ", top_k=1)
            self.assertEqual(len(results), 1)
            self.assertEqual(results[0].retriever_backend, model_name)
            self.assertFalse(is_degraded(results[0].retriever_backend))
            self.assertNotEqual(results[0].retriever_backend, CORPUS_UNSET)

    # 3. BM25Index stamps DEGRADED backend tag when rank_bm25 or pyvi is absent, clean tag when both present
    def test_bm25_backend_tags(self):
        chunks = [self.sample_chunk]

        # Case 3a: Both missing -> degraded with both reasons
        with patch.object(bm25, "HAS_RANK_BM25", False), \
             patch.object(bm25, "HAS_PYVI", False):
            index = BM25Index(chunks)
            self.assertTrue(is_degraded(index.backend))
            self.assertIn("rank_bm25 missing", index.backend)
            self.assertIn("pyvi missing", index.backend)

            results = index.search("thẻ", top_k=1)
            self.assertEqual(len(results), 1)
            self.assertEqual(results[0].retriever_backend, index.backend)
            self.assertTrue(is_degraded(results[0].retriever_backend))

        # Case 3b: rank_bm25 present, pyvi missing -> degraded("pyvi missing")
        mock_bm25_cls = MagicMock()
        mock_bm25_instance = MagicMock()
        mock_bm25_instance.get_scores.return_value = [0.9]
        mock_bm25_cls.return_value = mock_bm25_instance

        with patch.object(bm25, "HAS_RANK_BM25", True), \
             patch.object(bm25, "HAS_PYVI", False), \
             patch.object(bm25, "BM25Okapi", mock_bm25_cls):
            index = BM25Index(chunks)
            self.assertTrue(is_degraded(index.backend))
            self.assertIn("pyvi missing", index.backend)
            self.assertNotIn("rank_bm25 missing", index.backend)

            results = index.search("thẻ", top_k=1)
            self.assertEqual(len(results), 1)
            self.assertEqual(results[0].retriever_backend, index.backend)
            self.assertTrue(is_degraded(results[0].retriever_backend))

        # Case 3c: rank_bm25 missing, pyvi present -> degraded("rank_bm25 missing")
        mock_vitok = MagicMock()
        mock_vitok.tokenize.side_effect = lambda t: t

        with patch.object(bm25, "HAS_RANK_BM25", False), \
             patch.object(bm25, "HAS_PYVI", True), \
             patch.object(bm25, "ViTokenizer", mock_vitok):
            index = BM25Index(chunks)
            self.assertTrue(is_degraded(index.backend))
            self.assertIn("rank_bm25 missing", index.backend)
            self.assertNotIn("pyvi missing", index.backend)

            results = index.search("thẻ", top_k=1)
            self.assertEqual(len(results), 1)
            self.assertEqual(results[0].retriever_backend, index.backend)
            self.assertTrue(is_degraded(results[0].retriever_backend))

        # Case 3d: Both present -> clean "bm25-rank_bm25+pyvi"
        with patch.object(bm25, "HAS_RANK_BM25", True), \
             patch.object(bm25, "HAS_PYVI", True), \
             patch.object(bm25, "BM25Okapi", mock_bm25_cls), \
             patch.object(bm25, "ViTokenizer", mock_vitok):
            index = BM25Index(chunks)
            self.assertEqual(index.backend, "bm25-rank_bm25+pyvi")
            self.assertFalse(is_degraded(index.backend))

            results = index.search("thẻ", top_k=1)
            self.assertEqual(len(results), 1)
            self.assertEqual(results[0].retriever_backend, "bm25-rank_bm25+pyvi")
            self.assertFalse(is_degraded(results[0].retriever_backend))

    def test_tokenize_vietnamese_preserves_hyphenated_tokens(self):
        text = "Quy định tại Thông tư 18/2024/TT-NHNN và Nghị định 52/2024/NĐ-CP"
        tokens = tokenize_vietnamese(text)
        self.assertIn("tt-nhnn", tokens)
        self.assertIn("nđ-cp", tokens)

    # 4. assert_publishable raises on DEGRADED backend and CORPUS_TIER1, passes on CORPUS_TIER2 + real backend
    def test_assert_publishable(self):
        valid_rec = EvaluationRecord(
            question_id="Q001",
            model_name="qwen2.5-7b",
            retrieval_mode="rag_dense",
            is_answerable=True,
            corpus_source=CORPUS_TIER2,
            retriever_backend="BAAI/bge-m3",
        )
        # Valid row must pass cleanly
        self.assertEqual(assert_publishable([valid_rec]), [])

        # DEGRADED backend must raise ProvenanceError
        degraded_backend_rec = EvaluationRecord(
            question_id="Q002",
            model_name="qwen2.5-7b",
            retrieval_mode="rag_bm25",
            is_answerable=True,
            corpus_source=CORPUS_TIER2,
            retriever_backend="DEGRADED:rank_bm25 missing",
        )
        with self.assertRaises(ProvenanceError) as cm:
            assert_publishable([degraded_backend_rec])
        self.assertIn("unpublishable provenance", str(cm.exception))

        # CORPUS_TIER1 (fixture corpus) must raise ProvenanceError
        tier1_rec = EvaluationRecord(
            question_id="Q003",
            model_name="qwen2.5-7b",
            retrieval_mode="rag_dense",
            is_answerable=True,
            corpus_source=CORPUS_TIER1,
            retriever_backend="BAAI/bge-m3",
        )
        with self.assertRaises(ProvenanceError) as cm:
            assert_publishable([tier1_rec])
        self.assertIn("unpublishable provenance", str(cm.exception))

        # UNSET backend must raise ProvenanceError
        unset_backend_rec = EvaluationRecord(
            question_id="Q004",
            model_name="qwen2.5-7b",
            retrieval_mode="rag_dense",
            is_answerable=True,
            corpus_source=CORPUS_TIER2,
            retriever_backend=CORPUS_UNSET,
        )
        with self.assertRaises(ProvenanceError) as cm:
            assert_publishable([unset_backend_rec])
        self.assertIn("unpublishable provenance", str(cm.exception))

    # 5. evaluate_response self-identifies as unvalidated with non-empty placeholder_fields
    def test_evaluate_response_placeholder_and_provenance(self):
        gen = GenerationResult(
            question_id="Q001",
            model_name="qwen2.5-7b-instruct",
            retrieval_mode="rag_dense",
            prompt="Question: Hạn mức rút ngoại tệ...",
            raw_response="Theo Điều 14 Thông tư 18/2024/TT-NHNN, hạn mức...",
            answer_text="Theo Điều 14 Thông tư 18/2024/TT-NHNN, hạn mức...",
            corpus_source=CORPUS_TIER2,
            retriever_backend="BAAI/bge-m3",
        )
        gold = GoldQuestion(
            id="Q001",
            question="Hạn mức rút ngoại tệ tiền mặt tại nước ngoài là bao nhiêu?",
            is_answerable=True,
            gold_citations=[{"doc_id": "18/2024/TT-NHNN", "article_id": "14"}],
            reference_answer="Không quá tương đương 30 triệu đồng/ngày.",
        )

        record = evaluate_response(gen, gold)
        # The scorers are real but not yet validated against human labels, so the
        # record must self-identify as unvalidated and must never claim to be final.
        self.assertEqual(record.metric_status, METRIC_STATUS_UNVALIDATED)
        self.assertNotEqual(record.metric_status, METRIC_STATUS_FINAL)
        self.assertIsInstance(record.placeholder_fields, list)
        self.assertTrue(len(record.placeholder_fields) > 0)
        self.assertIn("correctness_score", record.placeholder_fields)
        self.assertIn("hallucinated", record.placeholder_fields)
        self.assertIn("abstained", record.placeholder_fields)
        self.assertIn("abstained_correctly", record.placeholder_fields)

        # Provenance must survive scoring
        self.assertEqual(record.corpus_source, CORPUS_TIER2)
        self.assertEqual(record.retriever_backend, "BAAI/bge-m3")

    # 6. file_repo raises instead of silently swallowing errors
    def test_file_repo_raises_on_write_error(self):
        temp_dir = tempfile.mkdtemp()
        try:
            repo = FileResultRepository(temp_dir)
            rec = EvaluationRecord(
                question_id="Q001",
                model_name="qwen2.5-7b",
                retrieval_mode="rag_bm25",
                is_answerable=True,
            )
            with patch("builtins.open", side_effect=IOError("Disk write failure")):
                with self.assertRaises(IOError) as cm:
                    repo.save_evaluation(rec)
                self.assertIn("Failed to persist", str(cm.exception))
        finally:
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)

    def test_file_repo_raises_on_load_parse_error(self):
        temp_dir = tempfile.mkdtemp()
        try:
            eval_path = os.path.join(temp_dir, "evaluations.json")
            with open(eval_path, "w", encoding="utf-8") as f:
                f.write("{ corrupt json content")

            with self.assertRaises(IOError) as cm:
                FileResultRepository(temp_dir)
            self.assertIn("Failed to load evaluations", str(cm.exception))
        finally:
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)


if __name__ == "__main__":
    unittest.main()
