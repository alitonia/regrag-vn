"""Unit tests for coverage invariant, doc_id resolution, citation bugfixes, and cache keys."""

import csv
import json
import os
import tempfile
import unittest

from regrag.corpus.qa_loader import load_gold_questions, load_and_report
from regrag.evaluation.citation import (
    extract_citations,
    compute_citation_precision_recall,
)
from regrag.models import GoldQuestion, LegalChunk
from regrag.provenance import CORPUS_TIER1
from scripts.regenerate import check_passage_coverage, cache_key


class TestCitationRegression(unittest.TestCase):
    """Regression tests for the two real bugs in regrag/evaluation/citation.py."""

    def test_bug1_broadcasting_prevention(self):
        """Bug 1: extract_citations must not broadcast Khoản 2 to both Điều 5 and Điều 8.

        Input: 'Theo Điều 5 và Điều 8 Thông tư 06/2019/TT-NHNN, Khoản 2 quy định...'
        Clause 2 belongs only to Điều 8 (nearest). Điều 5 must have clause_id=None.
        """
        text = "Theo Điều 5 và Điều 8 Thông tư 06/2019/TT-NHNN, Khoản 2 quy định..."
        citations = extract_citations(text)
        self.assertEqual(len(citations), 2)

        # First citation: Điều 5
        c_art5 = [c for c in citations if c["article_id"] == "5"][0]
        self.assertEqual(c_art5["doc_id"], "06/2019/TT-NHNN")
        self.assertIsNone(c_art5["clause_id"], "Clause 2 must NOT be broadcast to Điều 5")

        # Second citation: Điều 8
        c_art8 = [c for c in citations if c["article_id"] == "8"][0]
        self.assertEqual(c_art8["doc_id"], "06/2019/TT-NHNN")
        self.assertEqual(c_art8["clause_id"], "2")

    def test_bug2_pattern_coverage_named_laws(self):
        """Bug 2: Must recognize Luật Các tổ chức tín dụng 2024 and map to 32/2024/QH15.

        Input: 'Căn cứ Điều 134 Luật Các tổ chức tín dụng 2024, khoản 3'
        Previously returned doc_id: 'UNKNOWN'. Must return 32/2024/QH15.
        """
        text = "Căn cứ Điều 134 Luật Các tổ chức tín dụng 2024, khoản 3"
        citations = extract_citations(text)
        self.assertEqual(len(citations), 1)
        self.assertEqual(citations[0]["doc_id"], "32/2024/QH15")
        self.assertEqual(citations[0]["article_id"], "134")
        self.assertEqual(citations[0]["clause_id"], "3")

    def test_bug2_additional_legal_patterns(self):
        """Verify pattern coverage and canonical normalization for Decree, Law, Decision, and isolated Điều."""
        # Decree with ND-CP (ASCII D) -> normalized to NĐ-CP
        decree_text = "Theo Điều 23 Nghị định 21/2021/ND-CP, khoản 1"
        c_dec = extract_citations(decree_text)
        self.assertEqual(len(c_dec), 1)
        self.assertEqual(c_dec[0]["doc_id"], "21/2021/NĐ-CP")
        self.assertEqual(c_dec[0]["article_id"], "23")
        self.assertEqual(c_dec[0]["clause_id"], "1")

        # Law on Notarization 2024 -> 46/2024/QH15
        law_text = "Căn cứ Điều 10 Luật Công chứng 2024"
        c_law = extract_citations(law_text)
        self.assertEqual(len(c_law), 1)
        self.assertEqual(c_law[0]["doc_id"], "46/2024/QH15")
        self.assertEqual(c_law[0]["article_id"], "10")

        # Decision 2345/QĐ-NHNN
        dec_text = "Theo Điều 1 Quyết định 2345/QĐ-NHNN"
        c_dec2 = extract_citations(dec_text)
        self.assertEqual(len(c_dec2), 1)
        self.assertEqual(c_dec2[0]["doc_id"], "2345/QĐ-NHNN")

        # Isolated Điều without nearby document mention -> doc_id='UNKNOWN'
        iso_text = "Theo quy định tại Điều 15, việc công khai biểu phí dịch vụ thẻ là bắt buộc."
        c_iso = extract_citations(iso_text)
        self.assertEqual(len(c_iso), 1)
        self.assertEqual(c_iso[0]["doc_id"], "UNKNOWN")
        self.assertEqual(c_iso[0]["article_id"], "15")

    def test_precision_recall_tightened_doc_matching(self):
        """Tightened doc_id matching: document mismatch must fail, UNKNOWN tolerated."""
        # Doc mismatch with identical article: must NOT match
        pred_mismatch = [{"doc_id": "06/2019/TT-NHNN", "article_id": "8"}]
        gold_target = [{"doc_id": "17/2024/TT-NHNN", "article_id": "8"}]
        prec, rec = compute_citation_precision_recall(pred_mismatch, gold_target)
        self.assertEqual(prec, 0.0)
        self.assertEqual(rec, 0.0)

        # UNKNOWN doc on predicted side: tolerated for closed-book answers
        pred_unknown = [{"doc_id": "UNKNOWN", "article_id": "8"}]
        prec, rec = compute_citation_precision_recall(pred_unknown, gold_target)
        self.assertEqual(prec, 1.0)
        self.assertEqual(rec, 1.0)

        # Exact match
        pred_exact = [{"doc_id": "17/2024/TT-NHNN", "article_id": "8"}]
        prec, rec = compute_citation_precision_recall(pred_exact, gold_target)
        self.assertEqual(prec, 1.0)
        self.assertEqual(rec, 1.0)


class TestDocIdResolution(unittest.TestCase):
    """Test resolution of doc_ids from gold QA CSV."""

    def test_verified_gold_csv_state(self):
        """Verify structural invariants of data/gold/bank_qa_data.csv.

        The CSV is under active revision (64 rows on 2026-09-09, 88 on 2026-09-11),
        so this asserts invariants recomputed from the raw file rather than literal
        counts that would break on the next delivery: nothing is dropped, resolved
        and unresolved partition the answerable set, unresolved rows keep their
        marker, and probes plus answerable equal the total.
        """
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        csv_path = os.path.join(repo_root, "data", "gold", "bank_qa_data.csv")
        questions, report = load_and_report(csv_path, repo_root=repo_root)

        with open(csv_path, newline="", encoding="utf-8-sig") as f:
            raw_rows = [r for r in csv.DictReader(f) if (r.get("question") or "").strip()]

        self.assertEqual(report["questions"], len(raw_rows))
        self.assertEqual(len(questions), len(raw_rows), "no rows may be dropped")
        self.assertEqual(
            report["answerable_questions"] + report["unanswerable_probes"],
            report["questions"],
        )
        self.assertEqual(
            report["doc_ids_resolved"] + report["doc_ids_unresolved"],
            report["answerable_questions"],
        )
        self.assertEqual(len(report["unresolved_rows"]), report["doc_ids_unresolved"])
        for row in report["unresolved_rows"]:
            self.assertTrue(
                any(str(d).startswith("UNRESOLVED") for d in row["doc_ids"]),
                f"unresolved row {row['id']} carries no UNRESOLVED marker: {row['doc_ids']}",
            )
        self.assertGreater(len(report["distinct_doc_ids"]), 0)

    def test_fixture_preserves_unresolved_rows(self):
        """Synthetic fixture test: unresolved rows must retain UNRESOLVED prefix and not drop."""
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".csv", delete=False) as f:
            writer = csv.writer(f)
            writer.writerow(["ID", "question", "answer", "text_contains_answer_in_the_doc", "doc_link"])
            writer.writerow([
                "1",
                "Q1?",
                "Ans1",
                "Điều 1. Quy định chung",
                "https://congbao.chinhphu.vn/van-ban/thong-tu-so-06-2019-tt-nhnn-29358.htm",
            ])
            writer.writerow([
                "2",
                "Q2?",
                "Ans2",
                "Điều 2. Đối tượng",
                "https://example.com/unknown_circular.pdf",
            ])
            tmp_csv = f.name

        try:
            questions, report = load_and_report(tmp_csv, repo_root=repo_root)
            self.assertEqual(len(questions), 2)
            self.assertEqual(report["doc_ids_resolved"], 1)
            self.assertEqual(report["doc_ids_unresolved"], 1)
            self.assertTrue(questions[1].gold_doc_ids[0].startswith("UNRESOLVED:"))
            self.assertEqual(len(report["unresolved_rows"]), 1)
            self.assertEqual(report["unresolved_rows"][0]["id"], "Q002")
        finally:
            if os.path.exists(tmp_csv):
                os.remove(tmp_csv)


class TestCoverageInvariant(unittest.TestCase):
    """The coverage invariant: every answerable question's gold_passage must be found in chunks."""

    def setUp(self):
        self.repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        self.csv_path = os.path.join(self.repo_root, "data", "gold", "bank_qa_data.csv")
        self.questions = load_gold_questions(self.csv_path, repo_root=self.repo_root)

    def test_invariant_partitions_against_live_corpus(self):
        """Coverage against data/processed_chunks/corpus_chunks.json partitions cleanly.

        The corpus file is a live artifact (empty on 2026-09-09, 3,708 chunks from
        2026-09-11), so no fixed match count is asserted here. The invariant that must
        hold for ANY corpus is that every answerable row lands in exactly one bucket
        and that probes are excluded from all of them.
        """
        chunks_path = os.path.join(self.repo_root, "data", "processed_chunks", "corpus_chunks.json")
        self.assertTrue(os.path.exists(chunks_path), f"Missing {chunks_path}")

        with open(chunks_path, "r", encoding="utf-8") as f:
            chunks = json.load(f)

        cov = check_passage_coverage(self.questions, chunks)
        n_answerable = sum(1 for q in self.questions if q.is_answerable)
        self.assertEqual(
            cov["strict_matches"] + cov["squash_matches"] + cov["loose_matches"] + cov["not_found"],
            n_answerable,
            "every answerable row must land in exactly one coverage bucket",
        )
        self.assertEqual(cov["answerable_questions"], n_answerable)
        self.assertEqual(
            cov["unanswerable_probes_excluded"],
            sum(1 for q in self.questions if not q.is_answerable),
        )

    def test_invariant_passes_against_tier1_corpus(self):
        """Invariant PASSES against Tier 1 corpus built inline from CSV gold passages.

        Builds Tier 1 LegalChunk objects inline directly from CSV questions without
        relying on external builder scripts. All 64 answerable questions must match strictly.
        """
        tier1_chunks = []
        for q in self.questions:
            if q.is_answerable and q.gold_passage:
                tier1_chunks.append(
                    LegalChunk(
                        chunk_id=f"T1_{q.id}",
                        doc_id=q.gold_doc_ids[0] if q.gold_doc_ids else "UNKNOWN",
                        doc_title="Tier 1 Fixture",
                        chapter=None,
                        article_id=q.gold_citations[0].get("article_id", "") if q.gold_citations else "",
                        article_title="Tier 1 Fixture Passage",
                        clause_id=q.gold_citations[0].get("clause_id") if q.gold_citations else None,
                        text=q.gold_passage,
                        corpus_source=CORPUS_TIER1,
                    )
                )

        cov = check_passage_coverage(self.questions, tier1_chunks)
        n_answerable = sum(
            1 for q in self.questions if q.is_answerable and q.gold_passage
        )
        self.assertEqual(
            cov["strict_matches"], n_answerable,
            "All answerable questions with a gold passage must strictly match Tier 1",
        )
        self.assertEqual(cov["loose_matches"], 0, "No questions should match only at loose tier")
        self.assertEqual(cov["squash_matches"], 0, "No questions should match only at squash tier")
        self.assertEqual(cov["not_found"], 0, "No questions should be missing in Tier 1")
        self.assertEqual(cov["strict_coverage_pct"], 100.0)

    def test_loose_tier_diagnoses_diacritic_loss(self):
        """Loose tier matches ONLY when diacritics were stripped; reported distinctly."""
        # Simulated question with full Vietnamese diacritics
        q = GoldQuestion(
            id="Q_TEST",
            question="Thử nghiệm?",
            is_answerable=True,
            gold_passage="Điều 1. Phạm vi điều chỉnh và đối tượng áp dụng cho ngân hàng thương mại",
        )
        # Chunk with stripped diacritics (simulating ingestion damage)
        chunk = {
            "chunk_id": "C_TEST",
            "text": "Dieu 1. Pham vi dieu chinh va doi tuong ap dung cho ngan hang thuong mai",
        }

        cov = check_passage_coverage([q], [chunk])
        self.assertEqual(cov["strict_matches"], 0, "Stripped diacritics must NOT match at strict tier")
        self.assertEqual(cov["squash_matches"], 0, "Stripped diacritics must not match at squash tier either")
        self.assertEqual(cov["loose_matches"], 1, "Must match at loose diacritic-folded tier")
        self.assertEqual(cov["not_found"], 0)
        self.assertIn("Q_TEST", cov["loose_question_ids"])

    def test_squash_tier_diagnoses_intra_word_spacing_damage(self):
        """Squash tier matches ONLY when intra-word spacing is damaged; reported distinctly.

        The born-digital CÔNG BÁO layers ship with spurious spaces inside words
        ("vi ệc", "th ường") while keeping every diacritic. Such a chunk must fail
        the strict tier, pass the whitespace-squashed tier, and be reported under
        squash_question_ids - never counted as a strict match.
        """
        q = GoldQuestion(
            id="Q_SQUASH",
            question="Thử nghiệm?",
            is_answerable=True,
            gold_passage="Điều 1. Việc mở tài khoản thanh toán phải tuân thủ quy định",
        )
        # Chunk identical except spaces inserted inside words (CÔNG BÁO damage mode)
        chunk = {
            "chunk_id": "C_SQUASH",
            "text": "Điều 1. Vi ệc mở tài kho ản thanh to án phải tuân th ủ quy định",
        }

        cov = check_passage_coverage([q], [chunk])
        self.assertEqual(cov["strict_matches"], 0, "Spacing damage must NOT count as strict")
        self.assertEqual(cov["squash_matches"], 1, "Must match at whitespace-squashed tier")
        self.assertEqual(cov["loose_matches"], 0, "Diacritics intact: loose tier is not the diagnosis")
        self.assertEqual(cov["not_found"], 0)
        self.assertIn("Q_SQUASH", cov["squash_question_ids"])


class TestCacheKey(unittest.TestCase):
    """Test per-question invalidation and stability of cache_key."""

    def test_cache_key_stability_and_invalidation(self):
        """Editing 1 question must NOT invalidate cache keys of untouched questions."""
        csv_hash_v1 = "abcdef0123456789"
        csv_hash_v2 = "9876543210fedcba"  # Global CSV changed due to editing Q2
        corpus_hash = "corpus_hash_stable"

        q1_text = "Hạn mức rút tiền thẻ ATM tại nước ngoài là bao nhiêu?"
        q2_text_v1 = "Lãi suất quá hạn tối đa được áp dụng là bao nhiêu?"
        q2_text_v2 = "Mức trần lãi suất quá hạn theo Thông tư 39 là bao nhiêu?"

        # Initial cache keys
        k1_v1 = cache_key("Q001", "qwen2.5-7b", "rag_bm25", csv_hash_v1, corpus_hash, question_content=q1_text)
        k2_v1 = cache_key("Q002", "qwen2.5-7b", "rag_bm25", csv_hash_v1, corpus_hash, question_content=q2_text_v1)

        # After CSV update where Q1 is untouched and Q2 is edited
        k1_v2 = cache_key("Q001", "qwen2.5-7b", "rag_bm25", csv_hash_v2, corpus_hash, question_content=q1_text)
        k2_v2 = cache_key("Q002", "qwen2.5-7b", "rag_bm25", csv_hash_v2, corpus_hash, question_content=q2_text_v2)

        # Q1 cache key must remain identical despite global CSV hash change
        self.assertEqual(
            k1_v1, k1_v2,
            "Cache key for untouched question Q1 must stay identical when another question is edited",
        )

        # Q2 cache key must change because its content changed
        self.assertNotEqual(
            k2_v1, k2_v2,
            "Cache key for edited question Q2 must change",
        )

        # Corpus hash change must invalidate even untouched questions
        k1_new_corpus = cache_key("Q001", "qwen2.5-7b", "rag_bm25", csv_hash_v1, "new_corpus_hash", question_content=q1_text)
        self.assertNotEqual(k1_v1, k1_new_corpus)


if __name__ == "__main__":
    unittest.main()
