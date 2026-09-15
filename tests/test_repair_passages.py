"""Tests for scripts/repair_gold_passages.py - the verbatim gold-passage repair.

The repair pass rewrites annotator-summary passages with the corpus's verbatim
article text under three loud-failure guards (instrument absent, article
missing, lexical overlap low). A wrong repair attaches a question to the wrong
provision and poisons every downstream citation metric, so each guard must
refuse loudly rather than skip quietly.
"""

import csv
import json
import os
import sys
import tempfile
import unittest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO_ROOT)

from scripts import repair_gold_passages as mod

GOOD_ARTICLE = (
    "Điều 6. Tỷ lệ an toàn vốn\n"
    "1. Ngân hàng phải duy trì tỷ lệ an toàn vốn tối thiểu 8%.\n"
    "2. Ngân hàng có công ty con phải duy trì tỷ lệ hợp nhất tối thiểu 8%."
)
SUMMARY = "Điều 6: Ngân hàng phải duy trì tỷ lệ an toàn vốn tối thiểu 8%."


class RepairFixture(unittest.TestCase):
    """Shared fixture: temp CSV + canonical JSON + corpus around main()."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.csv_path = os.path.join(self.tmp.name, "gold.csv")
        self.canonical_path = os.path.join(self.tmp.name, "questions_canonical.json")
        self.corpus_path = os.path.join(self.tmp.name, "corpus.json")
        self.log_path = os.path.join(self.tmp.name, "repair_log.json")

    def write_fixture(self, rows, docs):
        """rows: (id, passage, doc_ids); docs: (doc_id, article_id, text)."""
        with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f, lineterminator="\r\n")
            w.writerow(["ID", "doc_link", "question", "answer",
                        "text_contains_answer_in_the_doc", "author"])
            canonical = []
            for rid, passage, doc_ids in rows:
                w.writerow([rid, "https://example.test/x", f"Q{rid}?", "A", passage, "Thành"])
                canonical.append({
                    "id": f"Q{int(rid):03d}", "question": "?", "is_answerable": True,
                    "gold_doc_ids": doc_ids, "gold_citations": [], "reference_answer": "A",
                    "category": "factual", "gold_passage": passage, "source_urls": [],
                    "doc_id_confidence": "manifest", "author": "Thành",
                })
        with open(self.canonical_path, "w", encoding="utf-8") as f:
            json.dump(canonical, f, ensure_ascii=False)
        chunks = [{
            "chunk_id": f"{doc_id}_D{art}", "doc_id": doc_id, "doc_title": "t",
            "chapter": None, "article_id": art, "article_title": "t", "clause_id": None,
            "text": text, "metadata": {"extraction": "pdf-text-layer:pdfplumber"},
            "corpus_source": "tier2_full",
        } for doc_id, art, text in docs]
        with open(self.corpus_path, "w", encoding="utf-8") as f:
            json.dump(chunks, f, ensure_ascii=False)

    def run_main(self, dry_run=True):
        argv, sys.argv = sys.argv, [
            "repair_gold_passages.py", "--csv", self.csv_path,
            "--corpus", self.corpus_path, "--canonical", self.canonical_path,
            "--log", self.log_path,
        ] + (["--dry-run"] if dry_run else [])
        try:
            return mod.main()
        finally:
            sys.argv = argv

    def read_csv_row(self, i=0):
        with open(self.csv_path, newline="", encoding="utf-8") as f:
            return list(csv.DictReader(f))[i]


class TestHappyPathAndGuards(RepairFixture):
    def test_summary_is_replaced_with_verbatim_article_and_logged(self):
        self.write_fixture(
            [("1", SUMMARY, ["41/2016/TT-NHNN"])],
            [("41/2016/TT-NHNN", "6", GOOD_ARTICLE)],
        )
        self.assertEqual(self.run_main(dry_run=False), 0)
        row = self.read_csv_row()
        self.assertEqual(row["text_contains_answer_in_the_doc"], GOOD_ARTICLE)
        self.assertIn("passage repaired 2026-09-11", row["author"])
        with open(self.log_path, encoding="utf-8") as f:
            log = json.load(f)
        self.assertEqual(log["repaired"][0]["original_summary"], SUMMARY)
        self.assertEqual(log["repaired"][0]["verbatim_text"], GOOD_ARTICLE)

    def test_second_run_is_idempotent(self):
        self.write_fixture(
            [("1", SUMMARY, ["41/2016/TT-NHNN"])],
            [("41/2016/TT-NHNN", "6", GOOD_ARTICLE)],
        )
        self.assertEqual(self.run_main(dry_run=False), 0)
        self.assertEqual(self.run_main(dry_run=False), 0)
        row = self.read_csv_row()
        self.assertEqual(row["text_contains_answer_in_the_doc"], GOOD_ARTICLE)
        self.assertEqual(row["author"].count("passage repaired"), 1)
        with open(self.log_path, encoding="utf-8") as f:
            self.assertEqual(len(json.load(f)["repaired"]), 0)
            # the row moved to already_verbatim instead

    def test_guard_a_instrument_absent_refuses(self):
        self.write_fixture(
            [("1", SUMMARY, ["41/2016/TT-NHNN"])],
            [("32/2024/QH15", "6", GOOD_ARTICLE)],  # different instrument only
        )
        self.assertEqual(self.run_main(dry_run=True), 1)
        self.assertFalse(os.path.exists(self.log_path))

    def test_guard_b_article_missing_refuses(self):
        self.write_fixture(
            [("1", "Điều 99: Quy định không tồn tại trong văn bản", ["41/2016/TT-NHNN"])],
            [("41/2016/TT-NHNN", "6", GOOD_ARTICLE)],
        )
        self.assertEqual(self.run_main(dry_run=True), 1)

    def test_guard_c_low_overlap_refuses(self):
        # Article number right, content wrong: the summary talks about cards
        # and wallets while Điều 6 is about capital adequacy.
        self.write_fixture(
            [("1", "Điều 6: Hoạt động thẻ ngân hàng và ví điện tử thanh toán trực tuyến",
              ["41/2016/TT-NHNN"])],
            [("41/2016/TT-NHNN", "6", GOOD_ARTICLE)],
        )
        self.assertEqual(self.run_main(dry_run=True), 1)

    def test_phu_luc_locator_reported_not_rewritten(self):
        passage = "Phụ lục 1, Phần A, mục I: Giá trị vốn cấp 2 tối đa bằng vốn cấp 1."
        self.write_fixture(
            [("1", passage, ["41/2016/TT-NHNN"])],
            [("41/2016/TT-NHNN", "6", GOOD_ARTICLE)],
        )
        self.assertEqual(self.run_main(dry_run=False), 0)
        self.assertEqual(self.read_csv_row()["text_contains_answer_in_the_doc"], passage)
        with open(self.log_path, encoding="utf-8") as f:
            self.assertEqual(json.load(f)["unrepairable_locator"], ["Q001"])

    def test_probes_are_ignored(self):
        canonical = [{
            "id": "Q001", "question": "?", "is_answerable": False, "gold_doc_ids": [],
            "gold_citations": [], "reference_answer": "Không có trong kho văn bản",
            "category": "unanswerable", "gold_passage": "", "source_urls": [],
            "doc_id_confidence": "n/a-unanswerable", "author": "Thành",
        }]
        with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f, lineterminator="\r\n")
            w.writerow(["ID", "doc_link", "question", "answer",
                        "text_contains_answer_in_the_doc", "author"])
            w.writerow(["1", "", "?", "Không có trong kho văn bản", "", "Thành"])
        with open(self.canonical_path, "w", encoding="utf-8") as f:
            json.dump(canonical, f, ensure_ascii=False)
        with open(self.corpus_path, "w", encoding="utf-8") as f:
            json.dump([], f)
        self.assertEqual(self.run_main(dry_run=False), 0)


class TestTokenRecall(unittest.TestCase):
    def test_recall_high_for_genuine_summary(self):
        self.assertGreaterEqual(mod._token_recall(SUMMARY, GOOD_ARTICLE), 0.5)

    def test_recall_low_for_disagreeing_summary(self):
        self.assertLess(
            mod._token_recall("Hoạt động thẻ ngân hàng ví điện tử", GOOD_ARTICLE),
            mod.MIN_TOKEN_RECALL,
        )


if __name__ == "__main__":
    unittest.main()
