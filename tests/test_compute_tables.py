"""Tests for scripts/compute_tables.py's pure helpers.

The script scores the campaign log with the unchanged repository scorers; the
only local logic worth pinning is the clause-accuracy match rule (which must
mirror compute_citation_precision_recall's two-pass article match) and the
F1/harmonic-mean helper.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.compute_tables import _f1, clause_match  # noqa: E402


class TestF1(unittest.TestCase):
    def test_harmonic_mean(self):
        self.assertAlmostEqual(_f1(0.5, 0.5), 0.5)
        self.assertAlmostEqual(_f1(0.2932, 0.3545), 0.3209, places=3)

    def test_zero_side_is_zero(self):
        self.assertEqual(_f1(0.0, 0.9), 0.0)
        self.assertEqual(_f1(0.9, 0.0), 0.0)


class TestClauseMatch(unittest.TestCase):
    GOLD = {"doc_id": "32/2024/QH15", "article_id": "3", "clause_id": "1"}

    def test_exact_doc_article_clause(self):
        pred = [{"doc_id": "32/2024/QH15", "article_id": "3", "clause_id": "1"}]
        self.assertTrue(clause_match(pred, self.GOLD))

    def test_right_article_wrong_clause(self):
        pred = [{"doc_id": "32/2024/QH15", "article_id": "3", "clause_id": "2"}]
        self.assertFalse(clause_match(pred, self.GOLD))

    def test_right_article_wrong_instrument(self):
        pred = [{"doc_id": "61/2025/TT-NHNN", "article_id": "3", "clause_id": "1"}]
        self.assertFalse(clause_match(pred, self.GOLD))

    def test_unknown_doc_tolerated_like_precision_recall_pass2(self):
        pred = [{"doc_id": "UNKNOWN", "article_id": "3", "clause_id": "1"}]
        self.assertTrue(clause_match(pred, self.GOLD))

    def test_document_numbering_variants_normalise(self):
        pred = [{"doc_id": "32/2024/QH15", "article_id": "03", "clause_id": "1"}]
        self.assertFalse(clause_match(pred, self.GOLD))  # article 03 != article 3

    def test_no_prediction_is_a_miss(self):
        self.assertFalse(clause_match([], self.GOLD))

    def test_gold_without_clause_needs_article_only(self):
        gold = {"doc_id": "23/2015/NĐ-CP", "article_id": "18", "clause_id": None}
        pred = [{"doc_id": "UNKNOWN", "article_id": "18", "clause_id": None}]
        self.assertTrue(clause_match(pred, gold))

    def test_gold_without_article_never_matches(self):
        gold = {"doc_id": "23/2015/NĐ-CP", "article_id": "", "clause_id": "2"}
        pred = [{"doc_id": "23/2015/NĐ-CP", "article_id": "18", "clause_id": "2"}]
        self.assertFalse(clause_match(pred, gold))


if __name__ == "__main__":
    unittest.main()
