"""Unit tests for corpus parsing, citation extraction, and evaluation agreement."""

import unittest
from regrag.corpus.parser import LegalDocumentParser
from regrag.evaluation.citation import (
    extract_citations,
    compute_citation_precision_recall,
)
from regrag.evaluation.agreement import compute_cohens_kappa


class TestComponents(unittest.TestCase):

    def test_legal_parser(self):
        sample_doc = """Chương II. HOẠT ĐỘNG PHÁT HÀNH VÀ SỬ DỤNG THẺ
Điều 14. Hạn mức giao dịch thẻ
1. Tổ chức phát hành thẻ thỏa thuận với chủ thẻ về hạn mức thanh toán.
2. Hạn mức rút ngoại tệ tiền mặt tại nước ngoài không quá tương đương 30 triệu đồng/ngày.
Điều 15. Phí dịch vụ thẻ
Tổ chức phát hành thẻ phải công khai biểu phí dịch vụ thẻ.
"""
        parser = LegalDocumentParser(doc_id="18/2024/TT-NHNN", doc_title="Quy định hoạt động thẻ ngân hàng")
        chunks = parser.parse(sample_doc)
        self.assertEqual(len(chunks), 3)

        # First chunk: Điều 14, Khoản 1
        self.assertEqual(chunks[0].article_id, "14")
        self.assertEqual(chunks[0].clause_id, "1")
        self.assertIn("Hạn mức giao dịch thẻ", chunks[0].article_title)

        # Third chunk: Điều 15, no specific clause
        self.assertEqual(chunks[2].article_id, "15")

    def test_citation_extraction(self):
        sample_response = "Theo quy định tại Điều 14 Thông tư số 18/2024/TT-NHNN, hạn mức rút ngoại tệ..."
        citations = extract_citations(sample_response)
        self.assertEqual(len(citations), 1)
        self.assertEqual(citations[0]["article_id"], "14")
        self.assertEqual(citations[0]["doc_id"], "18/2024/TT-NHNN")

        gold_citations = [{"doc_id": "18/2024/TT-NHNN", "article_id": "14"}]
        prec, rec = compute_citation_precision_recall(citations, gold_citations)
        self.assertEqual(prec, 1.0)
        self.assertEqual(rec, 1.0)

    def test_cohens_kappa(self):
        rater1 = ["valid", "valid", "unanswerable", "valid", "unanswerable"]
        rater2 = ["valid", "valid", "unanswerable", "valid", "unanswerable"]
        kappa = compute_cohens_kappa(rater1, rater2)
        self.assertEqual(kappa, 1.0)

        rater3 = ["unanswerable", "unanswerable", "valid", "unanswerable", "valid"]
        kappa_low = compute_cohens_kappa(rater1, rater3)
        self.assertLess(kappa_low, 0.5)


if __name__ == "__main__":
    unittest.main()
