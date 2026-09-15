"""Unit tests for the RegRAG-VN lexical scorers in regrag/evaluation/metrics.py.

Run with:  .venv/bin/python -m unittest tests.test_metrics

These tests exist to prove five regressions are fixed, in this order:

  1. a wrong answer carrying the RIGHT citation must not get full credit
     (the placeholder derived correctness entirely from citation P/R);
  2. a fluent fabrication carrying NO citation must be flagged as a
     hallucination (the placeholder flagged hallucination only when a citation
     WAS emitted, so the fabrication scored better than a wrong citation);
  3. a correct refusal on an unanswerable probe must be rewarded;
  4. a refusal on an ANSWERABLE question must be penalised as a false abstention;
  5. a wrong numeric threshold behind a correct Điều must be caught.

``LegacyPlaceholderScorer`` is a verbatim transcription of the scorer this module
replaced (git HEAD, regrag/evaluation/metrics.py). It is kept here ONLY so the
before/after difference is asserted rather than described.
"""

import dataclasses
import os
import unittest

from regrag.evaluation import metrics as M
from regrag.evaluation.citation import (
    compute_citation_precision_recall,
    extract_citations,
)
from regrag.generation.config import UNANSWERABLE_SENTINEL
from regrag.generation.prompts import ABSTENTION_KEYPHRASE, build_rag_prompt
from regrag.models import (
    EvaluationRecord,
    GenerationResult,
    GoldQuestion,
    LegalChunk,
    RetrievedResult,
)
from regrag.provenance import (
    CORPUS_TIER1,
    CORPUS_TIER2,
    CORPUS_UNSET,
    ProvenanceError,
    assert_publishable,
    degraded,
)

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
GOLD_CSV = os.path.join(REPO_ROOT, "data", "gold", "bank_qa_data.csv")

BACKEND_BM25 = "bm25-rank_bm25+pyvi"
BACKEND_CLOSED = "none-closed-book"

# --- fixtures ----------------------------------------------------------------

ARTICLE_14_CHUNK_TEXT = (
    "Hạn mức rút ngoại tệ tiền mặt tại nước ngoài không quá tương đương "
    "30 triệu đồng/ngày."
)
ARTICLE_14_CHUNK_HEADER = "[18/2024/TT-NHNN] Điều 14. Hạn mức giao dịch thẻ (Khoản 2)"
UNRELATED_CHUNK_TEXT = (
    "Tổ chức tín dụng phải ban hành quy định nội bộ về phòng chống rửa tiền "
    "và báo cáo Ngân hàng Nhà nước theo định kỳ hàng quý."
)

GOLD_ANSWER = (
    "Hạn mức rút ngoại tệ tiền mặt tại nước ngoài cho một thẻ tối đa tương đương "
    "30 triệu đồng Việt Nam trong một ngày."
)

CORRECT_CITED = (
    "Theo Điều 14 Thông tư 18/2024/TT-NHNN, hạn mức rút ngoại tệ tiền mặt tại "
    "nước ngoài không quá tương đương 30 triệu đồng/ngày."
)
WRONG_NUMBER_RIGHT_ARTICLE = (
    "Theo Điều 14 Thông tư 18/2024/TT-NHNN, hạn mức rút ngoại tệ tiền mặt tại "
    "nước ngoài không quá tương đương 50 triệu đồng/ngày."
)
WRONG_ANSWER_RIGHT_CITATION = (
    "Theo Điều 14 Thông tư 18/2024/TT-NHNN, khách hàng phải nộp thuế thu nhập "
    "cá nhân khi rút tiền mặt tại quầy giao dịch của ngân hàng."
)
FABRICATION_NO_CITATION = (
    "Khách hàng có thể rút tối đa 50 triệu đồng mỗi ngày tại bất kỳ máy ATM nào "
    "trên toàn quốc mà không cần đăng ký trước."
)
RIGHT_CONTENT_WRONG_CITATION = (
    "Hạn mức rút ngoại tệ tiền mặt tại nước ngoài không quá tương đương "
    "30 triệu đồng/ngày, theo Điều 99 Nghị định 21/2021/NĐ-CP."
)
HEDGED_THEN_ANSWERED = (
    "Tài liệu không đề cập đến vấn đề này. Tuy nhiên, hạn mức rút ngoại tệ "
    "tiền mặt là 50 triệu đồng/ngày."
)
SUBSTANTIVE_WITH_NEGATION = (
    "Công ty không thể mua ngoại tệ để thanh toán cho việc hoàn trả tiền ứng "
    "trước tiền công tác phí của chuyên gia nước ngoài."
)


def make_gold(**over) -> GoldQuestion:
    base = dict(
        id="Q001",
        question="Hạn mức rút ngoại tệ tiền mặt tại nước ngoài đối với một thẻ là bao nhiêu?",
        is_answerable=True,
        gold_doc_ids=["18/2024/TT-NHNN"],
        gold_citations=[{"doc_id": "18/2024/TT-NHNN", "article_id": "14", "clause_id": "2"}],
        reference_answer=GOLD_ANSWER,
        category="factual",
        gold_passage=(
            "Điều 14. Hạn mức giao dịch thẻ\n"
            "2. Hạn mức rút ngoại tệ tiền mặt tại nước ngoài không quá tương đương "
            "30 triệu đồng/ngày."
        ),
        source_urls=["https://congbao.chinhphu.vn/van-ban/thong-tu-18-2024-tt-nhnn.htm"],
        doc_id_confidence="slug",
    )
    base.update(over)
    return GoldQuestion(**base)


def make_probe(**over) -> GoldQuestion:
    """An unanswerable probe as the loader now emits it."""
    base = dict(
        id="Q090",
        question="Thời hạn cấp phép cho hoạt động phát hành tiền mã hoá là bao nhiêu ngày?",
        is_answerable=False,
        gold_doc_ids=[],
        gold_citations=[],
        reference_answer=UNANSWERABLE_SENTINEL,
        category="unanswerable",
        gold_passage="",
        source_urls=[],
        doc_id_confidence="n/a-unanswerable",
    )
    base.update(over)
    return GoldQuestion(**base)


def make_chunk(text=ARTICLE_14_CHUNK_TEXT, article_id="14", doc_id="18/2024/TT-NHNN",
               title="Hạn mức giao dịch thẻ", corpus_source=CORPUS_TIER2) -> LegalChunk:
    return LegalChunk(
        chunk_id=f"{doc_id}#{article_id}",
        doc_id=doc_id,
        doc_title="Quy định hoạt động thẻ ngân hàng",
        chapter="Chương II",
        article_id=article_id,
        article_title=title,
        clause_id="2",
        text=text,
        corpus_source=corpus_source,
    )


def make_gen(answer_text, mode="rag_bm25", prompt="", model="qwen-7b",
             corpus=CORPUS_TIER2, backend=BACKEND_BM25, qid="Q001") -> GenerationResult:
    return GenerationResult(
        question_id=qid,
        model_name=model,
        retrieval_mode=mode,
        prompt=prompt,
        raw_response=answer_text,
        answer_text=answer_text,
        corpus_source=corpus,
        retriever_backend=backend,
    )


def rag_prompt(chunks) -> str:
    return build_rag_prompt(
        "Hạn mức rút ngoại tệ?",
        [RetrievedResult(chunk=c, score=1.0 / (i + 1), rank=i + 1,
                         retriever_backend=BACKEND_BM25) for i, c in enumerate(chunks)],
    )


def score(answer_text, gold=None, contexts=None, retrieved=None, mode="rag_bm25",
          prompt="", backend=BACKEND_BM25, corpus=CORPUS_TIER2, validity=None):
    gold = gold if gold is not None else make_gold()
    gen = make_gen(answer_text, mode=mode, prompt=prompt, backend=backend, corpus=corpus)
    return M.score_response(gen, gold, retrieved=retrieved, contexts=contexts,
                            gold_citation_validity=validity)


def record(answer_text, **kw) -> EvaluationRecord:
    return M.evaluate_response(make_gen(answer_text, mode=kw.pop("mode", "rag_bm25"),
                                        prompt=kw.pop("prompt", ""),
                                        backend=kw.pop("backend", BACKEND_BM25),
                                        corpus=kw.pop("corpus", CORPUS_TIER2)),
                               kw.pop("gold", None) or make_gold(),
                               contexts=kw.pop("contexts", None),
                               retrieved=kw.pop("retrieved", None),
                               gold_citation_validity=kw.pop("validity", None))


def legacy_placeholder(answer_text: str, gold: GoldQuestion) -> EvaluationRecord:
    """Verbatim transcription of the PRE-FIX scorer (git HEAD metrics.py).

    Kept only so the tests can assert the direction of the fix instead of
    describing it in a comment. Not used by any production code path.
    """
    text = answer_text.strip()
    abstained = (
        ABSTENTION_KEYPHRASE.lower() in text.lower()
        or "không có trong tài liệu" in text.lower()
        or "không tìm thấy thông tin" in text.lower()
    )
    if not gold.is_answerable:
        abstained_correctly = abstained
        hallucinated = not abstained
        correctness = 2.0 if abstained_correctly else 0.0
        prec, rec = (1.0, 1.0) if abstained else (0.0, 0.0)
    else:
        abstained_correctly = False
        extracted = extract_citations(text)
        prec, rec = compute_citation_precision_recall(extracted, gold.gold_citations)
        hallucinated = False
        if abstained:
            correctness = 0.0
        else:
            if rec > 0.0:
                correctness = 2.0 if prec >= 0.5 else 1.0
            else:
                correctness = 0.5
                hallucinated = True if len(extracted) > 0 else False
    return EvaluationRecord(
        question_id=gold.id, model_name="legacy", retrieval_mode="rag_bm25",
        is_answerable=gold.is_answerable, citation_precision=prec, citation_recall=rec,
        abstained=abstained, abstained_correctly=abstained_correctly,
        correctness_score=correctness, hallucinated=hallucinated,
        metric_status="placeholder",
        placeholder_fields=["correctness_score", "hallucinated", "abstained",
                            "abstained_correctly"],
    )


# --- 1. correctness must read the answer, not only the citation --------------

class TestCorrectnessReadsTheAnswer(unittest.TestCase):

    def test_wrong_answer_with_right_citation_does_not_get_full_credit(self):
        """REGRESSION 1. The placeholder scored this 2.0 on citation recall alone."""
        gold = make_gold()
        old = legacy_placeholder(WRONG_ANSWER_RIGHT_CITATION, gold)
        self.assertEqual(old.citation_recall, 1.0)
        self.assertEqual(old.correctness_score, 2.0, "guard: the old bug must be present")

        new = record(WRONG_ANSWER_RIGHT_CITATION, gold=gold,
                     contexts=[make_chunk().formatted_context()])
        self.assertEqual(new.citation_recall, 1.0, "citation is still perfect")
        self.assertEqual(new.citation_precision, 1.0)
        self.assertLess(new.correctness_score, 2.0)
        self.assertEqual(new.correctness_score, 0.0)
        self.assertTrue(new.hallucinated)
        self.assertIn(M.H_UNSUPPORTED_BY_GOLD, new.notes)

    def test_correct_answer_gets_full_credit(self):
        rec = record(CORRECT_CITED, contexts=[make_chunk().formatted_context()])
        self.assertEqual(rec.correctness_score, 2.0)
        self.assertFalse(rec.hallucinated)

    def test_citation_match_can_never_raise_correctness(self):
        """A perfect citation with zero content agreement is still 0.0."""
        rec = record(WRONG_ANSWER_RIGHT_CITATION, contexts=[make_chunk().formatted_context()])
        self.assertEqual(rec.citation_precision, 1.0)
        self.assertEqual(rec.citation_recall, 1.0)
        self.assertEqual(rec.correctness_score, 0.0)

    def test_right_content_wrong_citation_is_capped_at_partial(self):
        s = score(RIGHT_CONTENT_WRONG_CITATION, contexts=[make_chunk().formatted_context()])
        self.assertEqual(s.correctness_score, 1.0)
        self.assertIn(M.H_WRONG_INSTRUMENT, s.hallucination_types)
        self.assertIn(M.H_NON_COVERING_PROVISION, s.hallucination_types)

    def test_empty_gold_answer_is_unassessable_not_zero(self):
        """A blank gold cell must not be averaged in as a plausible 0.0."""
        s = score(CORRECT_CITED, gold=make_gold(reference_answer=""))
        self.assertFalse(s.correctness.assessable)
        self.assertIn("correct=NA(gold_answer_empty)", s.notes)
        self.assertIn("ANOMALY=answerable_without_gold_answer", s.notes)
        rec = dataclasses.replace(
            record(CORRECT_CITED, gold=make_gold(reference_answer="")),
            metric_status=M.METRIC_STATUS_FINAL)
        group = M.summarise_records([rec])["groups"]["qwen-7b::rag_bm25"]
        self.assertIsNone(group["avg_correctness"])
        self.assertEqual(group["correctness_unassessable_rows"], 1)
        self.assertEqual(group["answerability_anomalies"], 1)

    def test_correct_answer_without_a_citation_is_not_a_hallucination(self):
        """UNCITED_ASSERTION is for fabrications, not for citation style.

        Without this, every closed-book row that omits "Điều N" would be counted as
        a hallucination and RQ1 would measure citation style, not unfaithfulness.
        """
        uncited_but_correct = (
            "Hạn mức rút ngoại tệ tiền mặt tại nước ngoài tối đa tương đương "
            "30 triệu đồng Việt Nam trong một ngày."
        )
        s = score(uncited_but_correct, mode="closed_book", backend=BACKEND_CLOSED)
        self.assertEqual(s.correctness_score, 2.0)
        self.assertNotIn(M.H_UNCITED_ASSERTION, s.hallucination_types)
        self.assertFalse(s.hallucinated)
        self.assertFalse(s.citation_emitted)
        self.assertIn("no_citation_emitted=1", s.notes)

    def test_incorrect_answer_without_a_citation_is_a_hallucination(self):
        s = score(FABRICATION_NO_CITATION, mode="closed_book", backend=BACKEND_CLOSED)
        self.assertIn(M.H_UNCITED_ASSERTION, s.hallucination_types)
        self.assertEqual(s.hallucination_severity, 3)


# --- 2. hallucination logic must not be inverted -----------------------------

class TestHallucinationIsNotInverted(unittest.TestCase):

    def test_fabrication_without_citation_is_flagged(self):
        """REGRESSION 2. The placeholder left hallucinated=False here."""
        gold = make_gold()
        old = legacy_placeholder(FABRICATION_NO_CITATION, gold)
        self.assertFalse(old.hallucinated, "guard: the old inversion must be present")
        self.assertEqual(old.correctness_score, 0.5)

        new = record(FABRICATION_NO_CITATION, gold=gold, mode="closed_book",
                     backend=BACKEND_CLOSED)
        self.assertTrue(new.hallucinated)
        self.assertIn(M.H_UNCITED_ASSERTION, new.notes)
        self.assertEqual(new.correctness_score, 0.0)

    def test_no_citation_scores_strictly_worse_than_a_wrong_citation(self):
        """Same fabricated content: citing the wrong Điều beats citing nothing."""
        ctx = [make_chunk().formatted_context()]
        no_cite = score(FABRICATION_NO_CITATION, contexts=ctx)
        wrong_cite = score(RIGHT_CONTENT_WRONG_CITATION, contexts=ctx)

        self.assertTrue(no_cite.hallucinated)
        self.assertTrue(wrong_cite.hallucinated)
        self.assertLess(no_cite.correctness_score, wrong_cite.correctness_score)
        self.assertGreaterEqual(no_cite.hallucination_severity,
                                wrong_cite.hallucination_severity)
        self.assertIn(M.H_UNCITED_ASSERTION, no_cite.hallucination_types)
        self.assertNotIn(M.H_UNCITED_ASSERTION, wrong_cite.hallucination_types)

    def test_taxonomy_distinguishes_the_four_required_kinds(self):
        ctx = [make_chunk().formatted_context()]
        # (a) unsupported claim: fluent, unrelated to the retrieved chunk
        unsupported = score(
            "Ngân hàng phải công khai biểu phí dịch vụ thẻ cho khách hàng biết "
            "trước khi phát hành thẻ tín dụng quốc tế.", contexts=ctx)
        self.assertIn(M.H_UNSUPPORTED_CLAIM, unsupported.hallucination_types)

        # (b) contradicts the corpus: a figure the chunk contradicts
        contradicts = score(WRONG_NUMBER_RIGHT_ARTICLE, contexts=ctx)
        self.assertIn(M.H_CONTRADICTS_CONTEXT, contradicts.hallucination_types)

        # (c) cited the wrong instrument
        wrong_instrument = score(RIGHT_CONTENT_WRONG_CITATION, contexts=ctx)
        self.assertIn(M.H_WRONG_INSTRUMENT, wrong_instrument.hallucination_types)

        # (d) cited a provision that does not cover the question (right circular,
        #     wrong Điều)
        wrong_article = score(
            "Theo Điều 23 Thông tư 18/2024/TT-NHNN, hạn mức rút ngoại tệ tiền mặt "
            "tại nước ngoài không quá tương đương 30 triệu đồng/ngày.", contexts=ctx)
        self.assertIn(M.H_NON_COVERING_PROVISION, wrong_article.hallucination_types)
        self.assertNotIn(M.H_WRONG_INSTRUMENT, wrong_article.hallucination_types)

    def test_empty_answer_is_degenerate_not_a_clean_refusal(self):
        s = score("   ", contexts=[make_chunk().formatted_context()])
        self.assertIn(M.H_DEGENERATE_OUTPUT, s.hallucination_types)
        self.assertFalse(s.abstention.abstained)
        self.assertEqual(s.correctness_score, 0.0)
        self.assertNotIn(M.H_UNCITED_ASSERTION, s.hallucination_types)


# --- 3. numeric thresholds ---------------------------------------------------

class TestNumericThresholds(unittest.TestCase):

    def test_wrong_number_behind_the_right_article_is_hallucination(self):
        """REGRESSION 5. 'Right Điều, wrong figure' must be caught."""
        rec = record(WRONG_NUMBER_RIGHT_ARTICLE,
                     contexts=[make_chunk().formatted_context()])
        self.assertEqual(rec.citation_recall, 1.0, "the article really is right")
        self.assertEqual(rec.citation_precision, 1.0)
        self.assertTrue(rec.hallucinated)
        self.assertIn(M.H_NUMERIC_THRESHOLD_MISMATCH, rec.notes)
        self.assertEqual(rec.correctness_score, 0.0)

    def test_percentage_threshold_is_caught(self):
        gold = make_gold(
            id="Q040",
            question="Tỷ lệ an toàn vốn tối thiểu là bao nhiêu?",
            reference_answer="Tổ chức tín dụng phải duy trì tỷ lệ an toàn vốn tối thiểu 8%.",
            gold_citations=[{"doc_id": "22/2023/TT-NHNN", "article_id": "9"}],
            gold_doc_ids=["22/2023/TT-NHNN"],
            gold_passage="Điều 9: Tỷ lệ an toàn vốn tối thiểu 8%.",
        )
        right = score("Tỷ lệ an toàn vốn tối thiểu phải duy trì là 8%.", gold=gold)
        wrong = score("Tỷ lệ an toàn vốn tối thiểu phải duy trì là 14%.", gold=gold)
        self.assertFalse(right.hallucinated)
        self.assertEqual(right.correctness_score, 2.0)
        self.assertTrue(wrong.hallucinated)
        self.assertIn(M.H_NUMERIC_THRESHOLD_MISMATCH, wrong.hallucination_types)
        self.assertEqual(wrong.correctness_score, 0.0)

    def test_money_units_are_rescaled_before_comparison(self):
        a = M.extract_numeric_facts("2 tỷ đồng")
        b = M.extract_numeric_facts("2000 triệu đồng")
        self.assertEqual({f.canonical for f in a}, {f.canonical for f in b})
        self.assertFalse(M.compare_numeric_facts(a, b).has_conflict)

    def test_legal_references_are_not_read_as_quantities(self):
        facts = M.extract_numeric_facts(
            "Theo Điều 14 Khoản 2 Thông tư 18/2024/TT-NHNN ngày 01/01/2025, "
            "hạn mức là 30 triệu đồng/ngày.")
        self.assertEqual({f.canonical for f in facts}, {(30000000.0, "VND")})

    def test_false_friend_units_are_not_invented(self):
        self.assertEqual(set(), {f for f in M.extract_numeric_facts("đồng thời phải báo cáo") if f.family})
        facts = M.extract_numeric_facts("tỷ lệ an toàn vốn tối thiểu 8%")
        self.assertEqual({f.canonical for f in facts}, {(8.0, "%")})

    def test_unit_narrowing_is_reported_not_silently_accepted(self):
        """'30 ngày' for a gold '30 ngày làm việc' is a real legal difference."""
        gold_facts = M.extract_numeric_facts("trong thời hạn 30 ngày làm việc")
        ans_facts = M.extract_numeric_facts("trong thời hạn 30 ngày")
        cmp = M.compare_numeric_facts(gold_facts, ans_facts)
        self.assertFalse(cmp.has_conflict, "same quantity, so not a value conflict")
        self.assertTrue(cmp.unit_mismatches, "but the unit changed and is reported")

    def test_unrecognised_unit_never_produces_a_conflict_claim(self):
        gold_facts = M.extract_numeric_facts("mức phạt 7 con trâu")
        ans_facts = M.extract_numeric_facts("mức phạt 9 con trâu")
        self.assertFalse(M.compare_numeric_facts(gold_facts, ans_facts).has_conflict)


# --- 4. abstention -----------------------------------------------------------

class TestAbstentionDetection(unittest.TestCase):

    def test_benchmark_defined_strings_are_abstentions(self):
        for text in (UNANSWERABLE_SENTINEL, ABSTENTION_KEYPHRASE):
            d = M.detect_abstention(text)
            self.assertTrue(d.abstained, text)
            self.assertEqual(d.tier, "exact")

    def test_vietnamese_paraphrases_are_abstentions(self):
        paraphrases = [
            "Các tài liệu được cung cấp không đề cập đến vấn đề này.",
            "Không tìm thấy thông tin về nội dung này trong văn bản.",
            "Văn bản pháp luật hiện hành không quy định về vấn đề này.",
            "Tôi không thể trả lời câu hỏi này dựa trên tài liệu được cung cấp.",
            "Câu hỏi không nằm trong phạm vi tài liệu.",
            "Không có thông tin về vấn đề này trong ngữ cảnh được cung cấp.",
            "Không đủ cơ sở để trả lời câu hỏi này.",
        ]
        for text in paraphrases:
            self.assertTrue(M.detect_abstention(text).abstained, text)

    def test_english_refusal_is_detected(self):
        self.assertTrue(M.detect_abstention(
            "I cannot answer this from the provided context.").abstained)
        self.assertTrue(M.detect_abstention(
            "The information is not provided in the given documents.").abstained)

    def test_substantive_answers_are_not_mistaken_for_refusals(self):
        """Two REAL gold answers from the CSV that contain negation."""
        for text in (
            SUBSTANTIVE_WITH_NEGATION,
            "Người từ đủ 15 tuổi trở lên, không bị hạn chế hoặc mất năng lực hành "
            "vi dân sự mới được mở tài khoản thanh toán tại ngân hàng.",
            "Việc thanh toán giá trị chuyển nhượng phần vốn góp KHÔNG thực hiện "
            "thông qua tài khoản vốn đầu tư trực tiếp.",
        ):
            self.assertFalse(M.detect_abstention(text).abstained, text)

    def test_hedged_then_answered_is_not_an_abstention(self):
        d = M.detect_abstention(HEDGED_THEN_ANSWERED)
        self.assertFalse(d.abstained)
        self.assertTrue(d.hedged_then_answered)
        self.assertEqual(d.reason, "hedged_then_answered")
        s = score(HEDGED_THEN_ANSWERED, contexts=[make_chunk().formatted_context()])
        self.assertTrue(s.hallucinated)
        self.assertIn("HEDGED_THEN_ANSWERED=1", s.notes)

    def test_refusal_that_names_the_article_it_checked_is_still_a_refusal(self):
        d = M.detect_abstention(
            "Không có quy định về vấn đề này tại Điều 14 Thông tư 18/2024/TT-NHNN.")
        self.assertTrue(d.abstained)

    def test_paraphrased_refusal_that_states_a_figure_is_not_a_refusal(self):
        d = M.detect_abstention(
            "Tài liệu không đề cập đến hạn mức, tuy nhiên mức tối đa là 50 triệu đồng.")
        self.assertFalse(d.abstained)
        self.assertTrue(d.states_figure)
        self.assertIn("soft_with_figures", d.reason)

    def test_empty_answer_is_not_an_abstention(self):
        d = M.detect_abstention("")
        self.assertFalse(d.abstained)
        self.assertTrue(d.empty_answer)

    def test_legacy_three_keyphrase_detector_is_superseded(self):
        """The placeholder missed every paraphrase below; the new one does not."""
        for text in ("Các tài liệu được cung cấp không đề cập đến vấn đề này.",
                     "Không đủ cơ sở để trả lời câu hỏi này."):
            old_hit = (ABSTENTION_KEYPHRASE.lower() in text.lower()
                       or "không có trong tài liệu" in text.lower()
                       or "không tìm thấy thông tin" in text.lower())
            self.assertFalse(old_hit, "guard: the old detector really did miss it")
            self.assertTrue(M.detect_abstention(text).abstained)


class TestAbstentionScoring(unittest.TestCase):

    def test_correct_refusal_on_a_probe_is_rewarded(self):
        """REGRESSION 3."""
        probe = make_probe()
        for text in (UNANSWERABLE_SENTINEL, ABSTENTION_KEYPHRASE,
                     "Các tài liệu được cung cấp không đề cập đến vấn đề này."):
            rec = record(text, gold=probe, mode="closed_book", backend=BACKEND_CLOSED)
            self.assertTrue(rec.abstained, text)
            self.assertTrue(rec.abstained_correctly, text)
            self.assertEqual(rec.correctness_score, 2.0, text)
            self.assertFalse(rec.hallucinated, text)

    def test_false_abstention_on_an_answerable_question_is_penalised(self):
        """REGRESSION 4."""
        rec = record(ABSTENTION_KEYPHRASE, contexts=[make_chunk().formatted_context()])
        self.assertTrue(rec.abstained)
        self.assertFalse(rec.abstained_correctly)
        self.assertEqual(rec.correctness_score, 0.0)
        self.assertIn("FALSE_ABSTENTION=1", rec.notes)
        # A refusal is a recall failure, not a fabrication: it must not be
        # counted in the hallucination rate that RQ1 reports.
        self.assertFalse(rec.hallucinated)

    def test_answered_probe_is_hallucination_with_spurious_citation(self):
        rec = record("Theo Điều 5 Thông tư 08/2023/TT-NHNN, doanh nghiệp được phép "
                     "phát hành tiền mã hóa trong 30 ngày.",
                     gold=make_probe(), mode="closed_book", backend=BACKEND_CLOSED)
        self.assertFalse(rec.abstained)
        self.assertFalse(rec.abstained_correctly)
        self.assertTrue(rec.hallucinated)
        self.assertIn(M.H_ANSWERED_UNANSWERABLE, rec.notes)
        self.assertIn(M.H_SPURIOUS_CITATION, rec.notes)
        self.assertEqual(rec.correctness_score, 0.0)


# --- 5. groundedness ---------------------------------------------------------

class TestGroundedness(unittest.TestCase):

    def test_supported_answer_scores_high_against_the_retrieved_chunk(self):
        rep = M.score_groundedness(CORRECT_CITED, [make_chunk().formatted_context()])
        self.assertTrue(rep.assessable)
        self.assertGreaterEqual(rep.groundedness, M.SUPPORT_CONTAINMENT)
        self.assertEqual(rep.n_supported, 1)
        self.assertEqual(rep.n_unsupported, 0)

    def test_unrelated_answer_scores_low_against_the_retrieved_chunk(self):
        rep = M.score_groundedness(
            "Ngân hàng phải công khai biểu phí dịch vụ thẻ tín dụng quốc tế cho "
            "khách hàng trước khi phát hành.",
            [make_chunk().formatted_context()])
        self.assertTrue(rep.assessable)
        self.assertLess(rep.groundedness, M.SUPPORT_CONTAINMENT)
        self.assertGreaterEqual(rep.n_unsupported + rep.n_partial, 1)

    def test_numeric_conflict_with_the_context_is_a_contradiction(self):
        rep = M.score_groundedness(WRONG_NUMBER_RIGHT_ARTICLE,
                                  [make_chunk().formatted_context()])
        self.assertEqual(rep.n_conflict, 1)
        self.assertIn("NUMERIC_CONFLICT", rep.flags)

    def test_citation_absent_from_the_context_is_flagged(self):
        rep = M.score_groundedness(
            "Theo Điều 23 Thông tư 18/2024/TT-NHNN, hạn mức rút ngoại tệ tiền mặt "
            "tại nước ngoài không quá tương đương 30 triệu đồng/ngày.",
            [make_chunk().formatted_context()])
        self.assertIn("REF_NOT_IN_CONTEXT", rep.flags)

    def test_groundedness_is_unassessable_without_context_not_zero(self):
        """No retrieved context => None + NA in notes. Never a plausible 0.0."""
        s = score(CORRECT_CITED, mode="closed_book", backend=BACKEND_CLOSED, contexts=[])
        self.assertFalse(s.groundedness.assessable)
        self.assertIsNone(s.groundedness.groundedness)
        self.assertIn("grounded=NA(no_retrieved_context)", s.notes)

    def test_groundedness_reference_is_the_retrieved_chunk_not_the_gold_passage(self):
        """The 46-row defect: gold_passage is an annotator summary, not source text.

        The answer quotes the DOCUMENT's wording, so it disagrees with the summary
        cell. Groundedness must stay high because the retrieved chunk - not the
        summary - is the reference.
        """
        summary_cell = "Điều 14: Khách hàng chỉ được rút tối đa ba mươi triệu đồng " \
                       "mỗi ngày khi đang ở nước ngoài."
        gold = make_gold(gold_passage=summary_cell)
        s = score(CORRECT_CITED, gold=gold, contexts=[make_chunk().formatted_context()])

        self.assertEqual(s.passage_quality.shape, M.PASSAGE_SHAPE_SUMMARY)
        self.assertIsNotNone(s.passage_quality.bigram_overlap)
        self.assertLess(s.passage_quality.bigram_overlap, M.VERBATIM_OVERLAP,
                        "the summary cell is not the document's wording")
        self.assertGreaterEqual(s.groundedness.groundedness, M.SUPPORT_CONTAINMENT,
                                "groundedness is measured against the chunk")
        self.assertNotIn(M.H_UNSUPPORTED_CLAIM, s.hallucination_types)

    def test_prompt_recovered_context_is_used_when_no_retrieval_is_passed(self):
        prompt = rag_prompt([make_chunk()])
        self.assertEqual(len(M.context_from_prompt(prompt)), 1)
        s = score(CORRECT_CITED, prompt=prompt)
        self.assertEqual(s.context_source, "prompt_recovered")
        self.assertTrue(s.groundedness.assessable)

    def test_retrieved_results_are_used_when_passed(self):
        rr = [RetrievedResult(chunk=make_chunk(), score=1.0, rank=1,
                              retriever_backend=BACKEND_BM25)]
        s = score(CORRECT_CITED, retrieved=rr)
        self.assertEqual(s.context_source, "retrieved_results")
        self.assertTrue(s.groundedness.assessable)

    def test_closed_book_prompt_yields_no_context(self):
        self.assertEqual(M.context_from_prompt("Câu hỏi: X\nTrả lời:"), [])


# --- 6. gold_passage data-quality flag ---------------------------------------

class TestPassageQualityFlag(unittest.TestCase):

    def test_shapes_are_distinguished(self):
        self.assertEqual(M.assess_passage_quality("", []).shape, M.PASSAGE_SHAPE_EMPTY)
        self.assertEqual(
            M.assess_passage_quality("Điều 11 Khoản 1: Hồ sơ gồm giấy đề nghị mở "
                                     "tài khoản và giấy tờ tùy thân còn hiệu lực.", []).shape,
            M.PASSAGE_SHAPE_SUMMARY)
        self.assertEqual(
            M.assess_passage_quality("Điều 14. Hạn mức giao dịch thẻ\n" + "x " * 300,
                                     []).shape,
            M.PASSAGE_SHAPE_EXCERPT)

    def test_overlap_is_measured_only_when_context_exists(self):
        cell = "Điều 14. Hạn mức giao dịch thẻ 2. Hạn mức rút ngoại tệ tiền mặt " \
               "tại nước ngoài không quá tương đương 30 triệu đồng/ngày."
        unmeasured = M.assess_passage_quality(cell, [])
        self.assertIsNone(unmeasured.bigram_overlap)
        self.assertEqual(unmeasured.overlap_class, M.OVERLAP_UNMEASURED)

        measured = M.assess_passage_quality(cell, [make_chunk().formatted_context()])
        self.assertIsNotNone(measured.bigram_overlap)
        self.assertEqual(measured.overlap_class, M.OVERLAP_HIGH)
        self.assertEqual(measured.basis, "measured_vs_retrieved_context")

    def test_stratum_appears_in_the_record_notes(self):
        rec = record(CORRECT_CITED, contexts=[make_chunk().formatted_context()])
        notes = M.parse_notes(rec.notes)
        self.assertIn("gold_passage", notes)
        self.assertIn("passage_overlap", notes)
        self.assertTrue(notes["passage_overlap"].endswith("/high"))


# --- 7. answerability resolution ---------------------------------------------

class TestAnswerabilityResolution(unittest.TestCase):

    def test_loader_flagged_probe(self):
        self.assertEqual(M.resolve_answerability(make_probe())[:2], (False, "gold_flag"))

    def test_legacy_probe_row_is_still_detected(self):
        """The loader may not have flagged the 23 probes yet; the scorer must cope."""
        legacy = make_probe(is_answerable=True, category="factual")
        answerable, basis, _ = M.resolve_answerability(legacy)
        self.assertFalse(answerable)
        self.assertEqual(basis, "sentinel_answer")

    def test_row_with_no_gold_evidence_at_all_is_flagged_not_turned_into_a_probe(self):
        """Mirrors qa_loader's "no-gold-without-sentinel" anomaly.

        Calling it a probe would reward a model for refusing; calling it answerable
        without saying so would score it against nothing. It stays answerable, is
        tagged, and every scorer reports itself unassessable for that row.
        """
        bare = GoldQuestion(id="Q091", question="Q?", is_answerable=True,
                            reference_answer="", category="factual", gold_passage="",
                            gold_doc_ids=[], gold_citations=[], source_urls=[])
        answerable, basis, anomaly = M.resolve_answerability(bare)
        self.assertTrue(answerable)
        self.assertEqual(basis, "gold_flag")
        self.assertEqual(anomaly, "answerable_without_any_gold_evidence")
        s = score("Doanh nghiệp phải mở tài khoản vốn đầu tư trực tiếp tại ngân "
                  "hàng được phép tại Việt Nam.", gold=bare)
        self.assertIn("ANOMALY=answerable_without_any_gold_evidence", s.notes)
        self.assertIn("correct=NA(gold_answer_empty)", s.notes)
        self.assertFalse(s.citation_scorable)

    def test_sentinel_answer_with_gold_is_an_anomaly_not_a_probe(self):
        """A row that HAS an instrument and a passage but whose answer cell is the
        sentinel must not become a probe: that would reward refusing a real question."""
        odd = make_gold(reference_answer=UNANSWERABLE_SENTINEL)
        answerable, basis, anomaly = M.resolve_answerability(odd)
        self.assertTrue(answerable)
        self.assertEqual(anomaly, "sentinel_answer_with_gold")
        s = score(CORRECT_CITED, gold=odd, contexts=[make_chunk().formatted_context()])
        self.assertIn("correct=NA(gold_answer_is_sentinel)", s.notes)

    def test_answerable_row_without_a_passage_is_flagged_not_reclassified(self):
        odd = make_gold(gold_passage="")
        answerable, _basis, anomaly = M.resolve_answerability(odd)
        self.assertTrue(answerable)
        self.assertEqual(anomaly, "answerable_without_gold_passage")
        self.assertIn("ANOMALY=answerable_without_gold_passage",
                      score(CORRECT_CITED, gold=odd).notes)


# --- 8. citation-gold robustness ---------------------------------------------

class TestCitationGoldRobustness(unittest.TestCase):

    def test_unresolved_gold_instrument_suppresses_wrong_instrument(self):
        """Plan §3.4: an unresolved instrument must not generate a false label."""
        gold = make_gold(gold_doc_ids=["UNRESOLVED:manifest-pending"],
                         gold_citations=[{"doc_id": "UNRESOLVED:manifest-pending",
                                          "article_id": "14"}],
                         doc_id_confidence="unresolved")
        s = score("Theo Điều 99 Thông tư 18/2024/TT-NHNN, hạn mức là 30 triệu đồng.",
                  gold=gold, contexts=[make_chunk().formatted_context()])
        self.assertNotIn(M.H_WRONG_INSTRUMENT, s.hallucination_types,
                         "the gold instrument is unknown, so no instrument can be 'wrong'")
        self.assertIn(M.H_NON_COVERING_PROVISION, s.hallucination_types,
                      "article-level gold is still usable")
        self.assertIn("gold_doc_unresolved=1", s.notes)

    def test_gold_citation_declared_invalid_suppresses_citation_judgement(self):
        gold = make_gold()
        s = score("Theo Điều 99 Thông tư 18/2024/TT-NHNN, hạn mức là 30 triệu đồng.",
                  gold=gold, contexts=[make_chunk().formatted_context()],
                  validity="invalid")
        self.assertFalse(s.citation_scorable)
        self.assertEqual(s.citation_unscorable_reason, "gold_citation_invalid")
        self.assertNotIn(M.H_WRONG_INSTRUMENT, s.hallucination_types)
        self.assertNotIn(M.H_NON_COVERING_PROVISION, s.hallucination_types)
        self.assertIn("citation_scorable=0:gold_citation_invalid", s.notes)

    def test_answerable_row_without_gold_citation_is_unscorable(self):
        s = score(CORRECT_CITED, gold=make_gold(gold_citations=[]))
        self.assertFalse(s.citation_scorable)
        self.assertEqual(s.citation_unscorable_reason, "no_gold_citation")

    def test_probe_citations_are_not_averaged_into_recall(self):
        """citation.py returns recall=1.0 for an empty gold set; that artefact must
        not enter the citation-recall column."""
        rec = record("Theo Điều 5 Thông tư 08/2023/TT-NHNN, được phép.",
                     gold=make_probe(), mode="closed_book", backend=BACKEND_CLOSED)
        self.assertIn("citation_scorable=0:probe_no_gold_citation", rec.notes)
        summary = M.summarise_records([rec], strict=False)
        group = summary["groups"]["qwen-7b::closed_book"]
        self.assertIsNone(group["citation_recall"])
        self.assertEqual(group["citation_scorable_rows"], 0)
        self.assertEqual(group["citation_unscorable_rows"], 1)


# --- 9. provenance discipline ------------------------------------------------

class TestProvenanceDiscipline(unittest.TestCase):

    def test_records_self_identify_as_unvalidated_never_final(self):
        rec = record(CORRECT_CITED, contexts=[make_chunk().formatted_context()])
        self.assertEqual(rec.metric_status, M.METRIC_STATUS_UNVALIDATED)
        self.assertNotEqual(rec.metric_status, M.METRIC_STATUS_FINAL)
        self.assertNotEqual(rec.metric_status, "placeholder",
                            "the scorers are implemented; 'placeholder' would now be false")
        for f in ("correctness_score", "hallucinated", "abstained", "abstained_correctly"):
            self.assertIn(f, rec.placeholder_fields)
        self.assertIn("status=unvalidated", rec.notes)
        self.assertIn(f"scorer={M.SCORER_NAME}", rec.notes)

    def test_corpus_and_backend_provenance_survive_scoring(self):
        rec = record(CORRECT_CITED, contexts=[make_chunk().formatted_context()],
                     corpus=CORPUS_TIER2, backend=BACKEND_BM25)
        self.assertEqual(rec.corpus_source, CORPUS_TIER2)
        self.assertEqual(rec.retriever_backend, BACKEND_BM25)

    def test_assert_publishable_still_refuses_tier1_and_degraded(self):
        """The pre-existing corpus/retriever guard must be untouched and effective."""
        tier1 = record(CORRECT_CITED, corpus=CORPUS_TIER1)
        with self.assertRaises(ProvenanceError) as cm:
            assert_publishable([tier1])
        self.assertIn("unpublishable provenance", str(cm.exception))

        deg = record(CORRECT_CITED, backend=degraded("no_rank_bm25"))
        with self.assertRaises(ProvenanceError):
            assert_publishable([deg])

        unset = record(CORRECT_CITED, corpus=CORPUS_UNSET, backend=CORPUS_UNSET)
        with self.assertRaises(ProvenanceError):
            assert_publishable([unset])

    def test_unvalidated_rows_are_refused_by_aggregation(self):
        """Clean corpus + clean backend is NOT enough: the scorer is unvalidated."""
        recs = [record(CORRECT_CITED, contexts=[make_chunk().formatted_context()])]
        self.assertEqual(assert_publishable(recs), [],
                         "guard: provenance.py itself does not look at metric_status")
        with self.assertRaises(ProvenanceError) as cm:
            M.assert_metrics_publishable(recs)
        self.assertIn("UNVALIDATED", str(cm.exception))
        with self.assertRaises(ProvenanceError):
            M.compute_abstention_metrics(recs)
        with self.assertRaises(ProvenanceError):
            M.summarise_records(recs)

    def test_non_strict_aggregation_is_labelled_not_silent(self):
        recs = [record(CORRECT_CITED, contexts=[make_chunk().formatted_context()])]
        out = M.compute_abstention_metrics(recs, strict=False)
        self.assertEqual(out["provenance"], "REFUSED_UNVALIDATED")
        self.assertEqual(M.summarise_records(recs, strict=False)["provenance"],
                         "REFUSED_UNVALIDATED")

    def test_final_status_is_required_before_aggregation(self):
        recs = [dataclasses.replace(
            record(CORRECT_CITED, contexts=[make_chunk().formatted_context()]),
            metric_status=M.METRIC_STATUS_FINAL)]
        self.assertEqual(M.assert_metrics_publishable(recs), None)
        out = M.compute_abstention_metrics(recs)
        self.assertNotIn("provenance", out)
        self.assertEqual(out["rows"], 1)


# --- 10. aggregation ---------------------------------------------------------

class TestAggregation(unittest.TestCase):

    def _mixed_records(self):
        probe = make_probe()
        ctx = [make_chunk().formatted_context()]
        recs = [
            record(CORRECT_CITED, contexts=ctx),                                   # TN (answer, answerable)
            record(ABSTENTION_KEYPHRASE, contexts=ctx),                            # FP false abstention
            record(FABRICATION_NO_CITATION, contexts=ctx),                          # TN + hallucination
            record(UNANSWERABLE_SENTINEL, gold=probe, mode="closed_book",
                   backend=BACKEND_CLOSED, corpus=CORPUS_TIER2),                    # TP correct refusal
            record("Theo Điều 5 Thông tư 08/2023/TT-NHNN, được phép phát hành.",
                   gold=probe, mode="closed_book", backend=BACKEND_CLOSED,
                   corpus=CORPUS_TIER2),                                            # FN answered probe
        ]
        return [dataclasses.replace(r, metric_status=M.METRIC_STATUS_FINAL) for r in recs]

    def test_abstention_confusion_matrix(self):
        out = M.compute_abstention_metrics(self._mixed_records())
        self.assertEqual(out["rows"], 5)
        self.assertEqual(out["probes"], 2)
        self.assertEqual(out["answerable"], 3)
        self.assertEqual(out["true_positive"], 1)
        self.assertEqual(out["false_abstention"], 1)
        self.assertEqual(out["true_negative"], 2)
        self.assertEqual(out["answered_probe"], 1)
        self.assertEqual(out["abstention_accuracy"], round(3 / 5, 4))
        self.assertEqual(out["false_abstention_rate"], round(1 / 3, 4))
        self.assertEqual(out["probe_abstention_rate"], 0.5)
        self.assertEqual(out["abstention_precision"], 0.5)
        self.assertEqual(out["abstention_recall"], 0.5)

    def test_summary_reports_denominators_and_never_fabricates_a_zero(self):
        out = M.summarise_records(self._mixed_records())
        closed = out["groups"]["qwen-7b::closed_book"]
        rag = out["groups"]["qwen-7b::rag_bm25"]
        # closed-book rows have no retrieved context: groundedness is None, not 0.0
        self.assertIsNone(closed["groundedness_mean"])
        self.assertEqual(closed["groundedness_assessable_rows"], 0)
        # Of the 3 RAG rows one is a bare refusal, which has no content sentence to
        # ground. It is excluded (NA), not counted as groundedness 0.0 - which is
        # why the denominator is 2 and the mean stays 1.0-ish rather than 0.67.
        self.assertEqual(rag["count"], 3)
        self.assertEqual(rag["groundedness_assessable_rows"], 2)
        self.assertIsNotNone(rag["groundedness_mean"])
        refusal = [r for r in self._mixed_records() if r.abstained and r.is_answerable]
        self.assertEqual(len(refusal), 1)
        self.assertIn("grounded=NA(no_content_sentence)", refusal[0].notes)

    def test_summary_stratifies_by_gold_passage_quality(self):
        summary_cell = "Điều 14: Khách hàng chỉ được rút tối đa ba mươi triệu đồng mỗi ngày."
        recs = [
            record(CORRECT_CITED, contexts=[make_chunk().formatted_context()]),
            record(CORRECT_CITED, gold=make_gold(gold_passage=summary_cell),
                   contexts=[make_chunk().formatted_context()]),
        ]
        recs = [dataclasses.replace(r, metric_status=M.METRIC_STATUS_FINAL) for r in recs]
        group = M.summarise_records(recs)["groups"]["qwen-7b::rag_bm25"]
        strata = group["groundedness_by_gold_passage_stratum"]
        self.assertEqual(len(group["gold_passage_strata"]), 2,
                         "summary-shaped and excerpt-shaped rows are not pooled")
        self.assertEqual(sum(v["rows"] for v in strata.values()), 2)


# --- 11. human validation ----------------------------------------------------

class TestValidationReport(unittest.TestCase):

    def _pairs(self, labels):
        return [
            {"question_id": f"Q{i:03d}",
             "human_correct": h[0], "auto_correct": a[0],
             "human_hallucinated": h[1], "auto_hallucinated": a[1],
             "human_abstained": h[2], "auto_abstained": a[2]}
            for i, (h, a) in enumerate(labels)
        ]

    def test_perfect_agreement(self):
        labels = [((True, False, False), (True, False, False)),
                  ((False, True, False), (False, True, False)),
                  ((False, False, True), (False, False, True)),
                  ((True, True, False), (True, True, False))]
        out = M.validation_report(self._pairs(labels))
        self.assertEqual(out["n"], 4)
        for f in ("correct", "hallucinated", "abstained"):
            self.assertEqual(out["fields"][f]["raw_agreement"], 1.0)
            self.assertEqual(out["fields"][f]["cohens_kappa"], 1.0)

    def test_disagreement_is_reported_with_direction_and_ids(self):
        labels = [((True, False, False), (True, False, False)),
                  ((False, True, False), (True, False, False)),   # auto over-credits
                  ((True, False, False), (False, True, False)),   # auto under-credits
                  ((False, False, True), (False, False, True))]
        out = M.validation_report(self._pairs(labels))["fields"]["correct"]
        self.assertEqual(out["raw_agreement"], 0.5)
        self.assertLess(out["cohens_kappa"], 0.5)
        self.assertEqual(out["confusion"]["human_no_auto_yes"], 1)
        self.assertEqual(out["confusion"]["human_yes_auto_no"], 1)
        self.assertEqual(sorted(out["disagreements"]), ["Q001", "Q002"])

    def test_incomplete_labels_raise(self):
        pairs = [{"question_id": "Q001", "human_correct": True, "auto_correct": True,
                  "human_hallucinated": False, "auto_hallucinated": False,
                  "human_abstained": False},
                 {"question_id": "Q002", "human_correct": False, "auto_correct": False,
                  "human_hallucinated": True, "auto_hallucinated": True,
                  "human_abstained": False, "auto_abstained": False}]
        with self.assertRaises(ProvenanceError) as cm:
            M.validation_report(pairs)
        self.assertIn("auto_abstained", str(cm.exception))

    def test_too_few_pairs_raise(self):
        with self.assertRaises(ProvenanceError):
            M.validation_report([{"question_id": "Q1", "human_correct": True,
                                  "auto_correct": True, "human_hallucinated": False,
                                  "auto_hallucinated": False, "human_abstained": False,
                                  "auto_abstained": False}])


# --- 12. determinism, notes round-trip, cross-module guard -------------------

class TestDeterminismAndIntegration(unittest.TestCase):

    def test_scoring_is_deterministic(self):
        ctx = [make_chunk().formatted_context()]
        a = score(CORRECT_CITED, contexts=ctx).to_dict()
        b = score(CORRECT_CITED, contexts=ctx).to_dict()
        self.assertEqual(a, b)

    def test_notes_round_trip(self):
        rec = record(WRONG_NUMBER_RIGHT_ARTICLE, contexts=[make_chunk().formatted_context()])
        notes = M.parse_notes(rec.notes)
        self.assertEqual(notes["scorer"], M.SCORER_NAME)
        self.assertEqual(notes["status"], "unvalidated")
        self.assertIn("NUMERIC_THRESHOLD_MISMATCH", notes["hallucination"])
        self.assertTrue(notes["correct"].startswith("0.0("))
        self.assertTrue(notes["grounded"].startswith("0.00(") or
                        notes["grounded"].startswith("NA("))

    def test_to_dict_is_json_serialisable(self):
        import json
        d = score(CORRECT_CITED, contexts=[make_chunk().formatted_context()]).to_dict()
        json.dumps(d, ensure_ascii=False)

    def test_abstention_module_compat_guard_passes(self):
        """The cross-module guard owned by regrag/generation/abstention.py."""
        from regrag.generation.abstention import assert_metrics_detects_sentinel
        assert_metrics_detects_sentinel(gold_is_answerable=False)
        assert_metrics_detects_sentinel(gold_is_answerable=True)

    def test_two_argument_call_still_works(self):
        """scripts/run_eval.py calls evaluate_response(gen, gold) with no retrieval."""
        rec = M.evaluate_response(make_gen(CORRECT_CITED, mode="closed_book",
                                           backend=BACKEND_CLOSED), make_gold())
        self.assertIsInstance(rec, EvaluationRecord)
        self.assertIn("grounded=NA(no_retrieved_context)", rec.notes)


# --- 13. the trusted CSV ----------------------------------------------------

@unittest.skipUnless(os.path.exists(GOLD_CSV), "trusted gold CSV not present")
class TestAgainstTrustedCsv(unittest.TestCase):
    """Data-driven checks on data/gold/bank_qa_data.csv (88 rows).

    Reads the CSV directly rather than through regrag.corpus.qa_loader, which is
    being edited concurrently; the only assumption is the column layout.
    """

    @classmethod
    def setUpClass(cls):
        import csv
        with open(GOLD_CSV, "r", encoding="utf-8-sig", newline="") as f:
            cls.rows = [r for r in csv.DictReader(f) if (r.get("question") or "").strip()]
        cls.answerable = [r for r in cls.rows
                          if (r.get("answer") or "").strip() != UNANSWERABLE_SENTINEL]
        cls.probes = [r for r in cls.rows
                      if (r.get("answer") or "").strip() == UNANSWERABLE_SENTINEL]

    def _gold(self, row, i):
        return GoldQuestion(
            id=row.get("ID") or f"Q{i:03d}",
            question=(row.get("question") or "").strip(),
            is_answerable=True,
            reference_answer=(row.get("answer") or "").strip(),
            gold_passage=(row.get("text_contains_answer_in_the_doc") or "").strip(),
            category="factual",
        )

    def test_row_counts_match_the_documented_benchmark(self):
        # 64 answerable + 24 probes since the 2026-09-11 gold correction
        # (Q037 converted to a coverage-gap probe; see
        # scripts/correct_gold_labels.py and TT41_FINDINGS.md).
        self.assertEqual(len(self.rows), 88)
        self.assertEqual(len(self.probes), 24)
        self.assertEqual(len(self.answerable), 64)

    def test_every_probe_resolves_as_unanswerable_from_raw_cells(self):
        """The scorer must not depend on the loader having flagged the probes."""
        for i, row in enumerate(self.probes, 1):
            gold = self._gold(row, i)
            answerable, basis, _ = M.resolve_answerability(gold)
            self.assertFalse(answerable, row["question"][:60])
            self.assertEqual(basis, "sentinel_answer")

    def test_gold_answer_scores_full_credit_against_itself(self):
        """A model that reproduces the gold answer verbatim must score 2.0."""
        for i, row in enumerate(self.answerable, 1):
            gold = self._gold(row, i)
            s = M.score_response(make_gen(gold.reference_answer, mode="closed_book",
                                          backend=BACKEND_CLOSED), gold)
            self.assertEqual(s.correctness_score, 2.0, f"row {i}: {gold.question[:60]}")
            self.assertFalse(s.hallucinated, f"row {i}: {s.hallucination_types}")

    def test_cross_matched_gold_answers_are_not_full_credit(self):
        """Wrong-answer control: row i's answer against row j's gold.

        Not a clean negative - the CSV holds sibling questions on the same topic -
        so the assertion is on the large majority, and the measured rate is printed
        for the paper's calibration paragraph.
        """
        full = 0
        total = 0
        for i in range(0, len(self.answerable), 5):
            j = (i + 7) % len(self.answerable)
            if i == j:
                continue
            gold = self._gold(self.answerable[j], j + 1)
            ans = (self.answerable[i].get("answer") or "").strip()
            s = M.score_response(make_gen(ans, mode="closed_book", backend=BACKEND_CLOSED),
                                 gold)
            total += 1
            if s.correctness_score == 2.0:
                full += 1
        self.assertLess(full / total, 0.10,
                        f"{full}/{total} cross-matched pairs scored full credit")

    def test_sentinel_refusal_is_credited_on_every_probe(self):
        for i, row in enumerate(self.probes, 1):
            gold = self._gold(row, i)
            s = M.score_response(make_gen(UNANSWERABLE_SENTINEL, mode="closed_book",
                                          backend=BACKEND_CLOSED), gold)
            self.assertTrue(s.abstention.abstained)
            self.assertTrue(s.abstained_correctly)
            self.assertEqual(s.correctness_score, 2.0)
            self.assertFalse(s.hallucinated)


if __name__ == "__main__":
    unittest.main()
