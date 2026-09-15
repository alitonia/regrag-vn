"""Unit tests for unanswerable probes: detection, loading, reporting, exclusion.

The 2026-09-11 revision of `data/gold/bank_qa_data.csv` added 23 rows that carry
a question and the sentinel answer "Không có trong kho văn bản" but no `doc_link`
and no gold passage. These are the abstention probes RQ2 needs. They must be

  * labelled `is_answerable=False` / `category="unanswerable"`,
  * given EMPTY `gold_doc_ids` (not `UNRESOLVED:empty` - nothing is unresolved,
    there is simply no instrument),
  * KEPT and counted, never dropped,
  * excluded from the gold-passage coverage invariant and from the
    unresolved-doc_id tally, so they neither penalise nor inflate anything.

Most assertions here are *invariants* recomputed from the raw CSV rather than
hardcoded counts, so they keep holding when the annotators revise the CSV again.
The one snapshot test that does pin the 2026-09-11 numbers is labelled as such.

Run by module name (pytest is not installed, discovery fails - no __init__.py):
    .venv/bin/python -m unittest tests.test_probes
"""

import csv
import json
import os
import tempfile
import unicodedata
import unittest

from regrag.corpus.qa_loader import (
    CONFIDENCE_NOT_APPLICABLE,
    UNANSWERABLE_CATEGORY,
    UNANSWERABLE_SENTINEL,
    coverage_report,
    is_sentinel_answer,
    is_unanswerable_probe,
    load_and_report,
    load_gold_questions,
    probe_anomaly,
    question_anomaly,
)
from regrag.models import GoldQuestion
from regrag.provenance import CORPUS_TIER1
from scripts.build_tier1_corpus import build_tier1_chunks
from scripts.regenerate import check_passage_coverage

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
GOLD_CSV = os.path.join(REPO_ROOT, "data", "gold", "bank_qa_data.csv")
CSV_COLUMNS = [
    "ID",
    "doc_link",
    "question",
    "answer",
    "text_contains_answer_in_the_doc",
    "author",
]

# A real answerable row from the CSV shape: resolvable slug URL + Điều header.
ANSWERABLE_LINK = "https://congbao.chinhphu.vn/van-ban/thong-tu-so-06-2019-tt-nhnn-29358.htm"
ANSWERABLE_PASSAGE = "Điều 8. Hạn mức rút tiền mặt tại nước ngoài\n1. Mức rút tiền mặt tối đa..."


def raw_rows(csv_path=GOLD_CSV):
    """Read the gold CSV as raw dicts, independently of the loader under test."""
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(rows):
    """Write rows (list of dicts) to a temp CSV and return its path."""
    f = tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", suffix=".csv", delete=False, newline=""
    )
    writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS, extrasaction="ignore")
    writer.writeheader()
    for r in rows:
        writer.writerow({c: r.get(c, "") for c in CSV_COLUMNS})
    f.close()
    return f.name


def probe_row(qid="", question="Probe question?", sentinel=UNANSWERABLE_SENTINEL):
    return {
        "ID": qid,
        "doc_link": "",
        "question": question,
        "answer": sentinel,
        "text_contains_answer_in_the_doc": "",
        "author": "",
    }


def answerable_row(qid="1", question="Real question?"):
    return {
        "ID": qid,
        "doc_link": ANSWERABLE_LINK,
        "question": question,
        "answer": "Mức rút tiền mặt tối đa là 30 triệu đồng/ngày.",
        "text_contains_answer_in_the_doc": ANSWERABLE_PASSAGE,
        "author": "",
    }


class TestSentinelNormalisation(unittest.TestCase):
    """The sentinel comparison must survive Google Sheets' cosmetic variation."""

    def test_exact_sentinel_recognised(self):
        self.assertTrue(is_sentinel_answer(UNANSWERABLE_SENTINEL))

    def test_nfd_decomposed_sentinel_recognised(self):
        """Sheets can emit NFD; NFC/NFD must not change probe detection."""
        nfd = unicodedata.normalize("NFD", UNANSWERABLE_SENTINEL)
        self.assertNotEqual(nfd, UNANSWERABLE_SENTINEL, "fixture must actually be NFD")
        self.assertTrue(is_sentinel_answer(nfd))

    def test_cosmetic_variants_recognised(self):
        for variant in (
            f" {UNANSWERABLE_SENTINEL} ",
            f"{UNANSWERABLE_SENTINEL}.",
            UNANSWERABLE_SENTINEL.upper(),
            UNANSWERABLE_SENTINEL.replace(" ", "  "),
            unicodedata.normalize("NFD", UNANSWERABLE_SENTINEL + "."),
        ):
            self.assertTrue(is_sentinel_answer(variant), f"not recognised: {variant!r}")

    def test_near_misses_rejected(self):
        """A substring or a different phrasing must NOT be treated as the sentinel."""
        for text in (
            "",
            "Không có",
            "Không có trong kho văn bản nhưng có trong Thông tư 48/2018/TT-NHNN.",
            "Có trong kho văn bản",
            "Không rõ",
            "Khong co trong kho van ban",  # diacritics are meaningful, not optional
        ):
            self.assertFalse(is_sentinel_answer(text), f"wrongly accepted: {text!r}")


class TestProbeDetection(unittest.TestCase):
    """Detection needs ALL THREE markers; a partial match is an anomaly, not a probe."""

    def test_all_three_markers_is_a_probe(self):
        self.assertTrue(is_unanswerable_probe("", "", UNANSWERABLE_SENTINEL))
        self.assertIsNone(probe_anomaly("", "", UNANSWERABLE_SENTINEL))

    def test_placeholder_doc_link_still_counts_as_no_link(self):
        """'N/A' / '-' cells yield no URL, so the row is still a probe."""
        for junk in ("N/A", "-", "không có", "   "):
            self.assertTrue(
                is_unanswerable_probe(junk, "", UNANSWERABLE_SENTINEL),
                f"placeholder doc_link {junk!r} should count as no link",
            )

    def test_sentinel_with_doc_link_is_anomaly_not_probe(self):
        reason = probe_anomaly(ANSWERABLE_LINK, "", UNANSWERABLE_SENTINEL)
        self.assertFalse(is_unanswerable_probe(ANSWERABLE_LINK, "", UNANSWERABLE_SENTINEL))
        self.assertEqual(reason, "sentinel-answer-with-gold")

    def test_sentinel_with_gold_passage_is_anomaly_not_probe(self):
        reason = probe_anomaly("", ANSWERABLE_PASSAGE, UNANSWERABLE_SENTINEL)
        self.assertFalse(is_unanswerable_probe("", ANSWERABLE_PASSAGE, UNANSWERABLE_SENTINEL))
        self.assertEqual(reason, "sentinel-answer-with-gold")

    def test_no_gold_without_sentinel_is_anomaly_not_probe(self):
        """A row with neither link nor passage that does NOT declare itself
        unanswerable cannot be scored as either class - it must be named."""
        reason = probe_anomaly("", "", "Mức rút tiền mặt tối đa là 30 triệu đồng.")
        self.assertFalse(is_unanswerable_probe("", "", "Mức rút tiền mặt tối đa là 30 triệu đồng."))
        self.assertEqual(reason, "no-gold-without-sentinel")

    def test_normal_answerable_row_is_clean(self):
        self.assertFalse(is_unanswerable_probe(ANSWERABLE_LINK, ANSWERABLE_PASSAGE, "30 triệu"))
        self.assertIsNone(probe_anomaly(ANSWERABLE_LINK, ANSWERABLE_PASSAGE, "30 triệu"))

    def test_missing_link_but_real_passage_is_not_flagged(self):
        """Link-less rows that still carry a gold passage are scorable for the
        passage invariant; they surface via empty_passage/unresolved, not here."""
        self.assertFalse(is_unanswerable_probe("", ANSWERABLE_PASSAGE, "30 triệu"))
        self.assertIsNone(probe_anomaly("", ANSWERABLE_PASSAGE, "30 triệu"))

    def test_question_anomaly_agrees_with_row_anomaly(self):
        """The loaded-object view and the raw-cell view must never disagree."""
        cases = [
            ("", "", UNANSWERABLE_SENTINEL),
            (ANSWERABLE_LINK, "", UNANSWERABLE_SENTINEL),
            ("", "", "30 triệu đồng"),
            (ANSWERABLE_LINK, ANSWERABLE_PASSAGE, "30 triệu đồng"),
        ]
        for link, passage, answer in cases:
            q = GoldQuestion(
                id="QX",
                question="?",
                is_answerable=not is_unanswerable_probe(link, passage, answer),
                reference_answer=answer,
                gold_passage="" if is_unanswerable_probe(link, passage, answer) else passage,
                source_urls=[] if not link else [link],
            )
            self.assertEqual(
                question_anomaly(q),
                probe_anomaly(link, passage, answer),
                f"views disagree for link={link!r} passage={bool(passage)} answer={answer!r}",
            )


class TestProbeLoading(unittest.TestCase):
    """Probes must be loaded as their own class and never dropped."""

    def setUp(self):
        # Models the real CSV: explicit numeric IDs first, probes appended with a
        # blank ID cell so they fall back to their row number.
        self.path = write_csv(
            [
                answerable_row("1"),
                probe_row("", "Probe A?"),
                probe_row("", "Probe B?"),
                answerable_row("4", "Second real question?"),
            ]
        )
        self.questions = load_gold_questions(self.path, manifest={})

    def tearDown(self):
        if os.path.exists(self.path):
            os.remove(self.path)

    def test_nothing_is_dropped(self):
        self.assertEqual(len(self.questions), 4)

    def test_probes_get_their_own_class(self):
        probes = [q for q in self.questions if not q.is_answerable]
        self.assertEqual(len(probes), 2)
        for q in probes:
            self.assertEqual(q.category, UNANSWERABLE_CATEGORY)
            self.assertEqual(q.gold_doc_ids, [], "probes have no instrument, so no UNRESOLVED tag")
            self.assertEqual(q.gold_citations, [])
            self.assertEqual(q.gold_passage, "")
            self.assertEqual(q.source_urls, [])
            self.assertEqual(q.doc_id_confidence, CONFIDENCE_NOT_APPLICABLE)
            self.assertEqual(q.reference_answer, UNANSWERABLE_SENTINEL)
            self.assertTrue(q.question.strip())

    def test_probes_are_not_counted_unresolved(self):
        """The old behaviour made every probe look like a missing manifest entry."""
        report = coverage_report(load_gold_questions(self.path, manifest={}))
        self.assertEqual(report["questions"], 4)
        self.assertEqual(report["answerable_questions"], 2)
        self.assertEqual(report["unanswerable_probes"], 2)
        self.assertEqual(report["doc_ids_resolved"], 2)
        self.assertEqual(report["doc_ids_unresolved"], 0)
        self.assertEqual(report["unresolved_rows"], [])
        self.assertEqual(report["empty_passage"], 0, "probes must not count as empty-passage defects")
        self.assertEqual(report["probe_anomalies"], [])

    def test_answerable_rows_are_untouched(self):
        answerable = [q for q in self.questions if q.is_answerable]
        self.assertEqual([q.id for q in answerable], ["Q001", "Q004"])
        for q in answerable:
            self.assertEqual(q.category, "factual")
            self.assertTrue(q.gold_doc_ids)
            self.assertTrue(q.gold_passage)
            self.assertTrue(q.gold_citations, "Điều header must still yield a gold citation")
            self.assertEqual(q.gold_doc_ids[0], "06/2019/TT-NHNN")

    def test_ids_are_stable_and_unique_without_a_csv_id(self):
        """Probes have a blank ID cell; ids fall back to row order and must not
        collide with the answerable ids."""
        ids = [q.id for q in self.questions]
        self.assertEqual(len(set(ids)), len(ids), f"id collision in {ids}")
        self.assertEqual([q.id for q in self.questions if not q.is_answerable], ["Q002", "Q003"])

    def test_duplicate_ids_fail_loudly(self):
        """A blank-ID row whose fallback equals an explicit ID must raise, not
        silently produce two questions under one id."""
        path = write_csv([answerable_row("2"), probe_row("", "Probe?")])
        try:
            with self.assertRaises(ValueError) as ctx:
                load_gold_questions(path, manifest={})
            msg = str(ctx.exception)
            self.assertIn("duplicate question id", msg)
            self.assertIn("Q002", msg, "the error must name the colliding id")
        finally:
            if os.path.exists(path):
                os.remove(path)


class TestRealGoldCsvProbes(unittest.TestCase):
    """The live `data/gold/bank_qa_data.csv`, checked as invariants."""

    @classmethod
    def setUpClass(cls):
        cls.rows = raw_rows()
        cls.questions = load_gold_questions(GOLD_CSV, repo_root=REPO_ROOT)
        cls.report = coverage_report(cls.questions)
        cls.probes = [q for q in cls.questions if not q.is_answerable]
        cls.answerable = [q for q in cls.questions if q.is_answerable]

    def test_every_csv_row_with_a_question_is_loaded(self):
        expected = sum(1 for r in self.rows if (r.get("question") or "").strip())
        self.assertEqual(len(self.questions), expected, "no row may be silently dropped")

    def test_probe_set_matches_raw_cell_detection(self):
        """The loader's probe set must be exactly the rows the raw-cell rule flags."""
        expected_ids = []
        for i, r in enumerate(self.rows, 1):
            raw_id = (r.get("ID") or "").strip()
            qid = f"Q{int(raw_id):03d}" if raw_id.isdigit() else (raw_id or f"Q{i:03d}")
            if is_unanswerable_probe(
                r.get("doc_link") or "",
                (r.get("text_contains_answer_in_the_doc") or "").strip(),
                (r.get("answer") or "").strip(),
            ):
                expected_ids.append(qid)
        self.assertEqual(sorted(q.id for q in self.probes), sorted(expected_ids))
        self.assertTrue(expected_ids, "the revised CSV is expected to contain probes")

    def test_probe_fields_are_all_empty_by_design(self):
        for q in self.probes:
            self.assertEqual(q.category, UNANSWERABLE_CATEGORY)
            self.assertEqual(q.gold_doc_ids, [])
            self.assertEqual(q.gold_citations, [])
            self.assertEqual(q.gold_passage, "")
            self.assertEqual(q.source_urls, [])
            self.assertEqual(q.doc_id_confidence, CONFIDENCE_NOT_APPLICABLE)
            self.assertEqual(q.reference_answer.strip(), UNANSWERABLE_SENTINEL)
            self.assertFalse(
                any(d.startswith("UNRESOLVED") for d in q.gold_doc_ids),
                f"{q.id} must not carry an UNRESOLVED doc_id",
            )

    def test_no_answerable_row_is_a_probe(self):
        for q in self.answerable:
            self.assertEqual(q.category, "factual")
            self.assertNotEqual(q.doc_id_confidence, CONFIDENCE_NOT_APPLICABLE)

    def test_report_arithmetic_is_consistent(self):
        r = self.report
        self.assertEqual(r["answerable_questions"] + r["unanswerable_probes"], r["questions"])
        self.assertEqual(
            r["doc_ids_resolved"] + r["doc_ids_unresolved"],
            r["answerable_questions"],
            "doc_id tallies must cover the answerable rows exactly - probes excluded",
        )
        self.assertEqual(len(r["unanswerable_probe_ids"]), r["unanswerable_probes"])
        self.assertEqual(r["by_confidence"].get(CONFIDENCE_NOT_APPLICABLE, 0), r["unanswerable_probes"])

    def test_probes_are_absent_from_every_defect_list(self):
        r = self.report
        probe_ids = set(r["unanswerable_probe_ids"])
        self.assertTrue(probe_ids)
        self.assertEqual(probe_ids & set(r["empty_passage_ids"]), set())
        self.assertEqual(probe_ids & {u["id"] for u in r["unresolved_rows"]}, set())
        self.assertEqual(r["probe_anomalies"], [], "no row may carry contradictory markers")

    def test_ids_unique_across_both_classes(self):
        ids = [q.id for q in self.questions]
        self.assertEqual(len(set(ids)), len(ids))

    def test_snapshot_2026_09_11_revision(self):
        """SNAPSHOT of the 2026-09-11 CSV revision - update deliberately, not silently.

        88 rows: 64 answerable and 24 unanswerable probes. Originally 65 + 23;
        the 2026-09-11 gold correction (scripts/correct_gold_labels.py, evidence
        in data/raw_legal/TT41_FINDINGS.md) converted Q037 to a coverage-gap
        probe: its authority (TT 35/2015/TT-NHNN, biểu 118/119-TTGS) is not in
        the corpus, so scoring it as answerable against TT 41/2016 would
        manufacture a false hallucination label. Probe rows now carry explicit
        IDs 66-88 so inserting a row can no longer renumber them.
        """
        r = self.report
        self.assertEqual(r["questions"], 88)
        self.assertEqual(r["answerable_questions"], 64)
        self.assertEqual(r["unanswerable_probes"], 24)
        self.assertEqual(
            r["unanswerable_probe_ids"],
            ["Q037"] + [f"Q{i:03d}" for i in range(66, 89)],
        )
        self.assertEqual(r["empty_passage"], 0)


class TestProbeCoverageExclusion(unittest.TestCase):
    """The gold-passage invariant must skip probes and count them separately."""

    @classmethod
    def setUpClass(cls):
        cls.questions = load_gold_questions(GOLD_CSV, repo_root=REPO_ROOT)
        cls.probes = [q for q in cls.questions if not q.is_answerable]
        cls.answerable_with_passage = [
            q for q in cls.questions if q.is_answerable and q.gold_passage.strip()
        ]

    def test_probes_excluded_and_answered_separately(self):
        cov = check_passage_coverage(self.questions, [])
        self.assertEqual(cov["total_questions"], len(self.questions))
        self.assertEqual(cov["unanswerable_probes_excluded"], len(self.probes))
        self.assertEqual(
            sorted(cov["unanswerable_probe_ids"]), sorted(q.id for q in self.probes)
        )
        self.assertEqual(cov["answerable_questions"], len(self.answerable_with_passage))
        self.assertEqual(cov["answerable_without_passage"], 0)

    def test_probes_never_appear_as_missing(self):
        """A probe has no passage, so it must never be reported as not-found."""
        cov = check_passage_coverage(self.questions, [])
        probe_ids = {q.id for q in self.probes}
        self.assertEqual(probe_ids & set(cov["missing_question_ids"]), set())
        self.assertEqual(probe_ids & set(cov["loose_question_ids"]), set())
        self.assertEqual(cov["not_found"], len(self.answerable_with_passage))

    def test_coverage_denominator_is_answerable_not_total(self):
        """Percentages are over the answerable rows; 23 probes must not dilute them."""
        chunks = [
            {"chunk_id": f"C_{q.id}", "text": q.gold_passage, "corpus_source": CORPUS_TIER1}
            for q in self.answerable_with_passage
        ]
        cov = check_passage_coverage(self.questions, chunks)
        self.assertEqual(cov["strict_matches"], len(self.answerable_with_passage))
        self.assertEqual(cov["not_found"], 0)
        self.assertEqual(cov["strict_coverage_pct"], 100.0)
        self.assertEqual(cov["loose_coverage_pct"], 0.0)

    def test_probes_excluded_from_live_corpus_coverage(self):
        """Probes stay out of every coverage bucket, whatever the corpus holds.

        The corpus file is a live artifact, so no fixed match count is asserted; the
        property under test is that probes are excluded and the answerable rows
        partition exactly.
        """
        path = os.path.join(REPO_ROOT, "data", "processed_chunks", "corpus_chunks.json")
        if not os.path.exists(path):
            self.skipTest(f"{path} not present")
        with open(path, "r", encoding="utf-8") as f:
            chunks = json.load(f)
        cov = check_passage_coverage(self.questions, chunks)
        self.assertEqual(
            cov["strict_matches"] + cov["squash_matches"] + cov["loose_matches"] + cov["not_found"],
            cov["answerable_questions"],
        )
        self.assertEqual(
            cov["unanswerable_probes_excluded"],
            len(self.questions) - cov["answerable_questions"],
        )
        self.assertEqual(len(cov["unanswerable_probe_ids"]),
                         cov["unanswerable_probes_excluded"])

    def test_answerable_row_without_passage_is_not_hidden_in_the_probe_bucket(self):
        """An answerable row with an empty passage is a defect and must be named."""
        qs = [
            GoldQuestion(id="Q_A", question="?", is_answerable=True, gold_passage=""),
            GoldQuestion(id="Q_P", question="?", is_answerable=False, gold_passage=""),
            GoldQuestion(id="Q_B", question="?", is_answerable=True, gold_passage="Điều 1. Phạm vi\n1. Quy định."),
        ]
        cov = check_passage_coverage(qs, [{"chunk_id": "C1", "text": "Điều 1. Phạm vi\n1. Quy định."}])
        self.assertEqual(cov["answerable_without_passage_ids"], ["Q_A"])
        self.assertEqual(cov["unanswerable_probe_ids"], ["Q_P"])
        self.assertEqual(cov["answerable_questions"], 1)
        self.assertEqual(cov["strict_matches"], 1)

    def test_empty_question_set_does_not_divide_by_zero(self):
        cov = check_passage_coverage([], [])
        self.assertEqual(cov["answerable_questions"], 0)
        self.assertEqual(cov["strict_coverage_pct"], 0.0)
        self.assertEqual(cov["loose_coverage_pct"], 0.0)
        self.assertEqual(cov["total_coverage_pct"], 0.0)


class TestProbeTier1Exclusion(unittest.TestCase):
    """Tier 1 is one chunk per gold passage, so probes contribute no chunk."""

    @classmethod
    def setUpClass(cls):
        cls.questions = load_gold_questions(GOLD_CSV, repo_root=REPO_ROOT)
        cls.probes = [q for q in cls.questions if not q.is_answerable]
        cls.chunks, cls.stats = build_tier1_chunks(cls.questions)

    def test_no_empty_text_chunk_is_built(self):
        self.assertTrue(self.chunks)
        for c in self.chunks:
            self.assertTrue(c.text.strip(), f"{c.chunk_id} has empty text")

    def test_probe_count_matches_exclusion_report(self):
        self.assertEqual(self.stats["probes_excluded"], len(self.probes))
        self.assertEqual(
            sorted(self.stats["probe_ids_excluded"]), sorted(q.id for q in self.probes)
        )
        self.assertEqual(self.stats["questions_in"], len(self.questions))
        self.assertEqual(self.stats["empty_passage_excluded"], 0)
        self.assertEqual(
            self.stats["chunks_built"], len(self.questions) - len(self.probes)
        )

    def test_no_chunk_carries_a_probe_question_id(self):
        probe_ids = {q.id for q in self.probes}
        built_ids = {c.metadata.get("question_id") for c in self.chunks}
        self.assertEqual(probe_ids & built_ids, set())

    def test_all_chunks_stamped_tier1(self):
        for c in self.chunks:
            self.assertEqual(c.corpus_source, CORPUS_TIER1)

    def test_chunk_ids_unique(self):
        ids = [c.chunk_id for c in self.chunks]
        self.assertEqual(len(set(ids)), len(ids))


class TestProbeSyntheticCsvEndToEnd(unittest.TestCase):
    """A synthetic probe-only CSV must not crash the reporting path."""

    def test_probe_only_csv(self):
        path = write_csv([probe_row("", "P1?"), probe_row("", "P2?")])
        try:
            questions, report = load_and_report(path, repo_root=REPO_ROOT)
            self.assertEqual(len(questions), 2)
            self.assertEqual(report["answerable_questions"], 0)
            self.assertEqual(report["unanswerable_probes"], 2)
            self.assertEqual(report["doc_ids_unresolved"], 0)
            self.assertEqual(report["distinct_doc_ids"], [])
            self.assertEqual(report["with_article_gold"], 0)
            cov = check_passage_coverage(questions, [])
            self.assertEqual(cov["answerable_questions"], 0)
            self.assertEqual(cov["unanswerable_probes_excluded"], 2)
            self.assertEqual(cov["strict_coverage_pct"], 0.0)
            chunks, stats = build_tier1_chunks(questions)
            self.assertEqual(chunks, [])
            self.assertEqual(stats["chunks_built"], 0)
            self.assertEqual(stats["probes_excluded"], 2)
        finally:
            if os.path.exists(path):
                os.remove(path)


if __name__ == "__main__":
    unittest.main()
