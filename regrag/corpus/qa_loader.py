"""Loader from the trusted QA CSV to canonical GoldQuestion objects.

This is the single place the CSV is interpreted, so that a revised CSV (review
is still in flight) regenerates every downstream artifact identically. Nothing
else in the codebase should call csv.DictReader on the gold file.

The CSV holds two different kinds of row and they must never be conflated:

* **Answerable rows** carry a ``doc_link`` and a verbatim gold passage. On the
  2026-09-11 revision that is rows 1-65, re-measured on that revision:
    * 61/65 passages name an ``Điều``  -> article-level gold is *usually*
      derivable. The 4 exceptions (Q031, Q032, Q034, Q035) quote a ``Phụ lục``
      risk-weight annex table, which has no article header at all.
    * 20/65 passages contain "Khoản N"  -> clause-level gold is optional
    * 6/65 preserve line-initial clause numbering -> never rely on layout
    * passages rarely name their instrument -> doc_id must come from doc_link

* **Unanswerable probes** (rows 66-88 of the same revision) carry a question and
  the sentinel answer ``Không có trong kho văn bản`` with *no* ``doc_link`` and
  *no* gold passage. They are deliberate: they carry the abstention arm of RQ2.
  They are loaded as ``is_answerable=False`` / ``category="unanswerable"`` with
  an EMPTY ``gold_doc_ids`` list - not ``UNRESOLVED:empty``, because nothing
  about them is unresolved; there is simply no instrument to resolve. They are
  kept and counted, never dropped, and they are excluded from every gold-passage
  invariant (see ``coverage_report`` and ``scripts/regenerate.py``).

Anything half-way between the two (a sentinel answer that *does* carry gold, or
a row with no gold that does *not* declare itself unanswerable) is an internal
contradiction in the CSV. It is kept, classified conservatively as answerable,
and reported loudly under ``probe_anomalies`` rather than silently absorbed.
"""

import csv
import os
import re
import unicodedata
from typing import Dict, List, Optional, Tuple

from regrag.models import GoldQuestion
from regrag.corpus.canonical import (
    canonicalize_doc_id,
    extract_article_id,
    extract_article_title,
    extract_gold_citation,
    load_manifest,
    manifest_path,
    prefer_passage_named_doc,
    split_urls,
)

REQUIRED_COLUMNS = ("question", "answer", "text_contains_answer_in_the_doc", "doc_link")

# --- unanswerable probes -----------------------------------------------------

#: The answer cell the annotators use to declare "this is not in the corpus".
UNANSWERABLE_SENTINEL = "Không có trong kho văn bản"

#: ``GoldQuestion.category`` value stamped on every probe.
UNANSWERABLE_CATEGORY = "unanswerable"

#: ``doc_id_confidence`` value for a probe. A probe has no instrument, so any of
#: the resolution confidences (manifest/slug/heuristic/unresolved) would be a
#: lie; this one self-identifies instead.
CONFIDENCE_NOT_APPLICABLE = "n/a-unanswerable"

_TRAILING_PUNCT_RE = re.compile(r"[\s\.\!\?\;\:\,\-]+$")


def normalize_answer_cell(text: str) -> str:
    """Normalise an ``answer`` cell for sentinel comparison.

    NFC + whitespace collapse + casefold + trailing punctuation strip. The CSV
    comes out of Google Sheets, so the same string can arrive NFC or NFD, with a
    trailing period, or with stray spaces; none of those should change whether a
    row is a probe.
    """
    t = unicodedata.normalize("NFC", text or "")
    t = re.sub(r"\s+", " ", t).strip()
    t = _TRAILING_PUNCT_RE.sub("", t)
    return t.casefold()


def is_sentinel_answer(answer: str) -> bool:
    """True when the answer cell is the unanswerable sentinel."""
    return normalize_answer_cell(answer) == normalize_answer_cell(UNANSWERABLE_SENTINEL)


def _classify(has_link: bool, has_passage: bool, sentinel: bool) -> Tuple[bool, Optional[str]]:
    """Single source of truth for probe detection, so the raw-cell view and the
    loaded-object view can never disagree.

    Returns ``(is_probe, anomaly_reason)``.
    """
    if not has_link and not has_passage and sentinel:
        return True, None
    if sentinel and (has_link or has_passage):
        return False, "sentinel-answer-with-gold"
    if not has_link and not has_passage and not sentinel:
        return False, "no-gold-without-sentinel"
    return False, None


def classify_row(doc_link_cell: str, passage: str, answer: str) -> Tuple[bool, Optional[str]]:
    """Classify a raw CSV row as ``(is_probe, anomaly_reason)``."""
    return _classify(
        bool(split_urls(doc_link_cell or "")),
        bool((passage or "").strip()),
        is_sentinel_answer(answer),
    )


def is_unanswerable_probe(doc_link_cell: str, passage: str, answer: str) -> bool:
    """Decide whether a CSV row is an explicitly-labelled unanswerable probe.

    Detection requires ALL THREE markers to agree, so a single blank cell in an
    otherwise answerable row can never silently turn it into a probe:

    1. no ``doc_link`` - the cell yields no URL at all (``split_urls`` is used,
       so placeholder junk such as "N/A" or "-" also counts as no link);
    2. no gold passage - ``text_contains_answer_in_the_doc`` is empty;
    3. the ``answer`` cell is the sentinel ``Không có trong kho văn bản``.

    Rows that satisfy only some of these are NOT probes; see ``probe_anomaly``.
    """
    return classify_row(doc_link_cell, passage, answer)[0]


def probe_anomaly(doc_link_cell: str, passage: str, answer: str) -> Optional[str]:
    """Name the contradiction when a raw row's unanswerable markers disagree.

    Returns None when the row is self-consistent (either a full probe or a
    normal answerable row). Otherwise returns a short reason code:

    * ``"sentinel-answer-with-gold"`` - the answer says "not in the corpus" but
      the row carries a link and/or a gold passage. Kept as answerable: real
      gold beats a stale sentinel, and the disagreement is reported.
    * ``"no-gold-without-sentinel"`` - the row has neither link nor passage but
      does not declare itself unanswerable, so it cannot be scored as either.
      Kept as answerable with an UNRESOLVED doc_id and reported.
    """
    return classify_row(doc_link_cell, passage, answer)[1]


def question_anomaly(q: GoldQuestion) -> Optional[str]:
    """Same contradiction check, applied to an already-loaded GoldQuestion.

    A loaded probe (``is_answerable=False``) is self-consistent by construction.
    """
    if not q.is_answerable:
        return None
    return _classify(
        bool(q.source_urls),
        bool((q.gold_passage or "").strip()),
        is_sentinel_answer(q.reference_answer),
    )[1]


def _row_id(raw_id: str, index: int) -> str:
    """Stable question id from the CSV `ID` column, falling back to row order.

    The fallback matters because probe rows are appended with a BLANK `ID` cell.
    Mixing explicit ids with row-order fallbacks can collide (a blank-id row at
    position 2 becomes Q002, which an explicit ``ID=2`` row also produces), so
    ``load_gold_questions`` refuses a CSV with duplicate ids rather than emit two
    questions that downstream cache keys and metric joins would conflate.
    """
    raw = (raw_id or "").strip()
    if raw.isdigit():
        return f"Q{int(raw):03d}"
    if raw:
        return raw
    return f"Q{index:03d}"


def load_gold_questions(
    csv_path: str,
    manifest: Optional[Dict[str, str]] = None,
    repo_root: Optional[str] = None,
) -> List[GoldQuestion]:
    """Parse the QA CSV into canonical GoldQuestion objects.

    Every row with a non-empty question is kept, including rows that cannot be
    scored:

    * rows whose instrument cannot be resolved are retained with an
      ``UNRESOLVED:`` doc_id, so the coverage report can name them;
    * rows that declare themselves unanswerable (no link, no gold passage, the
      sentinel answer) are retained as ``is_answerable=False`` /
      ``category="unanswerable"`` with an EMPTY ``gold_doc_ids`` list.

    Nothing is dropped: silently shrinking the benchmark is the failure mode
    this loader exists to prevent.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"QA CSV not found: {csv_path}")

    if manifest is None:
        root = repo_root or os.path.abspath(
            os.path.join(os.path.dirname(csv_path), "..", "..")
        )
        manifest = load_manifest(manifest_path(root))

    questions: List[GoldQuestion] = []
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        missing = [c for c in REQUIRED_COLUMNS if c not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"QA CSV missing required column(s): {missing}")

        for i, row in enumerate(reader, 1):
            q_text = (row.get("question") or "").strip()
            if not q_text:
                continue

            raw_link = row.get("doc_link") or ""
            passage = (row.get("text_contains_answer_in_the_doc") or "").strip()
            answer = (row.get("answer") or "").strip()
            author = (row.get("author") or "").strip()
            qid = _row_id(row.get("ID", ""), i)

            # An explicitly-labelled unanswerable probe: no instrument, no gold
            # passage, sentinel answer. It carries no citation target at all, so
            # gold_doc_ids stays EMPTY - "UNRESOLVED:empty" would wrongly imply a
            # missing manifest entry, and would make the CLI demand one.
            if is_unanswerable_probe(raw_link, passage, answer):
                questions.append(
                    GoldQuestion(
                        id=qid,
                        question=q_text,
                        is_answerable=False,
                        gold_doc_ids=[],
                        gold_citations=[],
                        reference_answer=answer,
                        category=UNANSWERABLE_CATEGORY,
                        gold_passage="",
                        source_urls=[],
                        doc_id_confidence=CONFIDENCE_NOT_APPLICABLE,
                        author=author,
                    )
                )
                continue

            urls = split_urls(raw_link)

            doc_ids: List[str] = []
            confidences: List[str] = []
            for u in urls:
                doc_id, conf = canonicalize_doc_id(u, manifest)
                if doc_id not in doc_ids:
                    doc_ids.append(doc_id)
                    confidences.append(conf)

            # Primary doc_id drives the gold citation. URL order is not evidence,
            # so when the passage itself names one of the candidates, prefer that.
            named = prefer_passage_named_doc(passage, doc_ids)
            if named and doc_ids and doc_ids[0] != named:
                doc_ids = [named] + [d for d in doc_ids if d != named]
            primary = doc_ids[0] if doc_ids else "UNRESOLVED:empty"
            citation = extract_gold_citation(passage, primary)

            if named:
                confidence = "passage-named"
            elif len(doc_ids) > 1:
                confidence = "ambiguous-multi-url"
            else:
                confidence = confidences[0] if confidences else "unresolved"

            questions.append(
                GoldQuestion(
                    id=qid,
                    question=q_text,
                    is_answerable=True,
                    gold_doc_ids=doc_ids,
                    gold_citations=[citation] if citation else [],
                    reference_answer=answer,
                    category="factual",
                    gold_passage=passage,
                    source_urls=urls,
                    doc_id_confidence=confidence,
                    author=author,
                )
            )

    _assert_unique_ids(questions)
    return questions


def _assert_unique_ids(questions: List[GoldQuestion]) -> None:
    """Refuse to emit two questions under one id.

    Duplicate ids would silently corrupt per-question cache keys and every
    metric join, so this fails loudly and names the rows instead of returning
    the first match as if it were the only one.
    """
    seen: Dict[str, List[int]] = {}
    for pos, q in enumerate(questions, 1):
        seen.setdefault(q.id, []).append(pos)
    dupes = {qid: pos for qid, pos in seen.items() if len(pos) > 1}
    if dupes:
        detail = "; ".join(
            f"{qid} at loaded positions {pos}" for qid, pos in sorted(dupes.items())
        )
        raise ValueError(
            f"QA CSV produced {len(dupes)} duplicate question id(s): {detail}. "
            "A blank ID cell falls back to the CSV row number, which can collide "
            "with an explicit numeric ID. Give every row an explicit unique ID."
        )


def coverage_report(questions: List[GoldQuestion]) -> Dict[str, object]:
    """Summarise what the CSV can and cannot support, without touching a corpus.

    Unanswerable probes are counted in their own bucket and are excluded from
    every doc_id and gold-passage tally: they have no instrument and no passage,
    so folding them into ``doc_ids_unresolved`` or ``empty_passage`` would make
    a deliberate design choice look like a data defect (and would make the CLI
    fail on 23 rows that need no manifest entry).
    """
    n = len(questions)
    probes = [q for q in questions if not q.is_answerable]
    answerable = [q for q in questions if q.is_answerable]
    resolved = [q for q in answerable if q.doc_ids_resolved]
    by_conf: Dict[str, int] = {}
    for q in questions:
        by_conf[q.doc_id_confidence] = by_conf.get(q.doc_id_confidence, 0) + 1
    return {
        "questions": n,
        "answerable_questions": len(answerable),
        "unanswerable_probes": len(probes),
        "unanswerable_probe_ids": [q.id for q in probes],
        "doc_ids_resolved": len(resolved),
        "doc_ids_unresolved": len(answerable) - len(resolved),
        "by_confidence": by_conf,
        "with_article_gold": sum(
            1 for q in answerable if q.gold_citations and q.gold_citations[0].get("article_id")
        ),
        "with_clause_gold": sum(
            1 for q in answerable if q.gold_citations and q.gold_citations[0].get("clause_id")
        ),
        # Answerable rows with no gold passage. Probes are NOT counted here - a
        # probe is supposed to have no passage. A non-zero value is a real defect.
        "empty_passage": sum(1 for q in answerable if not q.gold_passage),
        "empty_passage_ids": [q.id for q in answerable if not q.gold_passage],
        "distinct_doc_ids": sorted({d for q in resolved for d in q.gold_doc_ids}),
        "unresolved_rows": [
            {"id": q.id, "urls": q.source_urls, "doc_ids": q.gold_doc_ids}
            for q in answerable
            if not q.doc_ids_resolved
        ],
        # Rows whose unanswerable markers contradict each other. Kept, never
        # dropped, and named here so the contradiction is visible.
        "probe_anomalies": [
            {"id": q.id, "reason": reason}
            for q in questions
            for reason in [question_anomaly(q)]
            if reason
        ],
    }


def load_and_report(
    csv_path: str, repo_root: Optional[str] = None
) -> Tuple[List[GoldQuestion], Dict[str, object]]:
    qs = load_gold_questions(csv_path, repo_root=repo_root)
    return qs, coverage_report(qs)
