"""Benchmark question loading for the campaign, with loud unanswerable-probe derivation.

The upstream gap this module covers
-----------------------------------
The trusted CSV holds 88 rows: 65 answerable questions and 23 unanswerable probes
whose gold answer is the sentinel ``Không có trong kho văn bản`` and which carry
no ``doc_link`` and no gold passage. Those probes are what RQ2 (abstention) is
measured on, so if they enter the campaign as answerable a *correct* refusal is
scored as a failure and RQ2 comes out inverted with no warning.

``regrag/corpus/qa_loader.py`` now classifies them itself
(``_classify`` -> ``is_answerable=False`` / ``category="unanswerable"``, landed
2026-09-11). It did not when this harness was written - it hardcoded
``is_answerable=True`` for every row - and it is outside this harness's write
scope either way. So the derivation is re-done here, on the objects the loader
returns, and it is loud: every reclassified and anomalous id is listed and
written into the run manifest, making the derivation auditable rather than
implicit.

Deliberately defensive in both directions
-----------------------------------------
The loader is being changed concurrently to emit the probes itself. This module
must be correct whether or not that lands, so it never assumes the flag is wrong
and never assumes it is right - it derives from the row's own evidence and
reports the basis.

The rule mirrors ``regrag/evaluation/metrics._derive_answerability`` exactly, so
the harness and the scorer cannot disagree about which rows are probes. A
divergence would put ``is_answerable`` in the row metadata and a different value
in the scored record. Per that scorer's reasoning, the sentinel alone is NOT
enough: a row that has a gold passage and an instrument but whose answer cell
says "not in the corpus" is a data defect, and calling it a probe would reward a
model for refusing a question that has gold. It stays answerable and is flagged.

Unstable-id warning
-------------------
The 23 probe rows have an empty ``ID`` cell, so the loader falls back to
``Q{row_index:03d}``. Those ids hold only while the probe rows stay at the bottom
of the CSV in the same order. Inserting a row above them renumbers them, and
because cache keys bind to ``question_id``, that invalidates every cached probe
generation. Warned about explicitly.
"""

from __future__ import annotations

import csv
import sys
from dataclasses import dataclass, field, replace
from typing import Any, Dict, List, Optional, Sequence, Tuple

from regrag.corpus.canonical import normalize_ws
from regrag.corpus.qa_loader import coverage_report, load_gold_questions
from regrag.generation.config import UNANSWERABLE_SENTINEL
from regrag.models import GoldQuestion
from regrag.provenance import ProvenanceError

_SENTINEL = normalize_ws(UNANSWERABLE_SENTINEL).rstrip(".")


def _has_gold_evidence(q: GoldQuestion) -> bool:
    """True when the row carries any gold the model could be scored against."""
    return bool(
        (q.gold_passage or "").strip()
        or q.gold_doc_ids
        or q.gold_citations
        or q.source_urls
    )


def classify_answerability(q: GoldQuestion) -> Tuple[bool, str, str]:
    """Decide whether a gold row is answerable, and say on what evidence.

    Returns ``(is_answerable, basis, anomaly)``. ``anomaly`` is non-empty
    whenever the row's own evidence contradicts itself; such rows are reported
    for deliberate exclusion rather than dropped silently, because silently
    shrinking the benchmark is the failure this repo's design exists to prevent.
    """
    # The loader already classified it - trust the explicit flag.
    if not q.is_answerable:
        return False, "loader_flag", ""

    if q.category == "unanswerable":
        return (
            False,
            "loader_category",
            "category_unanswerable_with_gold" if _has_gold_evidence(q) else "",
        )

    ref = normalize_ws(q.reference_answer or "").rstrip(".")
    is_sentinel = bool(ref) and ref == _SENTINEL
    has_evidence = _has_gold_evidence(q)

    if is_sentinel and not has_evidence:
        return False, "sentinel_no_gold", ""

    if is_sentinel and has_evidence:
        # DATA DEFECT: says "not in the corpus" but has a gold passage/instrument.
        return True, "sentinel_with_gold", "sentinel_answer_but_has_gold"

    if not has_evidence:
        # Answerable-looking row with no gold at all. Most likely a question
        # whose source document never arrived. Plan §3.4: such a question must be
        # EXCLUDED, never left in, or it generates a false hallucination label.
        # Kept here and flagged, not dropped.
        return True, "no_evidence", "answerable_without_any_gold_evidence"

    return True, "gold_present", ""


@dataclass
class BenchmarkSet:
    """The loaded, classified question set plus the audit trail of doing so."""

    questions: List[GoldQuestion]
    csv_path: str
    csv_hash: str
    classification: Dict[str, Dict[str, str]] = field(default_factory=dict)
    reclassified_ids: List[str] = field(default_factory=list)
    anomaly_ids: Dict[str, List[str]] = field(default_factory=dict)
    unstable_ids: List[str] = field(default_factory=list)
    coverage: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)

    @property
    def answerable(self) -> List[GoldQuestion]:
        return [q for q in self.questions if q.is_answerable]

    @property
    def probes(self) -> List[GoldQuestion]:
        return [q for q in self.questions if not q.is_answerable]

    def by_id(self) -> Dict[str, GoldQuestion]:
        return {q.id: q for q in self.questions}

    def basis_counts(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for info in self.classification.values():
            counts[info["basis"]] = counts.get(info["basis"], 0) + 1
        return counts

    def manifest(self) -> Dict[str, Any]:
        return {
            "csv_path": self.csv_path,
            "csv_hash": self.csv_hash,
            "questions": len(self.questions),
            "answerable": len(self.answerable),
            "unanswerable_probes": len(self.probes),
            "probe_ids": [q.id for q in self.probes],
            "classification_basis_counts": self.basis_counts(),
            "reclassified_ids": self.reclassified_ids,
            "anomalies": self.anomaly_ids,
            "unstable_ids_derived_from_row_order": self.unstable_ids,
            "coverage_report": {
                k: self.coverage.get(k)
                for k in (
                    "questions",
                    "doc_ids_resolved",
                    "doc_ids_unresolved",
                    "with_article_gold",
                    "with_clause_gold",
                    "empty_passage",
                )
            },
            "warnings": self.warnings,
        }


def load_benchmark(
    csv_path: str,
    repo_root: Optional[str] = None,
    warn_stream=None,
) -> BenchmarkSet:
    """Load the trusted CSV, derive answerability, and report everything loudly."""
    stream = warn_stream if warn_stream is not None else sys.stderr
    warnings: List[str] = []

    from regrag.generation.corpus_io import file_sha256

    csv_hash = file_sha256(csv_path)
    questions = load_gold_questions(csv_path, repo_root=repo_root)
    if not questions:
        raise ProvenanceError(
            f"{csv_path} produced 0 questions. Refusing to run a campaign on an "
            "empty benchmark."
        )

    classification: Dict[str, Dict[str, str]] = {}
    reclassified: List[str] = []
    anomalies: Dict[str, List[str]] = {}
    out: List[GoldQuestion] = []

    for q in questions:
        answerable, basis, anomaly = classify_answerability(q)
        classification[q.id] = {"basis": basis, "anomaly": anomaly}
        if answerable != q.is_answerable:
            reclassified.append(q.id)
        if anomaly:
            anomalies.setdefault(anomaly, []).append(q.id)
        out.append(replace(q, is_answerable=answerable,
                           category="unanswerable" if not answerable else q.category))

    ids = [q.id for q in out]
    dupes = sorted({i for i in ids if ids.count(i) > 1})
    if dupes:
        raise ProvenanceError(
            f"Duplicate question id(s) {dupes} from {csv_path}. Question ids are the "
            "join key for cache keys, checkpoints and evaluation records; duplicates "
            "would silently overwrite rows."
        )

    n_probes = sum(1 for q in out if not q.is_answerable)
    n_answerable = len(out) - n_probes

    if n_probes == 0:
        msg = (
            "NO unanswerable probes were detected in the CSV. Plan §3.4 requires them "
            "for RQ2 (abstention) and 24 rows are expected as of 2026-09-11 (23 "
            "original probes + Q037, converted to a coverage-gap probe because its "
            "authority, TT 35/2015/TT-NHNN, is not in the corpus). Either the "
            "sentinel string changed or the probes are missing - do not report an "
            "abstention measurement from this set."
        )
        warnings.append(msg)
        stream.write(f"[WARNING] {msg}\n")
        stream.flush()

    for anomaly, qids in sorted(anomalies.items()):
        msg = (
            f"{len(qids)} row(s) carry the anomaly {anomaly!r}: {qids[:12]}"
            f"{' ...' if len(qids) > 12 else ''}. These are kept and reported, never "
            "dropped. Decide deliberately whether to exclude them before reporting."
        )
        warnings.append(msg)
        stream.write(f"[WARNING] {msg}\n")
        stream.flush()

    # Probe ids are only "unstable" while their ID cell is blank: a blank cell
    # makes the loader derive Q<row-number>, which a row insertion silently
    # renumbers (cache keys bind to question ids). Since 2026-09-11 the probe
    # rows carry explicit IDs, so this warning fires only for genuinely blank
    # cells - not for every numeric probe id.
    blank_id_positions = []
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        for pos, row in enumerate(csv.DictReader(f), 1):
            if (row.get("question") or "").strip() and not (row.get("ID") or "").strip():
                blank_id_positions.append(pos)
    derived_ids = {f"Q{pos:03d}" for pos in blank_id_positions}

    unstable = [q.id for q in out if not q.is_answerable and q.id in derived_ids]
    if unstable:
        msg = (
            f"{len(unstable)} probe id(s) ({unstable[0]}..{unstable[-1]}) were derived "
            "from CSV ROW ORDER because their ID cell is empty. They are stable only "
            "while the probe rows stay at the bottom of the CSV in the same order. "
            "Inserting a row above them renumbers them and, because cache keys bind to "
            "question_id, invalidates every cached probe generation. Give the probe "
            "rows explicit IDs in the CSV."
        )
        warnings.append(msg)
        stream.write(f"[WARNING] {msg}\n")
        stream.flush()

    coverage = coverage_report(out)

    stream.write(
        f"[BENCHMARK] {len(out)} questions from {csv_path} "
        f"(csv_hash={csv_hash[:12]}...): {n_answerable} answerable, "
        f"{n_probes} unanswerable probes.\n"
    )
    if reclassified:
        stream.write(
            f"[BENCHMARK] Derived is_answerable=False for {len(reclassified)} row(s) "
            f"({reclassified[0]}..{reclassified[-1]}); basis counts: "
            f"{ {b: sum(1 for v in classification.values() if v['basis'] == b) for b in sorted({v['basis'] for v in classification.values()})} }.\n"
        )
    stream.flush()

    return BenchmarkSet(
        questions=out,
        csv_path=csv_path,
        csv_hash=csv_hash,
        classification=classification,
        reclassified_ids=reclassified,
        anomaly_ids=anomalies,
        unstable_ids=unstable,
        coverage=coverage,
        warnings=warnings,
    )
