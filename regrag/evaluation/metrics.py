"""Deterministic lexical scorers for correctness, groundedness, hallucination and abstention.

Scorer identity
---------------
``SCORER_NAME = "regrag-lexical-v1"``. Every record this module produces is
stamped ``metric_status = "unvalidated"`` and lists the fields whose scorer has
been *implemented but not yet measured against human labels* in
``placeholder_fields``. ``metric_status`` never becomes ``"final"`` from inside
this module: only a human-validation run (``validation_report``) can justify
that, and it is a deliberate act by whoever owns the label set.

Why "unvalidated" and not "placeholder"
---------------------------------------
``regrag/models.py`` documents ``"placeholder"`` as "the scorer for this field is
not yet implemented". After this rewrite the scorers ARE implemented, so that tag
would itself become a misrepresentation. ``"unvalidated"`` says precisely what is
true: the number exists, it is deterministic and reproducible, and nobody has yet
measured whether it agrees with a human. ``assert_metrics_publishable`` refuses to
aggregate any record that is not ``"final"``, so an unvalidated row still cannot
reach a paper table.

What each scorer measures - and what it cannot
----------------------------------------------
No neural model is used anywhere in this module. Everything is lexical,
deterministic and CPU-cheap (no torch, no embeddings, no downloads), which is a
hard constraint of this project: the reference machine is a 12-core laptop with a
4 GB GPU and a previous local torch run froze it. The consequences are stated
per scorer below and repeated in the record's ``notes`` field.

* ``groundedness`` - is each content sentence of the answer supported by the
  RETRIEVED context? Signal: stopword-filtered syllable-token containment of the
  sentence in the retrieved chunk text, plus numeric-fact and legal-reference
  agreement with that same context. It CANNOT detect: a correct paraphrase that
  avoids the document's wording (scored as unsupported - false positive), a
  fluent sentence that reuses the document's vocabulary while inverting its
  meaning (scored as supported - false negative; negation and deontic modals are
  kept as content tokens for exactly this reason, but token overlap is still
  polarity-blind), and relevance (a sentence can be fully grounded in a chunk
  that does not answer the question).
  Reference text is ALWAYS the retrieved chunk(s). ``gold_passage`` is never used
  as a grounding reference - see "gold_passage" below.

* ``correctness_score`` - does the answer state what the gold answer states?
  Signal: content-token recall/F1 against ``gold.reference_answer`` plus
  numeric-fact agreement with it, on a 0.0 / 1.0 / 2.0 scale. Citations can only
  *lower* it (a correct-content answer carrying a wrong Điều is capped at 1.0);
  they can never raise it, which is the exact inversion of the placeholder scorer
  this replaces. It CANNOT detect: a correct answer that summarises a long gold
  answer in other words (under-credited), and a wrong answer that happens to
  reuse the gold answer's vocabulary (over-credited).

* ``hallucinated`` / ``hallucination_types`` - a typed taxonomy, not one bit:
  ``UNSUPPORTED_CLAIM`` (not backed by the retrieved chunk),
  ``UNSUPPORTED_BY_GOLD`` (not backed by the human gold answer),
  ``CONTRADICTS_CONTEXT``, ``WRONG_INSTRUMENT``,
  ``NON_COVERING_PROVISION``, ``NUMERIC_THRESHOLD_MISMATCH``,
  ``UNCITED_ASSERTION``, ``ANSWERED_UNANSWERABLE``, ``SPURIOUS_CITATION``,
  ``DEGENERATE_OUTPUT``. ``UNCITED_ASSERTION`` (a fluent fabrication carrying no
  citation at all) is scored strictly WORSE than a wrong citation, fixing the
  placeholder's inverted logic where a no-citation fabrication escaped the flag
  entirely. A *correct* answer that merely omits its citation is NOT a
  hallucination - that is a citation-recall failure, reported as
  ``no_citation_emitted=1`` - otherwise RQ1 would measure citation style.

* ``abstained`` - delegates the benchmark-defined strings (the gold sentinel
  ``UNANSWERABLE_SENTINEL`` and the prompted ``ABSTENTION_KEYPHRASE``) to
  ``regrag.generation.abstention.detect_abstention``, then adds two things that
  module deliberately does not do: a broader Vietnamese refusal-pattern tier,
  and an ASSERTION GATE. A refusal that is followed by a substantive answer
  ("Tài liệu không đề cập... tuy nhiên hạn mức là 50 triệu đồng") is the single
  most dangerous output in this benchmark - it reads as a refusal to a keyphrase
  matcher and as a fabrication to a human. The gate scores it as a
  non-abstention with ``hedged_then_answered=True``.

gold_passage is NOT trusted text
--------------------------------
An audit of ``data/gold/bank_qa_data.csv`` (88 rows) against the delivered
documents found that only the first rows carry verbatim source excerpts; 46 of
the 65 answerable rows hold a one-line ANNOTATOR SUMMARY with zero 2-gram overlap
with the document they cite. Therefore:

* groundedness never reads ``gold_passage``;
* numeric/legal-reference agreement is computed against ``reference_answer`` and
  the retrieved context only;
* ``gold_passage`` is used for exactly two things: answerability resolution (a
  probe has no passage) and a per-row DATA-QUALITY FLAG. That flag is measured,
  not guessed - when retrieved context is available the scorer computes the
  2-gram overlap between ``gold_passage`` and the context and reports it, so
  results can be stratified by whether the row really had source text. Rows with
  a paraphrase are never silently pooled with verbatim rows by
  ``summarise_records``.

Citation gold is used as-is through ``regrag.evaluation.citation`` (unmodified).
When the gold instrument is unresolved, or the caller reports the citation gold
as invalid, article/instrument judgements are SUPPRESSED and tagged rather than
emitted as false hallucinations.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, FrozenSet, List, Optional, Sequence, Set, Tuple

from regrag.models import EvaluationRecord, GenerationResult, GoldQuestion, RetrievedResult
from regrag.provenance import (
    CORPUS_UNSET,
    ProvenanceError,
    assert_publishable,
    require,
)
from regrag.corpus.canonical import fold_diacritics, normalize_ws
from regrag.generation.abstention import detect_abstention as detect_exact_abstention
from regrag.generation.config import UNANSWERABLE_SENTINEL
from regrag.generation.prompts import ABSTENTION_KEYPHRASE
from regrag.evaluation.citation import (
    CIRCULAR_PATTERN,
    EXPLICIT_DOC_PATTERN,
    NAMED_DOC_PATTERNS,
    compute_citation_precision_recall,
    extract_citations,
    normalize_doc_id,
)
from regrag.evaluation.agreement import compute_cohens_kappa

SCORER_NAME = "regrag-lexical-v1"

#: Status stamped on every record. See module docstring: the scorers exist, they
#: have not been measured against human labels, so they are not publishable.
METRIC_STATUS_UNVALIDATED = "unvalidated"
METRIC_STATUS_FINAL = "final"

#: EvaluationRecord fields whose value comes from an unvalidated scorer. Kept in
#: ``placeholder_fields`` so the existing self-identification mechanism carries
#: the new meaning; every name here is a real field of EvaluationRecord.
UNVALIDATED_FIELDS: Tuple[str, ...] = (
    "correctness_score",
    "hallucinated",
    "abstained",
    "abstained_correctly",
)

# --- hallucination taxonomy --------------------------------------------------
H_UNSUPPORTED_CLAIM = "UNSUPPORTED_CLAIM"                    # not supported by retrieved context
H_CONTRADICTS_CONTEXT = "CONTRADICTS_CONTEXT"                # numeric conflict with retrieved context
H_WRONG_INSTRUMENT = "WRONG_INSTRUMENT"                      # cited a different legal instrument
H_NON_COVERING_PROVISION = "NON_COVERING_PROVISION"          # cited a provision that does not cover the question
H_NUMERIC_THRESHOLD_MISMATCH = "NUMERIC_THRESHOLD_MISMATCH"  # right article, wrong figure
H_UNSUPPORTED_BY_GOLD = "UNSUPPORTED_BY_GOLD"                # assertion with ~no agreement with the gold answer
H_UNCITED_ASSERTION = "UNCITED_ASSERTION"                    # fluent assertion, no citation at all
H_ANSWERED_UNANSWERABLE = "ANSWERED_UNANSWERABLE"            # answered an unanswerable probe
H_SPURIOUS_CITATION = "SPURIOUS_CITATION"                    # cited law for a probe that has no law
H_DEGENERATE_OUTPUT = "DEGENERATE_OUTPUT"                    # empty/whitespace answer

#: Severity 3 = the answer has no verifiable basis at all; 2 = wrong legal basis
#: or wrong figure; 1 = a diagnostic that needs a human. Used to report which
#: hallucinations dominate, and to keep "no citation" worse than "wrong citation".
SEVERITY: Dict[str, int] = {
    H_DEGENERATE_OUTPUT: 3,
    H_UNCITED_ASSERTION: 3,
    H_ANSWERED_UNANSWERABLE: 3,
    H_NUMERIC_THRESHOLD_MISMATCH: 2,
    H_UNSUPPORTED_BY_GOLD: 2,
    H_CONTRADICTS_CONTEXT: 2,
    H_WRONG_INSTRUMENT: 2,
    H_NON_COVERING_PROVISION: 2,
    H_SPURIOUS_CITATION: 2,
    H_UNSUPPORTED_CLAIM: 1,
}

# --- thresholds --------------------------------------------------------------
# CALIBRATED on the trusted CSV (data/gold/bank_qa_data.csv, 65 answerable rows,
# measured 2026-09-11), not chosen by eye. Positive pairs = a gold answer's own
# sentences against its own gold passage; negative pairs = the same sentences
# against other rows' passages / other rows' gold answers.
#
#   sentence containment   POS p25=0.50 med=0.68 | NEG p75=0.25 p90~0.35
#   gold_recall            POS(answer vs own long passage) p25=0.41 med=0.58
#                          NEG(answer vs another row's gold answer) p90=0.36 p95=0.44
#   content_f1             POS p25=0.39 med=0.48 | NEG p95=0.26 max=0.40
#   self-match (answer vs itself) -> gold_recall 1.00, score 2.0 for 65/65 rows
#
# They are lexical cut-offs, not probabilities: a sentence at 0.49 is not
# meaningfully different from one at 0.51. That is why the record carries the
# underlying counts in ``notes``, and why metric_status stays "unvalidated" until
# validation_report() measures agreement with human labels.
SUPPORT_CONTAINMENT = 0.50     # sentence is grounded in the retrieved context
PARTIAL_CONTAINMENT = 0.30     # sentence shares vocabulary but is not supported
FULL_GOLD_RECALL = 0.55        # answer covers the gold answer's content
FULL_CONTENT_F1 = 0.45
PARTIAL_GOLD_RECALL = 0.35     # above NEG p90 (0.357): a wrong answer rarely reaches it
PARTIAL_CONTENT_F1 = 0.25      # at NEG p95 (0.259)
MIN_CONTENT_TOKENS = 3         # below this a sentence is a fragment, not a claim

# --- normalisation -----------------------------------------------------------

_WORD_RE = re.compile(r"[^\W\d_]+", re.UNICODE)
_SENT_SPLIT_RE = re.compile(r"[\.!?\n\r]+")
_URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
_DATE_RE = re.compile(r"\b\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4}\b")
_LEGAL_REF_RE = re.compile(
    r"\b(?:Điều|Khoản|Điểm|Mục|Chương|Phần|Phụ lục|Tiết)\s*\d+(?:[.,\/]\d+)*[a-zA-Z]?",
    re.IGNORECASE | re.UNICODE,
)
_LIST_MARKER_RE = re.compile(r"(?m)^\s*\d+[.)]\s+|\(\s*\d+\s*\)")

#: Grammatical function words plus non-discriminating legal boilerplate.
#: Negation and deontic words (không, chưa, cấm, được, phải, bị, trừ) are
#: deliberately NOT here: in legal text they carry the operative meaning, and
#: stripping them would make an inversion invisible to every overlap measure.
STOPWORDS: FrozenSet[str] = frozenset(
    """
    và hoặc hay cũng của trong ngoài trên dưới tại về với cho nhằm để mà thì là
    như nếu nhưng tuy do vì nên khi lúc sau trước từ đến vào ra này nọ kia đó đây
    ấy các mỗi mọi những một vài nhiều ít rất quá cả chỉ riêng cùng theo căn cứ
    gồm bao gồm thuộc gồm cả sau đây nêu trên kể trên dưới đây đối
    quy định việc trường hợp nội dung vấn đề liên quan đáp ứng phù hợp tương ứng
    """
    .split()
)


def normalize(text: str) -> str:
    """NFC + whitespace collapse + casefold. Diacritics preserved."""
    return normalize_ws(text or "").casefold()


def fold(text: str) -> str:
    """Second, looser tier: also strips diacritics.

    Used only to *relax* a match, never to create one, and every relaxed match is
    counted separately so a corpus that lost its diacritics is visible instead of
    silently passing (same philosophy as the §4.4 coverage invariant).
    """
    return fold_diacritics(normalize_ws(text or "")).casefold()


def content_tokens(text: str) -> Tuple[FrozenSet[str], FrozenSet[str]]:
    """(strict, folded) sets of content tokens: letters only, no stopwords."""
    strict: Set[str] = set()
    folded: Set[str] = set()
    for w in _WORD_RE.findall(normalize_ws(text or "")):
        t = w.casefold()
        if t in STOPWORDS or len(t) < 2:
            continue
        strict.add(t)
        folded.add(fold(t))
    return frozenset(strict), frozenset(folded)


# --- numeric facts -----------------------------------------------------------

#: Money multipliers, so "2 tỷ đồng" and "2000 triệu đồng" compare equal.
_MONEY: Dict[str, float] = {
    "đồng": 1.0,
    "vnd": 1.0,
    "vnđ": 1.0,
    "nghìn đồng": 1e3,
    "ngàn đồng": 1e3,
    "triệu đồng": 1e6,
    "tỷ đồng": 1e9,
    "nghìn": 1e3,
    "ngàn": 1e3,
    "triệu": 1e6,
    "tỷ": 1e9,
}

#: Unit vocabulary mined from the trusted CSV (answer + passage cells): the
#: units that actually occur are tuổi, ngày, ngày làm việc, tháng, năm, giờ,
#: tỷ đồng, % and a handful of counters. An unrecognised unit yields "" and is
#: then EXCLUDED from conflict detection - the scorer never claims a numeric
#: contradiction it cannot read.
_UNITS: Dict[str, str] = {
    "%": "%",
    "phần trăm": "%",
    "ngày làm việc": "ngày làm việc",
    "ngày": "ngày",
    "tháng": "tháng",
    "năm": "năm",
    "quý": "quý",
    "tuần": "tuần",
    "giờ": "giờ",
    "phút": "phút",
    "tuổi": "tuổi",
    "lần": "lần",
    "người": "người",
    "bộ": "bộ",
    "bản": "bản",
    "hồ sơ": "hồ sơ",
    "tài khoản": "tài khoản",
    "thẻ": "thẻ",
    "giao dịch": "giao dịch",
    "km": "km",
    "ha": "ha",
    "usd": "usd",
    "eur": "eur",
}
_UNITS.update(_MONEY)

#: A number followed by one of these words is not a quantity: "đồng thời",
#: "tỷ lệ an toàn vốn". Without this guard the scorer invents units.
_FALSE_FRIENDS: Dict[str, Tuple[str, ...]] = {
    "đồng": ("thời",),
    "tỷ": ("lệ",),
    "năm": ("nào", "nay", "ngoái"),
}

def _unit_alternation() -> str:
    """Regex alternation of the unit vocabulary, longest first.

    Multi-word units ("triệu đồng", "ngày làm việc") must be tried before their
    own head words, or "30 triệu đồng" is read as 30 triệu. The words are escaped
    individually and joined with ``\\s+`` - escaping the joined string would
    escape the backslash and silently match nothing.
    """
    alts = [r"\s+".join(re.escape(w) for w in u.split(" ")) for u in _UNITS]
    return "|".join(sorted(alts, key=len, reverse=True))


_UNIT_ALT = _unit_alternation()
_NUM_RE = re.compile(
    rf"(?P<num>\d+(?:[.,]\d+)*)\s*(?P<unit>{_UNIT_ALT})?(?P<per>\s*/\s*(?:ngày|tháng|năm|giờ))?",
    re.IGNORECASE | re.UNICODE,
)


@dataclass(frozen=True)
class NumericFact:
    """One quantity as written: ``30 triệu đồng/ngày`` -> (30.0, "triệu đồng/ngày")."""

    value: float
    unit: str
    surface: str

    @property
    def base_unit(self) -> str:
        """Unit without the "/per-day" tail, so "triệu đồng/ngày" == "triệu đồng"."""
        return self.unit.split("/")[0].strip()

    @property
    def family(self) -> str:
        """Comparison family: all money collapses to VND, else the head noun.

        "ngày làm việc" and "ngày" share the family "ngày" (working days vs
        calendar days is a real legal distinction, reported separately as a unit
        mismatch rather than a value mismatch).
        """
        b = self.base_unit
        if b in _MONEY:
            return "VND"
        if b == "%":
            return "%"
        return b.split(" ")[0] if b else ""

    @property
    def canonical(self) -> Tuple[float, str]:
        """(quantity, family) with money rescaled, so 2 tỷ đồng == 2000 triệu đồng."""
        mult = _MONEY.get(self.base_unit)
        if mult is not None:
            return (round(self.value * mult, 6), "VND")
        return (self.value, self.family)

    def __str__(self) -> str:  # pragma: no cover - reporting helper
        return self.surface


def _parse_number(raw: str) -> float:
    """Parse a Vietnamese-formatted number: 1.000.000 / 1,5 / 30."""
    raw = raw.strip()
    if re.fullmatch(r"\d{1,3}(?:[.,]\d{3})+", raw):        # thousand separators
        return float(re.sub(r"[.,]", "", raw))
    if re.fullmatch(r"\d+[.,]\d+", raw):                    # decimal , or .
        return float(raw.replace(",", "."))
    return float(raw.replace(",", "").replace(".", ""))     # e.g. "05"


def _mask_non_quantities(text: str) -> str:
    """Blank out numbers that are references, not quantities.

    Legal citations ("Điều 8.1", "Khoản 2"), instrument ids ("18/2024/TT-NHNN",
    "Thông tư số 06/2019/TT-NHNN"), URLs, dates and list markers are all digits
    that must never be read as thresholds.
    """
    t = normalize_ws(text or "")
    t = _URL_RE.sub(" ", t)
    t = _DATE_RE.sub(" ", t)
    t = _LEGAL_REF_RE.sub(" ", t)
    for pat in (CIRCULAR_PATTERN, EXPLICIT_DOC_PATTERN):
        t = pat.sub(" ", t)
    for pat, _doc in NAMED_DOC_PATTERNS:
        t = pat.sub(" ", t)
    t = _LIST_MARKER_RE.sub(" ", t)
    return t


def extract_numeric_facts(text: str) -> FrozenSet[NumericFact]:
    """Quantities with an explicit unit, as canonical (value, unit) pairs.

    A bare 4-digit 1900-2100 number is read as a calendar year and dropped;
    numbers with no recognised unit are dropped, because a conflict claim
    requires knowing what is being counted.
    """
    masked = _mask_non_quantities(text)
    facts: Set[NumericFact] = set()
    for m in _NUM_RE.finditer(masked):
        raw = m.group("num")
        unit = (m.group("unit") or "").strip()
        unit = re.sub(r"\s+", " ", unit).casefold()
        per = (m.group("per") or "").replace(" ", "").casefold()
        value = _parse_number(raw)
        if not unit:
            if re.fullmatch(r"\d{4}", raw) and 1900 <= value <= 2100:
                continue                                    # calendar year
            # No recognised unit: recorded with unit "" and excluded from
            # conflict detection by callers (family == "").
            facts.add(NumericFact(value, "", raw))
            continue
        friends = _FALSE_FRIENDS.get(unit.split(" ")[-1])
        if friends:
            tail = masked[m.end():].lstrip().casefold()
            if any(tail.startswith(f) for f in friends):
                facts.add(NumericFact(value, "", raw))
                continue
        if per:
            unit = f"{unit}/{per.lstrip('/')}"
        facts.add(NumericFact(value, unit, m.group(0).strip()))
    return frozenset(facts)


@dataclass(frozen=True)
class NumericComparison:
    """Result of comparing the answer's quantities against a reference set."""

    reference: FrozenSet[NumericFact]
    answer: FrozenSet[NumericFact]
    #: answer facts whose family exists in the reference but whose value does not
    conflicts: Tuple[NumericFact, ...] = ()
    #: reference facts the answer never states
    missing: Tuple[NumericFact, ...] = ()
    #: same value, narrower/wider unit ("30 ngày" vs "30 ngày làm việc")
    unit_mismatches: Tuple[Tuple[NumericFact, NumericFact], ...] = ()

    @property
    def has_conflict(self) -> bool:
        return bool(self.conflicts)

    def describe(self) -> str:
        parts = []
        if self.conflicts:
            parts.append("conflict=" + ",".join(c.surface for c in self.conflicts))
        if self.missing:
            parts.append("missing=" + ",".join(c.surface for c in self.missing))
        if self.unit_mismatches:
            parts.append(
                "unit=" + ",".join(f"{a.surface}!={b.surface}" for a, b in self.unit_mismatches)
            )
        return ";".join(parts) if parts else "ok"


def compare_numeric_facts(
    reference: Sequence[NumericFact], answer: Sequence[NumericFact]
) -> NumericComparison:
    """Find quantities the answer states that the reference contradicts.

    Only unit families present in BOTH sides are compared. A conflict means: the
    reference speaks about "% vốn tự có" and the answer states a different "%".
    Facts with no recognised unit are never used to claim a conflict.
    """
    ref = frozenset(reference)
    ans = frozenset(answer)
    ref_canon = {f.canonical for f in ref if f.family}
    ref_families: Dict[str, Set[Tuple[float, str]]] = {}
    for f in ref:
        if f.family:
            ref_families.setdefault(f.family, set()).add(f.canonical)

    conflicts: List[NumericFact] = []
    unit_mismatch: List[Tuple[NumericFact, NumericFact]] = []
    ref_by_family_unit: Dict[str, Set[str]] = {}
    for f in ref:
        if f.family:
            ref_by_family_unit.setdefault(f.family, set()).add(f.base_unit)

    for f in sorted(ans, key=lambda x: (x.family, x.value, x.unit)):
        if not f.family or f.family not in ref_families:
            continue
        if f.canonical in ref_canon:
            # Same quantity - but did the unit narrow or widen? "30 ngày" for a
            # gold "30 ngày làm việc" is a real legal difference.
            if f.base_unit not in ref_by_family_unit.get(f.family, set()):
                match = next(
                    (r for r in ref if r.canonical == f.canonical and r.base_unit != f.base_unit),
                    None,
                )
                if match is not None:
                    unit_mismatch.append((f, match))
            continue
        conflicts.append(f)

    ans_canon = {f.canonical for f in ans if f.family}
    missing = tuple(sorted((f for f in ref if f.family and f.canonical not in ans_canon),
                           key=lambda x: (x.family, x.value)))
    return NumericComparison(
        reference=ref,
        answer=ans,
        conflicts=tuple(conflicts),
        missing=missing,
        unit_mismatches=tuple(unit_mismatch),
    )


# --- abstention --------------------------------------------------------------

#: Refusal paraphrases beyond the two benchmark-defined strings. Deliberately
#: narrow: each one asserts something about the *evidence*, not about the world.
#: "không được quy định" is NOT here on its own because it is also a legitimate
#: substantive answer; it only counts when it is predicated of the corpus.
SOFT_ABSTENTION_PATTERNS: Tuple[Tuple[str, str], ...] = (
    ("corpus_silent", r"(?:tài\s+liệu|văn\s+bản|ngữ\s+cảnh|hồ\s+sơ|dữ\s+liệu|đoạn\s+trích|thông\s+tin\s+(?:được\s+cung\s+cấp|tra\s+cứu))[^.!?]{0,40}?\b(?:không|chưa)\s+(?:có|đề\s+cập|quy\s+định|nêu|nhắc\s+đến|bao\s+gồm|cung\s+cấp|thấy|phản\s+ánh)"),
    ("not_in_corpus", r"\bkhông\s+có\s+trong\s+(?:tài\s+liệu|kho\s+văn\s+bản|văn\s+bản|ngữ\s+cảnh|hồ\s+sơ|cơ\s+sở\s+dữ\s+liệu|kho\s+dữ\s+liệu|danh\s+mục\s+tài\s+liệu|phạm\s+vi)"),
    ("not_found", r"\bkhông\s+(?:tìm\s+thấy|thấy|tra\s+cứu\s+được|xác\s+định\s+được)\b[^.!?]{0,30}?\b(?:thông\s+tin|quy\s+định|căn\s+cứ|nội\s+dung|dữ\s+liệu)"),
    ("no_information", r"\bkhông\s+có\s+(?:thông\s+tin|căn\s+cứ|cơ\s+sở|dữ\s+liệu|quy\s+định)\b"),
    ("insufficient", r"\bkhông\s+đủ\s+(?:thông\s+tin|cơ\s+sở|căn\s+cứ|dữ\s+liệu|tài\s+liệu)"),
    ("cannot_answer", r"\bkhông\s+thể\s+(?:trả\s+lời|đưa\s+ra\s+(?:câu\s+trả\s+lời|kết\s+luận)|xác\s+định|kết\s+luận|trả\s+lời\s+chính\s+xác)"),
    ("no_basis_to_answer", r"\bkhông\s+có\s+cơ\s+sở\s+(?:pháp\s+lý\s+)?để\s+(?:trả\s+lời|xác\s+định|kết\s+luận|đánh\s+giá)"),
    ("law_silent", r"\b(?:pháp\s+luật|văn\s+bản\s+(?:bản\s+)?pháp\s+lý|quy\s+định\s+hiện\s+hành)\s+(?:không|chưa)\s+(?:quy\s+định|đề\s+cập|điều\s+chỉnh)"),
    ("not_regulated", r"\b(?:chưa|không)\s+được\s+(?:quy\s+định|đề\s+cập|nêu)\s+(?:trong|tại|bởi)"),
    ("out_of_scope", r"\b(?:không\s+nằm|không\s+thuộc|ngoài)\s+(?:trong\s+)?phạm\s+vi\b"),
    ("unable_english", r"\b(?:i\s+(?:do\s+not|don't|cannot|can't|am\s+unable)|unable\s+to|not\s+(?:specified|provided|mentioned|found)\s+in|no\s+information|outside\s+the\s+(?:scope|context))"),
)
def _fold_pattern(pat: str) -> str:
    """Fold a pattern's literal Vietnamese so it can match a folded answer.

    The patterns below are written with diacritics because they are Vietnamese
    and must stay readable to the annotators, but they are matched against
    ``fold(answer)``, which has the diacritics stripped. ``fold_diacritics`` only
    removes combining marks and maps đ->d, so regex syntax (``\\s``, ``\\b``,
    ``(?:...)``, ``{0,40}``) passes through untouched. Without this step a
    diacritic-rich pattern can never match, and every paraphrased refusal is
    silently missed - the failure mode is invisible because the exact tier still
    works.
    """
    return fold(pat)


_SOFT_COMPILED: Tuple[Tuple[str, "re.Pattern[str]"], ...] = tuple(
    (name, re.compile(_fold_pattern(pat), re.IGNORECASE | re.UNICODE))
    for name, pat in SOFT_ABSTENTION_PATTERNS
)


@dataclass(frozen=True)
class Sentence:
    """One answer sentence, classified."""

    text: str
    kind: str                          # "abstention" | "content" | "fragment"
    tokens: FrozenSet[str] = frozenset()
    folded: FrozenSet[str] = frozenset()
    facts: FrozenSet[NumericFact] = frozenset()


@dataclass(frozen=True)
class AbstentionDecision:
    """Three-tier abstention decision.

    ``tier`` is ``"exact"`` (a benchmark-defined string), ``"soft"`` (a refusal
    paraphrase confirmed by the assertion gate) or ``"none"``.
    """

    abstained: bool
    tier: str
    matched: Tuple[str, ...]
    soft_matched: Tuple[str, ...]
    hedged_then_answered: bool
    empty_answer: bool
    n_abstention_sentences: int
    n_assertion_sentences: int
    states_figure: bool = False

    @property
    def reason(self) -> str:
        if self.empty_answer:
            return "empty_answer"
        if self.abstained:
            return f"{self.tier}:{','.join(self.matched) or '-'}"
        if self.hedged_then_answered:
            return "hedged_then_answered"
        if self.soft_matched and self.states_figure:
            return "soft_with_figures:" + ",".join(self.soft_matched)
        if self.soft_matched:
            return "soft_with_assertions:" + ",".join(self.soft_matched)
        return "none"


def split_sentences(text: str) -> Tuple[Sentence, ...]:
    """Split an answer into sentences and classify each one.

    Splitting is on ``. ! ?`` and newlines only: Vietnamese legal prose chains
    enumerations with ``;`` and ``:``, and splitting there would turn one claim
    into fragments that individually look ungrounded. Fragments (fewer than
    ``MIN_CONTENT_TOKENS`` content tokens, no quantity, no refusal) are counted
    but never scored, so a bullet list cannot manufacture unsupported claims.
    """
    out: List[Sentence] = []
    for raw in _SENT_SPLIT_RE.split(normalize_ws(text or "")):
        s = raw.strip(" \t;:,()-")
        if not s:
            continue
        strict, folded = content_tokens(s)
        facts = extract_numeric_facts(s)
        exact, soft = _abstention_matches(s)
        if exact or soft:
            kind = "abstention"
        elif len(strict) >= MIN_CONTENT_TOKENS or facts:
            kind = "content"
        else:
            kind = "fragment"
        out.append(Sentence(text=s, kind=kind, tokens=strict, folded=folded, facts=facts))
    return tuple(out)


def _abstention_matches(text: str) -> Tuple[Tuple[str, ...], Tuple[str, ...]]:
    """(exact tier names, soft tier names) matched by one string."""
    folded = fold(text)
    verdict = detect_exact_abstention(text)
    exact: List[str] = []
    if verdict.abstained:
        exact.append("benchmark_keyphrase" if verdict.matched_exact == fold(
            ABSTENTION_KEYPHRASE) else "gold_sentinel")
    soft = tuple(
        name for name, pat in _SOFT_COMPILED if pat.search(folded)
    )
    # The exact-tier module also reports its own soft list; surface those names
    # so nothing is lost if the two lists ever diverge.
    extra = tuple(p for p in (verdict.matched_soft or []) if p not in soft)
    return tuple(exact), soft + tuple(f"abstention.py:{e}" for e in extra)


def detect_abstention(answer_text: str) -> AbstentionDecision:
    """Detect a refusal, then gate it on whether the answer still asserts.

    The gate is what makes a broader pattern list safe. ``abstention.py`` keeps
    paraphrases diagnostic-only because "không được quy định" can be a
    substantive answer; here a paraphrase only counts when the rest of the answer
    contains no assertion at all - no content sentence and no quantity. So a bare
    refusal in any wording is credited, and a refusal followed by a fabricated
    figure is not.
    """
    text = normalize_ws(answer_text or "")
    if not text:
        # An empty answer is a failed generation, not a refusal: crediting it
        # would hand the model a perfect abstention score for producing nothing.
        return AbstentionDecision(
            abstained=False, tier="none", matched=(), soft_matched=(),
            hedged_then_answered=False, empty_answer=True,
            n_abstention_sentences=0, n_assertion_sentences=0,
        )

    sentences = split_sentences(text)
    exact_names: List[str] = []
    soft_names: List[str] = []
    n_abst = 0
    n_assert = 0
    states_figure = False
    for s in sentences:
        e, so = _abstention_matches(s.text)
        exact_names.extend(e)
        soft_names.extend(so)
        if s.kind == "abstention":
            n_abst += 1
        elif s.kind == "content":
            n_assert += 1
        if any(f.family for f in s.facts):
            states_figure = True

    whole_e, whole_s = _abstention_matches(text)
    exact_names = list(dict.fromkeys(exact_names + list(whole_e)))
    soft_names = list(dict.fromkeys(soft_names + list(whole_s)))

    hedged = bool((exact_names or soft_names) and n_assert > 0)
    # A paraphrased refusal that also states a recognised quantity is not a
    # refusal: it answered. This extra condition applies to the SOFT tier only -
    # the two benchmark-defined strings are unambiguous by construction, and a
    # model that emits the registered keyphrase plus a stray figure has still
    # refused.
    soft_with_figure = bool(soft_names) and not exact_names and states_figure

    if exact_names and not hedged:
        return AbstentionDecision(True, "exact", tuple(exact_names), tuple(soft_names),
                                  False, False, n_abst, n_assert, states_figure)
    if exact_names and hedged:
        # The benchmark's own refusal string appears, but so does a substantive
        # answer. Scored as a non-abstention: the model did assert something.
        return AbstentionDecision(False, "none", tuple(exact_names), tuple(soft_names),
                                  True, False, n_abst, n_assert, states_figure)
    if soft_names and not hedged and not soft_with_figure:
        return AbstentionDecision(True, "soft", tuple(soft_names), tuple(soft_names),
                                  False, False, n_abst, n_assert, states_figure)
    return AbstentionDecision(False, "none", tuple(exact_names), tuple(soft_names),
                              hedged, False, n_abst, n_assert, states_figure)


# --- groundedness ------------------------------------------------------------

@dataclass(frozen=True)
class SentenceGrounding:
    text: str
    kind: str
    containment: float
    verdict: str                       # supported | partial | unsupported | conflict | na
    flags: Tuple[str, ...] = ()
    relaxed_hits: int = 0              # tokens matched only after diacritic folding


@dataclass(frozen=True)
class GroundednessReport:
    """Is the answer supported by the text the model was actually shown?

    ``groundedness`` is None - never 0.0 - when there is no retrieved context to
    check against (closed-book, or a caller that did not pass one). Returning 0.0
    there would be a plausible-looking default for "we could not measure this".
    """

    assessable: bool
    reason: str
    groundedness: Optional[float] = None
    mean_containment: Optional[float] = None
    n_content_sentences: int = 0
    n_supported: int = 0
    n_partial: int = 0
    n_unsupported: int = 0
    n_conflict: int = 0
    relaxed_hits: int = 0
    sentences: Tuple[SentenceGrounding, ...] = ()
    flags: Tuple[str, ...] = ()

    def describe(self) -> str:
        if not self.assessable:
            return f"NA({self.reason})"
        if self.groundedness is None:
            # Assessable context, but the answer contains no content sentence to
            # ground (a bare refusal, or a degenerate output). That is "nothing to
            # measure", not 0.00.
            return f"NA({self.reason})"
        return (
            f"{self.groundedness:.2f}"
            f"({self.n_supported}s/{self.n_partial}p/{self.n_unsupported}u"
            f"/{self.n_conflict}c of {self.n_content_sentences})"
        )


def score_groundedness(answer_text: str, contexts: Sequence[str]) -> GroundednessReport:
    """Score every content sentence of the answer against the retrieved context.

    Reference text is ``contexts`` ONLY - the chunks the generation was actually
    conditioned on. ``gold_passage`` is never used here (see module docstring).

    Per content sentence:
      * ``containment`` = |content tokens of the sentence found in the best
        matching context| / |content tokens of the sentence|;
      * every quantity in the sentence is looked for in the context; a quantity
        whose unit family occurs in the context with a DIFFERENT value is a
        ``NUMERIC_CONFLICT`` (this is what catches "right article, wrong figure"
        in RAG mode);
      * every ``Điều N`` / instrument named in the sentence must occur in the
        context, else ``REF_NOT_IN_CONTEXT`` - the model cited a provision the
        retrieval never gave it.

    Cannot detect paraphrase (a correct answer in fresh words scores low) or
    polarity inversion ("không được phép" vs "được phép" share almost all tokens).
    ``POLARITY_RISK`` is reported as a diagnostic flag and never on its own marks
    a sentence unsupported.
    """
    ctx_texts = [normalize_ws(c) for c in contexts if normalize_ws(c)]
    if not ctx_texts:
        return GroundednessReport(
            assessable=False,
            reason="no_retrieved_context",
            flags=("GROUNDEDNESS_UNASSESSABLE",),
        )

    ctx_strict: List[FrozenSet[str]] = []
    ctx_folded: List[FrozenSet[str]] = []
    ctx_norm: List[str] = []
    for c in ctx_texts:
        s, f = content_tokens(c)
        ctx_strict.append(s)
        ctx_folded.append(f)
        ctx_norm.append(normalize(c))
    ctx_facts = frozenset().union(*(extract_numeric_facts(c) for c in ctx_texts))
    ctx_fact_canon = {f.canonical for f in ctx_facts if f.family}
    ctx_families = {f.family for f in ctx_facts if f.family}
    ctx_refs = _legal_ref_surface(ctx_texts)

    sentences = split_sentences(answer_text)
    grounded: List[SentenceGrounding] = []
    n_sup = n_par = n_uns = n_conf = 0
    total_relaxed = 0

    for s in sentences:
        if s.kind != "content":
            grounded.append(SentenceGrounding(s.text, s.kind, 1.0 if s.kind == "abstention" else 0.0,
                                              "na", ()))
            continue
        if not s.tokens:
            grounded.append(SentenceGrounding(s.text, "fragment", 0.0, "na", ()))
            continue

        best = 0.0
        relaxed = 0
        for strict, folded in zip(ctx_strict, ctx_folded):
            hits = len(s.tokens & strict)
            extra = sum(1 for t in (s.tokens - strict) if fold(t) in folded)
            cont = (hits + extra) / len(s.tokens)
            if cont > best:
                best, relaxed = cont, extra
        total_relaxed += relaxed

        flags: List[str] = []
        if relaxed:
            flags.append("DIACRITIC_RELAXED")

        # quantities stated by this sentence vs the context
        sent_conflict = False
        for f in s.facts:
            if not f.family:
                continue
            if f.canonical in ctx_fact_canon:
                continue
            if f.family in ctx_families:
                flags.append("NUMERIC_CONFLICT")
                sent_conflict = True
            else:
                flags.append("NUMERIC_UNVERIFIED")

        # legal references stated by this sentence vs the context
        for ref in _legal_ref_surface([s.text]):
            if ref not in ctx_refs:
                flags.append("REF_NOT_IN_CONTEXT")

        # polarity diagnostic: negation/deontic markers present in the sentence
        # but absent from the context (weak signal, never decisive on its own)
        if _negation_markers(s.text) and not any(_negation_markers(c) for c in ctx_norm):
            flags.append("POLARITY_RISK")

        if sent_conflict:
            verdict = "conflict"
        elif best >= SUPPORT_CONTAINMENT and "REF_NOT_IN_CONTEXT" not in flags:
            verdict = "supported"
        elif best >= PARTIAL_CONTAINMENT:
            verdict = "partial"
        else:
            verdict = "unsupported"

        if verdict == "supported":
            n_sup += 1
        elif verdict == "partial":
            n_par += 1
        elif verdict == "conflict":
            n_conf += 1
        else:
            n_uns += 1
        grounded.append(SentenceGrounding(s.text, "content", round(best, 4), verdict,
                                          tuple(dict.fromkeys(flags)), relaxed))

    n_content = n_sup + n_par + n_uns + n_conf
    if n_content == 0:
        # Nothing to ground: a pure refusal, or an answer with no content
        # sentence at all. That is not "groundedness 0.0", it is "nothing to
        # measure", and the two must not be pooled.
        return GroundednessReport(
            assessable=True,
            reason="no_content_sentence",
            groundedness=None,
            mean_containment=None,
            n_content_sentences=0,
            sentences=tuple(grounded),
            flags=("NO_CONTENT_SENTENCE",),
        )

    return GroundednessReport(
        assessable=True,
        reason=f"contexts={len(ctx_texts)}",
        groundedness=round((n_sup + 0.5 * n_par) / n_content, 4),
        mean_containment=round(
            sum(g.containment for g in grounded if g.kind == "content") / n_content, 4
        ),
        n_content_sentences=n_content,
        n_supported=n_sup,
        n_partial=n_par,
        n_unsupported=n_uns,
        n_conflict=n_conf,
        relaxed_hits=total_relaxed,
        sentences=tuple(grounded),
        flags=tuple(dict.fromkeys(f for g in grounded for f in g.flags)),
    )


_NEGATION_RE = re.compile(
    r"\b(?:không|chưa|chẳng|cấm|trừ|ngoại\s+trừ|miễn|nghiêm\s+cấm)\b", re.IGNORECASE
)


def _negation_markers(text: str) -> FrozenSet[str]:
    return frozenset(m.group(0).casefold() for m in _NEGATION_RE.finditer(normalize_ws(text or "")))


_ARTICLE_SURFACE_RE = re.compile(r"\bđiều\s*\d+[a-z]?", re.IGNORECASE)


def _legal_ref_surface(texts: Sequence[str]) -> FrozenSet[str]:
    """Normalised legal references (article numbers + instrument ids) in ``texts``."""
    refs: Set[str] = set()
    for t in texts:
        n = normalize(t)
        for m in _ARTICLE_SURFACE_RE.finditer(n):
            refs.add(re.sub(r"\s+", "", m.group(0)))
        for c in extract_citations(t):
            doc = str(c.get("doc_id", ""))
            if doc and doc != "UNKNOWN":
                refs.add(normalize_ws(normalize_doc_id(doc)).casefold())
    return frozenset(refs)


# --- context acquisition -----------------------------------------------------

_CONTEXT_BLOCK_RE = re.compile(
    r"---\s*Tài\s+liệu\s+tham\s+khảo\s*\[\d+\]\s*---\s*(.*?)(?=\n\s*---\s*Tài\s+liệu\s+tham\s+khảo\s*\[\d+\]\s*---|\n---\s*\n|\Z)",
    re.IGNORECASE | re.DOTALL,
)


def context_from_prompt(prompt: str) -> List[str]:
    """Recover the retrieved chunks from a stored RAG prompt.

    ``GenerationResult`` keeps the prompt but not the retrieval, so this is the
    only way to ground a row after the fact. It matches the exact block marker
    ``build_rag_prompt`` emits. A closed-book prompt has no such block and
    yields [] - which the scorer reports as "unassessable", never as 0.0.
    """
    if not prompt:
        return []
    return [normalize_ws(b) for b in _CONTEXT_BLOCK_RE.findall(prompt) if normalize_ws(b)]


def resolve_contexts(
    retrieved: Optional[Sequence[Any]],
    contexts: Optional[Sequence[str]],
    gen: GenerationResult,
) -> Tuple[List[str], str]:
    """Pick the grounding reference, in priority order, and say which was used.

    Explicit ``contexts`` > ``retrieved`` RetrievedResults > blocks parsed out of
    the stored prompt. ``gold_passage`` is never a candidate.
    """
    if contexts:
        return [normalize_ws(c) for c in contexts if normalize_ws(c)], "caller_contexts"
    if retrieved:
        out: List[str] = []
        for r in retrieved:
            chunk = getattr(r, "chunk", r)
            fmt = getattr(chunk, "formatted_context", None)
            text = fmt() if callable(fmt) else str(getattr(chunk, "text", chunk) or "")
            if normalize_ws(text):
                out.append(normalize_ws(text))
        if out:
            return out, "retrieved_results"
    parsed = context_from_prompt(getattr(gen, "prompt", "") or "")
    if parsed:
        return parsed, "prompt_recovered"
    return [], "none"


# --- gold_passage data quality ----------------------------------------------
#
# The gold ``text_contains_answer_in_the_doc`` cell is NOT reliably source text.
# An audit of the 88-row CSV against the delivered documents (2026-09-11) found
# verbatim excerpts in only the first rows; 46 of the 65 answerable rows hold a
# one-line ANNOTATOR SUMMARY with zero 2-gram overlap with the instrument they
# cite. Two independent keys are therefore reported per row, and neither is
# allowed to masquerade as the other:
#
#   ``shape``  - a PRIOR read off the cell alone (a summary-shaped one-liner vs a
#                long excerpt). Cheap, always available, and only a guess.
#   ``overlap`` - a MEASUREMENT: the fraction of the cell's 2-grams that occur in
#                the retrieved context. This is the audit's verbatim test,
#                recomputed per row. It is ambiguous on its own - a low overlap
#                means either "the cell is a paraphrase" OR "retrieval missed the
#                gold provision" - so it is reported as a measurement next to the
#                row's retrieval result, never collapsed into a verdict.
#
# ``summarise_records`` stratifies by ``shape/overlap`` so summary-shaped rows are
# never silently pooled with verbatim ones in a groundedness mean.

#: Shape prior, from the cell alone.
PASSAGE_SHAPE_SUMMARY = "summary"      # "Điều N[ Khoản M]: ..." annotator one-liner
PASSAGE_SHAPE_EXCERPT = "excerpt"      # long cell: plausibly real source text
PASSAGE_SHAPE_SHORT = "short"          # short cell, not summary-shaped
PASSAGE_SHAPE_EMPTY = "empty"

#: Measured 2-gram overlap classes against the retrieved context.
OVERLAP_HIGH = "high"                  # >= VERBATIM_OVERLAP: the cell IS in the chunk
OVERLAP_MID = "mid"
OVERLAP_LOW = "low"                    # < PARAPHRASE_OVERLAP: the cell is NOT in the chunk
OVERLAP_UNMEASURED = "unmeasured"      # no retrieved context on this row

VERBATIM_OVERLAP = 0.50
PARAPHRASE_OVERLAP = 0.05

#: The annotator-summary shape: ``Điều 9: ...`` / ``Điều 11 Khoản 1: ...``. The
#: COLON is what distinguishes it from real legal text, which writes
#: ``Điều 14. Hạn mức giao dịch thẻ`` (dot + title + newline + clause numbering).
_SUMMARY_SHAPE_RE = re.compile(
    r"^\s*điều\s*\d+[a-z]?\s*(?:khoản\s*\d+\s*)?(?:điểm\s*[a-z0-9]+\s*)?\s*:",
    re.IGNORECASE | re.UNICODE,
)


@dataclass(frozen=True)
class PassageQuality:
    """Per-row data-quality flag for ``gold_passage`` (see the block comment)."""

    shape: str
    length: int
    bigram_overlap: Optional[float]
    overlap_class: str
    basis: str

    def describe(self) -> str:
        return f"{self.shape}(len={self.length})"

    def describe_overlap(self) -> str:
        if self.bigram_overlap is None:
            return f"{self.overlap_class}({self.basis})"
        return f"{self.bigram_overlap:.2f}/{self.overlap_class}"

    @property
    def stratum(self) -> str:
        return f"{self.shape}/{self.overlap_class}"


def _bigrams(text: str) -> Set[Tuple[str, str]]:
    toks = [t.casefold() for t in _WORD_RE.findall(normalize_ws(text or ""))]
    return {(a, b) for a, b in zip(toks, toks[1:])}


def assess_passage_quality(gold_passage: str, contexts: Sequence[str]) -> PassageQuality:
    """Classify the gold passage cell (prior) and, when possible, measure it."""
    p = normalize_ws(gold_passage or "")
    if not p:
        return PassageQuality(PASSAGE_SHAPE_EMPTY, 0, None, OVERLAP_UNMEASURED, "empty_cell")

    if _SUMMARY_SHAPE_RE.match(p):
        shape = PASSAGE_SHAPE_SUMMARY
    elif len(p) >= 300:
        shape = PASSAGE_SHAPE_EXCERPT
    else:
        shape = PASSAGE_SHAPE_SHORT

    overlap: Optional[float] = None
    klass = OVERLAP_UNMEASURED
    basis = "shape_only"
    ctx = [normalize_ws(c) for c in contexts if normalize_ws(c)]
    pg = _bigrams(p)
    if ctx and pg:
        cg: Set[Tuple[str, str]] = set()
        for c in ctx:
            cg |= _bigrams(c)
        overlap = round(len(pg & cg) / len(pg), 4)
        basis = "measured_vs_retrieved_context"
        if overlap >= VERBATIM_OVERLAP:
            klass = OVERLAP_HIGH
        elif overlap < PARAPHRASE_OVERLAP:
            klass = OVERLAP_LOW
        else:
            klass = OVERLAP_MID
    elif ctx:
        basis = "passage_has_no_bigrams"
    return PassageQuality(shape, len(p), overlap, klass, basis)


# --- answerability -----------------------------------------------------------

def resolve_answerability(gold: GoldQuestion) -> Tuple[bool, str, str]:
    """Decide whether a gold row is answerable, and say on what evidence.

    Defensive on purpose: the loader is being changed concurrently to emit the 23
    probes as ``is_answerable=False`` / ``category="unanswerable"``, and this must
    work either way. Returns ``(answerable, basis, anomaly)``; ``anomaly`` is
    non-empty whenever the row's own evidence contradicts itself, and is copied
    into ``notes`` so the row can be excluded from an aggregate deliberately
    instead of by accident.

    The decision mirrors ``qa_loader._classify``: a probe is a row with the
    sentinel answer AND no gold evidence. The sentinel alone is not enough - a row
    that has a passage and an instrument but whose answer cell says "not in the
    corpus" is a data defect, and calling it a probe would reward a model for
    refusing a question that has gold.
    """
    if not getattr(gold, "is_answerable", True):
        return False, "gold_flag", ""

    ref = normalize_ws(getattr(gold, "reference_answer", "") or "")
    sentinel = normalize_ws(UNANSWERABLE_SENTINEL).rstrip(".")
    is_sentinel = bool(ref) and ref.rstrip(".") == sentinel

    passage = normalize_ws(getattr(gold, "gold_passage", "") or "")
    docs = list(getattr(gold, "gold_doc_ids", []) or [])
    cites = list(getattr(gold, "gold_citations", []) or [])
    urls = list(getattr(gold, "source_urls", []) or [])
    has_evidence = bool(passage or docs or cites or urls)

    category = str(getattr(gold, "category", "")).casefold()
    if category == "unanswerable":
        anomaly = "category_unanswerable_with_gold" if has_evidence else ""
        return False, "category", anomaly
    if is_sentinel and not has_evidence:
        return False, "sentinel_answer", ""
    if is_sentinel and has_evidence:
        # Loader's "sentinel-answer-with-gold" anomaly: kept as answerable and
        # reported, never silently reclassified into a probe.
        return True, "gold_flag", "sentinel_answer_with_gold"
    if not has_evidence:
        return True, "gold_flag", "answerable_without_any_gold_evidence"
    if not normalize_ws(getattr(gold, "reference_answer", "") or ""):
        # No gold answer to compare against: correctness is unassessable for this
        # row and it is excluded from the mean instead of being scored 0.0.
        return True, "gold_flag", "answerable_without_gold_answer"
    if not passage:
        # An answerable row with no gold passage is a DATA DEFECT, not a probe.
        # Reclassifying it would reward a model for refusing a real question, so
        # it stays answerable and is flagged loudly for exclusion.
        return True, "gold_flag", "answerable_without_gold_passage"
    return True, "gold_flag", ""


# --- correctness -------------------------------------------------------------

@dataclass(frozen=True)
class CorrectnessReport:
    score: float                       # 0.0 | 1.0 | 2.0
    gold_recall: Optional[float]
    content_f1: Optional[float]
    answer_precision: Optional[float]
    numeric: Optional[NumericComparison]
    assessable: bool
    reason: str
    gates: Tuple[str, ...] = ()

    def describe(self) -> str:
        if not self.assessable:
            return f"NA({self.reason})"
        if self.gold_recall is None or self.content_f1 is None:
            # Probe / refusal path: correctness was decided by the abstention
            # judgement, so there is no lexical agreement to report. Printing
            # 0.00 here would invent a measurement.
            head = f"{self.score:.1f}({self.reason}"
        else:
            head = f"{self.score:.1f}(rec={self.gold_recall:.2f},f1={self.content_f1:.2f}"
        return (
            head
            + (f",num={self.numeric.describe()}" if self.numeric else "")
            + (f",gates={'+'.join(self.gates)}" if self.gates else "")
            + ")"
        )


def score_correctness(
    answer_text: str,
    gold_answer: str,
    abstention: AbstentionDecision,
    numeric_vs_context: Optional[NumericComparison],
    citation_error: bool,
    answerable: bool,
) -> CorrectnessReport:
    """Compare the answer TEXT to the gold answer text. 0.0 / 1.0 / 2.0.

    Citations are a one-way gate: a citation error caps the score at 1.0, and no
    citation match can raise it. Content agreement is what decides 2.0.

    Hard failures (0.0): a refusal on an answerable question, an answer on an
    unanswerable probe, an empty answer, or a quantity that contradicts the gold
    answer.
    """
    if abstention.empty_answer:
        return CorrectnessReport(0.0, None, None, None, None, False, "empty_answer",
                                 gates=("EMPTY",))

    gold_norm = normalize_ws(gold_answer or "")
    sentinel_norm = normalize_ws(UNANSWERABLE_SENTINEL).rstrip(".")
    if answerable and gold_norm.rstrip(".") == sentinel_norm:
        # The "sentinel-answer-with-gold" data defect: the row has an instrument
        # and a passage, but its gold answer says "not in the corpus". There is
        # nothing to compare the model's answer to, so correctness is
        # unassessable - not 0.0, and not the 2.0 a refusal would earn. Checked
        # BEFORE the false-abstention branch so a model is not penalised for
        # refusing a row whose own gold answer is a refusal.
        return CorrectnessReport(0.0, None, None, None, None, False,
                                 "gold_answer_is_sentinel", gates=("GOLD_DEFECT",))

    if answerable and abstention.abstained:
        return CorrectnessReport(0.0, None, None, None, None, True, "false_abstention",
                                 gates=("FALSE_ABSTENTION",))
    if not answerable:
        # For a probe the gold answer IS the refusal, so abstention is the whole
        # of correctness. No lexical comparison is attempted or needed.
        return CorrectnessReport(2.0 if abstention.abstained else 0.0, None, None, None,
                                 None, True,
                                 "probe_abstention" if abstention.abstained else "probe_answered",
                                 gates=() if abstention.abstained else ("ANSWERED_PROBE",))

    if not gold_norm:
        # An answerable row with no gold answer is a data defect. It is reported as
        # unassessable and excluded from the aggregate means by summarise_records,
        # rather than raising mid-campaign (generation is the expensive step and
        # must not be thrown away because one gold cell is blank) and rather than
        # being averaged in as a plausible 0.0.
        return CorrectnessReport(0.0, None, None, None, None, False,
                                 "gold_answer_empty", gates=("GOLD_DEFECT",))

    g_strict, g_folded = content_tokens(gold_answer)
    a_strict, a_folded = content_tokens(answer_text)

    if not g_strict:
        return CorrectnessReport(0.0, None, None, None, None, False,
                                 "gold_answer_has_no_content_tokens", gates=("GOLD_EMPTY",))

    def _overlap(src: FrozenSet[str], src_folded: FrozenSet[str],
                 dst: FrozenSet[str], dst_folded: FrozenSet[str]) -> float:
        hits = len(src & dst)
        hits += sum(1 for t in (src - dst) if fold(t) in dst_folded)
        return hits / len(src) if src else 0.0

    gold_recall = _overlap(g_strict, g_folded, a_strict, a_folded)
    answer_precision = _overlap(a_strict, a_folded, g_strict, g_folded)
    if gold_recall + answer_precision > 0:
        content_f1 = 2 * gold_recall * answer_precision / (gold_recall + answer_precision)
    else:
        content_f1 = 0.0

    gold_facts = extract_numeric_facts(gold_answer)
    ans_facts = extract_numeric_facts(answer_text)
    numeric = compare_numeric_facts(gold_facts, ans_facts) if gold_facts else None

    gates: List[str] = []
    if numeric and numeric.has_conflict:
        gates.append("NUMERIC_CONFLICT")
    if numeric and numeric.missing:
        gates.append("NUMERIC_INCOMPLETE")
    if numeric_vs_context and numeric_vs_context.has_conflict:
        gates.append("CONTEXT_NUMERIC_CONFLICT")
    if citation_error:
        gates.append("CITATION_ERROR")

    score: float
    if "NUMERIC_CONFLICT" in gates:
        score = 0.0                        # right article, wrong figure is wrong
    elif gold_recall >= FULL_GOLD_RECALL and content_f1 >= FULL_CONTENT_F1:
        score = 2.0
    elif gold_recall >= PARTIAL_GOLD_RECALL or content_f1 >= PARTIAL_CONTENT_F1:
        score = 1.0
    else:
        score = 0.0

    if score == 2.0 and ("CITATION_ERROR" in gates or "NUMERIC_INCOMPLETE" in gates
                         or "CONTEXT_NUMERIC_CONFLICT" in gates):
        score = 1.0                        # citations/figures can only downgrade

    return CorrectnessReport(
        score=score,
        gold_recall=round(gold_recall, 4),
        content_f1=round(content_f1, 4),
        answer_precision=round(answer_precision, 4),
        numeric=numeric,
        assessable=True,
        reason="gold_answer",
        gates=tuple(gates),
    )


# --- top-level score ---------------------------------------------------------

@dataclass(frozen=True)
class ResponseScore:
    """Full breakdown for one (generation, gold question) pair.

    ``EvaluationRecord`` has fixed fields, so everything beyond the four scored
    fields is serialised into ``notes`` as ``key=value`` pairs; ``parse_notes``
    reads it back. ``to_dict`` gives the lossless version for a validation harness.
    """

    question_id: str
    model_name: str
    retrieval_mode: str
    answerable: bool
    answerability_basis: str
    answerability_anomaly: str
    abstention: AbstentionDecision
    citation_precision: float
    citation_recall: float
    citation_scorable: bool
    citation_unscorable_reason: str
    groundedness: GroundednessReport
    context_source: str
    correctness: CorrectnessReport
    numeric_vs_context: Optional[NumericComparison]
    hallucinated: bool
    hallucination_types: Tuple[str, ...]
    hallucination_severity: int
    abstained_correctly: bool
    false_abstention: bool
    passage_quality: PassageQuality
    gold_citation_validity: str
    #: False when an answerable question was answered without naming any provision.
    #: A citation-recall failure, not by itself a hallucination - see
    #: UNCITED_ASSERTION in ``score_response``.
    citation_emitted: bool = True
    metric_status: str = METRIC_STATUS_UNVALIDATED
    unvalidated_fields: Tuple[str, ...] = UNVALIDATED_FIELDS

    @property
    def correctness_score(self) -> float:
        return self.correctness.score

    @property
    def notes(self) -> str:
        return format_notes(self)

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "question_id": self.question_id,
            "model_name": self.model_name,
            "retrieval_mode": self.retrieval_mode,
            "scorer": SCORER_NAME,
            "metric_status": self.metric_status,
            "unvalidated_fields": list(self.unvalidated_fields),
            "answerable": self.answerable,
            "answerability_basis": self.answerability_basis,
            "answerability_anomaly": self.answerability_anomaly,
            "abstained": self.abstention.abstained,
            "abstention_tier": self.abstention.tier,
            "abstention_reason": self.abstention.reason,
            "hedged_then_answered": self.abstention.hedged_then_answered,
            "abstained_correctly": self.abstained_correctly,
            "false_abstention": self.false_abstention,
            "citation_precision": self.citation_precision,
            "citation_recall": self.citation_recall,
            "citation_emitted": self.citation_emitted,
            "citation_scorable": self.citation_scorable,
            "citation_unscorable_reason": self.citation_unscorable_reason,
            "gold_citation_validity": self.gold_citation_validity,
            "context_source": self.context_source,
            "groundedness": self.groundedness.groundedness,
            "groundedness_assessable": self.groundedness.assessable,
            "groundedness_reason": self.groundedness.reason,
            "groundedness_sentences": {
                "content": self.groundedness.n_content_sentences,
                "supported": self.groundedness.n_supported,
                "partial": self.groundedness.n_partial,
                "unsupported": self.groundedness.n_unsupported,
                "conflict": self.groundedness.n_conflict,
            },
            "mean_containment": self.groundedness.mean_containment,
            "correctness_score": self.correctness.score,
            "correctness_assessable": self.correctness.assessable,
            "gold_recall": self.correctness.gold_recall,
            "content_f1": self.correctness.content_f1,
            "answer_precision": self.correctness.answer_precision,
            "correctness_gates": list(self.correctness.gates),
            "hallucinated": self.hallucinated,
            "hallucination_types": list(self.hallucination_types),
            "hallucination_severity": self.hallucination_severity,
            "gold_passage_quality": self.passage_quality.shape,
            "gold_passage_len": self.passage_quality.length,
            "gold_passage_bigram_overlap": self.passage_quality.bigram_overlap,
            "gold_passage_overlap_class": self.passage_quality.overlap_class,
            "gold_passage_stratum": self.passage_quality.stratum,
            "notes": self.notes,
        }
        if self.numeric_vs_context is not None:
            d["numeric_vs_context"] = self.numeric_vs_context.describe()
        return d

    def to_record(
        self,
        corpus_source: str = CORPUS_UNSET,
        retriever_backend: str = CORPUS_UNSET,
    ) -> EvaluationRecord:
        return EvaluationRecord(
            question_id=self.question_id,
            model_name=self.model_name,
            retrieval_mode=self.retrieval_mode,
            is_answerable=self.answerable,
            citation_precision=self.citation_precision,
            citation_recall=self.citation_recall,
            abstained=self.abstention.abstained,
            abstained_correctly=self.abstained_correctly,
            correctness_score=self.correctness.score,
            hallucinated=self.hallucinated,
            notes=self.notes,
            corpus_source=corpus_source,
            retriever_backend=retriever_backend,
            metric_status=self.metric_status,
            placeholder_fields=list(self.unvalidated_fields),
        )


def format_notes(s: ResponseScore) -> str:
    """Serialise the breakdown into ``notes`` as ``key=value`` pairs."""
    parts = [
        f"scorer={SCORER_NAME}",
        f"status={s.metric_status}",
        f"answerable={s.answerable}({s.answerability_basis})",
        f"abstention={s.abstention.reason}",
        f"context={s.context_source}",
        f"grounded={s.groundedness.describe()}",
        f"correct={s.correctness.describe()}",
        f"citation=P{s.citation_precision:.2f}/R{s.citation_recall:.2f}",
        f"hallucination={','.join(s.hallucination_types) or 'none'}"
        f"(sev={s.hallucination_severity})",
        f"gold_passage={s.passage_quality.describe()}",
        f"passage_overlap={s.passage_quality.describe_overlap()}",
        f"gold_citation_validity={s.gold_citation_validity}",
    ]
    if not s.citation_scorable:
        # Unscorable is not the same as 0.0: aggregation must drop these rows
        # rather than average a fabricated zero into citation precision.
        parts.append(f"citation_scorable=0:{s.citation_unscorable_reason or 'unknown'}")
    if "gold_doc_unresolved" in s.citation_unscorable_reason:
        parts.append("gold_doc_unresolved=1")
    if s.answerability_anomaly:
        parts.append(f"ANOMALY={s.answerability_anomaly}")
    if s.false_abstention:
        parts.append("FALSE_ABSTENTION=1")
    if not s.citation_emitted:
        parts.append("no_citation_emitted=1")
    if s.abstention.hedged_then_answered:
        parts.append("HEDGED_THEN_ANSWERED=1")
    if s.numeric_vs_context is not None:
        parts.append(f"numeric_vs_context={s.numeric_vs_context.describe()}")
    return "; ".join(parts)


_NOTE_SPLIT_RE = re.compile(r";\s*")


def parse_notes(notes: str) -> Dict[str, str]:
    """Read ``notes`` back into a dict, so results can be stratified downstream.

    Only flat ``key=value`` pairs are returned (``gold_passage=...``,
    ``gold_citation_validity=...``, ``ANOMALY=...``); nested parentheses stay
    inside the value.
    """
    out: Dict[str, str] = {}
    for part in _NOTE_SPLIT_RE.split(notes or ""):
        if "=" in part:
            k, v = part.split("=", 1)
            out[k.strip()] = v.strip()
    return out


def score_response(
    gen: GenerationResult,
    gold: GoldQuestion,
    retrieved: Optional[Sequence[RetrievedResult]] = None,
    contexts: Optional[Sequence[str]] = None,
    gold_citation_validity: Optional[str] = None,
) -> ResponseScore:
    """Score one generation against one gold question. Pure, deterministic, CPU-only.

    ``gold_citation_validity`` is the hook for the citation-gold audit: pass
    ``"invalid"`` for a row whose gold ``Điều N`` was shown not to exist in the
    instrument it cites, and article/instrument judgements are suppressed for that
    row instead of emitting a false hallucination. Default ``"unknown"`` keeps the
    judgement but tags it.
    """
    answer_text = normalize_ws(gen.answer_text or "")
    answerable, basis, anomaly = resolve_answerability(gold)
    validity = (gold_citation_validity or "unknown").strip() or "unknown"

    ctx_texts, ctx_source = resolve_contexts(retrieved, contexts, gen)

    abstention = detect_abstention(answer_text)
    groundedness = score_groundedness(answer_text, ctx_texts)
    passage_quality = assess_passage_quality(getattr(gold, "gold_passage", "") or "", ctx_texts)

    # --- citations (delegated, unmodified) ----------------------------------
    gold_citations = list(getattr(gold, "gold_citations", []) or [])
    extracted = extract_citations(answer_text)
    unresolved_gold_docs = [
        str(c.get("doc_id", "")) for c in gold_citations
        if str(c.get("doc_id", "")).upper().startswith("UNRESOLVED")
    ] + [d for d in (getattr(gold, "gold_doc_ids", []) or [])
         if str(d).upper().startswith("UNRESOLVED")]

    citation_scorable = True
    unscorable_reason = ""
    if not answerable:
        # A probe has NO gold citation, so precision/recall are undefined for it.
        # citation.py returns (0.0, 1.0) for "empty gold, citation emitted" - that
        # 1.0 recall is an artefact of an empty gold set, and averaging it in
        # would inflate the citation-recall column. The spurious citation is still
        # reported, as SPURIOUS_CITATION in the hallucination taxonomy.
        citation_scorable = False
        unscorable_reason = (
            "probe_with_gold_citations" if gold_citations else "probe_no_gold_citation"
        )
    elif not gold_citations:
        citation_scorable = False
        unscorable_reason = "no_gold_citation"
    elif validity == "invalid":
        citation_scorable = False
        unscorable_reason = "gold_citation_invalid"

    if citation_scorable:
        prec, rec = compute_citation_precision_recall(extracted, gold_citations)
    else:
        # Unscorable is not 0.0. Report the null and say why.
        prec, rec = 0.0, 0.0

    # Article-level gold exists for 64/64 CSV rows and stays usable even when the
    # instrument could not be resolved; doc-level judgement does not.
    gold_articles = {str(c.get("article_id", "")).strip() for c in gold_citations}
    gold_articles.discard("")
    gold_docs = {_norm_doc(str(c.get("doc_id", ""))) for c in gold_citations}
    gold_docs.discard("UNKNOWN")
    article_scoring_possible = bool(citation_scorable and answerable and gold_articles)
    doc_scoring_possible = bool(article_scoring_possible and gold_docs
                                and not unresolved_gold_docs)

    # --- hallucination taxonomy ---------------------------------------------
    types: List[str] = []

    if abstention.empty_answer:
        types.append(H_DEGENERATE_OUTPUT)

    if not answerable:
        if not abstention.abstained:
            types.append(H_ANSWERED_UNANSWERABLE)
            if extracted:
                types.append(H_SPURIOUS_CITATION)
    else:
        if not abstention.abstained:
            if article_scoring_possible:
                for c in extracted:
                    doc = _norm_doc(str(c.get("doc_id", "")))
                    art = str(c.get("article_id", "")).strip()
                    if (doc_scoring_possible and doc != "UNKNOWN" and gold_docs
                            and doc not in gold_docs
                            and H_WRONG_INSTRUMENT not in types):
                        types.append(H_WRONG_INSTRUMENT)
                    if art and art not in gold_articles:
                        if H_NON_COVERING_PROVISION not in types:
                            types.append(H_NON_COVERING_PROVISION)

        if groundedness.assessable and groundedness.groundedness is not None:
            if groundedness.n_unsupported > 0:
                types.append(H_UNSUPPORTED_CLAIM)
            if groundedness.n_conflict > 0:
                types.append(H_CONTRADICTS_CONTEXT)

    # numeric agreement against the retrieved context (RAG modes only)
    numeric_vs_context: Optional[NumericComparison] = None
    if ctx_texts and not abstention.abstained:
        numeric_vs_context = compare_numeric_facts(
            extract_numeric_facts(" ".join(ctx_texts)),
            extract_numeric_facts(answer_text),
        )
        if numeric_vs_context.has_conflict:
            types.append(H_CONTRADICTS_CONTEXT)

    # numeric agreement against the gold answer (all modes). This is the check
    # that catches "right Điều, wrong figure": the citation is perfect and the
    # prose is fluent, but 50 triệu đồng is not 30 triệu đồng.
    if answerable and not abstention.abstained:
        gold_facts = extract_numeric_facts(getattr(gold, "reference_answer", "") or "")
        if gold_facts:
            cmp_gold = compare_numeric_facts(gold_facts, extract_numeric_facts(answer_text))
            if cmp_gold.has_conflict:
                types.append(H_NUMERIC_THRESHOLD_MISMATCH)

    # --- correctness (computed before the taxonomy is closed, because one
    # --- hallucination type is defined in terms of gold agreement) -----------
    citation_error = bool(
        article_scoring_possible
        and not abstention.abstained
        and (H_WRONG_INSTRUMENT in types or H_NON_COVERING_PROVISION in types)
    )
    correctness = score_correctness(
        answer_text=answer_text,
        gold_answer=getattr(gold, "reference_answer", "") or "",
        abstention=abstention,
        numeric_vs_context=numeric_vs_context,
        citation_error=citation_error,
        answerable=answerable,
    )

    # UNCITED_ASSERTION is deliberately evaluated AFTER correctness. The failure
    # this type exists for is a fluent FABRICATION carrying no citation - the case
    # the placeholder scorer let through unflagged. A correct answer that simply
    # omits the citation is a citation-recall failure, not a hallucination, so it
    # is recorded as a flag and left out of the taxonomy; otherwise every
    # closed-book row without an "Điều N" would be counted as a hallucination and
    # RQ1 would measure citation style instead of unfaithfulness.
    uncited = bool(
        answerable and not abstention.abstained and not abstention.empty_answer
        and not extracted
    )
    content_unverified = bool(
        correctness.score == 0.0
        or groundedness.n_unsupported > 0
        or groundedness.n_conflict > 0
    )
    if uncited and content_unverified:
        types.append(H_UNCITED_ASSERTION)

    if (
        answerable
        and not abstention.abstained
        and not abstention.empty_answer
        and correctness.assessable
        and correctness.score == 0.0
    ):
        # The model asserted an answer and the assertion does not state what the
        # human gold answer states. Reported separately from UNSUPPORTED_CLAIM
        # (which is about the retrieved context) because the two references are
        # independent: an answer can be grounded in a chunk that does not answer
        # the question, and can agree with the gold answer while ignoring the
        # context. Both are recorded so the paper can report either definition.
        #
        # Known cost, stated plainly: a fully correct answer in fresh wording that
        # the lexical comparison under-credits to 0.0 is also flagged. That is the
        # paraphrase blind spot of every scorer in this module, it is the single
        # most important thing for the ~50-label human validation to measure, and
        # it is why metric_status stays "unvalidated".
        types.append(H_UNSUPPORTED_BY_GOLD)

    types = list(dict.fromkeys(types))
    hallucinated = bool(types)
    severity = max((SEVERITY.get(t, 1) for t in types), default=0)

    abstained_correctly = (not answerable) and abstention.abstained
    false_abstention = answerable and abstention.abstained

    return ResponseScore(
        question_id=gold.id,
        model_name=gen.model_name,
        retrieval_mode=gen.retrieval_mode,
        answerable=answerable,
        answerability_basis=basis,
        answerability_anomaly=anomaly,
        abstention=abstention,
        citation_precision=prec,
        citation_recall=rec,
        citation_scorable=citation_scorable,
        citation_unscorable_reason=unscorable_reason or (
            "gold_doc_unresolved" if unresolved_gold_docs else ""
        ),
        groundedness=groundedness,
        context_source=ctx_source,
        correctness=correctness,
        numeric_vs_context=numeric_vs_context,
        hallucinated=hallucinated,
        hallucination_types=tuple(types),
        hallucination_severity=severity,
        abstained_correctly=abstained_correctly,
        false_abstention=false_abstention,
        passage_quality=passage_quality,
        gold_citation_validity=validity,
        citation_emitted=bool(extracted) or abstention.abstained or not answerable,
    )


def _norm_doc(doc: str) -> str:
    """Normalise an instrument id for set comparison (mirrors citation.py's rule)."""
    d = normalize_ws(doc or "").upper()
    if not d or d == "UNKNOWN":
        return "UNKNOWN"
    d = d.replace("ND-CP", "NĐ-CP").replace("-", "/")
    parts = d.split("/")
    if len(parts) >= 2 and parts[0].isdigit() and len(parts[0]) == 1:
        parts[0] = "0" + parts[0]
    return "/".join(parts)


def evaluate_response(
    gen: GenerationResult,
    gold: GoldQuestion,
    retrieved: Optional[Sequence[RetrievedResult]] = None,
    contexts: Optional[Sequence[str]] = None,
    gold_citation_validity: Optional[str] = None,
) -> EvaluationRecord:
    """Score a generation and return the persistable ``EvaluationRecord``.

    Backwards-compatible: the two-argument call still works. Without a retrieval
    the groundedness fields are reported as unassessable in ``notes`` rather than
    defaulted to zero.
    """
    score = score_response(
        gen, gold, retrieved=retrieved, contexts=contexts,
        gold_citation_validity=gold_citation_validity,
    )
    return score.to_record(
        corpus_source=getattr(gen, "corpus_source", CORPUS_UNSET),
        retriever_backend=getattr(gen, "retriever_backend", CORPUS_UNSET),
    )


# --- aggregation, with the metric-status guard -------------------------------

def assert_metrics_publishable(records: Sequence[EvaluationRecord]) -> None:
    """Refuse to aggregate rows whose SCORER is not final.

    ``provenance.assert_publishable`` guards corpus and retriever provenance; it
    does not look at ``metric_status``, so it would happily average the numbers in
    this module into a table. This calls that guard FIRST (it is not weakened or
    bypassed) and then adds the missing check. Raises ``ProvenanceError``.
    """
    assert_publishable(records)
    bad = [
        f"{getattr(r, 'question_id', '?')}/{getattr(r, 'model_name', '?')}"
        f"/{getattr(r, 'retrieval_mode', '?')}: metric_status={getattr(r, 'metric_status', 'unset')}"
        for r in records
        if getattr(r, "metric_status", "unset") != METRIC_STATUS_FINAL
    ]
    if bad:
        raise ProvenanceError(
            f"{len(bad)} row(s) were scored by an UNVALIDATED scorer "
            f"({SCORER_NAME}) and cannot be aggregated into a paper table:\n  "
            + "\n  ".join(bad[:20])
            + "\nSet metric_status='final' only after validation_report() has "
              "measured agreement with human labels on the labelled subset."
        )


def compute_abstention_metrics(
    records: Sequence[EvaluationRecord], strict: bool = True
) -> Dict[str, Any]:
    """Binary abstention classification over the whole benchmark (RQ2).

    Positive class = "the model abstained". Over the 23 probes a correct refusal is
    a true positive; over the 65 answerable questions a refusal is a FALSE
    ABSTENTION (a recall failure the placeholder scorer had no name for).

    ``strict=True`` refuses to aggregate unvalidated rows. ``strict=False`` still
    computes, but stamps ``provenance="REFUSED_UNVALIDATED"`` on the result so a
    development number cannot be pasted into the paper unlabelled.
    """
    if strict:
        assert_metrics_publishable(records)
    tp = fp = tn = fn = 0
    hedged = 0
    for r in records:
        ans = bool(getattr(r, "is_answerable", True))
        abst = bool(getattr(r, "abstained", False))
        if not ans and abst:
            tp += 1
        elif ans and abst:
            fp += 1                        # false abstention
        elif ans and not abst:
            tn += 1
        else:
            fn += 1                          # answered an unanswerable probe
        if "HEDGED_THEN_ANSWERED=1" in (getattr(r, "notes", "") or ""):
            hedged += 1
    n = len(records)
    n_probes = tp + fn
    n_answerable = tn + fp
    out: Dict[str, Any] = {
        "rows": n,
        "probes": n_probes,
        "answerable": n_answerable,
        "true_positive": tp,
        "false_abstention": fp,
        "true_negative": tn,
        "answered_probe": fn,
        "abstention_accuracy": round((tp + tn) / n, 4) if n else None,
        "probe_abstention_rate": round(tp / n_probes, 4) if n_probes else None,
        "false_abstention_rate": round(fp / n_answerable, 4) if n_answerable else None,
        "abstention_precision": round(tp / (tp + fp), 4) if (tp + fp) else None,
        "abstention_recall": round(tp / (tp + fn), 4) if (tp + fn) else None,
        "hedged_then_answered": hedged,
        "scorer": SCORER_NAME,
    }
    if not strict:
        out["provenance"] = "REFUSED_UNVALIDATED"
    return out


def summarise_records(
    records: Sequence[EvaluationRecord], strict: bool = True
) -> Dict[str, Any]:
    """Per-(model, mode) summary, STRATIFIED by gold-passage quality.

    Three aggregation rules, each of which exists to stop a fabricated zero from
    being averaged into a headline number:

    * groundedness is averaged only over rows where it was ASSESSABLE, and the
      number of assessable rows is reported next to the mean (a closed-book row
      has no retrieved context, so it has no groundedness - that is not 0.0);
    * citation precision/recall are averaged only over rows whose citation gold
      was SCORABLE (``citation_scorable=0`` rows are counted, not averaged);
    * rows whose correctness was UNASSESSABLE (``correct=NA(...)`` in notes) are
      counted and left out of the correctness mean instead of being averaged in as
      a fabricated 0.0, ``ANOMALY`` rows are counted, and the gold-passage strata
      are reported separately so summary-shaped rows are never silently pooled
      with verbatim ones.
    """
    if strict:
        assert_metrics_publishable(records)

    groups: Dict[Tuple[str, str], List[EvaluationRecord]] = {}
    for r in records:
        groups.setdefault((r.model_name, r.retrieval_mode), []).append(r)

    summary: Dict[str, Any] = {"scorer": SCORER_NAME, "groups": {}}
    if not strict:
        summary["provenance"] = "REFUSED_UNVALIDATED"

    for (model, mode), recs in sorted(groups.items()):
        n = len(recs)
        grounded_vals: List[float] = []
        grounded_by_stratum: Dict[str, List[float]] = {}
        citation_rows: List[EvaluationRecord] = []
        correctness_rows: List[EvaluationRecord] = []
        strata: Dict[str, int] = {}
        anomalies = 0
        unscorable_citation = 0
        unresolved_doc = 0
        correctness_unassessable = 0
        for r in recs:
            notes = parse_notes(getattr(r, "notes", "") or "")
            shape = (notes.get("gold_passage") or "unknown").split("(")[0]
            ovl = notes.get("passage_overlap") or ""
            ovl_class = ovl.split("/")[-1].split("(")[0] if ovl else "unknown"
            stratum = f"{shape}/{ovl_class}"
            strata[stratum] = strata.get(stratum, 0) + 1
            if notes.get("ANOMALY"):
                anomalies += 1
            if notes.get("correct", "").startswith("NA("):
                correctness_unassessable += 1
            else:
                correctness_rows.append(r)
            if notes.get("citation_scorable", "1").startswith("0"):
                unscorable_citation += 1
            else:
                citation_rows.append(r)
            if notes.get("gold_doc_unresolved"):
                unresolved_doc += 1
            m = re.match(r"^(\d+\.\d+)\(", notes.get("grounded", ""))
            if m:
                val = float(m.group(1))
                grounded_vals.append(val)
                grounded_by_stratum.setdefault(stratum, []).append(val)

        def _mean(vals: Sequence[float]) -> Optional[float]:
            return round(sum(vals) / len(vals), 4) if vals else None

        group: Dict[str, Any] = {
            "count": n,
            "citation_precision": _mean([r.citation_precision for r in citation_rows]),
            "citation_recall": _mean([r.citation_recall for r in citation_rows]),
            "citation_scorable_rows": len(citation_rows),
            "citation_unscorable_rows": unscorable_citation,
            "gold_doc_unresolved_rows": unresolved_doc,
            "hallucination_rate": round(sum(1 for r in recs if r.hallucinated) / n, 4),
            "avg_correctness": _mean([r.correctness_score for r in correctness_rows]),
            "correctness_rows": len(correctness_rows),
            "correctness_unassessable_rows": correctness_unassessable,
            "groundedness_mean": _mean(grounded_vals),
            "groundedness_assessable_rows": len(grounded_vals),
            # Per-stratum groundedness: rows whose gold cell is an annotator
            # summary are reported separately from rows whose cell was measured
            # to be in the retrieved chunk, so the two are never pooled.
            "groundedness_by_gold_passage_stratum": {
                k: {"mean": _mean(v), "rows": len(v)}
                for k, v in sorted(grounded_by_stratum.items())
            },
            "gold_passage_strata": strata,
            "answerability_anomalies": anomalies,
        }
        abst = compute_abstention_metrics(recs, strict=strict)
        group.update(
            {k: abst[k] for k in ("abstention_accuracy", "false_abstention_rate",
                                  "probe_abstention_rate", "hedged_then_answered",
                                  "false_abstention", "answered_probe")}
        )
        summary["groups"][f"{model}::{mode}"] = group
    return summary


# --- human validation --------------------------------------------------------

def validation_report(pairs: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Turn a human-labelled subset into the agreement numbers the paper reports.

    ``pairs`` is one dict per labelled response, with a human judgement and the
    scorer's judgement for the same response::

        {"question_id": "Q021",
         "human_correct": True,   "auto_correct": True,     # bool
         "human_hallucinated": False, "auto_hallucinated": True,
         "human_abstained": False, "auto_abstained": False}

    ``auto_correct`` should be ``record.correctness_score == 2.0`` (the human
    judges a binary right/wrong, so the three-level scale is collapsed at the
    boundary the labeler was given). Reports raw agreement and Cohen's kappa per
    field via the existing ``agreement.compute_cohens_kappa``, plus the confusion
    counts, so a disagreement direction is visible and not just a coefficient.

    Raises ``ProvenanceError`` if fewer than two pairs or if any pair is missing a
    field - a half-labelled subset must not silently produce a smaller kappa.
    """
    require(len(pairs) >= 2, "validation_report: need at least 2 labelled pairs.")
    fields = ("correct", "hallucinated", "abstained")
    report: Dict[str, Any] = {"n": len(pairs), "scorer": SCORER_NAME, "fields": {}}

    for fname in fields:
        hk, ak = f"human_{fname}", f"auto_{fname}"
        missing = [p.get("question_id", "?") for p in pairs if hk not in p or ak not in p]
        require(not missing,
                f"validation_report: {len(missing)} pair(s) missing {hk}/{ak}: {missing[:5]}")
        human = ["yes" if bool(p[hk]) else "no" for p in pairs]
        auto = ["yes" if bool(p[ak]) else "no" for p in pairs]
        agree = sum(1 for h, a in zip(human, auto) if h == a)
        conf = {
            "both_yes": sum(1 for h, a in zip(human, auto) if h == "yes" and a == "yes"),
            "human_yes_auto_no": sum(1 for h, a in zip(human, auto) if h == "yes" and a == "no"),
            "human_no_auto_yes": sum(1 for h, a in zip(human, auto) if h == "no" and a == "yes"),
            "both_no": sum(1 for h, a in zip(human, auto) if h == "no" and a == "no"),
        }
        report["fields"][fname] = {
            "raw_agreement": round(agree / len(pairs), 4),
            "cohens_kappa": compute_cohens_kappa(human, auto),
            "confusion": conf,
            "disagreements": [
                p.get("question_id", "?") for h, a, p in zip(human, auto, pairs) if h != a
            ],
        }
    return report


# --- calibration record and the embedding upgrade ---------------------------
#
# Measured on data/gold/bank_qa_data.csv (88 rows; 65 answerable, 23 probes),
# 2026-09-11, CPU only, no model loaded:
#
#   gold answer vs itself                     gold_recall 1.00, f1 1.00 -> 2.0 for 65/65 rows
#   gold answer vs another row's gold answer  gold_recall p50 0.08 p90 0.36 p95 0.44
#                                             content_f1   p50 0.08 p95 0.26 max 0.40
#                                             -> 137/195 pairs 0.0, 54 partial, 4 full
#   answer sentence vs its OWN gold passage   containment p25 0.50 med 0.68
#   answer sentence vs ANOTHER row's passage  containment p25 0.05 med 0.11 p90 ~0.35
#
# The full/partial cut-offs sit between the negative p95 and the positive p25, so
# they are placed in a measured gap rather than chosen by eye. Two limits are
# stated here because they belong in the paper's threats-to-validity paragraph:
#   * the negative controls are not clean negatives - the CSV holds sibling
#     questions on the same topic, so a few cross-matched pairs are genuinely
#     correct answers (the 4 that scored 2.0);
#   * the positive containment distribution is pessimistic, because 46 of the 65
#     gold cells are annotator summaries rather than source text (see the
#     gold_passage block above).
#
# RECOMMENDATION (not implemented here, deliberately): the right long-term
# replacement for the two lexical agreement signals is a multilingual NLI or
# embedding scorer - groundedness as entailment of each answer sentence by the
# retrieved chunk, correctness as semantic similarity to the gold answer. That
# removes the paraphrase blind spot, which is this module's largest known error
# source. It must NOT run on the reference machine (12-core laptop, 4 GB GPU; a
# previous local torch run froze it), so it belongs on the Colab/RunPod side of
# the pipeline, as a second scorer whose agreement with this one is reported
# alongside the human agreement - not as a silent replacement. Until then every
# number here is tagged metric_status="unvalidated".
