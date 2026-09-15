"""Abstention detection and the cross-module abstention compatibility guard.

Why this module exists
----------------------
RQ2 (abstention) is measured on the 23 unanswerable probes in
``data/gold/bank_qa_data.csv``. Those rows carry the gold answer string

    "Không có trong kho văn bản"

but the prompt template in ``regrag/generation/prompts.py`` instructs the model
to refuse with a *different* string,

    ABSTENTION_KEYPHRASE = "THÔNG TIN KHÔNG CÓ TRONG TÀI LIỆU"

and ``regrag/evaluation/metrics.py`` tests only for that keyphrase plus
"không có trong tài liệu" and "không tìm thấy thông tin". None of those match
the gold sentinel. Left alone, every correct refusal would be scored as a
non-abstention and RQ2 would come out inverted with no warning - exactly the
silent-substitution failure class docs/PROJECT_PLAN.md §5 exists to prevent.

Two-tier detection
------------------
``EXACT``  - strings the benchmark itself defines (the gold sentinel and the
             prompted keyphrase). High precision. These set ``abstained``.
``SOFT``   - free paraphrases a model might produce. Reported as a diagnostic
             count for human review only, deliberately NOT folded into
             ``abstained``, because e.g. "không được quy định" is also a
             legitimate substantive answer to an answerable question.

``assert_metrics_detects_sentinel()`` is the loud guard: it exercises the real
``regrag.evaluation.metrics.evaluate_response`` on a synthetic sentinel refusal
and raises ``ProvenanceError`` if the scorer would miss it. It is CPU-only and
downloads nothing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

from regrag.corpus.canonical import fold_diacritics, normalize_ws
from regrag.generation.config import UNANSWERABLE_SENTINEL
from regrag.generation.prompts import ABSTENTION_KEYPHRASE
from regrag.provenance import ProvenanceError


def _fold(text: str) -> str:
    """Case/whitespace/diacritic-fold for robust substring matching.

    Diacritic folding is safe here because it is applied to *both* the pattern
    and the answer, and every pattern is a multi-word Vietnamese phrase long
    enough that folding cannot create a false positive out of ordinary prose.
    """
    return fold_diacritics(normalize_ws(text or "")).casefold()


# Strings the benchmark itself defines. Matching one of these IS an abstention.
EXACT_PATTERNS: Tuple[str, ...] = (
    _fold(UNANSWERABLE_SENTINEL),
    _fold(ABSTENTION_KEYPHRASE),
)

# Plausible paraphrases. Diagnostic only - see module docstring.
SOFT_PATTERNS: Tuple[str, ...] = (
    _fold("không có trong tài liệu"),
    _fold("không tìm thấy thông tin"),
    _fold("không có thông tin"),
    _fold("không đủ thông tin để trả lời"),
    _fold("không đủ thông tin"),
    _fold("không thể trả lời"),
    _fold("không có cơ sở"),
    _fold("không nằm trong phạm vi"),
    _fold("ngoài phạm vi"),
    _fold("không có trong kho"),
    _fold("không rõ"),
)


@dataclass
class AbstentionVerdict:
    """Result of abstention detection on one model answer."""

    abstained: bool
    matched_exact: Optional[str] = None
    matched_soft: List[str] = field(default_factory=list)

    @property
    def soft_only(self) -> bool:
        """True when a paraphrase matched but no benchmark-defined string did.

        These rows are the ones a human should look at: the model probably
        refused, but not in the registered wording, so the scorer may disagree.
        """
        return (not self.abstained) and bool(self.matched_soft)


def detect_abstention(answer_text: str) -> AbstentionVerdict:
    """Detect a refusal in a model answer. Pure function, no I/O."""
    folded = _fold(answer_text)
    if not folded:
        # An empty answer is NOT an abstention - it is a failed generation.
        # Scoring it as a refusal would hand the model credit for producing
        # nothing, so it is reported as non-abstention and flagged separately.
        return AbstentionVerdict(abstained=False)

    for pat in EXACT_PATTERNS:
        if pat and pat in folded:
            return AbstentionVerdict(abstained=True, matched_exact=pat)

    soft = [pat for pat in SOFT_PATTERNS if pat and pat in folded]
    return AbstentionVerdict(abstained=False, matched_soft=soft)


def summarize_verdicts(verdicts: Sequence[AbstentionVerdict]) -> dict:
    """Aggregate verdicts for the run manifest."""
    n = len(verdicts)
    return {
        "rows": n,
        "abstained_exact": sum(1 for v in verdicts if v.abstained),
        "soft_only_needs_human_review": sum(1 for v in verdicts if v.soft_only),
        "empty_answer": sum(1 for v in verdicts if not v.abstained and not v.matched_soft),
    }


def assert_metrics_detects_sentinel(gold_is_answerable: bool = False) -> None:
    """Fail loudly if regrag.evaluation.metrics cannot see the gold sentinel.

    This is a real cross-module check, not a unit test of this file: it imports
    the *actual* scorer and feeds it a synthetic refusal whose text is exactly
    the gold sentinel. If the scorer reports ``abstained=False``, the campaign
    would produce 23 x models x modes rows whose abstention is scored with the
    sign inverted, and nothing downstream would say so.

    Raises ``ProvenanceError`` naming the file that must change.
    """
    # Imported lazily so this module stays cheap and side-effect free.
    from regrag.evaluation.metrics import evaluate_response
    from regrag.models import GenerationResult, GoldQuestion

    gen = GenerationResult(
        question_id="Q_PROBE_COMPAT",
        model_name="compat-probe",
        retrieval_mode="closed_book",
        prompt="<compat probe>",
        raw_response=UNANSWERABLE_SENTINEL,
        answer_text=UNANSWERABLE_SENTINEL,
    )
    gold = GoldQuestion(
        id="Q_PROBE_COMPAT",
        question="<compat probe>",
        is_answerable=gold_is_answerable,
        reference_answer=UNANSWERABLE_SENTINEL,
        category="unanswerable",
    )
    record = evaluate_response(gen, gold)

    if not record.abstained:
        raise ProvenanceError(
            "ABSTENTION SCORER MISMATCH: regrag/evaluation/metrics.py did not "
            f"detect the gold unanswerable sentinel {UNANSWERABLE_SENTINEL!r} as an "
            "abstention. Every unanswerable probe would be scored as a "
            "non-abstention and RQ2 would be inverted. Fix the abstention check "
            "in regrag/evaluation/metrics.py to use "
            "regrag.generation.abstention.detect_abstention (or add the sentinel "
            "to its keyphrase list) before running the campaign. "
            "Note: metrics.py is outside this harness's write scope."
        )
    if not gold_is_answerable and not record.abstained_correctly:
        raise ProvenanceError(
            "ABSTENTION SCORER MISMATCH: the sentinel was detected as an "
            "abstention but not scored as abstained_correctly for an "
            "unanswerable probe. RQ2 abstention accuracy would be wrong."
        )
