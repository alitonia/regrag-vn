"""Provenance sentinels and publishability guards.

Every artifact that can reach a paper table carries a provenance tag. A tag that
is UNSET or DEGRADED means the number was produced without the real dependency
(missing sentence-transformers, missing pyvi, fixture corpus instead of the
ingested one). Such rows must be refused by aggregation rather than averaged in.

This module exists because silent substitution previously produced a plausible
but fictional dense-retrieval column: DenseIndex.search() returned the first
top_k chunks in corpus order with fabricated scores whenever its embedding
backend was unavailable.
"""

from typing import Iterable, List, Sequence

# --- corpus provenance -------------------------------------------------------
CORPUS_TIER1 = "tier1_passages"   # fixture built from CSV gold passages; never publishable
CORPUS_TIER2 = "tier2_full"       # segmented full legal instruments; the reported corpus
CORPUS_UNSET = "UNSET"

# --- retriever backend provenance -------------------------------------------
BACKEND_DEGRADED = "DEGRADED"     # prefix; real value is "DEGRADED:<reason>"


def degraded(reason: str) -> str:
    """Build a DEGRADED backend tag carrying the reason."""
    return f"{BACKEND_DEGRADED}:{reason}"


def is_degraded(backend: str) -> bool:
    return not backend or backend == CORPUS_UNSET or backend.startswith(BACKEND_DEGRADED)


def publishable_corpus(corpus_source: str) -> bool:
    return corpus_source == CORPUS_TIER2


class ProvenanceError(RuntimeError):
    """Raised when unpublishable rows are fed to aggregation."""


def assert_publishable(
    records: Sequence[object],
    backend_attr: str = "retriever_backend",
    corpus_attr: str = "corpus_source",
) -> List[str]:
    """Refuse to aggregate rows with UNSET/DEGRADED provenance.

    Returns the list of offending descriptions when called in report mode;
    raises ProvenanceError so a bad table cannot be produced silently.
    """
    offenders: Iterable[str] = []
    bad = []
    for r in records:
        backend = getattr(r, backend_attr, CORPUS_UNSET)
        corpus = getattr(r, corpus_attr, CORPUS_UNSET)
        if is_degraded(backend) or not publishable_corpus(corpus):
            bad.append(
                f"{getattr(r, 'question_id', '?')}/{getattr(r, 'model_name', '?')}"
                f"/{getattr(r, 'retrieval_mode', '?')}: backend={backend} corpus={corpus}"
            )
    offenders = bad
    if offenders:
        raise ProvenanceError(
            f"{len(list(offenders))} row(s) carry unpublishable provenance:\n  "
            + "\n  ".join(list(offenders)[:20])
        )
    return []


def require(condition: bool, message: str) -> None:
    """Loud precondition check. Never returns a plausible default."""
    if not condition:
        raise ProvenanceError(message)
