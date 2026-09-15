"""Evaluation module for citations, metrics, and inter-annotator agreement."""

from regrag.evaluation.citation import (
    extract_citations,
    compute_citation_precision_recall,
)
from regrag.evaluation.agreement import compute_cohens_kappa
from regrag.evaluation.metrics import evaluate_response

__all__ = [
    "extract_citations",
    "compute_citation_precision_recall",
    "compute_cohens_kappa",
    "evaluate_response",
]
