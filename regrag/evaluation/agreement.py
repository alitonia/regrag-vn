"""Inter-annotator agreement metrics for the 100 gold question benchmark.

Calculates Cohen's Kappa for categorical grading between two independent annotators.
"""

from typing import List, Dict, Any


def compute_cohens_kappa(rater1: List[str], rater2: List[str]) -> float:
    """Compute Cohen's Kappa coefficient between two raters.

    Args:
        rater1: List of category ratings from Annotator 1.
        rater2: List of category ratings from Annotator 2.

    Returns:
        Cohen's Kappa score (-1.0 to 1.0).
    """
    if len(rater1) != len(rater2) or not rater1:
        raise ValueError("Rater lists must be non-empty and of identical length.")

    n = len(rater1)
    categories = sorted(list(set(rater1).union(set(rater2))))

    # Observed agreement (Po)
    po = sum(1 for a, b in zip(rater1, rater2) if a == b) / n

    # Expected agreement by chance (Pe)
    pe = 0.0
    for cat in categories:
        p1 = sum(1 for a in rater1 if a == cat) / n
        p2 = sum(1 for b in rater2 if b == cat) / n
        pe += p1 * p2

    if pe == 1.0:
        return 1.0

    kappa = (po - pe) / (1.0 - pe)
    return round(kappa, 4)
