#!/usr/bin/env python3
"""Analyze the 50-response scorer-validation sheet (Sec. IV-C).

Reads data/human/scorer_validation_50.csv, compares the deterministic
scorer's per-row verdicts (auto_correct / auto_hallucinated /
auto_abstained) against the expert labels (human_*), and writes a dated
JSON log to data/eval/. Cohen's kappa uses the standard formula; the
expert is treated as the reference for precision/recall.

Usage:
    .venv/bin/python scripts/analyze_scorer_validation.py [--out PATH]
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SHEET = ROOT / "data" / "human" / "scorer_validation_50.csv"
DEFAULT_OUT = ROOT / "data" / "eval" / f"scorer_validation_{date.today().isoformat()}.json"

METRICS = ("correct", "hallucinated", "abstained")
BOOL_TOKENS = {"1": True, "true": True, "yes": True, "0": False, "false": False, "no": False}


def as_bool(tok: str) -> bool:
    return BOOL_TOKENS[tok.strip().lower()]


def cohens_kappa(a: list[bool], b: list[bool]) -> float:
    n = len(a)
    cats = sorted(set(a) | set(b))
    po = sum(1 for x, y in zip(a, b) if x == y) / n
    ca, cb = Counter(a), Counter(b)
    pe = sum((ca[c] / n) * (cb[c] / n) for c in cats)
    return (po - pe) / (1 - pe) if pe < 1 else float("nan")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sheet", type=Path, default=SHEET)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    with open(args.sheet, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    per_metric = {}
    for m in METRICS:
        human = [as_bool(r[f"human_{m}"]) for r in rows]
        auto = [as_bool(r[f"auto_{m}"]) for r in rows]
        conf = Counter(zip(human, auto))
        tp, fn = conf[(True, True)], conf[(True, False)]
        fp, tn = conf[(False, True)], conf[(False, False)]
        agree = tp + tn
        per_metric[m] = {
            "n": len(rows),
            "human_positive": sum(human),
            "auto_positive": sum(auto),
            "agreement": agree / len(rows),
            "cohens_kappa": cohens_kappa(human, auto),
            "confusion": {"tp": tp, "fn": fn, "fp": fp, "tn": tn},
            "auto_precision_vs_expert": (tp / (tp + fp)) if (tp + fp) else None,
            "auto_recall_vs_expert": (tp / (tp + fn)) if (tp + fn) else None,
        }

    report = {
        "analysis": "scorer_validation_50",
        "date": date.today().isoformat(),
        "sheet": str(args.sheet.relative_to(ROOT)),
        "n_rows": len(rows),
        "composition": {
            "models": dict(Counter(r["model_name"] for r in rows)),
            "modes": dict(Counter(r["retrieval_mode"] for r in rows)),
            "answerable": dict(Counter(as_bool(r["is_answerable"]) for r in rows)),
        },
        "per_metric": per_metric,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    for m, s in per_metric.items():
        print(
            f"{m:12s} agree={s['agreement']:.2f} kappa={s['cohens_kappa']:.3f} "
            f"human+={s['human_positive']:2d} auto+={s['auto_positive']:2d} "
            f"tp={s['confusion']['tp']:2d} fp={s['confusion']['fp']:2d} "
            f"fn={s['confusion']['fn']:2d} tn={s['confusion']['tn']:2d}"
        )
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
