#!/usr/bin/env python3
"""Unblind the 2026-09-15 RIVF blind-rank re-eval (ours = R33).

Reads the three scorer CSVs (armA deepseek std, armB glm std, armC gemini
hostile), computes strict percentiles for ours per metric per arm (share of
corpus entries ours beats outright), corpus distribution context, the arm
spread, and the paired delta against the 2026-09-13 run's R33 scores.

Usage: .venv/bin/python scripts/unblind_rivf_reeval.py
"""
from __future__ import annotations

import csv
from pathlib import Path
from statistics import median

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "reports" / "blind_rank_rivf2026"
ARMS = {
    "A_deepseek_std": "scores_reeval_armA_deepseek_2026-09-15.csv",
    "B_glm_std": "scores_reeval_armB_glm_2026-09-15.csv",
    "C_gemini_hostile": "scores_reeval_armC_gemini_2026-09-15.csv",
}
METRICS = ("novelty", "rigor", "evidence", "reproducibility", "clarity", "significance", "overall")
OLD = {"A_deepseek_std": 7, "B_glm_std": 6, "C_gemini_hostile": 5}  # 2026-09-13 overall for R33


def load(path: Path) -> dict[str, dict[str, str]]:
    with open(path, encoding="utf-8") as f:
        return {r["id"]: r for r in csv.DictReader(f)}


def strict_pct(values: list[int], ours: int) -> float:
    return 100.0 * sum(1 for v in values if v < ours) / len(values)


def main() -> None:
    present = {k: BASE / v for k, v in ARMS.items() if (BASE / v).exists()}
    missing = [k for k in ARMS if k not in present]
    if missing:
        print(f"WARNING: arms not yet landed: {missing} (percentiles below are partial)")

    rows_out = []
    for arm, path in present.items():
        data = load(path)
        if "R33" not in data or len(data) < 45:
            print(f"SKIP {arm}: incomplete ({len(data)} rows, R33 present: {'R33' in data})")
            continue
        ours = data.pop("R33")
        n = len(data)
        line = {"arm": arm, "n": n, "verdict": ours["verdict"], "note": ours["note"][:110]}
        for m in METRICS:
            vals = [int(r[m]) for r in data.values()]
            o = int(ours[m])
            line[m] = o
            line[f"{m}_pct"] = round(strict_pct(vals, o), 1)
            line[f"{m}_med"] = median(vals)
        rows_out.append(line)

    print(f"{'arm':<18}{'n':>3}  overall  pct   med   verdict          delta_vs_0913")
    for r in rows_out:
        print(f"{r['arm']:<18}{r['n']:>3}  {r['overall']:>7}  {r['overall_pct']:>4}  {r['overall_med']:>4}   {r['verdict']:<15}  {r['overall'] - OLD.get(r['arm'], 0):+d}")
    if len(rows_out) >= 2:
        overs = [r["overall"] for r in rows_out]
        pcts = [r["overall_pct"] for r in rows_out]
        print(f"\noverall spread: {min(overs)}-{max(overs)} | strict-pct spread: {min(pcts)}-{max(pcts)}")
    for r in rows_out:
        print(f"\n[{r['arm']}] R33: " + ", ".join(f"{m}={r[m]}({r[f'{m}_pct']}%)" for m in METRICS))
        print(f"  note: {r['note']}")


if __name__ == "__main__":
    main()
