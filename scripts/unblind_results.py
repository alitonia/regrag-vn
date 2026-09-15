#!/usr/bin/env python3
"""
Unblinds and analyzes blind-rank scoring results for RIVF 2026 and EASE 2027.
Computes strict rank, percentiles, corpus statistics, and arm spreads
for both standard arms and hostile arm.
"""

import csv
import json
import os
import sys

METRICS = ["novelty", "rigor", "evidence", "reproducibility", "clarity", "significance", "overall"]

def load_mapping(mapping_path):
    mapping = {}
    with open(mapping_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            mapping[row["id"]] = {
                "title": row["title"],
                "doi": row["doi"],
                "is_ours": row["is_ours"].lower() == "true",
                "n_chars": int(row["n_chars"])
            }
    return mapping

def load_scores(csv_path):
    scores = {}
    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pid = row["id"].strip()
            scores[pid] = {
                m: float(row[m]) for m in METRICS if m in row and row[m] != ""
            }
            if "verdict" in row:
                scores[pid]["verdict"] = row["verdict"].strip().lower()
            if "note" in row:
                scores[pid]["note"] = row["note"].strip()
            elif "fatal_flaw" in row:
                scores[pid]["note"] = row["fatal_flaw"].strip()
    return scores

def compute_percentiles(scores, ours_id, metric="overall"):
    ours_val = scores[ours_id][metric]
    all_vals = [scores[pid][metric] for pid in scores]
    n_total = len(all_vals)
    
    strict_beats = sum(1 for v in all_vals if v < ours_val)
    ties = sum(1 for v in all_vals if v == ours_val)
    strictly_better = sum(1 for v in all_vals if v > ours_val)
    
    # rank 1 is highest score
    rank = strictly_better + 1
    strict_pct = (strict_beats / (n_total - 1)) * 100 if n_total > 1 else 100.0
    midpoint_pct = ((strict_beats + 0.5 * (ties - 1)) / (n_total - 1)) * 100 if n_total > 1 else 100.0
    
    return {
        "ours_score": ours_val,
        "rank": rank,
        "out_of": n_total,
        "strict_beats": strict_beats,
        "ties": ties,
        "strictly_better": strictly_better,
        "strict_pct": strict_pct,
        "midpoint_pct": midpoint_pct,
        "corpus_min": min(all_vals),
        "corpus_median": sorted(all_vals)[len(all_vals)//2],
        "corpus_mean": sum(all_vals) / len(all_vals),
        "corpus_max": max(all_vals),
    }

def analyze_venue(venue_name, mapping_path, arms, out_md_path):
    mapping = load_mapping(mapping_path)
    ours_id = [pid for pid, meta in mapping.items() if meta["is_ours"]][0]
    
    print(f"=== {venue_name} (Ours ID: {ours_id}) ===")
    
    arm_results = {}
    for arm_name, csv_path, judge_name, arm_type in arms:
        if not os.path.exists(csv_path):
            print(f"Warning: {csv_path} not found.")
            continue
        scores = load_scores(csv_path)
        stats = compute_percentiles(scores, ours_id, "overall")
        arm_results[arm_name] = {
            "judge": judge_name,
            "type": arm_type,
            "scores": scores,
            "stats": stats,
            "verdict": scores.get(ours_id, {}).get("verdict", "n/a"),
            "note": scores.get(ours_id, {}).get("note", "n/a")
        }
        print(f"[{arm_name} - {judge_name}] Score: {stats['ours_score']} | Rank: {stats['rank']}/{stats['out_of']} (strict {stats['strict_pct']:.1f}%) | Corpus median: {stats['corpus_median']} | Verdict: {arm_results[arm_name]['verdict']}")

    return ours_id, arm_results

if __name__ == "__main__":
    print("Unblinding script compiled successfully.")
