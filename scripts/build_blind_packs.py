#!/usr/bin/env python3
"""
Builds blind ranking packs for RIVF 2026 and EASE 2027.
Draws 48 accepted papers from each venue corpus, adds RegRAG-VN,
and shuffles with a fixed seed (seed=20260913).

Outputs:
  /tmp/blindrank_rivf2026_pack.json
  /tmp/blindrank_ease2027_pack.json
  reports/blind_rank_rivf2026/mapping.csv
  reports/blind_rank_ease2027/mapping.csv
  reports/blind_rank_rivf2026/pack_stats.json
  reports/blind_rank_ease2027/pack_stats.json
"""

import csv
import json
import os
import random
import re

SEED = 20260913
N_COMPETITORS = 48

OURS_TITLE = "Citation-Level Unfaithfulness of Small Language Models on Vietnamese Banking Regulation: A Provenance-Guarded RAG Benchmark"
OURS_ABSTRACT = (
    "Retrieval-augmented generation (RAG) is the standard remedy for hallucination in small "
    "language models, but its benefit is usually reported as an aggregate accuracy delta that "
    "hides where an answer came from. We present RegRAG-VN, a benchmark and evaluation framework "
    "for Vietnamese banking and payment regulation in which the retrieval corpus is aligned with "
    "the benchmark questions by requiring each question's supporting gold passage to match an "
    "ingested legal chunk exactly. The benchmark holds 64 answerable questions drawn from State "
    "Bank of Vietnam circulars, decrees, and laws, alongside 24 unanswerable probes paired with an "
    "explicit abstention sentinel. We evaluate models of at most 7B parameters under closed-book, "
    "BM25, and dense retrieval, scoring article-level (Điều) and clause-level (Khoản) citation "
    "accuracy alongside abstention. Across a 3x3x88 generation campaign on a single GPU, retrieval "
    "augmentation raises article-level citation F1 from 0.005 (closed-book) to 0.32--0.43. However, "
    "43--49% of RAG-generated answers still cite an incorrect instrument or non-governing article, "
    "and models frequently emit the refusal sentinel before answering anyway. Finally, we describe "
    "a strict provenance discipline tracking corpus tiers and retriever backends, preventing "
    "degraded or fixture data from silently corrupting benchmark results."
)

def clean_text(text):
    if not text:
        return ""
    text = re.sub(r'\s+', ' ', str(text)).strip()
    return text

def build_pack(venue_name, raw_entries, id_prefix, pack_json_path, mapping_csv_path, stats_json_path):
    random.seed(SEED)
    
    # Filter valid entries
    candidates = []
    for e in raw_entries:
        title = clean_text(e.get("title", ""))
        abstract = clean_text(e.get("abstract", ""))
        doi = e.get("doi", "") or e.get("url", "") or e.get("oa_id", "")
        if title and abstract and len(abstract) > 100:
            candidates.append({
                "doi": doi,
                "title": title,
                "abstract": abstract,
                "is_ours": False
            })
            
    # Sample competitors
    selected = random.sample(candidates, N_COMPETITORS)
    
    # Add ours
    all_entries = selected + [{
        "doi": "LOCAL:paper/main.tex",
        "title": OURS_TITLE,
        "abstract": OURS_ABSTRACT,
        "is_ours": True
    }]
    
    # Shuffle
    random.shuffle(all_entries)
    
    # Assign IDs
    pack = []
    mapping = []
    char_lens = []
    
    for idx, item in enumerate(all_entries, start=1):
        pid = f"{id_prefix}{idx:02d}"
        pack.append({
            "id": pid,
            "title": item["title"],
            "abstract": item["abstract"],
            "n_chars": len(item["abstract"])
        })
        mapping.append({
            "id": pid,
            "title": item["title"],
            "doi": item["doi"],
            "is_ours": item["is_ours"],
            "n_chars": len(item["abstract"])
        })
        char_lens.append(len(item["abstract"]))
        if item["is_ours"]:
            ours_id = pid
            
    # Stats
    stats = {
        "venue": venue_name,
        "seed": SEED,
        "total_pack_entries": len(pack),
        "ours_id": ours_id,
        "ours_chars": len(OURS_ABSTRACT),
        "mean_chars": sum(char_lens) / len(char_lens),
        "min_chars": min(char_lens),
        "max_chars": max(char_lens),
        "all_chars": char_lens
    }
    
    # Write outputs
    os.makedirs(os.path.dirname(mapping_csv_path), exist_ok=True)
    
    with open(pack_json_path, "w", encoding="utf-8") as f:
        json.dump(pack, f, indent=2, ensure_ascii=False)
        
    with open(mapping_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "is_ours", "doi", "n_chars", "title"])
        writer.writeheader()
        for row in mapping:
            writer.writerow(row)
            
    with open(stats_json_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
        
    print(f"[{venue_name}] Built pack with {len(pack)} entries. Ours ID: {ours_id} ({len(OURS_ABSTRACT)} chars). Mean competitor chars: {stats['mean_chars']:.1f}")
    return ours_id, stats

def main():
    # RIVF Pack
    with open("/mnt/data/vifr_investigation/reports/tier_ladder_blind/raw/rivf2025.json", encoding="utf-8") as f:
        rivf_records = json.load(f)["records"]
    build_pack(
        venue_name="RIVF 2026",
        raw_entries=rivf_records,
        id_prefix="R",
        pack_json_path="/tmp/blindrank_rivf2026_pack.json",
        mapping_csv_path="/mnt/data/seminar_2/reports/blind_rank_rivf2026/mapping.csv",
        stats_json_path="/mnt/data/seminar_2/reports/blind_rank_rivf2026/pack_stats.json"
    )
    
    # EASE Pack
    with open("/mnt/data/ai4se/reports/ease2027_blind/ease_corpus.json", encoding="utf-8") as f:
        ease_records = json.load(f)["entries"]
    build_pack(
        venue_name="EASE 2027",
        raw_entries=ease_records,
        id_prefix="E",
        pack_json_path="/tmp/blindrank_ease2027_pack.json",
        mapping_csv_path="/mnt/data/seminar_2/reports/blind_rank_ease2027/mapping.csv",
        stats_json_path="/mnt/data/seminar_2/reports/blind_rank_ease2027/pack_stats.json"
    )

if __name__ == "__main__":
    main()
