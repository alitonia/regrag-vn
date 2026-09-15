#!/usr/bin/env python3
"""Rebuild the RIVF blind-rank pack for a paired re-eval (2026-09-15).

Keeps reports/blind_rank_rivf2026/mapping.csv as the source of truth for
id->DOI (identical positions to the 2026-09-13 run; only ours, R33, gets
its abstract refreshed from the current paper/main.tex). Fetches competitor
abstracts from OpenAlex by DOI, records OA availability for the tier-2 gate,
archives the raw corpus JSON repo-side, and writes the per-entry pack to
/tmp (agents cannot see the repo).

Usage: .venv/bin/python scripts/build_rivf_blind_reeval.py
"""
from __future__ import annotations

import csv
import json
import re
import time
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MAP = ROOT / "reports" / "blind_rank_rivf2026" / "mapping.csv"
CORPUS_OUT = ROOT / "reports" / "blind_rank_rivf2026" / "corpus_rivf2025_2026-09-15.json"
PACK_DIR = Path("/tmp/rivf_reeval_2026-09-15/pack")
OURS_TEX = ROOT / "paper" / "main.tex"


def openalex_batch(dois: list[str]) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for i in range(0, len(dois), 25):
        chunk = dois[i : i + 25]
        f = "doi:" + "|".join(d.lower() for d in chunk)
        url = "https://api.openalex.org/works?filter=" + urllib.parse.quote(f, safe=":|") + "&per-page=25&mailto=blindrank@example.org"
        req = urllib.request.Request(url, headers={"User-Agent": "blindrank/0.1"})
        with urllib.request.urlopen(req, timeout=60) as r:
            res = json.loads(r.read())
        for w in res.get("results", []):
            d = (w.get("doi") or "").replace("https://doi.org/", "").lower()
            if d in {x.lower() for x in chunk}:
                out[d] = w
        time.sleep(1)
    return out


def abstract_of(w: dict) -> str:
    inv = w.get("abstract_inverted_index")
    if not inv:
        return ""
    pos: dict[int, str] = {}
    for word, ps in inv.items():
        for p in ps:
            pos[p] = word
    return " ".join(pos[i] for i in sorted(pos))


def ours_abstract() -> str:
    tex = OURS_TEX.read_text(encoding="utf-8")
    m = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", tex, re.S)
    t = m.group(1)
    t = re.sub(r"\\emph\{([^}]*)\}", r"\1", t)
    t = re.sub(r"\$3\\times3\\times88\$", "3x3x88", t)
    t = t.replace("\\%", "%").replace("---", "-").replace("--", "-")
    t = t.replace("``", '"').replace("''", '"')
    t = re.sub(r"\s+", " ", t).strip()
    return t


def main() -> None:
    rows = list(csv.DictReader(open(MAP, encoding="utf-8")))
    comp = [r for r in rows if r["is_ours"] == "False"]
    dois = [r["doi"] for r in comp]
    works = openalex_batch(dois)

    corpus, missing = [], []
    for r in comp:
        w = works.get(r["doi"].lower())
        ab = abstract_of(w) if w else ""
        if not ab:
            missing.append(r["doi"])
            continue
        oa = bool((w.get("open_access") or {}).get("is_oa")) if w else False
        corpus.append({
            "id": r["id"], "doi": r["doi"], "title": w["title"] if w else r["title"],
            "abstract": ab, "is_oa": oa, "n_chars": len(r["title"]) + 1 + len(ab),
        })
    CORPUS_OUT.write_text(json.dumps(corpus, indent=1, ensure_ascii=False), encoding="utf-8")

    ours_ab = ours_abstract()
    ours_entry = {"id": "R33", "title": rows and next(r["title"] for r in rows if r["is_ours"] == "True"),
                  "abstract": ours_ab, "n_chars": 0}
    ours_entry["n_chars"] = len(ours_entry["title"]) + 1 + len(ours_ab)

    PACK_DIR.mkdir(parents=True, exist_ok=True)
    for e in corpus + [ours_entry]:
        (PACK_DIR / f"{e['id']}.txt").write_text(
            f"Title: {e['title']}\n\nAbstract: {e['abstract']}\n", encoding="utf-8")

    chars = sorted(e["n_chars"] for e in corpus + [ours_entry])
    n_oa = sum(1 for e in corpus if e["is_oa"])
    stats = {
        "rebuilt": "2026-09-15", "seed": "mapping held fixed from 20260913 run",
        "entries_written": len(corpus) + 1, "missing_abstracts": missing,
        "oa_competitors": n_oa, "corpus_median_chars": chars[len(chars)//2],
        "ours_chars": ours_entry["n_chars"], "min_chars": chars[0], "max_chars": chars[-1],
    }
    (Path("/tmp/rivf_reeval_2026-09-15") / "pack_stats.json").write_text(json.dumps(stats, indent=1))
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
