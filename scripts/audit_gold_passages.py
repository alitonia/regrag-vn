"""Audit the fidelity of every gold passage in the QA CSV against the delivered corpus.

The benchmark's central invariant is that each answerable question carries a *verbatim*
gold passage which string-matches a chunk of the ingested corpus. That invariant is what
makes coverage measurable, what makes groundedness scorable, and what stops a question
whose source never arrived from being scored as a hallucination.

This script tests whether the delivered CSV actually satisfies it, per question, and
separates three things that are easy to conflate:

  * the corpus is missing           -> no document contains the passage
  * the passage is a loose summary  -> a document is present and on-topic, but the
                                       annotator wrote the passage in their own words,
                                       so no string match is possible in principle
  * the citation is wrong           -> the passage names an `Điều N` that the document it
                                       points at does not contain

Matching is length-aware. A fixed 8-word window is meaningless on a 12-word passage,
which is what made the first pass of this analysis report a false 29% coverage: every
short passage scored 0 regardless of whether it was an accurate quotation.

Usage:
  python3 scripts/audit_gold_passages.py --inbox /path/to/extracted
  python3 scripts/audit_gold_passages.py --inbox DIR --json out.json
"""

import argparse
import csv
import json
import os
import re
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from regrag.corpus.canonical import fold_diacritics, normalize_ws, split_urls

from match_passages_to_documents import PROBE_SENTINEL, extract  # noqa: E402

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CSV_PATH = os.path.join(REPO_ROOT, "data", "gold", "bank_qa_data.csv")

_DIEU_RE = re.compile(r"Điều\s*(\d+[a-z]?)", re.IGNORECASE)
MAX_PROBES = 40

# Classification thresholds on small-window containment.
QUOTABLE = 0.70
PARTIAL = 0.35


def window_for(n_words: int) -> int:
    """Smaller windows for shorter passages, so a 12-word quote is not all-or-nothing."""
    if n_words < 12:
        return 3
    if n_words < 25:
        return 4
    if n_words < 60:
        return 6
    return 8


def probes(text: str, n: int, k: int = MAX_PROBES) -> list:
    words = text.split()
    if len(words) <= n:
        return [" ".join(words)] if words else []
    all_w = [" ".join(words[i:i + n]) for i in range(len(words) - n + 1)]
    if len(all_w) <= k:
        return all_w
    step = len(all_w) / float(k)
    return [all_w[int(i * step)] for i in range(k)]


def containment(ps: list, haystack: str) -> float:
    if not ps:
        return 0.0
    return sum(1 for p in ps if p and p in haystack) / float(len(ps))


def articles_in(text: str) -> set:
    return {m.group(1) for m in _DIEU_RE.finditer(text)}


def audit(inbox: str, csv_path: str):
    files = sorted(
        f for f in os.listdir(inbox)
        if os.path.isfile(os.path.join(inbox, f)) and not f.startswith(".")
    )
    docs = []
    for name in files:
        text, pages, has_layer = extract(os.path.join(inbox, name))
        norm = normalize_ws(text)
        docs.append({
            "file": name, "pages": pages, "chars": len(text),
            "has_text_layer": has_layer, "norm": norm,
            "folded": fold_diacritics(norm), "articles": articles_in(norm),
        })
        sys.stderr.write(f"[extract] {name:<40} {'TEXT' if has_layer else 'SCANNED'}\n")

    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))

    results = []
    for i, row in enumerate(rows, 1):
        question = (row.get("question") or "").strip()
        if not question:
            continue
        raw_id = (row.get("ID") or "").strip()
        qid = f"Q{int(raw_id):03d}" if raw_id.isdigit() else (raw_id or f"Q{i:03d}")
        passage = normalize_ws((row.get("text_contains_answer_in_the_doc") or "").strip())
        answer = (row.get("answer") or "").strip()
        urls = split_urls(row.get("doc_link") or "")
        if not urls and not passage and fold_diacritics(answer).lower() == fold_diacritics(PROBE_SENTINEL).lower():
            results.append({"id": qid, "kind": "probe"})
            continue
        if not passage:
            results.append({"id": qid, "kind": "no-passage", "urls": urls})
            continue

        claimed = _DIEU_RE.search(passage)
        claimed_article = claimed.group(1) if claimed else None
        n_words = len(passage.split())
        n = window_for(n_words)
        ps = probes(passage, n)

        best = {"file": None, "ratio": 0.0, "tier": "none"}
        for d in docs:
            if not d["has_text_layer"]:
                continue
            if passage and passage in d["norm"]:
                best = {"file": d["file"], "ratio": 1.0, "tier": "strict"}
                break
            r = containment(ps, d["norm"])
            if r > best["ratio"]:
                best = {"file": d["file"], "ratio": r, "tier": "fragment"}

        if best["tier"] != "strict" and best["ratio"] < PARTIAL:
            # Only if the diacritic-preserving match fails do we test for fold damage.
            fps = [fold_diacritics(p) for p in ps]
            for d in docs:
                if not d["has_text_layer"]:
                    continue
                r = containment(fps, d["folded"])
                if r > best["ratio"]:
                    best = {"file": d["file"], "ratio": r, "tier": "folded-only"}

        ratio = best["ratio"]
        if best["tier"] == "strict":
            fidelity = "verbatim"
        elif ratio >= QUOTABLE:
            fidelity = "quotable"
        elif ratio >= PARTIAL:
            fidelity = "partial"
        elif best["tier"] == "folded-only":
            fidelity = "diacritic-damaged"
        else:
            fidelity = "paraphrase-or-missing"

        match_doc = next((d for d in docs if d["file"] == best["file"]), None)
        article_present = None
        if claimed_article and match_doc and match_doc["has_text_layer"]:
            article_present = claimed_article in match_doc["articles"]

        results.append({
            "id": qid, "kind": "answerable", "passage_chars": len(passage),
            "passage_words": n_words, "window": n, "urls": urls,
            "best_file": best["file"], "ratio": round(ratio, 3), "tier": best["tier"],
            "fidelity": fidelity, "claimed_article": claimed_article,
            "article_present_in_best_doc": article_present,
            "n_articles_in_best_doc": len(match_doc["articles"]) if match_doc else None,
            "best_doc_has_text": match_doc["has_text_layer"] if match_doc else None,
            "passage_head": passage[:110],
        })
    return docs, results


def report(docs, results):
    ans = [r for r in results if r["kind"] == "answerable"]
    probes = [r for r in results if r["kind"] == "probe"]
    nopass = [r for r in results if r["kind"] == "no-passage"]

    print(f"\n=== {len(results)} rows: {len(ans)} answerable, {len(probes)} probes, "
          f"{len(nopass)} answerable-without-passage ===")

    print("\n=== passage fidelity against the delivered corpus ===")
    order = ["verbatim", "quotable", "partial", "diacritic-damaged", "paraphrase-or-missing"]
    for k in order:
        sel = [r for r in ans if r["fidelity"] == k]
        print(f"  {k:<22} {len(sel):>3} / {len(ans)}   {[r['id'] for r in sel][:12]}"
              f"{' ...' if len(sel) > 12 else ''}")
    strong = [r for r in ans if r["fidelity"] in ("verbatim", "quotable")]
    print(f"\n  => passages the corpus can actually ground: {len(strong)}/{len(ans)} "
          f"= {len(strong) / len(ans):.1%}" if ans else "")
    print("     (strict-verbatim only: "
          f"{sum(1 for r in ans if r['fidelity'] == 'verbatim')})")

    print("\n=== citation check: does the named Điều exist in the matched document? ===")
    checked = [r for r in ans if r["article_present_in_best_doc"] is not None]
    bad = [r for r in checked if not r["article_present_in_best_doc"]]
    print(f"  checkable              : {len(checked)}")
    print(f"  article IS present     : {len(checked) - len(bad)}")
    print(f"  article NOT present    : {len(bad)}")
    for r in bad:
        print(f"    {r['id']:<6} claims Điều {r['claimed_article']:<4} "
              f"best_doc={r['best_file']} (has {r['n_articles_in_best_doc']} articles, "
              f"ratio={r['ratio']})  {r['passage_head'][:70]!r}")

    print("\n=== blocked by a SCANNED delivery (no text layer anywhere to match) ===")
    scanned = {d["file"] for d in docs if not d["has_text_layer"]}
    blocked = [r for r in ans if r["best_file"] is None or r["ratio"] < PARTIAL]
    for r in blocked:
        bases = {os.path.basename(u.rstrip("/").split("?")[0]) for u in r["urls"]}
        hit = bases & scanned
        if hit:
            print(f"  {r['id']:<6} cites scanned {sorted(hit)}  ratio={r['ratio']}")

    print("\n=== documents by yield ===")
    by_file = {}
    for r in ans:
        if r["best_file"]:
            by_file.setdefault(r["best_file"], []).append(r["id"])
    for d in docs:
        ids = by_file.get(d["file"], [])
        state = "TEXT" if d["has_text_layer"] else "SCANNED"
        print(f"  {d['file']:<40} {state:<8} articles={len(d['articles']):<4} "
              f"{len(ids):>3} q")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--inbox", required=True, help="Directory holding the delivered files")
    ap.add_argument("--csv", default=CSV_PATH, help="Path to the trusted QA CSV")
    ap.add_argument("--json", dest="json_out", help="Also write the full audit as JSON")
    args = ap.parse_args()

    docs, results = audit(args.inbox, args.csv)
    report(docs, results)

    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump({
                "documents": [{k: v for k, v in d.items()
                               if k not in ("norm", "folded", "articles")} for d in docs],
                "rows": results,
            }, f, ensure_ascii=False, indent=2)
        print(f"\nwrote audit -> {args.json_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
