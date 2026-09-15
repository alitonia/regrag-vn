"""Match every gold passage in the QA CSV against the delivered documents, by content.

Filename matching cannot resolve this delivery. The revised CSV cites URLs whose
basenames do not match any delivered file (`22-nhnn.pdf`, `06-nhnn.pdf`,
`27480-1-2019611-61206-2019-tt-nhnn.pdf`) and one that has no basename at all
(`vanban.chinhphu.vn/?pageid=27160&docid=211190`, 12 questions), while the delivered
files are named after *different* URLs (`91.signed.pdf`, `6pl.pdf`, `tt-32.pdf`).

The only reliable mapping is content: each answerable row carries a verbatim gold
passage, so a document either contains it or it does not. This answers three
questions at once, before any ingestion happens:

  1. Which delivered file is which instrument (identity by content, not by filename).
  2. Which questions are satisfiable, i.e. the Gate B coverage forecast.
  3. Which questions are blocked, and why - no file, or a file with no text layer.

Matching is tiered so that extraction damage is *named* rather than absorbed:
  strict    normalised substring match (NFC + whitespace collapse, diacritics kept)
  fragment  a sampled word-window containment ratio, for PDFs whose line breaking
            or ligature handling breaks a contiguous match
  folded    fragment match only after diacritic folding -> the text lost diacritics,
            which the project treats as a distinct diagnostic, never a silent pass

Usage:
  python3 scripts/match_passages_to_documents.py --inbox /path/to/extracted
  python3 scripts/match_passages_to_documents.py --inbox DIR --json out.json
"""

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from regrag.corpus.canonical import fold_diacritics, normalize_ws, split_urls

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CSV_PATH = os.path.join(REPO_ROOT, "data", "gold", "bank_qa_data.csv")

# A row whose answer is this sentinel and which carries no source URL is an
# unanswerable probe, not a question with a missing document.
PROBE_SENTINEL = "không có trong kho văn bản"

WINDOW = 8          # words per probe fragment
MAX_PROBES = 24     # sampled fragments per passage, evenly spaced
FRAGMENT_PASS = 0.60  # containment ratio that counts as "the passage is here"


# --- extraction --------------------------------------------------------------

def extract(path: str) -> tuple:
    """Return (text, pages, has_text_layer) for one delivered file."""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        return _extract_pdf(path)
    if ext == ".doc":
        return _extract_doc(path)
    if ext in (".txt", ".html", ".htm"):
        with open(path, encoding="utf-8", errors="replace") as f:
            text = f.read()
        return text, 0, bool(text.strip())
    return "", 0, False


def _extract_pdf(path: str) -> tuple:
    from pypdf import PdfReader
    reader = PdfReader(path)
    pages = len(reader.pages)
    parts = []
    for page in reader.pages:
        try:
            parts.append(page.extract_text() or "")
        except Exception:
            parts.append("")
    text = "\n".join(parts)
    if text.strip():
        return text, pages, True
    # pypdf found nothing on any page. pdfplumber sometimes recovers a partial text
    # layer, but it is roughly 100x slower per page, so sample a few pages spread
    # across the document to confirm the absence rather than crawling all of it.
    try:
        import pdfplumber
        with pdfplumber.open(path) as pdf:
            idx = sorted(set([0, len(pdf.pages) // 2, max(0, len(pdf.pages) - 1)]))
            text = "\n".join((pdf.pages[i].extract_text() or "") for i in idx)
    except Exception:
        text = ""
    return text, pages, bool(text.strip())


def _extract_doc(path: str) -> tuple:
    if not shutil.which("libreoffice"):
        sys.stderr.write(f"[warn] libreoffice unavailable, cannot read {path}\n")
        return "", 0, False
    outdir = tempfile.mkdtemp(prefix="doc2txt-")
    try:
        subprocess.run(
            ["libreoffice", "-env:UserInstallation=file://" + outdir + "/profile",
             "--headless", "--convert-to", "txt:Text", "--outdir", outdir, path],
            check=False, capture_output=True, timeout=300,
        )
        made = [f for f in os.listdir(outdir) if f.endswith(".txt")]
        if not made:
            return "", 0, False
        with open(os.path.join(outdir, made[0]), encoding="utf-8", errors="replace") as f:
            return f.read(), 0, True
    finally:
        shutil.rmtree(outdir, ignore_errors=True)


# --- matching ----------------------------------------------------------------

def windows(text: str, n: int = WINDOW) -> list:
    words = text.split()
    if len(words) <= n:
        return [" ".join(words)] if words else []
    return [" ".join(words[i:i + n]) for i in range(len(words) - n + 1)]


def sample(seq: list, k: int = MAX_PROBES) -> list:
    if len(seq) <= k:
        return seq
    step = len(seq) / float(k)
    return [seq[int(i * step)] for i in range(k)]


def containment(probes: list, haystack: str) -> float:
    if not probes:
        return 0.0
    hit = sum(1 for p in probes if p and p in haystack)
    return hit / float(len(probes))


def match_passage(passage: str, docs: list) -> dict:
    """Best document for one passage, with the tier at which it matched."""
    norm = normalize_ws(passage)
    probes = sample(windows(norm))
    folded_probes = sample([fold_diacritics(w) for w in windows(norm)])

    best = {"file": None, "tier": "none", "ratio": 0.0}
    for d in docs:
        if not d["has_text_layer"]:
            continue
        if norm and norm in d["norm"]:
            return {"file": d["file"], "tier": "strict", "ratio": 1.0}
        ratio = containment(probes, d["norm"])
        if ratio > best["ratio"]:
            best = {"file": d["file"], "tier": "fragment", "ratio": ratio}

    if best["ratio"] >= FRAGMENT_PASS:
        return best

    # Only now consider diacritic folding, and label it as damage.
    for d in docs:
        if not d["has_text_layer"]:
            continue
        ratio = containment(folded_probes, d["folded"])
        if ratio >= FRAGMENT_PASS and ratio > best["ratio"]:
            return {"file": d["file"], "tier": "folded-only", "ratio": ratio}
    return best


# --- main --------------------------------------------------------------------

def load_rows(csv_path: str) -> list:
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        out = []
        for i, row in enumerate(reader, 1):
            q = (row.get("question") or "").strip()
            if not q:
                continue
            raw_id = (row.get("ID") or "").strip()
            qid = f"Q{int(raw_id):03d}" if raw_id.isdigit() else (raw_id or f"Q{i:03d}")
            passage = (row.get("text_contains_answer_in_the_doc") or "").strip()
            answer = (row.get("answer") or "").strip()
            urls = split_urls(row.get("doc_link") or "")
            probe = (
                not urls
                and not passage
                and fold_diacritics(answer).lower() == fold_diacritics(PROBE_SENTINEL).lower()
            )
            out.append({
                "id": qid, "question": q, "passage": passage,
                "urls": urls, "is_probe": probe,
                "unanswerable_no_passage": bool(not passage and not probe),
            })
        return out


def run(inbox: str, csv_path: str):
    files = sorted(
        f for f in os.listdir(inbox)
        if os.path.isfile(os.path.join(inbox, f)) and not f.startswith(".")
    )
    docs = []
    for name in files:
        path = os.path.join(inbox, name)
        text, pages, has_layer = extract(path)
        norm = normalize_ws(text)
        docs.append({
            "file": name, "pages": pages, "chars": len(text),
            "has_text_layer": has_layer, "norm": norm,
            "folded": fold_diacritics(norm),
        })
        sys.stderr.write(
            f"[extract] {name:<40} pages={pages:<4} chars={len(text):<7} "
            f"{'TEXT' if has_layer else 'SCANNED'}\n"
        )

    rows = load_rows(csv_path)
    answerable = [r for r in rows if r["passage"]]
    probes = [r for r in rows if r["is_probe"]]

    results = []
    for r in answerable:
        m = match_passage(r["passage"], docs)
        results.append({**r, "match": m, "passage_chars": len(r["passage"])})

    return docs, rows, answerable, probes, results


def report(docs, rows, answerable, probes, results):
    print(f"\n=== delivered documents ({len(docs)}) ===")
    for d in docs:
        state = "TEXT" if d["has_text_layer"] else "SCANNED (no text layer)"
        print(f"  {d['file']:<40} pages={d['pages']:<4} chars={d['chars']:<7} {state}")

    print(f"\n=== QA CSV: {len(rows)} rows = {len(answerable)} answerable + "
          f"{len(probes)} unanswerable probes"
          f"{' + ' + str(sum(1 for r in rows if r['unanswerable_no_passage'])) + ' ambiguous' if any(r['unanswerable_no_passage'] for r in rows) else ''} ===")

    strict = [r for r in results if r["match"]["tier"] == "strict"]
    frag = [r for r in results if r["match"]["tier"] == "fragment"]
    folded = [r for r in results if r["match"]["tier"] == "folded-only"]
    none = [r for r in results if r["match"]["tier"] == "none"]

    print(f"\n=== GATE B FORECAST (passage found in a delivered document) ===")
    print(f"  strict   (publishable)      : {len(strict):>3} / {len(results)}")
    print(f"  fragment (>= {FRAGMENT_PASS:.0%} containment): {len(frag):>3} / {len(results)}")
    print(f"  folded-only (DIACRITIC DAMAGE, not publishable): {len(folded):>3}")
    print(f"  no match                    : {len(none):>3}")
    usable = len(strict) + len(frag)
    print(f"  => usable coverage {usable}/{len(results)} = "
          f"{usable / len(results):.1%} of answerable questions" if results else "  => no answerable rows")

    print("\n=== per-document yield (which file backs which questions) ===")
    by_file = {}
    for r in results:
        f = r["match"]["file"]
        if f:
            by_file.setdefault(f, []).append((r["id"], r["match"]["tier"]))
    for d in docs:
        hits = by_file.get(d["file"], [])
        label = ", ".join(f"{i}({t[:4]})" for i, t in hits[:14])
        if len(hits) > 14:
            label += f", +{len(hits) - 14} more"
        print(f"  {d['file']:<40} {len(hits):>3} q  {label}")

    print("\n=== UNMATCHED answerable questions (blocked) ===")
    for r in none + folded:
        urls = " | ".join(os.path.basename(u.rstrip('/')) if '/' in u else u for u in r["urls"])
        print(f"  {r['id']}  tier={r['match']['tier']:<11} passage={r['passage_chars']:>5}ch  urls={urls[:100]}")
    if not (none or folded):
        print("  none - every answerable question matched a delivered document")

    scanned_cited = set()
    for r in results:
        for u in r["urls"]:
            base = os.path.basename(u.rstrip("/").split("?")[0])
            for d in docs:
                if d["file"] == base and not d["has_text_layer"]:
                    scanned_cited.add((r["id"], base))
    if scanned_cited:
        print("\n=== questions citing a SCANNED file (needs OCR or an HTML source) ===")
        for qid, base in sorted(scanned_cited):
            print(f"  {qid}  {base}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--inbox", required=True, help="Directory holding the delivered files")
    ap.add_argument("--csv", default=CSV_PATH, help="Path to the trusted QA CSV")
    ap.add_argument("--json", dest="json_out", help="Also write the full report as JSON")
    args = ap.parse_args()

    if not os.path.isdir(args.inbox):
        raise SystemExit(f"inbox is not a directory: {args.inbox}")

    docs, rows, answerable, probes, results = run(args.inbox, args.csv)
    report(docs, rows, answerable, probes, results)

    if args.json_out:
        payload = {
            "documents": [{k: v for k, v in d.items() if k not in ("norm", "folded")} for d in docs],
            "n_rows": len(rows), "n_answerable": len(answerable), "n_probes": len(probes),
            "matches": [
                {"id": r["id"], "tier": r["match"]["tier"], "file": r["match"]["file"],
                 "ratio": round(r["match"]["ratio"], 3), "urls": r["urls"],
                 "passage_chars": r["passage_chars"]}
                for r in results
            ],
        }
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"\nwrote report -> {args.json_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
