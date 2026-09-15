"""Identify delivered legal documents and map them onto the instruments the QA set cites.

A delivery arrives as a directory of files whose names are the *source URL basenames*
(`91.signed.pdf`, `6pl.pdf`, `62-vbhn-vpqh.pdf`), not the instrument ids. The benchmark
cannot resolve `gold.doc_id` from an opaque filename, and the ingestion manifest needs a
real `expected_file` per instrument. This script closes that gap mechanically instead of
by guesswork:

  * extracts text from each PDF (.doc is converted first via headless LibreOffice)
  * reports page count and whether a text layer exists at all, so a *scanned* delivery is
    named as unusable rather than silently producing an empty extraction
  * reads the instrument number off the document's own header (`Số: 17/2024/TT-NHNN`)
  * cross-references two independent sources of truth: the `source_url` basenames in
    INGEST_PLAN.json and every URL appearing in the QA CSV's `doc_link` column

It reports and changes nothing. Acting on the report means editing
data/raw_legal/DOC_MANIFEST.json and INGEST_PLAN.json.

Usage:
  python3 scripts/identify_documents.py --inbox /path/to/extracted
  python3 scripts/identify_documents.py --inbox DIR --json out.json
"""

import argparse
import csv
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from urllib.parse import unquote, urlparse

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RAW_DIR = os.path.join(REPO_ROOT, "data", "raw_legal")
PLAN_PATH = os.path.join(RAW_DIR, "INGEST_PLAN.json")
CSV_PATH = os.path.join(REPO_ROOT, "data", "gold", "bank_qa_data.csv")

# "Số: 17/2024/TT-NHNN", "số 32/2024/QH15", "22/2023/TT-NHNN"
_INSTRUMENT_RE = re.compile(
    r"\b(\d{1,3})\s*/\s*(\d{4})\s*/\s*"
    r"((?:[A-ZĐ]{1,12})(?:-[A-ZĐ]{1,12})*)",
)
# The instrument type words, used to confirm a candidate is a real header and not a
# date or a page range that happens to contain two slashes.
_TYPE_WORDS = re.compile(
    r"\b(THÔNG TƯ|NGHỊ ĐỊNH|LUẬT|PHÁP LỆNH|QUYẾT ĐỊNH|CÔNG VĂN|BỘ LUẬT|"
    r"THÔNG TƯ LIÊN TỊCH|VBHN|VĂN BẢN HỢP NHẤT)\b",
    re.IGNORECASE,
)
_URL_RE = re.compile(r"https?://[^\s,;'\"<>)\]]+")


def url_basename(url: str) -> str:
    """`.../vbpq/2016/01/91.signed.pdf` -> `91.signed.pdf`. Query-only URLs give ''."""
    path = unquote(urlparse(url).path).rstrip("/")
    return os.path.basename(path) if path else ""


# --- extraction --------------------------------------------------------------

def pdf_text(path: str, max_pages: int = 4):
    """Return (text_of_first_pages, page_count, has_text_layer)."""
    try:
        from pypdf import PdfReader
    except ImportError:
        raise SystemExit(
            "pypdf is not installed; cannot identify PDFs. "
            "Install with: .venv/bin/pip install pypdf pdfplumber"
        )
    reader = PdfReader(path)
    n = len(reader.pages)
    head = "\n".join(
        (reader.pages[i].extract_text() or "") for i in range(min(max_pages, n))
    )
    return head, n, bool(head.strip())


def doc_text(path: str, max_pages: int = 4):
    """Convert a legacy .doc via headless LibreOffice, then read the text."""
    if not shutil.which("libreoffice"):
        sys.stderr.write(
            f"[warn] libreoffice not available; cannot read {os.path.basename(path)}\n"
        )
        return "", 0, False
    outdir = tempfile.mkdtemp(prefix="doc2txt-")
    try:
        subprocess.run(
            ["libreoffice", "-env:UserInstallation=file://" + outdir + "/profile",
             "--headless", "--convert-to", "txt:Text", "--outdir", outdir, path],
            check=False, capture_output=True, timeout=300,
        )
        produced = [f for f in os.listdir(outdir) if f.endswith(".txt")]
        if not produced:
            return "", 0, False
        with open(os.path.join(outdir, produced[0]), encoding="utf-8", errors="replace") as f:
            text = f.read()
        return text[: max_pages * 3000], 0, bool(text.strip())
    finally:
        shutil.rmtree(outdir, ignore_errors=True)


def identify(text: str):
    """Pull the most likely instrument id out of a document's opening text."""
    if not text.strip():
        return None, []
    window = text[:6000]
    candidates = []
    for num, year, tail in _INSTRUMENT_RE.findall(window):
        tail = tail.strip("-.")
        # A genuine instrument suffix is short and hyphenated (TT-NHNN, NĐ-CP, QH15).
        if not re.fullmatch(r"[A-ZĐ]{1,12}(?:-[A-ZĐ0-9]{1,12}){0,2}", tail):
            continue
        candidates.append(f"{int(num)}/{year}/{tail}")
    if not candidates:
        return None, []
    # The header appears first and repeats; rank by frequency then first position.
    ranked = sorted(
        set(candidates),
        key=lambda c: (-candidates.count(c), candidates.index(c)),
    )
    has_type_word = bool(_TYPE_WORDS.search(window))
    return (ranked[0] if has_type_word else None), ranked


# --- cross-references --------------------------------------------------------

def plan_basenames(plan_path: str):
    """basename(source_url) -> doc_id, from the ingestion manifest."""
    if not os.path.exists(plan_path):
        return {}
    with open(plan_path, encoding="utf-8") as f:
        plan = json.load(f)
    out = {}
    for entry in plan.get("documents", []):
        doc_id = entry.get("doc_id")
        for key in ("source_url", "alt_source_url", "replacement_url", "also_at"):
            url = entry.get(key)
            if not url:
                continue
            base = url_basename(url)
            if base:
                out.setdefault(base, doc_id)
    for url in (plan.get("unresolved_urls") or {}):
        if url.startswith("_"):
            continue
        base = url_basename(url)
        if base:
            out.setdefault(base, "UNRESOLVED")
    return out


def csv_urls(csv_path: str):
    """Every URL cited by the QA set, basename -> question count."""
    if not os.path.exists(csv_path):
        return {}, 0
    counts, rows = {}, 0
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            rows += 1
            for url in _URL_RE.findall(row.get("doc_link") or ""):
                base = url_basename(url)
                if base:
                    counts[base] = counts.get(base, 0) + 1
                else:
                    counts[url] = counts.get(url, 0) + 1
    return counts, rows


# --- main --------------------------------------------------------------------

def scan(inbox: str, plan_path: str, csv_path: str):
    if not os.path.isdir(inbox):
        raise SystemExit(f"inbox is not a directory: {inbox}")

    plan_map = plan_basenames(plan_path)
    cited, n_rows = csv_urls(csv_path)

    files = sorted(
        f for f in os.listdir(inbox)
        if os.path.isfile(os.path.join(inbox, f)) and not f.startswith(".")
    )
    results = []
    for name in files:
        path = os.path.join(inbox, name)
        ext = os.path.splitext(name)[1].lower()
        rec = {
            "file": name,
            "bytes": os.path.getsize(path),
            "ext": ext,
            "plan_doc_id": plan_map.get(name),
            "cited_by_questions": cited.get(name, 0),
        }
        try:
            if ext == ".pdf":
                text, pages, has_layer = pdf_text(path)
            elif ext == ".doc":
                text, pages, has_layer = doc_text(path)
            elif ext in (".txt", ".html", ".htm"):
                with open(path, encoding="utf-8", errors="replace") as f:
                    text = f.read()
                pages, has_layer = 0, bool(text.strip())
            else:
                text, pages, has_layer = "", 0, False
                rec["note"] = f"no extractor for {ext!r}"
        except Exception as exc:
            text, pages, has_layer = "", 0, False
            rec["note"] = f"extraction failed: {exc}"

        ident, alts = identify(text)
        rec.update({
            "pages": pages,
            "head_chars": len(text.strip()),
            "text_layer": has_layer,
            "identified_as": ident,
            "id_candidates": alts[:6],
            "n_dieu_in_head": len(re.findall(r"Điều\s*\d+", text)),
        })
        if has_layer and ident is None:
            rec["note"] = (rec.get("note", "") + " text present but no instrument id matched").strip()
        if not has_layer:
            rec["note"] = (rec.get("note", "") + " SCANNED - no text layer, needs OCR").strip()
        results.append(rec)

    return results, cited, n_rows


def report(results, cited, n_rows):
    print(f"=== delivered files ({len(results)}) vs QA CSV ({n_rows} rows) ===\n")
    hdr = f"{'file':<38}{'MB':>6}{'pgs':>5}{'chars':>7}  {'identified_as':<22}{'plan_doc_id':<22}{'cites':>5}"
    print(hdr)
    print("-" * len(hdr))
    for r in results:
        mb = r["bytes"] / 1e6
        ident = r["identified_as"] or ("SCANNED" if not r["text_layer"] else "?")
        print(
            f"{r['file'][:37]:<38}{mb:>6.1f}{r['pages']:>5}{r['head_chars']:>7}  "
            f"{ident[:21]:<22}{(r['plan_doc_id'] or '-')[:21]:<22}{r['cited_by_questions']:>5}"
        )

    print("\n=== cited URLs with NO delivered file (basename match) ===")
    have = {r["file"] for r in results}
    missing = sorted(
        (base, n) for base, n in cited.items() if base not in have
    )
    if not missing:
        print("  none - every cited URL basename was delivered")
    for base, n in missing:
        print(f"  {n:>2} question(s)  {base[:110]}")

    print("\n=== delivered files cited by NO question (candidate distractors) ===")
    for r in results:
        if r["cited_by_questions"] == 0:
            print(f"  {r['file']:<38} identified_as={r['identified_as'] or '?'}")

    scanned = [r["file"] for r in results if not r["text_layer"]]
    if scanned:
        print("\n=== SCANNED (no text layer) - cannot ingest without OCR ===")
        for f in scanned:
            print(f"  {f}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--inbox", required=True, help="Directory holding the delivered files")
    ap.add_argument("--plan", default=PLAN_PATH, help="Path to INGEST_PLAN.json")
    ap.add_argument("--csv", default=CSV_PATH, help="Path to the trusted QA CSV")
    ap.add_argument("--json", dest="json_out", help="Also write the report as JSON here")
    args = ap.parse_args()

    results, cited, n_rows = scan(args.inbox, args.plan, args.csv)
    report(results, cited, n_rows)

    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump({"files": results, "cited_urls": cited, "csv_rows": n_rows},
                      f, ensure_ascii=False, indent=2)
        print(f"\nwrote report -> {args.json_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
