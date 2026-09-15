"""Ingest raw legal texts listed in data/raw_legal/INGEST_PLAN.json into
clause-level chunks at data/processed_chunks/corpus_chunks.json.

Rewritten 2026-09-09. The previous version hardcoded a one-entry DOCS_MAP and
did `if not os.path.exists(filepath): continue`, so a partial or misnamed
delivery produced an empty corpus while printing "Saved 0 total chunks" and
exiting 0. It also read only .txt, while 12 of the benchmark's instruments
arrive as ~7 PDFs and ~8 HTML pages.

This version is manifest-driven and fails loudly:
  * a missing file for a non-skipped entry is an error, not a skip
  * extraction is dispatched on format (txt / pdf / html)
  * a missing extraction dependency is an error naming the package to install
  * extracted text with no "Điều" anywhere is flagged as probably-garbage
  * every chunk is stamped corpus_source=CORPUS_TIER2 (see regrag/provenance.py)
  * the parser's report is surfaced rather than discarded

Usage:
  python3 scripts/build_corpus_chunks.py                # strict
  python3 scripts/build_corpus_chunks.py --allow-missing  # ingest what arrived
"""

import argparse
import json
import os
import re
import sys
from dataclasses import asdict
from typing import Any, Dict, List, Optional, Sequence, Tuple

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from regrag.corpus.parser import LegalDocumentParser
from regrag.models import LegalChunk
from regrag.provenance import CORPUS_TIER1, CORPUS_TIER2, ProvenanceError

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RAW_DIR = os.path.join(REPO_ROOT, "data", "raw_legal")
PLAN_PATH = os.path.join(RAW_DIR, "INGEST_PLAN.json")
OUTPUT_PATH = os.path.join(REPO_ROOT, "data", "processed_chunks", "corpus_chunks.json")

_ARTICLE_RE = re.compile(r"Điều\s*\d+", re.IGNORECASE)

_OCR_DIR = os.path.join(RAW_DIR, "ocr")
_VIET_LETTERS = set("ăâêôơưđĂÂÊÔƠƯĐáàảãạấầẩẫậắằẳẵặéèẻẽẹếềểễệ"
                    "íìỉĩịóòỏõọốồổỗộớờởỡợúùủũụứừửữựýỳỷỹỵ"
                    "ÁÀẢÃẠẤẦẨẪẬẮẰẲẴẶÉÈẺẼẸẾỀỂỄỆÍÌỈĨỊ"
                    "ÓÒỎÕỌỐỒỔỖỘỚỜỞỠỢÚÙỦŨỤỨỪỬỮỰÝỲỶỸ")
_ASCII_LETTERS = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ")
_MIN_VIET_RATIO = 0.05
_OCR_HEADER_RE = re.compile(r"^<!--.*?-->\s*", re.DOTALL)


def viet_ratio(text: str) -> float:
    """Share of letters carrying a Vietnamese-specific diacritic.

    Genuine Vietnamese legal text measures 0.26-0.31; a corrupt or diacritic-stripped
    layer measures 0.00. This is what separates a usable embedded PDF text layer from
    the garbage the government scans ship with.
    """
    letters = [c for c in text if c in _VIET_LETTERS or c in _ASCII_LETTERS]
    if not letters:
        return 0.0
    return sum(1 for c in letters if c in _VIET_LETTERS) / float(len(letters))


def _ocr_fallback(path: str) -> str:
    """Return the tesseract text for this PDF, or '' if none was produced."""
    stem = os.path.splitext(os.path.basename(path))[0]
    candidate = os.path.join(_OCR_DIR, stem + ".ocr.txt")
    if not os.path.exists(candidate):
        return ""
    with open(candidate, "r", encoding="utf-8", errors="replace") as f:
        return _OCR_HEADER_RE.sub("", f.read(), count=1)


# --- text extraction ---------------------------------------------------------

def extract_txt(path: str) -> tuple:
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        return f.read(), "plain-text"


def _looks_reversed(text: str) -> bool:
    """True when a page extracted as reversed single characters.

    Landscape/rotated annex tables (e.g. Phụ lục 1 of TT 41/2016, pages 31-44) extract
    as one character per line in reverse order. Crucially their viet_ratio still looks
    healthy (~0.29), because reversing characters preserves diacritics - so the quality
    gate alone cannot catch them and a pipeline would silently index garbage. Median
    line length separates them cleanly: ~2-4 chars reversed vs ~40-80 normal.
    """
    lines = [l for l in text.splitlines() if l.strip()]
    if len(lines) < 8 or len(text) < 400:
        return False
    median = sorted(len(l) for l in lines)[len(lines) // 2]
    return median <= 6


def _page_text(page) -> tuple:
    """Extract one page, repairing a 90-degree rotated layout if present."""
    text = page.extract_text() or ""
    if _looks_reversed(text):
        try:
            page.rotate(90)
            fixed = page.extract_text() or ""
            page.rotate(-90)
            if fixed.strip() and not _looks_reversed(fixed):
                return fixed, True
        except Exception:
            pass
    return text, False


def extract_pdf(path: str) -> tuple:
    """Extract text from a legal PDF, trusting only a usable embedded layer.

    Returns (text, extraction_tag). pdfplumber is the PREFERRED extractor: on
    several born-digital CÔNG BÁO PDFs of this delivery (06/2019, 21/2017,
    32/2024/QH15, 41/2016) the pypdf layer inserts spurious intra-word spaces
    ("T ỷ l ệ an toàn v ốn") - measured 2026-09-11 as a single-char-token
    ratio of 0.11-0.13 under pypdf versus 0.02-0.06 under pdfplumber - which
    silently poisons BM25 tokenisation and every downstream exact match.
    pypdf remains the fallback. An embedded layer is trusted only when it
    contains articles AND measures a real Vietnamese diacritic ratio. The
    government scans in this delivery ship corrupt layers that pass a naive
    "has text?" check - tens of thousands of characters, zero 'Điều',
    viet_ratio 0.00 - and would silently poison every downstream match. When
    no layer is usable, the tesseract text produced by
    scripts/ocr_scanned_documents.py is used and the chunk is stamped
    extraction=ocr:..., so an OCR row can never be presented as born-digital.
    """
    # 1. pdfplumber first (layout-aware, no intra-word spacing damage).
    try:
        import pdfplumber
        with pdfplumber.open(path) as pdf:
            pages, repaired = [], 0
            for p in pdf.pages:
                t, fixed = _page_text(p)
                repaired += 1 if fixed else 0
                pages.append(t)
            text = "\n".join(pages)
        if text.strip() and _ARTICLE_RE.search(text) and viet_ratio(text) >= _MIN_VIET_RATIO:
            tag = "pdf-text-layer:pdfplumber"
            if repaired:
                tag += f"+rot-repair:{repaired}"
            return text, tag
    except ImportError:
        sys.stderr.write(
            f"[warn] pdfplumber not installed; falling back to pypdf for "
            f"{os.path.basename(path)}. Install with: pip install pdfplumber\n"
        )
    except Exception as exc:
        sys.stderr.write(f"[warn] pdfplumber failed on {os.path.basename(path)}: {exc}\n")

    # 2. pypdf fallback.
    try:
        from pypdf import PdfReader
    except ImportError:
        raise ProvenanceError(
            f"No PDF library available to read {os.path.basename(path)}. "
            "Install with: pip install pypdf pdfplumber"
        )

    text = ""
    try:
        reader = PdfReader(path)
        text = "\n".join((page.extract_text() or "") for page in reader.pages)
    except Exception as exc:
        sys.stderr.write(f"[warn] pypdf failed on {os.path.basename(path)}: {exc}\n")

    if text.strip() and _ARTICLE_RE.search(text) and viet_ratio(text) >= _MIN_VIET_RATIO:
        return text, "pdf-text-layer:pypdf"

    # 3. OCR fallback.
    ocr = _ocr_fallback(path)
    if ocr.strip():
        return ocr, f"ocr:tesseract-vie:300dpi:{os.path.basename(path)}"

    stem = os.path.splitext(os.path.basename(path))[0] + ".ocr.txt"
    raise ProvenanceError(
        f"No usable text from {os.path.basename(path)}: the embedded layer is absent or "
        f"corrupt (viet_ratio={viet_ratio(text):.3f}) and no OCR text exists at "
        f"data/raw_legal/ocr/{stem}. Run scripts/ocr_scanned_documents.py first; "
        "ingesting the corrupt layer would fabricate the corpus."
    )


def extract_html(path: str) -> str:
    """Extract the readable body of a saved legal HTML page."""
    try:
        from bs4 import BeautifulSoup
    except ImportError:
        raise ProvenanceError(
            f"beautifulsoup4 is not installed; cannot read {os.path.basename(path)}. "
            "Install with: pip install beautifulsoup4 lxml"
        )
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        soup = BeautifulSoup(f.read(), "html.parser")
    for tag in soup(["script", "style", "nav", "header", "footer", "noscript", "form", "aside"]):
        tag.decompose()
    return soup.get_text("\n"), "html-body"


_EXTRACTORS = {"txt": extract_txt, "pdf": extract_pdf, "html": extract_html, "htm": extract_html}


def extract_text(path: str, fmt: str) -> tuple:
    fmt = (fmt or os.path.splitext(path)[1].lstrip(".")).lower()
    if fmt not in _EXTRACTORS:
        raise ProvenanceError(
            f"No extractor for format {fmt!r} ({os.path.basename(path)}). "
            f"Supported: {sorted(_EXTRACTORS)}"
        )
    return _EXTRACTORS[fmt](path)


def disambiguate_chunk_ids(chunks: Sequence[LegalChunk]) -> List[LegalChunk]:
    """Ensure every chunk has a unique and deterministic chunk_id.

    When chunk_ids collide (due to multiple clauses sharing an ID, amending
    provisions, repeated annexes, or multi-part documents under the same doc_id),
    appends a stable collision suffix (-2, -3, ...) in existing chunk order.
    Chunk text and all other attributes remain byte-identical.
    """
    seen: set = set()
    counts: Dict[str, int] = {}
    for c in chunks:
        base_id = c.chunk_id
        if base_id not in seen:
            seen.add(base_id)
            counts[base_id] = 1
        else:
            k = counts[base_id] + 1
            cand = f"{base_id}-{k}"
            while cand in seen:
                k += 1
                cand = f"{base_id}-{k}"
            counts[base_id] = k
            seen.add(cand)
            c.chunk_id = cand
    return list(chunks)


# --- main --------------------------------------------------------------------

def load_plan(path: str = PLAN_PATH):
    if not os.path.exists(path):
        raise ProvenanceError(f"Ingestion plan not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build(plan_path: str, allow_missing: bool, dry_run: bool) -> int:
    plan = load_plan(plan_path)
    documents = plan.get("documents") or []
    if not documents:
        raise ProvenanceError(f"{plan_path} lists no documents")

    all_chunks = []
    missing, ingested, garbage, reports = [], [], [], []

    for entry in sorted(documents, key=lambda d: d.get("priority", 999)):
        doc_id = entry.get("doc_id")
        filename = entry.get("expected_file")
        status = (entry.get("status") or "").lower()
        if status in ("skipped", "blocked", "quarantined"):
            print(f"[skip]    {doc_id}: status={status}")
            continue

        filepath = os.path.join(RAW_DIR, filename or "")
        if not filename or not os.path.exists(filepath):
            missing.append({"doc_id": doc_id, "file": filename, "questions": entry.get("questions")})
            msg = (
                f"[MISSING] {doc_id}: expected {filename} at {filepath} "
                f"({entry.get('questions')} questions)"
            )
            repl = entry.get("replacement_url")
            if repl:
                msg += f"\n          replacement source: {repl}"
            print(msg)
            if not allow_missing:
                continue
            continue

        try:
            raw_text, extraction = extract_text(filepath, entry.get("format", ""))
        except ProvenanceError as exc:
            print(f"[ERROR]   {doc_id}: {exc}")
            raise

        n_articles = len(_ARTICLE_RE.findall(raw_text))
        if n_articles == 0:
            garbage.append({"doc_id": doc_id, "file": filename, "chars": len(raw_text)})
            print(
                f"[SUSPECT] {doc_id}: extracted {len(raw_text)} chars but found NO 'Điều N' - "
                "probably a truncated, paywalled or mis-extracted document. Ingesting it would "
                "produce false hallucination labels."
            )
            if not (allow_missing or entry.get("no_articles_ok")):
                raise ProvenanceError(
                    f"Refusing to ingest {doc_id}: no articles found in {filename}. "
                    "Pass --allow-missing to ingest it anyway."
                )

        parser = LegalDocumentParser(
            doc_id=doc_id,
            doc_title=entry.get("doc_title") or "",
            # Explicit per-entry opt-in for article-less documents (Công văn):
            # INGEST_PLAN must carry "no_articles_ok": true, which is only set
            # after the document was verified complete by other means.
            whole_doc_when_articleless=bool(entry.get("no_articles_ok")),
        )
        # parser.parse_with_report is being added concurrently; use it if present.
        if hasattr(parser, "parse_with_report"):
            chunks, report = parser.parse_with_report(raw_text)
            reports.append({"doc_id": doc_id, "report": report})
        else:
            chunks = parser.parse(raw_text)

        for c in chunks:
            c.corpus_source = CORPUS_TIER2
            c.metadata.setdefault("source_file", filename)
            c.metadata.setdefault("source_url", entry.get("source_url"))
            c.metadata.setdefault("version", entry.get("version"))
            c.metadata.setdefault("questions_carried", entry.get("questions"))
            c.metadata.setdefault("extraction", extraction)

        print(f"[ok]      {doc_id}: {len(chunks):4d} chunks, {n_articles:4d} articles, "
              f"extraction={extraction}")
        ingested.append({"doc_id": doc_id, "chunks": len(chunks), "questions": entry.get("questions")})
        all_chunks.extend(chunks)

    # Disambiguate duplicate chunk IDs (stable collision suffix -2, -3, ... in chunk order)
    n_collided = len(all_chunks) - len({c.chunk_id for c in all_chunks})
    all_chunks = disambiguate_chunk_ids(all_chunks)
    if n_collided:
        print(f"\ndisambiguated {n_collided} collided chunk IDs with stable suffixes")

    print("\n=== ingestion summary ===")
    print(f"documents ingested : {len(ingested)}")
    print(f"chunks produced    : {len(all_chunks)}")
    print(f"missing files      : {len(missing)}")
    print(f"suspect extractions: {len(garbage)}")
    if missing:
        q = sum(m.get("questions") or 0 for m in missing)
        print(f"  -> {q} benchmark questions have NO corpus document:")
        for m in missing:
            print(f"     {m['doc_id']} ({m['questions']} q) <- {m['file']}")
    if garbage:
        for g in garbage:
            print(f"  -> SUSPECT {g['doc_id']}: {g['chars']} chars, no articles")

    if dry_run:
        print("\n[dry-run] not writing output")
        return 0 if not missing else 1

    if OUTPUT_PATH.endswith(f"{CORPUS_TIER1}.json"):
        raise ProvenanceError("Refusing to write the Tier 1 fixture path from the real ingestion script")

    # Guard BEFORE writing: an empty corpus must never clobber a usable one.
    if not all_chunks:
        raise ProvenanceError(
            "Ingestion produced ZERO chunks, so nothing was written and the existing "
            "output was left untouched. No corpus means no retrieval measurement is "
            "possible. Check INGEST_PLAN.json statuses and expected_file names, and "
            "confirm the source files actually arrived in data/raw_legal/."
        )

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    serialized = [asdict(c) for c in all_chunks]
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(serialized, f, ensure_ascii=False, indent=2)
    print(f"\nwrote {len(serialized)} chunks -> {os.path.relpath(OUTPUT_PATH, REPO_ROOT)}")

    if reports:
        report_path = os.path.join(REPO_ROOT, "data", "processed_chunks", "ingest_report.json")
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump({"parse_reports": reports, "missing": missing, "suspect": garbage},
                      f, ensure_ascii=False, indent=2)
        print(f"wrote parse/ingest report -> {os.path.relpath(report_path, REPO_ROOT)}")

    real_garbage = [g for g in garbage if not any(
        d.get("doc_id") == g["doc_id"] and d.get("no_articles_ok")
        for d in documents
    )]
    if allow_missing:
        return 0 if not missing else 1
    return 0 if not (missing or real_garbage) else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--plan", default=PLAN_PATH, help="Path to INGEST_PLAN.json")
    ap.add_argument("--allow-missing", action="store_true",
                    help="Ingest what arrived instead of erroring on gaps")
    ap.add_argument("--dry-run", action="store_true", help="Report only, write nothing")
    args = ap.parse_args()
    try:
        return build(args.plan, args.allow_missing, args.dry_run)
    except ProvenanceError as exc:
        sys.stderr.write(f"\nERROR: {exc}\n")
        return 2


if __name__ == "__main__":
    sys.exit(main())
