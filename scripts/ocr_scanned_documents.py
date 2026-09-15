"""Re-OCR the delivered legal PDFs whose embedded text layer is missing or corrupt.

Two distinct failure modes were measured in the 2026-09-11 delivery of 22 files:

  * no text layer at all  - pypdf returns ~0 chars (9 files, e.g. `91.signed.pdf`)
  * a CORRUPT text layer  - tens of thousands of chars, but Vietnamese diacritics are
    destroyed by whatever OCR produced the PDF, e.g. `17-nhnn.pdf` yields
    "NGAN ILANG lu,t xudc CQNG HoA xA HoI clru NGHia VIET NAM" and contains ZERO
    occurrences of "Điều" across 66,024 characters.

The second mode is the dangerous one: it passes any naive "does this file have text?"
check and then silently fails every downstream match, which is how a corpus-wide
extraction defect gets misdiagnosed as a bad annotation. So this script measures text
quality before deciding to OCR, and it records what it did.

Quality test: the share of letters that carry a Vietnamese-specific diacritic. Genuine
Vietnamese legal prose sits around 0.15-0.25; the corrupt layers measure 0.00.

OCR output is NOT equivalent to a born-digital text layer and must never be treated as
such. Every file written here is stamped with its provenance so the coverage invariant
can classify a match as `ocr` rather than `strict`:

    extraction = ocr:tesseract-<version>-vie:<dpi>dpi

Duplicates are skipped by content hash - `17-2024-tt-nhnn.pdf` and `17-nhnn.pdf` are
byte-identical, as are the two `06-2019` deliveries.

Usage:
  python3 scripts/ocr_scanned_documents.py --inbox DIR --out data/raw_legal/ocr
  python3 scripts/ocr_scanned_documents.py --inbox DIR --report-only
"""

import argparse
import concurrent.futures as cf
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEFAULT_OUT = os.path.join(REPO_ROOT, "data", "raw_legal", "ocr")

# Letters that only occur in Vietnamese with a diacritic attached.
_VIET_LETTERS = set("ăâêôơưđĂÂÊÔƠƯĐáàảãạấầẩẫậắằẳẵặéèẻẽẹếềểễệ"
                    "íìỉĩịóòỏõọốồổỗộớờởỡợúùủũụứừửữựýỳỷỹỵ"
                    "ÁÀẢÃẠẤẦẨẪẬẮẰẲẴẶÉÈẺẼẸẾỀỂỄỆÍÌỈĨỊ"
                    "ÓÒỎÕỌỐỒỔỖỘỚỜỞỠỢÚÙỦŨỤỨỪỬỮỰÝỲỶỸỴ")
_ASCII = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ")

_DIEU_RE = re.compile(r"Điều\s*\d+")
_MIN_QUALITY = 0.05     # below this the text layer is not usable Vietnamese
_workers_default = 6


def sha1(path: str) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def pdf_text(path: str) -> tuple:
    from pypdf import PdfReader
    reader = PdfReader(path)
    parts = []
    for page in reader.pages:
        try:
            parts.append(page.extract_text() or "")
        except Exception:
            parts.append("")
    return "\n".join(parts), len(reader.pages)


def viet_ratio(text: str) -> float:
    letters = [c for c in text if c in _VIET_LETTERS or c in _ASCII]
    if not letters:
        return 0.0
    return sum(1 for c in letters if c in _VIET_LETTERS) / float(len(letters))


def classify(path: str) -> dict:
    """Measure whether the embedded text layer is usable."""
    text, pages = pdf_text(path)
    chars = len(text.strip())
    q = viet_ratio(text)
    if chars < 50:
        state = "no-text-layer"
    elif q < _MIN_QUALITY:
        state = "corrupt-text-layer"
    else:
        state = "clean"
    return {
        "file": os.path.basename(path), "pages": pages, "chars": chars,
        "viet_ratio": round(q, 4), "dieu_hits": len(_DIEU_RE.findall(text)),
        "state": state, "needs_ocr": state != "clean",
    }


def ocr_page(args) -> tuple:
    """Render one page at `dpi` and OCR it with tesseract -l vie."""
    path, index, dpi = args
    tmp = tempfile.mkdtemp(prefix="ocr-")
    try:
        stem = os.path.join(tmp, "p")
        subprocess.run(
            ["pdftoppm", "-r", str(dpi), "-png", "-f", str(index + 1),
             "-l", str(index + 1), path, stem],
            check=True, capture_output=True, timeout=300,
        )
        pngs = sorted(f for f in os.listdir(tmp) if f.endswith(".png"))
        if not pngs:
            return index, ""
        proc = subprocess.run(
            ["tesseract", os.path.join(tmp, pngs[0]), "stdout", "-l", "vie", "--psm", "1"],
            check=False, capture_output=True, timeout=600,
        )
        return index, proc.stdout.decode("utf-8", "replace")
    except Exception as exc:
        return index, f"[OCR-ERROR page {index + 1}: {exc}]"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def ocr_document(path: str, dpi: int, workers: int) -> tuple:
    _text, pages = pdf_text(path)
    if pages <= 0:
        return "", 0
    jobs = [(path, i, dpi) for i in range(pages)]
    out = [""] * pages
    with cf.ThreadPoolExecutor(max_workers=workers) as ex:
        for index, text in ex.map(ocr_page, jobs):
            out[index] = text
    return "\n".join(out), pages


def tesseract_version() -> str:
    try:
        v = subprocess.run(["tesseract", "--version"], capture_output=True, timeout=30)
        return v.stdout.decode().splitlines()[0].split()[-1]
    except Exception:
        return "unknown"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--inbox", required=True, help="Directory holding the delivered PDFs")
    ap.add_argument("--out", default=DEFAULT_OUT, help="Where to write OCR'd text")
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--workers", type=int, default=_workers_default)
    ap.add_argument("--report-only", action="store_true",
                    help="Measure text-layer quality and stop; OCR nothing")
    ap.add_argument("--only", help="Comma-separated filenames to OCR (default: all that need it)")
    ap.add_argument("--json", dest="json_out", help="Write the quality/OCR report as JSON")
    args = ap.parse_args()

    for tool in ("pdftoppm", "tesseract"):
        if not shutil.which(tool):
            raise SystemExit(
                f"{tool} is not installed. On Debian/Ubuntu: "
                "sudo apt-get update && sudo apt-get install poppler-utils "
                "tesseract-ocr tesseract-ocr-vie"
            )
    langs = subprocess.run(["tesseract", "--list-langs"], capture_output=True).stdout.decode()
    if "vie" not in langs:
        raise SystemExit("tesseract has no 'vie' language pack: sudo apt-get install tesseract-ocr-vie")

    pdfs = sorted(f for f in os.listdir(args.inbox) if f.lower().endswith(".pdf"))
    infos, seen = [], {}
    for name in pdfs:
        info = classify(os.path.join(args.inbox, name))
        digest = sha1(os.path.join(args.inbox, name))
        if digest in seen:
            info.update(state="duplicate", needs_ocr=False, duplicate_of=seen[digest])
        else:
            seen[digest] = name
        infos.append(info)

    print("=== embedded text-layer quality ===")
    print(f"{'file':<40}{'pgs':>5}{'chars':>8}{'viet':>7}{'Điều':>6}  state")
    for i in infos:
        print(f"{i['file'][:39]:<40}{i['pages']:>5}{i['chars']:>8}"
              f"{i['viet_ratio']:>7.3f}{i['dieu_hits']:>6}  {i['state']}")

    todo = [i for i in infos if i["needs_ocr"]]
    if args.only:
        keep = set(args.only.split(","))
        todo = [i for i in todo if i["file"] in keep]
    print(f"\n{len(todo)} file(s) need OCR, "
          f"{sum(i['pages'] for i in todo)} pages total")

    if args.report_only:
        if args.json_out:
            with open(args.json_out, "w", encoding="utf-8") as f:
                json.dump({"quality": infos}, f, ensure_ascii=False, indent=2)
        return 0
    if not todo:
        print("nothing to do")
        return 0

    os.makedirs(args.out, exist_ok=True)
    version = tesseract_version()
    for i in todo:
        src = os.path.join(args.inbox, i["file"])
        print(f"[ocr] {i['file']} ({i['pages']} pages, was {i['state']}) ...", flush=True)
        text, pages = ocr_document(src, args.dpi, args.workers)
        q = viet_ratio(text)
        header = (
            f"<!-- provenance: ocr\n"
            f"     source_file: {i['file']}\n"
            f"     extraction: ocr:tesseract-{version}-vie:{args.dpi}dpi\n"
            f"     pages: {pages}\n"
            f"     prior_text_layer: {i['state']}\n"
            f"     ocr_viet_ratio: {round(q, 4)}\n"
            f"     ocr_dieu_hits: {len(_DIEU_RE.findall(text))}\n"
            f"     NOTE: OCR output is not a born-digital text layer. Matches against it\n"
            f"     must be classified 'ocr', never 'strict'. -->\n"
        )
        dest = os.path.join(args.out, os.path.splitext(i["file"])[0] + ".ocr.txt")
        with open(dest, "w", encoding="utf-8") as f:
            f.write(header + text)
        i.update(ocr_chars=len(text), ocr_viet_ratio=round(q, 4),
                 ocr_dieu_hits=len(_DIEU_RE.findall(text)), ocr_out=dest)
        print(f"      -> {len(text):>8} chars, viet={q:.3f}, "
              f"Điều={i['ocr_dieu_hits']:<4} {os.path.relpath(dest, REPO_ROOT)}", flush=True)

    print("\n=== after OCR ===")
    for i in todo:
        print(f"  {i['file']:<40} before: {i['chars']:>7}ch viet={i['viet_ratio']:.3f} "
              f"Điều={i['dieu_hits']:<4} | after: {i.get('ocr_chars', 0):>7}ch "
              f"viet={i.get('ocr_viet_ratio', 0):.3f} Điều={i.get('ocr_dieu_hits', 0)}")

    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump({"quality": infos, "tesseract": version, "dpi": args.dpi},
                      f, ensure_ascii=False, indent=2)
        print(f"\nwrote report -> {args.json_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
