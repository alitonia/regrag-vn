"""Canonicalisation from the trusted QA CSV to benchmark identifiers.

Two facts measured on the CSV drive this module's design:

* 64/64 gold passages name an ``Điều`` — article-level gold is always derivable.
* Only 2/64 gold passages name their own instrument — ``doc_id`` can therefore
  never be regexed out of passage or answer prose. It comes from ``doc_link``,
  and where the link is opaque (datafiles.chinhphu.vn PDF filenames, vanban
  docid query strings) it comes from an explicit human-verified manifest.

Anything that cannot be resolved returns an ``UNRESOLVED:`` sentinel rather than
a guess, so an unresolved instrument is visible in the coverage report instead
of silently attaching questions to the wrong document.
"""

import json
import os
import re
import unicodedata
from typing import Dict, List, Optional, Tuple

UNRESOLVED = "UNRESOLVED"

# --- URL extraction ----------------------------------------------------------

_URL_RE = re.compile(r"https?://[^\s\"'<>,;)]+")


def split_urls(cell: str) -> List[str]:
    """Split a ``doc_link`` cell into individual URLs.

    The CSV contains malformed cells: 7 of 64 rows hold more than one URL,
    separated by embedded newlines and blank lines. Naively treating the cell as
    a single link yields 16 "documents" where there are 19 real ones.
    """
    if not cell:
        return []
    seen: List[str] = []
    for u in _URL_RE.findall(cell):
        u = u.rstrip(".")
        if u not in seen:
            seen.append(u)
    return seen


# --- slug patterns for self-describing URLs ---------------------------------
# Each returns (doc_id, confidence). Confidence "slug" means the instrument
# number is spelled out in the URL itself and needs no human verification.

_SLUG_PATTERNS: List[Tuple[str, re.Pattern, str]] = [
    # congbao.chinhphu.vn/van-ban/thong-tu-so-06-2019-tt-nhnn-29358.htm
    ("TT-NHNN", re.compile(r"thong-tu-(?:so-)?(\d+)-(\d{4})-tt-nhnn", re.I), "{n}/{y}/TT-NHNN"),
    # congbao.chinhphu.vn/van-ban/nghi-dinh-so-21-2021-nd-cp-33477/35291.htm
    ("ND-CP", re.compile(r"nghi-dinh-(?:so-)?(\d+)-(\d{4})-nd-cp", re.I), "{n}/{y}/NĐ-CP"),
    # luatvietnam.vn/tu-phap/luat-cong-chung-2024-so-46-2024-qh15-379073-d1.html
    ("QH15", re.compile(r"-so-(\d+)-(\d{4})-qh15", re.I), "{n}/{y}/QH15"),
    # thuvienphapluat.vn/van-ban/.../Nghi-dinh-21-2021-ND-CP-thi-hanh-...
    ("ND-CP", re.compile(r"Nghi-dinh-(\d+)-(\d{4})-ND-CP", re.I), "{n}/{y}/NĐ-CP"),
    ("TT-NHNN", re.compile(r"Thong-tu-(\d+)-(\d{4})-TT-NHNN", re.I), "{n}/{y}/TT-NHNN"),
    # thuvienphapluat.vn/cong-van/.../Cong-van-276-NHNN-TTGSNH-2017-...
    ("CV", re.compile(r"Cong-van-(\d+)-([A-Za-z]+)-?([A-Za-z]*)-(\d{4})", re.I), None),  # special-cased
]

_CV_RE = re.compile(r"Cong-van-(\d+)-([A-Za-z-]+?)-(\d{4})", re.I)

# datafiles.chinhphu.vn/cpp/files/vbpq/<year>/<month>/<name>.pdf — the filename
# number is NOT reliably the instrument number (48-nhnn.pdf is 48/2018/TT-NHNN;
# 61-nhnn.pdf is 61/2025/TT-NHNN), and the path year is the publication year.
# These are resolved by manifest only; the heuristic below is advisory.
_DATAFILES_RE = re.compile(r"/vbpq/(\d{4})/(\d{1,2})/([^/?#]+)\.pdf", re.I)


def canonicalize_doc_id(url: str, manifest: Optional[Dict[str, str]] = None) -> Tuple[str, str]:
    """Map a source URL to a canonical instrument id.

    Returns ``(doc_id, confidence)`` where confidence is one of
    ``"manifest"`` (human-verified), ``"slug"`` (number spelled out in the URL),
    ``"heuristic"`` (advisory only — verify before use), or ``"unresolved"``.
    """
    url = (url or "").strip()
    if not url:
        return f"{UNRESOLVED}:empty", "unresolved"

    if manifest:
        for key, doc_id in manifest.items():
            if key and (key == url or key in url or url in key):
                if doc_id:
                    return doc_id, "manifest"
                return f"{UNRESOLVED}:manifest-pending", "unresolved"

    m = _CV_RE.search(url)
    if m:
        return f"{m.group(1)}/{m.group(3)}/{m.group(2).upper()}", "slug"

    for _kind, pat, fmt in _SLUG_PATTERNS:
        if fmt is None:
            continue
        m = pat.search(url)
        if m:
            n, y = m.group(1), m.group(2)
            return fmt.format(n=n.zfill(2) if len(n) < 2 else n, y=y), "slug"

    m = _DATAFILES_RE.search(url)
    if m:
        stem = m.group(3)
        num = re.search(r"(\d+)", stem)
        if num:
            return (
                f"{UNRESOLVED}:datafiles:{m.group(1)}/{stem}",
                "unresolved",
            )

    return f"{UNRESOLVED}:{_short_hash(url)}", "unresolved"


def _short_hash(s: str) -> str:
    import hashlib
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:8]


# --- gold citation extraction from the verbatim passage ----------------------

_ARTICLE_RE = re.compile(r"Điều\s*(\d+[a-z]?)", re.IGNORECASE)
_CLAUSE_RE = re.compile(r"(?:^|\n)\s*(\d+)\.\s+", re.MULTILINE)
_KHOAN_RE = re.compile(r"[Kk]hoản\s*(\d+)")
_ARTICLE_TITLE_RE = re.compile(r"^\s*Điều\s*(\d+[a-z]?)\s*[\.\:]\s*([^\n\r]+)", re.IGNORECASE)


def extract_article_id(passage: str) -> Optional[str]:
    """First ``Điều N`` in the passage. Available for 64/64 CSV rows."""
    m = _ARTICLE_RE.search(passage or "")
    return m.group(1) if m else None


def extract_article_title(passage: str) -> Optional[str]:
    m = _ARTICLE_TITLE_RE.match((passage or "").strip())
    return m.group(2).strip() if m else None


def extract_clause_ids(passage: str) -> List[str]:
    """Numbered clause labels at line start. Present in only ~24/64 passages,
    so clause-level scoring is optional and must never be required."""
    return _CLAUSE_RE.findall(passage or "")


def extract_gold_citation(passage: str, doc_id: str) -> Optional[Dict[str, object]]:
    """Build the gold citation from the passage header plus the link-derived doc_id."""
    art = extract_article_id(passage)
    if art is None:
        return None
    k = _KHOAN_RE.search(passage or "")
    return {
        "doc_id": doc_id,
        "article_id": art,
        "clause_id": k.group(1) if k else None,
        "confidence": "passage-header",
    }


def prefer_passage_named_doc(passage: str, doc_ids: List[str]) -> Optional[str]:
    """Return the candidate doc_id that the passage itself names, else None.

    Only 2/64 passages name their own instrument, so prose is never a valid
    *source* of doc_id. But when a row carries several doc_links and the passage
    explicitly names one of the candidates, that is direct evidence about which
    instrument is primary - and URL order is not evidence at all.

    Real case: Q007 lists ['23/2015/NĐ-CP', '48/2018/TT-NHNN'] while its passage
    reads "Điểm c Khoản 1 Điều 18 Thông tư 48/2018/TT-NHNN". Taking the first URL
    labelled the gold citation with the wrong instrument.
    """
    if not passage or not doc_ids:
        return None
    hay = normalize_ws(passage)
    for doc_id in doc_ids:
        if not doc_id or doc_id.startswith(UNRESOLVED):
            continue
        if normalize_ws(doc_id) in hay:
            return doc_id
        # "48/2018/TT-NHNN" also appears written as "48/2018/TT-NHNN" inside
        # "Thông tư 48/2018/TT-NHNN"; the number/year core is the stable part.
        core = doc_id.split("/")[0]
        parts = doc_id.split("/")
        if len(parts) >= 2 and f"{core}/{parts[1]}" in hay:
            return doc_id
    return None


# --- normalisation for the coverage test ------------------------------------

def normalize_ws(text: str) -> str:
    """NFC-normalise and collapse whitespace, preserving diacritics.

    Diacritics are kept at this tier on purpose: folding them would let a
    mis-extracted passage match the wrong provision and hide PDF damage.
    """
    text = unicodedata.normalize("NFC", text or "")
    return re.sub(r"\s+", " ", text).strip()


def fold_diacritics(text: str) -> str:
    """Second, looser matching tier — used only to *diagnose* extraction damage.

    A passage that matches only after folding means the ingested text lost
    diacritics. That is reported, never silently accepted.
    """
    nfd = unicodedata.normalize("NFD", normalize_ws(text))
    stripped = "".join(c for c in nfd if not unicodedata.combining(c))
    return unicodedata.normalize("NFC", stripped).replace("đ", "d").replace("Đ", "D")


def squash_ws(text: str) -> str:
    """Third matching tier — NFC-normalise, then remove ALL whitespace.

    The born-digital CÔNG BÁO PDF layers in this delivery insert spurious
    intra-word spaces ("vi ệc", "th ường", "tổ ch ức tín d ụng"), so a
    verbatim gold passage fails the strict tier even though every character
    is present and in order. Squashing whitespace on both sides matches that
    specific damage mode while still preserving diacritics and character
    order, so it cannot manufacture a match between different provisions.
    A match at this tier is reported as ``squash``, never as ``strict``:
    an exact-tier claim must never be manufactured by deleting evidence.
    """
    return re.sub(r"\s+", "", unicodedata.normalize("NFC", text or ""))


def manifest_path(repo_root: str) -> str:
    return os.path.join(repo_root, "data", "raw_legal", "DOC_MANIFEST.json")


def load_manifest(path: str) -> Dict[str, str]:
    """Load the human-verified URL -> doc_id manifest. Missing file = empty."""
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {k: (v or "") for k, v in (data.get("urls") or data).items()}
