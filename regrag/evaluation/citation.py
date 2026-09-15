"""Citation extraction and verification for Vietnamese legal text."""

import re
from typing import List, Dict, Any, Tuple


# Regex patterns for Vietnamese legal citations
# CIRCULAR_PATTERN kept for backwards-compatibility; expanded to cover full legal instruments
CIRCULAR_PATTERN = re.compile(
    r"(?:Thông\s*tư|TT|Nghị\s*định|NĐ|Quyết\s*định|QĐ|Pháp\s*lệnh|Công\s*văn|CV|Luật|Bộ\s*luật)\s*(?:số)?\s*"
    r"(\d+[\/\-][\w\-]+(?:[\/\-][A-ZĐa-zđ\d\-]+)*)",
    re.IGNORECASE,
)

NAMED_DOC_PATTERNS = [
    (re.compile(r"\bLuật\s+(?:các\s+)?tổ\s+chức\s+tín\s+dụng(?:\s+năm|\s+số)?(?:\s+\d{4})?\b", re.IGNORECASE), "32/2024/QH15"),
    (re.compile(r"\bLuật\s+công\s+chứng(?:\s+năm|\s+số)?(?:\s+\d{4})?\b", re.IGNORECASE), "46/2024/QH15"),
    (re.compile(r"\bBộ\s+luật\s+dân\s+sự(?:\s+năm|\s+số)?(?:\s+\d{4})?\b", re.IGNORECASE), "91/2015/QH13"),
]

EXPLICIT_DOC_PATTERN = re.compile(
    r"(?:"
    r"(?:Thông\s*tư|TT|Nghị\s*định|NĐ|Quyết\s*định|QĐ|Pháp\s*lệnh|Công\s*văn|CV|Luật|Bộ\s*luật)\s*(?:số)?\s*"
    r"(\d+[\/\-][\w\-]+(?:[\/\-][A-ZĐa-zđ\d\-]+)*)"
    r"|"
    r"\b(\d+[\/\-]\d{4}[\/\-][A-ZĐa-zđ\d\-]+)\b"
    r"|"
    r"\b(\d+[\/\-](?:QĐ|QD|NĐ|ND|TT)[\/\-][A-ZĐa-zđ\d\-]+)\b"
    r")",
    re.IGNORECASE,
)

ARTICLE_PATTERN = re.compile(r"\bĐiều\s*(\d+[a-zA-Z]?)", re.IGNORECASE)
CLAUSE_PATTERN = re.compile(r"\b[Kk]hoản\s*(\d+)", re.IGNORECASE)

MAX_DOC_DISTANCE = 150
MAX_CLAUSE_DISTANCE = 100


def normalize_doc_id(raw: str) -> str:
    """Normalize a document citation string to canonical form."""
    s = (raw or "").strip()
    if not s:
        return "UNKNOWN"
    for pat, target in NAMED_DOC_PATTERNS:
        if pat.search(s):
            return target
    s = re.sub(
        r"^(?:Thông\s*tư|TT|Nghị\s*định|NĐ|Quyết\s*định|QĐ|Pháp\s*lệnh|Công\s*văn|CV|Luật|Bộ\s*luật)\s*(?:số)?\s*",
        "",
        s,
        flags=re.IGNORECASE,
    ).strip()
    s = re.sub(r"[\/\-]ND[\/\-]CP\b", "/NĐ-CP", s, flags=re.IGNORECASE)
    s = re.sub(r"[\/\-]NĐ[\/\-]CP\b", "/NĐ-CP", s, flags=re.IGNORECASE)
    s = re.sub(r"[\/\-]QD[\/\-]NHNN\b", "/QĐ-NHNN", s, flags=re.IGNORECASE)
    s = re.sub(r"[\/\-]QĐ[\/\-]NHNN\b", "/QĐ-NHNN", s, flags=re.IGNORECASE)
    cv_m = re.match(r"(\d+)[\/\-]([A-Za-z-]+?)[\/\-](\d{4})$", s, re.IGNORECASE)
    if cv_m:
        return f"{cv_m.group(1)}/{cv_m.group(3)}/{cv_m.group(2).upper()}"
    m = re.match(r"^(\d+)[\/\-](\d{4})[\/\-](.+)$", s)
    if m:
        num, yr, rest = m.group(1), m.group(2), m.group(3).upper()
        if len(num) < 2:
            num = num.zfill(2)
        rest = rest.replace("ND-CP", "NĐ-CP")
        return f"{num}/{yr}/{rest}"
    m2 = re.match(r"^(\d+)[\/\-](.+)$", s)
    if m2:
        num, rest = m2.group(1), m2.group(2).upper()
        rest = rest.replace("ND-CP", "NĐ-CP").replace("QD-NHNN", "QĐ-NHNN")
        return f"{num}/{rest}"
    return s.upper()


def extract_citations(text: str) -> List[Dict[str, Any]]:
    """Extract circular/decree/law, article, and clause citations from generated text.

    Associates each Điều with the nearest document mention and nearest clause
    mention within proximity windows, preventing broadcast of unrelated clauses
    or documents across multiple articles.
    """
    articles = []
    for m in ARTICLE_PATTERN.finditer(text):
        articles.append({"id": m.group(1), "start": m.start(), "end": m.end()})

    if not articles:
        return []

    # Find all mentioned legal documents with character spans
    doc_spans = []
    for pat, doc_id in NAMED_DOC_PATTERNS:
        for m in pat.finditer(text):
            doc_spans.append((m.start(), m.end(), doc_id))
    for m in EXPLICIT_DOC_PATTERN.finditer(text):
        raw = m.group(1) or m.group(2) or m.group(3) or m.group(0)
        norm = normalize_doc_id(raw)
        doc_spans.append((m.start(), m.end(), norm))

    doc_spans.sort(key=lambda s: (s[0], -(s[1] - s[0])))
    merged_docs = []
    for s_start, s_end, d_id in doc_spans:
        if not merged_docs:
            merged_docs.append((s_start, s_end, d_id))
        else:
            last_start, last_end, _ = merged_docs[-1]
            if s_start < last_end:
                continue
            merged_docs.append((s_start, s_end, d_id))

    # Find clause spans
    clause_spans = []
    for m in CLAUSE_PATTERN.finditer(text):
        clause_spans.append((m.start(), m.end(), m.group(1)))

    # Assign clauses: each clause attaches to its single closest article within window
    article_clauses: Dict[int, List[str]] = {i: [] for i in range(len(articles))}
    for c_start, c_end, cl_id in clause_spans:
        best_art_idx = None
        min_dist = float("inf")
        for i, art in enumerate(articles):
            dist = max(0, art["start"] - c_end, c_start - art["end"])
            if dist < min_dist:
                min_dist = dist
                best_art_idx = i
        if best_art_idx is not None and min_dist <= MAX_CLAUSE_DISTANCE:
            article_clauses[best_art_idx].append(cl_id)

    # For each article, associate with the nearest document mention within window
    citations: List[Dict[str, Any]] = []
    for i, art in enumerate(articles):
        nearest_doc = "UNKNOWN"
        min_dist = float("inf")
        for d_start, d_end, d_id in merged_docs:
            dist = max(0, art["start"] - d_end, d_start - art["end"])
            if dist < min_dist:
                min_dist = dist
                nearest_doc = d_id
        if min_dist > MAX_DOC_DISTANCE:
            nearest_doc = "UNKNOWN"

        cl_list = article_clauses[i]
        clause_id = cl_list[0] if cl_list else None

        citations.append({
            "doc_id": nearest_doc,
            "article_id": art["id"],
            "clause_id": clause_id,
        })
    return citations


def _norm_doc_for_cmp(doc: str) -> str:
    d = (doc or "").strip().upper()
    if not d or d == "UNKNOWN":
        return "UNKNOWN"
    d = d.replace("ND-CP", "NĐ-CP").replace("-", "/")
    # normalize things like 6/2019 vs 06/2019
    parts = d.split("/")
    if len(parts) >= 2 and parts[0].isdigit() and len(parts[0]) == 1:
        parts[0] = "0" + parts[0]
    return "/".join(parts)


def compute_citation_precision_recall(
    predicted_citations: List[Dict[str, Any]],
    gold_citations: List[Dict[str, Any]],
) -> Tuple[float, float]:
    """Compute citation precision and recall at the Article / Điều level.

    Requires document matching when predicted doc is known (doc mismatch is
    not a match), while tolerating UNKNOWN on the predicted side. Clause
    agreement is not required.
    """
    if not gold_citations:
        if not predicted_citations:
            return 1.0, 1.0
        return 0.0, 1.0

    if not predicted_citations:
        return 0.0, 0.0

    def to_key(c: Dict[str, Any]) -> Tuple[str, str]:
        doc = _norm_doc_for_cmp(str(c.get("doc_id", "")))
        art = str(c.get("article_id", "")).strip()
        return doc, art

    pred_keys = list({to_key(c) for c in predicted_citations})
    gold_keys = list({to_key(c) for c in gold_citations})

    # Pass 1: exact doc + article matches
    matched_gold = set()
    matched_pred = set()
    for pi, (p_doc, p_art) in enumerate(pred_keys):
        if p_doc != "UNKNOWN":
            for gi, (g_doc, g_art) in enumerate(gold_keys):
                if gi not in matched_gold and p_art == g_art and p_doc == g_doc:
                    matched_gold.add(gi)
                    matched_pred.add(pi)
                    break

    # Pass 2: predicted citations with doc='UNKNOWN' match remaining gold by article
    for pi, (p_doc, p_art) in enumerate(pred_keys):
        if pi not in matched_pred and p_doc == "UNKNOWN":
            for gi, (g_doc, g_art) in enumerate(gold_keys):
                if gi not in matched_gold and p_art == g_art:
                    matched_gold.add(gi)
                    matched_pred.add(pi)
                    break

    true_positives = len(matched_gold)
    precision = true_positives / len(pred_keys) if pred_keys else 0.0
    recall = true_positives / len(gold_keys) if gold_keys else 0.0
    return min(1.0, precision), min(1.0, recall)

