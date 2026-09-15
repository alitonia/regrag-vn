"""Regex-based parser for Vietnamese legal instruments (Luật, Nghị định, Thông tư).

Segments legal texts into granular Khoản (Clause) units while retaining parent
metadata (Document ID, Title, Part, Chapter, Article ID, Article Title).
"""

import re
from typing import Any, Dict, List, Optional, Tuple

from regrag.models import LegalChunk
from regrag.provenance import CORPUS_TIER2


class LegalDocumentParser:
    """Parses structured Vietnamese legal documents (Luật, Nghị định, Thông tư)."""

    # Matches "Phần thứ nhất. NHỮNG QUY ĐỊNH CHUNG" or "PHẦN I. TỔNG QUAN"
    PART_REGEX = re.compile(
        r"^\s*(Phần\s+(?:thứ\s+[^\.\:\n\r\-]+|[IVXLCDM\d]+)|Phần)[\.\:\-\s]*(.*)$",
        re.IGNORECASE,
    )

    # Matches "Chương I. NHỮNG QUY ĐỊNH CHUNG" or similar
    CHAPTER_REGEX = re.compile(
        r"^\s*(Chương\s+[IVXLCDM\d]+)[\.\:\-\s]*(.*)$",
        re.IGNORECASE,
    )

    # Matches annex headers: "Phụ lục: Hệ số rủi ro 0% cho tiền mặt...", "PHỤ LỤC II. ..."
    # Annexes carry no Điều number, so they are kept under a distinct article id
    # ("PL", "PLII") rather than being dropped; citation gold at Điều level is
    # still not derivable for them and canonical.extract_article_id returns None.
    ANNEX_REGEX = re.compile(
        r"^\s*(Phụ\s*lục)\s*([IVXLCDM\d]*)[\.\:\-\s]*(.*)$",
        re.IGNORECASE,
    )

    # Tolerant match for article headers: "Điều 15. Hạn mức", "Đi ều 8", "Điều8", "ĐIỀU 12:",
    # "TT61 - Điều 3:", markdown-wrapped "**... Điều 3. Tiêu đề**"
    ARTICLE_REGEX = re.compile(
        r"^\s*(?:.*?\s*[-—]\s*)?(?:Đ|D|đ|d)\s*[iI]\s*(?:ề|ê|e|Ề|Ê|E)\s*[uU]\s*(\d+[a-zA-Z]?)(?:[\.\:\-\s]*(.*))?$",
        re.IGNORECASE,
    )

    # Matches numbered clauses at start of line: "1. ", "2. ", etc.
    CLAUSE_REGEX = re.compile(r"^(\d+)\.\s+", re.MULTILINE)

    # Matches inline clause definitions: "Khoản 1: ...", "1. ..."
    CLAUSE_IN_LINE_REGEX = re.compile(r"^(?:[Kk]hoản\s*(\d+)|(\d+)\.)[\.\:\s]*(.*)$")

    # Matches stray page headers/footers commonly found in extracted PDFs
    PAGE_HEADER_FOOTER_REGEX = re.compile(
        r"^(?:Trang\s+\d+(?:\s*/\s*\d+)?|Page\s+\d+(?:\s+(?:of|/)\s+\d+)?|-+\s*\d+\s*-+|\d+)\s*$",
        re.IGNORECASE,
    )

    # Matches a line that OPENS with a point/clause citation into an article, e.g.
    # "Điểm c Khoản 1 Điều 18 Thông tư 48/2018/TT-NHNN: ...". Such a line carries no
    # article header of its own; anchored at the start so ordinary preamble prose
    # that merely mentions an article is not mistaken for one.
    INLINE_ARTICLE_RE = re.compile(
        r"^\s*(?:Điểm\s*(?P<point>[a-zA-Z]+)\s*)?(?:Khoản\s*(?P<clause>\d+)\s*)?"
        r"Điều\s*(?P<art>\d+[a-zA-Z]?)\b",
        re.IGNORECASE,
    )

    def __init__(
        self,
        doc_id: str,
        doc_title: str,
        corpus_source: str = CORPUS_TIER2,
        include_preamble: bool = True,
        whole_doc_when_articleless: bool = False,
    ) -> None:
        self.doc_id = doc_id
        self.doc_title = doc_title
        self.corpus_source = corpus_source
        self.include_preamble = include_preamble
        # Opt-in for documents that legitimately have no numbered articles
        # (e.g. Công văn dispatches): emit the whole text as one chunk instead
        # of returning []. Default False keeps the truncation guard intact.
        self.whole_doc_when_articleless = whole_doc_when_articleless

    def parse(
        self, raw_text: str, corpus_source: Optional[str] = None
    ) -> List[LegalChunk]:
        """Parse raw text into a list of LegalChunk objects (backward compatible)."""
        chunks, _report = self.parse_with_report(
            raw_text, corpus_source=corpus_source
        )
        return chunks

    def parse_with_report(
        self, raw_text: str, corpus_source: Optional[str] = None
    ) -> Tuple[List[LegalChunk], Dict[str, Any]]:
        """Parse raw text and return both chunks and a comprehensive extraction report.

        Report contains:
          - articles_found: count of distinct article IDs discovered
          - clauses_found: count of granular clause chunks discovered
          - preamble_captured: bool, whether preamble text before first Điều was preserved
          - preamble_text: verbatim captured preamble text
          - lines_dropped: count of lines discarded (headers/footers or unparseable lines)
          - dropped_lines: list of discarded line strings
          - articles_with_missing_title: list of article IDs where title was empty/missing
          - missing_titles: alias for articles_with_missing_title
          - warnings: list of extraction warnings
        """
        source = corpus_source or self.corpus_source
        chunks: List[LegalChunk] = []
        dropped_lines: List[str] = []
        missing_titles: List[str] = []
        warnings: List[str] = []
        unique_article_ids: List[str] = []

        lines = raw_text.splitlines()

        current_part: Optional[str] = None
        current_chapter_title: Optional[str] = None
        current_chapter: Optional[str] = None

        current_article_id: Optional[str] = None
        current_article_title: str = ""
        current_clause_id: Optional[str] = None
        current_clause_lines: List[str] = []

        preamble_lines: List[str] = []
        article_header_pending: bool = False
        pending_title_for_article: Optional[str] = None

        def flush_clause() -> None:
            nonlocal current_clause_lines, current_clause_id, article_header_pending, pending_title_for_article
            if current_article_id and current_clause_lines:
                clause_text = "\n".join(current_clause_lines).strip()
                if clause_text:
                    chunk_id = f"{self.doc_id}_D{current_article_id}"
                    if current_clause_id:
                        chunk_id += f"_K{current_clause_id}"
                    else:
                        chunk_id += "_Kall"

                    chunk_meta: Dict[str, Any] = {}
                    if current_part:
                        chunk_meta["part"] = current_part
                    if current_chapter_title:
                        chunk_meta["chapter"] = current_chapter_title

                    chunk = LegalChunk(
                        chunk_id=chunk_id,
                        doc_id=self.doc_id,
                        doc_title=self.doc_title,
                        chapter=current_chapter or current_part,
                        article_id=current_article_id,
                        article_title=current_article_title,
                        clause_id=current_clause_id,
                        text=clause_text,
                        metadata=chunk_meta,
                        corpus_source=source,
                    )
                    chunks.append(chunk)
            current_clause_lines = []
            article_header_pending = False
            pending_title_for_article = None

        for line in lines:
            line_str = line.strip()
            if not line_str:
                continue

            # Check and filter stray PDF headers/footers
            if self.PAGE_HEADER_FOOTER_REGEX.match(line_str):
                dropped_lines.append(line_str)
                continue

            # Check Phần (Part)
            part_match = self.PART_REGEX.match(line_str)
            if part_match:
                flush_clause()
                part_label = part_match.group(1).strip()
                part_title = part_match.group(2).strip()
                current_part = (
                    f"{part_label}: {part_title}" if part_title else part_label
                )
                current_chapter = None
                current_chapter_title = None
                continue

            # Check Chương (Chapter)
            chap_match = self.CHAPTER_REGEX.match(line_str)
            if chap_match:
                flush_clause()
                chap_label = chap_match.group(1).strip()
                chap_title = chap_match.group(2).strip()
                chap_formatted = (
                    f"{chap_label}: {chap_title}" if chap_title else chap_label
                )
                current_chapter_title = chap_formatted
                current_chapter = (
                    f"{current_part} / {chap_formatted}"
                    if current_part
                    else chap_formatted
                )
                continue

            # Check Phụ lục (annex) before the preamble branch, otherwise a
            # single-line annex passage is swallowed as preamble text.
            annex_match = self.ANNEX_REGEX.match(line_str)
            if annex_match:
                flush_clause()
                annex_num = (annex_match.group(2) or "").strip()
                current_article_id = f"PL{annex_num}" if annex_num else "PL"
                if current_article_id not in unique_article_ids:
                    unique_article_ids.append(current_article_id)
                current_article_title = (annex_match.group(3) or "").strip()
                current_clause_id = None
                current_clause_lines.append(line_str)
                article_header_pending = False
                continue

            # Check Điều (Article)
            clean_line = line_str.strip("*").strip()
            art_match = self.ARTICLE_REGEX.match(clean_line)
            if art_match:
                flush_clause()
                art_num = art_match.group(1)
                rest = (art_match.group(2) or "").strip()
                current_article_id = art_num
                if art_num not in unique_article_ids:
                    unique_article_ids.append(art_num)

                current_clause_id = None
                current_clause_lines.append(line_str)
                article_header_pending = True

                # Check if the rest of the line starts with an inline clause
                inline_cl = self.CLAUSE_IN_LINE_REGEX.match(rest)
                if inline_cl:
                    current_article_title = ""
                    missing_titles.append(art_num)
                    current_clause_id = inline_cl.group(1) or inline_cl.group(2)
                    article_header_pending = False
                    pending_title_for_article = None
                elif rest:
                    current_article_title = rest
                    pending_title_for_article = None
                else:
                    current_article_title = ""
                    pending_title_for_article = art_num
                continue

            # A line opening with a point/clause citation into an article has no
            # header of its own; attribute it to the cited article so its text is
            # preserved instead of being swallowed as preamble and then discarded.
            inline = self.INLINE_ARTICLE_RE.match(line_str)
            if inline and current_article_id is None:
                flush_clause()
                current_article_id = inline.group("art")
                if current_article_id not in unique_article_ids:
                    unique_article_ids.append(current_article_id)
                current_article_title = ""
                current_clause_id = inline.group("clause")
                current_clause_lines.append(line_str)
                article_header_pending = False
                continue

            # Before first article, accumulate preamble lines
            if current_article_id is None:
                preamble_lines.append(line_str)
                continue

            # Within an article: check if resolving pending title from following line
            if pending_title_for_article is not None:
                # If next non-empty line starts a clause, then title was omitted
                cl_match = self.CLAUSE_REGEX.match(line_str)
                cl_inline = self.CLAUSE_IN_LINE_REGEX.match(line_str)
                if cl_match or cl_inline:
                    current_article_title = ""
                    missing_titles.append(pending_title_for_article)
                    pending_title_for_article = None
                    current_clause_id = (
                        cl_match.group(1)
                        if cl_match
                        else (cl_inline.group(1) or cl_inline.group(2))
                    )
                    current_clause_lines.append(line_str)
                    article_header_pending = False
                    continue
                else:
                    # Next line is the article title
                    current_article_title = line_str
                    pending_title_for_article = None
                    current_clause_lines.append(line_str)
                    continue

            # Check Clause if within an Article
            cl_match = self.CLAUSE_REGEX.match(line_str)
            if cl_match:
                if article_header_pending:
                    # First clause in article: keep article header attached to this clause
                    current_clause_id = cl_match.group(1)
                    current_clause_lines.append(line_str)
                    article_header_pending = False
                else:
                    flush_clause()
                    current_clause_id = cl_match.group(1)
                    current_clause_lines.append(line_str)
            else:
                current_clause_lines.append(line_str)

        flush_clause()

        # If no articles were found at all, refuse to return chunks and warn -
        # unless the caller explicitly opted in for article-less documents
        # (e.g. a Công văn dispatch, which is prose by nature).
        if not unique_article_ids:
            if self.whole_doc_when_articleless:
                warnings.append(
                    "No articles (Điều) found; emitting whole document as one chunk "
                    "(whole_doc_when_articleless opt-in)"
                )
                whole_text = "\n".join(
                    ln for ln in (preamble_lines + dropped_lines) if ln.strip()
                ).strip()
                chunk = LegalChunk(
                    chunk_id=f"{self.doc_id}_wholedoc",
                    doc_id=self.doc_id,
                    doc_title=self.doc_title,
                    chapter=None,
                    article_id="0",
                    article_title="Toàn văn (văn bản không có Điều)",
                    clause_id=None,
                    text=whole_text,
                    metadata={
                        "type": "whole_document_no_articles",
                        "is_preamble": False,
                    },
                    corpus_source=source,
                )
                report = {
                    "articles_found": 0,
                    "clauses_found": 0,
                    "preamble_captured": False,
                    "preamble_text": "",
                    "lines_dropped": 0,
                    "dropped_lines": [],
                    "articles_with_missing_title": missing_titles,
                    "missing_titles": missing_titles,
                    "warnings": warnings,
                }
                return [chunk], report
            warnings.append("No articles (Điều) found in document")
            dropped_lines.extend(preamble_lines)
            report = {
                "articles_found": 0,
                "clauses_found": 0,
                "preamble_captured": False,
                "preamble_text": "",
                "lines_dropped": len(dropped_lines),
                "dropped_lines": dropped_lines,
                "articles_with_missing_title": missing_titles,
                "missing_titles": missing_titles,
                "warnings": warnings,
            }
            return [], report

        # Emit preamble chunk if preamble text was captured before the first Điều
        preamble_captured = False
        preamble_text = ""
        if preamble_lines:
            preamble_text = "\n".join(preamble_lines).strip()
            if preamble_text and self.include_preamble:
                preamble_chunk = LegalChunk(
                    chunk_id=f"{self.doc_id}_preamble",
                    doc_id=self.doc_id,
                    doc_title=self.doc_title,
                    chapter=None,
                    article_id="0",
                    article_title="Lời nói đầu",
                    clause_id=None,
                    text=preamble_text,
                    metadata={"type": "preamble", "is_preamble": True},
                    corpus_source=source,
                )
                chunks.insert(0, preamble_chunk)
                preamble_captured = True

        clauses_count = sum(1 for c in chunks if c.clause_id is not None)

        report = {
            "articles_found": len(unique_article_ids),
            "clauses_found": clauses_count,
            "preamble_captured": preamble_captured,
            "preamble_text": preamble_text,
            "lines_dropped": len(dropped_lines),
            "dropped_lines": dropped_lines,
            "articles_with_missing_title": missing_titles,
            "missing_titles": missing_titles,
            "warnings": warnings,
        }
        return chunks, report
