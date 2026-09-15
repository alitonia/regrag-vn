#!/usr/bin/env python3
"""Replace annotator-summary gold passages with verbatim article text from the corpus.

46 of the original 65 answerable passages were annotator summaries, not
verbatim extracts, so the verbatim-string coverage invariant could never pass
on them (measured 2026-09-11: 1/65 strict). The benchmark decision recorded in
docs/HANDOFF_2026-09-11.md and settled with the authors on 2026-09-11:

* Gate B is provision-level (the cited Dieu N exists in the cited instrument).
* Verbatim string coverage is reported as a data-quality metric.
* This repair pass substitutes, for summary rows, the verbatim article text
  found in the Tier 2 corpus - which makes the verbatim metric partly
  self-satisfying for repaired rows, and that is disclosed in the paper.

Three loud-failure guards, per the handoff. A row is refused (the script exits
non-zero naming it) when:

* GUARD A - instrument absent: NO cited doc_id has chunks in the corpus (a
  co-cited instrument that cannot be ingested does not block the row; it is
  recorded in the repair log's absent_co_cited field);
* GUARD B - article missing: the passage names no "Dieu N", or no chunk of
  that article exists under any cited instrument (a Phu luc locator without an
  article number is reported as unrepairable-locator and kept as a summary,
  not silently rewritten);
* GUARD C - lexical overlap low: the content-token recall of the summary
  against the candidate article is below MIN_TOKEN_RECALL, meaning the
  article number and the passage disagree - rewriting would attach the row to
  the wrong provision.

Every repair is logged to data/gold/passage_repair_log.json with the original
summary preserved, and the author cell records the repair provenance, so a
repaired row can always be told apart from a human-verbatim one.
"""

import argparse
import csv
import json
import os
import re
import sys
import unicodedata

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CSV_PATH = os.path.join(REPO_ROOT, "data", "gold", "bank_qa_data.csv")
CORPUS_PATH = os.path.join(REPO_ROOT, "data", "processed_chunks", "corpus_chunks.json")
LOG_PATH = os.path.join(REPO_ROOT, "data", "gold", "passage_repair_log.json")

#: Minimum share of the summary's content tokens that must appear in the
#: candidate article for the repair to be trusted (GUARD C).
MIN_TOKEN_RECALL = 0.30

_ARTICLE_RE = re.compile(r"Điều\s*(\d+[a-z]?)", re.IGNORECASE)

_AUTHOR_NOTE = "; passage repaired 2026-09-11 from corpus {doc} Điều {art} (verbatim; was annotator summary)"


def _tokens(text: str) -> set:
    """Lowercased content tokens: >=2 chars, letters/digits, diacritics folded."""
    nfd = unicodedata.normalize("NFD", text or "")
    stripped = "".join(c for c in nfd if not unicodedata.combining(c))
    folded = unicodedata.normalize("NFC", stripped).replace("đ", "d").replace("Đ", "D")
    return {t for t in re.findall(r"[0-9a-zA-ZÀ-ỹ]{2,}", folded.lower())}


def _token_recall(summary: str, article: str) -> float:
    s, a = _tokens(summary), _tokens(article)
    if not s:
        return 0.0
    return len(s & a) / len(s)


def load_corpus(path):
    with open(path, encoding="utf-8") as f:
        chunks = json.load(f)
    by_doc_article = {}
    for ch in chunks:  # corpus order == document order
        key = (ch["doc_id"], ch.get("article_id") or "")
        by_doc_article.setdefault(key, []).append(ch)
    doc_ids = {ch["doc_id"] for ch in chunks}
    return chunks, by_doc_article, doc_ids


def article_text(chunks_for_article):
    return "\n".join(ch["text"] for ch in chunks_for_article)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--csv", default=CSV_PATH)
    ap.add_argument("--corpus", default=CORPUS_PATH)
    ap.add_argument("--log", default=LOG_PATH)
    ap.add_argument(
        "--canonical",
        default=os.path.join(REPO_ROOT, "data", "gold", "questions_canonical.json"),
        help="Canonical GoldQuestion JSON (regenerate with scripts/load_qa_csv.py first)",
    )
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    _chunks, by_doc_article, corpus_doc_ids = load_corpus(args.corpus)

    with open(args.csv, newline="", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = list(reader)
    idx = {name: i for i, name in enumerate(header)}

    with open(args.canonical, encoding="utf-8") as f:
        canonical = {q["id"]: q for q in json.load(f)}

    # Map CSV ID cell -> question id (the loader's convention: Q%03d).
    def qid_of(row):
        raw = (row[idx["ID"]] or "").strip()
        return f"Q{int(raw):03d}" if raw.isdigit() else raw

    repaired, already, unrepairable_locator, errors = [], [], [], []

    for row in rows:
        qid = qid_of(row)
        q = canonical.get(qid)
        if not q or not q.get("is_answerable"):
            continue
        passage = row[idx["text_contains_answer_in_the_doc"]]
        doc_ids = [d for d in q.get("gold_doc_ids", []) if not d.startswith("UNRESOLVED")]
        if not doc_ids:
            continue  # nothing citable; the loader already reports these

        # Idempotency: a row already carrying the joined corpus text of its
        # article is done.
        m = _ARTICLE_RE.search(passage)
        art = m.group(1) if m else None
        if art and any(
            (chs := by_doc_article.get((d, art)))
            and article_text(chs).strip() == passage.strip()
            for d in doc_ids
        ):
            already.append(qid)
            continue

        # GUARD B (locator half): a passage with no article number cannot be
        # repaired by article lookup. Reported, kept as a summary.
        if not art:
            unrepairable_locator.append(qid)
            continue

        # GUARD A: at least one cited instrument must be in the corpus. A row
        # may legitimately co-cite an instrument that cannot be ingested (e.g.
        # an unreadable .doc kept only as a secondary source); the repair then
        # draws from a present instrument and the absent one is recorded in
        # the log rather than blocking the row.
        present = [d for d in doc_ids if d in corpus_doc_ids]
        if not present:
            errors.append(f"{qid}: GUARD A: no cited instrument is in the corpus: {doc_ids}")
            continue
        absent_co_cited = [d for d in doc_ids if d not in corpus_doc_ids]

        # GUARD B (article exists) + GUARD C (lexical overlap high enough to
        # trust the article number). Try each cited instrument in citation
        # order; only fail when none of them clears both guards.
        attempts = []
        chosen = None
        for d in doc_ids:
            chs = by_doc_article.get((d, art))
            if not chs:
                attempts.append(f"{d}: no chunk of Điều {art}")
                continue
            recall = _token_recall(passage, article_text(chs))
            if recall >= MIN_TOKEN_RECALL:
                chosen = (d, chs, recall)
                break
            attempts.append(f"{d}: token recall {recall:.2f} < {MIN_TOKEN_RECALL}")
        if not chosen:
            errors.append(
                f"{qid}: GUARD B/C failed for Điều {art}: " + "; ".join(attempts)
            )
            continue

        d, chs, recall = chosen
        text = article_text(chs)
        extraction = (chs[0].get("metadata") or {}).get("extraction", "?")
        repaired.append({
            "id": qid,
            "doc_id": d,
            "article_id": art,
            "token_recall": round(recall, 3),
            "chunk_ids": [c["chunk_id"] for c in chs],
            "extraction": extraction,
            "absent_co_cited": absent_co_cited,
            "original_summary": passage,
            "verbatim_text": text,
        })
        row[idx["text_contains_answer_in_the_doc"]] = text
        note = _AUTHOR_NOTE.format(doc=d, art=art)
        if note not in row[idx["author"]]:
            row[idx["author"]] = row[idx["author"]] + note

    print(f"repaired:                {len(repaired)}")
    print(f"already verbatim:        {len(already)} -> {already}")
    print(f"unrepairable locator:    {len(unrepairable_locator)} -> {unrepairable_locator}")
    if errors:
        print(f"\nREFUSALS ({len(errors)}):")
        for e in errors:
            print(f"  - {e}")

    if args.dry_run:
        print("\n[dry-run] nothing written")
        return 1 if errors else 0

    if errors:
        print("\nrefusing to write any repair while a guard fails; fix the named rows first")
        return 1

    with open(args.csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, lineterminator="\r\n")
        writer.writerow(header)
        writer.writerows(rows)
    with open(args.log, "w", encoding="utf-8") as f:
        json.dump({
            "_README": "Gold passage repair log (2026-09-11). original_summary is preserved "
                       "verbatim; verbatim_text is the joined corpus chunks of the cited "
                       "article. Repaired rows make the verbatim coverage invariant partly "
                       "self-satisfying - disclose when reporting it.",
            "min_token_recall": MIN_TOKEN_RECALL,
            "repaired": repaired,
            "already_verbatim": already,
            "unrepairable_locator": unrepairable_locator,
        }, f, ensure_ascii=False, indent=2)
    print(f"\nwrote {len(repaired)} repairs -> {os.path.relpath(args.csv, REPO_ROOT)}")
    print(f"log -> {os.path.relpath(args.log, REPO_ROOT)}")
    print("now re-run: scripts/load_qa_csv.py && scripts/build_tier1_corpus.py && scripts/regenerate.py --force")
    return 0


if __name__ == "__main__":
    sys.exit(main())
