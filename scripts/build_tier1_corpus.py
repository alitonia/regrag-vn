"""Build Tier 1 development fixture corpus from QA CSV gold passages.

This script constructs a retrieval corpus directly from the verbatim passages in
`data/gold/bank_qa_data.csv` - one chunk per ANSWERABLE question that has a gold
passage. Unanswerable probes have no passage by design, so there is nothing to
build a chunk from; they are excluded here and reported by id, never silently
dropped (see `probes_excluded` in the build summary).

WARNING: Tier 1 is a DEVELOPMENT FIXTURE, not a scientific result. Because
each question's gold passage is in the corpus by construction, Recall@3 is
trivially ~1.0 and BM25 cannot be distinguished from dense retrieval. Chunks
are stamped with corpus_source = CORPUS_TIER1 and must never be aggregated
into publishable benchmark tables.
"""

import argparse
from collections import Counter
from dataclasses import asdict
import hashlib
import json
import os
import sys
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from regrag.corpus.canonical import (
    extract_article_id,
    extract_article_title,
    normalize_ws,
)
from regrag.corpus.qa_loader import load_gold_questions
from regrag.models import GoldQuestion, LegalChunk
from regrag.provenance import CORPUS_TIER1


def assert_safe_output_path(out_path: str) -> None:
    """Refuse to overwrite the real corpus path (corpus_chunks.json)."""
    norm = os.path.normpath(os.path.abspath(out_path))
    base = os.path.basename(norm)
    if base == "corpus_chunks.json":
        raise ValueError(
            f"Refusing to write to real corpus path: {out_path}. "
            "Tier 1 script must NEVER write to or overwrite corpus_chunks.json!"
        )


def build_tier1_chunks(questions: List[GoldQuestion]) -> Tuple[List[LegalChunk], Dict[str, object]]:
    """Convert GoldQuestion gold passages into LegalChunk objects.

    Unanswerable probes (`is_answerable=False`) carry no gold passage, so there
    is nothing to chunk; they are skipped and named in `probes_excluded` /
    `probe_ids_excluded` rather than dropped quietly. An *answerable* question
    with an empty passage is a data defect and is reported separately in
    `empty_passage_excluded` so it cannot hide behind the probe bucket.

    Returns:
        Tuple of (chunks, summary_stats)
    """
    chunks: List[LegalChunk] = []
    seen_hashes: Counter = Counter()
    probe_ids: List[str] = []
    empty_passage_ids: List[str] = []

    for q in questions:
        if not q.is_answerable:
            probe_ids.append(q.id)
            continue
        if not (q.gold_passage or "").strip():
            empty_passage_ids.append(q.id)
            continue

        # Stable chunk_id derived from normalized content hash
        norm_passage = normalize_ws(q.gold_passage)
        h = hashlib.sha256(norm_passage.encode("utf-8")).hexdigest()[:16]
        seen_hashes[h] += 1
        if seen_hashes[h] == 1:
            chunk_id = f"tier1_{h}"
        else:
            chunk_id = f"tier1_{h}_{seen_hashes[h]}"

        # Canonical doc_id resolution
        if q.doc_ids_resolved:
            doc_id = q.gold_doc_ids[0]
            is_unresolved = False
        else:
            doc_id = q.gold_doc_ids[0] if q.gold_doc_ids else "UNRESOLVED:empty"
            is_unresolved = True

        art_id = extract_article_id(q.gold_passage) or ""
        art_title = extract_article_title(q.gold_passage) or ""

        clause_id = None
        if q.gold_citations and q.gold_citations[0].get("clause_id"):
            clause_id = str(q.gold_citations[0]["clause_id"])

        meta = {
            "question_id": q.id,
            "doc_ids_resolved": q.doc_ids_resolved,
            "unresolved_doc_id": is_unresolved,
            "doc_id_confidence": q.doc_id_confidence,
            "source_urls": q.source_urls,
            "gold_doc_ids": q.gold_doc_ids,
            "author": q.author,
        }

        chunk = LegalChunk(
            chunk_id=chunk_id,
            doc_id=doc_id,
            doc_title="",
            chapter=None,
            article_id=art_id,
            article_title=art_title,
            clause_id=clause_id,
            text=q.gold_passage,
            metadata=meta,
            corpus_source=CORPUS_TIER1,
        )
        chunks.append(chunk)

    distinct_doc_ids = sorted(set(c.doc_id for c in chunks))
    unresolved_chunks = [c for c in chunks if c.metadata.get("unresolved_doc_id")]
    with_clause_count = sum(1 for c in chunks if c.clause_id is not None)

    stats: Dict[str, object] = {
        "chunks_built": len(chunks),
        "questions_in": len(questions),
        "probes_excluded": len(probe_ids),
        "probe_ids_excluded": probe_ids,
        "empty_passage_excluded": len(empty_passage_ids),
        "empty_passage_ids_excluded": empty_passage_ids,
        "distinct_doc_ids": distinct_doc_ids,
        "distinct_doc_ids_count": len(distinct_doc_ids),
        "unresolved_count": len(unresolved_chunks),
        "unresolved_question_ids": [c.metadata["question_id"] for c in unresolved_chunks],
        "with_clause_id_count": with_clause_count,
    }
    return chunks, stats


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build Tier 1 fixture retrieval corpus from QA CSV gold passages."
    )
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    parser.add_argument(
        "--csv",
        default=os.path.join(repo_root, "data", "gold", "bank_qa_data.csv"),
        help="Path to the gold QA CSV file",
    )
    parser.add_argument(
        "--out",
        default=os.path.join(repo_root, "data", "processed_chunks", "tier1_chunks.json"),
        help="Output JSON path for Tier 1 chunks (refuses corpus_chunks.json)",
    )
    args = parser.parse_args()

    assert_safe_output_path(args.out)

    print(f"Loading gold questions from: {args.csv}")
    questions = load_gold_questions(args.csv, repo_root=repo_root)
    print(f"Loaded {len(questions)} questions.")

    chunks, stats = build_tier1_chunks(questions)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    serialized = [asdict(c) for c in chunks]
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(serialized, f, ensure_ascii=False, indent=2)

    print("\n================ TIER 1 CORPUS BUILD SUMMARY ================")
    print(f"Questions in CSV:       {stats['questions_in']}")
    print(f"Chunks built:           {stats['chunks_built']}")
    print(f"Unanswerable probes excluded (no passage by design): {stats['probes_excluded']}")
    if stats["probe_ids_excluded"]:
        print(f"  probe ids:            {stats['probe_ids_excluded']}")
    if stats["empty_passage_excluded"]:
        print(
            f"!! ANSWERABLE rows with no gold passage excluded (DATA DEFECT, not a probe): "
            f"{stats['empty_passage_excluded']} -> {stats['empty_passage_ids_excluded']}",
            file=sys.stderr,
        )
    print(f"Distinct doc_ids ({stats['distinct_doc_ids_count']}):  {stats['distinct_doc_ids']}")
    print(f"UNRESOLVED doc_ids:     {stats['unresolved_count']} chunks: {stats['unresolved_question_ids']}")
    print(f"Chunks with clause_id:  {stats['with_clause_id_count']}")
    print(f"Corpus source:          {CORPUS_TIER1}")
    print(f"Saved output to:        {args.out}")
    print("=============================================================\n")


if __name__ == "__main__":
    main()
