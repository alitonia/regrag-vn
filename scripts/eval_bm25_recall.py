"""Gate C: BM25 and Dense (BGE-M3) retrieval recall on the answerable benchmark questions.

WHY: Table III of the IEEE-RIVF 2026 paper reports the retrieval quality on the
Tier-2 corpus (doc-level and article-level Recall@1 / Recall@3) that the RAG
pipeline builds on. These numbers are quoted verbatim in the paper, so this
script refuses to produce anything that could silently poison a table:

  * sparse arm runs on the project's own retrieval path (regrag.indexing.bm25.
    BM25Index + pyvi tokenize_vietnamese) - no parallel tokenizer, no second
    BM25 implementation;
  * dense arm runs on regrag.indexing.dense.DenseIndex with BAAI/bge-m3;
  * a DEGRADED backend (missing pyvi / rank_bm25 / sentence-transformers) is a
    hard error, per regrag/provenance.py - fallback scores must never reach a
    paper table;
  * any chunk not stamped corpus_source="tier2_full" is a hard error (Tier-1
    fixtures are development-only and must never reach a paper table);
  * zero chunks in the corpus, or zero retrieved chunks for any question, is
    a hard error, not a zero to average in.

Scoring population: the 64 answerable questions (is_answerable=True with
non-empty gold_doc_ids) from data/gold/questions_canonical.json. The 24
unanswerable abstention probes are EXCLUDED - they have no gold document by
design, so retrieval recall is undefined for them.

Article-level recall uses gold_citations[0]["article_id"]. Exactly one
answerable question (Q036, a Phụ lục annex locator) has no article-level gold;
it is excluded from the article-level denominator only and is named in the
output. The script verifies this expectation structurally (it lists whatever
it excludes) rather than hardcoding the id.

Usage:
  .venv/bin/python scripts/eval_bm25_recall.py
  .venv/bin/python scripts/eval_bm25_recall.py --retriever dense
  .venv/bin/python scripts/eval_bm25_recall.py --retriever dense --json data/eval/dense_recall_gate_c_YYYY-MM-DD.json
"""

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence, Tuple

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from regrag.models import GoldQuestion, LegalChunk
from regrag.provenance import (
    CORPUS_TIER2,
    ProvenanceError,
    is_degraded,
    publishable_corpus,
    require,
)

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
GOLD_PATH = os.path.join(REPO_ROOT, "data", "gold", "questions_canonical.json")
CORPUS_PATH = os.path.join(REPO_ROOT, "data", "processed_chunks", "corpus_chunks.json")
JSON_PATH = os.path.join(REPO_ROOT, "data", "eval", "bm25_recall_gate_c_2026-09-11.json")
TOP_K = 3

DEFAULT_JSON_NAME = "bm25_recall_gate_c_2026-09-11.json"


def load_corpus(path: str) -> List[LegalChunk]:
    """Load corpus_chunks.json into LegalChunk objects, refusing bad state."""
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    require(bool(raw), f"corpus at {path} has zero chunks - refusing to score")

    bad_sources = sorted({c.get("corpus_source", "<missing>") for c in raw
                          if not publishable_corpus(c.get("corpus_source", ""))})
    require(
        not bad_sources,
        "corpus contains non-tier2 chunks (corpus_source="
        f"{bad_sources}); Tier-1 fixtures must never reach a paper table",
    )
    require(
        len(raw) >= TOP_K,
        f"corpus has {len(raw)} chunks, fewer than top_k={TOP_K}",
    )
    chunk_fields = LegalChunk.__dataclass_fields__
    chunks = [
        LegalChunk(**{k: v for k, v in c.items() if k in chunk_fields})
        for c in raw
    ]
    return chunks


def load_questions(path: str) -> Tuple[List[GoldQuestion], List[GoldQuestion], List[GoldQuestion]]:
    """Load canonical gold questions; split answerable vs abstention probes."""
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    require(bool(raw), f"gold file at {path} is empty - refusing to score")

    questions = [
        GoldQuestion(**{k: v for k, v in item.items()
                        if k in GoldQuestion.__annotations__})
        for item in raw
    ]
    answerable = [q for q in questions if q.is_answerable and q.gold_doc_ids]
    probes = [q for q in questions if not (q.is_answerable and q.gold_doc_ids)]
    require(
        len(answerable) + len(probes) == len(questions),
        "question partition is inconsistent - every row must be one or the other",
    )
    return questions, answerable, probes


def score_retrieval(
    index: Any,
    answerable: Sequence[GoldQuestion],
    top_k: int = 3,
    ks: Sequence[int] = (1, 3),
) -> Tuple[Dict[int, int], Dict[int, int], List[Dict[str, Any]], List[str]]:
    """Score retrieval results against gold citations for answerable questions.

    Shared scoring logic for both BM25 and Dense retrieval arms.
    Returns:
        (doc_hits, art_hits, doc_misses_at_k, article_miss_ids_at_k)
    """
    doc_hits = {k: 0 for k in ks}
    art_hits = {k: 0 for k in ks}
    doc_misses_at_k: List[Dict[str, Any]] = []
    article_miss_ids_at_k: List[str] = []

    for q in answerable:
        results = index.search(q.question, top_k=top_k)
        require(
            bool(results),
            f"question {q.id} retrieved zero chunks - refusing to record a zero",
        )
        require(
            len(results) == top_k,
            f"question {q.id} retrieved {len(results)} chunks, expected {top_k}",
        )

        gold_docs = set(q.gold_doc_ids)
        gold_article = (q.gold_citations[0].get("article_id")
                        if q.gold_citations else None)

        for k in ks:
            top = results[:k]
            if any(r.chunk.doc_id in gold_docs for r in top):
                doc_hits[k] += 1
            if gold_article is not None and any(
                r.chunk.doc_id in gold_docs and r.chunk.article_id == gold_article
                for r in top
            ):
                art_hits[k] += 1

        hit_at_k = any(r.chunk.doc_id in gold_docs for r in results)
        if not hit_at_k:
            doc_misses_at_k.append({
                "question_id": q.id,
                "question": q.question,
                "gold_doc_ids": list(q.gold_doc_ids),
                "gold_article_id": gold_article,
                "retrieved_top3": [
                    {
                        "rank": r.rank,
                        "doc_id": r.chunk.doc_id,
                        "article_id": r.chunk.article_id,
                        "doc_title": r.chunk.doc_title,
                        "article_title": r.chunk.article_title,
                        "score": round(r.score, 4),
                    }
                    for r in results
                ],
            })
        if gold_article is not None and not any(
            r.chunk.doc_id in gold_docs and r.chunk.article_id == gold_article
            for r in results
        ):
            article_miss_ids_at_k.append(q.id)

    return doc_hits, art_hits, doc_misses_at_k, article_miss_ids_at_k


def build_recall_log(
    retriever_name: str,
    backend: str,
    corpus_path: str,
    gold_path: str,
    chunks: Sequence[LegalChunk],
    questions: Sequence[GoldQuestion],
    answerable: Sequence[GoldQuestion],
    probes: Sequence[GoldQuestion],
    article_scored: Sequence[GoldQuestion],
    article_excluded: Sequence[str],
    doc_hits: Dict[int, int],
    art_hits: Dict[int, int],
    doc_misses_at_k: List[Dict[str, Any]],
    article_miss_ids_at_k: List[str],
    top_k: int = 3,
    model_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Construct standardized evaluation JSON log dictionary."""
    log: Dict[str, Any] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "gate": "C",
        "metric": "dense_recall" if retriever_name == "dense" else "bm25_recall",
        "retriever_backend": backend,
        "corpus_source": CORPUS_TIER2,
    }
    if retriever_name == "dense":
        log["model_id"] = model_id or backend
    log.update({
        "corpus_path": corpus_path,
        "corpus_chunk_count": len(chunks),
        "instrument_count": len({c.doc_id for c in chunks}),
        "gold_path": gold_path,
        "n_canonical_questions": len(questions),
        "n_answerable_scored": len(answerable),
        "n_probes_excluded": len(probes),
        "article_level": {
            "n_scored": len(article_scored),
            "excluded_ids": list(article_excluded),
        },
        "doc_level": {
            "hits@1": doc_hits[1],
            "hits@3": doc_hits[top_k],
            "n": len(answerable),
            "recall@1": doc_hits[1] / len(answerable) if len(answerable) else 0.0,
            f"recall@{top_k}": doc_hits[top_k] / len(answerable) if len(answerable) else 0.0,
        },
        "article_level_hits": {
            "hits@1": art_hits[1],
            "hits@3": art_hits[top_k],
            "recall@1": art_hits[1] / len(article_scored) if len(article_scored) else 0.0,
            f"recall@{top_k}": art_hits[top_k] / len(article_scored) if len(article_scored) else 0.0,
        },
        "doc_misses_at_k3": doc_misses_at_k,
        "article_miss_ids_at_k3": article_miss_ids_at_k,
    })
    return log


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Gate C BM25 and Dense Recall@1/@3 evaluation")
    ap.add_argument(
        "--retriever",
        choices=["bm25", "dense"],
        default="bm25",
        help="retriever backend to evaluate (default: bm25)",
    )
    ap.add_argument("--gold", default=GOLD_PATH)
    ap.add_argument("--corpus", default=CORPUS_PATH)
    ap.add_argument(
        "--json",
        default=None,
        help="dated JSON log (default: data/eval/<retriever>_recall_gate_c_<date>.json)",
    )
    ap.add_argument("--top-k", type=int, default=TOP_K)
    ap.add_argument(
        "--model",
        default="BAAI/bge-m3",
        help="model name for dense retriever (default: BAAI/bge-m3)",
    )
    ap.add_argument(
        "--device",
        default=None,
        help="device for dense embedding (e.g. cuda, cpu; default: auto-detect)",
    )
    args = ap.parse_args(argv)
    top_k = args.top_k
    ks = sorted({1, top_k})

    date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    if args.json:
        json_path = args.json
    elif args.retriever == "dense":
        json_path = os.path.join(REPO_ROOT, "data", "eval", f"dense_recall_gate_c_{date_str}.json")
    else:
        json_path = os.path.join(REPO_ROOT, "data", "eval", DEFAULT_JSON_NAME)

    chunks = load_corpus(args.corpus)
    questions, answerable, probes = load_questions(args.gold)

    # The one structural caveat: questions whose gold citation has no article_id
    # cannot be scored at article level. Excluded loudly, never silently.
    article_excluded = [
        q.id for q in answerable
        if not (q.gold_citations and q.gold_citations[0].get("article_id"))
    ]
    article_scored = [q for q in answerable if q.id not in set(article_excluded)]

    print(f"[corpus] {args.corpus}")
    print(f"[corpus] {len(chunks)} chunks, "
          f"{len({c.doc_id for c in chunks})} instruments, "
          f"corpus_source={CORPUS_TIER2}")
    print(f"[gold]   {len(questions)} canonical questions: "
          f"{len(answerable)} answerable scored, {len(probes)} unanswerable "
          "probes EXCLUDED (no gold document by design)")
    if article_excluded:
        print(f"[gold]   article-level denominator {len(article_scored)}: excluded "
              f"{article_excluded} (no article_id in gold_citations - Phụ lục "
              "locator)")

    if args.retriever == "bm25":
        from regrag.indexing.bm25 import BM25Index

        index = BM25Index(chunks)
        require(
            not is_degraded(index.backend),
            f"BM25 backend is DEGRADED ({index.backend}) - results are not real "
            "BM25 scores and cannot be used for paper reporting",
        )
        print(f"[index]  backend: {index.backend}")
    elif args.retriever == "dense":
        from regrag.indexing.dense import DenseIndex

        print(f"[index]  building DenseIndex ({args.model})...")
        index = DenseIndex(chunks, model_name=args.model, device=args.device)
        index.build()
        require(
            not is_degraded(index.backend),
            f"Dense backend is DEGRADED ({index.backend}) - results are not real "
            "dense scores and cannot be used for paper reporting",
        )
        print(f"[index]  backend: {index.backend}")
    else:
        raise ValueError(f"Unknown retriever: {args.retriever}")

    doc_hits, art_hits, doc_misses_at_k, article_miss_ids_at_k = score_retrieval(
        index=index,
        answerable=answerable,
        top_k=top_k,
        ks=ks,
    )

    def recall_str(hits: int, n: int) -> str:
        return f"{hits / n:.4f}" if n else "n/a"

    retriever_label = "BM25" if args.retriever == "bm25" else f"Dense ({index.backend})"
    print()
    print(f"=== Gate C: {retriever_label} retrieval recall (Tier-2 corpus) ===")
    header = f"{'level':<15}{'Recall@1':>10}{'Recall@3':>10}{'n':>6}"
    print(header)
    print("-" * len(header))
    print(f"{'doc-level':<15}{recall_str(doc_hits[1], len(answerable)):>10}"
          f"{recall_str(doc_hits[top_k], len(answerable)):>10}{len(answerable):>6}")
    print(f"{'article-level':<15}{recall_str(art_hits[1], len(article_scored)):>10}"
          f"{recall_str(art_hits[top_k], len(article_scored)):>10}"
          f"{len(article_scored):>6}")
    print()
    print(f"doc-level misses @k={top_k} "
          f"({len(doc_misses_at_k)}): {[m['question_id'] for m in doc_misses_at_k]}")

    model_id = getattr(index, "model_name", args.model) if args.retriever == "dense" else None
    log = build_recall_log(
        retriever_name=args.retriever,
        backend=index.backend,
        corpus_path=args.corpus,
        gold_path=args.gold,
        chunks=chunks,
        questions=questions,
        answerable=answerable,
        probes=probes,
        article_scored=article_scored,
        article_excluded=article_excluded,
        doc_hits=doc_hits,
        art_hits=art_hits,
        doc_misses_at_k=doc_misses_at_k,
        article_miss_ids_at_k=article_miss_ids_at_k,
        top_k=top_k,
        model_id=model_id,
    )

    os.makedirs(os.path.dirname(os.path.abspath(json_path)), exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(log, f, ensure_ascii=False, indent=2)
    print(f"\n[log]    wrote {json_path}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except ProvenanceError as exc:
        sys.stderr.write(f"[FATAL] {exc}\n")
        sys.exit(2)
