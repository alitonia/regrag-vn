"""Compute the §V tables and results narrative statistics from the campaign log.

Every number that reaches paper Tables I and II is produced by this script from
data/eval/generations.json + generations_meta.jsonl (joined on cache_key);
nothing is hand-entered. Scoring reuses the repository scorer stack unchanged
(regrag.evaluation.metrics.score_response), with grounding contexts recovered
from the stored prompts exactly as evaluate_response does after the fact.

Provenance of the emitted tables, stated here because the JSON carries it too:
the 2026-09-13 campaign ran on the Tier-1 development fixture
(corpus_source=tier1_passages - the Tier-2 re-ingest was rejected on duplicate
chunk ids), and the lexical scorer is metric_status="unvalidated" until the
~50-response human-label validation runs. assert_metrics_publishable would
rightly refuse these rows, so aggregation runs with strict=False and every
emitted table is stamped with both facts; the paper's §V carries the same
disclosure. Table III (retrieval quality on Tier 2) is NOT computed here - it
comes from scripts/eval_bm25_recall.py logs, and its dense arm needs a GPU
embedding pass that must not run on this machine.

Usage:
    python3 scripts/compute_tables.py [--out data/eval/tables_paper_2026-09-13.json]
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from regrag.corpus.qa_loader import load_gold_questions  # noqa: E402
from regrag.evaluation.citation import extract_citations  # noqa: E402
from regrag.evaluation.citation import _norm_doc_for_cmp  # noqa: E402
from regrag.evaluation.metrics import parse_notes, score_response  # noqa: E402
from regrag.models import GenerationResult, GoldQuestion  # noqa: E402

GOLD_CSV = os.path.join(REPO_ROOT, "data", "gold", "bank_qa_data.csv")
GENERATIONS_JSON = os.path.join(REPO_ROOT, "data", "eval", "generations.json")
META_JSONL = os.path.join(REPO_ROOT, "data", "eval", "generations_meta.jsonl")

MODES = ("closed_book", "rag_bm25", "rag_dense")
EXPECTED_BACKENDS = {
    "closed_book": "none-closed-book",
    "rag_bm25": "bm25-rank_bm25+pyvi",
    "rag_dense": "BAAI/bge-m3",
}
EXPECTED_CORPUS_SOURCE = "tier1_passages"

#: The per-row citation-unfaithfulness types that define the RQ2 phenomenon:
#: a fluent, substantive answer whose legal basis is wrong or absent.
UNFAITHFULNESS_TYPES = frozenset(
    {"WRONG_INSTRUMENT", "NON_COVERING_PROVISION", "UNCITED_ASSERTION",
     "NUMERIC_THRESHOLD_MISMATCH"}
)


def _mean(vals):
    vals = [v for v in vals if v is not None]
    return round(sum(vals) / len(vals), 4) if vals else None


def _f1(p, r):
    if not p or not r:
        return 0.0
    return round(2 * p * r / (p + r), 4)


def load_inputs():
    with open(GENERATIONS_JSON, encoding="utf-8") as f:
        gens = json.load(f)
    meta: dict = {}
    with open(META_JSONL, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            meta[rec["cache_key"]] = rec
    return gens, meta


def check_grid(gens, golds):
    """Loud guards: 3 models x 3 modes x 88 questions, exactly once each."""
    models = sorted({g["model_name"] for g in gens})
    seen = Counter((g["model_name"], g["retrieval_mode"], g["question_id"]) for g in gens)
    problems = []
    if len(gens) != len(seen):
        problems.append(f"duplicate (model, mode, question) rows: {len(gens)} vs {len(seen)}")
    expected = {(m, mode, qid) for m in models for mode in MODES for qid in golds}
    missing = expected - set(seen)
    extra = set(seen) - expected
    if missing:
        problems.append(f"{len(missing)} grid cell(s) missing, e.g. {sorted(missing)[:3]}")
    if extra:
        problems.append(f"{len(extra)} unknown grid cell(s), e.g. {sorted(extra)[:3]}")
    bad_backend = [
        g for g in gens if g.get("retriever_backend") != EXPECTED_BACKENDS[g["retrieval_mode"]]
    ]
    if bad_backend:
        problems.append(f"{len(bad_backend)} row(s) with unexpected retriever_backend")
    bad_corpus = [g for g in gens if g.get("corpus_source") != EXPECTED_CORPUS_SOURCE]
    if bad_corpus:
        problems.append(
            f"{len(bad_corpus)} row(s) with corpus_source={bad_corpus[0].get('corpus_source')!r}"
        )
    empty = [g for g in gens if not (g.get("answer_text") or "").strip()]
    if empty:
        problems.append(f"{len(empty)} row(s) with empty answer_text")
    if problems:
        raise SystemExit("[FATAL] grid/provenance guards failed:\n  " + "\n  ".join(problems))
    return models


def clause_match(pred_citations, gold_cit):
    """Did a prediction matching the gold article also carry the gold clause?

    Mirrors compute_citation_precision_recall's two-pass article match (exact
    doc+article, then UNKNOWN-doc tolerated) and additionally requires the
    gold clause on the matched citation.
    """
    g_doc = _norm_doc_for_cmp(str(gold_cit.get("doc_id", "")))
    g_art = str(gold_cit.get("article_id", "")).strip()
    g_cl = gold_cit.get("clause_id")
    if not g_art:
        return False
    for c in pred_citations:
        p_doc = _norm_doc_for_cmp(str(c.get("doc_id", "")))
        p_art = str(c.get("article_id", "")).strip()
        doc_ok = (p_doc == g_doc) or (p_doc == "UNKNOWN")
        if doc_ok and p_art == g_art:
            if g_cl is not None:
                if str(c.get("clause_id") or "").strip() == str(g_cl).strip():
                    return True
            else:
                return True
    return False


def aggregate(records, extracted_by_key, golds):
    """All Table I / Table II / narrative statistics over one record set."""
    n = len(records)
    ans = [r for r in records if r.is_answerable]
    probes = [r for r in records if not r.is_answerable]

    grounded_vals = []
    for r in records:
        m = re.match(r"^(\d+\.\d+)\(", parse_notes(r.notes).get("grounded", ""))
        if m:
            grounded_vals.append(float(m.group(1)))

    tp = sum(1 for r in probes if r.abstained)
    fn = sum(1 for r in probes if not r.abstained)
    fp = sum(1 for r in ans if r.abstained)          # false abstention
    tn = sum(1 for r in ans if not r.abstained)
    hedged = sum(1 for r in records if "HEDGED_THEN_ANSWERED=1" in (r.notes or ""))

    cit_rows = [r for r in ans if "citation_scorable=0" not in (r.notes or "")]
    p_mean = _mean([r.citation_precision for r in cit_rows])
    r_mean = _mean([r.citation_recall for r in cit_rows])

    clause_den, clause_ok = 0, 0
    unfaithful = 0
    type_counts: Counter = Counter()
    for r in ans:
        types = set(getattr(r, "_types", ()))
        type_counts.update(types)
        if types & UNFAITHFULNESS_TYPES:
            unfaithful += 1
    for r in ans:
        gold_cits = golds[r.question_id].gold_citations
        if not gold_cits or not gold_cits[0].get("clause_id"):
            continue
        if "citation_scorable=0" in (r.notes or ""):
            continue
        clause_den += 1
        if clause_match(extracted_by_key[(r.model_name, r.retrieval_mode, r.question_id)],
                        gold_cits[0]):
            clause_ok += 1

    return {
        "rows": n,
        "answerable": len(ans),
        "probes": len(probes),
        "hallucination_rate_answerable": round(
            sum(1 for r in ans if r.hallucinated) / len(ans), 4) if ans else None,
        "groundedness_mean": _mean(grounded_vals),
        "groundedness_assessable_rows": len(grounded_vals),
        "probe_abstention_rate": round(tp / (tp + fn), 4) if (tp + fn) else None,
        "probe_abstained": tp,
        "probe_answered": fn,
        "false_abstention_rate": round(fp / len(ans), 4) if ans else None,
        "false_abstentions": fp,
        "hedged_then_answered": hedged,
        "citation_precision": p_mean,
        "citation_recall": r_mean,
        "citation_f1": _f1(p_mean, r_mean),
        "citation_scorable_rows": len(cit_rows),
        "citation_unscorable_rows": len(ans) - len(cit_rows),
        "clause_gold_rows": clause_den,
        "clause_accuracy": round(clause_ok / clause_den, 4) if clause_den else None,
        "citation_unfaithfulness_rows": unfaithful,
        "hallucination_types_answerable": dict(type_counts),
    }


def collect_examples(records, extracted_by_key, golds, gens_by_key, per_type=3):
    """Verbatim candidates for the §V citation-unfaithfulness examples."""
    out = []
    for r in records:
        if not r.is_answerable or r.abstained:
            continue
        types = set(getattr(r, "_types", ()))
        hit = types & UNFAITHFULNESS_TYPES
        if not hit or r.correctness_score < 1.0:
            continue
        g = gens_by_key[(r.model_name, r.retrieval_mode, r.question_id)]
        gold_cits = golds[r.question_id].gold_citations
        out.append({
            "types": sorted(hit),
            "question_id": r.question_id,
            "model_name": r.model_name,
            "retrieval_mode": r.retrieval_mode,
            "correctness_score": r.correctness_score,
            "gold_citation": gold_cits[0] if gold_cits else None,
            "extracted_citations": extracted_by_key[
                (r.model_name, r.retrieval_mode, r.question_id)],
            "answer_text": g["answer_text"][:900],
        })
    by_type: dict = defaultdict(list)
    for e in out:
        for t in e["types"]:
            by_type[t].append(e)
    picked, seen = [], set()
    for t in sorted(by_type):
        for e in by_type[t][:per_type]:
            k = (e["question_id"], e["model_name"], e["retrieval_mode"])
            if k not in seen:
                seen.add(k)
                picked.append(e)
    return picked


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(
        REPO_ROOT, "data", "eval", "tables_paper_2026-09-13.json"))
    args = ap.parse_args()

    golds_list = load_gold_questions(GOLD_CSV)
    golds = {q.id: q for q in golds_list}
    n_ans = sum(1 for q in golds_list if q.is_answerable)
    n_probe = sum(1 for q in golds_list if not q.is_answerable)
    n_clause = sum(1 for q in golds_list
                   if q.is_answerable and q.gold_citations
                   and q.gold_citations[0].get("clause_id"))
    print(f"[GOLD] {len(golds_list)} questions: {n_ans} answerable / {n_probe} probes / "
          f"{n_clause} with clause gold")

    gens, meta = load_inputs()
    models = check_grid(gens, golds)
    print(f"[GRID] {len(gens)} rows over models {models}")

    gens_by_key = {(g["model_name"], g["retrieval_mode"], g["question_id"]): g for g in gens}
    extracted_by_key = {
        k: extract_citations(g["answer_text"]) for k, g in gens_by_key.items()
    }

    records = []
    for i, g in enumerate(gens, 1):
        gen = GenerationResult(
            question_id=g["question_id"],
            model_name=g["model_name"],
            retrieval_mode=g["retrieval_mode"],
            prompt=g["prompt"],
            raw_response=g["raw_response"],
            answer_text=g["answer_text"],
            extracted_citations=g.get("extracted_citations", []),
            abstained=bool(g.get("abstained")),
            corpus_source=g.get("corpus_source", ""),
            retriever_backend=g.get("retriever_backend", ""),
            cache_key=g.get("cache_key", ""),
        )
        score = score_response(gen, golds[g["question_id"]])
        rec = score.to_record(corpus_source=gen.corpus_source,
                              retriever_backend=gen.retriever_backend)
        rec._types = tuple(score.hallucination_types)  # not a record field
        records.append(rec)
        if i % 200 == 0:
            print(f"[SCORE] {i}/{len(gens)}")

    # ---- Table I: per model x mode ------------------------------------------
    by_mm: dict = defaultdict(list)
    for r in records:
        by_mm[(r.model_name, r.retrieval_mode)].append(r)
    table1 = {
        f"{m}::{mode}": aggregate(by_mm[(m, mode)], extracted_by_key, golds)
        for m in models for mode in MODES
    }

    # ---- Table II: per mode, pooled over models ------------------------------
    by_mode: dict = defaultdict(list)
    for r in records:
        by_mode[r.retrieval_mode].append(r)
    table2 = {mode: aggregate(by_mode[mode], extracted_by_key, golds) for mode in MODES}

    # ---- RAG-arm equivalence on the Tier-1 fixture ---------------------------
    rag_identical = 0
    rag_total = 0
    for qid in golds:
        for m in models:
            a = gens_by_key[(m, "rag_bm25", qid)]["answer_text"]
            b = gens_by_key[(m, "rag_dense", qid)]["answer_text"]
            rag_total += 1
            if a == b:
                rag_identical += 1

    # ---- campaign metadata for the provenance / narrative block -------------
    metas = [meta.get(g.get("cache_key"), {}) for g in gens]
    lat = defaultdict(list)
    for g, mt in zip(gens, metas):
        if mt.get("latency_s") is not None:
            lat[(g["model_name"], g["retrieval_mode"])].append(mt["latency_s"])
    rag_ctx = [mt for mt in metas if "rag_context_chars" in mt]
    trunc = sum(1 for mt in rag_ctx if mt.get("rag_context_truncated_ranks"))
    drop = sum(1 for mt in rag_ctx if mt.get("rag_context_dropped_ranks"))

    hardware = {
        "gpu_names": sorted({mt.get("gpu_names", ["?"])[0] for mt in metas if mt}),
        "generation_backend": sorted({mt.get("generation_backend", "?") for mt in metas if mt}),
        "quant_config": sorted({mt.get("quant_config", "?") for mt in metas if mt}),
        "hf_model_ids": sorted({mt.get("hf_model_id", "?") for mt in metas if mt}),
        "finish_reasons": dict(Counter(mt.get("finish_reason", "?") for mt in metas if mt)),
        "prompt_exact_true": sum(1 for mt in metas if mt.get("prompt_exact")),
        "corpus_paths": sorted({mt.get("corpus_path", "?") for mt in metas if mt}),
        "generated_at": [min(mt.get("generated_at", "") for mt in metas if mt),
                         max(mt.get("generated_at", "") for mt in metas if mt)],
        "latency_s_mean_by_model_mode": {
            f"{m}::{mode}": _mean(v) for (m, mode), v in sorted(lat.items())
        },
        "rag_context": {
            "rows_stamped": len(rag_ctx),
            "truncated_rows": trunc,
            "dropped_rows": drop,
            "mean_chars": _mean([mt["rag_context_chars"] for mt in rag_ctx]),
            "max_chars": max(mt["rag_context_chars"] for mt in rag_ctx),
        },
    }

    report = {
        "scorer": "regrag-lexical-v1",
        "metric_status": "unvalidated (aggregated strict=False)",
        "corpus_source": EXPECTED_CORPUS_SOURCE,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "gold": {"questions": len(golds_list), "answerable": n_ans,
                 "probes": n_probe, "clause_gold": n_clause},
        "models": models,
        "table1_by_model_mode": table1,
        "table2_by_mode_pooled": table2,
        "tier1_rag_arm_identical_answers": {"identical": rag_identical, "of": rag_total},
        "campaign_provenance": hardware,
        "narrative_examples": collect_examples(records, extracted_by_key, golds, gens_by_key),
    }

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"[OUT] wrote {args.out}")

    # ---- console digest ------------------------------------------------------
    print("\n=== Table I (per model x mode) ===")
    hdr = f"{'model':<12}{'mode':<13}{'halluc':>8}{'ground':>8}{'abst':>7}{'false-abst':>11}"
    print(hdr)
    for m in models:
        for mode in MODES:
            s = table1[f"{m}::{mode}"]
            print(f"{m:<12}{mode:<13}"
                  f"{s['hallucination_rate_answerable']!s:>8}"
                  f"{s['groundedness_mean']!s:>8}"
                  f"{s['probe_abstention_rate']!s:>7}"
                  f"{s['false_abstention_rate']!s:>11}")
    print("\n=== Table II (per mode, pooled over models) ===")
    print(f"{'mode':<13}{'P':>7}{'R':>7}{'F1':>7}{'Khoản acc':>11}{'n(scored)':>11}")
    for mode in MODES:
        s = table2[mode]
        print(f"{mode:<13}{s['citation_precision']!s:>7}{s['citation_recall']!s:>7}"
              f"{s['citation_f1']!s:>7}{s['clause_accuracy']!s:>11}"
              f"{s['citation_scorable_rows']!s:>11}")
    print(f"\n[TIER1] rag_bm25 vs rag_dense byte-identical answers: "
          f"{rag_identical}/{rag_total}")
    print(f"[CTX] rag rows stamped {len(rag_ctx)}, truncated {trunc}, dropped {drop}, "
          f"mean {hardware['rag_context']['mean_chars']} chars")
    print(f"[HW] {hardware['gpu_names']} {hardware['generation_backend']} "
          f"{hardware['quant_config']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
