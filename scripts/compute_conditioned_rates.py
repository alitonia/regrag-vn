"""Conditioned re-analysis of the RQ2/RQ1 headline rates (round-2 red-team fix).

Splits the citation-unfaithfulness and hallucination rates by whether the
question's gold Tier-1 chunk actually reached the prompt:
  ABSENT     - gold chunk not in retrieved top-3
  TRUNCATED  - gold chunk in top-3 but its rank was clipped by the context budget
  INTACT     - gold chunk in top-3, rank not clipped

Sanity gate: overall unfaithful counts must reproduce 94 (rag_bm25) / 83
(rag_dense) from data/eval/tables_paper_2026-09-13.json before any split is
trusted.  Also emits answered-row hallucination rates and candidate
replacements for the Q059 example (gold INTACT + WRONG_INSTRUMENT + fully
correct answer text).
"""
from __future__ import annotations

import json
import os
import re
import sys
import unicodedata
from collections import Counter, defaultdict

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from regrag.corpus.qa_loader import load_gold_questions  # noqa: E402
from regrag.evaluation.metrics import score_response  # noqa: E402
from regrag.models import GenerationResult  # noqa: E402

GOLD_CSV = os.path.join(REPO_ROOT, "data", "gold", "bank_qa_data.csv")
UNFAITHFULNESS_TYPES = frozenset(
    {"WRONG_INSTRUMENT", "NON_COVERING_PROVISION", "UNCITED_ASSERTION",
     "NUMERIC_THRESHOLD_MISMATCH"})


def norm(s: str) -> str:
    s = unicodedata.normalize("NFC", s)
    return re.sub(r"\s+", " ", s).strip().lower()


golds = {q.id: q for q in load_gold_questions(GOLD_CSV)}
gens = json.load(open(os.path.join(REPO_ROOT, "data", "eval", "generations.json")))
meta = {json.loads(l)["cache_key"]: json.loads(l)
        for l in open(os.path.join(REPO_ROOT, "data", "eval", "generations_meta.jsonl"))}
t1 = json.load(open(os.path.join(REPO_ROOT, "data", "processed_chunks", "tier1_chunks.json")))
qid2chunk = {}
for c in t1:
    md = c.get("metadata") or {}
    q = md.get("question_id") or (md.get("question_ids") or [None])[0]
    if q:
        qid2chunk[q] = c["chunk_id"]

# ---- conditioning flag per RAG row ----------------------------------------
# Prompt-level semantics (fixed 2026-09-15 after the numeric-audit lane caught
# 15 rows whose gold chunk sat at rank 3 in rag_context_dropped_ranks and was
# never packed into the prompt, yet the chunk-id rule below had bucketed them
# INTACT; 6 of the 15 have no gold text in the prompt at all). Classification
# now honours what the model actually saw:
#   ABSENT    - gold chunk not retrieved in top-3, or its text head (first 40
#               normalised chars) does not appear in the final prompt
#   TRUNCATED - gold text head visible, but the gold chunk's own rank was
#               clipped by the budget, or the rank was dropped and the visible
#               copy is a duplicate passage at a clipped rank
#   INTACT    - gold text head visible at an unclipped rank
def norm_txt(s: str) -> str:
    s = unicodedata.normalize("NFC", s)
    return re.sub(r"\s+", " ", s).strip().lower()

qid2head = {}
for c in t1:
    md = c.get("metadata") or {}
    q = md.get("question_id") or (md.get("question_ids") or [None])[0]
    if q:
        qid2head[q] = norm_txt(c.get("text", ""))[:40]

cond = {}
for g in gens:
    if g["retrieval_mode"] == "closed_book":
        continue
    r = meta[g["cache_key"]]
    if not golds[g["question_id"]].is_answerable:
        continue
    gold = qid2chunk.get(g["question_id"])
    ret = r.get("retrieved_chunk_ids") or []
    tr = r.get("rag_context_truncated_ranks") or []
    dr = r.get("rag_context_dropped_ranks") or []
    head_in_prompt = qid2head.get(g["question_id"], "") in norm_txt(g["prompt"] or "")
    if gold not in ret or not head_in_prompt:
        c = "ABSENT"
    elif (ret.index(gold) + 1) in tr or (ret.index(gold) + 1) in dr:
        c = "TRUNCATED"
    else:
        c = "INTACT"
    cond[(g["model_name"], g["retrieval_mode"], g["question_id"])] = c

# ---- score every row exactly as compute_tables.py does --------------------
records = []
for g in gens:
    gen = GenerationResult(
        question_id=g["question_id"], model_name=g["model_name"],
        retrieval_mode=g["retrieval_mode"], prompt=g["prompt"],
        raw_response=g["raw_response"], answer_text=g["answer_text"],
        extracted_citations=g.get("extracted_citations", []),
        abstained=bool(g.get("abstained")),
        corpus_source=g.get("corpus_source", ""),
        retriever_backend=g.get("retriever_backend", ""),
        cache_key=g.get("cache_key", ""),
    )
    score = score_response(gen, golds[g["question_id"]])
    rec = score.to_record(corpus_source=gen.corpus_source,
                          retriever_backend=gen.retriever_backend)
    rec._types = tuple(score.hallucination_types)
    records.append(rec)

# ---- RQ2 conditioned split -------------------------------------------------
print("=== citation unfaithfulness by gold-condition (answerable rows) ===")
for mode in ("rag_bm25", "rag_dense"):
    rows = [r for r in records if r.retrieval_mode == mode and r.is_answerable]
    overall = sum(1 for r in rows if set(r._types) & UNFAITHFULNESS_TYPES)
    parts = {}
    for c in ("ABSENT", "TRUNCATED", "INTACT"):
        sub = [r for r in rows if cond.get((r.model_name, r.retrieval_mode, r.question_id)) == c]
        u = sum(1 for r in sub if set(r._types) & UNFAITHFULNESS_TYPES)
        ab = sum(1 for r in sub if r.abstained)
        parts[c] = (u, len(sub), ab)
    print(f"{mode}: overall_unfaithful={overall}/192 (gate: expect 94/83)")
    for c, (u, n, ab) in parts.items():
        print(f"    {c:9s}: {u:3d}/{n:3d} unfaithful ({u/n*100:.1f}%)  refused={ab}")

# per-question presence, two semantics: retrieval top-3 (chunk-id, matches
# the paper's Section III-B wording) and prompt-level (gold head visible)
for mode in ("rag_bm25", "rag_dense"):
    present_q, prompt_q = set(), set()
    for (m, md_, q), c in cond.items():
        if md_ != mode:
            continue
        if c in ("TRUNCATED", "INTACT"):
            prompt_q.add(q)
        g = next(x for x in gens if x["question_id"] == q
                 and x["retrieval_mode"] == mode and (m, md_, q) in cond)
        ret = meta[g["cache_key"]].get("retrieved_chunk_ids") or []
        if qid2chunk.get(q) in ret:
            present_q.add(q)
    print(f"{mode}: gold chunk in top-3 for {len(present_q)}/64 answerable questions "
          f"(gold text visible in prompt for {len(prompt_q)}/64)")

# ---- RQ1 answered-row hallucination rates ---------------------------------
print("=== hallucination: all-answerable vs answered-only (per model-mode) ===")
by_mm = defaultdict(list)
for r in records:
    by_mm[(r.model_name, r.retrieval_mode)].append(r)
for (m, mode), rows in sorted(by_mm.items()):
    ans = [r for r in rows if r.is_answerable]
    hall = [r for r in ans if r.hallucinated]
    refused = [r for r in ans if r.abstained]
    answered = [r for r in ans if not r.abstained]
    hall_a = [r for r in answered if r.hallucinated]
    if ans and answered:
        print(f"{m:10s} {mode:11s}: all={len(hall)}/64 ({len(hall)/64*100:.1f}%)  "
              f"answered-only={len(hall_a)}/{len(answered)} ({len(hall_a)/len(answered)*100:.1f}%)  "
              f"refused={len(refused)}")

# ---- Q059 replacement candidates ------------------------------------------
print("=== Q059 replacement candidates (gold INTACT, WRONG_INSTRUMENT, correct>=1.0) ===")
gens_by_key = {(g["model_name"], g["retrieval_mode"], g["question_id"]): g for g in gens}
picked = 0
for r in records:
    if picked >= 8:
        break
    if r.retrieval_mode == "closed_book" or not r.is_answerable or r.abstained:
        continue
    types = set(r._types)
    if "WRONG_INSTRUMENT" not in types or r.correctness_score < 1.0:
        continue
    if cond.get((r.model_name, r.retrieval_mode, r.question_id)) != "INTACT":
        continue
    g = gens_by_key[(r.model_name, r.retrieval_mode, r.question_id)]
    gold_cit = golds[r.question_id].gold_citations[0] if golds[r.question_id].gold_citations else None
    print(f"{r.question_id} {r.model_name} {r.retrieval_mode} types={sorted(types)}")
    print(f"   gold={gold_cit}")
    print(f"   extracted={g.get('extracted_citations')}")
    snippet = norm(g['answer_text'])[:220]
    print(f"   answer[:220]={snippet}")
    picked += 1
