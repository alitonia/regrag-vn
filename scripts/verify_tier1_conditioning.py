"""Verify round-2 red-team BLOCKER claims against the campaign logs.

BLOCKER-1: is the question's gold Tier-1 chunk actually in the retrieved top-3
(and untruncated) for RAG answerable rows?  Claims under test (laneH2):
  absent-from-top3: 93/192 BM25, 51/192 dense
  present-but-truncated: 42 BM25, 57 dense  -> intact 57/192 and 84/192
  text-level: gold passage absent from 144/384 RAG answerable prompts

BLOCKER-2: what did the Q059 (RAG-BM25) prompt actually contain, and what is
the canonical gold for Q059?

Also emits: answered-row hallucination rates (WARN-5) if per-row labels exist.
"""
from __future__ import annotations

import json
import re
import unicodedata
from collections import Counter, defaultdict


def norm(s: str) -> str:
    s = unicodedata.normalize("NFC", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip().lower()


meta = [json.loads(l) for l in open("data/eval/generations_meta.jsonl")]
t1 = json.load(open("data/processed_chunks/tier1_chunks.json"))
qc = {q["id"]: q for q in json.load(open("data/gold/questions_canonical.json"))}
gen = json.load(open("data/eval/generations.json"))
grows = {g["cache_key"]: g for g in gen} if isinstance(gen, list) else {}
print("generations.json rows:", len(grows), "keys:", sorted(next(iter(grows.values())).keys())[:20] if grows else "-")

# --- gold chunk per question (from tier1 metadata) -------------------------
qid2chunk = {}
for c in t1:
    md = c.get("metadata") or {}
    qids = md.get("question_ids") or md.get("question_id") or ([md["question_id"]] if md.get("question_id") else None)
    if qids is None:
        continue
    if isinstance(qids, str):
        qids = [qids]
    for q in qids:
        qid2chunk[q] = c["chunk_id"]
print("questions mapped to tier1 chunks:", len(qid2chunk))

ids_in_t1 = {c["chunk_id"] for c in t1}
stray = [r["retrieved_chunk_ids"] for r in meta if r.get("retrieved_chunk_ids")
         and not set(r["retrieved_chunk_ids"]) <= ids_in_t1]
print("rows with retrieved ids outside tier1 file:", len(stray))

# --- BLOCKER-1 counts -------------------------------------------------------
stats = defaultdict(Counter)
for r in meta:
    if r["retrieval_mode"] == "closed_book" or not r.get("is_answerable"):
        continue
    mode = r["retrieval_mode"]
    gold = qid2chunk.get(r["question_id"])
    ret = r.get("retrieved_chunk_ids") or []
    tr = r.get("rag_context_truncated_ranks") or []
    if gold is None or gold not in ret:
        stats[mode]["absent"] += 1
    else:
        rank = ret.index(gold) + 1
        if rank in tr:
            stats[mode]["present_truncated"] += 1
        else:
            stats[mode]["intact"] += 1
for mode, c in sorted(stats.items()):
    tot = sum(c.values())
    print(f"{mode}: absent={c['absent']} present_truncated={c['present_truncated']} "
          f"intact={c['intact']} total={tot}")

# --- text-level check --------------------------------------------------------
absent_text = Counter()
checked = Counter()
for r in meta:
    if r["retrieval_mode"] == "closed_book" or not r.get("is_answerable"):
        continue
    g = grows.get(r["cache_key"])
    if not g:
        continue
    prompt = norm(g.get("prompt") or "")
    gp = norm(qc[r["question_id"]].get("gold_passage") or "")[:80]
    checked[r["retrieval_mode"]] += 1
    if not gp or gp not in prompt:
        absent_text[r["retrieval_mode"]] += 1
print("text-level gold-passage-absent:", dict(absent_text), "of checked", dict(checked))

# --- BLOCKER-2: Q059 ---------------------------------------------------------
q59 = qc.get("Q059", {})
print("\nQ059 canonical: gold_doc_ids=", q59.get("gold_doc_ids"),
      "confidence=", q59.get("doc_id_confidence"),
      "gold_citations=", q59.get("gold_citations"))
print("Q059 gold_passage head:", norm(q59.get("gold_passage") or "")[:120])
for r in meta:
    if r["question_id"] == "Q059" and r["retrieval_mode"] == "rag_bm25":
        g = grows.get(r["cache_key"], {})
        print(f"\nQ059 {r['model_name']} retrieved:", r.get("retrieved_chunk_ids"),
              "truncated_ranks:", r.get("rag_context_truncated_ranks"))
        chunk_docs = {c["chunk_id"]: (c["doc_id"], c["article_id"], norm(c["text"])[:60])
                      for c in t1 if c["chunk_id"] in (r.get("retrieved_chunk_ids") or [])}
        for cid, (d, a, t) in chunk_docs.items():
            print("   ", cid, "->", d, "Điều", a, "|", t)
        print("    extracted_citations:", g.get("extracted_citations"))
        print("    prompt contains 'Mạng lưới hoạt động':",
              "mạng lưới hoạt động" in norm(g.get("prompt") or ""))
        print("    prompt contains 'tập quán thương mại':",
              "tập quán thương mại" in norm(g.get("prompt") or ""))

# --- Q036 annex check (WARN-9) ----------------------------------------------
q36 = qc.get("Q036", {})
print("\nQ036 gold_doc_ids:", q36.get("gold_doc_ids"), "confidence:", q36.get("doc_id_confidence"))
print("Q036 gold_passage head:", norm(q36.get("gold_passage") or "")[:100])
print("Q036 in tier1 map:", "Q036" in qid2chunk)

# --- WARN-5: answered-row hallucination rates -------------------------------
if grows:
    lbl = defaultdict(lambda: defaultdict(list))
    for r in meta:
        g = grows.get(r["cache_key"])
        if not g:
            continue
        row = r["retrieval_mode"] + "::" + r["model_name"]
        # find hallucination-ish label field robustly
        for k in ("hallucinated", "is_hallucinated", "hallucination"):
            if k in g:
                lbl[row][k].append(bool(g[k]) if not isinstance(g[k], list) else bool(g[k]))
        if "labels" in g and isinstance(g["labels"], dict):
            for k, v in g["labels"].items():
                lbl[row]["labels." + k].append(bool(v))
    printed = False
    for row, ks in sorted(lbl.items()):
        for k, vals in ks.items():
            if "halluc" in k:
                print("WARN5", row, k, "rate_all64ish:", round(sum(vals) / len(vals), 3), "n:", len(vals))
                printed = True
    if not printed:
        print("WARN5: no per-row hallucination label field found; enumerate one full row:")
        k0 = next(iter(grows))
        print(json.dumps({k: (v if not isinstance(v, str) or len(v) < 100 else v[:100] + "...") for k, v in grows[k0].items()}, ensure_ascii=False)[:1200])
