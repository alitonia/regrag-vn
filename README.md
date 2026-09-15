# RegRAG-VN

Measuring hallucination, citation accuracy and abstention behaviour of small open language models (≤7B) on **Vietnamese banking and payment regulation** under retrieval-augmented generation.

A benchmark plus a modular evaluation framework. The corpus is *QA-driven*: the legal instruments ingested are exactly those the benchmark questions cite, so ground truth is verifiably present in the retrieval corpus rather than assumed.

**Target venue:** IEEE-RIVF 2026, Hanoi — paper deadline **2026-09-15**. See [`docs/PROJECT_PLAN.md`](docs/PROJECT_PLAN.md) and [`docs/KANBAN.md`](docs/KANBAN.md).

## Authors

- Đinh Thị Lan Hương (20261261M)
- Vũ Đức Thành (20261082M)
- Nguyễn Khắc Duy Ngọc (20261206M)
- Nguyễn Huy Hoàng (20251325M)
- Nguyễn Thắng Phúc (20252263M)

## Research questions

- **RQ1 — Hallucination reduction.** How much does RAG reduce hallucination versus closed-book answering for small LLMs on Vietnamese regulatory questions?
- **RQ2 — Citation and abstention.** How accurately do models cite the supporting article (*Điều*) and clause (*Khoản*), and do they abstain when the corpus does not contain the answer?
- **RQ3 — Sparse versus dense.** How does BM25 with Vietnamese compound-word segmentation compare with multilingual dense embeddings (`BGE-M3`) on legal retrieval?

## Quickstart

```bash
# 1. dependencies (retrieval + corpus extraction)
python3 -m venv --system-site-packages .venv
.venv/bin/pip install -r requirements.txt

# 2. canonical load from the trusted QA CSV, with a coverage report
python3 scripts/load_qa_csv.py --report

# 3. Tier 1 development corpus (64 verbatim gold passages)
python3 scripts/build_tier1_corpus.py

# 4. real corpus ingestion once documents land in data/raw_legal/
python3 scripts/build_corpus_chunks.py --dry-run     # shows exactly what is missing

# 5. regenerate every CSV-derived artifact (idempotent)
python3 scripts/regenerate.py

# tests — stdlib unittest; pytest is not installed and discover fails (no tests/__init__.py)
python3 -m unittest tests.test_storage tests.test_components tests.test_provenance \
                    tests.test_coverage tests.test_parser_passages
```

## Data flow

```
data/gold/bank_qa_data.csv          ← the ONLY trusted artifact (human-authored)
        │  scripts/load_qa_csv.py  ·  regrag/corpus/qa_loader.py
        ▼
data/gold/questions_canonical.json   canonical doc_id + Điều-level gold citation
        │
        ├── scripts/build_tier1_corpus.py → tier1_chunks.json   (fixture, never publishable)
        │
data/raw_legal/*.{txt,pdf,html}      ← manually downloaded instruments
        │  scripts/build_corpus_chunks.py  ·  regrag/corpus/parser.py
        ▼
data/processed_chunks/corpus_chunks.json   (Tier 2 — the reported corpus)
        │  regrag/indexing/{bm25,dense}.py → regrag/retrieval/retriever.py
        ▼
regrag/generation → regrag/evaluation → regrag/storage
```

`data/raw_legal/INGEST_PLAN.json` is the priority-ordered download list, generated from what the questions actually cite. `data/raw_legal/DOC_MANIFEST.json` holds the human-verified URL→instrument-id map, the unreachable sources with working replacements, and the amendment decisions.

## Layout

| Path | Contents |
|---|---|
| `regrag/corpus/` | `canonical.py` (URL→`doc_id`, passage→`Điều`/`Khoản`, text normalisation), `qa_loader.py` (the only sanctioned CSV reader), `parser.py` (Điều/Khoản segmenter for Thông tư, Nghị định and Luật) |
| `regrag/indexing/` | `bm25.py` (sparse, `pyvi` segmentation), `dense.py` (`BGE-M3` embeddings) |
| `regrag/retrieval/` | top-*k* coordinator over closed-book / BM25 / dense |
| `regrag/generation/` | 4-bit inference prompting, abstention instruction |
| `regrag/evaluation/` | `citation.py` (extraction + precision/recall), `agreement.py` (Cohen's κ), `metrics.py` (scorers — currently tagged placeholder) |
| `regrag/storage/` | two persistence layers over one interface: `in_memory.py`, `file_repo.py` |
| `regrag/provenance.py` | corpus and retriever provenance tags, publishability guard |
| `scripts/` | load, build, ingest, regenerate |
| `data/` | `gold/` (trusted CSV + derived canonical JSON), `raw_legal/`, `processed_chunks/` |

## Provenance discipline

Every chunk, retrieved result, generation and evaluation record carries a provenance tag — `corpus_source` (`tier1_passages` / `tier2_full`) and `retriever_backend` (e.g. `bm25-rank_bm25+pyvi`, `BAAI/bge-m3`, or `DEGRADED:<reason>`). `provenance.assert_publishable()` **raises** rather than aggregate any row that is `UNSET`, `DEGRADED`, or Tier 1.

This exists because three code paths were found that silently substituted a degraded implementation and returned plausible-looking numbers — most seriously a dense-retrieval mock that returned corpus-order chunks with fabricated scores whenever `sentence_transformers` was missing, which would have produced an entirely fictional RAG-Dense column and invalidated RQ3 without any warning.

**Rule for contributors: a placeholder must fail loudly or self-identify. Never return a plausible default.**

Two further guarantees follow from it:

- **Coverage invariant.** Every retained question's verbatim gold passage must string-match a chunk in the Tier 2 corpus (`tests/test_coverage.py`). A passage that matches only after diacritic folding is reported as extraction damage, never silently accepted.
- **Unresolved is never dropped.** A question whose instrument id cannot be resolved is kept and reported, so the benchmark cannot silently shrink.

## Current status

Verified state as of 2026-09-09 — 33 tests passing, data layer and provenance complete, **0 of 10 real instruments ingested** (downloads in progress), coverage 0/64 against Tier 2 and 64/64 against Tier 1, no generation results, no manuscript. Full detail and the day-by-day schedule are in [`docs/KANBAN.md`](docs/KANBAN.md).

Note that `data/raw_legal/18_2024_TT_NHNN.txt` is quarantined: it is not an authentic legal text (43 lines, non-contiguous article numbers, none of the structural elements every Vietnamese circular carries) and its chunks were the entire previous corpus. See `INGEST_PLAN.json`.

## vLLM model serving

The benchmark generation models can be exposed through vLLM's
OpenAI-compatible API. Install vLLM separately in a CUDA environment, then
start one model (recommended for a single GPU):

```bash
python scripts/load_vllm_models.py --list
python scripts/load_vllm_models.py --model qwen-7b --gpu-devices 0
```

To keep all five configured models loaded, assign one CUDA device to each
server. They listen on consecutive ports starting at 8000:

```bash
python scripts/load_vllm_models.py --all --gpu-devices 0,1,2,3,4
```

Additional vLLM arguments may be placed after `--`, for example
`-- --max-model-len 4096 --gpu-memory-utilization 0.85`.
