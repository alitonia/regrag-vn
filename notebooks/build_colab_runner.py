#!/usr/bin/env python3
"""Build notebooks/colab_runner.ipynb.

The notebook is a generated artifact: its cells are long, contain Vietnamese text
and shell magic, and hand-editing JSON is how escaping bugs get shipped. Keeping
the generator here means the notebook can be rebuilt deterministically and
reviewed as Python.

    .venv/bin/python notebooks/build_colab_runner.py

Cell order follows the `colab` skill skeleton (setup -> Drive mount + RESTORE ->
data -> config -> run -> manual SAVE -> results + fetch) merged with the order
the harness brief requires (GPU assertion first, sanity probe gated before the
campaign, checkpointed campaign, save-to-Drive and copy-back last).
"""

from __future__ import annotations

import json
import os
import sys

CELLS = []


def md(text: str) -> None:
    CELLS.append({"cell_type": "markdown", "metadata": {}, "source": _split(text)})


def code(text: str) -> None:
    CELLS.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": _split(text),
    })


def _split(text: str):
    """ipynb stores source as a list of lines, each keeping its newline."""
    text = text.strip("\n")
    lines = text.split("\n")
    return [ln + "\n" for ln in lines[:-1]] + ([lines[-1]] if lines[-1] else [])


# --------------------------------------------------------------------------- 1
md("""
# RegRAG-VN — Colab Inference Campaign

Runs the **3 models x 3 retrieval modes x 88 questions** grid from
`docs/PROJECT_PLAN.md` §3.3 and writes provenance-tagged `GenerationResult` rows
that `regrag/storage/` and `regrag/evaluation/` consume directly.

**Read this before running anything.**

| | |
|---|---|
| Runtime | GPU required. T4 (16 GB) is the target; A100/L4 also work. |
| Grid | 3 models x 3 modes x 88 questions = **792 generations** |
| Resume | Every row is checkpointed as it lands. A disconnect costs at most the rows since the last Drive sync. |
| Re-run | Idempotent. Re-running a finished campaign generates **zero** rows. |

### The two hard gates

1. **Sanity-probe gate (cell 9-11).** A wrong chat template produces degenerate
   output that reads as *"this model hallucinates"* when the truth is *"we prompted
   it wrong"*. The probe must be run for every model and **explicitly
   human-approved** before the campaign cell will execute.
2. **Provenance gate.** No row is written without `corpus_source`,
   `retriever_backend`, `generation_backend` and `quant_config`. A degraded
   dependency, a quantization load that silently fell back, or a dead server
   **stops the run** instead of producing plausible numbers.

### Order of cells

`1 GPU assertion` -> `2 repo` -> `3 deps` -> `4 imports` -> `5 CONFIG` ->
`6 Drive + RESTORE` -> `7 questions + corpus` -> `8 indices` ->
`9 backend` -> `10 SANITY PROBE` -> `11 probe gate` -> `12 CAMPAIGN` ->
`13 manual SAVE` -> `14 audit` -> `15 copy back to repo`
""")

# --------------------------------------------------------------------------- 2
md("""
## 1. GPU / environment assertion

Fails loudly and immediately if there is no GPU or it is too small. This runs
before any `pip install` so a wrong runtime choice costs seconds, not minutes.
""")

code('''
# [CELL 1] GPU / environment assertion — FAILS LOUDLY, no silent CPU fallback.
import shutil, subprocess, sys

print("=" * 72)
print("RUNTIME ASSERTION")
print("=" * 72)

if shutil.which("nvidia-smi"):
    print(subprocess.run(["nvidia-smi"], capture_output=True, text=True).stdout.strip())
else:
    print("nvidia-smi NOT FOUND — this runtime has no NVIDIA driver.")

try:
    import torch
except ImportError as exc:
    raise SystemExit(
        "[FATAL] torch is not importable. On Colab use "
        "Runtime -> Change runtime type -> Hardware accelerator -> T4 GPU. "
        f"Original error: {exc}"
    )

if not torch.cuda.is_available():
    raise SystemExit(
        "[FATAL] torch.cuda.is_available() is False. This runtime has NO GPU.\\n"
        "The campaign will not run on CPU: a 7B model on CPU takes minutes per\\n"
        "question and produces rows whose latency and batching differ from every\\n"
        "other row. Fix the runtime (Runtime -> Change runtime type -> T4 GPU) and\\n"
        "re-run from cell 1."
    )

N = torch.cuda.device_count()
NAMES, MEM_GB = [], []
for i in range(N):
    p = torch.cuda.get_device_properties(i)
    NAMES.append(p.name)
    MEM_GB.append(round(p.total_memory / 1024**3, 2))

print(f"\\ntorch          : {torch.__version__}")
print(f"CUDA available : True")
print(f"device count   : {N}")
print(f"device names   : {NAMES}")
print(f"device memory  : {MEM_GB} GB")
print(f"CUDA version   : {torch.version.cuda}")

SMALLEST = min(MEM_GB)
if SMALLEST < 14.0:
    print(
        f"\\n[WARNING] Smallest GPU has {SMALLEST} GB. A 7B model in 4-bit needs\\n"
        "roughly 5-6 GB of weights plus KV cache. Expect OOM at\\n"
        "max_new_tokens=512; consider the 3B/4B models only, or an A100 runtime."
    )

# Record it now so every later cell can stamp the GPU on its rows.
GPU_INFO = {
    "cuda_available": True,
    "device_count": N,
    "device_names": NAMES,
    "device_memory_gb": MEM_GB,
    "cuda_version": torch.version.cuda,
    "torch_version": torch.__version__,
    "gpu_backend_tag": "cuda:" + ";".join(NAMES),
}
print("\\n[OK] GPU assertion passed.")
''')

# --------------------------------------------------------------------------- 3
md("""
## 2. Get the repository onto the runtime

`/content` is ephemeral, so the repo is fetched fresh every session. Three
sources are tried in order, because the campaign code and the 88-row CSV may not
be pushed to GitHub yet:

1. an existing `/content/rag_eval` (re-running a cell),
2. a copy synced to Google Drive (`<DRIVE>/repo/rag_eval`) — **use this if the
   harness modules are not committed yet**,
3. `git clone` / `git pull` from GitHub.

Then it **verifies** the harness modules and the data are actually present and
fails loudly with the exact remedy if they are not. A stale clone that silently
lacks `regrag/generation/backends.py` or still has a 64-row CSV would otherwise
produce a plausible-looking partial campaign.
""")

code('''
# [CELL 2] Repository acquisition + verification. Fails loud on a stale tree.
import os, subprocess, sys

REPO_ROOT   = "/content/rag_eval"
REPO_GITHUB = "https://github.com/alitonia/rag_eval.git"
# Set this in cell 5 if you sync the working tree to Drive instead of committing.
DRIVE_REPO_COPY = "/content/drive/MyDrive/regrag_vn_checkpoints/repo/rag_eval"

def _run(cmd, cwd=None):
    r = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"[git] failed: {' '.join(cmd)}\\n{r.stdout}\\n{r.stderr}")
    else:
        print(f"[git] ok: {' '.join(cmd)}")
    return r.returncode == 0

if os.path.isdir(os.path.join(REPO_ROOT, ".git")):
    print(f"[REPO] Using existing checkout at {REPO_ROOT}")
    _run(["git", "pull", "--ff-only"], cwd=REPO_ROOT)
elif os.path.isdir(os.path.join(DRIVE_REPO_COPY, "regrag")):
    import shutil
    print(f"[REPO] Copying from Drive: {DRIVE_REPO_COPY} -> {REPO_ROOT}")
    if os.path.isdir(REPO_ROOT):
        shutil.rmtree(REPO_ROOT)
    shutil.copytree(DRIVE_REPO_COPY, REPO_ROOT,
                    ignore=shutil.ignore_patterns(".git", ".venv", "__pycache__"))
elif os.path.isdir(REPO_ROOT):
    print(f"[REPO] Using non-git directory at {REPO_ROOT}")
else:
    print(f"[REPO] Cloning {REPO_GITHUB}")
    if not _run(["git", "clone", REPO_GITHUB, REPO_ROOT]):
        raise SystemExit(
            "[FATAL] Could not obtain the repository. If it is private or the "
            "harness is not committed yet, sync your working tree to "
            f"{DRIVE_REPO_COPY} and re-run this cell."
        )

if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
os.chdir(REPO_ROOT)
print(f"[REPO] cwd = {os.getcwd()}")

# --- verify the tree is the one this notebook needs --------------------------
REQUIRED_MODULES = [
    "regrag/generation/backends.py",
    "regrag/generation/campaign.py",
    "regrag/generation/checkpoint.py",
    "regrag/generation/sanity.py",
    "regrag/generation/abstention.py",
    "regrag/generation/benchmark.py",
    "regrag/generation/corpus_io.py",
    "regrag/generation/config.py",
    "regrag/provenance.py",
    "scripts/regenerate.py",
]
missing = [m for m in REQUIRED_MODULES if not os.path.exists(os.path.join(REPO_ROOT, m))]
if missing:
    raise SystemExit(
        "[FATAL] The checkout is missing harness module(s):\\n  "
        + "\\n  ".join(missing)
        + "\\nThis is almost certainly a stale clone: the generation harness is newer "
          "than the last push. Commit it, or sync the working tree to Drive at "
          f"{DRIVE_REPO_COPY} and re-run this cell."
    )

import csv as _csv
_csv_path = os.path.join(REPO_ROOT, "data/gold/bank_qa_data.csv")
with open(_csv_path, encoding="utf-8-sig", newline="") as f:
    _rows = [r for r in _csv.DictReader(f) if (r.get("question") or "").strip()]
print(f"[REPO] gold CSV rows with a question: {len(_rows)}")
if len(_rows) != 88:
    raise SystemExit(
        f"[FATAL] Expected 88 rows in the gold CSV (64 answerable + 24 unanswerable "
        f"probes), found {len(_rows)}. The 24 probes carry RQ2 (abstention); without "
        "them the campaign measures only RQ1/RQ3. The checkout is stale or the CSV "
        "was not pushed."
    )
print("[OK] Repository verified.")
''')

# --------------------------------------------------------------------------- 4
md("""
## 3. Dependencies

Everything is installed here and nowhere else. `vllm` is **optional** and off by
default — see cell 9.
""")

code('''
# [CELL 3] Dependency install. Nothing is installed later in this notebook.
# torch is preinstalled on Colab; do not reinstall it or the CUDA build breaks.
!pip install -q -U transformers accelerate bitsandbytes
!pip install -q sentence-transformers rank_bm25 pyvi
# requests is already present on Colab and is all the vLLM HTTP client needs.

# OPTIONAL — only if you will serve models through vLLM (cell 9).
# A plain T4 can serve ONE 7B model at a time; the default path below is
# transformers + 4-bit, which needs no server.
# !pip install -q vllm

print("[OK] Dependencies installed.")
''')

code('''
# [CELL 4] Imports, version report, and an honest dependency audit.
import importlib.util, json, os, sys, time
from datetime import datetime, timezone

import transformers
print(f"transformers        : {transformers.__version__}")
print(f"torch               : {__import__('torch').__version__}")

DEPS = ["transformers", "accelerate", "bitsandbytes", "sentence_transformers",
        "rank_bm25", "pyvi", "requests", "vllm"]
PRESENT = {d: importlib.util.find_spec(d) is not None for d in DEPS}
for d in DEPS:
    print(f"  {d:<22} {'PRESENT' if PRESENT[d] else 'MISSING'}")

# The consequence of each missing dependency, stated before it can bite.
problems = []
if not PRESENT["bitsandbytes"]:
    problems.append(
        "bitsandbytes MISSING -> the 4-bit load will raise ProvenanceError instead of "
        "quietly loading fp16 at 4x the VRAM. Install it, or set LOAD_IN_4BIT=False in "
        "cell 5 and accept the 'full-<dtype>' tag on every row."
    )
if not PRESENT["sentence_transformers"]:
    problems.append(
        "sentence_transformers MISSING -> DenseIndex.build() raises ProvenanceError, "
        "so the whole rag_dense column cannot be produced. This is deliberate: the "
        "old code returned corpus-order chunks with fabricated scores here and would "
        "have invented RQ3."
    )
if not PRESENT["rank_bm25"] or not PRESENT["pyvi"]:
    problems.append(
        "rank_bm25/pyvi missing -> BM25 self-declares DEGRADED and the campaign "
        "aborts (the Vietnamese compound-word handling is what RQ3 measures)."
    )
if problems:
    print("\\n[DEPENDENCY AUDIT]")
    for p in problems:
        print("  ! " + p)
else:
    print("\\n[DEPENDENCY AUDIT] all retrieval and quantization dependencies present.")

from regrag.generation.config import (
    DEFAULT_MODEL_ALIASES, MODEL_REGISTRY, RETRIEVAL_MODES, UNANSWERABLE_SENTINEL,
    SamplingConfig, select_models, summarize_selection, validate_matrix,
)
from regrag.generation.benchmark import load_benchmark
from regrag.generation.corpus_io import load_chunks, describe_corpus, file_sha256
from regrag.generation.backends import (
    TransformersQuantBackend, VLLMHttpBackend, gpu_report, resolve_backend, wait_for_vllm,
)
from regrag.generation.sanity import format_probe_report, probe_gate, run_probe
from regrag.generation.abstention import (
    assert_metrics_detects_sentinel, detect_abstention, summarize_verdicts,
)
from regrag.generation.checkpoint import (
    CheckpointStore, ProgressTracker, restore_from_drive, split_record,
    sync_to_drive, write_manifest,
)
from regrag.generation.campaign import CampaignRunner, build_manifest
from regrag.provenance import (
    CORPUS_TIER1, CORPUS_TIER2, ProvenanceError, assert_publishable, is_degraded,
    publishable_corpus,
)
from regrag.models import GenerationResult
from regrag.storage.file_repo import FileResultRepository

print("[OK] Harness imported.")
''')

# --------------------------------------------------------------------------- 5
md("""
## 5. CONFIG — single source of truth

Every path, model, mode and switch the campaign uses. **Nothing below is a magic
string repeated in a later cell.** The Drive directory in particular is defined
once here.
""")

code('''
# [CELL 5] CONFIG — the only place these values are defined.
RUN_NAME = "regrag_vn_" + datetime.now(timezone.utc).strftime("%Y%m%d")

# --- Drive (checkpoint medium, NOT the archive; see cell 15) -----------------
DRIVE_MOUNT = "/content/drive"                # parent of the mountpoint must exist
DRIVE_ROOT = f"{DRIVE_MOUNT}/MyDrive"         # valid only AFTER the mount
DRIVE_SUBDIR = "regrag_vn_checkpoints"        # <-- the ONE Drive parameter
DRIVE_DIR = f"{DRIVE_ROOT}/{DRIVE_SUBDIR}"

# --- local scratch (/content is ephemeral; Drive is the durable copy) --------
LOCAL_WORK = "/content/regrag_work"
CHECKPOINT_DIR = f"{LOCAL_WORK}/checkpoints"   # generations.jsonl lives here
RESULTS_DIR = f"{LOCAL_WORK}/{RUN_NAME}"       # repo-format output lands here

# --- inputs (relative to REPO_ROOT) -----------------------------------------
GOLD_CSV = "data/gold/bank_qa_data.csv"
# Tier 2 is the publishable corpus. Tier 1 is a development fixture and
# assert_publishable() refuses it, so it can never reach a paper table.
CORPUS_CANDIDATES = [
    "data/processed_chunks/corpus_chunks.json",    # Tier 2 — preferred
    "data/processed_chunks/tier1_chunks.json",     # Tier 1 — fixture, NOT publishable
]

# --- experimental matrix (plan §3.3) ----------------------------------------
MODEL_ALIASES = list(DEFAULT_MODEL_ALIASES)     # qwen-7b, qwen-3b, vistral-7b
RETRIEVAL_MODES_SELECTED = list(RETRIEVAL_MODES)  # closed_book, rag_bm25, rag_dense
TOP_K = 3
DENSE_MODEL_NAME = "BAAI/bge-m3"

# --- generation backend -----------------------------------------------------
# "auto"        : use a vLLM server if one is reachable, else local transformers.
# "vllm"        : require the server. NEVER falls back silently, because the two
#                 paths apply the chat template differently and their rows must
#                 not be pooled.
# "transformers": local 4-bit load, no server.
BACKEND_PREFERENCE = "auto"
VLLM_BASE_URL = "http://127.0.0.1:8000/v1"
# What the vLLM server was started with. The client CANNOT verify server-side
# weights, so leaving this None records quant_config="UNRECORDED:vllm-server-managed"
# rather than claiming a precision nobody measured.
VLLM_QUANT_DECLARED = None          # e.g. "bitsandbytes-4bit" or "fp16"
LOAD_IN_4BIT = True
STRICT_QUANTIZATION = True          # raise if the 4-bit load did not take effect

# --- safety switches (leave these alone) ------------------------------------
ABORT_ON_DEGRADED_RETRIEVAL = True  # stop rather than write DEGRADED rows
BACKEND_CONFLICT_POLICY = "raise"   # cached row from another backend -> stop
ALLOW_TIER1_RUN = True              # permit a Tier-1 fixture run, tagged unpublishable

# --- sampling and cadence ---------------------------------------------------
SAMPLING = SamplingConfig(max_new_tokens=512, temperature=0.0, do_sample=False, seed=1234)
MATERIALIZE_EVERY = 25              # rows between repo-format materialisations
SYNC_EVERY_MODELS = 1               # Drive sync after every N models
PROGRESS_EVERY = 10                 # rows between progress lines

# Models approved for the campaign (cell 12 sets this; pre-filled there).
HUMAN_APPROVED_MODELS = []

print(f"RUN_NAME   : {RUN_NAME}")
print(f"DRIVE_DIR  : {DRIVE_DIR}")
print(f"LOCAL_WORK : {LOCAL_WORK}")
print(f"RESULTS_DIR: {RESULTS_DIR}")
print(f"models     : {MODEL_ALIASES}")
print(f"modes      : {RETRIEVAL_MODES_SELECTED}")
print(f"backend    : {BACKEND_PREFERENCE}  4bit={LOAD_IN_4BIT}")
GRID = validate_matrix(MODEL_ALIASES, RETRIEVAL_MODES_SELECTED, 88)
print(f"grid       : {GRID}")
print(f"sampling   : {SAMPLING.as_dict()}")
''')


# --------------------------------------------------------------------------- 6
md("""
## 6. Google Drive mount + RESTORE from checkpoint

Mounts Drive, then restores any prior checkpoint for this campaign and **prints
what it found and when it was written**. A disconnect at any later point costs at
most the rows since the last sync, and the restart is "Run all" up to cell 12.

Checkpoints are copied *out* of the Drive mount into `/content` before use:
reading and fsync-ing every row through the FUSE mount stalls the loop.
""")

code('''
# [CELL 6] Drive mount + RESTORE. Idempotent; safe to re-run.
import os, shutil
from google.colab import drive

# Mount at /content/drive (its parent /content always exists on Colab); the
# MyDrive subtree appears after mounting. Mounting at .../MyDrive directly
# fails on current Colab ("Mountpoint must be in a directory that exists").
drive.mount(DRIVE_MOUNT)
os.makedirs(DRIVE_DIR, exist_ok=True)
os.makedirs(LOCAL_WORK, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
print(f"[DRIVE] checkpoint dir : {DRIVE_DIR}")

# A run is identified by the CSV + corpus + grid, not by RUN_NAME, so a session
# restarted on a new day resumes the SAME checkpoint instead of starting over.
RESTORE_REPORT = restore_from_drive(
    drive_dir=DRIVE_DIR,
    local_dir=CHECKPOINT_DIR,
    names=("generations.jsonl", "run_manifest.json"),
)

STORE = CheckpointStore(CHECKPOINT_DIR)
print(f"\\n[RESTORE] {len(STORE)} completed row(s) available for resume.")
if STORE.skipped_lines:
    print(f"[RESTORE] {len(STORE.skipped_lines)} unusable line(s) skipped "
          "(a torn final line is the expected artifact of a killed session).")
print(f"[RESTORE] rows per generation_backend so far: {STORE.backends_present()}")
''')

# --------------------------------------------------------------------------- 7
md("""
## 7. Questions and corpus

Loads the 88-row gold CSV through the harness loader and the chunk corpus, then
reports exactly what the campaign is bound to. Both are loud:

* the 24 unanswerable probes must be detected, or RQ2 cannot be measured;
* probe ids derived from CSV **row order** are flagged, because inserting a row
  renumbers them and invalidates every cached probe generation;
* a corpus with 0 chunks raises rather than letting the RAG modes silently
  degenerate into closed-book;
* a Tier-1 fixture corpus is announced as **not publishable**.
""")

code('''
# [CELL 7] Load the benchmark questions and the corpus. Fails loud on both.
BENCH = load_benchmark(os.path.join(REPO_ROOT, GOLD_CSV))
N_QUESTIONS = len(BENCH.questions)
print(f"\\n[QUESTIONS] {N_QUESTIONS} total | {len(BENCH.answerable)} answerable | "
      f"{len(BENCH.probes)} unanswerable probes")
print(f"[QUESTIONS] csv_hash = {BENCH.csv_hash[:16]}...")
print(f"[QUESTIONS] classification basis: {BENCH.basis_counts()}")
if BENCH.anomaly_ids:
    print(f"[QUESTIONS][WARNING] anomalies: "
          f"{ {k: len(v) for k, v in BENCH.anomaly_ids.items()} }")

# RQ2 depends on the probes. Without them the abstention column is meaningless.
if len(BENCH.probes) == 0:
    raise SystemExit(
        "[FATAL] 0 unanswerable probes were detected. RQ2 (abstention) cannot be "
        "measured on this CSV. Expected 23 rows whose gold answer is "
        f"{UNANSWERABLE_SENTINEL!r}."
    )

# The scorer and the harness must agree on what counts as a refusal.
assert_metrics_detects_sentinel()
print("[QUESTIONS] abstention scorer recognises the gold sentinel: OK")

# --- corpus ------------------------------------------------------------------
CHUNKS, CORPUS_PATH, CORPUS_DESC = [], None, None
for rel in CORPUS_CANDIDATES:
    p = os.path.join(REPO_ROOT, rel)
    if not os.path.exists(p):
        print(f"[CORPUS] not present: {rel}")
        continue
    try:
        CHUNKS = load_chunks(p)
        CORPUS_PATH = p
        CORPUS_DESC = describe_corpus(CHUNKS, p)
        break
    except ProvenanceError as exc:
        print(f"[CORPUS] rejected {rel}: {exc}")

if not CHUNKS:
    raise SystemExit(
        "[FATAL] No usable corpus. corpus_chunks.json is empty until the source "
        "instruments are ingested (scripts/build_corpus_chunks.py), and a RAG run "
        "over an empty corpus silently becomes a closed-book run. Ingest first."
    )

print(f"\\n[CORPUS] path           : {CORPUS_DESC['corpus_path']}")
print(f"[CORPUS] corpus_source  : {CORPUS_DESC['corpus_source']}")
print(f"[CORPUS] chunks         : {CORPUS_DESC['chunk_count']}")
print(f"[CORPUS] instruments    : {CORPUS_DESC['distinct_doc_ids']}")
print(f"[CORPUS] corpus_hash    : {CORPUS_DESC['corpus_hash'][:16]}...")
print(f"[CORPUS] publishable    : {CORPUS_DESC['publishable_corpus']}")

if not CORPUS_DESC["publishable_corpus"]:
    if not ALLOW_TIER1_RUN:
        raise SystemExit("[FATAL] corpus is not Tier 2 and ALLOW_TIER1_RUN is False.")
    print(
        "\\n" + "!" * 72 +
        "\\n[CORPUS][WARNING] This is a TIER 1 development fixture. Every row will be"
        "\\nstamped corpus_source='tier1_passages' and assert_publishable() will REFUSE"
        "\\nthem. Recall@3 is ~1.0 by construction, so BM25 cannot be distinguished from"
        "\\ndense: RQ3 is unmeasurable on this corpus. Fine for a harness dry run,"
        "\\nNEVER for a paper table.\\n" + "!" * 72
    )
''')

# --------------------------------------------------------------------------- 8
md("""
## 8. Retrieval indices

Builds BM25 (and the dense index, only if `rag_dense` is selected — BGE-M3 is a
~2 GB download and is not fetched for a closed-book/BM25-only run).

Both backends **self-report** and the cell refuses to continue on a degraded one:
BM25 without `rank_bm25`/`pyvi` degrades to term-overlap counting without the
Vietnamese compound-word segmentation RQ3 exists to measure, and `DenseIndex`
raises outright rather than returning fabricated scores.
""")

code('''
# [CELL 8] Build retrieval indices and report the real backend tags.
STORE_RUNNER = CampaignRunner(
    benchmark=BENCH,
    chunks=CHUNKS,
    corpus_path=CORPUS_PATH,
    checkpoint_store=STORE,
    modes=RETRIEVAL_MODES_SELECTED,
    top_k=TOP_K,
    dense_model_name=DENSE_MODEL_NAME,
    sampling=SAMPLING,
    gpu_info=GPU_INFO,
    abort_on_degraded_retrieval=ABORT_ON_DEGRADED_RETRIEVAL,
    backend_conflict_policy=BACKEND_CONFLICT_POLICY,
    results_dir=RESULTS_DIR,
)

t0 = time.time()
INDEX_TAGS = STORE_RUNNER.prepare_indices()
BM25_BACKEND = INDEX_TAGS["rag_bm25"]
DENSE_BACKEND = INDEX_TAGS["rag_dense"]
print(f"[INDEX] built in {time.time() - t0:.1f}s over {len(CHUNKS)} chunks")
print(f"[INDEX] bm25 backend  : {BM25_BACKEND}")
print(f"[INDEX] dense backend : {DENSE_BACKEND}")
if DENSE_BACKEND == "not-built":
    print("[INDEX] rag_dense not selected — dense index NOT built, BGE-M3 not downloaded.")

if is_degraded(BM25_BACKEND) or is_degraded(DENSE_BACKEND):
    raise SystemExit(
        f"[FATAL] A retrieval backend is DEGRADED (bm25={BM25_BACKEND!r}, "
        f"dense={DENSE_BACKEND!r}). Rows produced now would look clean and measure "
        "nothing. Install the missing dependency and re-run."
    )

# A quick retrieval smoke check: the top hit for a question should be on-topic.
_sample = BENCH.answerable[0]
_hits, _hit_backend = STORE_RUNNER.retrieve_for(_sample, "rag_bm25")
print(f"\\n[INDEX] smoke check — {_sample.id} via {_hit_backend}")
print(f"[INDEX]   Q: {_sample.question[:70]}...")
for h in _hits:
    print(f"    rank {h.rank} score {h.score:.3f} {h.chunk.chunk_id} "
          f"[{h.chunk.doc_id}] Điều {h.chunk.article_id}")
print("[OK] Indices ready.")
''')

# --------------------------------------------------------------------------- 9
md("""
## 9. Generation backend

Resolves the backend **per model** and keeps the object, so the sanity probe in
cell 10 tests the exact instance the campaign will use.

* `auto` (default) — probe `VLLM_BASE_URL` for a server serving the expected
  alias; if none, load locally with `transformers` + 4-bit. The fallback is
  printed, never silent.
* `vllm` — require the server. It will **not** fall back, because the two paths
  apply the chat template differently (server-side vs
  `tokenizer.apply_chat_template`) and their rows must not be pooled.
* `transformers` — local 4-bit only.

`generation_backend` recorded on every row is `vllm-serve:<alias>` or
`transformers-4bit` (or `DEGRADED:<reason>`), and `quant_config` records the
precision that was **verified**, not the one that was requested.

Serving through vLLM is the path `scripts/load_vllm_models.py` provides; it is
what makes a 5-model matrix one integration instead of five. On a single T4 only
one 7B server fits, so the default here is the local 4-bit path.
""")

code('''
# [CELL 9] Optional: start a vLLM server. Skipped unless USE_VLLM_SERVER is True.
# Requires `!pip install -q vllm` in cell 3. On one T4 this serves ONE model.
USE_VLLM_SERVER = False
VLLM_SERVER_PROC = None

if USE_VLLM_SERVER:
    import subprocess
    alias = MODEL_ALIASES[0]
    hf_id = MODEL_REGISTRY[alias].hf_id
    cmd = ["bash", os.path.join(REPO_ROOT, "scripts/serve_vllm.sh"), hf_id,
           "--host", "127.0.0.1", "--port", "8000",
           "--served-model-name", alias,
           "--max-model-len", "4096", "--gpu-memory-utilization", "0.90"]
    print("[vLLM] starting:", " ".join(cmd))
    VLLM_SERVER_PROC = subprocess.Popen(cmd, cwd=REPO_ROOT,
                                        stdout=open("/content/vllm.log", "w"),
                                        stderr=subprocess.STDOUT)
    print("[vLLM] log at /content/vllm.log — first load takes several minutes.")
else:
    print("[vLLM] Not starting a server; BACKEND_PREFERENCE="
          f"{BACKEND_PREFERENCE!r} will resolve per model in the next cell.")
''')

code('''
# [CELL 10] Resolve one backend per model. These instances are what the probe tests.
BACKENDS = {}
BACKEND_DESC = []

for alias in MODEL_ALIASES:
    spec = MODEL_REGISTRY[alias]
    print("\\n" + "=" * 72)
    print(f"[BACKEND] resolving {alias} ({spec.hf_id}) role={spec.role}")
    print("=" * 72)
    if not spec.is_peer:
        print(f"[BACKEND][WARNING] {alias} is role={spec.role}, NOT a peer. It must be "
              "reported in its own labelled column and must not be averaged into the "
              "cross-model headline number.")

    b = resolve_backend(
        spec,
        preference=BACKEND_PREFERENCE,
        vllm_base_url=VLLM_BASE_URL,
        quantization_declared=VLLM_QUANT_DECLARED,
        sampling=SAMPLING,
        strict_quantization=STRICT_QUANTIZATION,
        load_transformers=True,
    )
    BACKENDS[alias] = b
    BACKEND_DESC.append(b.describe())
    print(f"[BACKEND] {alias}: generation_backend={b.backend_tag()!r} "
          f"quant_config={b.quant_tag()!r}")

    # Record the precision NOW. A row that claims 4-bit without verification is
    # exactly the "plausible default" this repo forbids.
    if is_degraded(b.backend_tag()) or is_degraded(b.quant_tag()):
        raise SystemExit(
            f"[FATAL] backend for {alias} is degraded: "
            f"generation_backend={b.backend_tag()!r} quant_config={b.quant_tag()!r}"
        )
    if b.quant_tag().startswith("UNRECORDED"):
        print(f"[BACKEND][WARNING] quant_config={b.quant_tag()!r} for {alias}. Set "
              "VLLM_QUANT_DECLARED in cell 5 to record what the server was started "
              "with; otherwise the row states honestly that precision is unverified.")

print("\\n[OK] Backends resolved:")
for d in BACKEND_DESC:
    print(f"  {d.get('generation_backend'):<28} quant={d.get('quant_config'):<28} "
          f"template_by={d.get('template_applied_by')}")
''')

# --------------------------------------------------------------------------- 10
md("""
## 10. PER-MODEL PROMPT-FORMAT SANITY PROBE — human approval required

Three fixed questions per model, **on the backend the campaign will actually
use**, printed in full. This is the guard against the failure mode plan §3.3
calls out: a wrong chat template produces degenerate output that reads as *"this
model hallucinates"* when the truth is *"we prompted it wrong"*.

### What to look for before approving

| Check | Pass | Fail |
|---|---|---|
| Language | Fluent Vietnamese, diacritics intact | English, or diacritics stripped |
| Template | Clean prose | Chat-template control tokens (the `im_start` / `im_end` style markers, `[INST]`, `start_of_turn`) visible in the OUTPUT |
| Repetition | Varied vocabulary | A token or short phrase looping; distinct-token ratio under ~0.30 |
| Content | Answers the question | Echoes the question back, or returns an empty string |
| Refusal | Probe 3 refuses in the registered wording | Probe 3 invents a regulation |
| Prompt | If `prompt_exact` is True, the literal token string is printed and must look like the model's own format | If False (vLLM), the template is applied server-side — judge the OUTPUT, since the client cannot show the literal string |

A model that fails this probe has **not** been shown to hallucinate. It has been
shown to be mis-prompted, and its campaign column would measure our bug.
""")

code('''
# [CELL 11] SANITY PROBE — run this, then READ the output before cell 12.
PROBE_REPORTS = {}
PROBE_MODE = "closed_book"   # probe the prompt path with no retrieved context

for alias in MODEL_ALIASES:
    b = BACKENDS[alias]
    print("\\n\\n")
    reports = run_probe(b, mode=PROBE_MODE)
    PROBE_REPORTS[alias] = reports
    print(format_probe_report(alias, reports))

print("\\n\\n" + "=" * 72)
print("PROBE SUMMARY")
print("=" * 72)
for alias, reports in PROBE_REPORTS.items():
    status = "PASS" if all(r.ok for r in reports) else "FAIL"
    n_find = sum(len(r.findings) for r in reports)
    print(f"  {alias:<16} {status}  ({len(reports)} outputs, {n_find} finding(s))")
''')

code('''
# [CELL 12] PROBE GATE — programmatic block + model approval.
# Pre-filled with the selected model set (owner decision 2026-09-12: no manual
# edit needed on Colab). READ CELL 11'S OUTPUT BEFORE RUNNING CELL 13 anyway —
# the detectors below still block on template/repetition/refusal failures, but
# only a human can judge whether the Vietnamese is fluent. To re-arm the manual
# approval gate, set this list back to [].
HUMAN_APPROVED_MODELS = list(MODEL_ALIASES)

# 1. Programmatic gate: any blocking finding stops everything.
for alias, reports in PROBE_REPORTS.items():
    probe_gate(reports)
print("[GATE] No blocking detector findings on any model.")

# 2. Human gate: the detectors cannot judge whether the Vietnamese is actually
#    good, or whether the citation format is what the scorer expects.
missing = [a for a in MODEL_ALIASES if a not in HUMAN_APPROVED_MODELS]
if missing:
    raise SystemExit(
        "[GATE] SANITY PROBE NOT HUMAN-APPROVED for: " + ", ".join(missing) + "\\n"
        "Read cell 11's output for each model, then set\\n"
        "    HUMAN_APPROVED_MODELS = " + str(MODEL_ALIASES) + "\\n"
        "in this cell and re-run it. The campaign cell will not execute until you do.\\n"
        "A wrong chat template produces degenerate output that reads as 'this model\\n"
        "hallucinates' — approving the probe is how that is ruled out."
    )

for alias in HUMAN_APPROVED_MODELS:
    r0 = PROBE_REPORTS[alias][0]
    print(f"[GATE] {alias:<16} human-approved | backend={r0.backend_tag} "
          f"| prompt_exact={r0.prompt_exact}")
PROBE_APPROVED = True
print("\\n[OK] Probe gate passed. Cell 13 may now run.")
''')


# --------------------------------------------------------------------------- 13
md("""
## 13. CAMPAIGN — model x mode x question, checkpointed

Loops **model outer, then mode, then question**, because the scarce resource is
the model: all 264 cells for one model complete before it is unloaded, which is
what keeps three 7B/4B models inside a single 16 GB T4 session.

Retrieval is computed **once per (question, mode)** and reused across models, so
the three model columns are conditioned on byte-identical context — that is what
makes RQ1 and RQ3 comparisons valid.

Every row is appended to `generations.jsonl` and fsync'd the moment it lands, and
the checkpoint is pushed to Drive every `MATERIALIZE_EVERY` rows. A disconnect
therefore costs at most that many rows, and re-running this cell skips every cell
already in the checkpoint.
""")

code('''
# [CELL 13] THE CAMPAIGN. Re-runnable: completed cells are skipped, not redone.
if not globals().get("PROBE_APPROVED"):
    raise SystemExit(
        "[FATAL] The sanity-probe gate (cell 12) has not been passed. Run cells 11 "
        "and 12 and approve the prompt format for every model first."
    )

TOTAL_CELLS = len(MODEL_ALIASES) * len(RETRIEVAL_MODES_SELECTED) * N_QUESTIONS
ALREADY = len(STORE)
print("=" * 72)
print(f"[CAMPAIGN] grid          : {len(MODEL_ALIASES)} models x "
      f"{len(RETRIEVAL_MODES_SELECTED)} modes x {N_QUESTIONS} questions = {TOTAL_CELLS}")
print(f"[CAMPAIGN] resumed rows  : {ALREADY} / {TOTAL_CELLS}")
print(f"[CAMPAIGN] to generate   : {max(TOTAL_CELLS - ALREADY, 0)}")
print(f"[CAMPAIGN] corpus        : {CORPUS_DESC['corpus_source']} "
      f"(publishable={CORPUS_DESC['publishable_corpus']})")
print("=" * 72, flush=True)

def _sync():
    """Push the checkpoint and materialised results up to Drive."""
    sync_to_drive(
        local_dir=CHECKPOINT_DIR,
        drive_dir=DRIVE_DIR,
        names=("generations.jsonl", "run_manifest.json"),
        extra_dirs=[(RESULTS_DIR, "results")],
    )

def _usable(b):
    """A transformers backend that was closed to free VRAM must be re-loaded."""
    return not (isinstance(b, TransformersQuantBackend) and b.model is None)

PROGRESS = ProgressTracker(total=TOTAL_CELLS, every=PROGRESS_EVERY)
RUN_RESULTS = []
T_START = time.time()

for i, alias in enumerate(MODEL_ALIASES, 1):
    spec = MODEL_REGISTRY[alias]
    backend = BACKENDS.get(alias)
    if backend is None or not _usable(backend):
        print(f"\\n[CAMPAIGN] (re)resolving backend for {alias}", flush=True)
        backend = resolve_backend(
            spec, preference=BACKEND_PREFERENCE, vllm_base_url=VLLM_BASE_URL,
            quantization_declared=VLLM_QUANT_DECLARED, sampling=SAMPLING,
            strict_quantization=STRICT_QUANTIZATION, load_transformers=True,
        )
        BACKENDS[alias] = backend

    print(f"\\n{'#' * 72}")
    print(f"# [{i}/{len(MODEL_ALIASES)}] {alias}  ({spec.hf_id})  role={spec.role}")
    print(f"# backend={backend.backend_tag()}  quant={backend.quant_tag()}")
    print(f"{'#' * 72}", flush=True)

    t0 = time.time()
    res = STORE_RUNNER.run_model(
        spec, backend, progress=PROGRESS,
        materialize_every=MATERIALIZE_EVERY, on_error="raise",
        on_materialize=_sync,
    )
    res["elapsed_s"] = round(time.time() - t0, 1)
    RUN_RESULTS.append(res)
    print(f"[CAMPAIGN] {alias}: wrote {res['rows_written']}, "
          f"skipped {res['rows_skipped_cached']} cached, "
          f"{res['elapsed_s']}s", flush=True)

    _sync()                      # durable before the next model loads
    backend.close()              # free VRAM so model i+1 fits

ELAPSED = time.time() - T_START
print("\\n" + "=" * 72)
print(f"[CAMPAIGN] DONE in {ELAPSED/60:.1f} min")
print(f"[CAMPAIGN] rows in checkpoint : {len(STORE)} / {TOTAL_CELLS}")
print(f"[CAMPAIGN] progress summary   : {PROGRESS.summary()}")
print("=" * 72)

if len(STORE) < TOTAL_CELLS:
    print(f"[CAMPAIGN][WARNING] {TOTAL_CELLS - len(STORE)} cell(s) are still missing. "
          "Re-run this cell to continue from the checkpoint.")
''')

# --------------------------------------------------------------------------- 14
md("""
## 14. SAVE to Drive (manual checkpoint)

Run this at any time — before a risky step, when the session is getting slow, or
right after an interrupt. It is idempotent and cheap.
""")

code('''
# [CELL 14] MANUAL SAVE — checkpoint to Drive right now.
STORE.materialize(RESULTS_DIR)
STORE.assert_meta_complete(RESULTS_DIR)
SAVE_REPORT = _sync() if "_sync" in dir() else sync_to_drive(
    local_dir=CHECKPOINT_DIR, drive_dir=DRIVE_DIR,
    names=("generations.jsonl", "run_manifest.json"),
    extra_dirs=[(RESULTS_DIR, "results")],
)
print(f"[SAVE] {len(STORE)} row(s) checkpointed")
print(f"[SAVE] copied: {SAVE_REPORT['copied']}")
if SAVE_REPORT["errors"]:
    print(f"[SAVE][ERROR] {SAVE_REPORT['errors']}")
''')

# --------------------------------------------------------------------------- 15
md("""
## 15. Provenance audit

The gate from plan §5: **no number reaches a paper table without a clean
provenance tag.** This cell writes the run manifest and then states, without
softening it, whether the rows are publishable.

It reports rather than hides:
* rows per `generation_backend` — and refuses to pool a model whose rows came
  from two backends;
* every `DEGRADED` row, individually;
* whether `corpus_source` is Tier 2 (publishable) or the Tier 1 fixture (never);
* the abstention behaviour on the 23 probes, which is RQ2;
* `assert_publishable()` run over the materialised rows, so the verdict is the
  repo's own guard and not this notebook's opinion.
""")

code('''
# [CELL 15] Provenance audit + run manifest. Read this before touching a table.
MATERIALIZE_INFO = STORE.materialize(RESULTS_DIR)
STORE.assert_meta_complete(RESULTS_DIR)
print(f"[AUDIT] generations.json : {MATERIALIZE_INFO['generations_json']}")
print(f"[AUDIT] rows total       : {MATERIALIZE_INFO['rows_total']}")
print(f"[AUDIT] meta sidecar     : {MATERIALIZE_INFO['meta_jsonl']}")

PER_MODEL_BACKENDS = STORE_RUNNER.assert_no_backend_pooling()
PROV = STORE_RUNNER.provenance_summary()
print("\\n[AUDIT] rows per generation_backend :")
for tag, n in sorted(PROV["generation_backends"].items()):
    print(f"    {n:>5}  {tag}")
print("[AUDIT] retriever_backends          :", PROV["retriever_backends"])
print("[AUDIT] corpus_sources              :", PROV["corpus_sources"])
print(f"[AUDIT] publishable corpus          : {PROV['publishable_corpus']}")
print(f"[AUDIT] DEGRADED rows               : {PROV['degraded_row_count']}")
for row in PROV["degraded_rows"]:
    print(f"    ! {row}")

# --- abstention (RQ2) --------------------------------------------------------
RECS = STORE.records()
PROBE_ROWS = [r for r in RECS if r.get("is_answerable") is False]
ANS_ROWS = [r for r in RECS if r.get("is_answerable") is True]
VERDICTS = [detect_abstention(r.get("answer_text", "")) for r in PROBE_ROWS]
print(f"\\n[RQ2] probe rows                 : {len(PROBE_ROWS)}")
print(f"[RQ2] abstention on probes       : {summarize_verdicts(VERDICTS)}")
SOFT = [r["question_id"] for r, v in zip(PROBE_ROWS, VERDICTS) if v.soft_only]
if SOFT:
    print(f"[RQ2][REVIEW] {len(SOFT)} probe row(s) refused in unregistered wording "
          f"(human review): {sorted(set(SOFT))[:12]}")
EMPTY_ANS = sum(1 for r in RECS if not (r.get("answer_text") or "").strip())
print(f"[RQ2] empty answers (all rows)   : {EMPTY_ANS}")
if EMPTY_ANS:
    print("[RQ2][WARNING] Empty answers are FAILED GENERATIONS, not abstentions. "
          "They are counted as non-abstention; investigate before reporting.")

# --- the repo's own publishability guard ------------------------------------
ROWS = [GenerationResult(**split_record(r)[0]) for r in RECS]
try:
    assert_publishable(ROWS)
    PUBLISHABLE = True
    print("\\n[AUDIT] assert_publishable(): PASSED — rows may reach a paper table.")
except ProvenanceError as exc:
    PUBLISHABLE = False
    print("\\n[AUDIT] assert_publishable(): REFUSED")
    print(str(exc)[:2000])

MANIFEST = build_manifest(
    run_name=RUN_NAME,
    benchmark=BENCH,
    corpus_description=CORPUS_DESC,
    model_specs=[MODEL_REGISTRY[a] for a in MODEL_ALIASES],
    modes=RETRIEVAL_MODES_SELECTED,
    top_k=TOP_K,
    sampling=SAMPLING,
    gpu_info=GPU_INFO,
    backend_descriptions=BACKEND_DESC,
    run_results=RUN_RESULTS if "RUN_RESULTS" in dir() else [],
    progress_summary=PROGRESS.summary() if "PROGRESS" in dir() else {},
    provenance=PROV,
    probe_reports=[r.as_dict() for rs in PROBE_REPORTS.values() for r in rs]
                  if "PROBE_REPORTS" in dir() else [],
    extra={
        "human_approved_models": HUMAN_APPROVED_MODELS if "HUMAN_APPROVED_MODELS" in dir() else [],
        "assert_publishable_passed": PUBLISHABLE,
        "results_dir": RESULTS_DIR,
        "checkpoint_dir": CHECKPOINT_DIR,
        "drive_dir": DRIVE_DIR,
        "elapsed_s": round(ELAPSED, 1) if "ELAPSED" in dir() else None,
    },
)
write_manifest(os.path.join(CHECKPOINT_DIR, "run_manifest.json"), MANIFEST)
print(f"\\n[AUDIT] run manifest written to {CHECKPOINT_DIR}/run_manifest.json")
''')

# --------------------------------------------------------------------------- 16
md("""
## 16. Results + fetch back into the repo

Drive is a **checkpoint medium, not an archive**. The durable copy is the git
repository, so this cell packages everything and states the exact repo path it
must be committed to. It also offers a one-click download for a local machine.

Archive into: `colab_archive/<RUN_NAME>/` in the repo.
""")

code('''
# [CELL 16] Package results and state where they must be archived.
ARCHIVE_NAME = f"{RUN_NAME}_results"
ARCHIVE_BASE = os.path.join(LOCAL_WORK, ARCHIVE_NAME)
shutil.make_archive(ARCHIVE_BASE, "zip", root_dir=LOCAL_WORK, base_dir=RUN_NAME)
ZIP_PATH = ARCHIVE_BASE + ".zip"
shutil.copy2(ZIP_PATH, os.path.join(DRIVE_DIR, os.path.basename(ZIP_PATH)))
shutil.copy2(os.path.join(CHECKPOINT_DIR, "generations.jsonl"),
             os.path.join(DRIVE_DIR, "generations.jsonl"))
print(f"[FETCH] zip          : {ZIP_PATH} "
      f"({os.path.getsize(ZIP_PATH)/1024:.0f} KB)")
print(f"[FETCH] copied to    : {DRIVE_DIR}")

try:
    from google.colab import files as _colab_files
    _colab_files.download(ZIP_PATH)
    print("[FETCH] Browser download started.")
except Exception as exc:
    print(f"[FETCH] Browser download unavailable ({exc}); pull the zip from Drive.")

print("\\n" + "=" * 72)
print("ARCHIVE THIS INTO THE REPO — the run is not done until it is committed")
print("=" * 72)
print(f"""
  1. Copy the zip out of Drive, then in the repo:

         mkdir -p colab_archive/{RUN_NAME}
         unzip {os.path.basename(ZIP_PATH)} -d colab_archive/{RUN_NAME}/
         cp generations.jsonl colab_archive/{RUN_NAME}/checkpoints/

  2. Record the run (notebook name, date, headline numbers) in the project
     results registry.

  3. Commit. Never report this experiment as done while its only copy is in
     Drive or /content — both are ephemeral.

  Publishability verdict for these rows: {PUBLISHABLE if 'PUBLISHABLE' in dir() else 'UNKNOWN'}
  Rows in checkpoint                   : {len(STORE)}
""")
''')

# --------------------------------------------------------------------------- 17
md("""
## Appendix — runtime arithmetic

Rough budget for a T4, used to decide whether the grid fits in one session.

* 792 cells = 3 models x 3 modes x 88 questions.
* Prompt is ~350-600 tokens for the RAG modes (3 chunks of legal text) and
  ~180 tokens closed-book; output is capped at 512 new tokens.
* Greedy 4-bit decoding on a T4 runs roughly 15-25 new tokens/s for a 7B model
  and 30-45 for a 3B. A 512-token completion therefore costs ~20-35 s (7B) or
  ~12-18 s (3B); typical legal answers stop earlier, ~150-300 tokens.
* Model loads: ~2-4 min each for a 4-bit 7B (download + quantize), once per
  model. BGE-M3 download + encode: ~3-6 min once.

| Component | Estimate |
|---|---|
| 2 x 7B models x 264 cells x ~20 s | ~2.9 h |
| 1 x 3B model x 264 cells x ~12 s | ~0.9 h |
| 3 model loads + dense index build | ~0.3 h |
| **Total** | **~4.1 h** |

A free-tier Colab session is typically cut well before 4 h, and a T4 under load
is slower than this. **Plan on using the resume path**: the checkpoint makes a
restart cost only the rows since the last Drive sync, so the campaign is run as
several shorter sessions rather than one long gamble. Run cell 14 (manual SAVE)
before any risky step.

Scaling to 5 models x 5 modes would be ~2,100 cells (~8-14 T4-hours), which is
where serving all models through vLLM on one GPU (cell 9) pays for itself: one
integration instead of five, and the campaign loop does not change.
""")


def main() -> int:
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "colab_runner.ipynb")
    nb = {
        "nbformat": 4,
        "nbformat_minor": 0,
        "metadata": {
            "colab": {"provenance": [], "gpuType": "T4", "toc_visible": True},
            "kernelspec": {"name": "python3", "display_name": "Python 3"},
            "language_info": {"name": "python"},
            "accelerator": "GPU",
        },
        "cells": CELLS,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)
        f.write("\n")
    print(f"Wrote {path} with {len(CELLS)} cells "
          f"({sum(1 for c in CELLS if c['cell_type']=='code')} code, "
          f"{sum(1 for c in CELLS if c['cell_type']=='markdown')} markdown)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
