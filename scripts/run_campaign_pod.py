"""Unattended RunPod driver for the RegRAG-VN 3x3x88 campaign (2026-09-13).

Mirrors the Colab notebook cells 5-13 exactly (same constants, same guards,
same corpus-candidate order - corpus_chunks.json is deterministically rejected
by the duplicate-chunk_id guard, so the run lands on tier1 exactly like the
banked Colab rows). Pod adaptations, each deliberate:

* no Google Drive - the checkpoint JSONL is durable in /workspace and the
  driver uploads checkpoint + materialised results to the private HF
  artifacts repo (hunopapa/regrag-artifacts) after every model, so a pod
  death cannot strand the run's evidence;
* the human sanity-probe gate (notebook cell 12) is replaced by
  scripts.preflight_memory, which forces one worst-case-prompt generation per
  model and requires >=1.5 GiB free at peak - the human spot-check happens on
  the archived outputs instead;
* FRESH run - no Colab checkpoint restore - so all 792 rows share one GPU's
  provenance (greedy decoding is deterministic, but bnb kernels differ
  between sm75 T4 and the pod's Ampere card, and mixed-hardware rows would be
  sloppiness a reviewer can smell).

Usage (on the pod, from the repo root):
    python3 -u scripts/run_campaign_pod.py            # full campaign
    python3 -u scripts/run_campaign_pod.py --plan     # CPU data-path check

Environment:
    HF_TOKEN       required for Vistral (gated) and artifact uploads
    HF_ARTIFACTS_REPO  default hunopapa/regrag-artifacts
    POD_WORK       default /workspace/regrag_pod
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from typing import Optional

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from regrag.generation.config import (  # noqa: E402
    DEFAULT_MODEL_ALIASES,
    DENSE_MODEL_NAME,
    MODEL_REGISTRY,
    RETRIEVAL_MODES,
    SamplingConfig,
    UNANSWERABLE_SENTINEL,
    validate_matrix,
)
from regrag.generation.backends import gpu_report, resolve_backend  # noqa: E402
from regrag.generation.benchmark import load_benchmark  # noqa: E402
from regrag.generation.corpus_io import describe_corpus, load_chunks  # noqa: E402
from regrag.generation.campaign import CampaignRunner  # noqa: E402
from regrag.generation.checkpoint import (  # noqa: E402
    CheckpointStore,
    ProgressTracker,
    write_manifest,
)
from regrag.generation.abstention import assert_metrics_detects_sentinel  # noqa: E402
from regrag.provenance import ProvenanceError, is_degraded  # noqa: E402

# --- constants: notebook cell 5, verbatim ------------------------------------
GOLD_CSV = "data/gold/bank_qa_data.csv"
CORPUS_CANDIDATES = [
    "data/processed_chunks/corpus_chunks.json",  # Tier 2 - rejected on duplicate ids
    "data/processed_chunks/tier1_chunks.json",   # Tier 1 fixture (the settled run)
]
MODEL_ALIASES = list(DEFAULT_MODEL_ALIASES)
RETRIEVAL_MODES_SELECTED = list(RETRIEVAL_MODES)
TOP_K = 3
BACKEND_PREFERENCE = "auto"
VLLM_BASE_URL = "http://127.0.0.1:8000/v1"
VLLM_QUANT_DECLARED = None
LOAD_IN_4BIT = True
STRICT_QUANTIZATION = True
ABORT_ON_DEGRADED_RETRIEVAL = True
BACKEND_CONFLICT_POLICY = "raise"
ALLOW_TIER1_RUN = True
SAMPLING = SamplingConfig(max_new_tokens=512, temperature=0.0, do_sample=False, seed=1234)
MATERIALIZE_EVERY = 25
PROGRESS_EVERY = 10

POD_WORK = os.environ.get("POD_WORK", "/workspace/regrag_pod")


def latest_resumable_run(work_root: str) -> Optional[str]:
    """Newest pod_* run dir carrying a non-empty checkpoint, or None.

    A wedged generation kill must not cost the 600+ banked rows: relaunching
    the driver reuses the newest checkpointed run instead of opening a fresh
    one (dir names sort chronologically by construction).
    """
    if not os.path.isdir(work_root):
        return None
    best = None
    for name in sorted(os.listdir(work_root)):
        if not name.startswith("pod_"):
            continue
        ckpt = os.path.join(work_root, name, "checkpoints", "generations.jsonl")
        if os.path.isfile(ckpt) and os.path.getsize(ckpt) > 0:
            best = os.path.join(work_root, name)
    return best


def _upload_to_hub(local_dir: str, remote_path: str, commit_msg: str) -> bool:
    """Folder upload to the private artifacts repo; guarded, non-fatal."""
    repo = os.environ.get("HF_ARTIFACTS_REPO", "hunopapa/regrag-artifacts")
    if not os.environ.get("HF_TOKEN"):
        print(f"[HUB][WARN] no HF_TOKEN in env; skipping upload of {local_dir}")
        return False
    cmd = [
        "hf", "upload", repo, local_dir, remote_path,
        "--type", "dataset", "--commit-message", commit_msg,
    ]
    t0 = time.time()
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    except subprocess.TimeoutExpired:
        print(f"[HUB][WARN] upload of {local_dir} timed out")
        return False
    if r.returncode != 0:
        print(f"[HUB][WARN] upload rc={r.returncode}: {r.stderr.strip()[-400:]}")
        return False
    print(f"[HUB] uploaded {local_dir} -> {repo}/{remote_path} ({time.time()-t0:.0f}s)")
    return True


def _load_inputs():
    """Notebook cells 7 + 14: benchmark + corpus, with their loud guards."""
    bench = load_benchmark(os.path.join(REPO_ROOT, GOLD_CSV))
    print(f"[QUESTIONS] {len(bench.questions)} total | {len(bench.answerable)} answerable | "
          f"{len(bench.probes)} unanswerable probes")
    if len(bench.probes) == 0:
        raise SystemExit(
            f"[FATAL] 0 unanswerable probes detected; RQ2 unmeasurable. Expected rows "
            f"with gold answer {UNANSWERABLE_SENTINEL!r}."
        )
    assert_metrics_detects_sentinel()
    print("[QUESTIONS] abstention scorer recognises the gold sentinel: OK")

    chunks, corpus_path, desc = [], None, None
    for rel in CORPUS_CANDIDATES:
        p = os.path.join(REPO_ROOT, rel)
        if not os.path.exists(p):
            print(f"[CORPUS] not present: {rel}")
            continue
        try:
            chunks = load_chunks(p)
            corpus_path = p
            desc = describe_corpus(chunks, p)
            break
        except ProvenanceError as exc:
            print(f"[CORPUS] rejected {rel}: {exc}")
    if not chunks:
        raise SystemExit("[FATAL] No usable corpus.")
    print(f"[CORPUS] {desc['chunk_count']} chunks | source={desc['corpus_source']} | "
          f"publishable={desc['publishable_corpus']}")
    if not desc["publishable_corpus"] and not ALLOW_TIER1_RUN:
        raise SystemExit("[FATAL] corpus not Tier 2 and ALLOW_TIER1_RUN is False.")
    if not desc["publishable_corpus"]:
        print("[CORPUS][WARNING] Tier 1 fixture run - rows stamped tier1_passages; "
              "assert_publishable() will refuse them.")
    return bench, chunks, corpus_path, desc


def plan_mode() -> int:
    """CPU-only data-path validation: benchmark, corpus, BM25, one prompt."""
    from regrag.indexing.bm25 import BM25Index
    from regrag.generation.prompting import build_messages_with_report

    bench, chunks, corpus_path, desc = _load_inputs()
    idx = BM25Index(chunks)
    q = bench.answerable[0]
    hits = idx.search(q.question, top_k=TOP_K)
    messages, report = build_messages_with_report(q.question, "rag_bm25", hits)
    grid = validate_matrix(MODEL_ALIASES, RETRIEVAL_MODES_SELECTED, len(bench.questions))
    print(f"[PLAN] grid={grid} chunks={len(chunks)} corpus={desc['corpus_source']}")
    print(f"[PLAN] sample prompt: {len(messages[0]['content'])} chars, "
          f"context {report.context_chars}/{report.budget_chars}, "
          f"truncated={report.truncated_ranks} dropped={report.dropped_ranks}")
    print(f"[PLAN] sample top-hit: {hits[0].chunk.chunk_id} [doc {hits[0].chunk.doc_id}]")
    print("[PLAN] OK - data path validated on CPU; nothing GPU-side was touched.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", action="store_true", help="CPU data-path check only")
    args = ap.parse_args()
    if args.plan:
        return plan_mode()

    if not os.environ.get("HF_TOKEN"):
        raise SystemExit("[FATAL] HF_TOKEN not set - Vistral (gated) and artifact "
                         "uploads both need it.")

    resumed_dir = latest_resumable_run(POD_WORK)
    if resumed_dir:
        run_id = os.path.basename(resumed_dir)
        local_work = resumed_dir
        print(f"[RESUME] reusing {run_id} (existing checkpoint found; "
              "cached rows are skipped, not regenerated)")
    else:
        run_id = datetime.now(timezone.utc).strftime("pod_%Y%m%d_%H%M")
        local_work = os.path.join(POD_WORK, run_id)
        print(f"[RUN] fresh run {run_id}")
    ckpt_dir = os.path.join(local_work, "checkpoints")
    results_dir = os.path.join(local_work, "results")
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    bench, chunks, corpus_path, desc = _load_inputs()
    gpu_info = gpu_report()
    print(f"[GPU] {gpu_info}")

    store = CheckpointStore(ckpt_dir)
    runner = CampaignRunner(
        benchmark=bench,
        chunks=chunks,
        corpus_path=corpus_path,
        checkpoint_store=store,
        modes=RETRIEVAL_MODES_SELECTED,
        top_k=TOP_K,
        dense_model_name=DENSE_MODEL_NAME,
        sampling=SAMPLING,
        gpu_info=gpu_info,
        abort_on_degraded_retrieval=ABORT_ON_DEGRADED_RETRIEVAL,
        backend_conflict_policy=BACKEND_CONFLICT_POLICY,
        results_dir=results_dir,
    )

    t0 = time.time()
    index_tags = runner.prepare_indices()
    print(f"[INDEX] built in {time.time()-t0:.1f}s over {len(chunks)} chunks")
    print(f"[INDEX] bm25={index_tags['rag_bm25']!r} dense={index_tags['rag_dense']!r}")
    if is_degraded(index_tags["rag_bm25"]) or is_degraded(index_tags["rag_dense"]):
        raise SystemExit("[FATAL] a retrieval backend is DEGRADED; rows would measure nothing.")

    backends = {}
    for alias in MODEL_ALIASES:
        spec = MODEL_REGISTRY[alias]
        print(f"[BACKEND] resolving {alias} ({spec.hf_id})")
        b = resolve_backend(
            spec,
            preference=BACKEND_PREFERENCE,
            vllm_base_url=VLLM_BASE_URL,
            quantization_declared=VLLM_QUANT_DECLARED,
            sampling=SAMPLING,
            strict_quantization=STRICT_QUANTIZATION,
            load_transformers=True,
        )
        backends[alias] = b
        print(f"[BACKEND] {alias}: backend={b.backend_tag()!r} quant={b.quant_tag()!r}")
        if is_degraded(b.backend_tag()) or is_degraded(b.quant_tag()):
            raise SystemExit(f"[FATAL] backend for {alias} is degraded.")

    # Notebook cell 12's human probe gate is replaced by the memory preflight:
    # one worst-case-prompt generation per model, >=1.5 GiB free at peak.
    from scripts.preflight_memory import preflight_memory
    report = preflight_memory(backends)
    bad = [a for a, r in report.items() if r.get("verdict") != "PASS"]
    if bad:
        raise SystemExit(f"[FATAL] memory preflight below pass bar for {bad}; "
                         "not spending 3 GPU-hours on this envelope.")
    print("[PREFLIGHT] all models PASS - campaign envelope proven; the human "
          "prompt-format spot-check moves to the archived outputs.")

    total = len(MODEL_ALIASES) * len(RETRIEVAL_MODES_SELECTED) * len(bench.questions)
    print("=" * 72)
    print(f"[CAMPAIGN] grid={total} (fresh run, run_id={run_id})")
    print("=" * 72, flush=True)

    progress = ProgressTracker(total=total, every=PROGRESS_EVERY)
    t_start = time.time()
    uploads_ok = True

    def _hub_checkpoint():
        _upload_to_hub(ckpt_dir, f"regrag/outputs/{run_id}/checkpoints",
                       f"checkpoint after model pass {run_id}")

    for i, alias in enumerate(MODEL_ALIASES, 1):
        spec = MODEL_REGISTRY[alias]
        backend = backends[alias]
        print(f"\n{'#' * 72}\n# [{i}/{len(MODEL_ALIASES)}] {alias} ({spec.hf_id})\n{'#' * 72}",
              flush=True)
        t0 = time.time()
        res = runner.run_model(
            spec, backend, progress=progress,
            materialize_every=MATERIALIZE_EVERY, on_error="raise",
            on_materialize=_hub_checkpoint,
        )
        res["elapsed_s"] = round(time.time() - t0, 1)
        print(f"[CAMPAIGN] {alias}: wrote {res['rows_written']}, "
              f"skipped {res['rows_skipped_cached']}, {res['elapsed_s']}s", flush=True)
        backend.close()
        uploads_ok &= _upload_to_hub(results_dir, f"regrag/outputs/{run_id}/results",
                                     f"results after {alias} {run_id}")
        uploads_ok &= _upload_to_hub(ckpt_dir, f"regrag/outputs/{run_id}/checkpoints",
                                     f"checkpoint after {alias} {run_id}")

    runner.materialize(results_dir)
    write_manifest(os.path.join(ckpt_dir, "run_manifest.json"), {
        "run_id": run_id,
        "grid": validate_matrix(MODEL_ALIASES, RETRIEVAL_MODES_SELECTED, len(bench.questions)),
        "rows_in_checkpoint": len(store),
        "corpus": desc["corpus_desc"] if "corpus_desc" in desc else desc["corpus_path"],
        "gpu_info": gpu_info,
        "elapsed_s": round(time.time() - t_start, 1),
        "sampling": SAMPLING.as_dict(),
    })
    uploads_ok &= _upload_to_hub(results_dir, f"regrag/outputs/{run_id}/results",
                                 f"final results {run_id}")
    uploads_ok &= _upload_to_hub(ckpt_dir, f"regrag/outputs/{run_id}/checkpoints",
                                 f"final checkpoint {run_id}")

    print("\n" + "=" * 72)
    print(f"[CAMPAIGN] DONE in {(time.time()-t_start)/60:.1f} min; "
          f"rows={len(store)}/{total}; hub_uploads={'OK' if uploads_ok else 'PARTIAL'}")
    print("=" * 72)
    if len(store) < total:
        print(f"[CAMPAIGN][WARNING] {total - len(store)} cells missing - "
              "re-run the driver; the checkpoint resumes.")
        return 3
    if not uploads_ok:
        print("[CAMPAIGN][WARNING] some hub uploads failed - re-run "
              "scripts/pod_fetch_campaign.sh later may miss files; push manually.")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
