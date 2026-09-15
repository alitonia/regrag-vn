#!/usr/bin/env python3
"""CPU-only dry run of the Colab harness against the REAL benchmark data.

This exists to de-risk the 2026-09-12 Colab session. It exercises exactly the
code path the notebook runs - real CSV, real chunks, real BM25 index, real cache
keys, real checkpoint/resume, real materialisation, real provenance audit - with
a **fake generation backend** standing in for the models.

It therefore proves everything except the two things that need a GPU:
  * that a model loads in 4-bit and does not OOM;
  * that a model's chat template produces fluent Vietnamese (the sanity probe,
    which is why the probe is a gated cell in the notebook).

What it does NOT do, deliberately:
  * no model download, no torch inference, no network;
  * no `rag_dense` - that would download BGE-M3. Dense is covered by the
    unit tests and by `DenseIndex`'s own provenance guard.

    .venv/bin/python notebooks/dry_run_harness.py

Exit code is 0 only if every stage passes.
"""

from __future__ import annotations

import io
import json
import os
import shutil
import sys
import tempfile

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from regrag.generation.backends import Generation, GenerationBackend, RenderedPrompt
from regrag.generation.benchmark import load_benchmark
from regrag.generation.campaign import CampaignRunner, build_manifest
from regrag.generation.checkpoint import (
    CheckpointStore,
    ProgressTracker,
    restore_from_drive,
    split_record,
    sync_to_drive,
    write_manifest,
)
from regrag.generation.config import (
    DEFAULT_MODEL_ALIASES,
    MODEL_REGISTRY,
    SamplingConfig,
    select_models,
    validate_matrix,
)
from regrag.generation.corpus_io import describe_corpus, load_chunks
from regrag.generation.sanity import format_probe_report, probe_gate, run_probe
from regrag.models import GenerationResult
from regrag.provenance import ProvenanceError, assert_publishable, is_degraded
from regrag.storage.file_repo import FileResultRepository

GOLD_CSV = os.path.join(REPO_ROOT, "data", "gold", "bank_qa_data.csv")
TIER1 = os.path.join(REPO_ROOT, "data", "processed_chunks", "tier1_chunks.json")
TIER2 = os.path.join(REPO_ROOT, "data", "processed_chunks", "corpus_chunks.json")

# rag_dense is excluded: it would download BGE-M3. See module docstring.
DRY_RUN_MODES = ("closed_book", "rag_bm25")

FAKE_ANSWER = (
    "Theo quy định hiện hành, cá nhân từ đủ 18 tuổi trở lên có năng lực hành vi "
    "dân sự đầy đủ được mở tài khoản thanh toán tại ngân hàng thương mại và phải "
    "xuất trình giấy tờ tùy thân hợp lệ. Căn cứ pháp lý: Điều 5 Thông tư "
    "17/2024/TT-NHNN."
)
FAKE_REFUSAL = "Không có trong kho văn bản"


class DryRunBackend(GenerationBackend):
    """Stands in for a model. Refuses on unanswerable probes, answers otherwise."""

    kind = "dry-run-fake"

    def __init__(self, alias: str, fail_after: int | None = None,
                 probe_questions: set[str] | None = None) -> None:
        self.alias = alias
        self.fail_after = fail_after
        # The real probe questions, so the fake model refuses exactly the rows a
        # well-behaved model should refuse. Without this the abstention path
        # (RQ2) would never be exercised and `abstained` would stay untested.
        self.probe_questions = set(probe_questions or ())
        self.calls = 0
        self.sampling = SamplingConfig()

    def backend_tag(self) -> str:
        return f"dryrun-fake:{self.alias}"

    def quant_tag(self) -> str:
        return "dryrun:not-a-real-model"

    def render_prompt(self, messages):
        text = json.dumps(list(messages), ensure_ascii=False)
        return RenderedPrompt(text=text, exact=True, applied_by="dry-run",
                              messages=[dict(m) for m in messages])

    def generate(self, messages) -> Generation:
        self.calls += 1
        if self.fail_after is not None and self.calls > self.fail_after:
            raise ProvenanceError("simulated Colab session death")
        content = messages[-1]["content"]
        is_probe = any(pq and pq in content for pq in self.probe_questions)
        text = FAKE_REFUSAL if is_probe else FAKE_ANSWER
        return Generation(text=text, raw=text, latency_s=0.0,
                          finish_reason="stop", usage={"completion_tokens": 40})

    def describe(self):
        return {
            "kind": self.kind,
            "generation_backend": self.backend_tag(),
            "quant_config": self.quant_tag(),
            "served_alias": self.alias,
            "template_applied_by": "dry-run",
            "prompt_exact": True,
        }


def stage(n: int, title: str) -> None:
    print(f"\n{'=' * 74}\nSTAGE {n} — {title}\n{'=' * 74}", flush=True)


def main() -> int:
    failures: list[str] = []

    def check(label: str, ok: bool, detail: str = "") -> None:
        mark = "PASS" if ok else "FAIL"
        print(f"  [{mark}] {label}{(' — ' + detail) if detail else ''}", flush=True)
        if not ok:
            failures.append(label)

    tmp = tempfile.mkdtemp(prefix="regrag_dryrun_")
    try:
        ckpt_dir = os.path.join(tmp, "checkpoints")
        results_dir = os.path.join(tmp, "results")
        drive_dir = os.path.join(tmp, "drive")
        os.makedirs(drive_dir, exist_ok=True)

        # ---------------- 1. inputs -----------------------------------------
        stage(1, "Load the real benchmark and corpus")
        bench = load_benchmark(GOLD_CSV, warn_stream=sys.stderr)
        check("88 questions loaded", len(bench.questions) == 88, f"got {len(bench.questions)}")
        check("64 answerable", len(bench.answerable) == 64, f"got {len(bench.answerable)}")
        check("24 unanswerable probes", len(bench.probes) == 24, f"got {len(bench.probes)}")
        check("question ids unique",
              len({q.id for q in bench.questions}) == len(bench.questions))

        corpus_path = TIER2 if os.path.exists(TIER2) else TIER1
        try:
            chunks = load_chunks(corpus_path)
        except ProvenanceError as exc:
            # corpus_chunks.json is empty until the instruments are ingested.
            print(f"  [INFO] {corpus_path} rejected: {str(exc)[:120]}")
            corpus_path, chunks = TIER1, load_chunks(TIER1)
        desc = describe_corpus(chunks, corpus_path)
        check("corpus loaded", len(chunks) > 0, f"{len(chunks)} chunks")
        print(f"  [INFO] corpus_source={desc['corpus_source']} "
              f"publishable={desc['publishable_corpus']} hash={desc['corpus_hash'][:12]}")

        specs = select_models(DEFAULT_MODEL_ALIASES)
        grid = validate_matrix(DEFAULT_MODEL_ALIASES, DRY_RUN_MODES, len(bench.questions))
        expected = grid["cells"]
        print(f"  [INFO] dry-run grid = {grid} (full campaign would be "
              f"{validate_matrix(DEFAULT_MODEL_ALIASES, ('closed_book','rag_bm25','rag_dense'), len(bench.questions))['cells']} cells)")
        check("all selected models are peers", all(s.is_peer for s in specs))

        # ---------------- 2. sanity probe ------------------------------------
        stage(2, "Per-model sanity probe (fake model, real prompt path)")
        from regrag.generation.sanity import PROBE_QUESTIONS
        sanity_probes = {p["question"] for p in PROBE_QUESTIONS
                         if "token" in p["question"]}
        probe_backend = DryRunBackend("probe", probe_questions=sanity_probes)
        reports = run_probe(probe_backend, mode="closed_book")
        print(format_probe_report("probe (dry-run)", reports)[:1600])
        probe_gate(reports)
        check("probe ran 3 fixed questions", probe_backend.calls == 3,
              f"got {probe_backend.calls}")
        check("probe gate passed", all(r.ok for r in reports))
        check("the out-of-scope probe question was refused",
              [r for r in reports if r.probe_id.startswith("PROBE-3")][0].abstained)
        check("probe recorded the backend tag",
              all(r.backend_tag == "dryrun-fake:probe" for r in reports))

        # The real CSV probe questions, so the fake model refuses exactly the
        # rows a well-behaved model must refuse. This is what exercises RQ2.
        probe_qs = {q.question for q in bench.probes}

        # ---------------- 3. campaign, interrupted ---------------------------
        stage(3, "Campaign, killed mid-grid, then resumed")
        store = CheckpointStore(ckpt_dir)
        runner = CampaignRunner(
            benchmark=bench, chunks=chunks, corpus_path=corpus_path,
            checkpoint_store=store, modes=DRY_RUN_MODES, top_k=3,
            sampling=SamplingConfig(), results_dir=results_dir,
        )
        progress = ProgressTracker(total=expected, every=200, stream=sys.stdout)

        dying = DryRunBackend("qwen-7b", fail_after=100, probe_questions=probe_qs)
        try:
            runner.run_model(MODEL_REGISTRY["qwen-7b"], dying, progress=progress,
                             materialize_every=25)
            check("interrupted run raised", False, "expected ProvenanceError")
        except ProvenanceError as exc:
            check("interrupted run raised loudly", "session death" in str(exc))
        rows_after_crash = len(store)
        check("rows survived the crash", rows_after_crash == 100,
              f"got {rows_after_crash}")
        check("the 101st call raised", dying.calls == 101, f"got {dying.calls}")

        sync_to_drive(ckpt_dir, drive_dir, extra_dirs=[(results_dir, "results")])

        # ---------------- 4. resume in a "new session" -----------------------
        stage(4, "New session: restore from Drive and resume")
        local2 = os.path.join(tmp, "restored")
        rep = restore_from_drive(drive_dir, local2)
        check("Drive restore found the checkpoint", len(rep["restored"]) >= 1,
              str([r["name"] for r in rep["restored"]]))
        store2 = CheckpointStore(local2)
        check("restored row count matches", len(store2) == rows_after_crash,
              f"got {len(store2)} want {rows_after_crash}")

        runner2 = CampaignRunner(
            benchmark=bench, chunks=chunks, corpus_path=corpus_path,
            checkpoint_store=store2, modes=DRY_RUN_MODES, top_k=3,
            sampling=SamplingConfig(), results_dir=results_dir,
        )
        # A fresh tracker per session, exactly as notebook cell 13 does: reusing
        # one across sessions would report processed > total.
        progress2 = ProgressTracker(total=expected, every=200, stream=sys.stdout)
        total_calls = 0
        run_results = []
        for alias in DEFAULT_MODEL_ALIASES:
            b = DryRunBackend(alias, probe_questions=probe_qs)
            res = runner2.run_model(MODEL_REGISTRY[alias], b, progress=progress2,
                                    materialize_every=25)
            run_results.append(res)
            total_calls += b.calls
        check("resumed run completed the grid", len(store2) == expected,
              f"{len(store2)}/{expected}")
        check("cached cells were NOT regenerated",
              total_calls == expected - rows_after_crash,
              f"{total_calls} calls, expected {expected - rows_after_crash}")
        check("no duplicate cache keys",
              len({r["cache_key"] for r in store2.records()}) == expected)

        # ---------------- 5. idempotent re-run -------------------------------
        stage(5, "Re-run the whole campaign (must be a no-op)")
        store3 = CheckpointStore(local2)
        runner3 = CampaignRunner(
            benchmark=bench, chunks=chunks, corpus_path=corpus_path,
            checkpoint_store=store3, modes=DRY_RUN_MODES, top_k=3,
            sampling=SamplingConfig(), results_dir=results_dir,
        )
        calls = 0
        for alias in DEFAULT_MODEL_ALIASES:
            b = DryRunBackend(alias)
            runner3.run_model(MODEL_REGISTRY[alias], b, materialize_every=25)
            calls += b.calls
        check("re-run generated nothing", calls == 0, f"got {calls} calls")
        check("row count unchanged", len(store3) == expected)

        # ---------------- 6. repo-format output ------------------------------
        stage(6, "Materialise into the repo's record format")
        info = store3.materialize(results_dir)
        check("generations.json written", os.path.exists(info["generations_json"]))
        check("row count in repo format", info["rows_total"] == expected,
              f"got {info['rows_total']}")
        repo = FileResultRepository(results_dir)  # the plain class
        check("plain FileResultRepository loads it", len(repo.list_generations()) == expected)
        store3.assert_meta_complete(results_dir)
        check("every row has a provenance sidecar entry", True)

        with open(info["generations_json"], encoding="utf-8") as f:
            raw = json.load(f)
        fields = set(GenerationResult.__dataclass_fields__)
        check("generations.json is schema-pure",
              all(set(r.keys()) == fields for r in raw))

        # ---------------- 7. provenance audit --------------------------------
        stage(7, "Provenance audit")
        recs = store3.records()
        check("every row has corpus_source",
              all(r.get("corpus_source") for r in recs))
        check("every row has retriever_backend",
              all(r.get("retriever_backend") for r in recs))
        check("every row has generation_backend",
              all(r.get("generation_backend") for r in recs))
        check("every row has quant_config", all(r.get("quant_config") for r in recs))
        check("no row is silently DEGRADED",
              not any(is_degraded(r["retriever_backend"]) for r in recs))
        closed = [r for r in recs if r["retrieval_mode"] == "closed_book"]
        rag = [r for r in recs if r["retrieval_mode"] == "rag_bm25"]
        check("closed-book self-identifies",
              all(r["retriever_backend"] == "none-closed-book" for r in closed))
        check("BM25 rows carry the real backend",
              all(r["retriever_backend"] == "bm25-rank_bm25+pyvi" for r in rag),
              rag[0]["retriever_backend"] if rag else "")
        check("closed-book retrieved nothing",
              all(r["retrieved_chunk_ids"] == [] for r in closed))
        check("BM25 retrieved top_k=3",
              all(len(r["retrieved_chunk_ids"]) == 3 for r in rag))
        runner3.assert_no_backend_pooling()
        check("backend-pooling guard passes", True)

        probe_rows = [r for r in recs if r.get("is_answerable") is False]
        n_probe_expected = 24 * len(DEFAULT_MODEL_ALIASES) * len(DRY_RUN_MODES)
        check("probe rows are marked unanswerable",
              len(probe_rows) == n_probe_expected,
              f"got {len(probe_rows)}, want {n_probe_expected}")
        refused = sum(1 for r in probe_rows if r.get("abstained"))
        check("RQ2 plumbing: every probe refusal is detected as abstained",
              refused == len(probe_rows), f"{refused}/{len(probe_rows)}")
        answerable_rows = [r for r in recs if r.get("is_answerable") is True]
        false_refusals = sum(1 for r in answerable_rows if r.get("abstained"))
        check("no answerable row is falsely marked abstained",
              false_refusals == 0, f"{false_refusals} false refusals")

        rows = [GenerationResult(**split_record(r)[0]) for r in recs]
        try:
            assert_publishable(rows)
            publishable = True
        except ProvenanceError as exc:
            publishable = False
            print(f"  [INFO] assert_publishable refused (expected on Tier 1): "
                  f"{str(exc)[:160]}")
        if desc["publishable_corpus"]:
            check("Tier 2 rows are publishable", publishable)
        else:
            check("Tier 1 rows are REFUSED by assert_publishable", not publishable)

        # ---------------- 8. manifest ----------------------------------------
        stage(8, "Run manifest")
        manifest = build_manifest(
            run_name="dry_run", benchmark=bench, corpus_description=desc,
            model_specs=specs, modes=list(DRY_RUN_MODES), top_k=3,
            sampling=SamplingConfig(), gpu_info={"device_names": ["dry-run"]},
            backend_descriptions=[DryRunBackend("x").describe()],
            run_results=run_results, progress_summary=progress2.summary(),
            provenance=runner3.provenance_summary(),
            probe_reports=[r.as_dict() for r in reports],
            extra={"assert_publishable_passed": publishable, "dry_run": True},
        )
        write_manifest(os.path.join(ckpt_dir, "run_manifest.json"), manifest)
        check("manifest written",
              os.path.exists(os.path.join(ckpt_dir, "run_manifest.json")))
        check("manifest records the benchmark split",
              manifest["benchmark"]["unanswerable_probes"] == 24)
        check("manifest records the corpus tag",
              manifest["corpus"]["corpus_source"] == desc["corpus_source"])
        check("manifest records expected cell count",
              manifest["grid"]["expected_cells"] == expected)

        print(f"\n  progress summary: {progress2.summary()}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print("\n" + "=" * 74)
    if failures:
        print(f"DRY RUN FAILED — {len(failures)} check(s):")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("DRY RUN PASSED — every stage OK.")
    print("Not covered here (needs a GPU): 4-bit model load, OOM headroom, and")
    print("the real chat-template sanity probe. Those are cells 10-12 on Colab.")
    print("=" * 74)
    return 0


if __name__ == "__main__":
    sys.exit(main())
