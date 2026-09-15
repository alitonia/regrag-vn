"""CPU-only tests for the Colab generation harness.

Constraints honoured by every test here:
  * no GPU required and no CUDA tensor allocated;
  * no model downloaded and no generation run - the vLLM path is exercised
    through an injected fake HTTP transport, and the transformers path is only
    inspected in its unloaded state;
  * no network access at all;
  * runs in seconds on a laptop.

The one test class that touches torch (``TestGpuAssertion``) only calls
``torch.cuda.is_available()`` / ``get_device_properties`` - the same thing the
notebook's assertion cell does. It allocates nothing and runs no inference.

Run with (pytest is not installed and tests/ has no __init__.py):
    .venv/bin/python -m unittest tests.test_harness -v

What is covered, and why it matters
-----------------------------------
The repo's standing rule is that a placeholder must fail loudly or
self-identify and never return a plausible default. So the tests are weighted
towards the failure paths: a degraded backend, a missing dependency, an
unreachable server, an empty corpus, a wrong served model, a quantization load
that silently fell back, and a checkpoint truncated by a killed session. Each
must either raise or stamp a tag - never produce a row that looks clean.
"""

from __future__ import annotations

import io
import json
import os
import shutil
import tempfile
import unittest
from typing import Any, Dict, List, Optional, Sequence

from regrag.generation.abstention import (
    assert_metrics_detects_sentinel,
    detect_abstention,
)
from regrag.generation.backends import (
    Generation,
    GenerationBackend,
    RenderedPrompt,
    TransformersQuantBackend,
    VLLMHttpBackend,
    _evict_to_make_room,
    _max_resident_backends,
    classify_quantization,
    coerce_input_ids,
    gpu_report,
    resolve_backend,
)
from regrag.generation.benchmark import BenchmarkSet, classify_answerability, load_benchmark
from regrag.generation.campaign import CampaignRunner
from regrag.generation.checkpoint import (
    CheckpointStore,
    ProgressTracker,
    split_record,
)
from regrag.generation.config import (
    DEFAULT_MODEL_ALIASES,
    MODEL_REGISTRY,
    RETRIEVAL_MODES,
    UNANSWERABLE_SENTINEL,
    SamplingConfig,
    backend_vllm,
    select_models,
    validate_matrix,
)
from regrag.generation.corpus_io import (
    chunk_from_dict,
    corpus_hash,
    describe_corpus,
    load_chunks,
)
from regrag.generation.sanity import (
    PROBE_QUESTIONS,
    check_output,
    detect_language,
    detect_repetition,
    detect_template_leak,
    format_probe_report,
    output_stats,
    probe_gate,
    run_probe,
)
from regrag.models import GenerationResult, GoldQuestion, LegalChunk, RetrievedResult
from regrag.provenance import (
    CORPUS_TIER1,
    CORPUS_TIER2,
    CORPUS_UNSET,
    ProvenanceError,
    assert_publishable,
    degraded,
    is_degraded,
)
from regrag.storage.file_repo import FileResultRepository

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
GOLD_CSV = os.path.join(REPO_ROOT, "data", "gold", "bank_qa_data.csv")


# --- fakes -------------------------------------------------------------------

FLUENT_VI = (
    "Theo quy định hiện hành, cá nhân từ đủ 18 tuổi trở lên có năng lực hành vi "
    "dân sự đầy đủ được mở tài khoản thanh toán tại ngân hàng thương mại. "
    "Căn cứ pháp lý: Điều 5 Thông tư 17/2024/TT-NHNN."
)


class FakeBackend(GenerationBackend):
    """A backend that returns canned text. Stands in for a model or a server."""

    kind = "fake"

    def __init__(
        self,
        tag: str = "vllm-serve:fake-model",
        quant: str = "vllm-server:fp16",
        text: str = FLUENT_VI,
        fail_after: Optional[int] = None,
        fail_message: str = "simulated Colab session death",
        prompt_exact: bool = False,
    ) -> None:
        self._tag = tag
        self._quant = quant
        self._text = text
        self._fail_after = fail_after
        self._fail_message = fail_message
        self._prompt_exact = prompt_exact
        self.calls = 0
        self.prompts_seen: List[str] = []
        self.sampling = SamplingConfig()

    def backend_tag(self) -> str:
        return self._tag

    def quant_tag(self) -> str:
        return self._quant

    def render_prompt(self, messages):
        text = json.dumps(list(messages), ensure_ascii=False)
        return RenderedPrompt(
            text=text,
            exact=self._prompt_exact,
            applied_by="fake-backend",
            messages=[dict(m) for m in messages],
        )

    def generate(self, messages) -> Generation:
        self.calls += 1
        self.prompts_seen.append(json.dumps(list(messages), ensure_ascii=False))
        if self._fail_after is not None and self.calls > self._fail_after:
            raise ProvenanceError(self._fail_message)
        return Generation(
            text=self._text,
            raw=self._text,
            latency_s=0.01,
            finish_reason="stop",
            usage={"completion_tokens": 10},
        )

    def describe(self) -> Dict[str, Any]:
        return {
            "kind": self.kind,
            "generation_backend": self._tag,
            "quant_config": self._quant,
            "served_alias": "fake-model",
            "template_applied_by": "fake",
            "prompt_exact": self._prompt_exact,
        }


def fake_transport_factory(status: int = 200, body: Optional[Dict[str, Any]] = None,
                           raise_exc: Optional[Exception] = None):
    """An injected HTTP transport. Records calls; never opens a socket."""
    calls: List[Dict[str, Any]] = []

    def transport(method, url, payload, timeout):
        calls.append({"method": method, "url": url, "payload": payload, "timeout": timeout})
        if raise_exc is not None:
            raise raise_exc
        return status, body if body is not None else {}

    transport.calls = calls  # type: ignore[attr-defined]
    return transport


def make_chunks(n: int = 4, corpus_source: str = CORPUS_TIER2) -> List[LegalChunk]:
    return [
        LegalChunk(
            chunk_id=f"C{i:03d}",
            doc_id="17/2024/TT-NHNN",
            doc_title="Thông tư quy định về thanh toán",
            chapter="Chương II",
            article_id=str(i + 1),
            article_title=f"Điều {i+1} mẫu",
            clause_id=None,
            text=f"Điều {i+1}. Quy định mẫu số {i+1} về tài khoản thanh toán và thẻ ngân hàng.",
            corpus_source=corpus_source,
        )
        for i in range(n)
    ]


def make_questions() -> List[GoldQuestion]:
    return [
        GoldQuestion(
            id="Q001",
            question="Điều kiện mở tài khoản thanh toán là gì?",
            is_answerable=True,
            gold_doc_ids=["17/2024/TT-NHNN"],
            reference_answer="Cá nhân từ đủ 18 tuổi.",
            gold_passage="Điều 5. Cá nhân từ đủ 18 tuổi được mở tài khoản thanh toán.",
            category="factual",
        ),
        GoldQuestion(
            id="Q002",
            question="Hạn mức rút tiền mặt bằng thẻ là bao nhiêu?",
            is_answerable=True,
            gold_doc_ids=["17/2024/TT-NHNN"],
            reference_answer="Không quá 100 triệu đồng/tháng.",
            gold_passage="Điều 6. Hạn mức rút tiền mặt không quá 100 triệu đồng.",
            category="factual",
        ),
        GoldQuestion(
            id="Q003",
            question="Quy định về phát hành token chứng khoán hoá bất động sản?",
            is_answerable=False,
            reference_answer=UNANSWERABLE_SENTINEL,
            category="unanswerable",
        ),
    ]


def make_benchmark(questions=None, csv_hash="deadbeef" * 8) -> BenchmarkSet:
    qs = questions if questions is not None else make_questions()
    return BenchmarkSet(questions=qs, csv_path="<synthetic>", csv_hash=csv_hash)


# --- cache keys --------------------------------------------------------------


class TestCacheKeyStability(unittest.TestCase):
    """A partial run must resume exactly, and a CSV revision touching a few
    questions must not invalidate thousands of cached generations."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="regrag_harness_")
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.chunks = make_chunks()
        self.store = CheckpointStore(os.path.join(self.tmp, "ckpt"), warn_stream=io.StringIO())
        self.runner = CampaignRunner(
            benchmark=make_benchmark(),
            chunks=self.chunks,
            corpus_path="<synthetic>",
            checkpoint_store=self.store,
            modes=("closed_book", "rag_bm25"),
            warn_stream=io.StringIO(),
        )

    def test_key_is_stable_across_runner_instances(self):
        """Rebuilding the runner (i.e. restarting the session) reproduces keys."""
        q = self.runner.benchmark.questions[0]
        k1 = self.runner.cache_key_for(q, "qwen-7b", "rag_bm25")

        store2 = CheckpointStore(os.path.join(self.tmp, "ckpt2"), warn_stream=io.StringIO())
        runner2 = CampaignRunner(
            benchmark=make_benchmark(),
            chunks=make_chunks(),
            corpus_path="<synthetic>",
            checkpoint_store=store2,
            modes=("closed_book", "rag_bm25"),
            warn_stream=io.StringIO(),
        )
        k2 = runner2.cache_key_for(runner2.benchmark.questions[0], "qwen-7b", "rag_bm25")
        self.assertEqual(k1, k2)
        self.assertEqual(len(k1), 24, "key length must match scripts/regenerate.cache_key")

    def test_key_separates_model_mode_and_question(self):
        q0, q1 = self.runner.benchmark.questions[0], self.runner.benchmark.questions[1]
        base = self.runner.cache_key_for(q0, "qwen-7b", "rag_bm25")
        self.assertNotEqual(base, self.runner.cache_key_for(q0, "qwen-3b", "rag_bm25"))
        self.assertNotEqual(base, self.runner.cache_key_for(q0, "qwen-7b", "closed_book"))
        self.assertNotEqual(base, self.runner.cache_key_for(q1, "qwen-7b", "rag_bm25"))

    def test_editing_one_question_invalidates_only_that_question(self):
        """The property the 09-10 CSV review depends on."""
        before = {q.id: self.runner.cache_key_for(q, "qwen-7b", "rag_bm25")
                  for q in self.runner.benchmark.questions}

        edited = make_questions()
        edited[1].question = edited[1].question + " (đã sửa câu hỏi)"
        runner2 = CampaignRunner(
            benchmark=make_benchmark(questions=edited, csv_hash="c0ffee" * 8),
            chunks=self.chunks,
            corpus_path="<synthetic>",
            checkpoint_store=self.store,
            modes=("closed_book", "rag_bm25"),
            warn_stream=io.StringIO(),
        )
        after = {q.id: runner2.cache_key_for(q, "qwen-7b", "rag_bm25")
                 for q in runner2.benchmark.questions}

        self.assertNotEqual(before["Q002"], after["Q002"], "edited question must invalidate")
        self.assertEqual(before["Q001"], after["Q001"], "untouched question must survive")
        self.assertEqual(before["Q003"], after["Q003"], "untouched probe must survive")

    def test_corpus_rebuild_invalidates_rag_keys(self):
        before = self.runner.cache_key_for(self.runner.benchmark.questions[0], "qwen-7b", "rag_bm25")
        changed = make_chunks()
        changed[0].text = changed[0].text + " Nội dung đã được sửa đổi hoàn toàn khác."
        runner2 = CampaignRunner(
            benchmark=make_benchmark(),
            chunks=changed,
            corpus_path="<synthetic>",
            checkpoint_store=self.store,
            modes=("closed_book", "rag_bm25"),
            warn_stream=io.StringIO(),
        )
        after = runner2.cache_key_for(runner2.benchmark.questions[0], "qwen-7b", "rag_bm25")
        self.assertNotEqual(before, after)

    def test_planned_keys_cover_the_whole_grid(self):
        planned = self.runner.planned_keys(["qwen-7b", "qwen-3b"])
        self.assertEqual(len(planned), 2)
        # 2 models x 2 modes x 3 questions
        self.assertEqual(sum(len(v) for v in planned.values()), 12)
        self.assertEqual(len({k for v in planned.values() for k in v}), 12, "keys must be unique")


# --- checkpoint / resume -----------------------------------------------------


class TestCheckpointResume(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="regrag_ckpt_")
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.ckpt_dir = os.path.join(self.tmp, "ckpt")

    def _runner(self, store):
        return CampaignRunner(
            benchmark=make_benchmark(),
            chunks=make_chunks(),
            corpus_path="<synthetic>",
            checkpoint_store=store,
            modes=("closed_book", "rag_bm25"),
            results_dir=os.path.join(self.tmp, "results"),
            warn_stream=io.StringIO(),
        )

    def test_resume_after_simulated_session_death(self):
        """Kill the run mid-grid, rebuild everything, finish without redoing work."""
        spec = MODEL_REGISTRY["qwen-7b"]

        store1 = CheckpointStore(self.ckpt_dir, warn_stream=io.StringIO())
        self.assertEqual(len(store1), 0)
        dying = FakeBackend(fail_after=3)
        runner1 = self._runner(store1)
        with self.assertRaises(ProvenanceError):
            runner1.run_model(spec, dying)
        rows_after_crash = len(store1)
        self.assertEqual(rows_after_crash, 3, "exactly the rows produced before the kill survive")
        self.assertEqual(dying.calls, 4, "the 4th call raised")

        # Simulated new Colab session: brand new store object reading the same dir.
        log = io.StringIO()
        store2 = CheckpointStore(self.ckpt_dir, warn_stream=log)
        self.assertEqual(len(store2), rows_after_crash, "resume must see the prior rows")
        self.assertIn("Restored", log.getvalue())

        fresh = FakeBackend()
        runner2 = self._runner(store2)
        result = runner2.run_model(spec, fresh)

        self.assertEqual(result["rows_skipped_cached"], rows_after_crash)
        self.assertEqual(result["rows_written"], 6 - rows_after_crash)
        self.assertEqual(fresh.calls, 6 - rows_after_crash,
                         "cached cells must NOT be regenerated")
        self.assertEqual(len(store2), 6, "grid is 1 model x 2 modes x 3 questions")
        self.assertEqual(len({r["cache_key"] for r in store2.records()}), 6, "no duplicates")

    def test_rerun_is_a_noop(self):
        """Idempotency: re-running a completed campaign generates nothing."""
        spec = MODEL_REGISTRY["qwen-7b"]
        store = CheckpointStore(self.ckpt_dir, warn_stream=io.StringIO())
        runner = self._runner(store)
        b1 = FakeBackend()
        runner.run_model(spec, b1)
        self.assertEqual(b1.calls, 6)

        store2 = CheckpointStore(self.ckpt_dir, warn_stream=io.StringIO())
        runner2 = self._runner(store2)
        b2 = FakeBackend()
        res = runner2.run_model(spec, b2)
        self.assertEqual(b2.calls, 0, "a complete run must not regenerate anything")
        self.assertEqual(res["rows_skipped_cached"], 6)
        self.assertEqual(res["rows_written"], 0)

    def test_truncated_final_line_is_tolerated_and_reported(self):
        """A session killed mid-write must cost one row, not the whole run."""
        store = CheckpointStore(self.ckpt_dir, warn_stream=io.StringIO())
        store.append({
            "cache_key": "k1", "question_id": "Q001", "model_name": "qwen-7b",
            "retrieval_mode": "closed_book", "prompt": "p", "raw_response": "r",
            "answer_text": "a", "corpus_source": CORPUS_TIER2,
            "retriever_backend": "none-closed-book",
        })
        with open(store.path, "a", encoding="utf-8") as f:
            f.write('{"cache_key": "k2", "question_id": "Q002", "answ')  # torn write

        log = io.StringIO()
        store2 = CheckpointStore(self.ckpt_dir, warn_stream=log)
        self.assertEqual(len(store2), 1, "the intact row survives")
        self.assertTrue(store2.has("k1"))
        self.assertFalse(store2.has("k2"))
        self.assertEqual(len(store2.skipped_lines), 1)
        self.assertIn("WARNING", log.getvalue())

    def test_append_refuses_a_record_without_cache_key(self):
        store = CheckpointStore(self.ckpt_dir, warn_stream=io.StringIO())
        with self.assertRaises(ProvenanceError):
            store.append({"question_id": "Q001"})

    def test_append_refuses_a_record_missing_generation_fields(self):
        store = CheckpointStore(self.ckpt_dir, warn_stream=io.StringIO())
        with self.assertRaises(ProvenanceError) as ctx:
            store.append({"cache_key": "k1", "question_id": "Q001"})
        self.assertIn("missing GenerationResult field", str(ctx.exception))

    def test_materialized_rows_are_loadable_by_the_plain_repo(self):
        """storage/ and evaluation/ must consume the output directly.

        FileResultRepository reloads with GenerationResult(**item), so an extra
        key in generations.json would make it unloadable. This proves the
        materialised file stays schema-pure and the extra provenance lives in the
        sidecar instead.
        """
        spec = MODEL_REGISTRY["qwen-7b"]
        store = CheckpointStore(self.ckpt_dir, warn_stream=io.StringIO())
        runner = self._runner(store)
        runner.run_model(spec, FakeBackend())
        results_dir = os.path.join(self.tmp, "results")
        info = store.materialize(results_dir)
        self.assertEqual(info["rows_total"], 6)

        repo = FileResultRepository(results_dir)  # the plain class, not the subclass
        self.assertEqual(len(repo.list_generations()), 6)
        with open(os.path.join(results_dir, "generations.json"), encoding="utf-8") as f:
            raw = json.load(f)
        expected = set(GenerationResult.__dataclass_fields__)
        for row in raw:
            self.assertEqual(set(row.keys()), expected,
                             "generations.json must carry exactly GenerationResult's fields")

        store.assert_meta_complete(results_dir)
        with open(os.path.join(results_dir, "generations_meta.jsonl"), encoding="utf-8") as f:
            metas = [json.loads(line) for line in f if line.strip()]
        self.assertEqual(len(metas), 6)
        for m in metas:
            self.assertIn("generation_backend", m)
            self.assertIn("quant_config", m)

    def test_materialize_is_idempotent(self):
        spec = MODEL_REGISTRY["qwen-7b"]
        store = CheckpointStore(self.ckpt_dir, warn_stream=io.StringIO())
        runner = self._runner(store)
        runner.run_model(spec, FakeBackend())
        results_dir = os.path.join(self.tmp, "results")
        store.materialize(results_dir)
        second = store.materialize(results_dir)
        self.assertEqual(second["rows_added"], 0, "re-materialising must not duplicate rows")
        self.assertEqual(second["rows_total"], 6)

    def test_assert_meta_complete_catches_orphan_rows(self):
        spec = MODEL_REGISTRY["qwen-7b"]
        store = CheckpointStore(self.ckpt_dir, warn_stream=io.StringIO())
        runner = self._runner(store)
        runner.run_model(spec, FakeBackend())
        results_dir = os.path.join(self.tmp, "results")
        store.materialize(results_dir)
        os.remove(os.path.join(results_dir, "generations_meta.jsonl"))
        with self.assertRaises(ProvenanceError) as ctx:
            store.assert_meta_complete(results_dir)
        self.assertIn("generation_backend", str(ctx.exception))

    def test_split_record_routes_fields_and_reports_unknown(self):
        rec = {
            "cache_key": "k", "question_id": "Q001", "model_name": "m",
            "retrieval_mode": "closed_book", "prompt": "p", "raw_response": "r",
            "answer_text": "a", "generation_backend": "vllm-serve:x",
            "quant_config": "vllm-server:fp16", "made_up_field": 1,
        }
        gen, meta, unknown = split_record(rec)
        self.assertEqual(unknown, ["made_up_field"])
        self.assertIn("answer_text", gen)
        self.assertNotIn("generation_backend", gen)
        self.assertEqual(meta["generation_backend"], "vllm-serve:x")


# --- provenance tagging ------------------------------------------------------


class TestProvenanceEnforcement(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="regrag_prov_")
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.store = CheckpointStore(os.path.join(self.tmp, "ckpt"), warn_stream=io.StringIO())

    def _runner(self, store=None, **kw):
        args = dict(
            benchmark=make_benchmark(),
            chunks=make_chunks(),
            corpus_path="<synthetic>",
            # NOTE: `store or self.store` would be WRONG here - CheckpointStore
            # defines __len__, so an empty store is falsy and would silently be
            # replaced by the shared one.
            checkpoint_store=store if store is not None else self.store,
            modes=("closed_book", "rag_bm25"),
            results_dir=os.path.join(self.tmp, "results"),
            warn_stream=io.StringIO(),
        )
        args.update(kw)
        return CampaignRunner(**args)

    def test_append_refuses_a_row_without_provenance(self):
        """The specific hole: a defaulted corpus_source reads like a real value."""
        store = CheckpointStore(os.path.join(self.tmp, "ckpt_p"), warn_stream=io.StringIO())
        with self.assertRaises(ProvenanceError) as ctx:
            store.append({
                "cache_key": "k", "question_id": "Q001", "model_name": "m",
                "retrieval_mode": "closed_book", "prompt": "p", "raw_response": "r",
                "answer_text": "a",
            })
        self.assertIn("corpus_source", str(ctx.exception))
        self.assertEqual(len(store), 0)

    def test_every_row_carries_corpus_and_backend_tags(self):
        spec = MODEL_REGISTRY["qwen-7b"]
        self._runner().run_model(spec, FakeBackend())
        recs = self.store.records()
        self.assertEqual(len(recs), 6)
        for r in recs:
            self.assertEqual(r["corpus_source"], CORPUS_TIER2)
            self.assertIn("generation_backend", r)
            self.assertIn("quant_config", r)
            self.assertEqual(r["model_set_role"], "peer")
            self.assertEqual(r["hf_model_id"], "Qwen/Qwen2.5-7B-Instruct")
            if r["retrieval_mode"] == "closed_book":
                self.assertEqual(r["retriever_backend"], "none-closed-book",
                                 "closed-book must self-identify, not borrow a backend name")
            else:
                self.assertEqual(r["retriever_backend"], "bm25-rank_bm25+pyvi")
            self.assertFalse(is_degraded(r["retriever_backend"]))
            self.assertFalse(is_degraded(r["generation_backend"]))

    def test_probe_rows_are_recorded_as_unanswerable(self):
        """RQ2 depends on the probe rows being identifiable in the output."""
        self._runner(modes=("closed_book",)).run_model(MODEL_REGISTRY["qwen-7b"], FakeBackend())
        recs = self.store.records()
        probe_rows = [r for r in recs if r["question_id"] == "Q003"]
        self.assertEqual(len(probe_rows), 1)
        self.assertFalse(probe_rows[0]["is_answerable"])
        answerable_rows = [r for r in recs if r["question_id"] != "Q003"]
        self.assertTrue(all(r["is_answerable"] for r in answerable_rows))

    def test_closed_book_and_rag_rows_are_distinguishable(self):
        self._runner().run_model(MODEL_REGISTRY["qwen-7b"], FakeBackend())
        recs = self.store.records()
        closed = [r for r in recs if r["retrieval_mode"] == "closed_book"]
        rag = [r for r in recs if r["retrieval_mode"] == "rag_bm25"]
        self.assertTrue(all(r["retrieved_chunk_ids"] == [] for r in closed))
        self.assertTrue(all(len(r["retrieved_chunk_ids"]) == 3 for r in rag),
                        "top_k=3 must be recorded per row")

    def test_all_models_see_identical_context(self):
        """RQ1/RQ3 comparability: retrieval is shared, not recomputed per model."""
        runner = self._runner()
        runner.run_model(MODEL_REGISTRY["qwen-7b"], FakeBackend())
        store2 = CheckpointStore(os.path.join(self.tmp, "ckpt2"), warn_stream=io.StringIO())
        runner2 = self._runner(store=store2)
        runner2.run_model(MODEL_REGISTRY["qwen-3b"], FakeBackend())

        a = {(r["question_id"], r["retrieval_mode"]): r["retrieved_chunk_ids"]
             for r in runner.store.records() if r["retrieval_mode"] == "rag_bm25"}
        b = {(r["question_id"], r["retrieval_mode"]): r["retrieved_chunk_ids"]
             for r in store2.records() if r["retrieval_mode"] == "rag_bm25"}
        self.assertEqual(a, b)

    def test_degraded_generation_backend_is_refused_not_written(self):
        """A degraded backend must stop the run rather than emit a clean row."""
        runner = self._runner()
        bad = FakeBackend(tag=degraded("simulated-precision-loss"))
        with self.assertRaises(ProvenanceError) as ctx:
            runner.run_model(MODEL_REGISTRY["qwen-7b"], bad)
        self.assertIn("DEGRADED", str(ctx.exception))
        self.assertEqual(len(self.store), 0, "no row may be written")

    def test_backend_change_on_resume_is_refused_by_default(self):
        """Same cache key, different backend: rows would be silently pooled."""
        self._runner().run_model(MODEL_REGISTRY["qwen-7b"], FakeBackend(tag="vllm-serve:qwen-7b"))
        self.assertEqual(len(self.store), 6)

        store2 = CheckpointStore(os.path.join(self.tmp, "ckpt"), warn_stream=io.StringIO())
        runner2 = self._runner(store=store2)
        other = FakeBackend(tag="transformers-4bit", prompt_exact=True)
        with self.assertRaises(ProvenanceError) as ctx:
            runner2.run_model(MODEL_REGISTRY["qwen-7b"], other)
        self.assertIn("must not be pooled", str(ctx.exception))

    def test_backend_conflict_policy_reuse_is_allowed_explicitly(self):
        self._runner().run_model(MODEL_REGISTRY["qwen-7b"], FakeBackend(tag="vllm-serve:qwen-7b"))
        store2 = CheckpointStore(os.path.join(self.tmp, "ckpt"), warn_stream=io.StringIO())
        runner2 = self._runner(store=store2, backend_conflict_policy="reuse")
        res = runner2.run_model(MODEL_REGISTRY["qwen-7b"], FakeBackend(tag="transformers-4bit"))
        self.assertEqual(res["rows_skipped_cached"], 6)
        self.assertEqual(res["rows_written"], 0)

    def test_invalid_backend_conflict_policy_is_rejected(self):
        with self.assertRaises(ValueError):
            self._runner(backend_conflict_policy="silently-mix")

    def test_pooling_guard_detects_mixed_backends_for_one_model(self):
        self._runner().run_model(MODEL_REGISTRY["qwen-7b"], FakeBackend(tag="vllm-serve:qwen-7b"))
        store2 = CheckpointStore(os.path.join(self.tmp, "ckpt"), warn_stream=io.StringIO())
        runner2 = self._runner(store=store2)
        # Force the mixed state a mid-run fallback would produce.
        for rec in list(store2.records())[:2]:
            rec2 = dict(rec)
            rec2["cache_key"] = rec["cache_key"] + "_x"
            rec2["generation_backend"] = "transformers-4bit"
            store2.append(rec2)
        with self.assertRaises(ProvenanceError) as ctx:
            runner2.assert_no_backend_pooling()
        self.assertIn("different generation backends", str(ctx.exception))

    def test_pooling_guard_passes_when_each_model_has_one_backend(self):
        runner = self._runner()
        runner.run_model(MODEL_REGISTRY["qwen-7b"], FakeBackend(tag="vllm-serve:qwen-7b"))
        per_model = runner.assert_no_backend_pooling()
        self.assertEqual(per_model, {"qwen-7b": {"vllm-serve:qwen-7b": 6}})

    def test_tier1_corpus_rows_are_not_publishable(self):
        """The only corpus that exists today is Tier 1 - it must never look publishable."""
        store = CheckpointStore(os.path.join(self.tmp, "ckpt_t1"), warn_stream=io.StringIO())
        runner = CampaignRunner(
            benchmark=make_benchmark(),
            chunks=make_chunks(corpus_source=CORPUS_TIER1),
            corpus_path="<synthetic>",
            checkpoint_store=store,
            modes=("closed_book",),
            results_dir=os.path.join(self.tmp, "results"),
            warn_stream=io.StringIO(),
        )
        runner.run_model(MODEL_REGISTRY["qwen-7b"], FakeBackend())
        self.assertFalse(runner.provenance_summary()["publishable_corpus"])
        rows = [GenerationResult(**split_record(r)[0]) for r in store.records()]
        with self.assertRaises(ProvenanceError):
            assert_publishable(rows)

    def test_tier2_corpus_rows_are_publishable(self):
        runner = self._runner(modes=("closed_book",))
        runner.run_model(MODEL_REGISTRY["qwen-7b"], FakeBackend())
        self.assertTrue(runner.provenance_summary()["publishable_corpus"])
        rows = [GenerationResult(**split_record(r)[0]) for r in self.store.records()]
        self.assertEqual(assert_publishable(rows), [])

    def test_empty_corpus_refuses_rag_modes(self):
        """The live state of corpus_chunks.json - must not silently become closed-book."""
        with self.assertRaises(ProvenanceError) as ctx:
            CampaignRunner(
                benchmark=make_benchmark(),
                chunks=[],
                corpus_path="<empty>",
                checkpoint_store=self.store,
                modes=("rag_bm25",),
                warn_stream=io.StringIO(),
            )
        self.assertIn("empty corpus", str(ctx.exception))

    def test_empty_corpus_closed_book_warns_and_tags_unset(self):
        log = io.StringIO()
        runner = CampaignRunner(
            benchmark=make_benchmark(),
            chunks=[],
            corpus_path="<empty>",
            checkpoint_store=self.store,
            modes=("closed_book",),
            results_dir=os.path.join(self.tmp, "results"),
            warn_stream=log,
        )
        runner.run_model(MODEL_REGISTRY["qwen-7b"], FakeBackend())
        self.assertIn("WARNING", log.getvalue())
        for r in self.store.records():
            self.assertEqual(r["corpus_source"], CORPUS_UNSET)
        runner2 = CampaignRunner(
            benchmark=make_benchmark(), chunks=[], corpus_path="<empty>",
            checkpoint_store=self.store, modes=("closed_book",),
            results_dir=os.path.join(self.tmp, "results"), warn_stream=io.StringIO(),
        )
        self.assertFalse(runner2.provenance_summary()["publishable_corpus"])

    def test_unknown_mode_is_rejected(self):
        with self.assertRaises(ValueError):
            self._runner(modes=("rag_magic",))

    def test_prepare_indices_reports_the_real_bm25_backend(self):
        """The notebook's index cell uses this instead of private methods."""
        runner = self._runner(modes=("closed_book", "rag_bm25"))
        tags = runner.prepare_indices()
        self.assertEqual(tags["rag_bm25"], "bm25-rank_bm25+pyvi")
        self.assertEqual(tags["rag_dense"], "not-built",
                         "dense must NOT be built when rag_dense is not selected")
        self.assertFalse(runner._dense_built)

    def test_prepare_indices_on_empty_corpus_reports_unset(self):
        store = CheckpointStore(os.path.join(self.tmp, "ckpt_e"), warn_stream=io.StringIO())
        runner = CampaignRunner(
            benchmark=make_benchmark(), chunks=[], corpus_path="<empty>",
            checkpoint_store=store, modes=("closed_book",),
            results_dir=os.path.join(self.tmp, "results"), warn_stream=io.StringIO(),
        )
        self.assertEqual(runner.prepare_indices()["rag_bm25"], CORPUS_UNSET)

    def test_on_materialize_hook_fires(self):
        """The notebook uses this hook to push the checkpoint up to Drive."""
        fired: List[int] = []
        runner = self._runner(modes=("closed_book",))
        runner.run_model(MODEL_REGISTRY["qwen-7b"], FakeBackend(),
                         materialize_every=2,
                         on_materialize=lambda: fired.append(len(self.store)))
        self.assertTrue(fired, "the hook must fire at least once")
        self.assertTrue(all(n > 0 for n in fired))

    def test_unrecorded_quantization_warns_once_per_model_not_per_row(self):
        """792 identical warnings would bury the progress output."""
        log = io.StringIO()
        store = CheckpointStore(os.path.join(self.tmp, "ckpt_q"), warn_stream=log)
        runner = CampaignRunner(
            benchmark=make_benchmark(), chunks=make_chunks(), corpus_path="<syn>",
            checkpoint_store=store, modes=("closed_book", "rag_bm25"),
            results_dir=os.path.join(self.tmp, "results"), warn_stream=log,
        )
        runner.run_model(MODEL_REGISTRY["qwen-7b"],
                         FakeBackend(quant="UNRECORDED:vllm-server-managed"))
        self.assertEqual(len(store), 6)
        self.assertEqual(log.getvalue().count("UNVERIFIED"), 1,
                         "one warning per model, not one per row")
        # Every row still records the unverified tag - the warning is cosmetic,
        # the record is not.
        self.assertTrue(all(r["quant_config"].startswith("UNRECORDED")
                            for r in store.records()))

    def test_recorded_quantization_does_not_warn(self):
        log = io.StringIO()
        store = CheckpointStore(os.path.join(self.tmp, "ckpt_q2"), warn_stream=log)
        runner = CampaignRunner(
            benchmark=make_benchmark(), chunks=make_chunks(), corpus_path="<syn>",
            checkpoint_store=store, modes=("closed_book",),
            results_dir=os.path.join(self.tmp, "results"), warn_stream=log,
        )
        runner.run_model(MODEL_REGISTRY["qwen-7b"], FakeBackend(quant="bnb-4bit-nf4+float16"))
        self.assertNotIn("UNVERIFIED", log.getvalue())


# --- quantization ------------------------------------------------------------


class TestQuantizationClassification(unittest.TestCase):
    """`load_in_4bit=True` is a request, not evidence. These pin the verdicts."""

    def test_verified_4bit_from_linear_census(self):
        """The real T4 signature: every Linear swapped to Linear4bit while the
        embedding table (first parameter) legitimately stays bfloat16."""
        v = classify_quantization(
            True, True, "torch.bfloat16", linear_census={"Linear4bit": 204}
        )
        self.assertTrue(v.verified_4bit)
        self.assertFalse(v.degraded)
        self.assertEqual(v.quant_config, "bnb-4bit-nf4+float16")

    def test_verified_4bit_tolerates_unconverted_lm_head(self):
        v = classify_quantization(
            True, True, "torch.bfloat16", linear_census={"Linear4bit": 203, "Linear": 1}
        )
        self.assertTrue(v.verified_4bit)
        self.assertFalse(v.degraded)

    def test_requested_4bit_but_not_applied(self):
        """bitsandbytes missing -> transformers loads fp16 and says nothing."""
        v = classify_quantization(
            True, False, "torch.float16", linear_census={"Linear": 204}
        )
        self.assertFalse(v.verified_4bit)
        self.assertTrue(v.degraded)
        self.assertTrue(v.quant_config.startswith("DEGRADED:"))
        self.assertIn("float16", v.quant_config)
        self.assertTrue(is_degraded(v.quant_config))

    def test_requested_4bit_but_nothing_converted(self):
        """Config recorded but zero Linear modules converted: the true
        fallback case the strict gate must still stop."""
        v = classify_quantization(
            True, True, "torch.bfloat16", linear_census={"Linear": 204}
        )
        self.assertTrue(v.degraded)
        self.assertFalse(v.verified_4bit)
        self.assertIn("fell-back-to-bfloat16", v.quant_config)

    def test_partial_4bit_is_degraded(self):
        v = classify_quantization(
            True, True, "torch.bfloat16",
            linear_census={"Linear4bit": 100, "Linear": 104},
        )
        self.assertTrue(v.degraded)
        self.assertFalse(v.verified_4bit)
        self.assertIn("partial-4bit", v.quant_config)

    def test_census_without_linear_modules_is_degraded(self):
        v = classify_quantization(True, True, "torch.bfloat16", linear_census={})
        self.assertTrue(v.degraded)
        self.assertFalse(v.verified_4bit)

    def test_legacy_no_census_probe_still_supported(self):
        """Callers without a census keep the old dtype-biased verdicts."""
        v = classify_quantization(True, True, "torch.uint8")
        self.assertTrue(v.verified_4bit)
        self.assertEqual(v.quant_config, "bnb-4bit-nf4+float16")
        v = classify_quantization(True, True, "torch.float32")
        self.assertTrue(v.degraded)
        self.assertIn("fell-back-to-float32", v.quant_config)
        v = classify_quantization(True, True, "unknown")
        self.assertTrue(v.degraded)
        self.assertFalse(v.verified_4bit)
        v = classify_quantization(True, False, "torch.float16")
        self.assertTrue(v.degraded)
        self.assertTrue(is_degraded(v.quant_config))

    def test_full_precision_is_honest_not_degraded(self):
        v = classify_quantization(False, False, "torch.bfloat16")
        self.assertFalse(v.degraded)
        self.assertFalse(v.verified_4bit)
        self.assertEqual(v.quant_config, "full-bfloat16")

    def test_unloaded_backend_self_identifies(self):
        """Never claim a precision for a model that has not been loaded."""
        b = TransformersQuantBackend(hf_id="Qwen/Qwen2.5-7B-Instruct", warn_stream=io.StringIO())
        self.assertEqual(b.backend_tag(), "transformers:not-loaded")
        self.assertEqual(b.quant_tag(), "UNRECORDED:model-not-loaded")
        d = b.describe()
        self.assertEqual(d["quantization_reason"], "not loaded")
        self.assertTrue(d["requested"]["load_in_4bit"])

    def test_unloaded_backend_refuses_to_generate(self):
        b = TransformersQuantBackend(hf_id="x/y", warn_stream=io.StringIO())
        with self.assertRaises(ProvenanceError):
            b.generate([{"role": "user", "content": "hi"}])
        with self.assertRaises(ProvenanceError):
            b.render_prompt([{"role": "user", "content": "hi"}])


class TestCoerceInputIds(unittest.TestCase):
    """apply_chat_template(tokenize=True) is a bare tensor on some transformers
    versions and a BatchEncoding on 5.x; generate() needs the ids tensor."""

    def test_bare_tensor_passes_through(self):
        tensor = object()
        self.assertIs(coerce_input_ids(tensor), tensor)

    def test_batch_encoding_is_unwrapped(self):
        tensor = object()
        encoding = {"input_ids": tensor}  # dict stand-in: has .keys/__getitem__
        self.assertIs(coerce_input_ids(encoding), tensor)

    def test_batch_encoding_without_input_ids_fails_loud(self):
        encoding = {"attention_mask": object()}
        with self.assertRaises(ProvenanceError):
            coerce_input_ids(encoding)


class TestGpuResidency(unittest.TestCase):
    """One quantized model resident at a time: a new load FIFO-evicts the
    oldest (the 2026-09-12 OOM at row 81 had all three peers resident)."""

    class _FakeResident:
        def __init__(self, hf_id):
            self.hf_id = hf_id
            self.evictions = 0
            self.model = object()

        def _evict_model(self):
            self.evictions += 1
            self.model = None

    def test_cap_one_evicts_fifo(self):
        a = self._FakeResident("a")
        b = self._FakeResident("b")
        reg = [a, b]
        _evict_to_make_room(reg, 1, "c", io.StringIO())
        self.assertEqual(reg, [])
        self.assertEqual(a.evictions, 1)
        self.assertEqual(b.evictions, 1)
        self.assertIsNone(a.model)
        self.assertIsNone(b.model)

    def test_room_available_evicts_nothing(self):
        a = self._FakeResident("a")
        reg = [a]
        _evict_to_make_room(reg, 2, "b", io.StringIO())
        self.assertEqual(reg, [a])
        self.assertEqual(a.evictions, 0)

    def test_cap_below_one_disables_the_cap(self):
        reg = [self._FakeResident("a"), self._FakeResident("b")]
        _evict_to_make_room(reg, 0, "c", io.StringIO())
        self.assertEqual(len(reg), 2)

    def test_empty_registry_is_noop(self):
        reg = []
        _evict_to_make_room(reg, 1, "c", io.StringIO())
        self.assertEqual(reg, [])

    def test_default_cap_is_one(self):
        import os
        from unittest import mock

        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("REGRAG_MAX_RESIDENT_BACKENDS", None)
            self.assertEqual(_max_resident_backends(), 1)
            os.environ["REGRAG_MAX_RESIDENT_BACKENDS"] = "3"
            self.assertEqual(_max_resident_backends(), 3)
            os.environ["REGRAG_MAX_RESIDENT_BACKENDS"] = "not-a-number"
            self.assertEqual(_max_resident_backends(), 1)


def _budget_chunk(text: str, cid: str = "c") -> LegalChunk:
    return LegalChunk(
        chunk_id=cid,
        doc_id="18/2024/TT-NHNN",
        doc_title="Quy định hoạt động thẻ ngân hàng",
        chapter="Chương II",
        article_id="14",
        article_title="Hạn mức giao dịch thẻ",
        clause_id="2",
        text=text,
        corpus_source=CORPUS_TIER1,
    )


def _budget_retrieved(*texts: str):
    return [
        RetrievedResult(chunk=_budget_chunk(t, cid=f"c{i}"), score=1.0 / (i + 1),
                        rank=i + 1, retriever_backend="bm25-rank_bm25+pyvi")
        for i, t in enumerate(texts)
    ]


class TestRagContextBudget(unittest.TestCase):
    """The 2026-09-12 OOM: unbounded top-k concatenation built an ~18K-token
    prompt (tier1 has 14.6k-char chunks plus duplicates). RAG prompts are now
    budgeted, and what was cut is reported, never silent."""

    def test_budgeted_prompt_equals_legacy_when_everything_fits(self):
        from regrag.generation.prompts import build_rag_prompt, build_rag_prompt_with_report

        retrieved = _budget_retrieved("điều 14 quy định", "khoản 2 nêu rõ", "hạn mức 100 triệu")
        budgeted, report = build_rag_prompt_with_report("Hạn mức?", retrieved)
        legacy = build_rag_prompt("Hạn mức?", retrieved)
        self.assertEqual(budgeted, legacy)
        self.assertFalse(report.touched)
        self.assertGreater(report.budget_chars, report.context_chars)

    def test_giant_passage_is_truncated_and_reported(self):
        from regrag.generation.prompts import (
            CONTEXT_TRUNCATION_MARKER,
            build_rag_prompt_with_report,
        )

        giant = "x" * 20_000
        prompt, report = build_rag_prompt_with_report("Q?", _budget_retrieved(giant))
        self.assertIn(CONTEXT_TRUNCATION_MARKER, prompt)
        self.assertEqual(report.truncated_ranks, (1,))
        self.assertEqual(report.dropped_ranks, ())
        # The context block stays inside the per-passage cap; the marker adds a
        # few chars, hence the small slack.
        self.assertLess(report.context_chars, 3_800)

    def test_budget_exhaustion_drops_later_ranks(self):
        from regrag.generation.prompts import build_rag_prompt_with_report

        # A tighter budget than the defaults exercises the drop path
        # deterministically (explicit per-passage keeps the arithmetic stable
        # against config changes): with 8k total and 5k per passage, two
        # blocks can never exhaust the budget at top_k=3.
        retrieved = _budget_retrieved("y" * 5_000, "y" * 5_000, "ngắn")
        prompt, report = build_rag_prompt_with_report(
            "Q?", retrieved, total_context_chars=8_000, per_passage_chars=5_000
        )
        # rank 1 spills just past the per-passage cap (formatted_context adds
        # a header); rank 2 is truncated into what is left; rank 3 finds less
        # than the floor remaining and is dropped entirely.
        self.assertEqual(report.truncated_ranks, (1, 2))
        self.assertEqual(report.dropped_ranks, (3,))
        self.assertNotIn("[3]", prompt.split("Câu hỏi")[0])

    def test_build_messages_with_report_carries_report_for_rag_only(self):
        from regrag.generation.prompting import build_messages_with_report

        messages, report = build_messages_with_report("Q?", "closed_book")
        self.assertIsNone(report)
        self.assertIn("Câu hỏi: Q?", messages[0]["content"])

        messages, report = build_messages_with_report(
            "Q?", "rag_bm25", _budget_retrieved("z" * 20_000)
        )
        self.assertIsNotNone(report)
        self.assertEqual(report.truncated_ranks, (1,))
        self.assertIn("Tài liệu tham khảo", messages[0]["content"])

    def test_campaign_row_stamps_the_context_report(self):
        """A row whose context was cut must say so - never a silent trim."""
        tmp = tempfile.mkdtemp()
        try:
            store = CheckpointStore(os.path.join(tmp, "ckpt"), warn_stream=io.StringIO())
            runner = CampaignRunner(
                benchmark=make_benchmark(),
                chunks=[_budget_chunk("y" * 20_000)],
                corpus_path="tier1://fixture",
                checkpoint_store=store,
                modes=("rag_bm25",),
                results_dir=None,
            )
            runner.run_model(MODEL_REGISTRY["qwen-7b"], FakeBackend())
            rows = store.records()
            self.assertEqual(len(rows), len(make_benchmark().questions))
            for row in rows:
                self.assertIn("rag_context_chars", row)
                self.assertIn("rag_context_budget_chars", row)
                self.assertEqual(row["rag_context_truncated_ranks"], [1])
                self.assertEqual(row["rag_context_dropped_ranks"], [])
                self.assertLess(row["rag_context_chars"], 3_800)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_purge_retires_cached_rows_for_regeneration(self):
        """A cached row is skipped forever; the only way to regenerate it under
        a new prompt policy is to purge it from the store first. With
        results_dir, the purge must ALSO clean generations.json - materialise()
        only ever adds rows there, so a purged row would otherwise survive in
        the analysis output forever."""
        tmp = tempfile.mkdtemp()
        try:
            ckpt = os.path.join(tmp, "ckpt_p")
            results = os.path.join(tmp, "results")
            store = CheckpointStore(ckpt, warn_stream=io.StringIO())
            runner = CampaignRunner(
                benchmark=make_benchmark(),
                chunks=make_chunks(),
                corpus_path="<synthetic>",
                checkpoint_store=store,
                modes=("closed_book", "rag_bm25"),
                results_dir=results,
            )
            runner.run_model(MODEL_REGISTRY["qwen-7b"], FakeBackend())
            n_closed = sum(
                1 for r in store.records() if r["retrieval_mode"] == "closed_book"
            )
            self.assertEqual(len(store), n_closed * 2)
            store.materialize(results)
            with open(os.path.join(results, "generations.json"), encoding="utf-8") as f:
                self.assertEqual(len(json.load(f)), n_closed * 2)

            purged = store.purge(
                lambda r: r["retrieval_mode"] != "closed_book", results_dir=results
            )
            self.assertEqual(len(purged), n_closed)
            self.assertEqual(len(store), n_closed)
            self.assertTrue(all(
                r["retrieval_mode"] == "closed_book" for r in store.records()
            ))
            with open(os.path.join(results, "generations.json"), encoding="utf-8") as f:
                on_disk = json.load(f)
            self.assertEqual(len(on_disk), n_closed)
            self.assertTrue(all(
                r["retrieval_mode"] == "closed_book" for r in on_disk
            ))
            with open(os.path.join(results, "generations_meta.jsonl"), encoding="utf-8") as f:
                meta_modes = [json.loads(line)["retrieval_mode"] for line in f if line.strip()]
            self.assertTrue(all(m == "closed_book" for m in meta_modes))

            # The rewrite is durable: a fresh store over the same dir agrees.
            store2 = CheckpointStore(ckpt, warn_stream=io.StringIO())
            self.assertEqual(len(store2), n_closed)

            # Idempotent second purge touches nothing.
            self.assertEqual(store.purge(lambda r: r["retrieval_mode"] != "closed_book"), [])
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


class TestPreflightMemory(unittest.TestCase):
    """The worst-case prompt builder for the preflight script: if its length
    ever drifts below the budget max, the preflight stops proving the
    campaign's true memory envelope."""

    def test_worst_case_prompt_hits_max_length(self):
        from scripts.preflight_memory import (
            MAX_PROMPT_CHARS,
            build_worst_case_messages,
        )

        messages = build_worst_case_messages()
        self.assertEqual(len(messages), 1)
        self.assertEqual(messages[0]["role"], "user")
        self.assertEqual(len(messages[0]["content"]), MAX_PROMPT_CHARS)
        self.assertGreater(MAX_PROMPT_CHARS, 6_500)  # tracks the budget config

    def test_explicit_max_is_respected(self):
        from scripts.preflight_memory import build_worst_case_messages

        self.assertEqual(len(build_worst_case_messages(500)[0]["content"]), 500)


class TestPodDriverResume(unittest.TestCase):
    """After a wedged-generation kill, the driver must reuse the newest
    checkpointed run - a fresh dir would silently discard 600+ banked rows."""

    def _make_run(self, root, name, with_checkpoint):
        ckpt = os.path.join(root, name, "checkpoints")
        os.makedirs(ckpt, exist_ok=True)
        if with_checkpoint:
            with open(os.path.join(ckpt, "generations.jsonl"), "w") as f:
                f.write('{"cache_key": "k"}\n')

    def test_picks_newest_nonempty_checkpoint(self):
        from scripts.run_campaign_pod import latest_resumable_run

        root = tempfile.mkdtemp()
        try:
            self.assertIsNone(latest_resumable_run(root))
            self._make_run(root, "pod_20260912_1800", with_checkpoint=True)
            self._make_run(root, "pod_20260912_1854", with_checkpoint=True)
            self._make_run(root, "pod_20260912_1900", with_checkpoint=False)  # empty ckpt
            self.assertEqual(
                latest_resumable_run(root),
                os.path.join(root, "pod_20260912_1854"),
            )
        finally:
            shutil.rmtree(root, ignore_errors=True)

    def test_ignores_non_pod_dirs_and_missing_root(self):
        from scripts.run_campaign_pod import latest_resumable_run

        self.assertIsNone(latest_resumable_run("/nonexistent_root_for_test"))
        root = tempfile.mkdtemp()
        try:
            os.makedirs(os.path.join(root, "not_a_run", "checkpoints"))
            self.assertIsNone(latest_resumable_run(root))
        finally:
            shutil.rmtree(root, ignore_errors=True)


# --- backend selection -------------------------------------------------------


class TestBackendResolution(unittest.TestCase):
    """How the two backends are told apart, and that neither is assumed."""

    def test_the_two_backend_tags_are_distinct(self):
        t = fake_transport_factory(200, {"data": [{"id": "qwen-7b"}]})
        vllm_b = VLLMHttpBackend("http://127.0.0.1:8000/v1", "qwen-7b",
                                 "Qwen/Qwen2.5-7B-Instruct", transport=t)
        tf_b = TransformersQuantBackend(hf_id="Qwen/Qwen2.5-7B-Instruct",
                                        warn_stream=io.StringIO())
        self.assertEqual(vllm_b.backend_tag(), "vllm-serve:qwen-7b")
        self.assertEqual(tf_b.backend_tag(), "transformers:not-loaded")
        self.assertNotEqual(vllm_b.backend_tag(), tf_b.backend_tag())
        self.assertNotEqual(vllm_b.kind, tf_b.kind)
        # The loaded transformers tag is what a real 4-bit run records.
        self.assertEqual(backend_vllm("qwen-7b"), "vllm-serve:qwen-7b")

    def test_vllm_quant_is_never_assumed(self):
        b = VLLMHttpBackend("http://127.0.0.1:8000/v1", "qwen-7b", "x/y",
                            transport=fake_transport_factory())
        self.assertTrue(b.quant_tag().startswith("UNRECORDED"))
        b2 = VLLMHttpBackend("http://127.0.0.1:8000/v1", "qwen-7b", "x/y",
                             quantization_declared="bitsandbytes-4bit",
                             transport=fake_transport_factory())
        self.assertEqual(b2.quant_tag(), "vllm-server:bitsandbytes-4bit")

    def test_chat_completions_payload_uses_alias_and_records_sampling(self):
        t = fake_transport_factory(
            200, {"choices": [{"message": {"content": FLUENT_VI}, "finish_reason": "stop"}],
                  "usage": {"completion_tokens": 12}})
        b = VLLMHttpBackend("http://127.0.0.1:8000/v1", "qwen-7b", "x/y", transport=t)
        gen = b.generate([{"role": "user", "content": "Câu hỏi?"}])
        self.assertEqual(gen.text, FLUENT_VI)
        payload = t.calls[0]["payload"]
        self.assertEqual(payload["model"], "qwen-7b",
                         "must send the SERVED ALIAS, not the HF repo id")
        self.assertEqual(payload["temperature"], 0.0)
        self.assertEqual(payload["max_tokens"], 512)
        self.assertTrue(t.calls[0]["url"].endswith("/v1/chat/completions"))

    def test_wrong_served_model_is_refused(self):
        """A leftover server on the same port is the silent-wrong-model failure."""
        t = fake_transport_factory(200, {"data": [{"id": "qwen-3b"}]})
        b = VLLMHttpBackend("http://127.0.0.1:8000/v1", "qwen-7b", "x/y", transport=t)
        with self.assertRaises(ProvenanceError) as ctx:
            b.verify_serves_alias()
        self.assertIn("not the requested alias", str(ctx.exception))

    def test_http_error_is_loud(self):
        t = fake_transport_factory(500, {"error": "boom"})
        b = VLLMHttpBackend("http://127.0.0.1:8000/v1", "qwen-7b", "x/y", transport=t)
        with self.assertRaises(ProvenanceError):
            b.generate([{"role": "user", "content": "hi"}])

    def test_connection_failure_is_loud(self):
        t = fake_transport_factory(raise_exc=ConnectionError("connection refused"))
        b = VLLMHttpBackend("http://127.0.0.1:8000/v1", "qwen-7b", "x/y", transport=t)
        self.assertFalse(b.is_reachable())
        with self.assertRaises(ProvenanceError):
            b.generate([{"role": "user", "content": "hi"}])

    def test_missing_content_is_refused(self):
        t = fake_transport_factory(200, {"choices": [{"message": {}, "finish_reason": "stop"}]})
        b = VLLMHttpBackend("http://127.0.0.1:8000/v1", "qwen-7b", "x/y", transport=t)
        with self.assertRaises(ProvenanceError) as ctx:
            b.generate([{"role": "user", "content": "hi"}])
        self.assertIn("no message.content", str(ctx.exception))

    def test_explicit_vllm_preference_does_not_silently_fall_back(self):
        t = fake_transport_factory(raise_exc=ConnectionError("refused"))
        with self.assertRaises(ProvenanceError) as ctx:
            resolve_backend(
                MODEL_REGISTRY["qwen-7b"], preference="vllm",
                vllm_base_url="http://127.0.0.1:8000/v1", transport=t,
                warn_stream=io.StringIO(),
            )
        self.assertIn("Not falling back", str(ctx.exception))

    def test_auto_preference_falls_back_loudly(self):
        """auto -> transformers needs a GPU, so here it must raise the GPU error
        rather than quietly returning something usable."""
        t = fake_transport_factory(raise_exc=ConnectionError("refused"))
        log = io.StringIO()
        try:
            import torch
            has_cuda = torch.cuda.is_available()
        except ImportError:
            has_cuda = False
        if has_cuda:
            self.skipTest("a GPU is visible; the fallback would try to load a model")
        with self.assertRaises(ProvenanceError) as ctx:
            resolve_backend(
                MODEL_REGISTRY["qwen-3b"], preference="auto",
                vllm_base_url="http://127.0.0.1:8000/v1", transport=t,
                warn_stream=log, load_transformers=False,
            )
        self.assertIn("CUDA", str(ctx.exception))
        self.assertIn("Falling back", log.getvalue())

    def test_auto_preference_selects_vllm_when_reachable(self):
        t = fake_transport_factory(200, {"data": [{"id": "qwen-7b"}]})
        log = io.StringIO()
        b = resolve_backend(
            MODEL_REGISTRY["qwen-7b"], preference="auto",
            vllm_base_url="http://127.0.0.1:8000/v1", transport=t,
            quantization_declared="fp16", warn_stream=log,
        )
        self.assertIsInstance(b, VLLMHttpBackend)
        self.assertEqual(b.backend_tag(), "vllm-serve:qwen-7b")
        self.assertEqual(b.quant_tag(), "vllm-server:fp16")
        self.assertIn("Using vLLM server", log.getvalue())

    def test_unknown_preference_is_rejected(self):
        with self.assertRaises(ValueError):
            resolve_backend(MODEL_REGISTRY["qwen-7b"], preference="magic",
                            transport=fake_transport_factory(), warn_stream=io.StringIO())

    def test_prompt_exactness_differs_by_backend(self):
        """The sanity probe must say whether the shown prompt is literal."""
        t = fake_transport_factory(200, {"data": [{"id": "qwen-7b"}]})
        b = VLLMHttpBackend("http://127.0.0.1:8000/v1", "qwen-7b", "x/y", transport=t)
        r = b.render_prompt([{"role": "user", "content": "Câu hỏi?"}])
        self.assertFalse(r.exact)
        self.assertIn("SERVER-SIDE", r.note)
        self.assertIn("NOT the literal token string", r.prompt_field())


# --- GPU assertion -----------------------------------------------------------


class TestGpuAssertion(unittest.TestCase):
    """Reads device properties only: no tensor is allocated and nothing runs."""

    def test_require_gpu_raises_when_unavailable(self):
        try:
            import torch
        except ImportError:
            self.skipTest("torch not importable")
        if torch.cuda.is_available():
            self.skipTest("a GPU is visible on this machine")
        with self.assertRaises(ProvenanceError) as ctx:
            gpu_report(require=True, warn_stream=io.StringIO())
        self.assertIn("No CUDA device", str(ctx.exception))

    def test_report_without_requirement_describes_the_environment(self):
        try:
            import torch
        except ImportError:
            self.skipTest("torch not importable")
        r = gpu_report(require=False, warn_stream=io.StringIO())
        self.assertTrue(r["torch_present"])
        self.assertEqual(r["cuda_available"], bool(torch.cuda.is_available()))
        self.assertIn("gpu_backend_tag", r)
        self.assertEqual(r["device_count"], torch.cuda.device_count())


# --- abstention --------------------------------------------------------------


class TestAbstention(unittest.TestCase):
    def test_gold_sentinel_is_detected(self):
        v = detect_abstention(UNANSWERABLE_SENTINEL)
        self.assertTrue(v.abstained)
        self.assertIsNotNone(v.matched_exact)

    def test_prompted_keyphrase_is_detected(self):
        from regrag.generation.prompts import ABSTENTION_KEYPHRASE
        self.assertTrue(detect_abstention(f"Xin lỗi. {ABSTENTION_KEYPHRASE}.").abstained)

    def test_sentinel_detected_despite_case_and_whitespace_noise(self):
        self.assertTrue(detect_abstention("KHÔNG CÓ TRONG KHO VĂN BẢN").abstained)
        self.assertTrue(detect_abstention("  Không   có   trong   kho   văn   bản  ").abstained)

    def test_substantive_answer_is_not_an_abstention(self):
        self.assertFalse(detect_abstention(FLUENT_VI).abstained)

    def test_empty_answer_is_not_credited_as_a_refusal(self):
        """Crediting an empty string would hand a perfect abstention score for nothing."""
        v = detect_abstention("")
        self.assertFalse(v.abstained)
        self.assertFalse(v.matched_soft)

    def test_soft_paraphrase_is_diagnostic_only(self):
        v = detect_abstention("Tôi không tìm thấy thông tin liên quan đến câu hỏi này.")
        self.assertFalse(v.abstained, "paraphrases must not set abstained")
        self.assertTrue(v.soft_only)

    def test_metrics_scorer_detects_the_gold_sentinel(self):
        """Guards RQ2 against a scorer rewrite that loses the sentinel."""
        assert_metrics_detects_sentinel()  # raises ProvenanceError on mismatch

    def test_sentinel_constant_has_not_diverged_from_the_loader(self):
        """qa_loader.py keeps its own copy of the sentinel; metrics.py imports ours."""
        from regrag.corpus.qa_loader import UNANSWERABLE_SENTINEL as LOADER_SENTINEL
        self.assertEqual(LOADER_SENTINEL, UNANSWERABLE_SENTINEL)


# --- benchmark classification ------------------------------------------------


class TestBenchmarkClassification(unittest.TestCase):
    def _q(self, **kw):
        base = dict(id="Q001", question="q", is_answerable=True, category="factual",
                    reference_answer="a", gold_passage="p", gold_doc_ids=["17/2024/TT-NHNN"])
        base.update(kw)
        return GoldQuestion(**base)

    def test_sentinel_without_gold_is_a_probe(self):
        q = self._q(reference_answer=UNANSWERABLE_SENTINEL, gold_passage="",
                    gold_doc_ids=[], source_urls=[], gold_citations=[])
        answerable, basis, anomaly = classify_answerability(q)
        self.assertFalse(answerable)
        self.assertEqual(basis, "sentinel_no_gold")
        self.assertEqual(anomaly, "")

    def test_sentinel_WITH_gold_is_a_data_defect_not_a_probe(self):
        """Calling it a probe would reward refusing a question that has gold."""
        q = self._q(reference_answer=UNANSWERABLE_SENTINEL)
        answerable, basis, anomaly = classify_answerability(q)
        self.assertTrue(answerable)
        self.assertEqual(basis, "sentinel_with_gold")
        self.assertEqual(anomaly, "sentinel_answer_but_has_gold")

    def test_loader_flag_is_trusted(self):
        q = self._q(is_answerable=False, category="unanswerable")
        answerable, basis, _ = classify_answerability(q)
        self.assertFalse(answerable)
        self.assertEqual(basis, "loader_flag")

    def test_answerable_row_with_no_gold_is_flagged_not_dropped(self):
        q = self._q(gold_passage="", gold_doc_ids=[], source_urls=[], gold_citations=[])
        answerable, basis, anomaly = classify_answerability(q)
        self.assertTrue(answerable)
        self.assertEqual(anomaly, "answerable_without_any_gold_evidence")

    @unittest.skipUnless(os.path.exists(GOLD_CSV), "gold CSV not present")
    def test_real_csv_yields_88_rows_with_24_probes(self):
        # 64 answerable + 24 probes since the 2026-09-11 gold correction:
        # Q037 was converted to a coverage-gap probe (authority TT 35/2015 is
        # not in the corpus). Probe rows carry explicit IDs 66-88.
        b = load_benchmark(GOLD_CSV, warn_stream=io.StringIO())
        self.assertEqual(len(b.questions), 88)
        self.assertEqual(len(b.probes), 24)
        self.assertEqual(len(b.answerable), 64)
        self.assertEqual(len({q.id for q in b.questions}), 88)
        m = b.manifest()
        self.assertEqual(m["unanswerable_probes"], 24)
        self.assertTrue(all(q.category == "unanswerable" for q in b.probes))

    @unittest.skipUnless(os.path.exists(GOLD_CSV), "gold CSV not present")
    def test_probe_ids_no_longer_derived_from_row_order(self):
        """Probe rows carry explicit IDs since 2026-09-11, so no id may be
        flagged as row-order-derived - inserting a row must not renumber
        probe ids or invalidate cached probe generations."""
        b = load_benchmark(GOLD_CSV, warn_stream=io.StringIO())
        self.assertEqual(b.unstable_ids, [])
        self.assertFalse(any("ROW ORDER" in w for w in b.warnings))


# --- corpus io ---------------------------------------------------------------


class TestCorpusIO(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="regrag_corpus_")
        self.addCleanup(shutil.rmtree, self.tmp, True)

    def _write(self, name, obj):
        p = os.path.join(self.tmp, name)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False)
        return p

    def test_roundtrip_from_asdict(self):
        from dataclasses import asdict
        chunks = make_chunks(2)
        p = self._write("c.json", [asdict(c) for c in chunks])
        loaded = load_chunks(p)
        self.assertEqual(len(loaded), 2)
        self.assertEqual(loaded[0].chunk_id, chunks[0].chunk_id)
        self.assertEqual(loaded[0].corpus_source, CORPUS_TIER2)

    def test_real_tier1_fixture_loads_and_is_unpublishable(self):
        p = os.path.join(REPO_ROOT, "data", "processed_chunks", "tier1_chunks.json")
        if not os.path.exists(p):
            self.skipTest("tier1 fixture not present")
        chunks = load_chunks(p)
        self.assertTrue(chunks)
        d = describe_corpus(chunks, p)
        self.assertEqual(d["corpus_source"], CORPUS_TIER1)
        self.assertFalse(d["publishable_corpus"], "Tier 1 must never look publishable")

    def test_empty_corpus_raises(self):
        p = self._write("empty.json", [])
        with self.assertRaises(ProvenanceError) as ctx:
            load_chunks(p)
        self.assertIn("0 chunks", str(ctx.exception))

    def test_missing_file_raises(self):
        with self.assertRaises(FileNotFoundError):
            load_chunks(os.path.join(self.tmp, "nope.json"))

    def test_mixed_corpus_sources_raise(self):
        from dataclasses import asdict
        rows = [asdict(c) for c in make_chunks(2)]
        rows[1]["corpus_source"] = CORPUS_TIER1
        p = self._write("mixed.json", rows)
        with self.assertRaises(ProvenanceError) as ctx:
            load_chunks(p)
        self.assertIn("mixes corpus_source", str(ctx.exception))

    def test_unset_corpus_source_raises(self):
        from dataclasses import asdict
        rows = [asdict(c) for c in make_chunks(1)]
        rows[0]["corpus_source"] = CORPUS_UNSET
        p = self._write("unset.json", rows)
        with self.assertRaises(ProvenanceError):
            load_chunks(p)

    def test_missing_required_key_raises(self):
        with self.assertRaises(ProvenanceError):
            chunk_from_dict({"chunk_id": "C1", "text": "x", "corpus_source": CORPUS_TIER2})

    def test_empty_text_raises(self):
        with self.assertRaises(ProvenanceError):
            chunk_from_dict({"chunk_id": "C1", "doc_id": "d", "article_id": "1",
                             "text": "   ", "corpus_source": CORPUS_TIER2})

    def test_duplicate_chunk_ids_raise(self):
        from dataclasses import asdict
        c = make_chunks(1)[0]
        p = self._write("dup.json", [asdict(c), asdict(c)])
        with self.assertRaises(ProvenanceError) as ctx:
            load_chunks(p)
        self.assertIn("duplicate chunk_id", str(ctx.exception))

    def test_corpus_hash_is_order_insensitive_but_content_sensitive(self):
        a, b = make_chunks(3), make_chunks(3)
        self.assertEqual(corpus_hash(a), corpus_hash(list(reversed(b))))
        b[0].text = b[0].text + " đã sửa"
        self.assertNotEqual(corpus_hash(a), corpus_hash(b))

    def test_corpus_hash_ignores_cosmetic_whitespace(self):
        a = make_chunks(2)
        b = make_chunks(2)
        b[0].text = b[0].text.replace(" ", "   ")
        self.assertEqual(corpus_hash(a), corpus_hash(b))


# --- sanity probe ------------------------------------------------------------


class TestSanityDetectors(unittest.TestCase):
    """The detectors that separate 'model hallucinates' from 'we prompted wrong'."""

    def test_fluent_vietnamese_passes(self):
        self.assertEqual(check_output(FLUENT_VI, question="Điều kiện mở tài khoản?"), [])

    def test_empty_output_is_blocking(self):
        f = check_output("", question="q")
        self.assertEqual(len(f), 1)
        self.assertEqual(f[0].code, "empty-output")
        self.assertEqual(f[0].severity, "blocking")

    def test_template_markup_is_blocking(self):
        marker = "<" + "|im_start|" + ">"
        out = f"{marker}assistant\nCâu trả lời ở đây.\n" + "<" + "|im_end|" + ">"
        f = check_output(out, question="Điều kiện mở tài khoản?")
        codes = [x.code for x in f]
        self.assertIn("template-markup-in-output", codes)
        self.assertTrue(any(x.severity == "blocking" for x in f))
        self.assertTrue(detect_template_leak(out))

    def test_repetition_loop_is_blocking(self):
        out = " ".join(["không"] * 60)
        codes = [x.code for x in check_output(out, question="q")]
        self.assertIn("repetition-single-token", codes)
        self.assertTrue(detect_repetition(out))

    def test_ngram_loop_is_blocking(self):
        unit = "theo quy định của ngân hàng nhà nước "
        out = unit * 20
        codes = [x.code for x in check_output(out, question="q")]
        self.assertTrue(
            "repetition-ngram" in codes or "vocabulary-collapse" in codes,
            f"a 20x repeated sentence must be caught, got {codes}",
        )

    def test_normal_vietnamese_is_not_flagged_as_repetition(self):
        self.assertEqual(detect_repetition(FLUENT_VI), [])

    def test_english_answer_is_blocking(self):
        out = (
            "According to the current regulations, an individual must be at least "
            "18 years old and have full civil capacity to open a payment account "
            "at a commercial bank in the country."
        )
        codes = [x.code for x in check_output(out, question="Điều kiện mở tài khoản?")]
        self.assertIn("not-vietnamese", codes)
        self.assertTrue(detect_language(out))

    def test_vietnamese_is_not_flagged_as_english(self):
        self.assertEqual(detect_language(FLUENT_VI), [])

    def test_short_answer_is_not_language_flagged(self):
        """A terse correct answer must not trip the diacritic-ratio heuristic."""
        self.assertEqual(detect_language("Điều 5."), [])

    def test_question_echo_is_blocking(self):
        q = "Điều kiện để cá nhân mở tài khoản thanh toán tại ngân hàng là gì?"
        codes = [x.code for x in check_output(q, question=q)]
        self.assertIn("echoes-question", codes)

    def test_truncation_is_review_not_blocking(self):
        f = check_output(FLUENT_VI, question="q", finish_reason="length")
        self.assertEqual([x.code for x in f], ["hit-max-new-tokens"])
        self.assertEqual(f[0].severity, "review")

    def test_output_stats_are_computed(self):
        s = output_stats(FLUENT_VI)
        self.assertGreater(s["chars"], 50)
        self.assertGreater(s["vietnamese_diacritic_ratio"], 0.01)
        self.assertGreater(s["vi_function_word_hits"], 2)
        self.assertEqual(output_stats(""), {"chars": 0, "tokens_whitespace": 0,
                                            "distinct_token_ratio": 0.0,
                                            "vietnamese_diacritic_ratio": 0.0,
                                            "vi_function_word_hits": 0})


class TestProbeRun(unittest.TestCase):
    """The probe end-to-end against a fake backend, and the gate that blocks a run."""

    def test_probe_runs_three_fixed_questions_and_passes(self):
        b = FakeBackend()
        reports = run_probe(b, mode="closed_book")
        self.assertEqual(len(reports), len(PROBE_QUESTIONS))
        self.assertEqual(b.calls, 3)
        self.assertTrue(all(r.ok for r in reports), [r.findings for r in reports])
        self.assertTrue(all(r.backend_tag == "vllm-serve:fake-model" for r in reports))
        self.assertTrue(all(r.quant_tag == "vllm-server:fp16" for r in reports))

    def test_probe_gate_blocks_a_degenerate_model(self):
        good = FakeBackend(text=FLUENT_VI)
        probe_gate(run_probe(good, mode="closed_book"))  # must not raise

        bad = FakeBackend(text="")
        reports = run_probe(bad, mode="closed_book")
        self.assertFalse(any(r.ok for r in reports))
        with self.assertRaises(ProvenanceError) as ctx:
            probe_gate(reports)
        self.assertIn("Refusing to run the campaign", str(ctx.exception))
        self.assertIn("empty-output", str(ctx.exception))

    def test_probe_report_is_human_readable_and_names_the_backend(self):
        b = FakeBackend(prompt_exact=True)
        text = format_probe_report("qwen-7b", run_probe(b, mode="closed_book"))
        self.assertIn("PROMPT-FORMAT SANITY PROBE", text)
        self.assertIn("qwen-7b", text)
        self.assertIn("generation_backend", text)
        self.assertIn("vllm-serve:fake-model", text)
        self.assertIn("EXACT", text)
        self.assertIn("PROBE PASSED", text)
        for pq in PROBE_QUESTIONS:
            self.assertIn(pq["id"], text)

    def test_probe_report_flags_non_exact_prompt(self):
        b = FakeBackend(prompt_exact=False)
        text = format_probe_report("qwen-7b", run_probe(b, mode="closed_book"))
        self.assertIn("NOT EXACT", text)

    def test_probe_report_shows_failure_clearly(self):
        b = FakeBackend(text="ok " * 80)
        text = format_probe_report("qwen-7b", run_probe(b, mode="closed_book"))
        self.assertIn("PROBE FAILED", text)
        self.assertIn("Do NOT run the campaign", text)


# --- model set config --------------------------------------------------------


class TestModelSetConfig(unittest.TestCase):
    """The model list is explicit config, and non-peers cannot be reported as peers."""

    def test_default_matrix_matches_plan_section_3_3(self):
        self.assertEqual(len(DEFAULT_MODEL_ALIASES), 3)
        self.assertEqual(tuple(RETRIEVAL_MODES), ("closed_book", "rag_bm25", "rag_dense"))
        grid = validate_matrix(DEFAULT_MODEL_ALIASES, RETRIEVAL_MODES, 88)
        self.assertEqual(grid["cells"], 3 * 3 * 88)
        self.assertEqual(grid["cells"], 792)

    def test_default_selection_is_all_peers(self):
        specs = select_models(DEFAULT_MODEL_ALIASES, warn_stream=io.StringIO())
        self.assertTrue(all(s.is_peer for s in specs))

    def test_aliases_mirror_the_vllm_launcher_registry(self):
        """load_vllm_models.py sets VLLM_SERVED_MODEL_NAME = alias, so the alias is
        what the client must send. Drift here means a 404 or the wrong model."""
        launcher = {
            "qwen-7b": "Qwen/Qwen2.5-7B-Instruct",
            "llama-3b": "meta-llama/Llama-3.2-3B-Instruct",
            "qwen-3b": "Qwen/Qwen2.5-3B-Instruct",
            "vistral-7b": "Viet-Mistral/Vistral-7B-Chat",
            "qwen-1.5b": "Qwen/Qwen2.5-1.5B-Instruct",
        }
        for alias, hf_id in launcher.items():
            self.assertIn(alias, MODEL_REGISTRY, f"{alias} missing from harness registry")
            self.assertEqual(MODEL_REGISTRY[alias].hf_id, hf_id)
            self.assertTrue(MODEL_REGISTRY[alias].in_vllm_registry)

    def test_llama_is_labelled_a_weak_vietnamese_contrast_and_warns(self):
        """Plan §3.3: never a peer, or it confounds every cross-model number."""
        log = io.StringIO()
        specs = select_models(["llama-3b"], warn_stream=log)
        self.assertEqual(specs[0].role, "weak-vietnamese-contrast")
        self.assertFalse(specs[0].is_peer)
        self.assertIn("WARNING", log.getvalue())
        self.assertIn("NOT 'peer'", log.getvalue())
        self.assertIn("must not be averaged", log.getvalue())

    def test_qwen_1_5b_is_labelled_out_of_matrix_and_warns(self):
        log = io.StringIO()
        specs = select_models(["qwen-1.5b"], warn_stream=log)
        self.assertEqual(specs[0].role, "not-in-experimental-matrix")
        self.assertIn("WARNING", log.getvalue())

    def test_gemma_is_flagged_as_absent_from_the_launcher_registry(self):
        log = io.StringIO()
        specs = select_models(["gemma-3-4b-it"], warn_stream=log)
        self.assertFalse(specs[0].in_vllm_registry)
        self.assertIn("serve_vllm.sh", log.getvalue())

    def test_unknown_alias_raises(self):
        with self.assertRaises(KeyError):
            select_models(["gpt-5"], warn_stream=io.StringIO())

    def test_unknown_mode_raises(self):
        with self.assertRaises(ValueError):
            validate_matrix(["qwen-7b"], ("rag_quantum",), 88)

    def test_sampling_defaults_are_reproducible(self):
        s = SamplingConfig()
        self.assertEqual(s.temperature, 0.0)
        self.assertFalse(s.do_sample)
        self.assertEqual(s.vllm_payload()["max_tokens"], 512)


# --- progress + Drive sync ---------------------------------------------------


class TestProgressTracker(unittest.TestCase):
    def test_counts_and_eta(self):
        out = io.StringIO()
        p = ProgressTracker(total=10, every=5, stream=out)
        for _ in range(4):
            p.tick("done")
        p.tick("skipped")
        p.tick("done")
        s = p.summary()
        self.assertEqual(s["rows_written"], 5)
        self.assertEqual(s["cached_skipped"], 1)
        self.assertEqual(s["failed"], 0)
        text = out.getvalue()
        # The line printed at processed=5 reflects 4 generations + 1 cache hit.
        self.assertIn("cached_skipped=1", text)
        self.assertIn("ETA", text)
        self.assertIn("5/10", text)

    def test_eta_ignores_cached_skips(self):
        """A resumed run must not report an ETA derived from cache hits."""
        out = io.StringIO()
        p = ProgressTracker(total=100, every=1, stream=out)
        for _ in range(50):
            p.tick("skipped")
        self.assertIn("ETA n/a", out.getvalue())


class TestDriveSync(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="regrag_drive_")
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.local = os.path.join(self.tmp, "local")
        self.drive = os.path.join(self.tmp, "drive")
        os.makedirs(self.local, exist_ok=True)

    def test_fresh_run_when_no_checkpoint_exists(self):
        from regrag.generation.checkpoint import restore_from_drive
        log = io.StringIO()
        rep = restore_from_drive(self.drive, self.local, warn_stream=log)
        self.assertEqual(rep["restored"], [])
        self.assertIn("FRESH RUN", log.getvalue())

    def test_restore_reports_what_and_when(self):
        from regrag.generation.checkpoint import restore_from_drive, sync_to_drive
        os.makedirs(self.drive, exist_ok=True)
        with open(os.path.join(self.local, "generations.jsonl"), "w", encoding="utf-8") as f:
            f.write('{"cache_key":"k1"}\n{"cache_key":"k2"}\n')
        sync_to_drive(self.local, self.drive, warn_stream=io.StringIO())

        local2 = os.path.join(self.tmp, "local2")
        log = io.StringIO()
        rep = restore_from_drive(self.drive, local2, warn_stream=log)
        self.assertEqual(len(rep["restored"]), 1)
        self.assertIn("Restored", log.getvalue())
        self.assertIn("2 completed row(s)", log.getvalue())
        self.assertTrue(os.path.exists(os.path.join(local2, "generations.jsonl")))

    def test_sync_is_idempotent(self):
        from regrag.generation.checkpoint import sync_to_drive
        with open(os.path.join(self.local, "generations.jsonl"), "w", encoding="utf-8") as f:
            f.write('{"cache_key":"k1"}\n')
        sync_to_drive(self.local, self.drive, warn_stream=io.StringIO())
        second = sync_to_drive(self.local, self.drive, warn_stream=io.StringIO())
        self.assertEqual(len(second["copied"]), 1)
        self.assertEqual(second["errors"], [])

    def test_sync_copies_extra_result_dirs(self):
        from regrag.generation.checkpoint import sync_to_drive
        results = os.path.join(self.local, "results")
        os.makedirs(results, exist_ok=True)
        with open(os.path.join(results, "generations.json"), "w", encoding="utf-8") as f:
            f.write("[]")
        rep = sync_to_drive(self.local, self.drive, extra_dirs=[("results", "results")],
                            warn_stream=io.StringIO())
        self.assertIn("results/generations.json", rep["copied"])
        self.assertTrue(os.path.exists(os.path.join(self.drive, "results", "generations.json")))


if __name__ == "__main__":
    unittest.main()

