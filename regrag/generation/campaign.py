"""The campaign loop: model x retrieval-mode x question, checkpointed.

Loop order is **model outer, then mode, then question**, because the expensive
resource is the model: on Colab you serve or load one model at a time, and all
88 x 3 cells for that model should complete before it is swapped out.

Retrieval is **shared across models**. The retrieved context for (question, mode)
does not depend on which model will read it, so it is computed once and memoised.
That also makes the three model columns genuinely comparable: they are conditioned
on byte-identical context, which is the point of RQ1 and RQ3.

Everything the loop writes goes through one function, ``_make_record``, so there
is exactly one place where provenance is stamped and exactly one place that can
be audited for the "never write a row that looks clean when it is not" rule.
"""

from __future__ import annotations

import gc
import os
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from regrag.evaluation.citation import extract_citations
from regrag.generation.abstention import detect_abstention
from regrag.generation.benchmark import BenchmarkSet
from regrag.generation.checkpoint import (
    CheckpointStore,
    META_FIELDS,
    ProgressTracker,
    split_record,
)
from regrag.generation.config import (
    BACKEND_CLOSED_BOOK,
    RETRIEVAL_MODES,
    SamplingConfig,
)
from regrag.generation.corpus_io import corpus_hash, corpus_source_of, describe_corpus
from regrag.generation.prompting import build_messages_with_report
from regrag.models import GenerationResult, GoldQuestion, LegalChunk, RetrievedResult
from regrag.provenance import CORPUS_UNSET, ProvenanceError, is_degraded, publishable_corpus

# Use the repo's own cache-key function so the harness and scripts/regenerate.py
# can never disagree about what invalidates a cached generation.
from scripts.regenerate import cache_key as repo_cache_key

#: Closed-book does not consult the corpus, but the cache key still binds to it so
#: that a corpus rebuild marks a new campaign rather than silently reusing rows
#: produced beside a different corpus. Documented deviation-from-zero, not a
#: special case in the key function itself.
CLOSED_BOOK_CORPUS_HASH_MODE = "bind"


class CampaignRunner:
    """Runs the grid and checkpoints every row as it lands."""

    def __init__(
        self,
        benchmark: BenchmarkSet,
        chunks: Sequence[LegalChunk],
        corpus_path: str,
        checkpoint_store: CheckpointStore,
        modes: Sequence[str] = RETRIEVAL_MODES,
        top_k: int = 3,
        dense_model_name: str = "BAAI/bge-m3",
        sampling: Optional[SamplingConfig] = None,
        gpu_info: Optional[Dict[str, Any]] = None,
        abort_on_degraded_retrieval: bool = True,
        backend_conflict_policy: str = "raise",
        results_dir: Optional[str] = None,
        warn_stream=None,
    ) -> None:
        self.benchmark = benchmark
        self.chunks = list(chunks)
        self.corpus_path = corpus_path
        self.store = checkpoint_store
        self.results_dir = results_dir
        self.modes = tuple(modes)
        self.top_k = top_k
        self.dense_model_name = dense_model_name
        self.sampling = sampling or SamplingConfig()
        self.gpu_info = gpu_info or {}
        self.abort_on_degraded_retrieval = abort_on_degraded_retrieval
        if backend_conflict_policy not in ("raise", "reuse", "regenerate"):
            raise ValueError(
                f"backend_conflict_policy must be raise|reuse|regenerate, got {backend_conflict_policy!r}"
            )
        self.backend_conflict_policy = backend_conflict_policy
        self._stream = warn_stream if warn_stream is not None else sys.stderr

        bad = [m for m in self.modes if m not in RETRIEVAL_MODES]
        if bad:
            raise ValueError(f"Unknown retrieval mode(s) {bad}; known: {list(RETRIEVAL_MODES)}")

        self._retriever = None
        self._retrieval_cache: Dict[Tuple[str, str], List[RetrievedResult]] = {}
        self._dense_built = False
        self._warned_unrecorded_quant: set = set()

        self.corpus_src = corpus_source_of(self.chunks) if self.chunks else CORPUS_UNSET
        self.corpus_h = corpus_hash(self.chunks) if self.chunks else CORPUS_UNSET

        if not self.chunks and any(m != "closed_book" for m in self.modes):
            raise ProvenanceError(
                f"Modes {list(self.modes)} need a corpus but 0 chunks were supplied. "
                "Retrieving from an empty corpus returns nothing, the model answers "
                "from parametric memory, and the RAG column becomes indistinguishable "
                "from closed-book with no warning."
            )
        if not self.chunks:
            self._stream.write(
                "[CAMPAIGN][WARNING] Running closed_book with NO corpus: every row is "
                f"stamped corpus_source={CORPUS_UNSET!r} and assert_publishable() will "
                "refuse it. These rows are diagnostics, not paper numbers.\n"
            )
            self._stream.flush()

    # -- cache keys ----------------------------------------------------------
    def cache_key_for(self, q: GoldQuestion, model_name: str, mode: str) -> str:
        """Per-question, per-model, per-mode key, via scripts/regenerate.cache_key.

        Binding to the question's own content (not the whole-file CSV hash) is
        what makes a CSV revision that touches a few rows invalidate only those
        rows, so thousands of cached generations survive the 09-10 review edits.
        """
        return repo_cache_key(
            question_id=q.id,
            model_name=model_name,
            retrieval_mode=mode,
            csv_hash=self.benchmark.csv_hash,
            corpus_hash=self.corpus_h,
            question_content=q,
        )

    def planned_keys(self, model_names: Sequence[str]) -> Dict[str, List[str]]:
        """Every cache key the grid will need, grouped by model. No generation."""
        out: Dict[str, List[str]] = {}
        for name in model_names:
            out[name] = [
                self.cache_key_for(q, name, m) for m in self.modes for q in self.benchmark.questions
            ]
        return out

    # -- retrieval -----------------------------------------------------------
    def _ensure_retriever(self):
        if self._retriever is not None:
            return self._retriever
        from regrag.retrieval.retriever import RegulatoryRetriever

        self._retriever = RegulatoryRetriever(
            self.chunks, dense_model_name=self.dense_model_name
        )
        bm25_backend = self._retriever.bm25_index.backend
        self._stream.write(f"[CAMPAIGN] BM25 backend: {bm25_backend}\n")
        if is_degraded(bm25_backend):
            msg = (
                f"BM25 index is running DEGRADED ({bm25_backend}). Plan §5: the "
                "compound-word segmentation RQ3 measures is exactly what is missing. "
                "Install rank_bm25 and pyvi."
            )
            if self.abort_on_degraded_retrieval:
                raise ProvenanceError(msg + " abort_on_degraded_retrieval=True, so the run stops.")
            self._stream.write(f"[CAMPAIGN][ERROR] {msg} Continuing because "
                               "abort_on_degraded_retrieval=False; rows are tagged.\n")
        self._stream.flush()
        return self._retriever

    def _ensure_dense(self):
        """Build the dense index lazily.

        ``RegulatoryRetriever.__init__`` constructs a ``DenseIndex`` but never
        calls ``build()``, so ``search()`` would raise "has not been built".
        Building is also the point where ``sentence_transformers`` is proven
        present - DenseIndex.build() calls provenance.require() and raises
        rather than falling back to the old corpus-order mock.
        Done lazily so a closed_book/BM25-only run never downloads BGE-M3.
        """
        if self._dense_built:
            return
        retr = self._ensure_retriever()
        self._stream.write(
            f"[CAMPAIGN] Building dense index ({self.dense_model_name}) over "
            f"{len(self.chunks)} chunks...\n"
        )
        self._stream.flush()
        t0 = time.time()
        retr.dense_index.build()
        self._dense_built = True
        self._stream.write(
            f"[CAMPAIGN] Dense index built in {time.time()-t0:.1f}s; "
            f"backend={retr.dense_index.backend!r}\n"
        )
        self._stream.flush()

    def retrieve_for(self, q: GoldQuestion, mode: str) -> Tuple[List[RetrievedResult], str]:
        """Retrieve (memoised) and return (results, retriever_backend tag)."""
        if mode == "closed_book":
            return [], BACKEND_CLOSED_BOOK

        cached = self._retrieval_cache.get((q.id, mode))
        if cached is not None:
            return cached, self._backend_for(mode)

        retr = self._ensure_retriever()
        if mode == "rag_dense":
            self._ensure_dense()
        results = retr.retrieve(q.question, mode=mode, top_k=self.top_k)
        backend = self._backend_for(mode)

        if is_degraded(backend):
            msg = (
                f"Retrieval backend for mode={mode} is DEGRADED ({backend}). Refusing "
                "to write rows that look clean when the ranking is not real."
            )
            if self.abort_on_degraded_retrieval:
                raise ProvenanceError(msg)
            self._stream.write(f"[CAMPAIGN][ERROR] {msg} Tagging and continuing.\n")
            self._stream.flush()

        self._retrieval_cache[(q.id, mode)] = results
        return results, backend

    def prepare_indices(self) -> Dict[str, str]:
        """Build the indices the selected modes need and report their real tags.

        Public entry point for the notebook, so it does not have to reach into
        ``_ensure_retriever`` / ``_ensure_dense``. The dense index is built only
        when ``rag_dense`` is selected: BGE-M3 is a multi-GB download and there
        is no reason to fetch it for a closed-book/BM25-only run.

        Returns ``{"rag_bm25": <tag>, "rag_dense": <tag or "not-built">}``.
        """
        tags: Dict[str, str] = {}
        if not self.chunks:
            return {"rag_bm25": CORPUS_UNSET, "rag_dense": "not-built"}
        retr = self._ensure_retriever()
        tags["rag_bm25"] = retr.bm25_index.backend
        if "rag_dense" in self.modes:
            self._ensure_dense()
            tags["rag_dense"] = retr.dense_index.backend
        else:
            tags["rag_dense"] = "not-built"
        return tags

    def _backend_for(self, mode: str) -> str:
        retr = self._ensure_retriever()
        if mode == "rag_bm25":
            return retr.bm25_index.backend
        if mode == "rag_dense":
            return retr.dense_index.backend
        return BACKEND_CLOSED_BOOK

    # -- record construction: the ONLY place provenance is stamped ------------
    def _make_record(
        self,
        q: GoldQuestion,
        model_spec,
        mode: str,
        key: str,
        prompt_field: str,
        generation,
        retrieved: Sequence[RetrievedResult],
        retriever_backend: str,
        backend,
        rag_context=None,
    ) -> Dict[str, Any]:
        text = generation.text or ""
        answer_text = text.strip()
        verdict = detect_abstention(answer_text)

        gen = GenerationResult(
            question_id=q.id,
            model_name=model_spec.alias,
            retrieval_mode=mode,
            prompt=prompt_field,
            raw_response=generation.raw,
            answer_text=answer_text,
            extracted_citations=extract_citations(answer_text),
            abstained=verdict.abstained,
            corpus_source=self.corpus_src,
            retriever_backend=retriever_backend,
            cache_key=key,
        )
        record = asdict(gen)

        desc = backend.describe()
        record["generation_backend"] = desc.get("generation_backend", CORPUS_UNSET)
        record["quant_config"] = desc.get("quant_config", CORPUS_UNSET)
        record["hf_model_id"] = model_spec.hf_id
        record["served_alias"] = desc.get("served_alias", model_spec.alias)
        record["model_set_role"] = model_spec.role
        record["template_applied_by"] = desc.get("template_applied_by", "")
        record["prompt_exact"] = bool(desc.get("prompt_exact", False))
        record["gpu_names"] = self.gpu_info.get("device_names", [])
        record["gpu_memory_gb"] = self.gpu_info.get("device_memory_gb", [])
        record["sampling"] = self.sampling.as_dict()
        record["latency_s"] = generation.latency_s
        record["finish_reason"] = generation.finish_reason
        record["usage"] = dict(generation.usage or {})
        record["is_answerable"] = q.is_answerable
        record["abstention_matched_exact"] = verdict.matched_exact
        record["abstention_soft_patterns"] = list(verdict.matched_soft)
        record["corpus_path"] = self.corpus_path
        record["retrieved_chunk_ids"] = [r.chunk.chunk_id for r in retrieved]
        record["retrieved_scores"] = [round(float(r.score), 6) for r in retrieved]
        if rag_context is not None:
            # What the RAG_CONTEXT_* budget did to this prompt. Absent on
            # closed-book rows and on pre-budget rows = nothing was cut.
            record["rag_context_chars"] = rag_context.context_chars
            record["rag_context_budget_chars"] = rag_context.budget_chars
            record["rag_context_truncated_ranks"] = list(rag_context.truncated_ranks)
            record["rag_context_dropped_ranks"] = list(rag_context.dropped_ranks)
        record["generated_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")

        # The rule: never write a row that looks clean when it is not.
        if is_degraded(record["generation_backend"]):
            raise ProvenanceError(
                f"Refusing to write {q.id}/{model_spec.alias}/{mode}: "
                f"generation_backend={record['generation_backend']!r} is DEGRADED. "
                "A degraded backend means the precision or the serving path is not "
                "what the campaign claims."
            )
        if record["quant_config"].startswith("UNRECORDED"):
            # Once per model, not once per row: precision affects every mode
            # including closed-book, but 792 identical warnings would bury the
            # progress output that the operator is actually watching.
            warn_key = (model_spec.alias, record["quant_config"])
            if warn_key not in self._warned_unrecorded_quant:
                self._warned_unrecorded_quant.add(warn_key)
                self._stream.write(
                    f"[CAMPAIGN][WARNING] {model_spec.alias} rows carry "
                    f"quant_config={record['quant_config']!r}: the precision is "
                    "UNVERIFIED for this backend. Rows are still written, but state "
                    "the precision explicitly before reporting them. (This warning "
                    "is emitted once per model, not once per row.)\n"
                )
                self._stream.flush()
        return record

    # -- the loop ------------------------------------------------------------
    def run_model(
        self,
        model_spec,
        backend,
        progress: Optional[ProgressTracker] = None,
        materialize_every: int = 25,
        on_error: str = "raise",
        on_materialize: Optional[Callable[[], None]] = None,
    ) -> Dict[str, Any]:
        """Run every (mode, question) cell for one model on one backend.

        ``on_materialize`` is called after each periodic materialisation. The
        notebook uses it to push the checkpoint up to Google Drive, so a session
        killed mid-model loses at most ``materialize_every`` rows rather than
        every row generated since the last model finished.
        """
        errors: List[Dict[str, Any]] = []
        written = 0
        skipped = 0
        since_materialize = 0

        for mode in self.modes:
            if mode == "rag_dense":
                self._ensure_dense()
            for q in self.benchmark.questions:
                key = self.cache_key_for(q, model_spec.alias, mode)

                prior = self.store.get(key)
                if prior is not None:
                    prior_backend = prior.get("generation_backend", "<missing>")
                    current_backend = backend.backend_tag()
                    if prior_backend != current_backend:
                        if self.backend_conflict_policy == "raise":
                            raise ProvenanceError(
                                f"Cache key {key} ({q.id}/{model_spec.alias}/{mode}) was "
                                f"produced by generation_backend={prior_backend!r} but the "
                                f"current backend is {current_backend!r}. Rows from "
                                "different backends must not be pooled: the chat template, "
                                "sampling and quantization differ. Choose a policy "
                                "explicitly - 'reuse' (keep the old row and accept the mix, "
                                "which the final guard will report) or 'regenerate'."
                            )
                        if self.backend_conflict_policy == "reuse":
                            skipped += 1
                            if progress:
                                progress.tick("skipped")
                            continue
                        # "regenerate": fall through and overwrite.
                    else:
                        skipped += 1
                        if progress:
                            progress.tick("skipped")
                        continue

                retrieved, retriever_backend = self.retrieve_for(q, mode)
                messages, rag_context = build_messages_with_report(q.question, mode, retrieved)
                rendered = backend.render_prompt(messages)

                try:
                    generation = backend.generate(messages)
                except ProvenanceError:
                    raise  # a dead server or a degraded backend must stop the run
                except Exception as exc:
                    errors.append(
                        {
                            "cache_key": key,
                            "question_id": q.id,
                            "model_name": model_spec.alias,
                            "retrieval_mode": mode,
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                    )
                    if progress:
                        progress.tick("failed")
                    if on_error == "raise":
                        raise
                    continue

                record = self._make_record(
                    q, model_spec, mode, key, rendered.prompt_field(),
                    generation, retrieved, retriever_backend, backend,
                    rag_context=rag_context,
                )
                self.store.append(record)
                written += 1
                since_materialize += 1
                if progress:
                    progress.tick("done")

                if since_materialize >= materialize_every:
                    since_materialize = 0
                    self.materialize()
                    if on_materialize is not None:
                        on_materialize()

        if since_materialize:
            self.materialize()
            if on_materialize is not None:
                on_materialize()

        return {
            "model": model_spec.alias,
            "generation_backend": backend.backend_tag(),
            "quant_config": backend.quant_tag(),
            "rows_written": written,
            "rows_skipped_cached": skipped,
            "errors": errors,
        }

    def materialize(self, results_dir: Optional[str] = None) -> Dict[str, Any]:
        """Write the checkpointed rows out in the repo's own record format."""
        gc.collect()
        target = results_dir or self.results_dir
        if target is None:
            return {"skipped": "no results_dir configured"}
        return self.store.materialize(target)

    # -- final guards --------------------------------------------------------
    def assert_no_backend_pooling(self) -> Dict[str, Any]:
        """Refuse a run whose rows for one model came from two backends."""
        per_model: Dict[str, Dict[str, int]] = {}
        for rec in self.store.records():
            name = rec.get("model_name", "?")
            tag = rec.get("generation_backend", "<missing>")
            per_model.setdefault(name, {})
            per_model[name][tag] = per_model[name].get(tag, 0) + 1

        mixed = {m: tags for m, tags in per_model.items() if len(tags) > 1}
        if mixed:
            raise ProvenanceError(
                f"Rows for the same model were produced by different generation "
                f"backends: {mixed}. They are not the same measurement (chat template, "
                "sampling and quantization differ) and must never be pooled into one "
                "table cell. Re-run the affected model on a single backend, or split "
                "the results by backend before aggregating."
            )
        return per_model

    def provenance_summary(self) -> Dict[str, Any]:
        recs = self.store.records()
        degraded = self.store.degraded_rows()
        sources = sorted({r.get("corpus_source", CORPUS_UNSET) for r in recs})
        backends = sorted({r.get("retriever_backend", CORPUS_UNSET) for r in recs})
        gen_backends = self.store.backends_present()
        return {
            "rows": len(recs),
            "corpus_sources": sources,
            "retriever_backends": backends,
            "generation_backends": gen_backends,
            "publishable_corpus": all(publishable_corpus(s) for s in sources) if sources else False,
            "degraded_row_count": len(degraded),
            "degraded_rows": degraded[:20],
        }


def build_manifest(
    *,
    run_name: str,
    benchmark: BenchmarkSet,
    corpus_description: Dict[str, Any],
    model_specs: Sequence[Any],
    modes: Sequence[str],
    top_k: int,
    sampling: SamplingConfig,
    gpu_info: Dict[str, Any],
    backend_descriptions: Sequence[Dict[str, Any]],
    run_results: Sequence[Dict[str, Any]],
    progress_summary: Dict[str, Any],
    provenance: Dict[str, Any],
    probe_reports: Optional[Sequence[Dict[str, Any]]] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Assemble the run manifest - the provenance record for the whole campaign."""
    manifest: Dict[str, Any] = {
        "run_name": run_name,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "grid": {
            "models": [asdict(s) if hasattr(s, "__dataclass_fields__") else s for s in model_specs],
            "retrieval_modes": list(modes),
            "top_k": top_k,
            "expected_cells": len(model_specs) * len(modes) * len(benchmark.questions),
        },
        "benchmark": benchmark.manifest(),
        "corpus": corpus_description,
        "sampling": sampling.as_dict(),
        "gpu": gpu_info,
        "backends": list(backend_descriptions),
        "run_results": list(run_results),
        "progress": progress_summary,
        "provenance_summary": provenance,
        "sanity_probe": list(probe_reports or []),
    }
    if extra:
        manifest.update(extra)
    return manifest
