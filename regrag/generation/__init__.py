"""Generation package for prompt construction, model serving and the campaign.

Deliberately kept to the cheap prompt re-exports. ``regrag.evaluation.metrics``
imports ``regrag.generation.prompts``, and ``regrag.evaluation.__init__`` imports
``metrics``; if this file imported ``campaign`` (which imports
``regrag.evaluation.citation``) the two packages would initialise each other in a
cycle. Import the heavier modules by path instead::

    from regrag.generation.campaign   import CampaignRunner, build_manifest
    from regrag.generation.backends   import resolve_backend, gpu_report
    from regrag.generation.benchmark  import load_benchmark
    from regrag.generation.corpus_io  import load_chunks, describe_corpus
    from regrag.generation.checkpoint import CheckpointStore, restore_from_drive, sync_to_drive
    from regrag.generation.sanity     import run_probe, format_probe_report, probe_gate
    from regrag.generation.abstention import detect_abstention, assert_metrics_detects_sentinel
    from regrag.generation.config     import MODEL_REGISTRY, DEFAULT_MODEL_ALIASES, select_models

Module map
----------
    config      explicit model set (mirrors scripts/load_vllm_models.py aliases),
                modes, sampling, Drive path helpers
    benchmark   trusted CSV -> GoldQuestion, deriving is_answerable for the 23
                unanswerable probes loudly (qa_loader hardcodes it to True)
    corpus_io   processed-chunk JSON -> LegalChunk, corpus hash and description
    prompting   chat-message assembly over the repo's sanctioned prompt templates
    backends    vLLM OpenAI-compatible HTTP (primary) and transformers 4-bit
                (fallback); each records a distinct generation_backend tag
    sanity      per-model prompt-format probe + degeneracy detectors
    abstention  abstention detection and the metrics compatibility guard
    checkpoint  crash-safe append-only JSONL resume store, repo-format
                materialisation, Drive sync, progress tracker
    campaign    the model x mode x question loop and its provenance guards
"""

from regrag.generation.prompts import (
    ABSTENTION_KEYPHRASE,
    build_closed_book_prompt,
    build_rag_prompt,
)

__all__ = [
    "build_closed_book_prompt",
    "build_rag_prompt",
    "ABSTENTION_KEYPHRASE",
]
