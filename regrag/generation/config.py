"""Campaign configuration for the RegRAG-VN generation harness.

Single source of truth for the model set, the retrieval modes, the sampling
config and the paths the Colab notebook uses. Nothing here imports torch,
transformers or requests, so this module is importable on a CPU-only machine
and inside the fast test suite.

Design constraints this file encodes
------------------------------------
* docs/PROJECT_PLAN.md §3.3 fixes the matrix at **3 models x 3 retrieval modes**,
  top_k = 3. The model list is therefore an explicit config the campaign reads,
  not something discovered at runtime.
* The same section warns that ``Llama-3.2-3B-Instruct`` is English-centric and
  "may only appear as a labelled weak-Vietnamese contrast, never as a peer, or
  it confounds every cross-model number". Selecting it emits a loud warning and
  stamps ``model_set_role`` on every row so it cannot be silently pooled with
  the peers.
* scripts/load_vllm_models.py (branch ``feat/load_model``, commit 3532f36) owns
  the vLLM alias registry. The aliases below mirror it exactly, and each spec
  records whether the alias exists in that registry, because a model that is
  *not* in it can only be served by calling scripts/serve_vllm.sh directly with
  the HF id and ``VLLM_SERVED_MODEL_NAME`` set by hand.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

# --- model-set roles ---------------------------------------------------------
# A "peer" is a model the cross-model numbers may be averaged over.
# Anything else must stay in its own labelled column.
ROLE_PEER = "peer"
ROLE_WEAK_VIETNAMESE = "weak-vietnamese-contrast"
ROLE_NOT_IN_MATRIX = "not-in-experimental-matrix"


@dataclass(frozen=True)
class ModelSpec:
    """One candidate generation model.

    ``alias`` is the vLLM *served model name* (load_vllm_models.py sets
    ``VLLM_SERVED_MODEL_NAME = alias``), which is what an OpenAI-compatible
    client must put in the ``model`` field. ``hf_id`` is the Hugging Face repo
    id, which is what the transformers fallback loads. The two are deliberately
    kept distinct: sending ``hf_id`` to a vLLM server that was started with an
    alias returns a 404, and sending ``alias`` to ``AutoModelForCausalLM`` fails
    to resolve. Confusing them is a silent-wrong-model failure mode.
    """

    alias: str
    hf_id: str
    role: str
    params_b: float
    in_vllm_registry: bool
    note: str = ""

    @property
    def is_peer(self) -> bool:
        return self.role == ROLE_PEER


# Mirrors scripts/load_vllm_models.py:MODELS plus the plan §3.3 alternative
# third model (gemma-3-4b-it), which that registry does not contain.
MODEL_REGISTRY: Dict[str, ModelSpec] = {
    "qwen-7b": ModelSpec(
        alias="qwen-7b",
        hf_id="Qwen/Qwen2.5-7B-Instruct",
        role=ROLE_PEER,
        params_b=7.0,
        in_vllm_registry=True,
    ),
    "qwen-3b": ModelSpec(
        alias="qwen-3b",
        hf_id="Qwen/Qwen2.5-3B-Instruct",
        role=ROLE_PEER,
        params_b=3.0,
        in_vllm_registry=True,
    ),
    "vistral-7b": ModelSpec(
        alias="vistral-7b",
        hf_id="Viet-Mistral/Vistral-7B-Chat",
        role=ROLE_PEER,
        params_b=7.0,
        in_vllm_registry=True,
        note="Architecturally different Vietnamese-capable model (plan §3.3 option A).",
    ),
    "gemma-3-4b-it": ModelSpec(
        alias="gemma-3-4b-it",
        hf_id="google/gemma-3-4b-it",
        role=ROLE_PEER,
        params_b=4.0,
        in_vllm_registry=False,
        note=(
            "Plan §3.3 option B for the third model. NOT in load_vllm_models.py's "
            "registry: serve it via scripts/serve_vllm.sh google/gemma-3-4b-it with "
            "VLLM_SERVED_MODEL_NAME=gemma-3-4b-it."
        ),
    ),
    "llama-3b": ModelSpec(
        alias="llama-3b",
        hf_id="meta-llama/Llama-3.2-3B-Instruct",
        role=ROLE_WEAK_VIETNAMESE,
        params_b=3.0,
        in_vllm_registry=True,
        note=(
            "English-centric. Plan §3.3: include ONLY as a labelled "
            "weak-Vietnamese contrast, never as a peer."
        ),
    ),
    "qwen-1.5b": ModelSpec(
        alias="qwen-1.5b",
        hf_id="Qwen/Qwen2.5-1.5B-Instruct",
        role=ROLE_NOT_IN_MATRIX,
        params_b=1.5,
        in_vllm_registry=True,
        note="In the vLLM registry but not in the plan §3.3 experimental matrix.",
    ),
}

#: The plan §3.3 baseline matrix.
DEFAULT_MODEL_ALIASES: Tuple[str, ...] = ("qwen-7b", "qwen-3b", "vistral-7b")

#: The three retrieval conditions. Order is fixed so runs are comparable.
RETRIEVAL_MODES: Tuple[str, ...] = ("closed_book", "rag_bm25", "rag_dense")

TOP_K: int = 3
DENSE_MODEL_NAME: str = "BAAI/bge-m3"

#: RAG context budget, in characters. Calibrated against the 2026-09-13
#: RunPod preflight on an RTX 3090: a 12,900-char prompt peaked at
#: 13.66 GiB (qwen-7b), 13.27 GiB (qwen-3b) and 16.63 GiB (vistral-7b),
#: because Vietnamese legal text tokenizes at ~1.1-1.5 chars/token - a
#: 12.9k-char prompt is ~11-12k TOKENS, past Vistral's 8,192-token window
#: (Mistral-arch) and into math-backend attention whose L x L matrix
#: dominates peak memory. 7,000 chars keeps even the worst tokenization
#: inside Vistral's window with the 512-token answer to spare, and the
#: measured peak scales to ~7-9 GiB. The budget is in characters (not
#: tokens) so it is tokenizer-agnostic and byte-reproducible; it only
#: bites outlier passages (p95 chunk ~1.4k chars). Truncation and drops
#: are stamped per row.
RAG_CONTEXT_TOTAL_CHARS: int = 7_000
RAG_CONTEXT_PER_PASSAGE_CHARS: int = 3_500
#: A passage whose remaining budget share is below this floor is dropped
#: entirely (a 300-char fragment of a legal article is noise, not evidence).
RAG_CONTEXT_MIN_PASSAGE_CHARS: int = 400

#: The gold CSV marks an unanswerable probe with this exact answer string.
UNANSWERABLE_SENTINEL: str = "Không có trong kho văn bản"

#: retriever_backend tag for the closed-book condition. Self-identifying: it
#: says "no retrieval happened" rather than reusing a real backend's name or
#: leaving the field UNSET (which assert_publishable would refuse).
BACKEND_CLOSED_BOOK: str = "none-closed-book"

#: generation_backend tags. Rows from different backends are NOT
#: interchangeable - sampling, chat-template application and quantization all
#: differ - so the tag is recorded per row and pooling across tags is refused.
BACKEND_VLLM_PREFIX: str = "vllm-serve"
BACKEND_TRANSFORMERS_4BIT: str = "transformers-4bit"
BACKEND_TRANSFORMERS_PREFIX: str = "transformers"


def backend_vllm(alias: str) -> str:
    """Provenance tag for a row produced through a vLLM OpenAI-compatible server."""
    return f"{BACKEND_VLLM_PREFIX}:{alias}"


@dataclass(frozen=True)
class SamplingConfig:
    """Decoding config, recorded verbatim on every row.

    Greedy by default: a hallucination benchmark must be reproducible, and a
    temperature-sampled run cannot be re-verified after the session dies.
    """

    max_new_tokens: int = 512
    temperature: float = 0.0
    top_p: float = 1.0
    do_sample: bool = False
    seed: int = 1234

    def as_dict(self) -> Dict[str, object]:
        return {
            "max_new_tokens": self.max_new_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "do_sample": self.do_sample,
            "seed": self.seed,
        }

    def vllm_payload(self) -> Dict[str, object]:
        """OpenAI-compatible /v1/chat/completions sampling fields."""
        return {
            "max_tokens": self.max_new_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "seed": self.seed,
        }


def select_models(
    aliases: Sequence[str],
    warn_stream=None,
) -> List[ModelSpec]:
    """Resolve aliases to specs, failing loudly on unknown aliases.

    Emits a warning for every selected model that is not a peer, because the
    failure being guarded against is a weak-Vietnamese model being averaged
    into the cross-model headline number without a label.
    """
    stream = warn_stream if warn_stream is not None else sys.stderr
    unknown = [a for a in aliases if a not in MODEL_REGISTRY]
    if unknown:
        raise KeyError(
            f"Unknown model alias(es) {unknown}. Known: {sorted(MODEL_REGISTRY)}. "
            "The model set is explicit config (plan §3.3), not runtime discovery."
        )

    specs = [MODEL_REGISTRY[a] for a in aliases]
    for s in specs:
        if s.role != ROLE_PEER:
            stream.write(
                f"[WARNING] Model '{s.alias}' ({s.hf_id}) has role={s.role}, NOT 'peer'. "
                f"{s.note} It must be reported in its own labelled column and must not "
                "be averaged into the cross-model headline number.\n"
            )
            stream.flush()
        if not s.in_vllm_registry:
            stream.write(
                f"[WARNING] Alias '{s.alias}' is not in scripts/load_vllm_models.py's "
                f"registry. {s.note}\n"
            )
            stream.flush()
    return specs


def validate_matrix(
    model_aliases: Sequence[str],
    modes: Sequence[str],
    questions: int,
) -> Dict[str, int]:
    """Return the expected grid size and check the matrix is what the plan says."""
    bad_modes = [m for m in modes if m not in RETRIEVAL_MODES]
    if bad_modes:
        raise ValueError(
            f"Unknown retrieval mode(s) {bad_modes}. Known: {list(RETRIEVAL_MODES)}."
        )
    return {
        "models": len(model_aliases),
        "modes": len(modes),
        "questions": questions,
        "cells": len(model_aliases) * len(modes) * questions,
    }


# --- paths -------------------------------------------------------------------
# Drive is a checkpoint medium, not an archive (colab skill STEP 3). The Drive
# directory is a single parameter so it is never a magic string repeated
# across cells, and so the notebook can point a run somewhere else without an
# edit in six places.
DEFAULT_DRIVE_SUBDIR: str = "regrag_vn_checkpoints"
DEFAULT_DRIVE_ROOT: str = "/content/drive/MyDrive"
DEFAULT_LOCAL_RESULTS_SUBDIR: str = "results"


def drive_dir(drive_root: str = DEFAULT_DRIVE_ROOT, subdir: str = DEFAULT_DRIVE_SUBDIR) -> str:
    """Absolute Drive directory for checkpoints of this campaign."""
    return f"{drive_root.rstrip('/')}/{subdir.strip('/')}"


def default_sampling() -> SamplingConfig:
    return SamplingConfig()


def summarize_selection(specs: Sequence[ModelSpec]) -> str:
    """Human-readable one-liner per selected model, for the notebook header."""
    lines = []
    for s in specs:
        flag = "" if s.is_peer else f"  <<< role={s.role}"
        lines.append(f"  {s.alias:<16} {s.hf_id:<38} {s.params_b}B{flag}")
    return "\n".join(lines)
