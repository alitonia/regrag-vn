"""Generation backends: vLLM OpenAI-compatible HTTP (primary) and transformers 4-bit (fallback).

Why two backends, and why the distinction is recorded on every row
------------------------------------------------------------------
``scripts/load_vllm_models.py`` (branch ``feat/load_model``, commit 3532f36)
starts each benchmark model as a vLLM OpenAI-compatible server and sets
``VLLM_SERVED_MODEL_NAME`` to the **alias**, on port ``base_port + offset``.
That is the "collapse N fragile integrations into 1" path docs/PROJECT_PLAN.md
§3.3 recommends, and it is the harness's primary backend: one HTTP client code
path serves every model, so adding a model is a registry entry rather than a new
tokenizer/quantization integration.

A plain Colab T4 has no vLLM server, so a direct ``transformers`` + 4-bit
BitsAndBytes backend is the fallback.

**Rows from the two backends are not interchangeable.** The chat template is
applied server-side by vLLM's tokenizer but client-side by
``tokenizer.apply_chat_template``; sampling is implemented differently; and the
quantization is the server's business in one path and ours in the other. Two
rows for the same question therefore are not the same measurement, and pooling
them would be a silent substitution of exactly the kind §5 forbids. Every row
carries ``generation_backend``:

    vllm-serve:<alias>       e.g. "vllm-serve:qwen-7b"
    transformers-4bit        verified 4-bit load
    DEGRADED:<reason>        quantization fell back, or precision unrecorded

``resolve_backend()`` never silently switches: an explicit request for a backend
that is unavailable raises. Only ``preference="auto"`` falls back, and it prints
the reason.

vLLM extensibility note
-----------------------
The campaign loop in ``regrag/generation/campaign.py`` talks only to the
``GenerationBackend`` interface below (``backend_tag``, ``quant_tag``,
``render_prompt``, ``generate``, ``describe``). Scaling the model set from 3 to 5
needs NO change to the loop: add the alias to ``config.MODEL_REGISTRY``, start
the servers, and the same HTTP path serves all of them. What would change if a
future backend (e.g. a remote endpoint, or an llama.cpp server) were added is
only a new class implementing those five members plus a branch in
``resolve_backend`` - the loop, the cache keys, the checkpoint format and the
provenance stamping are untouched.
"""

from __future__ import annotations

import json
import os
import sys
import time


# Fragmentation guard for the varying prompt lengths of a 3x3x88 campaign.
# Must be set BEFORE the CUDA caching allocator initialises - i.e. before the
# first CUDA tensor is allocated anywhere in the process, which includes the
# dense-index build in an earlier notebook cell - so it lives at import time,
# not inside load(). (Verified the hard way: set inside load(), it was too
# late and the 2026-09-12 run fragmentated anyway.)
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from regrag.generation.config import (
    BACKEND_TRANSFORMERS_4BIT,
    BACKEND_TRANSFORMERS_PREFIX,
    BACKEND_VLLM_PREFIX,
    SamplingConfig,
    backend_vllm,
)
from regrag.generation.prompting import messages_to_prompt_field
from regrag.provenance import ProvenanceError, degraded, is_degraded, require

# A transport performs one HTTP request and returns (status_code, parsed_body).
# Injectable so the CPU-only test suite can exercise the vLLM path without a
# server, a network, or the `requests` package.
Transport = Callable[[str, str, Optional[Dict[str, Any]], float], Tuple[int, Any]]


def requests_transport(method: str, url: str, payload: Optional[Dict[str, Any]], timeout: float):
    """Default HTTP transport. Imports requests lazily."""
    import requests  # noqa: PLC0415 - lazy so this module imports without requests

    resp = requests.request(method, url, json=payload, timeout=timeout)
    try:
        body = resp.json()
    except Exception:
        body = {"_raw_text": (resp.text or "")[:2000]}
    return resp.status_code, body


# --- GPU assertion -----------------------------------------------------------


def gpu_report(require: bool = True, warn_stream=None) -> Dict[str, Any]:
    """Describe the visible CUDA devices. Fails loudly when a GPU is required.

    Never runs inference and never allocates a tensor: it reads device
    properties only, so it is safe on any machine.
    """
    stream = warn_stream if warn_stream is not None else sys.stderr
    report: Dict[str, Any] = {
        "torch_present": False,
        "cuda_available": False,
        "device_count": 0,
        "device_names": [],
        "device_memory_gb": [],
        "cuda_version": None,
        "gpu_backend_tag": "DEGRADED:no-torch",
    }
    try:
        import torch  # noqa: PLC0415 - lazy
    except ImportError as exc:
        if require:
            raise ProvenanceError(
                "torch is not importable, so no GPU can be verified and no local "
                f"generation backend can run. Original error: {exc}"
            ) from exc
        stream.write("[GPU] torch not importable; reporting CPU-only environment.\n")
        stream.flush()
        return report

    report["torch_present"] = True
    report["cuda_version"] = getattr(torch.version, "cuda", None)
    report["cuda_available"] = bool(torch.cuda.is_available())
    if report["cuda_available"]:
        n = torch.cuda.device_count()
        report["device_count"] = n
        for i in range(n):
            props = torch.cuda.get_device_properties(i)
            report["device_names"].append(props.name)
            report["device_memory_gb"].append(
                round(props.total_memory / (1024**3), 2)
            )
        report["gpu_backend_tag"] = "cuda:" + ";".join(report["device_names"])

    if require and not report["cuda_available"]:
        raise ProvenanceError(
            "No CUDA device is visible. The campaign requires a GPU: on Colab use "
            "Runtime -> Change runtime type -> T4 GPU. Refusing to run generation on "
            "CPU because a 7B model on CPU would take hours per question and would "
            "produce rows whose latency and batching differ from every other row."
        )
    if require and report["device_memory_gb"]:
        smallest = min(report["device_memory_gb"])
        if smallest < 14.0:
            stream.write(
                f"[GPU][WARNING] Smallest visible device has {smallest} GB. A 7B model "
                "in 4-bit needs roughly 5-6 GB of weights plus KV cache; below 14 GB "
                "expect OOM at max_new_tokens=512 with batch>1.\n"
            )
            stream.flush()
    return report


# --- quantization classification (pure, CPU-testable) ------------------------


@dataclass(frozen=True)
class QuantizationVerdict:
    """What precision a model ACTUALLY loaded at, versus what was requested."""

    quant_config: str
    verified_4bit: bool
    degraded: bool
    reason: str


# bitsandbytes packs 4-bit weights into uint8 tensors -- but the FIRST
# parameter of these models is the embedding table, which bnb deliberately
# never quantizes (embeddings and norms stay at checkpoint dtype). A
# first-param dtype probe therefore reports bfloat16 even on a fully
# successful 4-bit load; it cannot distinguish success from fallback. The
# trustworthy evidence is the module census: bnb swaps every nn.Linear for a
# Linear4bit, so "4bit" in the Linear-family module-type names (or uint8
# params, in the legacy no-census path below) is what proves the load.
_BNB_4BIT_DTYPE = "uint8"
# lm_head is occasionally left unconverted (tied/kept-in-hf modules): a real
# Qwen2.5-7B load is 203/204 = 0.995. A true fallback measures 0.0.
_QUANTIZED_SHARE_FLOOR = 0.95


def _census_summary(census: Optional[Dict[str, int]]) -> str:
    if not census:
        return "(no Linear-family modules)"
    return " ".join(
        f"{name}={count}" for name, count in sorted(census.items(), key=lambda kv: -kv[1])
    )


def coerce_input_ids(encoded: Any) -> Any:
    """Unwrap apply_chat_template(tokenize=True) output to the ids tensor.

    Some transformers versions return a bare tensor there; 5.x returns a
    BatchEncoding. Both must feed model.generate() as the input_ids tensor.
    """
    if hasattr(encoded, "keys"):
        try:
            return encoded["input_ids"]
        except KeyError:
            raise ProvenanceError(
                f"apply_chat_template returned {type(encoded).__name__} with keys "
                f"{sorted(encoded.keys())} and no 'input_ids'; cannot build the "
                "generation input."
            ) from None
    return encoded


def classify_quantization(
    requested_4bit: bool,
    quant_config_present: bool,
    observed_param_dtype: str,
    requested_quant_type: str = "nf4",
    compute_dtype: str = "float16",
    linear_census: Optional[Dict[str, int]] = None,
) -> QuantizationVerdict:
    """Classify a loaded model's precision. Pure function - no torch, no model.

    This exists because "we passed load_in_4bit=True" is not evidence: if
    bitsandbytes is unusable on the runtime, transformers loads the model in
    full precision and the run continues happily at 4x the VRAM, either OOMing
    later or - worse - succeeding slowly while every row claims 4-bit.

    ``observed_param_dtype`` is the FIRST parameter's dtype -- for these models
    the embedding table, which bnb never quantizes -- so it only names the
    unquantized payload precision. ``linear_census`` (module type name ->
    count over Linear-family modules) is the decisive evidence and should
    always be supplied by a real load.
    """
    dtype = (observed_param_dtype or "unknown").replace("torch.", "")

    if not requested_4bit:
        return QuantizationVerdict(
            quant_config=f"full-{dtype}",
            verified_4bit=False,
            degraded=False,
            reason="4-bit was not requested; model loaded at observed precision.",
        )

    if linear_census is None:
        # Legacy no-census path: judge from the first-param dtype. Biased
        # degraded-by-default (only uint8 passes) because this probe cannot
        # see the Linear layers where 4-bit actually lives.
        if not quant_config_present:
            return QuantizationVerdict(
                quant_config=degraded(f"quantization-not-applied-loaded-{dtype}"),
                verified_4bit=False,
                degraded=True,
                reason=(
                    "load_in_4bit was requested but model.config.quantization_config "
                    "is absent, so transformers did not quantize. Most likely "
                    "bitsandbytes is missing or unsupported on this runtime."
                ),
            )
        if dtype != _BNB_4BIT_DTYPE:
            return QuantizationVerdict(
                quant_config=degraded(f"quantization-fell-back-to-{dtype}"),
                verified_4bit=False,
                degraded=True,
                reason=(
                    f"quantization_config is present but the first parameter is "
                    f"{dtype}, not {_BNB_4BIT_DTYPE}. No module census was "
                    "supplied, so 4-bit could not be verified from the Linear "
                    "layers; refusing to assume it took effect."
                ),
            )
        return QuantizationVerdict(
            quant_config=f"bnb-4bit-{requested_quant_type}+{compute_dtype}",
            verified_4bit=True,
            degraded=False,
            reason="4-bit verified from parameter dtype (legacy probe, no census).",
        )

    total = sum(linear_census.values())
    quantized = sum(
        count for name, count in linear_census.items() if "4bit" in name.lower()
    )
    census_note = _census_summary(linear_census)

    if total == 0:
        return QuantizationVerdict(
            quant_config=degraded("quantization-unverifiable-no-linear-modules"),
            verified_4bit=False,
            degraded=True,
            reason=(
                "load_in_4bit was requested but the model exposes no "
                "Linear-family modules, so a 4-bit load cannot be verified."
            ),
        )

    if not quant_config_present:
        return QuantizationVerdict(
            quant_config=degraded(f"quantization-not-applied-loaded-{dtype}"),
            verified_4bit=False,
            degraded=True,
            reason=(
                "load_in_4bit was requested but model.config.quantization_config is "
                f"absent, so transformers did not quantize. Linear census: "
                f"{census_note}. Most likely bitsandbytes is missing or "
                "unsupported on this runtime."
            ),
        )

    share = quantized / total
    if share >= _QUANTIZED_SHARE_FLOOR:
        return QuantizationVerdict(
            quant_config=f"bnb-4bit-{requested_quant_type}+{compute_dtype}",
            verified_4bit=True,
            degraded=False,
            reason=(
                f"{quantized}/{total} Linear-family modules are 4-bit "
                f"({census_note}); embeddings/norms stay {dtype} by design."
            ),
        )

    if quantized == 0:
        return QuantizationVerdict(
            quant_config=degraded(f"quantization-fell-back-to-{dtype}"),
            verified_4bit=False,
            degraded=True,
            reason=(
                f"quantization_config is present but no Linear module was "
                f"converted to a 4-bit type; parameters are {dtype}. Linear "
                f"census: {census_note}. The 4-bit load did not take effect."
            ),
        )

    return QuantizationVerdict(
        quant_config=degraded(f"partial-4bit-{quantized}of{total}-linear-modules"),
        verified_4bit=False,
        degraded=True,
        reason=(
            f"only {quantized}/{total} Linear-family modules are 4-bit "
            f"({census_note}); a partial quantization is a mixed-precision "
            "load, not a verified 4-bit one."
        ),
    )


# --- shared result types -----------------------------------------------------


@dataclass(frozen=True)
class RenderedPrompt:
    """What the sanity probe shows a human.

    ``exact`` is the honesty flag. True means ``text`` is the literal string the
    model tokenizes. False means the template is applied somewhere the client
    cannot see (a vLLM server), so ``text`` is the request payload and the human
    must judge the model's OUTPUT format instead of the prompt string.
    """

    text: str
    exact: bool
    applied_by: str
    note: str = ""
    messages: List[Dict[str, str]] = field(default_factory=list)

    def prompt_field(self) -> str:
        """Value stored in GenerationResult.prompt."""
        if self.exact:
            return self.text
        marker = f"[{self.applied_by} | NOT the literal token string]"
        return messages_to_prompt_field(self.messages, marker)


@dataclass(frozen=True)
class Generation:
    """One model completion plus the metadata worth recording."""

    text: str
    raw: str
    latency_s: float
    finish_reason: Optional[str] = None
    usage: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None


class GenerationBackend(ABC):
    """The only interface the campaign loop and the sanity probe depend on."""

    kind: str = "abstract"

    @abstractmethod
    def backend_tag(self) -> str:
        """Provenance value recorded as generation_backend on every row."""

    @abstractmethod
    def quant_tag(self) -> str:
        """Provenance value recorded as quant_config on every row."""

    @abstractmethod
    def render_prompt(self, messages: Sequence[Dict[str, str]]) -> RenderedPrompt:
        """Show what the model will actually be given."""

    @abstractmethod
    def generate(self, messages: Sequence[Dict[str, str]]) -> Generation:
        """Produce one completion."""

    @abstractmethod
    def describe(self) -> Dict[str, Any]:
        """Manifest-ready description of this backend."""

    def close(self) -> None:
        """Release resources. Default no-op."""

    # Convenience for the campaign loop.
    def supports_exact_prompt(self) -> bool:
        return False


# --- vLLM OpenAI-compatible HTTP backend (PRIMARY) ---------------------------


class VLLMHttpBackend(GenerationBackend):
    """Talks to ``{base_url}/chat/completions`` with ``model=<served alias>``.

    ``base_url`` must include the ``/v1`` prefix that ``vllm serve`` exposes,
    e.g. ``http://127.0.0.1:8000/v1``.
    """

    kind = "vllm-serve"

    def __init__(
        self,
        base_url: str,
        served_alias: str,
        hf_id: str,
        sampling: Optional[SamplingConfig] = None,
        quantization_declared: Optional[str] = None,
        transport: Optional[Transport] = None,
        timeout_s: float = 600.0,
        extra_body: Optional[Dict[str, Any]] = None,
        warn_stream=None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.served_alias = served_alias
        self.hf_id = hf_id
        self.sampling = sampling or SamplingConfig()
        self.quantization_declared = quantization_declared
        self._transport = transport or requests_transport
        self.timeout_s = timeout_s
        self.extra_body = dict(extra_body or {})
        self._stream = warn_stream if warn_stream is not None else sys.stderr
        self._served_ids: Optional[List[str]] = None

        if not self.base_url:
            raise ProvenanceError("VLLMHttpBackend needs a non-empty base_url.")

    # -- provenance ----------------------------------------------------------
    def backend_tag(self) -> str:
        return backend_vllm(self.served_alias)

    def quant_tag(self) -> str:
        """Precision is the SERVER's business; we record what was declared.

        A vLLM client cannot verify the server's weights, so claiming "4bit"
        here would be an assumption dressed as a measurement. When the notebook
        does not state what the server was started with, the tag says so
        explicitly instead of defaulting to something plausible.
        """
        if self.quantization_declared:
            return f"vllm-server:{self.quantization_declared}"
        return "UNRECORDED:vllm-server-managed"

    def describe(self) -> Dict[str, Any]:
        return {
            "kind": self.kind,
            "generation_backend": self.backend_tag(),
            "quant_config": self.quant_tag(),
            "base_url": self.base_url,
            "served_alias": self.served_alias,
            "hf_id": self.hf_id,
            "template_applied_by": "vllm-server-tokenizer",
            "prompt_exact": False,
            "sampling": self.sampling.as_dict(),
            "served_ids_observed": self._served_ids,
        }

    # -- HTTP ----------------------------------------------------------------
    def _call(self, method: str, path: str, payload: Optional[Dict[str, Any]] = None):
        url = f"{self.base_url}{path}"
        try:
            status, body = self._transport(method, url, payload, self.timeout_s)
        except Exception as exc:  # connection refused, timeout, DNS
            raise ProvenanceError(
                f"vLLM request {method} {url} failed: {type(exc).__name__}: {exc}. "
                "Is the server still alive? A dead server mid-campaign must stop "
                "the run rather than produce empty answers."
            ) from exc
        return status, body

    def served_model_ids(self) -> List[str]:
        """GET /v1/models and return the served ids."""
        status, body = self._call("GET", "/models")
        if status != 200:
            raise ProvenanceError(
                f"GET {self.base_url}/models returned HTTP {status}: {str(body)[:400]}"
            )
        data = body.get("data") if isinstance(body, dict) else None
        if not isinstance(data, list):
            raise ProvenanceError(
                f"Unexpected /models response from {self.base_url}: {str(body)[:400]}"
            )
        ids = [str(d.get("id")) for d in data if isinstance(d, dict) and d.get("id")]
        self._served_ids = ids
        return ids

    def verify_serves_alias(self) -> List[str]:
        """Refuse to generate unless the server really serves THIS alias.

        Guards the silent-wrong-model failure: a server left running from the
        previous model on the same port answers requests perfectly well and
        produces fluent Vietnamese - for the wrong model. Every row would then
        be attributed to a model that never saw the question.
        """
        ids = self.served_model_ids()
        if self.served_alias not in ids:
            raise ProvenanceError(
                f"vLLM server at {self.base_url} serves {ids}, not the requested "
                f"alias {self.served_alias!r}. Refusing to generate: rows would be "
                "attributed to a model that is not running. Start it with "
                f"scripts/load_vllm_models.py --model {self.served_alias} or check "
                "--base-port."
            )
        return ids

    def is_reachable(self) -> bool:
        try:
            self.served_model_ids()
            return True
        except ProvenanceError:
            return False

    # -- generation ----------------------------------------------------------
    def render_prompt(self, messages: Sequence[Dict[str, str]]) -> RenderedPrompt:
        msgs = [dict(m) for m in messages]
        return RenderedPrompt(
            text=json.dumps(msgs, ensure_ascii=False, indent=2),
            exact=False,
            applied_by=f"vllm-server-tokenizer({self.hf_id})",
            note=(
                "The chat template is applied SERVER-SIDE by vLLM using the "
                f"tokenizer of {self.hf_id}. The client cannot render the literal "
                "token string, so judge the model's OUTPUT format below, not this "
                "payload. What is shown is exactly the messages array that is sent."
            ),
            messages=msgs,
        )

    def generate(self, messages: Sequence[Dict[str, str]]) -> Generation:
        msgs = [dict(m) for m in messages]
        payload: Dict[str, Any] = {
            "model": self.served_alias,
            "messages": msgs,
            **self.sampling.vllm_payload(),
        }
        payload.update(self.extra_body)

        t0 = time.time()
        status, body = self._call("POST", "/chat/completions", payload)
        latency = time.time() - t0

        if status != 200:
            raise ProvenanceError(
                f"POST {self.base_url}/chat/completions returned HTTP {status}: "
                f"{str(body)[:600]}"
            )
        if not isinstance(body, dict):
            raise ProvenanceError(f"Non-JSON-object response from vLLM: {str(body)[:300]}")

        choices = body.get("choices") or []
        if not choices:
            raise ProvenanceError(f"vLLM returned no choices: {str(body)[:400]}")
        choice = choices[0]
        message = choice.get("message") or {}
        text = message.get("content")
        if text is None:
            raise ProvenanceError(
                f"vLLM choice has no message.content: {str(choice)[:400]}. "
                "Refusing to record an empty row as a successful generation."
            )
        return Generation(
            text=text,
            raw=text,
            latency_s=round(latency, 3),
            finish_reason=choice.get("finish_reason"),
            usage=body.get("usage") or {},
        )


def wait_for_vllm(
    base_url: str,
    served_alias: str,
    hf_id: str = "",
    timeout_s: float = 900.0,
    poll_s: float = 10.0,
    transport: Optional[Transport] = None,
    quantization_declared: Optional[str] = None,
    sampling: Optional[SamplingConfig] = None,
    warn_stream=None,
) -> VLLMHttpBackend:
    """Poll until the server is up AND serves the expected alias."""
    stream = warn_stream if warn_stream is not None else sys.stderr
    backend = VLLMHttpBackend(
        base_url=base_url,
        served_alias=served_alias,
        hf_id=hf_id,
        sampling=sampling,
        quantization_declared=quantization_declared,
        transport=transport,
        warn_stream=stream,
    )
    deadline = time.time() + timeout_s
    last = ""
    while time.time() < deadline:
        try:
            ids = backend.verify_serves_alias()
            stream.write(f"[vLLM] {base_url} is serving {ids} - alias {served_alias!r} confirmed.\n")
            stream.flush()
            return backend
        except ProvenanceError as exc:
            last = str(exc)
            stream.write(f"[vLLM] waiting for {base_url} ({served_alias})... {last[:120]}\n")
            stream.flush()
            time.sleep(poll_s)
    raise ProvenanceError(
        f"vLLM server at {base_url} did not serve alias {served_alias!r} within "
        f"{timeout_s}s. Last error: {last}"
    )


# --- transformers 4-bit backend (FALLBACK) -----------------------------------


# --- GPU residency: one quantized model on the card at a time ----------------
#
# A 16 GB T4 fits roughly two 4-bit 7B models, so loading every peer up front
# OOMs mid-campaign once generation KV-cache grows. Loaded transformers
# backends register below; a new load FIFO-evicts the oldest resident (weights
# freed, tokenizer and provenance tags kept, lazy reload on next use).
# Override with REGRAG_MAX_RESIDENT_BACKENDS (values < 1 disable the cap).
_RESIDENT_QUANT_BACKENDS: List["TransformersQuantBackend"] = []


def _max_resident_backends() -> int:
    try:
        return int(os.environ.get("REGRAG_MAX_RESIDENT_BACKENDS", "1"))
    except ValueError:
        return 1


def _evict_to_make_room(
    registry: List["TransformersQuantBackend"],
    cap: int,
    arriving_hf_id: str,
    stream,
) -> None:
    """FIFO-evict resident backends until one slot is free. cap < 1 = no cap."""
    if cap < 1:
        return
    while len(registry) >= cap and registry:
        victim = registry.pop(0)
        victim._evict_model()
        stream.write(
            f"[BACKEND] GPU residency cap {cap}: freed {victim.hf_id} "
            f"before loading {arriving_hf_id}; it will lazily reload from "
            "cache if used again.\n"
        )
        stream.flush()


class TransformersQuantBackend(GenerationBackend):
    """Direct HF load with 4-bit BitsAndBytes, verified rather than assumed.

    ``load()`` must be called before use; it is separated from ``__init__`` so
    the object can be constructed and inspected (and its provenance tags read)
    without touching the GPU or downloading weights.
    """

    kind = "transformers"

    def __init__(
        self,
        hf_id: str,
        sampling: Optional[SamplingConfig] = None,
        load_in_4bit: bool = True,
        bnb_4bit_quant_type: str = "nf4",
        compute_dtype_name: str = "float16",
        bnb_4bit_use_double_quant: bool = True,
        device_map: str = "auto",
        strict_quantization: bool = True,
        warn_stream=None,
    ) -> None:
        self.hf_id = hf_id
        self.sampling = sampling or SamplingConfig()
        self.load_in_4bit = load_in_4bit
        self.bnb_4bit_quant_type = bnb_4bit_quant_type
        self.compute_dtype_name = compute_dtype_name
        self.bnb_4bit_use_double_quant = bnb_4bit_use_double_quant
        self.device_map = device_map
        self.strict_quantization = strict_quantization
        self._stream = warn_stream if warn_stream is not None else sys.stderr

        self.model = None
        self.tokenizer = None
        self._torch = None
        self._evicted = False
        self.quant_verdict: Optional[QuantizationVerdict] = None
        self.template_name: Optional[str] = None

    # -- provenance ----------------------------------------------------------
    def backend_tag(self) -> str:
        if self.quant_verdict is None:
            # Not loaded yet: say so rather than claiming a precision.
            return f"{BACKEND_TRANSFORMERS_PREFIX}:not-loaded"
        if self.quant_verdict.verified_4bit:
            return BACKEND_TRANSFORMERS_4BIT
        if self.quant_verdict.degraded:
            return self.quant_verdict.quant_config  # already "DEGRADED:<reason>"
        return f"{BACKEND_TRANSFORMERS_PREFIX}-{self.quant_verdict.quant_config}"

    def quant_tag(self) -> str:
        if self.quant_verdict is None:
            return "UNRECORDED:model-not-loaded"
        return self.quant_verdict.quant_config

    def describe(self) -> Dict[str, Any]:
        return {
            "kind": self.kind,
            "generation_backend": self.backend_tag(),
            "quant_config": self.quant_tag(),
            "hf_id": self.hf_id,
            "template_applied_by": "tokenizer.apply_chat_template",
            "prompt_exact": True,
            "chat_template_present": self.template_name,
            "sampling": self.sampling.as_dict(),
            "requested": {
                "load_in_4bit": self.load_in_4bit,
                "bnb_4bit_quant_type": self.bnb_4bit_quant_type,
                "compute_dtype": self.compute_dtype_name,
                "double_quant": self.bnb_4bit_use_double_quant,
                "device_map": self.device_map,
            },
            "quantization_reason": (
                self.quant_verdict.reason if self.quant_verdict else "not loaded"
            ),
        }

    # -- loading -------------------------------------------------------------
    def load(self):
        """Load tokenizer + model. Raises rather than degrading silently."""
        if self.model is not None:
            return self  # idempotent: safe to call again after a crash/re-run
        _evict_to_make_room(
            _RESIDENT_QUANT_BACKENDS,
            _max_resident_backends(),
            self.hf_id,
            self._stream,
        )
        try:
            import torch
            import transformers
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise ProvenanceError(
                "The transformers fallback backend needs torch and transformers "
                f"installed. Original error: {exc}"
            ) from exc

        if self.load_in_4bit:
            try:
                import bitsandbytes  # noqa: F401
            except ImportError as exc:
                raise ProvenanceError(
                    "4-bit quantization was requested but 'bitsandbytes' is not "
                    "importable. Refusing to fall back to fp16/fp32 silently: that "
                    "would 4x VRAM and record a precision the row does not have. "
                    "Install bitsandbytes, or set load_in_4bit=False explicitly and "
                    f"accept the 'full-<dtype>' tag. Original error: {exc}"
                ) from exc

        self._torch = torch
        if not torch.cuda.is_available():
            raise ProvenanceError(
                f"No CUDA device visible for transformers backend loading {self.hf_id}. "
                "Refusing to load a multi-billion-parameter model on CPU."
            )

        quantization_config = None
        if self.load_in_4bit:
            from transformers import BitsAndBytesConfig

            compute_dtype = getattr(torch, self.compute_dtype_name, torch.float16)
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type=self.bnb_4bit_quant_type,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=self.bnb_4bit_use_double_quant,
            )

        self.tokenizer = AutoTokenizer.from_pretrained(self.hf_id)
        load_kwargs: Dict[str, Any] = {"device_map": self.device_map}
        if quantization_config is not None:
            # Do NOT also pass torch_dtype/dtype here: on current transformers
            # an explicit dtype alongside quantization_config makes the plain
            # fp16 load win and no Linear is ever converted to Linear4bit
            # (observed on Colab T4 2026-09-12: params came up float16 and the
            # strict guard stopped the run). The compute dtype lives inside
            # BitsAndBytesConfig(bnb_4bit_compute_dtype=...).
            load_kwargs["quantization_config"] = quantization_config
        else:
            load_kwargs["torch_dtype"] = getattr(
                torch, self.compute_dtype_name, torch.float16)
        self.model = AutoModelForCausalLM.from_pretrained(self.hf_id, **load_kwargs)
        self.model.eval()

        if self.tokenizer.chat_template is None:
            # Not fatal, but it means apply_chat_template will use a generic
            # default and the prompt will NOT match what the model was trained
            # on. That is precisely the "we prompted it wrong" failure the plan
            # warns about, so it is surfaced rather than swallowed.
            self._stream.write(
                f"[WARNING] {self.hf_id} ships no chat_template; "
                "apply_chat_template will use a generic default and the prompt may "
                "not match the model's training format.\n"
            )
            self._stream.flush()
            self.template_name = "generic-default"
        else:
            self.template_name = "repo-chat-template"

        observed_dtype = "unknown"
        try:
            observed_dtype = str(next(self.model.parameters()).dtype)
        except StopIteration:
            observed_dtype = "no-parameters"

        # The decisive evidence: bnb swaps nn.Linear -> Linear4bit on a
        # successful 4-bit load. The first parameter (embed_tokens) is never
        # quantized and cannot serve as the probe.
        linear_census: Dict[str, int] = {}
        dtype_histogram: Dict[str, int] = {}
        for module in self.model.modules():
            type_name = type(module).__name__
            if "linear" in type_name.lower():
                linear_census[type_name] = linear_census.get(type_name, 0) + 1
        for param in self.model.parameters():
            param_dtype = str(param.dtype).replace("torch.", "")
            dtype_histogram[param_dtype] = dtype_histogram.get(param_dtype, 0) + 1

        self.quant_verdict = classify_quantization(
            requested_4bit=self.load_in_4bit,
            quant_config_present=getattr(self.model.config, "quantization_config", None) is not None,
            observed_param_dtype=observed_dtype,
            requested_quant_type=self.bnb_4bit_quant_type,
            compute_dtype=self.compute_dtype_name,
            linear_census=linear_census,
        )
        if self.quant_verdict.verified_4bit:
            self._stream.write(
                f"[OK] 4-bit verified for {self.hf_id}: "
                f"{self.quant_verdict.reason} Param dtypes: {dtype_histogram}.\n"
            )
            self._stream.flush()

        if self.quant_verdict.degraded:
            try:
                from importlib.metadata import version as _pkg_version
                env = (f" [torch {torch.__version__}, "
                       f"transformers {_pkg_version('transformers')}, "
                       f"bitsandbytes {_pkg_version('bitsandbytes')}]")
            except Exception:
                env = ""
            msg = (
                f"QUANTIZATION DID NOT TAKE EFFECT for {self.hf_id}: "
                f"{self.quant_verdict.reason} Tag recorded as "
                f"{self.quant_verdict.quant_config!r}.{env} "
                f"Param dtypes: {dtype_histogram}."
            )
            self._stream.write(f"[ERROR] {msg}\n")
            self._stream.flush()
            if self.strict_quantization:
                raise ProvenanceError(msg + " strict_quantization=True, so the run stops.")
        self._evicted = False
        if self not in _RESIDENT_QUANT_BACKENDS:
            _RESIDENT_QUANT_BACKENDS.append(self)
        return self

    # -- generation ----------------------------------------------------------
    def render_prompt(self, messages: Sequence[Dict[str, str]]) -> RenderedPrompt:
        if self.tokenizer is None:
            raise ProvenanceError(
                f"TransformersQuantBackend for {self.hf_id} is not loaded; call load() first."
            )
        msgs = [dict(m) for m in messages]
        text = self.tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True
        )
        return RenderedPrompt(
            text=text,
            exact=True,
            applied_by=f"tokenizer.apply_chat_template({self.hf_id})",
            note=(
                "This IS the literal string the model will tokenize, rendered by the "
                "checkpoint's own chat template."
            ),
            messages=msgs,
        )

    def generate(self, messages: Sequence[Dict[str, str]]) -> Generation:
        if self.model is None or self.tokenizer is None:
            if self._evicted and self.tokenizer is not None:
                self._stream.write(
                    f"[BACKEND] {self.hf_id} was evicted for GPU residency; "
                    "reloading weights from cache.\n"
                )
                self.load()
            else:
                raise ProvenanceError(
                    f"TransformersQuantBackend for {self.hf_id} is not loaded; call load() first."
                )
        torch = self._torch
        msgs = [dict(m) for m in messages]
        encoded = self.tokenizer.apply_chat_template(
            msgs,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        )
        # transformers 5.x returns a BatchEncoding here; older versions a bare
        # tensor. Either way generate() needs the input_ids tensor.
        input_ids = coerce_input_ids(encoded).to(self.model.device)
        in_len = input_ids.shape[-1]

        gen_kwargs: Dict[str, Any] = {
            "max_new_tokens": self.sampling.max_new_tokens,
            "do_sample": self.sampling.do_sample,
            "pad_token_id": self.tokenizer.eos_token_id,
        }
        if self.sampling.do_sample:
            gen_kwargs["temperature"] = self.sampling.temperature
            gen_kwargs["top_p"] = self.sampling.top_p

        t0 = time.time()
        with torch.inference_mode():
            out = self.model.generate(input_ids, **gen_kwargs)
        latency = time.time() - t0

        new_tokens = out[0][in_len:]
        text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        raw = self.tokenizer.decode(new_tokens, skip_special_tokens=False)
        n_new = int(new_tokens.shape[0])
        finish = "length" if n_new >= self.sampling.max_new_tokens else "stop"
        return Generation(
            text=text,
            raw=raw,
            latency_s=round(latency, 3),
            finish_reason=finish,
            usage={"prompt_tokens": int(in_len), "completion_tokens": n_new},
        )

    def _evict_model(self) -> None:
        """Registry eviction: free the weights only. The tokenizer and the
        recorded quantization verdict stay, so prompts can still be rendered
        and generate() can lazily reload."""
        if self.model is None:
            return
        self.model = None
        self._evicted = True
        if self._torch is not None:
            try:
                self._torch.cuda.empty_cache()
            except Exception:
                pass

    def close(self) -> None:
        """Free VRAM before the next model loads. Without this, loading model N+1
        OOMs on a 16 GB T4 even though model N would have fit alone."""
        if self in _RESIDENT_QUANT_BACKENDS:
            _RESIDENT_QUANT_BACKENDS.remove(self)
        self.model = None
        self.tokenizer = None
        self._evicted = False
        if self._torch is not None:
            try:
                self._torch.cuda.empty_cache()
            except Exception:
                pass


# --- backend resolution ------------------------------------------------------


def resolve_backend(
    spec,
    preference: str = "auto",
    vllm_base_url: str = "http://127.0.0.1:8000/v1",
    vllm_timeout_s: float = 900.0,
    quantization_declared: Optional[str] = None,
    sampling: Optional[SamplingConfig] = None,
    transport: Optional[Transport] = None,
    warn_stream=None,
    strict_quantization: bool = True,
    load_transformers: bool = True,
) -> GenerationBackend:
    """Pick the backend the campaign will actually use, and say which it picked.

    ``preference``:
      * ``"vllm"``         - require a reachable server serving the alias. NO
                             fallback: silently switching to transformers would
                             change the chat-template path and produce rows that
                             must not be pooled with vLLM rows.
      * ``"transformers"`` - require a GPU and load locally.
      * ``"auto"``         - try vLLM, fall back to transformers, printing why.

    ``load_transformers=False`` returns an unloaded TransformersQuantBackend so
    the notebook can inspect provenance tags before spending VRAM.
    """
    stream = warn_stream if warn_stream is not None else sys.stderr
    if preference not in ("auto", "vllm", "transformers"):
        raise ValueError(f"Unknown backend preference {preference!r}.")

    def try_vllm() -> Optional[VLLMHttpBackend]:
        backend = VLLMHttpBackend(
            base_url=vllm_base_url,
            served_alias=spec.alias,
            hf_id=spec.hf_id,
            sampling=sampling,
            quantization_declared=quantization_declared,
            transport=transport,
        )
        try:
            ids = backend.verify_serves_alias()
        except ProvenanceError as exc:
            stream.write(f"[BACKEND] vLLM probe at {vllm_base_url} failed: {exc}\n")
            stream.flush()
            return None
        stream.write(
            f"[BACKEND] Using vLLM server at {vllm_base_url} (serves {ids}); "
            f"generation_backend={backend.backend_tag()!r}\n"
        )
        if backend.quant_tag().startswith("UNRECORDED"):
            stream.write(
                "[BACKEND][WARNING] Quantization for this vLLM server was not declared, "
                f"so rows are tagged {backend.quant_tag()!r}. The client cannot verify "
                "server-side weights; pass quantization_declared= (e.g. 'bitsandbytes-4bit' "
                "or 'fp16') to record it.\n"
            )
        stream.flush()
        return backend

    if preference in ("auto", "vllm"):
        backend = try_vllm()
        if backend is not None:
            return backend
        if preference == "vllm":
            raise ProvenanceError(
                f"Backend preference is 'vllm' but no server at {vllm_base_url} serves "
                f"alias {spec.alias!r}. Not falling back to transformers: the two paths "
                "apply the chat template differently and their rows must not be pooled. "
                "Start the server (scripts/load_vllm_models.py --model "
                f"{spec.alias}) or set preference='auto' deliberately."
            )
        stream.write(
            "[BACKEND] Falling back to the local transformers backend because no vLLM "
            "server was reachable. Rows will be tagged 'transformers-*' and must NOT be "
            "pooled with 'vllm-serve:*' rows.\n"
        )
        stream.flush()

    gpu = gpu_report(require=True, warn_stream=stream)
    stream.write(
        f"[BACKEND] Using local transformers backend for {spec.hf_id} on "
        f"{gpu['device_names']}.\n"
    )
    stream.flush()
    tb = TransformersQuantBackend(
        hf_id=spec.hf_id,
        sampling=sampling,
        strict_quantization=strict_quantization,
        warn_stream=stream,
    )
    if load_transformers:
        tb.load()
    return tb
