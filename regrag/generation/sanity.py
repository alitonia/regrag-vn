"""Per-model prompt-format sanity probe.

docs/PROJECT_PLAN.md §3.3 names this as the dangerous silent failure of a
multi-model campaign:

    "a wrong chat template produces degenerate output that reads as 'this model
     hallucinates' when the truth is 'we prompted it wrong'"

So before any full campaign run, every model is put through three fixed
questions on **the backend the campaign will actually use**, and the output is
printed in full for a human to eyeball. A model that fails this probe has not
been shown to hallucinate - it has been shown to be mis-prompted, and its
campaign column would be measuring our bug, not its behaviour.

The probe must run on the real backend because that is where the template lives:
vLLM applies the served model's chat template server-side, while the
transformers path applies it client-side via ``tokenizer.apply_chat_template``.
The same messages array can therefore become two different token strings, and
only the backend in use can tell you which one the model saw.

All detectors are pure functions of a string, so the CPU-only test suite can
exercise them without a GPU, a server, or a model download.
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

# --- fixed probe questions ---------------------------------------------------
# Three questions, deliberately spanning the shapes the campaign needs:
#   1. a normal answerable regulatory question -> must produce fluent Vietnamese
#      prose with a legal citation;
#   2. an answerable question about a specific numeric threshold -> must not
#      collapse into repetition;
#   3. a clearly out-of-scope question -> must produce the registered refusal
#      wording, which is the behaviour RQ2 depends on.
PROBE_QUESTIONS: Tuple[Dict[str, str], ...] = (
    {
        "id": "PROBE-1-answerable",
        "question": "Điều kiện để cá nhân mở tài khoản thanh toán tại ngân hàng thương mại ở Việt Nam là gì?",
        "expect": "Fluent Vietnamese prose, cites an Điều/Thông tư, no template markup.",
    },
    {
        "id": "PROBE-2-numeric",
        "question": "Hạn mức rút tiền mặt bằng thẻ ghi nợ tại Việt Nam hiện nay là bao nhiêu?",
        "expect": "A concrete number plus its legal basis; must not degenerate into a repeated token loop.",
    },
    {
        "id": "PROBE-3-out-of-scope",
        "question": "Quy định của Việt Nam về việc phát hành token chứng khoán hóa cho bất động sản là gì?",
        "expect": "A refusal in the registered abstention wording (this is an unanswerable probe).",
    },
)

# --- detector parameters -----------------------------------------------------
#: Chat-template control tokens. Their presence in the OUTPUT means the template
#: was not applied (or was applied twice), so the model saw raw markup.
TEMPLATE_MARKERS: Tuple[str, ...] = (
    "<|im_start|>",
    "<|im_end|>",
    "<|endoftext|>",
    "<|system|>",
    "<|user|>",
    "<|assistant|>",
    "<|pad_token|>",
    "<start_of_turn>",
    "<end_of_turn>",
    "[INST]",
    "[/INST]",
    "<<SYS>>",
    "<</SYS>>",
    "<s>",
    "</s>",
    "<tool_response>",
    "<|fim_middle|>",
)

#: Vietnamese-specific code points. Text with none of these over a long span is
#: almost certainly not Vietnamese.
_VI_CODEPOINTS = set("ăâêôơưđĂÂÊÔƠƯĐáàảãạấầẩẫậắằẳẵặéèẻẽẹếềểễệíìỉĩịóòỏõọốồổỗộớờởỡợúùủũụứừửữựýỳỷỹỵ")

#: Words that only occur in Vietnamese prose. Cheap, high-precision signal.
_VI_FUNCTION_WORDS = (
    "của", "và", "là", "trong", "không", "được", "các", "một", "hoặc", "tại",
    "theo", "khi", "nếu", "phải", "việc", "đối với", "ngân hàng", "quy định",
)

#: Repetition-loop evidence. A model given the wrong template very often falls
#: into an n-gram loop; that is a prompting defect, not hallucination.
_MAX_UNIQUE_RATIO = 0.30
_MIN_TOKENS_FOR_RATIO = 40
_MAX_NGRAM_REPEATS = 8
_NGRAM_N = 4
_MAX_SINGLE_TOKEN_RUN = 12


@dataclass
class ProbeFinding:
    """One detected defect in a probe output."""

    code: str
    detail: str
    severity: str = "blocking"  # "blocking" | "review"

    def as_dict(self) -> Dict[str, str]:
        return {"code": self.code, "detail": self.detail, "severity": self.severity}


@dataclass
class ProbeOutputReport:
    """Detector results for one (model, probe question) pair."""

    probe_id: str
    question: str
    expected: str
    output: str
    findings: List[ProbeFinding] = field(default_factory=list)
    stats: Dict[str, Any] = field(default_factory=dict)
    backend_tag: str = ""
    quant_tag: str = ""
    prompt_exact: bool = False
    prompt_shown: str = ""
    latency_s: float = 0.0
    finish_reason: Optional[str] = None
    abstained: bool = False

    @property
    def ok(self) -> bool:
        return not any(f.severity == "blocking" for f in self.findings)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "probe_id": self.probe_id,
            "question": self.question,
            "expected": self.expected,
            "ok": self.ok,
            "findings": [f.as_dict() for f in self.findings],
            "stats": self.stats,
            "backend_tag": self.backend_tag,
            "quant_tag": self.quant_tag,
            "prompt_exact": self.prompt_exact,
            "latency_s": self.latency_s,
            "finish_reason": self.finish_reason,
            "abstained": self.abstained,
        }


# --- detectors ---------------------------------------------------------------


def detect_template_leak(text: str) -> List[ProbeFinding]:
    """Raw chat-template markup in the output."""
    found = [m for m in TEMPLATE_MARKERS if m in text]
    if found:
        return [
            ProbeFinding(
                "template-markup-in-output",
                f"Output contains chat-template control token(s) {found}. The model saw "
                "raw markup, so the template was not applied (or was applied twice). "
                "This is a prompting defect, not hallucination.",
            )
        ]
    return []


def detect_empty(text: str) -> List[ProbeFinding]:
    if not (text or "").strip():
        return [
            ProbeFinding(
                "empty-output",
                "Model returned an empty/whitespace-only string. A row like this would "
                "be scored as a non-answer and inflate the hallucination rate.",
            )
        ]
    return []


def detect_repetition(text: str) -> List[ProbeFinding]:
    """Degenerate repetition loops."""
    findings: List[ProbeFinding] = []
    tokens = text.split()
    if not tokens:
        return findings

    # 1. Single-token run, e.g. "không không không ..."
    longest_run = 1
    current = 1
    for i in range(1, len(tokens)):
        if tokens[i] == tokens[i - 1]:
            current += 1
            longest_run = max(longest_run, current)
        else:
            current = 1
    if longest_run >= _MAX_SINGLE_TOKEN_RUN:
        findings.append(
            ProbeFinding(
                "repetition-single-token",
                f"Longest identical-token run is {longest_run} (threshold "
                f"{_MAX_SINGLE_TOKEN_RUN}). Classic degenerate decoding.",
            )
        )

    # 2. Repeated n-gram
    if len(tokens) >= _NGRAM_N:
        grams = Counter(tuple(tokens[i : i + _NGRAM_N]) for i in range(len(tokens) - _NGRAM_N + 1))
        top_gram, top_count = grams.most_common(1)[0]
        if top_count >= _MAX_NGRAM_REPEATS:
            findings.append(
                ProbeFinding(
                    "repetition-ngram",
                    f"The {_NGRAM_N}-gram {' '.join(top_gram)!r} repeats {top_count} times "
                    f"(threshold {_MAX_NGRAM_REPEATS}).",
                )
            )

    # 3. Vocabulary collapse
    if len(tokens) >= _MIN_TOKENS_FOR_RATIO:
        ratio = len(set(t.casefold() for t in tokens)) / len(tokens)
        if ratio < _MAX_UNIQUE_RATIO:
            findings.append(
                ProbeFinding(
                    "vocabulary-collapse",
                    f"Only {ratio:.1%} of {len(tokens)} tokens are distinct (threshold "
                    f"{_MAX_UNIQUE_RATIO:.0%}).",
                )
            )

    return findings


def detect_language(text: str) -> List[ProbeFinding]:
    """Flag output that is not Vietnamese.

    Two independent signals must BOTH fail before this is reported, to avoid
    flagging a short correct answer that happens to contain few diacritics:
      * the ratio of Vietnamese-specific code points to alphabetic characters;
      * the presence of Vietnamese function words / domain words.
    """
    stripped = (text or "").strip()
    if len(stripped) < 20:
        return []

    letters = [c for c in stripped if unicodedata.category(c).startswith("L")]
    if len(letters) < 20:
        return []

    vi_chars = sum(1 for c in stripped if c in _VI_CODEPOINTS)
    vi_ratio = vi_chars / len(letters)

    lowered = stripped.casefold()
    hits = sum(1 for w in _VI_FUNCTION_WORDS if w.casefold() in lowered)

    if vi_ratio < 0.01 and hits == 0:
        return [
            ProbeFinding(
                "not-vietnamese",
                f"Vietnamese-specific diacritics are {vi_ratio:.2%} of alphabetic "
                f"characters and 0 of {len(_VI_FUNCTION_WORDS)} Vietnamese function "
                "words appear. The model is answering in English (or another "
                "language), which for a Vietnamese legal benchmark means the prompt "
                "or the model choice is wrong.",
            )
        ]
    if vi_ratio < 0.01 and hits > 0:
        return [
            ProbeFinding(
                "low-diacritic-density",
                f"Diacritic density is only {vi_ratio:.2%} though {hits} Vietnamese "
                "function words are present - possibly diacritics were stripped "
                "somewhere in the pipeline.",
                severity="review",
            )
        ]
    return []


def detect_question_echo(text: str, question: str) -> List[ProbeFinding]:
    """The model repeats the question instead of answering it."""
    q = re.sub(r"\s+", " ", (question or "").strip()).casefold()
    t = re.sub(r"\s+", " ", (text or "").strip()).casefold()
    if len(q) >= 25 and q in t and len(t) < len(q) * 1.6:
        return [
            ProbeFinding(
                "echoes-question",
                "Output is essentially the question repeated back with little or no "
                "answer content - a common symptom of a missing generation prompt "
                "(the model continues the user turn instead of starting the assistant turn).",
            )
        ]
    return []


def detect_truncation(finish_reason: Optional[str], max_new_tokens: int) -> List[ProbeFinding]:
    """The completion hit the token cap."""
    if finish_reason == "length":
        return [
            ProbeFinding(
                "hit-max-new-tokens",
                f"Generation stopped at max_new_tokens={max_new_tokens}. Either the "
                "model is looping or the cap is too low for a cited legal answer; "
                "check the output before trusting any row produced with this setting.",
                severity="review",
            )
        ]
    return []


def output_stats(text: str) -> Dict[str, Any]:
    """Numbers a human uses to eyeball the probe output."""
    tokens = (text or "").split()
    letters = [c for c in (text or "") if unicodedata.category(c).startswith("L")]
    vi_chars = sum(1 for c in (text or "") if c in _VI_CODEPOINTS)
    return {
        "chars": len(text or ""),
        "tokens_whitespace": len(tokens),
        "distinct_token_ratio": (
            round(len(set(t.casefold() for t in tokens)) / len(tokens), 3) if tokens else 0.0
        ),
        "vietnamese_diacritic_ratio": (
            round(vi_chars / len(letters), 4) if letters else 0.0
        ),
        "vi_function_word_hits": sum(
            1 for w in _VI_FUNCTION_WORDS if w.casefold() in (text or "").casefold()
        ),
    }


def check_output(
    text: str,
    question: str = "",
    finish_reason: Optional[str] = None,
    max_new_tokens: int = 512,
) -> List[ProbeFinding]:
    """Run every detector on one output. Pure function."""
    findings: List[ProbeFinding] = []
    findings += detect_empty(text)
    if findings:
        return findings  # nothing else is meaningful for an empty string
    findings += detect_template_leak(text)
    findings += detect_repetition(text)
    findings += detect_language(text)
    if question:
        findings += detect_question_echo(text, question)
    findings += detect_truncation(finish_reason, max_new_tokens)
    return findings


# --- running the probe -------------------------------------------------------


def run_probe(
    backend,
    mode: str = "closed_book",
    questions: Sequence[Dict[str, str]] = PROBE_QUESTIONS,
    abstention_detector=None,
) -> List[ProbeOutputReport]:
    """Run the fixed probe questions through a REAL backend.

    ``backend`` is any ``GenerationBackend``. No model is loaded here, so the
    caller controls whether this touches a GPU or an HTTP server.
    """
    from regrag.generation.prompting import build_messages

    if abstention_detector is None:
        from regrag.generation.abstention import detect_abstention as abstention_detector

    reports: List[ProbeOutputReport] = []
    for pq in questions:
        messages = build_messages(pq["question"], mode)
        rendered = backend.render_prompt(messages)
        gen = backend.generate(messages)
        findings = check_output(
            gen.text,
            question=pq["question"],
            finish_reason=gen.finish_reason,
            max_new_tokens=getattr(backend.sampling, "max_new_tokens", 512),
        )
        verdict = abstention_detector(gen.text)
        reports.append(
            ProbeOutputReport(
                probe_id=pq["id"],
                question=pq["question"],
                expected=pq["expect"],
                output=gen.text,
                findings=findings,
                stats=output_stats(gen.text),
                backend_tag=backend.backend_tag(),
                quant_tag=backend.quant_tag(),
                prompt_exact=rendered.exact,
                prompt_shown=rendered.text,
                latency_s=gen.latency_s,
                finish_reason=gen.finish_reason,
                abstained=verdict.abstained,
            )
        )
    return reports


# --- human-readable rendering ------------------------------------------------


def format_probe_report(
    model_alias: str, reports: Sequence[ProbeOutputReport]
) -> str:
    """Render the probe the way a human must read it before approving a run.

    Prints the prompt EXACTLY as the backend will send/apply it, flags whether
    that string is the literal token sequence, and then prints each output in
    full with its detector findings.
    """
    lines: List[str] = []
    bar = "=" * 78
    lines.append(bar)
    lines.append(f"PROMPT-FORMAT SANITY PROBE  ::  {model_alias}")
    lines.append(bar)

    if reports:
        r0 = reports[0]
        lines.append(f"generation_backend : {r0.backend_tag}")
        lines.append(f"quant_config       : {r0.quant_tag}")
        lines.append(
            "prompt string      : "
            + (
                "EXACT - this is the literal text the model tokenizes"
                if r0.prompt_exact
                else "NOT EXACT - chat template applied server-side; judge the OUTPUT"
            )
        )
        lines.append("")
        lines.append("--- prompt as the backend will render/send it ---")
        lines.append(r0.prompt_shown)
        lines.append("--- end prompt ---")

    all_ok = True
    for r in reports:
        lines.append("")
        lines.append("-" * 78)
        status = "OK" if r.ok else "FAIL"
        if not r.ok:
            all_ok = False
        lines.append(f"[{status}] {r.probe_id}   ({r.latency_s}s, finish={r.finish_reason}, abstained={r.abstained})")
        lines.append(f"  question : {r.question}")
        lines.append(f"  expected : {r.expected}")
        lines.append(f"  stats    : {r.stats}")
        lines.append("  --- model output (verbatim) ---")
        for out_line in (r.output or "<EMPTY>").splitlines() or ["<EMPTY>"]:
            lines.append(f"  | {out_line}")
        lines.append("  --- end output ---")
        if r.findings:
            lines.append("  findings:")
            for f in r.findings:
                lines.append(f"    [{f.severity.upper()}] {f.code}: {f.detail}")
        else:
            lines.append("  findings: none")

    lines.append("")
    lines.append(bar)
    if all_ok:
        lines.append(
            "PROBE PASSED for this backend. A HUMAN must still read the outputs above and\n"
            "confirm they are fluent Vietnamese in the expected format before the campaign runs."
        )
    else:
        lines.append(
            "PROBE FAILED. Do NOT run the campaign for this model on this backend.\n"
            "A failure here means the model was prompted wrong, NOT that it hallucinates.\n"
            "Fix the template/backend and re-probe; an unprobed model produces a column\n"
            "that measures our bug."
        )
    lines.append(bar)
    return "\n".join(lines)


def probe_gate(reports: Sequence[ProbeOutputReport]) -> None:
    """Raise if any probe output has a blocking finding.

    The notebook calls this between the probe cell and the campaign cell so the
    gate cannot be skipped by scrolling past a wall of text.
    """
    from regrag.provenance import ProvenanceError

    bad = [(r.probe_id, f.code) for r in reports for f in r.findings if f.severity == "blocking"]
    if bad:
        raise ProvenanceError(
            f"Prompt-format sanity probe FAILED on {len(bad)} output(s): {bad}. "
            "Refusing to run the campaign: a wrong chat template produces degenerate "
            "output that reads as hallucination. Fix the backend and re-probe."
        )
