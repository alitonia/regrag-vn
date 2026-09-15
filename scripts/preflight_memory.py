"""Worst-case memory preflight for the RegRAG-VN campaign (added 2026-09-12).

Three mid-campaign OOMs on 2026-09-12 each had a different mechanism. This
script proves the memory envelope of the whole 3x3x88 grid in ~10 minutes by
forcing one generation per model at the maximum prompt the RAG_CONTEXT_*
budget can produce. Peak memory is driven by prompt length, not call count
(rows 1-171 only ever failed on long-prompt rows), so surviving the worst
case bounds every real row.

Run it in Colab AFTER cell 10 (backends resolved) and BEFORE the campaign:

    from scripts.preflight_memory import preflight_memory
    preflight_memory(BACKENDS)

Cost: three cached model loads + three greedy 512-token generations.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List

from regrag.generation.config import RAG_CONTEXT_TOTAL_CHARS

# The budget bounds the context at RAG_CONTEXT_TOTAL_CHARS; add the template,
# instructions, and a long question to get the true worst-case prompt length.
_TEMPLATE_HEADROOM_CHARS = 900
MAX_PROMPT_CHARS = RAG_CONTEXT_TOTAL_CHARS + _TEMPLATE_HEADROOM_CHARS

# Keep this much slack at peak for the CUDA context and allocator overheads
# that max_memory_allocated() cannot see. The card size is read from torch at
# runtime - hardcoding the T4's 14.56 GiB misfired the gate on a 24 GB 3090
# (2026-09-13: peaks of 13.7-16.6 GiB were actually inside that card).
_PASS_HEADROOM_GIB = 1.5


def _gpu_total_gib() -> float:
    import torch

    try:
        return torch.cuda.get_device_properties(0).total_memory / 2**30
    except Exception:
        return 14.56  # T4-class fallback if the properties read fails

_FILLER = (
    "Tổ chức cung ứng dịch vụ trung gian thanh toán phải đảm bảo an toàn, "
    "bảo mật và tuân thủ các quy định của Ngân hàng Nhà nước Việt Nam về hạn "
    "mức giao dịch điện tử trong ngày đối với từng loại giao dịch và từng "
    "kênh thanh toán, đồng thời thông báo cho khách hàng về các hạn mức này "
    "trước khi thực hiện giao dịch. "
)


def build_worst_case_messages(max_chars: int = MAX_PROMPT_CHARS) -> List[Dict[str, str]]:
    """One user message at the largest length the budgeted prompt builder can
    emit, in realistic Vietnamese so chars/token stays honest per tokenizer."""
    body = _FILLER * (max_chars // len(_FILLER) + 1)
    payload = ("Câu hỏi: " + body)[:max_chars]
    return [{"role": "user", "content": payload}]


def preflight_memory(backends: Dict[str, Any]) -> Dict[str, Any]:
    """Generate once per backend at the worst-case prompt; report peak VRAM."""
    import torch

    budget_gib = _gpu_total_gib()
    report: Dict[str, Any] = {}
    messages = build_worst_case_messages()
    print(f"[PREFLIGHT] worst-case prompt: {len(messages[0]['content'])} chars; "
          f"card budget {budget_gib:.2f} GiB")
    for alias, backend in backends.items():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
        t0 = time.time()
        try:
            gen = backend.generate(messages)
            ok = bool(gen.text and gen.text.strip())
            peak_gib = torch.cuda.max_memory_allocated() / 2**30
            free_gib = budget_gib - peak_gib
            verdict = "PASS" if (ok and free_gib >= _PASS_HEADROOM_GIB) else "WARN"
            report[alias] = {
                "generated_chars": len(gen.text),
                "peak_gib": round(peak_gib, 2),
                "free_gib": round(free_gib, 2),
                "verdict": verdict,
            }
            print(
                f"[PREFLIGHT] {alias:<12} {verdict}  peak={peak_gib:.2f} GiB  "
                f"free={free_gib:.2f} GiB  out={len(gen.text)} chars  "
                f"({time.time() - t0:.0f}s)"
            )
        except Exception as exc:  # report and continue: other models still matter
            report[alias] = {"verdict": "FAIL", "error": repr(exc)}
            print(f"[PREFLIGHT] {alias:<12} FAIL  {exc!r}")
        torch.cuda.empty_cache()
    n_bad = sum(1 for r in report.values() if r.get("verdict") != "PASS")
    if n_bad == 0:
        print("[PREFLIGHT] ALL PASS - campaign memory envelope proven")
    else:
        print(f"[PREFLIGHT] {n_bad} model(s) below the pass bar - do NOT start the campaign")
    return report
