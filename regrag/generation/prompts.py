"""Prompt templates for Closed-book and RAG generation on Vietnamese banking regulations."""

from dataclasses import dataclass
from typing import List, Tuple

from regrag.generation.config import (
    RAG_CONTEXT_MIN_PASSAGE_CHARS,
    RAG_CONTEXT_PER_PASSAGE_CHARS,
    RAG_CONTEXT_TOTAL_CHARS,
)
from regrag.models import RetrievedResult


SYSTEM_PROMPT_VI = (
    "Bạn là chuyên gia tư vấn pháp lý chuyên sâu về các quy định ngân hàng, "
    "thanh toán điện tử và nghiệp vụ thẻ của Ngân hàng Nhà nước Việt Nam. "
    "Nhiệm vụ của bạn là trả lời chính xác, trung thực dựa trên các văn bản pháp luật hiện hành."
)

ABSTENTION_KEYPHRASE = "THÔNG TIN KHÔNG CÓ TRONG TÀI LIỆU"

CONTEXT_TRUNCATION_MARKER = "[...Nội dung bị cắt do vượt giới hạn ngữ cảnh...]"


@dataclass(frozen=True)
class RagContextReport:
    """What the context budget did to one RAG prompt. Stamped on the row so a
    truncated context is visible at analysis time, never silently applied."""

    context_chars: int
    budget_chars: int
    per_passage_chars: int
    truncated_ranks: Tuple[int, ...]  # 1-based retrieval ranks
    dropped_ranks: Tuple[int, ...]    # 1-based retrieval ranks

    @property
    def touched(self) -> bool:
        return bool(self.truncated_ranks or self.dropped_ranks)


def _clip(text: str, limit: int) -> Tuple[str, bool]:
    if len(text) <= limit:
        return text, False
    return text[:limit].rstrip() + "\n" + CONTEXT_TRUNCATION_MARKER, True


def build_rag_prompt_with_report(
    question: str,
    retrieved_results: List[RetrievedResult],
    total_context_chars: int = RAG_CONTEXT_TOTAL_CHARS,
    per_passage_chars: int = RAG_CONTEXT_PER_PASSAGE_CHARS,
) -> Tuple[str, RagContextReport]:
    """RAG prompt under a character budget, plus a report of what was cut.

    Passages are packed in retrieval-rank order; each contributes at most
    ``per_passage_chars`` and packing stops when the total budget is spent.
    A passage whose remaining share is under the floor is dropped whole.
    When everything fits (the normal case: p95 chunk is ~1.4k chars) the
    output string is byte-identical to the legacy unbounded builder.
    """
    context_blocks: List[str] = []
    truncated: List[int] = []
    dropped: List[int] = []
    used = 0
    for i, res in enumerate(retrieved_results, 1):
        full = res.chunk.formatted_context()
        remaining = total_context_chars - used
        if remaining < RAG_CONTEXT_MIN_PASSAGE_CHARS:
            dropped.extend(range(i, len(retrieved_results) + 1))
            break
        allowance = min(per_passage_chars, remaining)
        clipped, was_truncated = _clip(full, allowance)
        if was_truncated:
            truncated.append(i)
        block = f"--- Tài liệu tham khảo [{i}] ---\n{clipped}"
        used += len(block)
        context_blocks.append(block)
    context_str = "\n\n".join(context_blocks)
    report = RagContextReport(
        context_chars=len(context_str),
        budget_chars=total_context_chars,
        per_passage_chars=per_passage_chars,
        truncated_ranks=tuple(truncated),
        dropped_ranks=tuple(dropped),
    )
    return _render_rag_prompt(question, context_str), report


def _render_rag_prompt(question: str, context_str: str) -> str:
    return f"""{SYSTEM_PROMPT_VI}

Dưới đây là các đoạn trích từ quy định pháp luật của Ngân hàng Nhà nước Việt Nam:

{context_str}

---
Câu hỏi: {question}

Yêu cầu nghiêm ngặt:
1. Chỉ sử dụng thông tin trong các đoạn trích trên để trả lời.
2. Phải trích dẫn cụ thể căn cứ pháp lý (Tên văn bản/Thông tư, Điều, Khoản) hỗ trợ cho câu trả lời.
3. Nếu các đoạn trích trên KHÔNG chứa đủ thông tin để trả lời câu hỏi, bạn PHẢI trả lời: "{ABSTENTION_KEYPHRASE}". Tuyệt đối không tự suy đoán thông tin ngoài văn bản.

Trả lời:"""


def build_closed_book_prompt(question: str) -> str:
    """Prompt for closed-book condition."""
    return f"""{SYSTEM_PROMPT_VI}

Câu hỏi: {question}

Yêu cầu:
1. Trả lời chính xác nội dung câu hỏi.
2. Nêu rõ căn cứ pháp lý nếu biết (Tên văn bản, Điều, Khoản).
3. Nếu bạn không chắc chắn hoặc quy định pháp luật không quy định, hãy trả lời chính xác: "{ABSTENTION_KEYPHRASE}". Không được suy diễn hoặc tạo ra căn cứ pháp lý không có thật.

Trả lời:"""


def build_rag_prompt(question: str, retrieved_results: List[RetrievedResult]) -> str:
    """Legacy unbounded RAG prompt. Kept for callers that need the historical
    string; the campaign path uses :func:`build_rag_prompt_with_report`."""
    context_blocks = []
    for i, res in enumerate(retrieved_results, 1):
        context_blocks.append(
            f"--- Tài liệu tham khảo [{i}] ---\n{res.chunk.formatted_context()}"
        )
    context_str = "\n\n".join(context_blocks)
    return _render_rag_prompt(question, context_str)
