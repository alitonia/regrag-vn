# Hướng Dẫn Thẩm Định Bộ Chấm Tự Động (Scorer Validation Instructions)

**RegRAG-VN Benchmark — Scorer vs Human Validation**
**Target**: 50 sampled model responses across 3x3 grid (50 câu trả lời của mô hình)
**Evaluator**: Human Domain Expert

---

## I. Mục đích / Purpose

### Tiếng Việt
Đánh giá độ tương đồng giữa bộ chấm tự động theo quy tắc (`regrag.evaluation.metrics.score_response`) và đánh giá của con người trên 50 câu trả lời của 3 mô hình (`qwen-7b`, `qwen-3b`, `vistral-7b`) qua 3 chế độ (`closed_book`, `rag_bm25`, `rag_dense`).
Kết quả phục vụ công bố độ tin cậy và hệ số Cohen's Kappa của bộ chấm trong mục §IV-C của bài báo khoa học.

### English
Measure agreement between the deterministic rule-based scorer (`regrag.evaluation.metrics.score_response`) and human judgment on 50 sampled model responses across 3 models (`qwen-7b`, `qwen-3b`, `vistral-7b`) and 3 retrieval modes (`closed_book`, `rag_bm25`, `rag_dense`).
The resulting metrics support the scorer reliability analysis and Cohen's Kappa reported in §IV-C of the paper.

---

## II. Ba phán đoán nhị phân cần chấm / The Three Binary Judgments

Người đánh giá đọc câu hỏi (`question`), câu trả lời chuẩn (`reference_answer`), ngữ cảnh trích xuất (`retrieved_context`), và câu trả lời của mô hình (`model_answer`). Sau đó điền giá trị `1` (Đúng / Có) hoặc `0` (Sai / Không) vào 3 cột trống:

### 1. `human_correct` (1 / 0)
- **Định nghĩa**: Câu trả lời của mô hình có nêu được nội dung cốt lõi mà câu trả lời chuẩn (`reference_answer`) đưa ra hay không?
- **Quy tắc**:
  - `1`: Mô hình trả lời đúng thực chất yêu cầu pháp lý (chấp nhận cách diễn đạt tương đương).
  - `0`: Mô hình trả lời sai, thiếu ý cốt lõi, hoặc từ chối trả lời một câu hỏi có căn cứ trong tài liệu.

### 2. `human_hallucinated` (1 / 0)
- **Định nghĩa**: Câu trả lời có chứa thông tin sai lệch, bịa đặt trích dẫn hoặc không có căn cứ từ ngữ cảnh trích xuất (`retrieved_context`) / kho văn bản hay không?
- **Quy tắc**:
  - `1`: Mô hình bịa đặt số hiệu văn bản/Điều luật không tồn tại; trích dẫn sai văn bản điều chỉnh; khẳng định sự kiện trái ngược với tài liệu; hoặc đưa ra thông tin không được ngữ cảnh hỗ trợ (ở chế độ RAG).
  - `0`: Câu trả lời hoàn toàn trung thực với ngữ cảnh hoặc căn cứ pháp lý được cung cấp.

### 3. `human_abstained` (1 / 0)
- **Định nghĩa**: Mô hình có chủ động từ chối đưa ra câu trả lời thực chất hay không?
- **Quy tắc**:
  - `1`: Mô hình từ chối trả lời (ví dụ: *"Thông tin không có trong tài liệu"*, *"Kho văn bản không quy định vấn đề này"*, *"Tôi không thể trả lời"*).
  - `0`: Mô hình đưa ra câu trả lời thực chất.
  - **QUY TẮC QUAN TRỌNG (REFUSAL RULE)**: Nếu mô hình đưa ra lời rào đón/từ chối nhưng sau đó VẪN TRẢ LỜI nội dung thực chất (ví dụ: *"Tài liệu không đề cập rõ, tuy nhiên theo quy định chung thì hạn mức là 100 triệu"*), thì trường hợp này **KHÔNG ĐƯỢC COI LÀ TỪ CHỐI** (`human_abstained = 0`).

---

## III. Lệnh hoàn tất & Tính toán kết quả / Finisher Command

Sau khi hoàn thành việc điền các cột `human_correct`, `human_hallucinated`, `human_abstained` vào file `data/human/scorer_validation_50.csv`, chạy đoạn script sau để tạo báo cáo so sánh §IV-C:

```bash
/mnt/data/seminar_2/.venv/bin/python3 -c '
import csv, json
from regrag.evaluation.metrics import validation_report

pairs = []
with open("data/human/scorer_validation_50.csv", "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        pairs.append({
            "question_id": row["question_id"],
            "human_correct": row["human_correct"].strip() in ("1", "true", "True", "yes", "Yes"),
            "auto_correct": bool(int(row["auto_correct"])),
            "human_hallucinated": row["human_hallucinated"].strip() in ("1", "true", "True", "yes", "Yes"),
            "auto_hallucinated": bool(int(row["auto_hallucinated"])),
            "human_abstained": row["human_abstained"].strip() in ("1", "true", "True", "yes", "Yes"),
            "auto_abstained": bool(int(row["auto_abstained"])),
        })

report = validation_report(pairs)
print(json.dumps(report, indent=2, ensure_ascii=False))
'
```
