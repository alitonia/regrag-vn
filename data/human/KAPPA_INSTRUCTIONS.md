# Hướng Dẫn Đánh Giá Kappa (Inter-Annotator Agreement Instructions)

**RegRAG-VN Benchmark — Dual Independent Annotation Protocol**
**Target**: 30-question cross-annotated subset (Tập 30 câu hỏi gán nhãn chéo)
**Annotators**: Đinh Thị Lan Hương & Nguyễn Thành (hoặc người đánh giá độc lập)

---

## I. Giới thiệu / Introduction

### Tiếng Việt
Tài liệu này hướng dẫn hai chuyên viên gán nhãn (Hương và Thành) thực hiện đánh giá độc lập trên cùng một tập 30 câu hỏi được chọn mẫu đại diện từ bộ chuẩn `data/gold/bank_qa_data.csv`. Mỗi người sẽ điền vào file CSV riêng của mình (`kappa_grading_huong.csv` và `kappa_grading_thanh.csv`).
Mục tiêu là đo lường hệ số tương đồng Cohen's Kappa ($\kappa$) theo thiết kế tại mục §III-D của bài báo khoa học.

### English
This document instructs the two independent annotators (Hương and Thành) to grade the SAME 30-question subset sampled from `data/gold/bank_qa_data.csv`. Each annotator fills their own dedicated CSV file (`kappa_grading_huong.csv` and `kappa_grading_thanh.csv`).
The objective is to compute Cohen's Kappa ($\kappa$) inter-annotator agreement as specified in Section §III-D of the paper.

---

## II. Quy tắc chấm điểm từng cột / Column Grading Guidelines

Đối với mỗi dòng trong file CSV, người đánh giá đối chiếu câu hỏi, câu trả lời tham chiếu và trích dẫn chuẩn với các văn bản quy phạm pháp luật gốc trong `data/raw_legal/`. Điền các cột đánh giá (từ cột 10 đến cột 15):

| Tên cột (Column) | Giá trị hợp lệ (Allowed Values) | Hướng dẫn chi tiết (Detailed Instructions) |
| :--- | :--- | :--- |
| `grade_doc` | `correct` / `incorrect` | **Đúng văn bản pháp luật?**<br>- `correct`: Văn bản trích dẫn (`gold_doc_id`) đúng là căn cứ pháp lý của câu trả lời. Đối với câu hỏi bẫy (`is_answerable=False`), điền `correct` nếu kho văn bản thực sự không có quy định.<br>- `incorrect`: Trích dẫn sai văn bản. |
| `grade_article` | `correct` / `incorrect` / `n-a-when-no-article-gold` | **Đúng Điều luật?**<br>- `correct`: Đúng số thứ tự Điều luật quy định nội dung.<br>- `incorrect`: Sai Điều luật. **QUY TẮC BẮT BUỘC: Nếu trích dẫn đúng số Điều nhưng ở SAI văn bản pháp quy thì BẮT BUỘC chấm `incorrect`** (ví dụ: Điều 8 TT 06/2019 quy định về chuẩn bị đầu tư, nếu ghi Điều 8 nhưng ở văn bản TT 17/2024 thì là `incorrect`).<br>- `n-a-when-no-article-gold`: Áp dụng cho câu hỏi bẫy (`is_answerable=False`) hoặc câu hỏi trích dẫn Phụ lục/không có Điều cụ thể. |
| `grade_clause` | `correct` / `incorrect` / `n-a-when-no-clause-gold` | **Đúng Khoản?**<br>- `correct`: Đúng số Khoản quy định nội dung.<br>- `incorrect`: Sai Khoản.<br>- `n-a-when-no-clause-gold`: Khi văn bản chuẩn không xác định Khoản cụ thể hoặc câu hỏi bẫy. |
| `grade_answerable` | `yes` / `no` | **Có thể trả lời được từ kho văn bản không?**<br>- `yes`: Nội dung câu hỏi có căn cứ trả lời rõ ràng trong kho văn bản ngân hàng.<br>- `no`: Câu hỏi bẫy (probe), không có căn cứ trong kho văn bản hiện hành. |
| `grade_overall` | `correct` / `incorrect` | **Đánh giá tổng thể nhãn chuẩn:**<br>- `correct`: Cả văn bản, điều khoản và câu trả lời tham chiếu đều chính xác, đủ cơ sở pháp lý.<br>- `incorrect`: Có sai sót đáng kể về căn cứ pháp lý hoặc nội dung trả lời. |
| `notes` | Văn bản tự do (Free text) | Ghi chú lý do nếu chấm `incorrect`, hoặc các trường hợp đặc biệt / điều khoản sửa đổi bổ sung. |

---

## III. Kế hoạch phân tích & Tính toán Kappa / Analysis Plan

Hệ số Cohen's Kappa được tính trên các cặp đánh giá `(Hương grade, Thành grade)` cho từng hạng mục thông qua module `regrag.evaluation.agreement.compute_cohens_kappa`.

Báo cáo sẽ phân tầng theo 3 cấp độ:
1. **Overall agreement**: Tính trên toàn bộ 30 câu hỏi.
2. **Genuinely cross-annotated half (Phần gán nhãn chéo thực chất)**: 15 câu do người kia là tác giả gốc (`author_of_record`), người chấm phản biện câu của đồng nghiệp.
3. **Self-authored half (Phần tự đánh giá)**: 15 câu do chính người chấm soạn thảo ban đầu, được báo cáo riêng biệt để minh bạch độ thiên lệch tác giả (author bias).

### Lệnh tính toán (Verification snippet):

```python
import csv
from regrag.evaluation.agreement import compute_cohens_kappa

def load_grades(csv_path, column_name):
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return [row[column_name].strip() for row in reader]

huong_grades = load_grades("data/human/kappa_grading_huong.csv", "grade_overall")
thanh_grades = load_grades("data/human/kappa_grading_thanh.csv", "grade_overall")

kappa = compute_cohens_kappa(huong_grades, thanh_grades)
print(f"Cohen's Kappa (grade_overall): {kappa:.4f}")
```
