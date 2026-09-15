#!/usr/bin/env python3
"""Build human-labeling instruments for RegRAG-VN benchmark paper.

Produces:
1. Kappa grading sheets (30 questions, cross-annotated by Huong and Thanh):
   - data/human/kappa_grading_huong.csv
   - data/human/kappa_grading_thanh.csv
   - data/human/KAPPA_INSTRUCTIONS.md
   - data/human/kappa_subset_2026-09-13.json

2. Scorer validation set (50 sampled model responses across 3x3 grid):
   - data/human/scorer_validation_50.csv
   - data/human/VALIDATION_INSTRUCTIONS.md
   - data/human/scorer_validation_sample_2026-09-13.json
"""

import argparse
import csv
import hashlib
import json
import os
import random
import sys
from collections import Counter, defaultdict
from typing import Any, Dict, List, Tuple

# Ensure repository root is on sys.path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from regrag.corpus.qa_loader import load_gold_questions
from regrag.models import GenerationResult, GoldQuestion
from regrag.evaluation.metrics import score_response, context_from_prompt, validation_report
from regrag.evaluation.agreement import compute_cohens_kappa


def file_sha256(filepath: str) -> str:
    """Calculate SHA256 checksum of a file."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


# -----------------------------------------------------------------------------
# 1. KAPPA SUBSET GENERATOR
# -----------------------------------------------------------------------------

KAPPA_INSTRUCTIONS_TEXT = r"""# Hướng Dẫn Đánh Giá Kappa (Inter-Annotator Agreement Instructions)

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
"""


def build_kappa_sheets(
    gold_csv: str,
    out_dir: str,
    seed: int = 20260913,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Build the 30-question cross-annotated Kappa subset and grading sheets."""
    questions = load_gold_questions(gold_csv, repo_root=REPO_ROOT)
    
    # Parse author and separate answerable vs probes
    # Annotator is the first part of author cell before semicolon
    ans_huong: List[GoldQuestion] = []
    ans_thanh: List[GoldQuestion] = []
    probes: List[GoldQuestion] = []
    
    for q in questions:
        raw_author = q.author or ""
        author_base = raw_author.split(";")[0].strip()
        if not q.is_answerable:
            probes.append(q)
        else:
            if author_base == "Hương":
                ans_huong.append(q)
            elif author_base == "Thành":
                ans_thanh.append(q)
            else:
                # Default fallback
                ans_thanh.append(q)

    rng = random.Random(seed)
    
    # Target: 24 answerable + 6 probes = 30 questions
    # Authorship split: 15 Hương, 15 Thành
    # For answerable: sample 12 from Hương (out of 14) and 12 from Thành (out of 50)
    # For probes: sample 6 probes (out of 24), and assign round-robin 3 to Hương, 3 to Thành
    sampled_ans_h = rng.sample(ans_huong, 12)
    sampled_ans_t = rng.sample(ans_thanh, 12)
    sampled_probes = rng.sample(probes, 6)

    # Round-robin assignment of probes for authorship bookkeeping:
    # probe 0, 2, 4 -> Hương; probe 1, 3, 5 -> Thành
    probe_authors = ["Hương", "Thành", "Hương", "Thành", "Hương", "Thành"]

    selected_rows: List[Dict[str, Any]] = []

    for q in sampled_ans_h:
        selected_rows.append({
            "question_obj": q,
            "author_of_record": "Hương",
            "is_probe": False,
        })

    for q in sampled_ans_t:
        selected_rows.append({
            "question_obj": q,
            "author_of_record": "Thành",
            "is_probe": False,
        })

    for p_obj, p_auth in zip(sampled_probes, probe_authors):
        selected_rows.append({
            "question_obj": p_obj,
            "author_of_record": p_auth,
            "is_probe": True,
        })

    # Sort deterministically by question_id (e.g. Q001, Q002, ...)
    selected_rows.sort(key=lambda x: x["question_obj"].id)

    # Prepare CSV fieldnames and row dictionaries
    fieldnames = [
        "question_id",
        "author_of_record",
        "question",
        "is_answerable",
        "gold_doc_id",
        "gold_article_id",
        "gold_clause_id",
        "reference_answer",
        "gold_passage",
        "grade_doc",
        "grade_article",
        "grade_clause",
        "grade_answerable",
        "grade_overall",
        "notes",
    ]

    csv_rows: List[Dict[str, str]] = []
    for item in selected_rows:
        q: GoldQuestion = item["question_obj"]
        auth: str = item["author_of_record"]
        
        gold_doc = ""
        gold_art = ""
        gold_clause = ""
        if q.gold_citations:
            c = q.gold_citations[0]
            gold_doc = str(c.get("doc_id") or "")
            gold_art = str(c.get("article_id") or "")
            gold_clause = str(c.get("clause_id") or "")
        elif q.gold_doc_ids:
            gold_doc = q.gold_doc_ids[0]

        row = {
            "question_id": q.id,
            "author_of_record": auth,
            "question": q.question,
            "is_answerable": str(q.is_answerable),
            "gold_doc_id": gold_doc,
            "gold_article_id": gold_art,
            "gold_clause_id": gold_clause,
            "reference_answer": q.reference_answer,
            "gold_passage": q.gold_passage,
            "grade_doc": "",
            "grade_article": "",
            "grade_clause": "",
            "grade_answerable": "",
            "grade_overall": "",
            "notes": "",
        }
        csv_rows.append(row)

    # Write kappa_grading_huong.csv and kappa_grading_thanh.csv (identical rows)
    huong_csv_path = os.path.join(out_dir, "kappa_grading_huong.csv")
    thanh_csv_path = os.path.join(out_dir, "kappa_grading_thanh.csv")

    for path in (huong_csv_path, thanh_csv_path):
        with open(path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(csv_rows)

    # Write KAPPA_INSTRUCTIONS.md
    instructions_path = os.path.join(out_dir, "KAPPA_INSTRUCTIONS.md")
    with open(instructions_path, "w", encoding="utf-8") as f:
        f.write(KAPPA_INSTRUCTIONS_TEXT.strip() + "\n")

    # Build JSON metadata record
    composition = {
        "answerable": sum(1 for r in selected_rows if not r["is_probe"]),
        "probes": sum(1 for r in selected_rows if r["is_probe"]),
        "total": len(selected_rows),
    }
    authorship_split = {
        "Hương": sum(1 for r in selected_rows if r["author_of_record"] == "Hương"),
        "Thành": sum(1 for r in selected_rows if r["author_of_record"] == "Thành"),
    }
    authorship_composition = {
        "Hương": {
            "answerable": sum(1 for r in selected_rows if r["author_of_record"] == "Hương" and not r["is_probe"]),
            "probes": sum(1 for r in selected_rows if r["author_of_record"] == "Hương" and r["is_probe"]),
            "total": 15,
        },
        "Thành": {
            "answerable": sum(1 for r in selected_rows if r["author_of_record"] == "Thành" and not r["is_probe"]),
            "probes": sum(1 for r in selected_rows if r["author_of_record"] == "Thành" and r["is_probe"]),
            "total": 15,
        },
    }

    subset_metadata = {
        "benchmark": "RegRAG-VN",
        "subset_name": "kappa_cross_annotated_30",
        "date": "2026-09-13",
        "seed": seed,
        "total_questions": len(selected_rows),
        "composition": composition,
        "authorship_split": authorship_split,
        "authorship_composition": authorship_composition,
        "probe_assignment_rule": (
            "Probes have unassigned or collective authorship; 6 sampled probes are assigned "
            "round-robin (3 to Hương, 3 to Thành) for balanced cross-annotation bookkeeping."
        ),
        "question_ids": [r["question_obj"].id for r in selected_rows],
        "items": [
            {
                "question_id": r["question_obj"].id,
                "author_of_record": r["author_of_record"],
                "is_answerable": r["question_obj"].is_answerable,
                "gold_doc_id": csv_rows[idx]["gold_doc_id"],
                "gold_article_id": csv_rows[idx]["gold_article_id"],
                "gold_clause_id": csv_rows[idx]["gold_clause_id"],
                "question": r["question_obj"].question,
            }
            for idx, r in enumerate(selected_rows)
        ],
    }

    json_path = os.path.join(out_dir, "kappa_subset_2026-09-13.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(subset_metadata, f, ensure_ascii=False, indent=2)

    return csv_rows, subset_metadata


# -----------------------------------------------------------------------------
# 2. SCORER VALIDATION SET GENERATOR
# -----------------------------------------------------------------------------

VALIDATION_INSTRUCTIONS_TEXT = r"""# Hướng Dẫn Thẩm Định Bộ Chấm Tự Động (Scorer Validation Instructions)

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
"""


def build_scorer_validation_set(
    generations_json: str,
    gold_csv: str,
    out_dir: str,
    seed: int = 20260913,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Build the 50-response stratified scorer-validation set."""
    golds = {q.id: q for q in load_gold_questions(gold_csv, repo_root=REPO_ROOT)}

    with open(generations_json, "r", encoding="utf-8") as f:
        gens = json.load(f)

    # Classify all generations into 4 mutually exclusive categories per cell:
    # 1. probe: unanswerable question probe
    # 2. false_abstention: answerable question where model abstained
    # 3. ans_hallucinated: answerable question, non-abstained, flagged hallucinated
    # 4. ans_clean: answerable question, non-abstained, clean (not hallucinated)
    by_cell_cat: Dict[Tuple[str, str], Dict[str, List[Dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )

    scored_records: Dict[str, Any] = {}
    for g in gens:
        gen = GenerationResult(
            question_id=g["question_id"],
            model_name=g["model_name"],
            retrieval_mode=g["retrieval_mode"],
            prompt=g["prompt"],
            raw_response=g["raw_response"],
            answer_text=g["answer_text"],
            extracted_citations=g.get("extracted_citations", []),
            abstained=bool(g.get("abstained")),
            corpus_source=g.get("corpus_source", ""),
            retriever_backend=g.get("retriever_backend", ""),
            cache_key=g.get("cache_key", ""),
        )
        gold = golds[g["question_id"]]
        score = score_response(gen, gold)
        
        if not gold.is_answerable:
            cat = "probe"
        elif score.abstention.abstained:
            cat = "false_abstention"
        elif score.hallucinated:
            cat = "ans_hallucinated"
        else:
            cat = "ans_clean"

        entry = {
            "gen": g,
            "gold": gold,
            "score": score,
            "category": cat,
        }
        cell = (g["model_name"], g["retrieval_mode"])
        by_cell_cat[cell][cat].append(entry)

    # 3x3 Grid Quotas summing to exactly 50:
    # Models: qwen-7b (17), qwen-3b (17), vistral-7b (16)
    # Modes: closed_book (17), rag_bm25 (17), rag_dense (16)
    quotas: Dict[Tuple[str, str], Dict[str, int]] = {
        ("qwen-7b", "closed_book"): {"probe": 1, "false_abstention": 0, "ans_hallucinated": 3, "ans_clean": 2},  # 6
        ("qwen-7b", "rag_bm25"):    {"probe": 1, "false_abstention": 2, "ans_hallucinated": 2, "ans_clean": 1},  # 6
        ("qwen-7b", "rag_dense"):   {"probe": 1, "false_abstention": 2, "ans_hallucinated": 1, "ans_clean": 1},  # 5
        ("qwen-3b", "closed_book"): {"probe": 1, "false_abstention": 1, "ans_hallucinated": 2, "ans_clean": 2},  # 6
        ("qwen-3b", "rag_bm25"):    {"probe": 1, "false_abstention": 2, "ans_hallucinated": 2, "ans_clean": 1},  # 6
        ("qwen-3b", "rag_dense"):   {"probe": 1, "false_abstention": 1, "ans_hallucinated": 2, "ans_clean": 1},  # 5
        ("vistral-7b", "closed_book"): {"probe": 1, "false_abstention": 0, "ans_hallucinated": 2, "ans_clean": 2}, # 5
        ("vistral-7b", "rag_bm25"):    {"probe": 1, "false_abstention": 1, "ans_hallucinated": 2, "ans_clean": 1}, # 5
        ("vistral-7b", "rag_dense"):   {"probe": 1, "false_abstention": 0, "ans_hallucinated": 3, "ans_clean": 2}, # 6
    }

    rng = random.Random(seed)
    sampled_entries: List[Dict[str, Any]] = []

    for cell, qdict in sorted(quotas.items()):
        for cat, count in sorted(qdict.items()):
            pool = sorted(by_cell_cat[cell][cat], key=lambda x: x["gen"]["question_id"])
            if count > 0:
                selected = rng.sample(pool, count)
                sampled_entries.extend(selected)

    assert len(sampled_entries) == 50, f"Expected 50 sampled entries, got {len(sampled_entries)}"

    # Sort deterministically by question_id, then model_name, then retrieval_mode
    sampled_entries.sort(key=lambda x: (x["gen"]["question_id"], x["gen"]["model_name"], x["gen"]["retrieval_mode"]))

    fieldnames = [
        "question_id",
        "model_name",
        "retrieval_mode",
        "question",
        "reference_answer",
        "gold_citation_json",
        "retrieved_context",
        "model_answer",
        "is_answerable",
        "auto_correct",
        "auto_hallucinated",
        "auto_abstained",
        "human_correct",
        "human_hallucinated",
        "human_abstained",
        "notes",
    ]

    csv_rows: List[Dict[str, Any]] = []
    meta_rows: List[Dict[str, Any]] = []

    for item in sampled_entries:
        g = item["gen"]
        gold: GoldQuestion = item["gold"]
        score = item["score"]
        cat = item["category"]

        # Recover retrieved contexts
        contexts = context_from_prompt(g.get("prompt", "") or "")
        retrieved_context_str = "\n\n---\n\n".join(contexts) if contexts else ""

        gold_cit_json = json.dumps(gold.gold_citations, ensure_ascii=False)
        auto_corr = 1 if score.correctness_score == 2.0 else 0
        auto_hall = 1 if score.hallucinated else 0
        auto_abst = 1 if score.abstention.abstained else 0

        row = {
            "question_id": g["question_id"],
            "model_name": g["model_name"],
            "retrieval_mode": g["retrieval_mode"],
            "question": gold.question,
            "reference_answer": gold.reference_answer,
            "gold_citation_json": gold_cit_json,
            "retrieved_context": retrieved_context_str,
            "model_answer": g.get("answer_text", "") or "",
            "is_answerable": str(gold.is_answerable),
            "auto_correct": str(auto_corr),
            "auto_hallucinated": str(auto_hall),
            "auto_abstained": str(auto_abst),
            "human_correct": "",
            "human_hallucinated": "",
            "human_abstained": "",
            "notes": "",
        }
        csv_rows.append(row)

        meta_rows.append({
            "question_id": g["question_id"],
            "model_name": g["model_name"],
            "retrieval_mode": g["retrieval_mode"],
            "category": cat,
            "is_answerable": gold.is_answerable,
            "correctness_score": score.correctness_score,
            "auto_correct": auto_corr,
            "auto_hallucinated": auto_hall,
            "auto_abstained": auto_abst,
            "hallucination_types": list(score.hallucination_types),
            "contexts_count": len(contexts),
            "gold_citations": gold.gold_citations,
        })

    # Write scorer_validation_50.csv
    csv_path = os.path.join(out_dir, "scorer_validation_50.csv")
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)

    # Write VALIDATION_INSTRUCTIONS.md
    instructions_path = os.path.join(out_dir, "VALIDATION_INSTRUCTIONS.md")
    with open(instructions_path, "w", encoding="utf-8") as f:
        f.write(VALIDATION_INSTRUCTIONS_TEXT.strip() + "\n")

    # Stratification summary
    models_counter = Counter(r["model_name"] for r in meta_rows)
    modes_counter = Counter(r["retrieval_mode"] for r in meta_rows)
    cats_counter = Counter(r["category"] for r in meta_rows)
    grid_counter = {
        f"{m}::{mode}": sum(1 for r in meta_rows if r["model_name"] == m and r["retrieval_mode"] == mode)
        for m in sorted(models_counter.keys())
        for mode in sorted(modes_counter.keys())
    }
    auto_labels_counter = {
        "auto_correct_count": sum(1 for r in meta_rows if r["auto_correct"] == 1),
        "auto_hallucinated_count": sum(1 for r in meta_rows if r["auto_hallucinated"] == 1),
        "auto_abstained_count": sum(1 for r in meta_rows if r["auto_abstained"] == 1),
    }

    sample_metadata = {
        "benchmark": "RegRAG-VN",
        "subset_name": "scorer_validation_50",
        "date": "2026-09-13",
        "seed": seed,
        "total_samples": len(meta_rows),
        "stratification_table": {
            "by_model": dict(models_counter),
            "by_mode": dict(modes_counter),
            "by_category": dict(cats_counter),
            "by_grid_cell": grid_counter,
            "auto_labels": auto_labels_counter,
        },
        "sampled_rows": meta_rows,
    }

    json_path = os.path.join(out_dir, "scorer_validation_sample_2026-09-13.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(sample_metadata, f, ensure_ascii=False, indent=2)

    return csv_rows, sample_metadata


# -----------------------------------------------------------------------------
# MAIN CLI & VERIFICATION
# -----------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Build human-labeling instruments for RegRAG-VN")
    parser.add_argument("--gold-csv", default=os.path.join(REPO_ROOT, "data", "gold", "bank_qa_data.csv"))
    parser.add_argument("--generations-json", default=os.path.join(REPO_ROOT, "data", "eval", "generations.json"))
    parser.add_argument("--out-dir", default=os.path.join(REPO_ROOT, "data", "human"))
    parser.add_argument("--seed", type=int, default=20260913)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print("================================================================================")
    print(f"Generating RegRAG-VN Human Label Instruments (seed={args.seed})")
    print(f"Output directory: {args.out_dir}")
    print("================================================================================")

    # 1. Build Kappa sheets
    kappa_rows, kappa_meta = build_kappa_sheets(args.gold_csv, args.out_dir, seed=args.seed)
    print(f"\n[1] Kappa subset generated: {len(kappa_rows)} rows")
    print(f"    - Answerable: {kappa_meta['composition']['answerable']}, Probes: {kappa_meta['composition']['probes']}")
    print(f"    - Authorship split: {kappa_meta['authorship_split']}")
    print(f"    - Files written:")
    print(f"      * {os.path.join(args.out_dir, 'kappa_grading_huong.csv')}")
    print(f"      * {os.path.join(args.out_dir, 'kappa_grading_thanh.csv')}")
    print(f"      * {os.path.join(args.out_dir, 'KAPPA_INSTRUCTIONS.md')}")
    print(f"      * {os.path.join(args.out_dir, 'kappa_subset_2026-09-13.json')}")

    # 2. Build Scorer validation set
    val_rows, val_meta = build_scorer_validation_set(args.generations_json, args.gold_csv, args.out_dir, seed=args.seed)
    print(f"\n[2] Scorer validation sample generated: {len(val_rows)} rows")
    print(f"    - Models: {val_meta['stratification_table']['by_model']}")
    print(f"    - Modes:  {val_meta['stratification_table']['by_mode']}")
    print(f"    - Categories: {val_meta['stratification_table']['by_category']}")
    print(f"    - Auto labels: {val_meta['stratification_table']['auto_labels']}")
    print(f"    - Files written:")
    print(f"      * {os.path.join(args.out_dir, 'scorer_validation_50.csv')}")
    print(f"      * {os.path.join(args.out_dir, 'VALIDATION_INSTRUCTIONS.md')}")
    print(f"      * {os.path.join(args.out_dir, 'scorer_validation_sample_2026-09-13.json')}")

    print("\n================================================================================")
    print("Generation complete.")
    print("================================================================================")


if __name__ == "__main__":
    main()
