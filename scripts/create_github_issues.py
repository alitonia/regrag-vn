"""Script to create milestone tasks as GitHub issues for RegRAG-VN."""

import subprocess
import json

TASKS = [
    {
        "title": "[Phase 1] Architecture Scaffolding & Dual Storage Repositories",
        "label": "phase:1-scaffolding",
        "body": """### Objective
Implement the core project architecture and dual-persistence storage layers (In-Memory and File JSON/CSV) sharing an identical interface without changing business logic.

### Assignees / Lead
- **Lead:** Nguyễn Thắng Phúc (20252263M)
- **Co-Lead:** Nguyễn Huy Hoàng (20251325M)

### Key Deliverables
- [x] Abstract base class `BenchmarkResultRepository` in `regrag/storage/base.py`
- [x] `InMemoryResultRepository` in `regrag/storage/in_memory.py`
- [x] `FileResultRepository` with atomic JSON and CSV persistence in `regrag/storage/file_repo.py`
- [x] Unit test suite verifying interface parity in `tests/test_storage.py`
""",
    },
    {
        "title": "[Phase 2] Curate 10-12 active SBV circulars & Decree 52/2024/NĐ-CP",
        "label": "phase:2-corpus",
        "body": """### Objective
Curate and verify 10–12 active, non-repealed legal documents from the State Bank of Vietnam (SBV) and Government in card payments and electronic banking.

### Assignees / Lead
- **Lead:** Đinh Thị Lan Hương (20261261M)

### Key Deliverables
- [ ] List of 10–12 circulars/decrees with active effective dates (avoid repealed circulars).
  - Must include: Thông tư 18/2024/TT-NHNN, Thông tư 17/2024/TT-NHNN, Nghị định 52/2024/NĐ-CP.
- [ ] Save raw clean text files in `data/raw_legal/`.
- [ ] Document metadata catalog with document numbers, titles, issue dates, and effective dates.
""",
    },
    {
        "title": "[Phase 2] Clause-level Legal Document Segmenter & JSON Serializer",
        "label": "phase:2-corpus",
        "body": """### Objective
Implement regex-based segmenter for Vietnamese legal texts to produce clause-level (`Khoản`) chunks with prepended parent metadata (`Điều`, Chương, Tên văn bản).

### Assignees / Lead
- **Lead:** Đinh Thị Lan Hương (20261261M)

### Key Deliverables
- [ ] Robust regex parser handling `Điều`, `Khoản`, `Điểm` hierarchies.
- [ ] Serializer saving parsed chunks into `data/processed_chunks/corpus_chunks.json`.
- [ ] Unit tests verifying chunk integrity and metadata continuity in `tests/test_components.py`.
""",
    },
    {
        "title": "[Phase 2] BM25 Indexing with Vietnamese Word Segmentation (pyvi)",
        "label": "phase:2-corpus",
        "body": """### Objective
Build the sparse retrieval pipeline using BM25, incorporating Vietnamese compound word segmentation via `pyvi` / `underthesea`.

### Assignees / Lead
- **Lead:** Nguyễn Khắc Duy Ngọc (20261206M)

### Key Deliverables
- [ ] Tokenizer wrapper supporting Vietnamese compound word boundary detection.
- [ ] BM25 indexer wrapping `rank-bm25` over all legal chunks in `regrag/indexing/bm25.py`.
- [ ] Top-$k$ search interface returning `RetrievedResult` objects with ranked relevance scores.
""",
    },
    {
        "title": "[Phase 2] Dense Vector Indexer with Multilingual Embeddings (BGE-M3)",
        "label": "phase:2-corpus",
        "body": """### Objective
Build the dense vector retrieval pipeline using state-of-the-art multilingual embeddings (`BAAI/bge-m3` or `multilingual-e5`).

### Assignees / Lead
- **Lead:** Nguyễn Khắc Duy Ngọc (20261206M)

### Key Deliverables
- [ ] SentenceTransformer encoder wrapper in `regrag/indexing/dense.py`.
- [ ] Chunk embedding generator with normalization.
- [ ] Cosine similarity retrieval module with top-$k$ ranking.
""",
    },
    {
        "title": "[Phase 3] Design 100 Gold QA Benchmark Taxonomy (70 answerable, 30 unanswerable)",
        "label": "phase:3-gold-set",
        "body": """### Objective
Author 100 high-quality Vietnamese legal questions across card payments and e-banking regulations, including exact article/clause ground-truth citations.

### Assignees / Lead
- **Lead:** Vũ Đức Thành (20261082M)
- **Contributors:** All Group Members

### Key Deliverables
- [ ] 70 Answerable questions (factual extraction, multi-clause synthesis, boundary definitions).
- [ ] 30 Unanswerable questions (out-of-scope, repealed rules, cross-domain banking queries) to test abstention.
- [ ] Complete schema JSON: `data/gold/questions_100.json`.
""",
    },
    {
        "title": "[Phase 3] Dual Independent Annotation Protocol & Cohen's Kappa Calculation",
        "label": "phase:3-gold-set",
        "body": """### Objective
Execute a dual-reviewer annotation pass on the 100 gold QA pairs to validate benchmark reliability and report inter-annotator agreement.

### Assignees / Lead
- **Lead:** Vũ Đức Thành (20261082M)
- **Reviewers:** All Group Members (pairwise review)

### Key Deliverables
- [ ] Annotation guideline rubric in `data/gold/annotation_guidelines.md`.
- [ ] Two independent validation passes on each question.
- [ ] Compute Cohen's $\kappa$ agreement score using `regrag/evaluation/agreement.py`.
""",
    },
    {
        "title": "[Phase 4] Colab Execution Runner with Drive Checkpoint Backup/Restore",
        "label": "phase:4-inference",
        "body": """### Objective
Finalize the Google Colab runner notebook for 4-bit quantized model inference, complete with Google Drive checkpoint backup and restore for session resilience.

### Assignees / Lead
- **Lead:** Nguyễn Thắng Phúc (20252263M)
- **Co-Lead:** Nguyễn Huy Hoàng (20251325M)

### Key Deliverables
- [ ] Colab environment installation cells (`bitsandbytes`, `accelerate`, `transformers`).
- [ ] Drive mount and automatic checkpoint restore/sync cells in `notebooks/colab_runner.ipynb`.
- [ ] Idempotent runner skipping already evaluated queries.
""",
    },
    {
        "title": "[Phase 4] Execute 4-bit Inference across 9 Experimental Configurations (900 runs)",
        "label": "phase:4-inference",
        "body": """### Objective
Run inference across 3 small LLMs ($\le 7\text{B}$) under Closed-book, RAG-BM25, and RAG-Dense settings on the 100 gold questions.

### Assignees / Lead
- **Lead:** Nguyễn Huy Hoàng (20251325M)
- **Co-Lead:** Nguyễn Thắng Phúc (20252263M)

### Key Deliverables
- [ ] 3 Models: `Qwen2.5-7B-Instruct`, `Llama-3.2-3B-Instruct`, `Qwen2.5-3B-Instruct` (or `Vistral-7B-Chat`).
- [ ] 9 Configurations $\times$ 100 Questions = 900 raw generation outputs.
- [ ] Results saved via `FileResultRepository` to `results/generations.json`.
""",
    },
    {
        "title": "[Phase 5] Automated Citation Extraction & Precision/Recall Scoring",
        "label": "phase:5-evaluation",
        "body": """### Objective
Evaluate citation accuracy by parsing legal citations (circular number, Điều, Khoản) from model outputs and comparing with gold metadata.

### Assignees / Lead
- **Lead:** Nguyễn Khắc Duy Ngọc (20261206M)
- **Co-Lead:** Nguyễn Huy Hoàng (20251325M)

### Key Deliverables
- [ ] Robust regex and citation normalizer in `regrag/evaluation/citation.py`.
- [ ] Compute Citation Precision and Citation Recall per model and retrieval condition.
""",
    },
    {
        "title": "[Phase 5] Automated Abstention Classification & Hallucination Rate Analysis",
        "label": "phase:5-evaluation",
        "body": """### Objective
Measure abstention accuracy on the 30 unanswerable queries and quantify hallucination rates across all 9 configurations.

### Assignees / Lead
- **Lead:** Nguyễn Huy Hoàng (20251325M)

### Key Deliverables
- [ ] Abstention classifier evaluating compliance with abstention prompt instruction.
- [ ] Hallucination detection heuristic and comparative analysis across Closed-book vs RAG.
- [ ] Export evaluated records to `results/evaluations.json` and `results/evaluations.csv`.
""",
    },
    {
        "title": "[Phase 5] Result Aggregation & Comparison Tables/Figures Generation",
        "label": "phase:5-evaluation",
        "body": """### Objective
Aggregate evaluation records into summary tables and publication-grade visualization plots.

### Assignees / Lead
- **Lead:** Nguyễn Huy Hoàng (20251325M)
- **Co-Lead:** Nguyễn Khắc Duy Ngọc (20261206M)

### Key Deliverables
- [ ] Table I: Retrieval performance (Recall@1, Recall@3, MRR for BM25 vs Dense).
- [ ] Table II: Generation performance (Citation P/R, Abstention Accuracy, Hallucination Rate).
- [ ] Visualization plots for paper and presentation.
""",
    },
    {
        "title": "[Phase 6] Course LaTeX Report & IEEE 6-Page Manuscript Drafting",
        "label": "phase:6-paper",
        "body": """### Objective
Draft the academic research report / 6-page IEEE paper following the course template and target conference format.

### Assignees / Lead
- **Lead:** Nguyễn Huy Hoàng (20251325M)
- **Section Authors:**
  - §I Intro & §II Related Work: Hoàng & Hương
  - §III Legal Corpus & Gold Benchmark: Hương & Thành
  - §IV System Architecture & Dual Persistence: Phúc & Ngọc
  - §V Experimental Results & RQ Discussion: Hoàng & Ngọc
  - §VI Conclusion & Ethics: Thành & Phúc
""",
    },
    {
        "title": "[Phase 6] Final Verification, Reproducibility Audit & Release Packaging",
        "label": "phase:6-paper",
        "body": """### Objective
Perform end-to-end verification of results, check tests, verify all member Git commits, and package release artifacts.

### Assignees / Lead
- **Lead:** All Group Members

### Key Deliverables
- [ ] Full test pass across storage, parsing, indexing, and evaluation.
- [ ] Verification that all 5 members have substantial, distinct git commits.
- [ ] Tag final release version on GitHub.
""",
    },
]

def main():
    repo = "alitonia/rag_eval"
    for i, t in enumerate(TASKS, 1):
        print(f"Creating issue {i}/{len(TASKS)}: {t['title']}")
        cmd = [
            "gh", "issue", "create",
            "-R", repo,
            "--title", t["title"],
            "--body", t["body"],
            "--label", t["label"],
        ]
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode == 0:
            print(f" -> Created: {res.stdout.strip()}")
        else:
            print(f" -> Error: {res.stderr.strip()}")

if __name__ == "__main__":
    main()
