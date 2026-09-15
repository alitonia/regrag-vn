# Data Directory

This directory contains the regulatory corpus and benchmark question sets:

- `raw_legal/`: Raw text / PDF files of State Bank of Vietnam (SBV) circulars and decrees.
- `processed_chunks/`: Cleaned and parsed JSON chunks at the `Khoản` (Clause) level.
- `gold/`: The 100 gold question set and annotation guidelines:
  - `questions_100.json`: 100 curated questions (70 answerable, 30 unanswerable) with dual-annotation labels.
  - `sample_gold.json`: Benchmark schema template for question authoring.
