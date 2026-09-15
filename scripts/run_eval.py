"""CLI runner to execute benchmark evaluation using either in-memory or file repository."""

import argparse
import json
import os
import sys

# Ensure repository root is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from regrag.models import GenerationResult, GoldQuestion
from regrag.evaluation.metrics import evaluate_response
from regrag.storage.in_memory import InMemoryResultRepository
from regrag.storage.file_repo import FileResultRepository


def main():
    parser = argparse.ArgumentParser(description="Run RegRAG-VN Benchmark Evaluation")
    parser.add_argument(
        "--storage",
        choices=["in_memory", "file"],
        default="file",
        help="Persistence backend to use",
    )
    parser.add_argument(
        "--results-dir",
        default="results",
        help="Directory for file storage",
    )
    parser.add_argument(
        "--gold-file",
        default="data/gold/sample_gold.json",
        help="Path to gold questions JSON",
    )
    args = parser.parse_args()

    # Choose persistence layer via dependency injection
    if args.storage == "in_memory":
        repo = InMemoryResultRepository()
        print("[Storage] Using InMemoryResultRepository")
    else:
        repo = FileResultRepository(args.results_dir)
        print(f"[Storage] Using FileResultRepository at {args.results_dir}")

    # Load sample gold questions
    if not os.path.exists(args.gold_file):
        print(f"Gold file not found: {args.gold_file}")
        return

    with open(args.gold_file, "r", encoding="utf-8") as f:
        gold_data = json.load(f)

    print(f"Loaded {len(gold_data)} gold question(s).")
    for item in gold_data:
        gold = GoldQuestion(**{k: v for k, v in item.items() if k in GoldQuestion.__annotations__})

        # Simulate sample response
        sample_gen = GenerationResult(
            question_id=gold.id,
            model_name="qwen2.5-7b-instruct",
            retrieval_mode="rag_bm25",
            prompt=f"Question: {gold.question}",
            raw_response=gold.reference_answer,
            answer_text=gold.reference_answer,
        )

        repo.save_generation(sample_gen)
        record = evaluate_response(sample_gen, gold)
        repo.save_evaluation(record)

    summary = repo.export_summary()
    print("\nBenchmark Summary:")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
