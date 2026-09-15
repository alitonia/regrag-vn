#!/usr/bin/env python3
"""CLI loader from trusted QA CSV to canonical GoldQuestion JSON.

Thin CLI over regrag.corpus.qa_loader. Enforces loud failure on unresolved doc_ids
unless explicitly overridden via --allow-unresolved.
"""

import argparse
from dataclasses import asdict
import json
import os
import sys

# Ensure repo root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from regrag.corpus.qa_loader import load_and_report


def main() -> None:
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    default_csv = os.path.join(repo_root, "data", "gold", "bank_qa_data.csv")
    default_out = os.path.join(repo_root, "data", "gold", "questions_canonical.json")

    parser = argparse.ArgumentParser(
        description="Load QA CSV into canonical GoldQuestion JSON and report coverage."
    )
    parser.add_argument(
        "--csv",
        default=default_csv,
        help=f"Path to QA CSV file (default: {default_csv})",
    )
    parser.add_argument(
        "--out",
        default=default_out,
        help=f"Path to output JSON file (default: {default_out})",
    )
    parser.add_argument(
        "--report",
        action="store_true",
        help="Print coverage report as JSON",
    )
    parser.add_argument(
        "--allow-unresolved",
        action="store_true",
        help="Allow unresolved doc_ids without failing with non-zero exit code",
    )
    args = parser.parse_args()

    csv_path = os.path.abspath(args.csv)
    out_path = os.path.abspath(args.out)

    questions, report = load_and_report(csv_path, repo_root=repo_root)

    # Write canonical GoldQuestion list as JSON
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    serialized = [asdict(q) for q in questions]
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(serialized, f, ensure_ascii=False, indent=2)
    print(f"Wrote {len(questions)} canonical questions to {out_path}", file=sys.stderr)

    if args.report:
        print(json.dumps(report, ensure_ascii=False, indent=2))

    # Unanswerable probes are a deliberate benchmark class, not a data gap:
    # report them separately so they are never mistaken for unresolved rows.
    probe_count = report.get("unanswerable_probes", 0)
    if probe_count:
        print(
            f"\nUnanswerable probes: {probe_count} "
            f"(is_answerable=False, category='unanswerable', no doc_id, no gold passage)",
            file=sys.stderr,
        )
        print(
            f"  ids: {', '.join(report.get('unanswerable_probe_ids', []))}",
            file=sys.stderr,
        )

    anomalies = report.get("probe_anomalies", []) or []
    if anomalies:
        print(
            f"\nERROR: {len(anomalies)} row(s) have CONTRADICTORY unanswerable markers. "
            "They were KEPT (never dropped) and classified as answerable:",
            file=sys.stderr,
        )
        for a in anomalies:
            print(f"  - {a['id']}: {a['reason']}", file=sys.stderr)
        print(
            "  'sentinel-answer-with-gold' = answer says 'not in the corpus' but the row "
            "carries a doc_link and/or gold passage.",
            file=sys.stderr,
        )
        print(
            "  'no-gold-without-sentinel'   = row has neither doc_link nor gold passage but "
            "does not declare itself unanswerable.",
            file=sys.stderr,
        )

    empty_answerable = report.get("empty_passage", 0)
    if empty_answerable:
        print(
            f"\nERROR: {empty_answerable} ANSWERABLE row(s) have no gold passage and cannot be "
            f"scored for citation accuracy: {report.get('empty_passage_ids', [])}",
            file=sys.stderr,
        )

    # Check unresolved doc_ids (answerable rows only - probes have no instrument
    # to resolve and must not be counted here)
    unresolved_count = report.get("doc_ids_unresolved", 0)
    if unresolved_count > 0:
        print(
            f"\nWARNING: {unresolved_count} answerable question(s) have unresolved doc_ids!",
            file=sys.stderr,
        )
        print(
            "Unresolved rows to add to data/raw_legal/DOC_MANIFEST.json:",
            file=sys.stderr,
        )
        for row in report.get("unresolved_rows", []):
            urls_str = ", ".join(row.get("urls", []))
            print(
                f"  - {row['id']}: URLs: {urls_str} -> {row.get('doc_ids')}",
                file=sys.stderr,
            )

    blocking = unresolved_count > 0 or bool(anomalies) or empty_answerable > 0
    if blocking and not args.allow_unresolved:
        reasons = []
        if unresolved_count > 0:
            reasons.append(f"{unresolved_count} unresolved doc_id(s)")
        if anomalies:
            reasons.append(f"{len(anomalies)} contradictory unanswerable row(s)")
        if empty_answerable > 0:
            reasons.append(f"{empty_answerable} answerable row(s) with no gold passage")
        print(
            f"\nError: {'; '.join(reasons)}. "
            "Failing non-zero (pass --allow-unresolved to bypass).",
            file=sys.stderr,
        )
        sys.exit(1)

    sys.exit(0)


if __name__ == "__main__":
    main()
