#!/usr/bin/env python3
"""Single-command regeneration entry point for RegRAG-VN benchmark.

Rebuilds all CSV-derived artifacts in dependency order with content hashing
for idempotent re-runs.
"""

from dataclasses import asdict
from datetime import datetime, timezone
import glob
import hashlib
import json
import os
import subprocess
import sys
from typing import Any, Dict, List, Optional, Sequence

# Ensure repo root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from regrag.corpus.qa_loader import load_and_report, load_gold_questions
from regrag.corpus.canonical import normalize_ws, fold_diacritics, squash_ws
from regrag.provenance import CORPUS_UNSET


def check_passage_coverage(
    questions: Sequence[Any],
    chunks: Sequence[Any],
) -> Dict[str, Any]:
    """Check passage coverage of canonical questions against legal chunks.

    Two-tier evaluation:
      - Strict tier: matches normalized text (NFC, collapsed whitespace, preserving diacritics).
      - Loose tier: matches diacritic-folded text. Matches at this tier ONLY mean
        the ingested chunk text lost diacritics and are reported as a distinct category.

    A third, intermediate squash tier (NFC with ALL whitespace removed on both
    sides) matches the born-digital CÔNG BÁO layers whose intra-word spacing is
    damaged ("vi ệc"); a squash match means the characters are all present and
    in order, and it is reported as its own category, never counted as strict.

    Unanswerable probes (``is_answerable=False``) have no gold passage by design,
    so they are EXCLUDED from the invariant and counted separately in
    ``unanswerable_probes_excluded`` - they must never drag coverage down, and
    they must never be silently invisible either. An *answerable* row with an
    empty passage is a different thing: that is a data defect, and it is counted
    in ``answerable_without_passage`` so it cannot hide behind the probe bucket.
    """
    total = len(questions)
    answerable = 0
    strict_matches = 0
    squash_matches = 0
    loose_matches = 0
    not_found = 0
    missing_ids: List[str] = []
    squash_ids: List[str] = []
    loose_ids: List[str] = []
    probe_ids: List[str] = []
    no_passage_ids: List[str] = []

    # Pre-process chunks for faster comparison
    chunk_data = []
    for c in chunks:
        cid = getattr(c, "chunk_id", None) or (c.get("chunk_id", "") if isinstance(c, dict) else "")
        raw_text = getattr(c, "text", None) or (c.get("text", "") if isinstance(c, dict) else "")
        source = getattr(c, "corpus_source", None) or (c.get("corpus_source", CORPUS_UNSET) if isinstance(c, dict) else CORPUS_UNSET)
        s_norm = normalize_ws(raw_text)
        l_norm = fold_diacritics(raw_text)
        q_norm = squash_ws(raw_text)
        chunk_data.append((cid, s_norm, l_norm, q_norm, source))

    for q in questions:
        qid = getattr(q, "id", None) or (q.get("id", "") if isinstance(q, dict) else "")
        is_ans = getattr(q, "is_answerable", True) if not isinstance(q, dict) else q.get("is_answerable", True)
        passage = getattr(q, "gold_passage", None) or (q.get("gold_passage", "") if isinstance(q, dict) else "")

        if not is_ans:
            # Unanswerable probe: excluded from the gold-passage invariant.
            probe_ids.append(qid)
            continue
        if not (passage or "").strip():
            # Answerable but has no gold passage: a real defect, reported apart.
            no_passage_ids.append(qid)
            continue

        answerable += 1
        p_strict = normalize_ws(passage)
        p_loose = fold_diacritics(passage)
        p_squash = squash_ws(passage)

        # 1. Strict tier check
        matched_strict = False
        for cid, c_strict, _, _, _ in chunk_data:
            if not c_strict:
                continue
            if p_strict in c_strict or p_strict == c_strict or (len(c_strict) >= 30 and c_strict in p_strict):
                matched_strict = True
                break

        if matched_strict:
            strict_matches += 1
            continue

        # 2. Squash tier check: whitespace removed on both sides. This matches
        # intra-word spacing damage in the CÔNG BÁO born-digital layers while
        # still requiring every character, diacritics included, in order.
        matched_squash = False
        for cid, _, _, c_squash, _ in chunk_data:
            if not c_squash:
                continue
            if p_squash in c_squash or p_squash == c_squash or (len(c_squash) >= 30 and c_squash in p_squash):
                matched_squash = True
                break

        if matched_squash:
            squash_matches += 1
            squash_ids.append(qid)
            continue

        # 3. Loose tier check (diacritic folded)
        matched_loose = False
        for cid, _, c_loose, _, _ in chunk_data:
            if not c_loose:
                continue
            if p_loose in c_loose or p_loose == c_loose or (len(c_loose) >= 30 and c_loose in p_loose):
                matched_loose = True
                break

        if matched_loose:
            loose_matches += 1
            loose_ids.append(qid)
        else:
            not_found += 1
            missing_ids.append(qid)

    strict_pct = (strict_matches / answerable * 100.0) if answerable else 0.0
    total_pct = ((strict_matches + squash_matches + loose_matches) / answerable * 100.0) if answerable else 0.0
    squash_pct = (squash_matches / answerable * 100.0) if answerable else 0.0
    loose_pct = (loose_matches / answerable * 100.0) if answerable else 0.0

    return {
        "total_questions": total,
        "answerable_questions": answerable,
        "unanswerable_probes_excluded": len(probe_ids),
        "unanswerable_probe_ids": probe_ids,
        "answerable_without_passage": len(no_passage_ids),
        "answerable_without_passage_ids": no_passage_ids,
        "strict_matches": strict_matches,
        "squash_matches": squash_matches,
        "loose_matches": loose_matches,
        "not_found": not_found,
        "strict_coverage_pct": round(strict_pct, 2),
        "squash_coverage_pct": round(squash_pct, 2),
        "loose_coverage_pct": round(loose_pct, 2),
        "total_coverage_pct": round(total_pct, 2),
        "squash_question_ids": squash_ids,
        "loose_question_ids": loose_ids,
        "missing_question_ids": missing_ids,
    }


def cache_key(
    question_id: str,
    model_name: str,
    retrieval_mode: str,
    csv_hash: str,
    corpus_hash: str,
    question_content: Optional[Any] = None,
) -> str:
    """Compute a deterministic cache key for a generated model response.

    Why per-question invalidation matters:
    The CSV undergoes frequent edits during benchmark authoring (e.g. fixing typos,
    adding references). If the cache key depended directly on the whole-file `csv_hash`,
    editing 6 questions in a 100-question CSV would invalidate all 2,000+ cached model
    generations, requiring costly re-runs.

    To prevent this, when `question_content` (the question text, GoldQuestion, or
    per-question content string) is provided, the cache key incorporates the hash of that
    specific question's content and the corpus hash, deliberately omitting the global
    `csv_hash`. Thus, untouched questions keep their stable cache keys across CSV revisions,
    while edited questions are cleanly invalidated. If `question_content` is omitted,
    the key falls back to binding to `csv_hash`.
    """
    if question_content is not None:
        if isinstance(question_content, str):
            content_str = question_content
        elif hasattr(question_content, "question"):
            # GoldQuestion object or similar
            q_text = getattr(question_content, "question", "")
            ans = getattr(question_content, "reference_answer", "")
            pas = getattr(question_content, "gold_passage", "")
            content_str = f"{q_text}|{ans}|{pas}"
        elif isinstance(question_content, dict):
            content_str = f"{question_content.get('question', '')}|{question_content.get('reference_answer', '')}|{question_content.get('gold_passage', '')}"
        else:
            content_str = str(question_content)

        q_hash = hashlib.sha256(content_str.encode("utf-8")).hexdigest()[:16]
        payload = f"{question_id}:{model_name}:{retrieval_mode}:{q_hash}:{corpus_hash}"
    else:
        payload = f"{question_id}:{model_name}:{retrieval_mode}:{csv_hash}:{corpus_hash}"

    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]


def regenerate(repo_root: Optional[str] = None) -> Dict[str, Any]:
    root = repo_root or os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    csv_path = os.path.join(root, "data", "gold", "bank_qa_data.csv")
    state_path = os.path.join(root, "data", "gold", ".regenerate_state.json")
    canonical_out = os.path.join(root, "data", "gold", "questions_canonical.json")
    chunks_dir = os.path.join(root, "data", "processed_chunks")

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Gold QA CSV not found at {csv_path}")

    # Compute CSV content hash
    with open(csv_path, "rb") as f:
        csv_bytes = f.read()
    current_csv_hash = hashlib.sha256(csv_bytes).hexdigest()

    # Check previous state for idempotency / no-op detection
    is_noop = False
    if os.path.exists(state_path) and os.path.exists(canonical_out):
        try:
            with open(state_path, "r", encoding="utf-8") as f:
                prev_state = json.load(f)
            if prev_state.get("csv_hash") == current_csv_hash:
                is_noop = True
        except Exception:
            is_noop = False

    if is_noop:
        print(f"[*] CSV unchanged (hash: {current_csv_hash[:12]}...). Skipping re-derivation (NO-OP).")
        questions = load_gold_questions(csv_path, repo_root=root)
        from regrag.corpus.qa_loader import coverage_report
        report = coverage_report(questions)
    else:
        print(f"[*] CSV modified or initial run (hash: {current_csv_hash[:12]}...). Regenerating artifacts...")
        # Step 1 & 2: load CSV and write canonical JSON
        questions, report = load_and_report(csv_path, repo_root=root)
        os.makedirs(os.path.dirname(canonical_out), exist_ok=True)
        serialized = [asdict(q) for q in questions]
        with open(canonical_out, "w", encoding="utf-8") as f:
            json.dump(serialized, f, ensure_ascii=False, indent=2)
        print(f"    - Wrote {len(questions)} canonical questions to {canonical_out}")

        # Step 3: Invoke Tier 1 corpus builder if it exists
        tier1_script = os.path.join(root, "scripts", "build_tier1_corpus.py")
        if os.path.exists(tier1_script):
            print(f"    - Invoking Tier 1 corpus builder: {tier1_script}")
            try:
                res = subprocess.run([sys.executable, tier1_script], capture_output=True, text=True)
                if res.returncode != 0:
                    print(f"    ! Warning: build_tier1_corpus.py exited with code {res.returncode}:\n{res.stderr.strip()}", file=sys.stderr)
                else:
                    print("    - Tier 1 corpus builder finished successfully.")
            except Exception as e:
                print(f"    ! Failed to invoke Tier 1 corpus builder: {e}", file=sys.stderr)
        else:
            print(f"    - Note: Tier 1 builder script not found at {tier1_script} (skipping)")

        # Record new state
        new_state = {
            "csv_hash": current_csv_hash,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "questions_count": len(questions),
            "answerable_questions": report.get("answerable_questions", 0),
            "unanswerable_probes": report.get("unanswerable_probes", 0),
            "doc_ids_resolved": report.get("doc_ids_resolved", 0),
            "doc_ids_unresolved": report.get("doc_ids_unresolved", 0),
        }
        with open(state_path, "w", encoding="utf-8") as f:
            json.dump(new_state, f, indent=2)

    # Step 4: Report coverage against whatever chunk files exist in data/processed_chunks/
    chunk_files = sorted(glob.glob(os.path.join(chunks_dir, "*.json")))
    chunk_reports = {}
    for cf in chunk_files:
        rel_cf = os.path.relpath(cf, root)
        try:
            with open(cf, "r", encoding="utf-8") as f:
                c_data = json.load(f)
            by_source: Dict[str, int] = {}
            for chunk in c_data:
                src = chunk.get("corpus_source", CORPUS_UNSET) if isinstance(chunk, dict) else getattr(chunk, "corpus_source", CORPUS_UNSET)
                by_source[src] = by_source.get(src, 0) + 1
            cov = check_passage_coverage(questions, c_data)
            chunk_reports[rel_cf] = {
                "chunk_count": len(c_data),
                "chunks_by_source": by_source,
                "coverage": cov,
            }
        except Exception as e:
            chunk_reports[rel_cf] = {"error": str(e)}

    # Print clean final summary
    print("\n" + "=" * 60)
    print("REGENERATION SUMMARY")
    print("=" * 60)
    status_label = "NO-OP (CSV unchanged)" if is_noop else "REGENERATED"
    print(f"Status:             {status_label}")
    print(f"CSV Content Hash:   {current_csv_hash[:16]}...")
    print(f"Total Questions:    {len(questions)}")
    print(f"  - Answerable:     {report.get('answerable_questions', 0)}")
    print(f"  - Unanswerable probes: {report.get('unanswerable_probes', 0)} (excluded from gold-passage coverage)")
    print(f"  - Resolved IDs:   {report.get('doc_ids_resolved', 0)}")
    print(f"  - Unresolved IDs: {report.get('doc_ids_unresolved', 0)} (answerable rows only)")
    print(f"Distinct Instruments: {len(report.get('distinct_doc_ids', []))}")

    anomalies = report.get("probe_anomalies", []) or []
    if anomalies:
        print(
            f"\n!! {len(anomalies)} row(s) have CONTRADICTORY unanswerable markers "
            "(kept, not dropped - fix the CSV):",
            file=sys.stderr,
        )
        for a in anomalies:
            print(f"  - {a['id']}: {a['reason']}", file=sys.stderr)

    if chunk_reports:
        print("\nProcessed Chunk Files Coverage:")
        for cf, cr in chunk_reports.items():
            if "error" in cr:
                print(f"  - {cf}: error ({cr['error']})")
                continue
            cov = cr["coverage"]
            print(f"  - {cf} ({cr['chunk_count']} chunks):")
            print(f"      By corpus source: {cr['chunks_by_source']}")
            print(
                f"      Strict match:    {cov['strict_matches']}/{cov['answerable_questions']} "
                f"({cov['strict_coverage_pct']}%)"
            )
            print(
                f"      Squash match:    {cov['squash_matches']}/{cov['answerable_questions']} "
                f"({cov['squash_coverage_pct']}%) [intra-word spacing damaged layers]"
            )
            print(
                f"      Loose match:     {cov['loose_matches']}/{cov['answerable_questions']} "
                f"({cov['loose_coverage_pct']}%)"
            )
            print(f"      Not found:       {cov['not_found']}/{cov['answerable_questions']}")
            print(
                f"      Probes excluded: {cov['unanswerable_probes_excluded']}"
            )
            if cov.get("answerable_without_passage"):
                print(
                    f"      !! Answerable rows with NO gold passage (defect, not a probe): "
                    f"{cov['answerable_without_passage']} -> {cov['answerable_without_passage_ids']}",
                    file=sys.stderr,
                )
    else:
        print("\nNo chunk files found in data/processed_chunks/ to evaluate.")

    print("=" * 60 + "\n")

    return {
        "is_noop": is_noop,
        "csv_hash": current_csv_hash,
        "questions_count": len(questions),
        "report": report,
        "chunk_reports": chunk_reports,
    }


if __name__ == "__main__":
    regenerate()
