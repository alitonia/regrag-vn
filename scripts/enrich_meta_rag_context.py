"""Merge rag_context_* fields from a checkpoint JSONL into a meta sidecar.

The 2026-09-13 pod campaign predates the materialize() fix that merges
unknown store fields into generations_meta.jsonl, so the fetched meta lacks
the rag_context_* stamping that the store records carry. This backfills the
meta from the checkpoint, joined on cache_key.

Usage:
    python3 scripts/enrich_meta_rag_context.py <checkpoint.jsonl> <meta.jsonl>
"""

from __future__ import annotations

import json
import sys

FIELDS = (
    "rag_context_chars",
    "rag_context_budget_chars",
    "rag_context_truncated_ranks",
    "rag_context_dropped_ranks",
)


def main() -> int:
    if len(sys.argv) != 3:
        print(__doc__)
        return 2
    ckpt_path, meta_path = sys.argv[1], sys.argv[2]

    store: dict = {}
    with open(ckpt_path, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            store[rec["cache_key"]] = rec

    updated = 0
    lines = []
    with open(meta_path, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            src = store.get(rec.get("cache_key"))
            if src is not None:
                for field in FIELDS:
                    if field in src:
                        rec[field] = src[field]
                if any(field in rec for field in FIELDS):
                    updated += 1
            lines.append(json.dumps(rec, ensure_ascii=False, sort_keys=True))

    tmp = meta_path + ".enrich-tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    import os

    os.replace(tmp, meta_path)
    print(f"[ENRICH] {updated} meta rows updated in {meta_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
