"""Crash-safe checkpointing and repo-format materialisation of campaign rows.

Two artifacts, two jobs
-----------------------
``checkpoints/generations.jsonl``  - append-only, one complete record per line,
    fsync'd as soon as the row is produced. This is the resume medium. JSONL is
    chosen deliberately over a single JSON array: when a Colab session is killed
    mid-write, an array is left unparseable and the WHOLE run is lost, whereas a
    truncated final *line* costs exactly one row. The loader tolerates and
    reports that truncation instead of raising.

``results/<run>/generations.json`` - the repo's own record format, written
    through ``regrag/storage/file_repo.FileResultRepository`` so
    ``regrag/evaluation/`` consumes it directly. Materialised periodically and
    at the end, never per row.

Why a subclass of FileResultRepository
--------------------------------------
``FileResultRepository.save_generation`` rewrites the entire JSON file on every
call. For a 3 x 3 x 88 = 792-row campaign that is ~792 full rewrites of a
growing file (hundreds of MB of pointless I/O, and far worse on the Drive FUSE
mount). ``BatchedFileResultRepository`` defers the flush to explicit calls and
reuses the parent's own atomic writer, so the on-disk format is byte-for-byte
the repo's - nothing new is invented.

Schema contract
---------------
``GenerationResult`` has a FIXED field set, and ``FileResultRepository`` reloads
rows with ``GenerationResult(**item)``. An extra key therefore makes
``generations.json`` unloadable by the repo's own storage layer. So the
generation-backend and quantization provenance that the campaign must record
cannot live in that file; it is written to ``results/<run>/generations_meta.jsonl``
and joined on ``cache_key``. ``assert_meta_complete`` refuses to call a run
publishable when any row lacks its meta entry, so the provenance cannot be
silently dropped between the two files.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import time
from dataclasses import MISSING as _MISSING, asdict
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from regrag.models import GenerationResult
from regrag.provenance import ProvenanceError, is_degraded
from regrag.storage.file_repo import FileResultRepository

#: Fields that make up a repo-format GenerationResult row.
GENERATION_FIELDS: Tuple[str, ...] = tuple(GenerationResult.__dataclass_fields__)  # type: ignore[attr-defined]

#: Fields with no dataclass default - a record without them cannot be built.
_MANDATORY_GENERATION_FIELDS: Tuple[str, ...] = tuple(
    name
    for name, f in GenerationResult.__dataclass_fields__.items()  # type: ignore[attr-defined]
    if f.default is _MISSING and f.default_factory is _MISSING  # type: ignore[misc]
)

#: Provenance fields that DO have defaults but must never be left to default.
#: A row missing corpus_source would silently become CORPUS_UNSET, which reads
#: like a legitimate dataclass default rather than "this row has no provenance".
_REQUIRED_PROVENANCE_FIELDS: Tuple[str, ...] = ("corpus_source", "retriever_backend")

#: Extra provenance carried per row in the sidecar, joined on cache_key.
META_FIELDS: Tuple[str, ...] = (
    "generation_backend",
    "quant_config",
    "hf_model_id",
    "served_alias",
    "model_set_role",
    "template_applied_by",
    "prompt_exact",
    "gpu_names",
    "gpu_memory_gb",
    "sampling",
    "latency_s",
    "finish_reason",
    "usage",
    "is_answerable",
    "abstention_matched_exact",
    "abstention_soft_patterns",
    "corpus_path",
    "retrieved_chunk_ids",
    "retrieved_scores",
    "generated_at",
)


class BatchedFileResultRepository(FileResultRepository):
    """FileResultRepository with a deferred flush. Identical on-disk format."""

    def save_generation(self, result: GenerationResult) -> None:
        self._gen_cache.append(result)

    def flush(self) -> None:
        """Persist generations (and evaluations) using the parent's atomic writer."""
        self._flush_generations()


def split_record(record: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any], List[str]]:
    """Split one checkpoint record into (GenerationResult kwargs, meta, unknown).

    Unknown keys are returned rather than dropped silently, so a schema drift
    shows up as a warning instead of vanishing from the results.
    """
    known = set(GENERATION_FIELDS) | set(META_FIELDS) | {"cache_key"}
    gen_kwargs = {k: record[k] for k in GENERATION_FIELDS if k in record}
    meta = {k: record[k] for k in META_FIELDS if k in record}
    unknown = sorted(k for k in record if k not in known)
    return gen_kwargs, meta, unknown


class CheckpointStore:
    """Append-only resume store keyed by cache_key."""

    def __init__(
        self,
        checkpoint_dir: str,
        filename: str = "generations.jsonl",
        warn_stream=None,
    ) -> None:
        self.checkpoint_dir = checkpoint_dir
        self.path = os.path.join(checkpoint_dir, filename)
        self._stream = warn_stream if warn_stream is not None else sys.stderr
        os.makedirs(checkpoint_dir, exist_ok=True)

        self._records: Dict[str, Dict[str, Any]] = {}
        self._insertion_order: List[str] = []
        self.skipped_lines: List[Dict[str, Any]] = []
        self.duplicate_keys: List[str] = []
        self._load()

    # -- load / resume -------------------------------------------------------
    def _load(self) -> None:
        if not os.path.exists(self.path):
            self._stream.write(
                f"[CHECKPOINT] No checkpoint at {self.path} - starting fresh.\n"
            )
            self._stream.flush()
            return

        n_ok = 0
        with open(self.path, "r", encoding="utf-8") as f:
            for lineno, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError as exc:
                    self.skipped_lines.append(
                        {"lineno": lineno, "reason": f"unparseable: {exc}", "chars": len(line)}
                    )
                    continue
                if not isinstance(rec, dict):
                    self.skipped_lines.append(
                        {"lineno": lineno, "reason": "not a JSON object", "chars": len(line)}
                    )
                    continue
                key = rec.get("cache_key")
                if not key:
                    self.skipped_lines.append(
                        {"lineno": lineno, "reason": "record has no cache_key", "chars": len(line)}
                    )
                    continue
                if key in self._records:
                    self.duplicate_keys.append(key)
                else:
                    self._insertion_order.append(key)
                # Last write wins: a re-run of the same cell supersedes the older row.
                self._records[key] = rec
                n_ok += 1

        if self.skipped_lines:
            self._stream.write(
                f"[CHECKPOINT][WARNING] {len(self.skipped_lines)} line(s) in {self.path} "
                f"could not be used and were skipped: {self.skipped_lines[:3]}. A single "
                "truncated final line is the expected artifact of a session killed "
                "mid-write; more than one suggests real corruption.\n"
            )
            self._stream.flush()
        if self.duplicate_keys:
            self._stream.write(
                f"[CHECKPOINT] {len(set(self.duplicate_keys))} cache_key(s) appear more "
                "than once; the last occurrence wins.\n"
            )
            self._stream.flush()

        self._stream.write(
            f"[CHECKPOINT] Restored {len(self._records)} completed row(s) from {self.path} "
            f"({n_ok} lines parsed).\n"
        )
        self._stream.flush()

    # -- query ---------------------------------------------------------------
    def __len__(self) -> int:
        return len(self._records)

    @property
    def completed_keys(self) -> Set[str]:
        return set(self._records)

    def has(self, cache_key: str) -> bool:
        return cache_key in self._records

    def get(self, cache_key: str) -> Optional[Dict[str, Any]]:
        return self._records.get(cache_key)

    def records(self) -> List[Dict[str, Any]]:
        return [self._records[k] for k in self._insertion_order if k in self._records]

    def purge(self, predicate, results_dir: Optional[str] = None) -> List[str]:
        """Drop rows whose record matches ``predicate(record)``; rewrite the
        checkpoint JSONL atomically. Returns the purged cache_keys.

        For retiring cached rows whose prompt policy changed mid-campaign
        (e.g. rag rows generated before the RAG_CONTEXT_* budget): a cached
        row is skipped forever, so the only way to regenerate it under the
        new policy is to remove it from the store first. When ``results_dir``
        is given, matching rows are also removed from the materialised
        generations.json / generations_meta.jsonl - materialise() only ever
        adds to those files, so a purged row would otherwise survive in the
        analysis output forever.
        """
        doomed = [k for k, rec in self._records.items() if predicate(rec)]
        if not doomed:
            return []
        doomed_set = set(doomed)
        for k in doomed:
            self._records.pop(k, None)
        self._insertion_order = [k for k in self._insertion_order if k in self._records]
        tmp = self.path + ".purge-tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            for rec in self.records():
                f.write(json.dumps(rec, ensure_ascii=False, sort_keys=True) + "\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, self.path)

        if results_dir:
            gen_path = os.path.join(results_dir, "generations.json")
            if os.path.exists(gen_path):
                with open(gen_path, "r", encoding="utf-8") as f:
                    rows = json.load(f)
                kept = [r for r in rows if r.get("cache_key") not in doomed_set]
                tmp_gen = gen_path + ".purge-tmp"
                with open(tmp_gen, "w", encoding="utf-8") as f:
                    json.dump(kept, f, ensure_ascii=False, indent=2)
                    f.flush()
                    os.fsync(f.fileno())
                os.replace(tmp_gen, gen_path)
            meta_path = os.path.join(results_dir, "generations_meta.jsonl")
            if os.path.exists(meta_path):
                kept_lines = []
                with open(meta_path, "r", encoding="utf-8") as f:
                    for line in f:
                        try:
                            if json.loads(line).get("cache_key") in doomed_set:
                                continue
                        except json.JSONDecodeError:
                            pass  # torn line: keep as-is, not our concern here
                        kept_lines.append(line if line.endswith("\n") else line + "\n")
                tmp_meta = meta_path + ".purge-tmp"
                with open(tmp_meta, "w", encoding="utf-8") as f:
                    f.writelines(kept_lines)
                    f.flush()
                    os.fsync(f.fileno())
                os.replace(tmp_meta, meta_path)
        return doomed

    def backends_present(self) -> Dict[str, int]:
        """Row counts per generation_backend, for the pooling guard."""
        counts: Dict[str, int] = {}
        for rec in self._records.values():
            tag = rec.get("generation_backend", "<missing>")
            counts[tag] = counts.get(tag, 0) + 1
        return counts

    def degraded_rows(self) -> List[Dict[str, Any]]:
        """Rows whose generation_backend or retriever_backend is DEGRADED/UNSET."""
        out = []
        for rec in self._records.values():
            gb = rec.get("generation_backend", "")
            rb = rec.get("retriever_backend", "")
            if is_degraded(gb) or is_degraded(rb):
                out.append(
                    {
                        "cache_key": rec.get("cache_key"),
                        "question_id": rec.get("question_id"),
                        "model_name": rec.get("model_name"),
                        "retrieval_mode": rec.get("retrieval_mode"),
                        "generation_backend": gb,
                        "retriever_backend": rb,
                    }
                )
        return out

    # -- write ---------------------------------------------------------------
    def append(self, record: Dict[str, Any]) -> None:
        """Persist one row durably. Fsync'd: a session kill loses at most this row."""
        key = record.get("cache_key")
        if not key:
            raise ProvenanceError(
                "Refusing to checkpoint a record with no cache_key: it could never be "
                "recognised on resume, so the cell would be re-run and duplicated."
            )
        missing = [k for k in _MANDATORY_GENERATION_FIELDS if k not in record]
        missing_prov = [
            k for k in _REQUIRED_PROVENANCE_FIELDS
            if k not in record or not str(record[k] or "").strip()
        ]
        if missing or missing_prov:
            raise ProvenanceError(
                f"Record for cache_key={key} is missing GenerationResult field(s) "
                f"{missing + missing_prov}. Refusing to checkpoint it: a row without "
                "corpus_source/retriever_backend would fall back to the dataclass "
                "default UNSET, which reads like a legitimate value rather than "
                "'this row has no provenance'."
            )

        line = json.dumps(record, ensure_ascii=False, sort_keys=True)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(line + "\n")
            f.flush()
            os.fsync(f.fileno())

        if key in self._records:
            self.duplicate_keys.append(key)
        else:
            self._insertion_order.append(key)
        self._records[key] = record

    # -- materialise into the repo's own format ------------------------------
    def materialize(
        self,
        results_dir: str,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """Write generations.json (+ generations_meta.jsonl) in the repo format."""
        os.makedirs(results_dir, exist_ok=True)
        repo = BatchedFileResultRepository(results_dir)

        # Rows already on disk, so re-materialising is idempotent.
        existing = {
            g.cache_key for g in repo.list_generations() if getattr(g, "cache_key", "")
        }
        added = 0
        for rec in self.records():
            if rec.get("cache_key") in existing:
                continue
            gen_kwargs, _meta, unknown = split_record(rec)
            if unknown and verbose:
                self._stream.write(
                    f"[CHECKPOINT][WARNING] record {rec.get('cache_key')} carries "
                    f"unknown field(s) {unknown}; they are not part of GenerationResult "
                    "and are kept only in the meta sidecar.\n"
                )
                self._stream.flush()
            repo.save_generation(GenerationResult(**gen_kwargs))
            added += 1
        repo.flush()

        meta_path = os.path.join(results_dir, "generations_meta.jsonl")
        meta_rows = 0
        with open(meta_path, "w", encoding="utf-8") as f:
            for rec in self.records():
                _gen, meta, unknown = split_record(rec)
                # Unknown fields (e.g. rag_context_* stamped by newer drivers)
                # belong in the meta sidecar, not in the void: the warning
                # above promises they survive here, so merge them.
                for k in unknown:
                    meta[k] = rec[k]
                meta["cache_key"] = rec.get("cache_key")
                meta["question_id"] = rec.get("question_id")
                meta["model_name"] = rec.get("model_name")
                meta["retrieval_mode"] = rec.get("retrieval_mode")
                f.write(json.dumps(meta, ensure_ascii=False, sort_keys=True) + "\n")
                meta_rows += 1

        if verbose:
            self._stream.write(
                f"[CHECKPOINT] Materialised {added} new row(s) into {results_dir} "
                f"(generations.json total={len(repo.list_generations())}; "
                f"{meta_rows} meta rows in generations_meta.jsonl).\n"
            )
            self._stream.flush()

        return {
            "results_dir": results_dir,
            "rows_added": added,
            "rows_total": len(repo.list_generations()),
            "meta_rows": meta_rows,
            "generations_json": os.path.join(results_dir, "generations.json"),
            "meta_jsonl": meta_path,
        }

    def assert_meta_complete(self, results_dir: str) -> None:
        """Fail loudly if any repo-format row lacks its provenance sidecar entry."""
        repo = FileResultRepository(results_dir)
        meta_path = os.path.join(results_dir, "generations_meta.jsonl")
        if not os.path.exists(meta_path):
            if repo.list_generations():
                raise ProvenanceError(
                    f"{results_dir}/generations.json holds {len(repo.list_generations())} "
                    "rows but generations_meta.jsonl is missing, so no row has a "
                    "generation_backend or quant_config. Refusing to treat this run as "
                    "publishable: rows from different backends must never be pooled."
                )
            return
        keys: Set[str] = set()
        with open(meta_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    keys.add(json.loads(line).get("cache_key"))
                except json.JSONDecodeError:
                    continue
        orphans = [
            g.cache_key for g in repo.list_generations() if g.cache_key and g.cache_key not in keys
        ]
        if orphans:
            raise ProvenanceError(
                f"{len(orphans)} row(s) in generations.json have no entry in "
                f"generations_meta.jsonl (e.g. {orphans[:5]}). Their generation_backend "
                "and quant_config are unknown, so they cannot be safely pooled."
            )


# --- Drive sync --------------------------------------------------------------


def restore_from_drive(
    drive_dir: str,
    local_dir: str,
    names: Sequence[str] = ("generations.jsonl", "run_manifest.json"),
    warn_stream=None,
) -> Dict[str, Any]:
    """Copy checkpoints from Drive into /content so a restart resumes.

    Prints a status report either way, per the colab skill: never silently start
    over when a checkpoint exists.
    """
    stream = warn_stream if warn_stream is not None else sys.stderr
    os.makedirs(local_dir, exist_ok=True)
    report: Dict[str, Any] = {"drive_dir": drive_dir, "restored": [], "missing": [], "errors": []}

    if not os.path.isdir(drive_dir):
        stream.write(
            f"[RESTORE] Drive directory {drive_dir} does not exist - FRESH RUN, no "
            "prior checkpoint. (Is Drive mounted?)\n"
        )
        stream.flush()
        return report

    for name in names:
        src = os.path.join(drive_dir, name)
        if not os.path.exists(src):
            report["missing"].append(name)
            continue
        dst = os.path.join(local_dir, name)
        try:
            st = os.stat(src)
            shutil.copy2(src, dst)
            report["restored"].append(
                {
                    "name": name,
                    "bytes": st.st_size,
                    "mtime_utc": datetime.fromtimestamp(st.st_mtime, tz=timezone.utc).isoformat(),
                }
            )
        except Exception as exc:
            report["errors"].append({"name": name, "error": f"{type(exc).__name__}: {exc}"})

    if report["restored"]:
        stream.write(f"[RESTORE] Restored from {drive_dir}:\n")
        for r in report["restored"]:
            stream.write(
                f"    - {r['name']}  {r['bytes']} bytes  modified {r['mtime_utc']}\n"
            )
        jsonl = os.path.join(local_dir, "generations.jsonl")
        if os.path.exists(jsonl):
            with open(jsonl, "r", encoding="utf-8") as f:
                n = sum(1 for line in f if line.strip())
            stream.write(f"[RESTORE] {n} completed row(s) available for resume.\n")
    else:
        stream.write(
            f"[RESTORE] FRESH RUN - no checkpoint found in {drive_dir} "
            f"(looked for {list(names)}).\n"
        )
    if report["errors"]:
        stream.write(f"[RESTORE][ERROR] {report['errors']}\n")
    stream.flush()
    return report


def sync_to_drive(
    local_dir: str,
    drive_dir: str,
    names: Sequence[str] = ("generations.jsonl", "run_manifest.json"),
    extra_dirs: Sequence[Tuple[str, str]] = (),
    warn_stream=None,
) -> Dict[str, Any]:
    """Copy checkpoints (and optionally whole result dirs) up to Drive.

    ``extra_dirs`` is a list of (local_subdir, drive_subdir) pairs, used to push
    the materialised repo-format results next to the checkpoint.
    """
    stream = warn_stream if warn_stream is not None else sys.stderr
    os.makedirs(drive_dir, exist_ok=True)
    report: Dict[str, Any] = {"drive_dir": drive_dir, "copied": [], "errors": []}

    for name in names:
        src = os.path.join(local_dir, name)
        if not os.path.exists(src):
            continue
        try:
            shutil.copy2(src, os.path.join(drive_dir, name))
            report["copied"].append(name)
        except Exception as exc:
            report["errors"].append({"name": name, "error": f"{type(exc).__name__}: {exc}"})

    for local_sub, drive_sub in extra_dirs:
        src_dir = os.path.join(local_dir, local_sub) if not os.path.isabs(local_sub) else local_sub
        if not os.path.isdir(src_dir):
            continue
        dst_dir = os.path.join(drive_dir, drive_sub)
        try:
            os.makedirs(dst_dir, exist_ok=True)
            for fn in sorted(os.listdir(src_dir)):
                sp = os.path.join(src_dir, fn)
                if os.path.isfile(sp):
                    shutil.copy2(sp, os.path.join(dst_dir, fn))
                    report["copied"].append(f"{drive_sub}/{fn}")
        except Exception as exc:
            report["errors"].append(
                {"name": f"{local_sub}->{drive_sub}", "error": f"{type(exc).__name__}: {exc}"}
            )

    stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    stream.write(f"[SYNC] {stamp}  copied {len(report['copied'])} file(s) to {drive_dir}\n")
    if report["errors"]:
        stream.write(f"[SYNC][ERROR] {report['errors']}\n")
    stream.flush()
    return report


def write_manifest(path: str, manifest: Dict[str, Any]) -> None:
    """Atomically write the run manifest."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


class ProgressTracker:
    """Real progress with periodic ETA and a running count of rows written."""

    def __init__(self, total: int, every: int = 10, stream=None) -> None:
        self.total = total
        self.every = max(1, every)
        self._stream = stream if stream is not None else sys.stdout
        self.done = 0
        self.skipped = 0
        self.failed = 0
        self._t0 = time.time()
        self._last_print = 0.0

    def tick(self, kind: str = "done") -> None:
        if kind == "done":
            self.done += 1
        elif kind == "skipped":
            self.skipped += 1
        else:
            self.failed += 1
        processed = self.done + self.skipped + self.failed
        now = time.time()
        if processed - self._last_print >= self.every or processed >= self.total:
            self._last_print = processed
            self._print(processed, now)

    def _print(self, processed: int, now: float) -> None:
        elapsed = max(now - self._t0, 1e-9)
        # ETA is based on cells actually GENERATED, not skipped cache hits: a
        # resumed run that counts cached rows would report a nonsense ETA.
        rate = self.done / elapsed if self.done else 0.0
        remaining = max(self.total - processed, 0)
        if rate > 0:
            eta_s = remaining / rate
            eta = f"ETA {eta_s/60:.1f} min"
            per = f"{1.0/rate:.1f}s/gen"
        else:
            eta = "ETA n/a (no generations yet)"
            per = ""
        pct = 100.0 * processed / self.total if self.total else 0.0
        bar_n = 30
        filled = int(bar_n * processed / self.total) if self.total else bar_n
        bar = "#" * filled + "-" * (bar_n - filled)
        self._stream.write(
            f"[{bar}] {processed}/{self.total} ({pct:5.1f}%)  "
            f"rows_written={self.done} cached_skipped={self.skipped} failed={self.failed}  "
            f"elapsed={elapsed/60:.1f}min {per} {eta}\n"
        )
        self._stream.flush()

    def summary(self) -> Dict[str, Any]:
        elapsed = time.time() - self._t0
        return {
            "total": self.total,
            "rows_written": self.done,
            "cached_skipped": self.skipped,
            "failed": self.failed,
            "elapsed_s": round(elapsed, 1),
            "mean_s_per_generation": (
                round(elapsed / self.done, 2) if self.done else None
            ),
        }
