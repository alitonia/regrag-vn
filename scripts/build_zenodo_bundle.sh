#!/usr/bin/env bash
# build_zenodo_bundle.sh — assemble the Zenodo deposit bundle from git HEAD.
#
# The bundle is built ONLY from tracked files at HEAD (git archive), so what
# goes to Zenodo is exactly what is committed — never a dirty working tree.
# pod_artifacts/ (raw campaign archive) and paper/main.pdf are tracked and
# included. Excluded by nature of git archive: .venv, caches, data/human/
# label sheets if untracked (add them to git first if they belong in the
# release).
#
# Usage: bash scripts/build_zenodo_bundle.sh [output-dir]
# Output: <output-dir>/regrag-vn-rivf2026/ + SHA256SUMS, size printed.
set -euo pipefail
cd "$(dirname "$0")/.."

OUT="${1:-dist/zenodo_bundle}"
NAME="regrag-vn-rivf2026"

# Refuse a dirty tree: the DOI must point at exactly one commit state.
if [ -n "$(git status --porcelain)" ]; then
  echo "[FATAL] working tree is dirty — commit or stash before building the bundle:" >&2
  git status --porcelain >&2 | head -10
  exit 1
fi

COMMIT="$(git rev-parse --short HEAD)"
rm -rf "$OUT"
mkdir -p "$OUT"

git archive --format=tar.gz --prefix="$NAME/" -o "$OUT/$NAME-$COMMIT.tar.gz" HEAD

# Also lay the tree out unpacked for Zenodo's web UI (which shows archives
# poorly): same content, plain files.
git archive --format=tar --prefix="$NAME/" HEAD | tar xf - -C "$OUT"

# The raw ingested sources are gitignored (repo policy: manifest carries
# URL+sha1 provenance), but the paper's Data and Code Availability paragraph
# promises "legal texts redistributed as downloaded" — so the deposit includes
# them from the working tree, pinned by DOC_MANIFEST.json sha1s.
cp -r data/raw_legal "$OUT/$NAME/data/raw_legal"

( cd "$OUT/$NAME" && find . -type f -exec sha256sum {} \; ) > "$OUT/SHA256SUMS.txt"

SIZE=$(du -sh "$OUT" | cut -f1)
echo "[OK] bundle at $OUT (commit $COMMIT, $SIZE)"
echo "     archive: $OUT/$NAME-$COMMIT.tar.gz"
echo "     tree:    $OUT/$NAME/"
echo "     next:    bash scripts/zenodo_deposit.sh $OUT/$NAME   (needs ZENODO_ACCESS_TOKEN)"
