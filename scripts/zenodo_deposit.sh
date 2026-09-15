#!/usr/bin/env bash
# zenodo_deposit.sh — create the Zenodo draft deposit, reserve the DOI, upload
# the bundle. Publishing is a SEPARATE explicit act: the paper's Data and Code
# Availability paragraph cites the reserved DOI, and the record goes public
# only when --publish is passed on submission day (owner rule: release is a
# decision, not a side effect).
#
# Prerequisites:
#   export ZENODO_ACCESS_TOKEN=...   (zenodo.org -> Applications -> Personal
#                                     access tokens, scope deposit:write)
#   bundle built: bash scripts/build_zenodo_bundle.sh
#
# Usage:
#   bash scripts/zenodo_deposit.sh dist/zenodo_bundle/regrag-vn-rivf2026
#   bash scripts/zenodo_deposit.sh dist/zenodo_bundle/regrag-vn-rivf2026 --publish
set -euo pipefail
cd "$(dirname "$0")/.."

BUNDLE_DIR="${1:?Usage: bash scripts/zenodo_deposit.sh <bundle-dir> [--publish]}"
PUBLISH=0
[ "${2:-}" = "--publish" ] && PUBLISH=1

TOKEN="${ZENODO_ACCESS_TOKEN:?Set ZENODO_ACCESS_TOKEN (zenodo.org -> Applications -> Personal access tokens, scope deposit:write)}"
API="https://zenodo.org/api/deposit/deposit"

# Never echo the token; redact it from any API error output.
redact() { sed -e "s#$(printf '%s' "$TOKEN" | sed 's/[.[\*^$]/\\&/g')#***TOKEN***#g"; }

echo "[zenodo] creating draft deposit..."
PAYLOAD=$(python3 -c "import json; print(json.dumps({'metadata': json.load(open('zenodo_metadata.json'))}))")
RESPONSE=$(curl -sS -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json" \
  -X POST "$API" -d "$PAYLOAD" 2>&1 | redact)

DEP_ID=$(echo "$RESPONSE" | python3 -c "
import sys, json
d = json.load(sys.stdin)
if 'id' not in d:
    sys.stderr.write(json.dumps(d)[:1500] + '\n'); sys.exit(1)
print(d['id'])")

RESPONSE=$(curl -sS -H "Authorization: Bearer $TOKEN" "$API/$DEP_ID" | redact)
PRERESERVED=$(echo "$RESPONSE" | python3 -c "import sys,json; print(json.load(sys.stdin)['metadata']['prereserve_doi']['doi'])")
BUCKET=$(echo "$RESPONSE" | python3 -c "import sys,json; print(json.load(sys.stdin)['links']['bucket'])")

echo
echo "=============================================================="
echo " RESERVED DOI (put this in paper/main.tex): $PRERESERVED"
echo "=============================================================="

echo "[zenodo] uploading bundle files from $BUNDLE_DIR ..."
find "$BUNDLE_DIR" -type f | sort | while read -r f; do
  rel="${f#"$BUNDLE_DIR"/}"
  echo "  -> $rel ($(du -h "$f" | cut -f1))"
  code=""
  for attempt in 1 2 3; do
    code=$(curl -sS -o /tmp/zenodo_up.json -w '%{http_code}' -X PUT \
      -H "Authorization: Bearer $TOKEN" --data-binary @"$f" \
      "$BUCKET/$rel")
    [ "$code" = "200" ] && break
    echo "     upload http=$code (attempt $attempt)"; sleep 5
  done
  if [ "$code" != "200" ]; then
    echo "[FATAL] upload failed for $rel"; cat /tmp/zenodo_up.json; exit 1
  fi
done

if [ "$PUBLISH" -eq 1 ]; then
  echo "[zenodo] PUBLISHING (record goes PUBLIC immediately)..."
  curl -sS -H "Authorization: Bearer $TOKEN" -X POST "$API/$DEP_ID/actions/publish" | redact >/dev/null
  echo "[zenodo] published. The reserved DOI now resolves."
else
  echo
  echo "[zenodo] draft ready, files uploaded, DOI reserved but NOT public."
  echo "         To publish on submission day (explicit, deliberate):"
  echo "           curl -H \"Authorization: Bearer \$ZENODO_ACCESS_TOKEN\" -X POST \\"
  echo "             \"$API/$DEP_ID/actions/publish\""
fi
