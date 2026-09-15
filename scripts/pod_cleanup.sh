#!/usr/bin/env bash
# pod_cleanup.sh — terminate the pod and delete any orphaned network volume.
# Run ONLY after outputs are verified on HF (pod_fetch_campaign.sh --dry-run
# or a completed fetch). Idempotent and safe to run late.
# Usage: bash scripts/pod_cleanup.sh <pod_name>
set -euo pipefail
POD_NAME="${1:?Usage: bash scripts/pod_cleanup.sh <pod_name>}"
KEY="${RUNPOD_API_KEY:?Set RUNPOD_API_KEY}"
GQL() { curl -s "https://api.runpod.io/graphql?api_key=${KEY}" -H "Content-Type: application/json" \
  -d "{\"query\": \"$1\"}"; }

POD_ID=$(GQL "{ myself { pods { id name } } }" \
  | python3 -c "import sys,json; ps=[p for p in json.load(sys.stdin)['data']['myself']['pods'] if p['name']=='$POD_NAME']; print(ps[0]['id'] if ps else '')")
if [ -n "$POD_ID" ]; then
  echo "[CLEANUP] terminating pod $POD_NAME ($POD_ID)"
  GQL "mutation { podTerminate(input: {podId: \\\"$POD_ID\\\"}) }"
  echo
else
  echo "[CLEANUP] pod '$POD_NAME' not found (already terminated?)"
fi

echo "[CLEANUP] orphaned network volumes (delete any left from this project):"
GQL "{ myself { networkVolumes { id name } } }"
echo
echo "[CLEANUP] remaining pods:"
GQL "{ myself { pods { id name desiredStatus } } }"
