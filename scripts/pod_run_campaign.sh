#!/usr/bin/env bash
# pod_run_campaign.sh — deploy + launch the RegRAG-VN campaign on a RunPod pod.
# Usage: bash scripts/pod_run_campaign.sh <pod_name>   (invoke BARE, no | tail)
#
# Endpoint resolve -> ssh wait -> token pipe -> pip (bg, marker) ∥ code pull ->
# import-chain + GPU verify -> --plan CPU gate -> nohup launch with sentinels.
set -euo pipefail
POD_NAME="${1:?Usage: bash scripts/pod_run_campaign.sh <pod_name>}"
KEY="${RUNPOD_API_KEY:?Set RUNPOD_API_KEY}"
HF_REPO="${HF_ARTIFACTS_REPO:-hunopapa/regrag-artifacts}"

SSH() { ssh -p "$PORT" -i ~/.ssh/id_ed25519 -o StrictHostKeyChecking=accept-new \
  -o UserKnownHostsFile=/dev/null -o ServerAliveInterval=15 -o ServerAliveCountMax=4 \
  -o TCPKeepAlive=yes -o LogLevel=ERROR root@"$IP" "$@"; }
GQL() { curl -s "https://api.runpod.io/graphql?api_key=${KEY}" -H "Content-Type: application/json" \
  -d "{\"query\": \"$1\"}"; }

# --- local HF token (file -> CLI -> venv), piped to the pod as /root/.hfenv
#     WITH export lines (rescop format; source with set -a everywhere)
HF_TOKEN="$(cat ~/.cache/huggingface/token 2>/dev/null || hf auth token 2>/dev/null \
  || .venv/bin/python -c 'from huggingface_hub import get_token; print(get_token())' 2>/dev/null || true)"
[ -n "$HF_TOKEN" ] || { echo "[FATAL] no local HF token"; exit 1; }

# [1] endpoint — filter privatePort==22 (port-list decoy)
POD_JSON=$(GQL "{ myself { pods { name desiredStatus runtime { ports { ip privatePort publicPort type } } } } }")
IP=$(echo "$POD_JSON" | python3 -c "import sys,json; ps=[p for p in json.load(sys.stdin)['data']['myself']['pods'] if p['name']=='$POD_NAME' and p.get('runtime')]; print(next((pt['ip'] for pt in ps[0]['runtime']['ports'] if pt['privatePort']==22),'')) if ps else print('')")
PORT=$(echo "$POD_JSON" | python3 -c "import sys,json; ps=[p for p in json.load(sys.stdin)['data']['myself']['pods'] if p['name']=='$POD_NAME' and p.get('runtime')]; print(next((pt['publicPort'] for pt in ps[0]['runtime']['ports'] if pt['privatePort']==22),'')) if ps else print('')")
[ -n "$IP" ] && [ -n "$PORT" ] || { echo "[FATAL] no SSH endpoint for pod '$POD_NAME' (booting? runtime null?) — retry in a minute"; exit 1; }
echo "[1] endpoint $IP:$PORT"

# [2] wait for sshd (fresh secure pods can take 10-20 min)
for i in $(seq 1 90); do SSH "echo SSH-up" >/dev/null 2>&1 && break; echo "ssh retry $i/90"; sleep 10; done
SSH "echo SSH-up" >/dev/null

# [3] token pipe — OWN ssh call, nothing else in it (truncation trap)
printf 'export HF_TOKEN=%s\nexport HF_ARTIFACTS_REPO=%s\n' "$HF_TOKEN" "$HF_REPO" \
  | SSH "cat > /root/.hfenv && chmod 600 /root/.hfenv && wc -c /root/.hfenv"

# [4] pip in background (marker pattern). Decide on OUR marker only: the
#     pytorch template runs its own boot-time pip, and pgrep -f 'pip instal[l]'
#     matched it (2026-09-13, zesty_coffee_herring) - the skip branch fired
#     spuriously and hf was missing at the code-pull step.
SSH "if test -f /workspace/.pip_done; then echo 'pip done - skip'; else nohup bash -c 'pip install --break-system-packages \"numpy<2\" transformers accelerate bitsandbytes sentence-transformers rank_bm25 pyvi requests \"huggingface_hub[cli]\" > /workspace/pip.log 2>&1 && touch /workspace/.pip_done' >/dev/null 2>&1 & echo 'pip launched'; fi" </dev/null

# [5] code + inputs pull (per-block auth: source .hfenv with set -a)
SSH "set -a; . /root/.hfenv 2>/dev/null; set +a; [ -n \"\${HF_TOKEN:-}\" ] || { echo 'no HF_TOKEN'; exit 1; }; rm -rf /ws_code; hf download '$HF_REPO' --repo-type dataset --include 'regrag/code/*' --include 'regrag/inputs/*' --local-dir /ws_code && mkdir -p /workspace/rag_eval && cp -r /ws_code/regrag/code/regrag /ws_code/regrag/code/scripts /workspace/rag_eval/ && mkdir -p /workspace/rag_eval/data/gold /workspace/rag_eval/data/processed_chunks && cp /ws_code/regrag/inputs/bank_qa_data.csv /workspace/rag_eval/data/gold/ && cp /ws_code/regrag/inputs/processed_chunks/*.json /workspace/rag_eval/data/processed_chunks/ && find /workspace/rag_eval -name '*.py' | wc -l"

# [6] wait for pip, then verify imports + GPU
echo "[6] waiting for pip..."
SSH "for i in \$(seq 1 120); do test -f /workspace/.pip_done && break; sleep 10; done; test -f /workspace/.pip_done || { echo '[FATAL] pip did not finish'; tail -5 /workspace/pip.log 2>/dev/null; exit 1; }; echo pip-done"
SSH "cd /workspace/rag_eval && python3 -c 'from regrag.generation.campaign import CampaignRunner; from regrag.generation.backends import resolve_backend; from scripts.run_campaign_pod import main; from scripts.preflight_memory import preflight_memory; print(\"import-chain OK\")'"
SSH "python3 -c 'import torch; print(torch.__version__, torch.cuda.get_device_name(0), torch.cuda.get_device_properties(0).total_memory//2**30, \"GiB\")'"

# [7] CPU plan gate on REAL data (fatal) — catches staging/data bugs pre-GPU
SSH "cd /workspace/rag_eval && set -o pipefail && python3 -u scripts/run_campaign_pod.py --plan 2>&1 | tail -6"

# [8] runner with sentinels + empty-log guards, launched under nohup.
#     mkdir BEFORE the redirect: a missing logs dir kills the runner's
#     stdout redirect before bash even execs it (2026-09-13, phantom
#     LAUNCHED on zesty_coffee_herring - the runner never started).
SSH "mkdir -p /workspace/logs && cat > /workspace/rag_eval/runner_campaign.sh" <<'RUNNER'
#!/usr/bin/env bash
set -uo pipefail
cd /workspace/rag_eval
mkdir -p /workspace/logs
echo "[start] campaign $(date -u +%H:%M:%S)" >> /workspace/logs/status.log
set -a; . /root/.hfenv 2>/dev/null; set +a
[ -n "${HF_TOKEN:-}" ] || { echo "[FAILED] no HF_TOKEN in runner" >> /workspace/logs/status.log; exit 1; }
python3 -u scripts/run_campaign_pod.py > /workspace/logs/campaign.log 2>&1
rc=$?
if [ $rc -ne 0 ] || [ ! -s /workspace/logs/campaign.log ]; then
  echo "[FAILED] campaign rc=$rc $(date -u +%H:%M:%S)" >> /workspace/logs/status.log
  tail -5 /workspace/logs/campaign.log >> /workspace/logs/status.log
  exit $rc
fi
echo "[done] campaign rc=0 $(date -u +%H:%M:%S)" >> /workspace/logs/status.log
RUNNER
SSH "chmod +x /workspace/rag_eval/runner_campaign.sh && cd /workspace/rag_eval && nohup bash runner_campaign.sh > /workspace/logs/driver.log 2>&1 </dev/null & sleep 8; if grep -q '^\[start\] campaign' /workspace/logs/status.log 2>/dev/null; then echo LAUNCHED-VERIFIED; nohup tail -f /workspace/logs/campaign.log > /proc/1/fd/1 2>/dev/null & else echo LAUNCH-FAILED; tail -3 /workspace/logs/driver.log 2>/dev/null; exit 95; fi"

echo
echo "[OK] campaign launched. Monitor with:"
echo "  ssh -p $PORT -i ~/.ssh/id_ed25519 root@$IP 'tail -3 /workspace/logs/status.log; tail -2 /workspace/logs/campaign.log'"
echo "Fetch after done:  bash scripts/pod_fetch_campaign.sh <run_id>"
echo "Then cleanup:      bash scripts/pod_cleanup.sh $POD_NAME"
