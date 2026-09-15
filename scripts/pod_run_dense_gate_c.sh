#!/usr/bin/env bash
# pod_run_dense_gate_c.sh — deploy + launch Gate C Dense (BGE-M3) Recall eval on a RunPod pod.
# Usage:
#   From host:   bash scripts/pod_run_dense_gate_c.sh <pod_name>
#   On pod:      bash scripts/pod_run_dense_gate_c.sh (inside /workspace/rag_eval)
#
# Endpoint resolve -> ssh wait -> token pipe -> pip (marker, aliyun mirror) ->
# verify GPU/imports -> nohup launch with sentinels (LAUNCHED-VERIFIED / [DONE]).
set -euo pipefail

# -----------------------------------------------------------------------------
# Direct on-pod execution branch (when run inside /workspace/rag_eval without args)
# -----------------------------------------------------------------------------
if [ -d "/workspace/rag_eval" ] && [ $# -eq 0 ]; then
  cd /workspace/rag_eval
  mkdir -p /workspace/logs data/eval
  echo "[start] dense_gate_c $(date -u +%H:%M:%S)" >> /workspace/logs/status.log
  set -a; [ -f /root/.hfenv ] && . /root/.hfenv 2>/dev/null; set +a

  if ! test -f /workspace/.pip_dense_done && ! test -f /workspace/.pip_done; then
    echo "[pip] installing dependencies via aliyun mirror..."
    pip install --break-system-packages -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com \
      "numpy<2" sentence-transformers rank_bm25 pyvi requests "huggingface_hub[cli]" > /workspace/pip_dense.log 2>&1
    touch /workspace/.pip_dense_done
  fi

  DATE_TAG=$(date -u +%Y-%m-%d)
  OUT_JSON="data/eval/dense_recall_gate_c_${DATE_TAG}.json"

  echo "[run] executing dense recall eval -> $OUT_JSON"
  rc=0
  python3 -u scripts/eval_bm25_recall.py --retriever dense --json "$OUT_JSON" > /workspace/logs/dense_gate_c.log 2>&1 || rc=$?
  if [ $rc -ne 0 ] || [ ! -s "$OUT_JSON" ]; then
    echo "[FAILED] dense_gate_c rc=$rc $(date -u +%H:%M:%S)" >> /workspace/logs/status.log
    tail -10 /workspace/logs/dense_gate_c.log >> /workspace/logs/status.log 2>/dev/null || true
    exit "$rc"
  fi

  HF_REPO="${HF_ARTIFACTS_REPO:-hunopapa/regrag-artifacts}"
  if [ -n "${HF_TOKEN:-}" ] && command -v hf >/dev/null 2>&1; then
    echo "[push] uploading $OUT_JSON to $HF_REPO (gate-c/)"
    hf upload "$HF_REPO" "$OUT_JSON" "gate-c/dense_recall_gate_c_${DATE_TAG}.json" --type dataset \
      --commit-message "gate-c dense recall eval ${DATE_TAG}" >> /workspace/logs/dense_gate_c.log 2>&1
  fi

  echo "[done] dense_gate_c rc=0 $(date -u +%H:%M:%S)" >> /workspace/logs/status.log
  echo "[DONE] dense_gate_c complete: $OUT_JSON"
  exit 0
fi

# -----------------------------------------------------------------------------
# Host orchestration branch (dispatches to remote RunPod instance via SSH)
# -----------------------------------------------------------------------------
POD_NAME="${1:?Usage: bash scripts/pod_run_dense_gate_c.sh <pod_name>}"
KEY="${RUNPOD_API_KEY:?Set RUNPOD_API_KEY}"
HF_REPO="${HF_ARTIFACTS_REPO:-hunopapa/regrag-artifacts}"

SSH() { ssh -p "$PORT" -i ~/.ssh/id_ed25519 -o StrictHostKeyChecking=accept-new \
  -o UserKnownHostsFile=/dev/null -o ServerAliveInterval=15 -o ServerAliveCountMax=4 \
  -o TCPKeepAlive=yes -o LogLevel=ERROR root@"$IP" "$@"; }
GQL() { curl -s "https://api.runpod.io/graphql?api_key=${KEY}" -H "Content-Type: application/json" \
  -d "{\"query\": \"$1\"}"; }

# --- local HF token (file -> CLI -> venv), piped to the pod as /root/.hfenv
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

# [4] pip in background (marker pattern with aliyun mirror fallback). Skip when
#     a marker exists OR a pip is already mid-flight (re-run safety: never two
#     pips racing; campaign lesson 2026-09-13).
SSH "if test -f /workspace/.pip_dense_done || test -f /workspace/.pip_done || pgrep -f 'pip instal[l]' >/dev/null; then echo 'pip done/running - skip'; else nohup bash -c 'pip install --break-system-packages -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com \"numpy<2\" sentence-transformers rank_bm25 pyvi requests \"huggingface_hub[cli]\" > /workspace/pip_dense.log 2>&1 && touch /workspace/.pip_dense_done' >/dev/null 2>&1 & echo 'pip launched'; fi" </dev/null

# [4b] code + inputs pull — a fresh pod has no /workspace/rag_eval; stage the
#      CURRENT repo state (fixed corpus + dense arm) from the HF artifacts
#      repo, exactly like pod_run_campaign.sh [5]. Waits for pip FIRST: the
#      pull needs the hf CLI that pip installs (first deploy failed with
#      'hf: command not found' because the pull raced the background pip).
echo "[4b] waiting for pip, then pulling code + inputs from $HF_REPO..."
SSH "for i in \$(seq 1 120); do (test -f /workspace/.pip_dense_done || test -f /workspace/.pip_done) && break; sleep 5; done; (test -f /workspace/.pip_dense_done || test -f /workspace/.pip_done) || { echo '[FATAL] pip did not finish'; tail -5 /workspace/pip_dense.log 2>/dev/null; exit 1; }; command -v hf >/dev/null || { echo '[FATAL] hf missing after pip'; tail -20 /workspace/pip_dense.log 2>/dev/null; exit 1; }; set -a; . /root/.hfenv 2>/dev/null; set +a; [ -n \"\${HF_TOKEN:-}\" ] || { echo 'no HF_TOKEN'; exit 1; }; rm -rf /ws_code; hf download '$HF_REPO' --repo-type dataset --include 'regrag/code/*' --include 'regrag/inputs/*' --local-dir /ws_code && mkdir -p /workspace/rag_eval && cp -r /ws_code/regrag/code/regrag /ws_code/regrag/code/scripts /workspace/rag_eval/ && mkdir -p /workspace/rag_eval/data/gold /workspace/rag_eval/data/processed_chunks && cp /ws_code/regrag/inputs/bank_qa_data.csv /workspace/rag_eval/data/gold/ && cp /ws_code/regrag/inputs/processed_chunks/*.json /workspace/rag_eval/data/processed_chunks/ && find /workspace/rag_eval -name '*.py' | wc -l"

# [5] sync eval script to the pod (assumes repo structure in /workspace/rag_eval)
echo "[5] syncing eval script to pod..."
SSH "mkdir -p /workspace/rag_eval/scripts /workspace/rag_eval/data/eval /workspace/logs"
cat scripts/eval_bm25_recall.py | SSH "cat > /workspace/rag_eval/scripts/eval_bm25_recall.py"

# [5b] canonical gold (own ssh call, md5 both sides — truncation trap)
echo "[5b] syncing canonical gold..."
md5sum data/gold/questions_canonical.json
cat data/gold/questions_canonical.json | SSH "cat > /workspace/rag_eval/data/gold/questions_canonical.json && md5sum /workspace/rag_eval/data/gold/questions_canonical.json"

# [6] wait for pip, then verify imports + GPU + staged corpus
echo "[6] waiting for pip..."
SSH "for i in \$(seq 1 120); do (test -f /workspace/.pip_dense_done || test -f /workspace/.pip_done) && break; sleep 5; done; (test -f /workspace/.pip_dense_done || test -f /workspace/.pip_done) || { echo '[FATAL] pip did not finish'; tail -5 /workspace/pip_dense.log 2>/dev/null; exit 1; }; echo pip-done"
SSH "python3 -c 'import torch; print(torch.__version__, torch.cuda.get_device_name(0), torch.cuda.get_device_properties(0).total_memory//2**30, \"GiB\")'"
SSH "cd /workspace/rag_eval && python3 -c 'from regrag.indexing.dense import DenseIndex; print(\"dense import OK\")'"
SSH "python3 -c \"import json; ch=json.load(open('/workspace/rag_eval/data/processed_chunks/corpus_chunks.json')); ids=[c['chunk_id'] for c in ch]; assert len(ids)==len(set(ids)), 'DUPLICATE CHUNK IDS STAGED'; print('staged corpus:', len(ch), 'chunks, ids unique')\""

# [7] runner with sentinels + empty-log guards, launched under nohup.
#     mkdir BEFORE redirecting: a missing logs dir kills stdout redirect before bash execs.
SSH "mkdir -p /workspace/logs && cat > /workspace/rag_eval/runner_dense_gate_c.sh" <<'RUNNER'
#!/usr/bin/env bash
set -uo pipefail
cd /workspace/rag_eval
mkdir -p /workspace/logs data/eval
echo "[start] dense_gate_c $(date -u +%H:%M:%S)" >> /workspace/logs/status.log
set -a; . /root/.hfenv 2>/dev/null; set +a
[ -n "${HF_TOKEN:-}" ] || { echo "[FAILED] no HF_TOKEN in runner" >> /workspace/logs/status.log; exit 1; }
HF_REPO="${HF_ARTIFACTS_REPO:-hunopapa/regrag-artifacts}"

DATE_TAG=$(date -u +%Y-%m-%d)
OUT_JSON="data/eval/dense_recall_gate_c_${DATE_TAG}.json"

echo "[run] executing dense recall eval -> $OUT_JSON"
python3 -u scripts/eval_bm25_recall.py --retriever dense --json "$OUT_JSON" > /workspace/logs/dense_gate_c.log 2>&1
rc=$?
if [ $rc -ne 0 ] || [ ! -s "$OUT_JSON" ]; then
  echo "[FAILED] dense_gate_c rc=$rc $(date -u +%H:%M:%S)" >> /workspace/logs/status.log
  tail -10 /workspace/logs/dense_gate_c.log >> /workspace/logs/status.log 2>/dev/null || true
  exit $rc
fi

# Push artifact to HF repository under gate-c path
echo "[push] uploading $OUT_JSON to $HF_REPO (gate-c/)"
command -v hf >/dev/null || { echo "[FAILED] hf CLI not found" >> /workspace/logs/status.log; exit 1; }
hf upload "$HF_REPO" "$OUT_JSON" "gate-c/dense_recall_gate_c_${DATE_TAG}.json" --type dataset \
  --commit-message "gate-c dense recall eval ${DATE_TAG}" >> /workspace/logs/dense_gate_c.log 2>&1
push_rc=$?
if [ $push_rc -ne 0 ]; then
  echo "[FAILED] hf push rc=$push_rc $(date -u +%H:%M:%S)" >> /workspace/logs/status.log
  exit $push_rc
fi

echo "[done] dense_gate_c rc=0 $(date -u +%H:%M:%S)" >> /workspace/logs/status.log
echo "[DONE] dense_gate_c complete: $OUT_JSON uploaded to $HF_REPO"
RUNNER

SSH "chmod +x /workspace/rag_eval/runner_dense_gate_c.sh && cd /workspace/rag_eval && nohup bash runner_dense_gate_c.sh > /workspace/logs/driver_dense.log 2>&1 </dev/null & sleep 8; if grep -q '^\[start\] dense_gate_c' /workspace/logs/status.log 2>/dev/null; then echo LAUNCHED-VERIFIED; nohup tail -f /workspace/logs/dense_gate_c.log > /proc/1/fd/1 2>/dev/null & else echo LAUNCH-FAILED; tail -3 /workspace/logs/driver_dense.log 2>/dev/null; exit 95; fi"

echo
echo "[OK] dense gate C eval launched. Monitor with:"
echo "  ssh -p $PORT -i ~/.ssh/id_ed25519 root@$IP 'tail -3 /workspace/logs/status.log; tail -10 /workspace/logs/dense_gate_c.log'"
echo "HF artifacts destination: $HF_REPO (gate-c/dense_recall_gate_c_*.json)"
