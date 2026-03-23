#!/usr/bin/env bash
set -euo pipefail
LOG="training_live.log"
OUT="monitor_status.txt"
VERDICT="final_auc_verdict.txt"

while true; do
  ts=$(date '+%Y-%m-%d %H:%M:%S')

  running=0
  if ps -ef | grep -E "python(3)? .*train.py|train.py" | grep -v grep >/dev/null; then
    running=1
  fi

  current_epoch=$(grep -E "^  Epoch [0-9]+/20" "$LOG" | tail -n 1 | sed -E 's/.*Epoch ([0-9]+)\/20.*/\1/' || true)
  last_mean=$(grep -E "Mean AUC:" "$LOG" | tail -n 1 | sed -E 's/.*Mean AUC: ([0-9.]+).*/\1/' || true)

  echo "[$ts] running=$running epoch=${current_epoch:-na} last_mean_auc=${last_mean:-na}" > "$OUT"

  if grep -q "Epoch 20/20" "$LOG" && grep -q "\[History\] Saved training history" "$LOG"; then
    python3 - <<'PY' > "$VERDICT"
import json
from statistics import mean
from pathlib import Path

p = Path('results/training_history.json')
if not p.exists():
    print('Could not find results/training_history.json')
    raise SystemExit(0)

h = json.loads(p.read_text())
means = h.get('mean_auc', [])
if not means:
    print('No mean_auc found in training history')
    raise SystemExit(0)

best = max(means)
best_epoch = means.index(best) + 1
last = means[-1]

stan5 = {'Atelectasis':0.858,'Cardiomegaly':0.832,'Consolidation':0.899,'Edema':0.924,'Pleural Effusion':0.968}
stan6 = {**stan5,'Pneumothorax':0.943}
stan5_mean = mean(stan5.values())
stan6_mean = mean(stan6.values())

print(f'Best mean AUC: {best:.4f} (history index epoch {best_epoch})')
print(f'Last mean AUC: {last:.4f}')
print(f'Stanford mean (5-label): {stan5_mean:.4f}')
print(f'Stanford mean (6-label incl Pneumothorax): {stan6_mean:.4f}')
print(f'Gap vs Stanford 5-label (best): {best - stan5_mean:+.4f}')
print(f'Gap vs Stanford 6-label (best): {best - stan6_mean:+.4f}')
print('Verdict: ABOVE Stanford mean' if best > stan5_mean else 'Verdict: BELOW Stanford mean')
PY
    echo "[$ts] epoch20 complete; verdict written to $VERDICT" >> "$OUT"
    exit 0
  fi

  if [[ "$running" -eq 0 ]]; then
    echo "[$ts] training process not running; monitor exiting" >> "$OUT"
    exit 0
  fi

  sleep 120

done
