#!/usr/bin/env bash
# Sequential lane queue. Usage (inside lane.sh): evid_queue.sh <queue-name>
# Reads /work/campaigns/mps-evidence-20260912-a01/queues/<queue-name>.txt line by
# line (re-read each iteration, so lines may be appended while running). Each
# non-empty, non-# line is a python3 argument list run from the harness dir.
# Failures are journaled and the queue continues with the next line; a line
# starting with STOP ends the queue.
set -uo pipefail
name=$1
QUEUE=/work/campaigns/mps-evidence-20260912-a01/queues/$name.txt
JOURNAL=/work/results/mps-evidence-20260912-a01/job-18012/queues/$name.jsonl
mkdir -p "$(dirname "$JOURNAL")"
[[ -f "$QUEUE" ]] || { echo "no queue file $QUEUE" >&2; exit 2; }
i=0
while :; do
    i=$((i+1))
    line=$(sed -n "${i}p" "$QUEUE" || true)
    total=$(wc -l < "$QUEUE")
    if (( i > total )); then
        echo "queue $name drained after $((i-1)) lines $(date -u +%FT%TZ)"
        break
    fi
    [[ -z "${line// }" || "$line" == \#* ]] && continue
    [[ "$line" == STOP* ]] && { echo "queue $name STOP at line $i"; break; }
    start=$(date -u +%FT%TZ)
    echo "== queue $name line $i start $start: python3 $line"
    # shellcheck disable=SC2086
    python3 $line
    rc=$?
    end=$(date -u +%FT%TZ)
    echo "== queue $name line $i exit $rc end $end"
    printf '{"queue":"%s","line":%d,"command":%s,"exit":%d,"start":"%s","end":"%s"}\n' \
        "$name" "$i" "$(python3 -c 'import json,sys; print(json.dumps(sys.argv[1]))' "$line")" "$rc" "$start" "$end" >> "$JOURNAL"
    # Never let a failed cell's leftovers poison the next boot on this lane.
    left=$(nvidia-smi --id="$CUDA_VISIBLE_DEVICES" --query-compute-apps=pid,used_memory --format=csv,noheader || true)
    if [[ -n "$left" ]]; then
        echo "GPU processes remain after line $i: $left; waiting up to 120s"
        for _ in $(seq 24); do
            sleep 5
            left=$(nvidia-smi --id="$CUDA_VISIBLE_DEVICES" --query-compute-apps=pid,used_memory --format=csv,noheader || true)
            [[ -z "$left" ]] && break
        done
        [[ -n "$left" ]] && { echo "GPU still busy; stopping queue $name to avoid contaminating later cells: $left"; break; }
    fi
    sleep 10
done
