#!/usr/bin/env bash
# Container-side lane wrapper. Usage: lane.sh <log-name> <command...>
# Requires LANE_GPU_UUID, LANE_CPUS, LANE_PORT_BASE from the launcher. Binds the
# lane GPU by UUID, pins to the lane CPUs, and journals stdout/stderr per run.
set -euo pipefail
: "${LANE_GPU_UUID:?}" "${LANE_CPUS:?}" "${LANE_PORT_BASE:?}"
case "$LANE_GPU_UUID" in GPU-76fd1c0c*) echo "refusing GPU 0 (other session)" >&2; exit 5;; esac
case ",$LANE_CPUS," in *,0-*|*,0,*|*,1-*|*,2-*|*,3-*|*,4-*|*,5-*|*,6-*|*,7-*|*,8-*|*,9-*|*,1[0-9]-*|*,2[0-9]-*|*,3[01]-*)
    echo "refusing CPUs inside 0-31 (other session)" >&2; exit 5;; esac
export CUDA_VISIBLE_DEVICES="$LANE_GPU_UUID"
# One live lane per GPU node-wide (/tmp shared, no PID namespace): refuse a second launch on the same GPU.
exec 9>"/tmp/evid-lane-$LANE_GPU_UUID.lock"
flock -n 9 || { echo "lane $LANE_GPU_UUID already has a live step; refusing" >&2; exit 6; }
BASE=/work/campaigns/mps-evidence-20260912-a01
EVID=$BASE/extras/evid-v1
export PYTHONPATH="$EVID:$BASE/sources/evidence-9b148e05f:/work/campaigns/mps-host-gpu-green-context-20260912-a01/deps"
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
logdir=/work/results/mps-evidence-20260912-a01/job-18012/lane-logs
mkdir -p "$logdir"
name=$1; shift
log="$logdir/$name-$(date -u +%Y%m%dT%H%M%SZ).log"
{
  echo "lane gpu=$LANE_GPU_UUID cpus=$LANE_CPUS port_base=$LANE_PORT_BASE step=${SLURM_STEP_ID:-?} host=$(hostname) start=$(date -u +%FT%TZ)"
  echo "affinity: $(taskset -pc $$ | cut -d: -f2)"
  echo "cmd: $*"
} > "$log"
cd "$EVID"
exec taskset -c "$LANE_CPUS" "$@" >> "$log" 2>&1
