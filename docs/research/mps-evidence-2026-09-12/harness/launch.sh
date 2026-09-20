#!/usr/bin/env bash
# Login-node lane launcher. Usage: launch.sh <A|B|C> <name> <cmd...>
# Starts one overlap step on job 18012 bound to the lane GPU/CPUs/ports and runs
# <cmd...> inside the container through lane.sh (which holds the per-GPU lock).
set -euo pipefail
MY=/mnt/nfs/sa-shared/wenyao-minimax-h3/work/campaigns/mps-evidence-20260912-a01
lane=$1; name=$2; shift 2
case $lane in
  A) GPU=GPU-e7fb109f-a132-36c9-55b4-489eec809700; CPUS=56-95; PORT=19000;;
  B) GPU=GPU-a7730a60-aa85-6f5b-72fd-905679046585; CPUS=32-55; PORT=19100;;
  C) GPU=GPU-8fd719df-ee0b-bda1-f591-b57193bce909; CPUS=96-111; PORT=19200;;
  *) echo "lane must be A|B|C" >&2; exit 2;;
esac
out=$MY/launch-logs/$lane-$name.out
[[ -e $out ]] && { echo "launch log exists: $out" >&2; exit 3; }
mkdir -p "$MY/launch-logs"
nohup setsid env -u SLURM_GPUS_PER_NODE -u SLURM_TRES_PER_TASK \
  LANE_GPU_UUID=$GPU LANE_CPUS=$CPUS LANE_PORT_BASE=$PORT \
  srun --jobid=18012 --nodelist=hpc-gpu-1-1 --overlap --nodes=1 --ntasks=1 --gpus-per-task=8 --immediate=60 \
  --job-name=evid-$lane-$name bash "$MY/control/entry-overlap.sh" \
  bash /work/campaigns/mps-evidence-20260912-a01/extras/evid-v1/lane.sh "$lane-$name" "$@" > "$out" 2>&1 &
echo "launched $lane-$name pid $! log $out"
