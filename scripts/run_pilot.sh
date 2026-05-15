#!/bin/bash
# Parameter-granularity pilot: 2 ablations x 5 tasks x 5 seeds = 50 runs.
# Sequential, --resume safe. Expect ~6-8 hours wall-clock on one GPU.
#
# Usage:
#   bash scripts/run_pilot.sh
# Or to run detached so it survives terminal disconnect:
#   nohup bash scripts/run_pilot.sh > pilot.log 2>&1 &
#   tail -f pilot.log   # to watch progress

set -u

export CUDA_VISIBLE_DEVICES=0
export WANDB_CACHE_DIR=/tmp/wandb_cache
export HF_HOME=/tmp/hf_cache
export TOKENIZERS_PARALLELISM=false

cd "$(dirname "$0")/.."

mkdir -p logs
STAMP=$(date +%Y%m%d_%H%M%S)
PROGRESS=logs/progress_${STAMP}.log

echo "=== paramgrain pilot started $(date) ==="                  | tee -a "$PROGRESS"
echo "    50 runs total (2 ablations x 5 tasks x 5 seeds)"        | tee -a "$PROGRESS"
echo "    progress log: $PROGRESS"                                | tee -a "$PROGRESS"
echo ""                                                            | tee -a "$PROGRESS"

# Order: fast tasks first so any crash surfaces on cheap runs.
# Within each task, both ablations run before moving on -- this means
# salsvd-vs-random pairs are produced as we go, so partial analysis is
# possible even before the full pilot finishes.
TASKS=(rte mrpc cola stsb sst2)
ABLATIONS=(salsvd random)

i=0
TOTAL=$(( ${#TASKS[@]} * ${#ABLATIONS[@]} ))

for task in "${TASKS[@]}"; do
  for abl in "${ABLATIONS[@]}"; do
    i=$((i+1))
    echo "[$i/$TOTAL]  $(date +%H:%M:%S)  task=$task  ablation=$abl  -- starting" \
      | tee -a "$PROGRESS"

    python scripts/paramgrain_lora.py \
        --task "$task" \
        --ablation "$abl" \
        --resume

    status=$?
    if [ $status -eq 0 ]; then
      echo "[$i/$TOTAL]  $(date +%H:%M:%S)  task=$task  ablation=$abl  -- DONE" \
        | tee -a "$PROGRESS"
    else
      echo "[$i/$TOTAL]  $(date +%H:%M:%S)  task=$task  ablation=$abl  -- FAILED (code $status)" \
        | tee -a "$PROGRESS"
      # do not abort; --resume will pick up next time
    fi
    echo ""                                                        | tee -a "$PROGRESS"
  done
done

echo "=== paramgrain pilot finished $(date) ===" | tee -a "$PROGRESS"
