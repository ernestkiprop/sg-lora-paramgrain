#!/bin/bash
# Top-up to n=8 seeds on the 3 most promising tasks (MRPC, CoLA, RTE).
# Adds 3 new seeds (65, 75, 85) for 2 ablations -> 18 additional runs.
# At n=8 the one-sided Wilcoxon floor is 1/2^8 = 0.004, well below
# Holm's alpha/3 = 0.017 first-rank threshold.
#
# Usage:
#   nohup bash scripts/run_topup.sh > topup.log 2>&1 &
#   tail -f topup.log

set -u

export CUDA_VISIBLE_DEVICES=0
export WANDB_CACHE_DIR=/tmp/wandb_cache
export HF_HOME=/tmp/hf_cache
export TOKENIZERS_PARALLELISM=false

cd "$(dirname "$0")/.."

mkdir -p logs
STAMP=$(date +%Y%m%d_%H%M%S)
PROGRESS=logs/topup_${STAMP}.log

NEW_SEEDS="65 75 85"
TASKS=(rte mrpc cola)
ABLATIONS=(salsvd random)

echo "=== paramgrain top-up started $(date) ===" | tee -a "$PROGRESS"
echo "    seeds: $NEW_SEEDS"                      | tee -a "$PROGRESS"
echo "    tasks: ${TASKS[*]}"                     | tee -a "$PROGRESS"
echo "    18 runs total (2 abl x 3 tasks x 3 seeds), ~1.5-2 hrs" | tee -a "$PROGRESS"
echo ""                                            | tee -a "$PROGRESS"

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
        --seeds $NEW_SEEDS \
        --resume

    status=$?
    if [ $status -eq 0 ]; then
      echo "[$i/$TOTAL]  $(date +%H:%M:%S)  task=$task  ablation=$abl  -- DONE" \
        | tee -a "$PROGRESS"
    else
      echo "[$i/$TOTAL]  $(date +%H:%M:%S)  task=$task  ablation=$abl  -- FAILED (code $status)" \
        | tee -a "$PROGRESS"
    fi
    echo "" | tee -a "$PROGRESS"
  done
done

echo "=== paramgrain top-up finished $(date) ===" | tee -a "$PROGRESS"
