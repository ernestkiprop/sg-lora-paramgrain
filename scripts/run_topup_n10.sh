#!/bin/bash
# Second top-up: n=8 -> n=10 across ALL 5 tasks.
# Adds 2 new seeds (95, 105) for 2 ablations -> 20 additional runs.
#
# At n=10 the one-sided Wilcoxon floor is 1/2^10 = 0.001, well below
# Holm's alpha/5 = 0.01 first-rank threshold. With 5/5 already directional
# and CoLA + STS-B already trending Holm-significant at n=8, this pushes
# borderline tasks (STS-B, RTE) over the threshold if the signal is real.
#
# Usage:
#   nohup bash scripts/run_topup_n10.sh > topup_n10.log 2>&1 &
#   tail -f topup_n10.log

set -u

export CUDA_VISIBLE_DEVICES=0
export WANDB_CACHE_DIR=/tmp/wandb_cache
export HF_HOME=/tmp/hf_cache
export TOKENIZERS_PARALLELISM=false

cd "$(dirname "$0")/.."

mkdir -p logs
STAMP=$(date +%Y%m%d_%H%M%S)
PROGRESS=logs/topup_n10_${STAMP}.log

NEW_SEEDS="95 105"
# Fast tasks first; SST-2 (slow) last.
TASKS=(rte mrpc stsb cola sst2)
ABLATIONS=(salsvd random)

echo "=== paramgrain n=10 top-up started $(date) ===" | tee -a "$PROGRESS"
echo "    seeds: $NEW_SEEDS"                          | tee -a "$PROGRESS"
echo "    tasks: ${TASKS[*]}"                         | tee -a "$PROGRESS"
echo "    20 runs total (2 abl x 5 tasks x 2 seeds), ~3-4 hrs" | tee -a "$PROGRESS"
echo ""                                                | tee -a "$PROGRESS"

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

echo "=== paramgrain n=10 top-up finished $(date) ===" | tee -a "$PROGRESS"
