#!/bin/bash
# RoBERTa-large validation of paramgrain pilot.
#
# Tasks: CoLA + STS-B (the two tasks with the strongest n=10 base-model signal:
#        CoLA Holm-significant, STS-B 9/10 paired wins).
# Ablations: salsvd, random  -- matched LoRA-parameter budget, greedy alloc.
# Seeds: 15,25,35,45,55,65,75,85,95,105  (same as base; pairs by seed).
# Total runs: 2 tasks x 2 ablations x 10 seeds = 40
#
# Budget = 1,572,864 = 24 layers * 4 attn matrices * 1024 hidden * 2 (B+A) * 8 rank
#                    = LoRA-AllAttn r=8 on roberta-large
#          (this matches the parent paper's "fair" capacity budget at the new scale)
#
# Batch size = 16 to fit roberta-large in ~16-20 GB.
# Expect ~12-16 hours total wall-clock on a single 24 GB GPU.
#
# Usage:
#   nohup bash scripts/run_large_validation.sh > large_val.log 2>&1 &
#   tail -f large_val.log
#
# Resume-safe: --resume checks W&B for already-finished (seed, budget) pairs.

set -u

export CUDA_VISIBLE_DEVICES=0
export WANDB_CACHE_DIR=/tmp/wandb_cache
export HF_HOME=/tmp/hf_cache
export TOKENIZERS_PARALLELISM=false

cd "$(dirname "$0")/.."

mkdir -p logs
STAMP=$(date +%Y%m%d_%H%M%S)
PROGRESS=logs/large_val_${STAMP}.log

# CoLA first (faster + primary). STSB second.
TASKS=(cola stsb)
ABLATIONS=(salsvd random)
SEEDS="15 25 35 45 55 65 75 85 95 105"
MODEL="roberta-large"
BUDGET=1572864
BATCH=16

echo "=== paramgrain RoBERTa-large validation started $(date) ===" | tee -a "$PROGRESS"
echo "    tasks:      ${TASKS[*]}"                                  | tee -a "$PROGRESS"
echo "    ablations:  ${ABLATIONS[*]}"                              | tee -a "$PROGRESS"
echo "    seeds:      $SEEDS"                                        | tee -a "$PROGRESS"
echo "    model:      $MODEL"                                        | tee -a "$PROGRESS"
echo "    budget:     $BUDGET (AllAttn r=8 on $MODEL)"              | tee -a "$PROGRESS"
echo "    batch:      $BATCH"                                        | tee -a "$PROGRESS"
echo "    40 runs total, ~12-16 hrs"                                | tee -a "$PROGRESS"
echo ""                                                              | tee -a "$PROGRESS"

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
        --model "$MODEL" \
        --budget "$BUDGET" \
        --batch-size "$BATCH" \
        --seeds $SEEDS \
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

echo "=== paramgrain RoBERTa-large validation finished $(date) ===" | tee -a "$PROGRESS"
