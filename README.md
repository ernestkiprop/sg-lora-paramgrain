# SG-LoRA Parameter-Granularity Follow-up

Per-parameter saliency, aggregated by SVD into per-module LoRA rank
allocations. Final follow-up to the paper **When Does Saliency Help
LoRA? A Rigorous Study of One-Shot Gradient Criteria for Adapter
Placement**.

## What this tests

The parent paper's no-go theorem (gradient saliency = capacity controller)
held at:

- **Layer granularity** with gradient signals (SNIP / GN / Fisher) — ties Random.
- **Sub-module granularity** with gradient signals — small RTE-only effect dominated by module-type bias.
- **Layer granularity** with information-theoretic signals (linear probing) — trails Random in every cell.

The only granularity-by-signal-class combination not yet tested is
**parameter granularity** — per-weight-entry saliency, decomposed by SVD
into per-rank-direction value curves, and greedily allocated across
modules. This is the natural matching of saliency granularity to LoRA's
actual unit of operation (rank-r subspaces of each weight matrix).

## Method

For each linear module `m` of `roberta-base`:

1. Compute saliency matrix `S_m = |grad · weight|` on the same 5% sample
   used throughout the parent paper.
2. SVD: `S_m = U_m Σ_m V_m^T`. The singular values `σ_{m,1} ≥ σ_{m,2} ≥ ...`
   are the per-rank value curve.
3. LoRA cost of adding rank +1 to module `m`: `d_out + d_in`.
4. **Greedy allocation:** at each step, add +1 rank to the (module, next-rank)
   maximizing `σ_{m,r+1} / (d_out + d_in)`. Stop when total LoRA parameter
   budget is hit.

Result: a per-module integer rank vector `(r_1, ..., r_72)` summing (in
parameters) to ~600k (matching LoRA-AllAttn r=8, the parent paper's
recommended baseline).

## Ablations

- `salsvd` — saliency-SVD greedy allocation
- `random` — uniform random scores at each rank, same greedy budget-respecting routine

Same module pool, same allocation algorithm, same total budget — only
the per-rank scoring differs. This isolates the contribution of
saliency from the contribution of the allocator.

## Decision rule

The parent paper's five-criterion checklist:

1. Beats matched-budget **Random** at every comparison where it claims to win.
2. Beats **LoRA-AllAttn** at matched parameter count on ≥3 of 5 tasks.
3. Survives **Holm–Bonferroni** with ≥5 seeds and Cohen's `d_z`.
4. Reports **best-budget** in addition to a single budget cell.
5. Validates on **at least one larger model** (RoBERTa-large minimum).

Outcomes:

- If salsvd beats random on ≥2 tasks → first one-shot signal escaping the
  no-go; promote to short companion paper.
- If salsvd ties or loses to random everywhere → no-go closes for all
  one-shot signals at all granularities.

## Pipeline

```
freeze roberta-base
  -> forward+backward over 5% of train (one (task, seed) pass)
  -> per-Linear |grad · weight| accumulated, then SVD'd
  -> greedy rank allocation across 72 modules within budget
  -> PEFT-LoRA with rank_pattern (per-module ranks)
  -> train + eval on full GLUE validation split
```

## Quick start

```bash
pip install -r requirements.txt

# Smoke test (~3 min: saliency pass + 8 training steps)
python scripts/paramgrain_lora.py --task rte --ablation salsvd --smoke

# Full pilot: 2 ablations × 5 tasks × 5 seeds = 50 runs
for abl in salsvd random; do
  for task in sst2 mrpc cola stsb rte; do
    python scripts/paramgrain_lora.py --task $task --ablation $abl
  done
done
```

## Tasks and seeds

| Task  | Metric              | Seeds                  |
|-------|---------------------|------------------------|
| SST-2 | Accuracy            | 15, 25, 35, 45, 55     |
| MRPC  | Accuracy            | 15, 25, 35, 45, 55     |
| CoLA  | Matthews correlation| 15, 25, 35, 45, 55     |
| STS-B | Pearson             | 15, 25, 35, 45, 55     |
| RTE   | Accuracy            | 15, 25, 35, 45, 55     |

## W&B logging

Each run writes to project `{TASK}-ParamGrain-{Salsvd|Random}-LoRA-5-Seeds-2`.
Logged: per-module rank allocation (table of 72 modules), trainable parameter
count, budget used / target, active-module count, max/mean rank, saliency-pass
time, per-step training/eval metrics, best-checkpoint model artifact.

## Hardware

Single GPU sufficient. Tested on NVIDIA RTX 3080 Ti and V100 SXM2. fp16
mixed-precision is on by default; disable in `fine_tune_model` for
CPU/MPS. SVD pass adds a few seconds per run on top of the saliency forward-backward.

## Citation

If you use this repo, please cite the parent paper:

```bibtex
@article{kiprop2026sglora,
  title  = {When Does Saliency Help LoRA? A Rigorous Study of One-Shot
            Gradient Criteria for Adapter Placement},
  author = {Kiprop, Ernest and Nderu, Lawrence and Karanja, Mwangi},
  year   = {2026}
}
```
