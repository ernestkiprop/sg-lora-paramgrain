# Design rationale

Concise version of the reasoning. See parent paper for theoretical
context (Jensen theorem, capacity-controller account, no-go theorem
extending across granularities and signal classes).

## Why parameter granularity

The parent paper's no-go closure list:

| Granularity | Gradient signal | Info-theoretic | Open? |
|---|---|---|---|
| Layer | SNIP/GN/Fisher: ties Random | Probe: trails Random | closed |
| Sub-module (head/FFN) | SNIP: RTE-only, FFN bias | not run | partly closed |
| Parameter (row/column) | not run | — | OPEN |

Parameter granularity is the natural granularity for LoRA: LoRA modifies
rank-r subspaces of each weight matrix, so per-rank-direction saliency
maps directly onto LoRA's allocation problem. The coarser granularities
necessarily aggregate across rank directions; this experiment doesn't.

## Why SVD of the saliency matrix

LoRA's update is `ΔW = BA` with `A ∈ ℝ^{r×d_in}, B ∈ ℝ^{d_out×r}`. The
rank-r constraint means LoRA can capture **at most r singular directions**
of the optimal update direction. The saliency matrix `S = |∇L ⊙ W|`,
SVD'd, gives a value curve `σ_1 ≥ σ_2 ≥ ...` that estimates how much
"importance mass" lives in each rank direction *of the saliency*. Two
intuitions support using this curve to drive LoRA rank allocation:

- A module whose saliency matrix has fast singular-value decay can be
  well-approximated by a low-rank update (small `r` captures most of
  the variance). Allocate moderate rank.
- A module with slow decay needs more rank to capture the same
  fraction of importance mass. Allocate more rank.
- A module whose top singular values are very small is uninformative
  at any rank. Allocate zero.

This is the closest one-shot analogue to AdaLoRA's singular-value-based
online importance, which is the only existing PEFT method that operates
on the right granularity.

## Greedy allocation, not threshold-based

A simpler approach — "rank ∝ total saliency mass" — confounds the
*scale* of a module's saliency with the *distribution* across rank
directions. Two modules can have the same total mass but very different
singular-value decay; one wants high rank, the other wants low.

Greedy adds +1 rank at the highest *marginal* value-per-parameter
`σ_{m,r+1} / (d_out_m + d_in_m)`. This:

1. Respects the parameter budget exactly.
2. Allocates within and across modules using the same currency
   (saliency-per-parameter).
3. Has no free hyperparameters beyond the budget itself.
4. Is the natural matched-budget control: replace the SVD singular
   values with uniform random draws and run the same algorithm to get
   the Random arm.

## Budget choice

Target: ~600k LoRA params, matching LoRA-AllAttn at rank 8 (the parent
paper's recommended baseline). This puts the experiment at the same
parameter cost as the strongest existing uniform baseline. If salsvd
wins, it wins against an already-strong reference.

The two ablations (`salsvd`, `random`) use the same greedy allocator
with the same budget, so any difference in trainable parameter count
between arms is within 1–2 rank-update steps (~3000 params) — well
below the 5% capacity threshold for confounding the parent paper used.

## Max-rank cap

Caps per-module rank at 64. Without a cap the greedy could in principle
assign rank-768 to one FFN module if its singular-value decay is very
slow, which would be:

- Mathematically the greedy optimum given the score.
- Practically dubious (extreme rank concentration far outside the
  rank-8 regime the parent paper studied).
- Hard to interpret.

64 is well above any rank the parent paper or its baselines use, so the
cap is unlikely to bind for reasonable saliency-matrix shapes. We log
`alloc/max_rank` per run to verify.

## Random control as the discriminating test

The interesting comparison is *not* "salsvd vs uniform-rank-everywhere".
A uniform rank of `r=4` across all 72 modules also costs ~600k params but
is a wholly different placement (no zero-rank modules) — it tests a
different hypothesis. The proper matched control is **same allocator,
same budget, different scoring** — i.e., salsvd vs random-greedy.

If random-greedy wins or ties: the allocator does the work, not the
saliency. This is the no-go conclusion at parameter granularity.

If salsvd-greedy wins: the saliency contributes information that random
does not. This is the first one-shot placement signal escaping the
parent paper's closure.

## What the parent paper's checklist requires

Any positive claim from this experiment must clear:

1. Beats Random at every budget where it claims to win.
2. Beats LoRA-AllAttn at matched parameter count on ≥3 of 5 tasks.
3. Holm–Bonferroni survival with ≥5 seeds and Cohen's d_z.
4. Best-budget reported alongside the headline budget cell.
5. Validates on RoBERTa-large minimum.

This pilot covers (1) and (3) at budget ≈ 600k; (2) is an immediate
follow-up against the existing LoRA-AllAttn r=8 numbers in the parent
paper (no new runs needed); (4) and (5) are gated on positive
(1)/(2)/(3) results.
