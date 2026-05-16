"""
Analyze paramgrain pilot: salsvd vs random at matched parameter budget.

For each task at budget=600k:
  - Fetch salsvd results from {TASK}-ParamGrain-Salsvd-LoRA-5-Seeds-2
  - Fetch matched-budget random results from {TASK}-ParamGrain-Random-LoRA-5-Seeds-2
  - Pair runs by seed
  - One-sided Wilcoxon signed-rank (salsvd > random)
  - Cohen's d_z, mean delta, paired wins
  - Holm-Bonferroni across the 5 task contrasts

Decision: salsvd beats random on >=2 tasks (post-Holm) -> first one-shot
          signal escaping the no-go theorem.
          salsvd ties or loses everywhere               -> no-go closes for
          all granularity-by-signal-class combinations.

Usage:
  python scripts/analyze_paramgrain.py
  python scripts/analyze_paramgrain.py --csv results.csv
"""

import argparse
import sys
from collections import defaultdict

import numpy as np
import wandb
from scipy.stats import wilcoxon


# ----- Config -----------------------------------------------------------

TASKS = ["SST2", "MRPC", "COLA", "STSB", "RTE"]
# Base seeds always run; topup seeds added for MRPC/CoLA/RTE to unlock
# Holm under tighter Wilcoxon floors. Analysis uses whatever pairs exist
# in both arms per task.
SEEDS = [15, 25, 35, 45, 55, 65, 75, 85, 95, 105]
TARGET_BUDGET = 600_000

# Best-eval metric key per task (W&B summary field; matches paramgrain_lora.py).
TASK_METRIC = {
    "SST2":  "eval/accuracy",
    "MRPC":  "eval/accuracy",
    "COLA":  "eval/matthews_correlation",
    "STSB":  "eval/pearson",
    "RTE":   "eval/accuracy",
}


# ----- Helpers ----------------------------------------------------------

def fetch_seed_metrics(entity, project, metric_key, target_budget=TARGET_BUDGET):
    """Return {seed: best_eval_metric} for finished runs at budget=target."""
    api = wandb.Api()
    out = {}
    try:
        runs = api.runs(f"{entity}/{project}", filters={"state": "finished"})
        for r in runs:
            b = r.config.get("budget")
            s = r.config.get("seed")
            if b is None or s is None:
                continue
            if abs(int(b) - target_budget) > 1000:  # tolerate small budget drift
                continue
            # Skip smoke tests
            if r.config.get("is_smoke_test", False):
                continue
            val = r.summary.get(metric_key)
            if val is None:
                for alt in ("final_metric", metric_key.replace("eval/", "eval_")):
                    val = r.summary.get(alt)
                    if val is not None:
                        break
            if val is None:
                continue
            seed = int(s)
            if seed not in out or float(val) > out[seed]:
                out[seed] = float(val)
    except Exception as e:
        print(f"  WARNING: project {project} not accessible: {e}")
    return out


def paired_stats(sal, rnd, seeds):
    """Paired comparison on the intersection of seeds."""
    pairs = [(sal[s], rnd[s]) for s in seeds if s in sal and s in rnd]
    if len(pairs) < 3:
        return None
    p = np.array([x for x, _ in pairs])
    r = np.array([y for _, y in pairs])
    diff = p - r
    mean_d = float(diff.mean())
    sd_d   = float(diff.std(ddof=1)) if len(diff) > 1 else 0.0
    dz     = mean_d / sd_d if sd_d > 0 else float("inf") if mean_d > 0 else 0.0
    try:
        _, p_two = wilcoxon(p, r, zero_method="wilcox")
        if mean_d > 0:
            p_one = p_two / 2.0
        else:
            p_one = 1.0 - p_two / 2.0
    except ValueError:
        p_one = float("nan")
        p_two = float("nan")
    return {
        "n":           len(pairs),
        "salsvd_mean": float(p.mean()),
        "random_mean": float(r.mean()),
        "delta":       mean_d,
        "dz":          dz,
        "p_one":       float(p_one),
        "p_two":       float(p_two),
        "wins":        int((diff > 0).sum()),
    }


def holm_bonferroni(records):
    """Holm step-down on one-sided p-values."""
    valid = [r for r in records if r["stats"] and not np.isnan(r["stats"]["p_one"])]
    valid.sort(key=lambda r: r["stats"]["p_one"])
    m = len(valid)
    for i, r in enumerate(valid):
        p = r["stats"]["p_one"]
        r["stats"]["p_holm"] = min(1.0, p * (m - i))


# ----- Main -------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--entity", default=None)
    ap.add_argument("--csv", default=None)
    ap.add_argument("--alpha", type=float, default=0.05)
    args = ap.parse_args()

    api = wandb.Api()
    entity = args.entity or api.viewer.entity
    print(f"\nEntity: {entity}   alpha={args.alpha}   target budget={TARGET_BUDGET}\n")

    records = []
    for task in TASKS:
        p_sal = f"{task}-ParamGrain-Salsvd-LoRA-5-Seeds-2"
        p_rnd = f"{task}-ParamGrain-Random-LoRA-5-Seeds-2"
        print(f"[fetch] {p_sal}")
        sal = fetch_seed_metrics(entity, p_sal, TASK_METRIC[task])
        print(f"[fetch] {p_rnd}")
        rnd = fetch_seed_metrics(entity, p_rnd, TASK_METRIC[task])
        stats = paired_stats(sal, rnd, SEEDS)
        records.append({
            "task": task, "sal": sal, "rnd": rnd, "stats": stats,
        })

    holm_bonferroni(records)

    # ----- Report ------------------------------------------------------
    print("\n" + "=" * 90)
    print(f"{'Task':<6} {'n':>3} {'salsvd':>9} {'random':>9} {'delta':>9} "
          f"{'dz':>7} {'wins':>6} {'p_one':>9} {'p_holm':>9}")
    print("-" * 90)
    for rec in records:
        s = rec["stats"]
        if s is None:
            print(f"{rec['task']:<6}  -- insufficient paired data --")
            continue
        sig = ""
        ph = s.get("p_holm", float("nan"))
        if not np.isnan(ph):
            if ph < 0.001:    sig = "***"
            elif ph < 0.01:   sig = "**"
            elif ph < args.alpha: sig = "*"
        print(f"{rec['task']:<6} {s['n']:>3} "
              f"{s['salsvd_mean']:>9.4f} {s['random_mean']:>9.4f} "
              f"{s['delta']:>+9.4f} {s['dz']:>+7.2f} "
              f"{s['wins']}/{s['n']:<4} {s['p_one']:>9.4f} "
              f"{ph:>8.4f}{sig}")
    print("=" * 90)

    # ----- Verdict ------------------------------------------------------
    print("\n--- Verdict ---")
    significant_wins = []
    for rec in records:
        s = rec["stats"]
        if not s: continue
        ph = s.get("p_holm", 1.0)
        if ph < args.alpha and s["delta"] > 0:
            significant_wins.append(rec["task"])
    directional_wins = sum(1 for rec in records
                            if rec["stats"] and rec["stats"]["delta"] > 0)
    print(f"  Directional (delta>0): {directional_wins}/5  "
          f"{[r['task'] for r in records if r['stats'] and r['stats']['delta']>0]}")
    print(f"  Holm-significant wins (alpha={args.alpha}): "
          f"{len(significant_wins)}/5  {significant_wins}")

    print()
    if len(significant_wins) >= 2:
        print("Result: SalSVD beats Random on >=2 tasks post-Holm.")
        print("        FIRST one-shot placement signal escaping the no-go.")
        print("        Promote to companion paper / extension section.")
    elif directional_wins >= 3:
        print("Result: SalSVD directionally favored on >=3 tasks but no Holm-significant win.")
        print("        Suggestive but underpowered; consider topping up seeds or scaling.")
    else:
        print("Result: SalSVD does not consistently beat Random.")
        print("        No-go theorem extends to parameter granularity.")
        print("        All granularity-by-signal-class combinations now closed.")

    # ----- CSV ---------------------------------------------------------
    if args.csv:
        with open(args.csv, "w") as f:
            f.write("task,n,salsvd_mean,random_mean,delta,dz,wins,p_one,p_holm\n")
            for rec in records:
                s = rec["stats"]
                if not s: continue
                f.write(f"{rec['task']},{s['n']},"
                        f"{s['salsvd_mean']:.6f},{s['random_mean']:.6f},"
                        f"{s['delta']:.6f},{s['dz']:.4f},{s['wins']},"
                        f"{s['p_one']:.6f},{s.get('p_holm', float('nan')):.6f}\n")
        print(f"\nCSV written: {args.csv}")


if __name__ == "__main__":
    sys.exit(main())
