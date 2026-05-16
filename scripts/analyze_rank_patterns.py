"""
Pull SalSVD/Random rank allocations from W&B and aggregate.

For each (task, ablation, seed) ParamGrain run, the training script
logs an `allocation/ranks` Table -- one row per target module with
its chosen LoRA rank. This script:

  * Fetches that table for every requested seed and ablation
  * Parses module names into (block_idx, module_type)
  * Aggregates per (module_type, block_idx): mean rank across seeds
  * Reports allocation profile: which module types receive rank?
                                which layer depths?

Intended use: generate the rank-allocation breakdown for the
companion paper (Figure 1 candidate -- "where does saliency put
the rank?").

Usage:
  python scripts/analyze_rank_patterns.py --task cola --ablation salsvd
  python scripts/analyze_rank_patterns.py --task cola --ablation salsvd \\
        --seeds 15 25 35 45 55 65 75 85 95 105 --csv cola_salsvd_ranks.csv
"""

import argparse
import json
import re
import sys
from collections import defaultdict

import numpy as np
import wandb


# ----- Module-name parsing ---------------------------------------------

# RoBERTa target modules look like:
#   roberta.encoder.layer.{block_idx}.attention.self.query
#   roberta.encoder.layer.{block_idx}.attention.self.key
#   roberta.encoder.layer.{block_idx}.attention.self.value
#   roberta.encoder.layer.{block_idx}.attention.output.dense
#   roberta.encoder.layer.{block_idx}.intermediate.dense
#   roberta.encoder.layer.{block_idx}.output.dense
# We group them by short module-type label.

MODULE_TYPE_RULES = [
    (re.compile(r"attention\.self\.query$"),   "Q"),
    (re.compile(r"attention\.self\.key$"),     "K"),
    (re.compile(r"attention\.self\.value$"),   "V"),
    (re.compile(r"attention\.output\.dense$"), "O"),
    (re.compile(r"intermediate\.dense$"),      "FC1"),
    (re.compile(r"^(?!.*attention).*output\.dense$"), "FC2"),
]

MODULE_TYPE_ORDER = ["Q", "K", "V", "O", "FC1", "FC2"]


def parse_module_name(name: str):
    """(block_idx, module_type) or (None, None) if unparseable."""
    m = re.search(r"layer\.(\d+)\.", name)
    block_idx = int(m.group(1)) if m else None
    mtype = None
    for pat, label in MODULE_TYPE_RULES:
        if pat.search(name):
            mtype = label
            break
    return block_idx, mtype


# ----- W&B fetch -------------------------------------------------------

def project_name(task: str, ablation: str, model: str) -> str:
    suffix = ""
    if model == "roberta-large":
        suffix = "-Large"
    elif model != "roberta-base":
        short = model.split("/")[-1]
        suffix = "-" + short.replace("-", "").capitalize()
    return f"{task.upper()}-ParamGrain-{ablation.capitalize()}{suffix}-LoRA-5-Seeds-2"


def fetch_rank_tables(entity: str, task: str, ablation: str, model: str,
                       seeds: list) -> dict:
    """Return {seed: {module_name: rank}}."""
    api = wandb.Api()
    proj = project_name(task, ablation, model)
    out = {}
    runs = api.runs(f"{entity}/{proj}", filters={"state": "finished"})
    wanted = set(seeds)
    for r in runs:
        s = r.config.get("seed")
        if s is None or int(s) not in wanted:
            continue
        seed = int(s)
        if seed in out:
            continue
        # The table is logged as wandb.Table; pull via run.logged_artifacts
        # Or directly via run.history(); cleanest is to pull the artifact-free
        # logged table from run.summary["allocation/ranks"] when present, else
        # iterate run.history() looking for it.
        try:
            tbl = None
            # 1. Try summary slot
            slot = r.summary.get("allocation/ranks", None)
            if slot is not None and hasattr(slot, "get_pandas_df"):
                tbl = slot.get_pandas_df()
            # 2. Fall back to history scan
            if tbl is None:
                for row in r.scan_history(keys=["allocation/ranks"]):
                    val = row.get("allocation/ranks")
                    if val is None:
                        continue
                    # W&B stores tables as JSON refs; fetch via api file
                    path = val.get("path") if isinstance(val, dict) else None
                    if path:
                        f = r.file(path)
                        f.download(replace=True, root="/tmp/rank_tables")
                        with open(f"/tmp/rank_tables/{path}") as fh:
                            data = json.load(fh)
                        cols = data.get("columns", [])
                        rows = data.get("data", [])
                        if "module" in cols and "rank" in cols:
                            i_mod = cols.index("module")
                            i_rnk = cols.index("rank")
                            tbl = {row[i_mod]: int(row[i_rnk]) for row in rows}
                        break
            if isinstance(tbl, dict):
                out[seed] = tbl
            elif tbl is not None:
                # pandas DataFrame branch
                out[seed] = {row["module"]: int(row["rank"]) for _, row in tbl.iterrows()}
        except Exception as e:
            print(f"  WARN seed={seed}: {e}")
            continue
    return out


# ----- Aggregation ------------------------------------------------------

def aggregate(rank_tables: dict):
    """Aggregate per (module_type, block_idx) across seeds."""
    by_type     = defaultdict(list)   # type   -> [total rank per seed]
    by_depth    = defaultdict(list)   # block  -> [total rank per seed]
    by_cell     = defaultdict(list)   # (type, block) -> [rank per seed]
    n_active    = []                  # active modules per seed
    rank_max    = []                  # max rank per seed
    used_budget = []                  # used budget per seed (if computable; skip)

    for seed, tbl in rank_tables.items():
        per_type_seed  = defaultdict(int)
        per_depth_seed = defaultdict(int)
        active = 0
        rmax   = 0
        for mod, r in tbl.items():
            r = int(r)
            block, mtype = parse_module_name(mod)
            if mtype is None:
                continue
            by_cell[(mtype, block)].append(r)
            per_type_seed[mtype] += r
            if block is not None:
                per_depth_seed[block] += r
            if r > 0:
                active += 1
            if r > rmax:
                rmax = r
        for mt in MODULE_TYPE_ORDER:
            by_type[mt].append(per_type_seed.get(mt, 0))
        for d in sorted(per_depth_seed):
            by_depth[d].append(per_depth_seed[d])
        n_active.append(active)
        rank_max.append(rmax)

    return by_type, by_depth, by_cell, n_active, rank_max


# ----- Report ----------------------------------------------------------

def report(task, ablation, model, rank_tables, csv_path=None):
    if not rank_tables:
        print("No rank tables fetched.")
        return
    by_type, by_depth, by_cell, n_active, rank_max = aggregate(rank_tables)
    n = len(rank_tables)

    print(f"\n=== {task.upper()} / {ablation} / {model} ===")
    print(f"seeds fetched: {n}    active modules per seed: "
          f"mean={np.mean(n_active):.1f} ± {np.std(n_active):.1f}    "
          f"max rank per seed: mean={np.mean(rank_max):.1f}")

    print(f"\nTotal rank per module type (mean across seeds):")
    print(f"{'Type':<6} {'mean':>8} {'std':>7} {'share':>7}")
    type_totals = {mt: np.mean(by_type[mt]) for mt in MODULE_TYPE_ORDER}
    grand = sum(type_totals.values()) or 1.0
    for mt in MODULE_TYPE_ORDER:
        m = np.mean(by_type[mt]); s = np.std(by_type[mt])
        print(f"{mt:<6} {m:>8.1f} {s:>7.1f} {100*m/grand:>6.1f}%")

    if by_depth:
        print(f"\nTotal rank per encoder block (mean across seeds):")
        print(f"{'block':>6} {'mean':>8} {'std':>7}")
        for d in sorted(by_depth):
            print(f"{d:>6} {np.mean(by_depth[d]):>8.1f} {np.std(by_depth[d]):>7.1f}")

    if csv_path:
        with open(csv_path, "w") as f:
            f.write("module_type,block_idx,mean_rank,std_rank,n_seeds\n")
            for (mt, block), values in sorted(by_cell.items(),
                                              key=lambda kv: (MODULE_TYPE_ORDER.index(kv[0][0])
                                                              if kv[0][0] in MODULE_TYPE_ORDER else 99,
                                                              kv[0][1] if kv[0][1] is not None else -1)):
                f.write(f"{mt},{block},{np.mean(values):.3f},"
                        f"{np.std(values):.3f},{len(values)}\n")
        print(f"\nCSV written: {csv_path}")


# ----- Main ------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True,
                    choices=["sst2", "mrpc", "cola", "stsb", "rte"])
    ap.add_argument("--ablation", required=True, choices=["salsvd", "random"])
    ap.add_argument("--model", default="roberta-base")
    ap.add_argument("--seeds", type=int, nargs="+",
                    default=[15, 25, 35, 45, 55, 65, 75, 85, 95, 105])
    ap.add_argument("--entity", default=None)
    ap.add_argument("--csv", default=None)
    args = ap.parse_args()

    api = wandb.Api()
    entity = args.entity or api.viewer.entity
    print(f"Entity: {entity}    project: "
          f"{project_name(args.task, args.ablation, args.model)}")
    print(f"Fetching seeds: {args.seeds}")

    tables = fetch_rank_tables(entity, args.task, args.ablation,
                                args.model, args.seeds)
    print(f"Got {len(tables)} rank tables.")
    report(args.task, args.ablation, args.model, tables, csv_path=args.csv)


if __name__ == "__main__":
    sys.exit(main())
