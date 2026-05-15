"""
Parameter-granularity saliency for LoRA placement.

The granularity hierarchy of the parent paper:
  layer       -- score whole nn.Linear modules (parent paper, ties Random)
  sub-module  -- score attention heads / FFN modules (Beyond Layer Granularity)
  parameter   -- score individual weight entries, aggregated by SVD into
                 per-module rank allocations   <-- this experiment

Per-parameter |grad . weight| saliency is computed on a 5% sample with
RoBERTa-base frozen. Each module's saliency matrix is SVD-decomposed; the
singular values form a per-module "value curve" V_m(r) = sum_{i<=r} sigma_{m,i}.
A greedy allocator picks the (module, rank) with the largest marginal value
per LoRA parameter cost until a fixed total parameter budget is hit. The
result is a rank allocation across the 72 target modules of RoBERTa-base.

Three ablations are run at matched total LoRA-parameter budget
(default ~600k params, matching LoRA-AllAttn r=8 from the parent paper):

  salsvd   -- rank allocation by saliency-SVD greedy
  random   -- rank allocation by uniform-random greedy (matched-budget control)

If salsvd beats random on >=2 of 5 GLUE tasks under Holm correction, this
is the first one-shot signal escaping the no-go theorem of the parent paper.

W&B projects: {TASK}-ParamGrain-{Salsvd|Random}-LoRA-5-Seeds-2

Usage:
  python scripts/paramgrain_lora.py --task rte --ablation salsvd --smoke
  python scripts/paramgrain_lora.py --task rte --ablation salsvd
  python scripts/paramgrain_lora.py --task rte --ablation random --resume
"""

import argparse
import os
import random
import shutil
import tempfile
import time
from typing import Dict, Tuple

import evaluate
import numpy as np
import torch
import wandb
from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


# ----- Constants --------------------------------------------------------

MODEL_NAME = "roberta-base"
LORA_ALPHA_PER_RANK = 2  # so per-module alpha = 2*r (matches r=8/alpha=16)
LORA_DROPOUT = 0.1
WEIGHT_DECAY = 0.01
SEEDS_TO_RUN = [15, 25, 35, 45, 55]
DEFAULT_BUDGET = 600_000  # ~LoRA-AllAttn r=8 (the parent paper's recommended baseline)
SAL_BATCH_SIZE = 8
SAL_PERCENT = 0.05  # 5% sample for saliency pass

# Module-name suffixes for the 6 LoRA-targetable Linears per encoder block.
# (We use full module paths in target_modules so attention.output.dense and
# output.dense are unambiguous.)
TARGET_SUFFIXES = (
    "attention.self.query",
    "attention.self.key",
    "attention.self.value",
    "attention.output.dense",
    "intermediate.dense",
    "output.dense",
)


# ----- Per-task dataset / training config -------------------------------

DATASET_CONFIGS = {
    "sst2":  {"sentence_keys": ["sentence"],            "num_labels": 2, "eval_split": "validation", "metric": "accuracy"},
    "mrpc":  {"sentence_keys": ["sentence1","sentence2"], "num_labels": 2, "eval_split": "validation", "metric": "accuracy"},
    "cola":  {"sentence_keys": ["sentence"],            "num_labels": 2, "eval_split": "validation", "metric": "matthews_correlation"},
    "stsb":  {"sentence_keys": ["sentence1","sentence2"], "num_labels": 1, "eval_split": "validation", "metric": "pearson"},
    "rte":   {"sentence_keys": ["sentence1","sentence2"], "num_labels": 2, "eval_split": "validation", "metric": "accuracy"},
}

TASK_HPARAMS = {
    "sst2": {"num_train_epochs": 3, "learning_rate": 3e-4, "eval_steps": 200,  "logging_steps": 100},
    "mrpc": {"num_train_epochs": 5, "learning_rate": 3e-4, "eval_steps": 50,   "logging_steps": 25},
    "cola": {"num_train_epochs": 5, "learning_rate": 3e-4, "eval_steps": 100,  "logging_steps": 50},
    "stsb": {"num_train_epochs": 5, "learning_rate": 3e-4, "eval_steps": 100,  "logging_steps": 50},
    "rte":  {"num_train_epochs": 10,"learning_rate": 3e-4, "eval_steps": 50,   "logging_steps": 25},
}

BATCH_SIZE = 32


# ----- Helpers ----------------------------------------------------------

def set_seed(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def is_target_module(name: str) -> bool:
    return any(name.endswith(suffix) for suffix in TARGET_SUFFIXES)


# ----- Saliency pass: per-parameter |grad . weight| ---------------------

def collect_saliency_matrices(task: str, seed: int,
                               is_smoke_test: bool = False) -> Dict[str, torch.Tensor]:
    """Run one forward-backward over a 5% train sample with the frozen
    backbone. Return |grad . weight| accumulated per target Linear (mean
    over batches). Matrices have the same shape as the underlying weight."""
    set_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    d_cfg = DATASET_CONFIGS[task]

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=d_cfg["num_labels"],
        problem_type="regression" if task == "stsb" else None,
    ).to(device)
    model.train()  # need .grad set; we don't actually update weights

    # Collect target Linear modules
    targets = {name: mod for name, mod in model.named_modules()
               if isinstance(mod, torch.nn.Linear) and is_target_module(name)}
    print(f"[saliency] {len(targets)} target modules identified")

    # Load 5% of train (or just 64 examples in smoke mode)
    ds = load_dataset("glue", task)["train"]
    n = 64 if is_smoke_test else max(64, int(len(ds) * SAL_PERCENT))
    idx = np.random.RandomState(seed).choice(len(ds), n, replace=False)
    sub = ds.select(idx.tolist())

    keys = d_cfg["sentence_keys"]
    def tok(x):
        args = [x[k] for k in keys]
        return tokenizer(*args, truncation=True, padding="max_length", max_length=128)
    sub = sub.map(tok, batched=True)
    sub = sub.rename_column("label", "labels")
    sub.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
    loader = DataLoader(sub, batch_size=SAL_BATCH_SIZE, shuffle=False)

    saliencies = {name: torch.zeros_like(mod.weight, device=device)
                  for name, mod in targets.items()}
    n_batches = 0
    for batch in loader:
        model.zero_grad()
        out = model(
            input_ids=batch["input_ids"].to(device),
            attention_mask=batch["attention_mask"].to(device),
            labels=batch["labels"].to(device).float() if task == "stsb" else batch["labels"].to(device),
        )
        out.loss.backward()
        with torch.no_grad():
            for name, mod in targets.items():
                if mod.weight.grad is not None:
                    saliencies[name] += (mod.weight.grad * mod.weight).abs().detach()
        n_batches += 1
    for name in saliencies:
        saliencies[name] = (saliencies[name] / max(n_batches, 1)).cpu()

    del model
    torch.cuda.empty_cache()
    return saliencies


# ----- SVD value curves + greedy allocation ----------------------------

def svd_value_curves(saliencies: Dict[str, torch.Tensor]) -> Dict[str, np.ndarray]:
    """SVD each saliency matrix; return descending singular values."""
    out = {}
    for name, S in saliencies.items():
        _, sig, _ = torch.linalg.svd(S.float(), full_matrices=False)
        out[name] = sig.numpy()
    return out


def per_rank_cost(saliencies: Dict[str, torch.Tensor]) -> Dict[str, int]:
    """LoRA cost of adding rank +1 to a module = d_out + d_in (B and A matrices)."""
    return {name: int(S.shape[0] + S.shape[1]) for name, S in saliencies.items()}


def greedy_allocate(scores_at_rank: Dict[str, np.ndarray],
                     costs_per_rank: Dict[str, int],
                     budget: int,
                     max_rank_cap: int = 64) -> Tuple[Dict[str, int], int]:
    """Greedy rank allocator.

    At each step, add +1 rank to the (module, next-rank) with the highest
    marginal score-per-parameter, subject to the running parameter budget.
    Returns (ranks, used_budget).
    """
    ranks = {name: 0 for name in scores_at_rank}
    used = 0
    max_ranks = {name: min(len(scores_at_rank[name]), max_rank_cap)
                  for name in scores_at_rank}
    while True:
        best_name, best_pps = None, -np.inf
        for name in scores_at_rank:
            r = ranks[name]
            if r >= max_ranks[name]:
                continue
            cost = costs_per_rank[name]
            if used + cost > budget:
                continue
            pps = float(scores_at_rank[name][r]) / cost
            if pps > best_pps:
                best_pps = pps
                best_name = name
        if best_name is None:
            break
        ranks[best_name] += 1
        used += costs_per_rank[best_name]
    return ranks, used


def allocate_salsvd(saliencies: Dict[str, torch.Tensor], budget: int) -> Tuple[Dict[str, int], int]:
    sigmas = svd_value_curves(saliencies)
    costs = per_rank_cost(saliencies)
    return greedy_allocate(sigmas, costs, budget)


def allocate_random(saliencies: Dict[str, torch.Tensor], budget: int,
                    seed: int) -> Tuple[Dict[str, int], int]:
    """Random-rank control: replace SVD scores with uniform random scores
    at each rank. Same greedy budget-respecting routine, same costs."""
    rng = np.random.RandomState(seed)
    costs = per_rank_cost(saliencies)
    max_rank = max(min(S.shape) for S in saliencies.values())
    scores = {name: rng.rand(max_rank).astype(np.float32) for name in saliencies}
    return greedy_allocate(scores, costs, budget)


# ----- Training ---------------------------------------------------------

def fine_tune_model(cfg: dict, ranks: Dict[str, int],
                     run_name: str, ckpt_dir: str) -> None:
    is_smoke = cfg.get("is_smoke_test", False)
    d_cfg = DATASET_CONFIGS[cfg["dataset_name"]]
    keys = d_cfg["sentence_keys"]; eval_split = d_cfg["eval_split"]
    os.makedirs(ckpt_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"])
    dataset = load_dataset("glue", cfg["dataset_name"])

    def tok(x):
        args = [x[k] for k in keys]
        return tokenizer(*args, truncation=True, padding=False, max_length=128)

    tokenized = dataset.map(tok, batched=True)
    tokenized = tokenized.rename_column("label", "labels")
    tokenized.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

    base = AutoModelForSequenceClassification.from_pretrained(
        cfg["model_name"], num_labels=d_cfg["num_labels"],
        problem_type="regression" if cfg["dataset_name"] == "stsb" else None,
    )

    active = [name for name, r in ranks.items() if r > 0]
    if not active:
        raise RuntimeError("No modules selected (budget too small).")
    rank_pattern = {name: r for name, r in ranks.items() if r > 0}
    alpha_pattern = {name: LORA_ALPHA_PER_RANK * r for name, r in ranks.items() if r > 0}

    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=1,                            # overridden per-module by rank_pattern
        lora_alpha=LORA_ALPHA_PER_RANK, # overridden per-module by alpha_pattern
        lora_dropout=LORA_DROPOUT,
        target_modules=active,
        rank_pattern=rank_pattern,
        alpha_pattern=alpha_pattern,
        bias="none",
    )
    model = get_peft_model(base, lora_config)
    trainable, total = 0, 0
    for p in model.parameters():
        total += p.numel()
        if p.requires_grad:
            trainable += p.numel()
    wandb.log({
        "params/trainable": trainable, "params/total": total,
        "params/trainable_pct": 100.0 * trainable / total,
    })

    metric_obj = evaluate.load("glue", cfg["dataset_name"])
    def compute_metrics(p):
        preds = p.predictions[:, 0] if cfg["dataset_name"] == "stsb" else np.argmax(p.predictions, axis=1)
        return metric_obj.compute(predictions=preds, references=p.label_ids)

    metric_key = d_cfg["metric"]
    eval_steps = cfg["eval_steps"]; logging_steps = cfg["logging_steps"]
    max_steps = 8 if is_smoke else -1
    epochs = 1 if is_smoke else cfg["num_train_epochs"]

    targs = TrainingArguments(
        output_dir=ckpt_dir,
        num_train_epochs=epochs,
        max_steps=max_steps,
        per_device_train_batch_size=cfg["batch_size"],
        per_device_eval_batch_size=cfg["batch_size"],
        learning_rate=cfg["learning_rate"],
        weight_decay=cfg["weight_decay"],
        eval_strategy="steps",
        eval_steps=eval_steps,
        logging_steps=logging_steps,
        save_strategy="steps",
        save_steps=eval_steps,
        save_total_limit=1,
        load_best_model_at_end=not is_smoke,
        metric_for_best_model=f"eval_{metric_key}",
        greater_is_better=True,
        fp16=torch.cuda.is_available(),
        report_to=["wandb"],
        run_name=run_name,
        dataloader_num_workers=2,
        seed=cfg["seed"],
    )

    trainer = Trainer(
        model=model, args=targs,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized[eval_split],
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
    )

    t0 = time.time()
    trainer.train()
    wandb.log({"time/training_seconds": time.time() - t0})

    eval_out = trainer.evaluate()
    final_metric = eval_out.get(f"eval_{metric_key}", None)
    wandb.run.summary["final_metric"] = final_metric
    wandb.run.summary[f"eval/{metric_key}"] = final_metric

    if not is_smoke:
        with tempfile.TemporaryDirectory() as art_dir:
            trainer.save_model(art_dir)
            art = wandb.Artifact(name=f"best-{run_name}", type="model")
            art.add_dir(art_dir)
            wandb.log_artifact(art)

    del model, trainer
    torch.cuda.empty_cache()


# ----- W&B resume helper ------------------------------------------------

def fetch_completed_runs(task: str, ablation: str) -> set:
    api = wandb.Api()
    project = f"{task.upper()}-ParamGrain-{ablation.capitalize()}-LoRA-5-Seeds-2"
    try:
        entity = api.viewer.entity
        runs = api.runs(f"{entity}/{project}", filters={"state": "finished"})
        done = set()
        for r in runs:
            s = r.config.get("seed"); b = r.config.get("budget")
            if s is not None and b is not None:
                done.add((int(s), int(b)))
        print(f"[resume] {len(done)} already-finished runs in {project}")
        return done
    except Exception:
        print(f"[resume] project {project} not found yet, starting fresh")
        return set()


# ----- Single experiment / sweep ---------------------------------------

def run_single_experiment(cfg: dict) -> None:
    set_seed(cfg["seed"])
    project = f"{cfg['dataset_name'].upper()}-ParamGrain-{cfg['ablation'].capitalize()}-LoRA-5-Seeds-2"
    wandb.init(project=project, name=cfg["run_name"], group=cfg["group"], config=cfg)

    print(f"\n--- Saliency pass ({cfg['dataset_name']}, seed={cfg['seed']}) ---")
    t0 = time.time()
    saliencies = collect_saliency_matrices(
        cfg["dataset_name"], cfg["seed"], is_smoke_test=cfg.get("is_smoke_test", False)
    )
    t_sal = time.time() - t0

    if cfg["ablation"] == "salsvd":
        ranks, used = allocate_salsvd(saliencies, cfg["budget"])
    elif cfg["ablation"] == "random":
        ranks, used = allocate_random(saliencies, cfg["budget"], cfg["seed"])
    else:
        raise ValueError(f"unknown ablation: {cfg['ablation']}")

    active_modules = [n for n, r in ranks.items() if r > 0]
    rank_values = [r for r in ranks.values() if r > 0]
    wandb.log({
        "time/saliency_seconds": t_sal,
        "budget/target": cfg["budget"],
        "budget/used": used,
        "alloc/num_active_modules": len(active_modules),
        "alloc/max_rank": int(max(rank_values)) if rank_values else 0,
        "alloc/mean_rank_active": float(np.mean(rank_values)) if rank_values else 0.0,
    })

    # Log full rank vector as a small W&B table
    tab = wandb.Table(columns=["module", "rank"])
    for name, r in sorted(ranks.items(), key=lambda x: -x[1]):
        tab.add_data(name, int(r))
    wandb.log({"allocation/ranks": tab})

    print(f"  saliency pass: {t_sal:.1f}s")
    print(f"  budget used: {used}/{cfg['budget']} ({100*used/cfg['budget']:.1f}%)")
    print(f"  active modules: {len(active_modules)} / 72")
    if rank_values:
        print(f"  rank stats: max={max(rank_values)} mean={np.mean(rank_values):.1f} "
              f"median={int(np.median(rank_values))}")

    print(f"\n--- Fine-tuning ---")
    fine_tune_model(cfg, ranks, cfg["run_name"],
                     ckpt_dir=os.path.join(".paramgrain_ckpt", cfg["run_name"]))
    wandb.finish()


def run_study(task: str, seed: int, ablation: str, budget: int,
              is_smoke: bool = False, skip_completed: set = None) -> None:
    hp = TASK_HPARAMS[task]
    base_cfg = {
        "model_name": MODEL_NAME, "dataset_name": task,
        "ablation": ablation, "ablation_type": f"PARAMGRAIN_{ablation.upper()}",
        "lora_dropout": LORA_DROPOUT, "weight_decay": WEIGHT_DECAY,
        "batch_size": BATCH_SIZE, "seed": seed, "budget": budget,
        "learning_rate": hp["learning_rate"], "num_train_epochs": hp["num_train_epochs"],
        "eval_steps": hp["eval_steps"], "logging_steps": hp["logging_steps"],
        "is_smoke_test": is_smoke, "lora_alpha_per_rank": LORA_ALPHA_PER_RANK,
    }
    prefix = task.upper()
    abl = ablation.capitalize()
    if is_smoke:
        c = {**base_cfg}
        c.update({"max_steps": 8, "num_train_epochs": 1,
                   "logging_steps": 4, "eval_steps": 4})
        c["run_name"] = f"{prefix}-ParamGrain-{abl}-smoke"
        c["group"] = f"{prefix}-ParamGrain-{abl}"
        run_single_experiment(c)
        return

    if skip_completed and (seed, budget) in skip_completed:
        print(f"[resume] skipping completed run: seed={seed} budget={budget}")
        return

    c = {**base_cfg}
    c["run_name"] = f"{prefix}-ParamGrain-{abl}-b{budget//1000}k-seed{seed}"
    c["group"] = f"{prefix}-ParamGrain-{abl}-b{budget//1000}k"
    run_single_experiment(c)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=list(DATASET_CONFIGS))
    ap.add_argument("--ablation", required=True, choices=["salsvd", "random"])
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--seeds", type=int, nargs="+", default=SEEDS_TO_RUN)
    ap.add_argument("--budget", type=int, default=DEFAULT_BUDGET,
                    help=f"Total LoRA parameter budget (default {DEFAULT_BUDGET}).")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--batch-size", type=int, default=None)
    args = ap.parse_args()

    if args.batch_size is not None:
        global BATCH_SIZE
        BATCH_SIZE = args.batch_size

    if args.smoke:
        print(f"\n[SMOKE TEST] task={args.task}  ablation={args.ablation}\n")
        run_study(args.task, seed=42, ablation=args.ablation,
                   budget=args.budget, is_smoke=True)
        return

    skip = fetch_completed_runs(args.task, args.ablation) if args.resume else None
    print(f"\n[FULL RUN] task={args.task}  ablation={args.ablation}  "
          f"seeds={args.seeds}  budget={args.budget}"
          + ("  (resuming)" if args.resume else "") + "\n")
    for seed in args.seeds:
        run_study(args.task, seed=seed, ablation=args.ablation,
                   budget=args.budget, is_smoke=False, skip_completed=skip)


if __name__ == "__main__":
    main()
