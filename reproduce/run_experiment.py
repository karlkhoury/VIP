"""
reproduce/run_experiment.py
───────────────────────────
Entry point used by the paper notebook. Same behaviour as run.py for the tokenalloc
pipeline, plus three things the notebook needs:

  * eval.splits: evaluate on several splits in one run (e.g. [val, test]), so the
    price lambda can be selected on validation without looking at the test set;
  * allocator-only checkpoints: a two-stage run saves allocator.pt (a few hundred KB)
    instead of the full model (~440 MB), since the pipeline is unchanged;
  * load_allocator: re-evaluate a saved allocator on its frozen pipeline without
    retraining (priority sweep, ablation of the selected model).

    python -m reproduce.run_experiment --config <path> [--seeds 0 1 2] [--device cpu]
"""

import argparse
import os
import time

import pandas as pd
import torch

from common.data import PhraseBankDataset, class_weights, make_loaders, print_label_distribution
from common.utils import load_config, load_encoder, save_json, set_seed


def run_once(cfg, seed, out_dir, device, log):
    from tokenalloc.encoder import check_attention_rules
    from tokenalloc.evaluate import evaluate
    from tokenalloc.model import TokenAllocSystem
    from tokenalloc.train import train_allocator_frozen, train_tokenalloc

    bert, tok = load_encoder(cfg["encoder"])
    ds, split, loaders = make_loaders(cfg["data"], tok, cfg["train"].get("batch_size", 16), seed)
    model = TokenAllocSystem(bert, cfg.get("allocator", {}), cfg.get("tokens_per_task", 4),
                             cfg.get("max_tokens_per_task")).to(device)

    batch = next(iter(loaders["val"]))
    ok, diffs = check_attention_rules(model.encoder, batch["input_ids"].to(device),
                                      batch["attention_mask"].to(device))
    log(f"attention-rule check: {'OK' if ok else 'FAILED'} {diffs}")
    if not ok:
        raise RuntimeError("Task-token attention masks are not respected by this transformers version")

    frozen = cfg["train"].get("allocator_stage", "joint") == "frozen"
    if frozen or cfg.get("load_allocator"):
        path = cfg["init_checkpoint"].format(seed=seed)
        state = {k: v for k, v in torch.load(path, map_location=device).items()
                 if not k.startswith("allocator.")}
        missing, unexpected = model.load_state_dict(state, strict=False)
        assert all(k.startswith("allocator.") for k in missing) and not unexpected, (missing, unexpected)
        log(f"pipeline from {path}")

    if cfg.get("load_allocator"):
        path = cfg["load_allocator"].format(seed=seed)
        model.allocator.load_state_dict(torch.load(path, map_location=device))
        log(f"loaded allocator {path} (no training)")
        history = []
    elif frozen:
        log("training the allocator only")
        cw = class_weights(ds, split["train"], cfg["train"].get("class_weighted", ["esg"]), device)
        history = train_allocator_frozen(model, loaders, cw, cfg, seed, device, log)
    else:
        cw = class_weights(ds, split["train"], cfg["train"].get("class_weighted", ["esg"]), device)
        history = train_tokenalloc(model, loaders, cw, cfg, seed, device, log)

    if cfg.get("save_checkpoint", False) and not cfg.get("load_allocator"):
        if frozen:
            torch.save(model.allocator.state_dict(), os.path.join(out_dir, "allocator.pt"))
        else:
            torch.save(model.state_dict(), os.path.join(out_dir, "model.pt"))

    ev = cfg["eval"]
    splits = ev.get("splits", [ev.get("split", "test")])
    df = pd.concat([evaluate(model, loaders[s], cfg, seed, device, None, split=s) for s in splits],
                   ignore_index=True)
    return df, history


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--seeds", type=int, nargs="*", default=None)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    cfg = load_config(args.config)
    seeds = args.seeds if args.seeds else cfg["seeds"]
    if cfg["data"].get("synthetic"):
        from common.tiny import synthetic_parquet
        synthetic_parquet(cfg["data"]["path"], n=cfg["data"].get("n", 96))
    os.makedirs(cfg["out_dir"], exist_ok=True)
    save_json(cfg, os.path.join(cfg["out_dir"], "config.json"))
    print(f"== {cfg['name']}  cap={cfg.get('cap')}  seeds={seeds}  device={args.device}")
    print_label_distribution(PhraseBankDataset(cfg["data"]["path"]))

    for seed in seeds:
        out_dir = os.path.join(cfg["out_dir"], f"seed{seed}")
        os.makedirs(out_dir, exist_ok=True)
        set_seed(seed)
        t0 = time.time()
        df, history = run_once(cfg, seed, out_dir, args.device, print)
        df.to_csv(os.path.join(out_dir, "per_sentence.csv.gz"), index=False)
        save_json({"seed": seed, "minutes": (time.time() - t0) / 60, "history": history},
                  os.path.join(out_dir, "train_log.json"))
        print(f"seed {seed} done in {(time.time() - t0) / 60:.1f} min -> {out_dir}")


if __name__ == "__main__":
    main()
