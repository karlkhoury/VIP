"""
run.py — launch one experiment from its config file (CLAUDE.md §10).

    python run.py --config runs/configs/exp2_main_cap8.yaml
    python run.py --config runs/configs/exp2_main_cap8.yaml --seeds 0      # one seed
    python run.py --config runs/configs/smoke.yaml                         # CPU smoke test

Outputs, per seed, in runs/<name>/seed<k>/:
    per_sentence.csv.gz   one row per (method, channel, SNR, priority, sentence)
    train_log.json        loss history and settings
    model*.pt             weights (if save_checkpoint: true)
"""

import argparse
import os
import time

import torch

from common.data import class_weights, make_loaders, print_label_distribution
from common.utils import load_config, load_encoder, save_json, set_seed


def run_tokenalloc(cfg, seed, out_dir, device, log):
    from tokenalloc.encoder import check_attention_rules
    from tokenalloc.evaluate import evaluate, mean_val_curves
    from tokenalloc.model import TokenAllocSystem
    from tokenalloc.train import train_tokenalloc

    bert, tok = load_encoder(cfg["encoder"])
    ds, split, loaders = make_loaders(cfg["data"], tok, cfg["train"].get("batch_size", 16), seed)
    model = TokenAllocSystem(bert, cfg.get("allocator", {})).to(device)

    batch = next(iter(loaders["val"]))
    ok, diffs = check_attention_rules(model.encoder, batch["input_ids"].to(device),
                                      batch["attention_mask"].to(device))
    log(f"attention-rule check: {'OK' if ok else 'FAILED'} {diffs}")
    if not ok:
        raise RuntimeError("Task-token attention masks are not respected by this transformers version")

    if cfg.get("load_checkpoint"):           # evaluate an already-trained model (e.g. Exp. 3)
        path = cfg["load_checkpoint"].format(seed=seed)
        model.load_state_dict(torch.load(path, map_location=device))
        log(f"loaded {path} (no training)")
        history = []
    else:
        cw = class_weights(ds, split["train"], cfg["train"].get("class_weighted", ["esg"]), device)
        history = train_tokenalloc(model, loaders, cw, cfg, seed, device, log)
    if cfg.get("save_checkpoint", False) and not cfg.get("load_checkpoint"):
        torch.save(model.state_dict(), os.path.join(out_dir, "model.pt"))

    ev = cfg["eval"]
    val_curves = None
    if "snr_only" in ev.get("policies", []):
        val_curves = mean_val_curves(model, loaders["val"], ev["channels"], ev["snrs"],
                                     cfg["noise_seed"], device)
    df = evaluate(model, loaders[ev.get("split", "test")], cfg, seed, device, val_curves,
                  split=ev.get("split", "test"))
    return df, history


def run_legacy(cfg, seed, out_dir, device, log):
    import pandas as pd

    from legacy.model import FinDeepSC
    from legacy.train import evaluate_legacy, train_legacy

    frames, history = [], []
    for channel in cfg["eval"]["channels"]:                  # one model per channel (paper)
        set_seed(seed)
        bert, tok = load_encoder(cfg["encoder"])
        ds, split, loaders = make_loaders(cfg["data"], tok, cfg["train"].get("batch_size", 16), seed)
        lg = cfg["legacy"]
        model = FinDeepSC(bert, lg.get("ch_dim", 32), lg.get("readout", "shared"),
                          lg.get("tx_mode", "per_token")).to(device)
        cw = class_weights(ds, split["train"], cfg["train"].get("class_weighted", ["esg"]), device)
        history += train_legacy(model, loaders, cw, cfg, channel, device, log)
        if cfg.get("save_checkpoint", False):
            torch.save(model.state_dict(), os.path.join(out_dir, f"model_{channel}.pt"))
        frames.append(evaluate_legacy(model, loaders[cfg["eval"].get("split", "test")], cfg,
                                      channel, seed, device))
    return pd.concat(frames, ignore_index=True), history


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--seeds", type=int, nargs="*", default=None, help="override cfg seeds")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    cfg = load_config(args.config)
    seeds = args.seeds if args.seeds else cfg["seeds"]
    if cfg["data"].get("synthetic"):                          # smoke tests only
        from common.tiny import synthetic_parquet
        synthetic_parquet(cfg["data"]["path"], n=cfg["data"].get("n", 96))
    os.makedirs(cfg["out_dir"], exist_ok=True)
    save_json(cfg, os.path.join(cfg["out_dir"], "config.json"))

    from common.data import PhraseBankDataset
    print(f"== {cfg['name']}  pipeline={cfg['pipeline']}  cap={cfg.get('cap')}  "
          f"seeds={seeds}  device={args.device}")
    print_label_distribution(PhraseBankDataset(cfg["data"]["path"]))

    for seed in seeds:
        out_dir = os.path.join(cfg["out_dir"], f"seed{seed}")
        os.makedirs(out_dir, exist_ok=True)
        set_seed(seed)
        t0 = time.time()
        runner = run_tokenalloc if cfg["pipeline"] == "tokenalloc" else run_legacy
        df, history = runner(cfg, seed, out_dir, args.device, print)
        df.to_csv(os.path.join(out_dir, "per_sentence.csv.gz"), index=False)
        save_json({"seed": seed, "minutes": (time.time() - t0) / 60, "history": history},
                  os.path.join(out_dir, "train_log.json"))
        summary = df.groupby(["method", "channel"])[["correct_S", "correct_F", "correct_E", "symbols"]].mean()
        print(summary.round(3).to_string())
        print(f"seed {seed} done in {(time.time() - t0) / 60:.1f} min -> {out_dir}")


if __name__ == "__main__":
    main()
