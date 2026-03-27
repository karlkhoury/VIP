"""
main.py  (updated)
───────────────────────────────────
New vs previous version:
  • --freeze-finbert : FinBERT fully frozen entire training (Task 4)
  • --weighted-loss  : inverse-frequency weights for FLS + ESG (fixes imbalance)
  • print_label_distribution() : prints class counts at startup (Task 3 diagnostic)
  • checkpoint names now include freeze/weight tags so runs don't overwrite each other
"""

import argparse
import os
import json
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
from transformers import AutoTokenizer

from dataset                import PhraseBankTextDataset, make_collate_fn
from models.fin_transceiver import DeepSCFinDecision, SNR_to_noise
from utils                  import run_epoch, snr_sweep_multitask


# ─────────────────────────────────────────────────────────────────────────────
# Args
# ─────────────────────────────────────────────────────────────────────────────

def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data",            default="phrasebank_multitask.parquet")
    p.add_argument("--train-split",     type=float, default=0.8)
    p.add_argument("--tasks",           nargs="+",
                   choices=["sentiment","fls","esg"],
                   default=["sentiment","fls","esg"])
    p.add_argument("--channel",         default="AWGN",
                   choices=["AWGN","Rician","Rayleigh"])
    p.add_argument("--finbert",         default="ProsusAI/finbert")
    p.add_argument("--ch-dim",          type=int,   default=32)
    p.add_argument("--batch-size",      type=int,   default=16)
    p.add_argument("--stage1-epochs",   type=int,   default=8)
    p.add_argument("--epochs",          type=int,   default=30)
    p.add_argument("--lr-init",         type=float, default=1e-3)
    p.add_argument("--lr-finetune",     type=float, default=2e-5)
    p.add_argument("--unfreeze-last-n", type=int,   default=2)
    p.add_argument("--patience",        type=int,   default=6)
    p.add_argument("--snr-low",         type=float, default=5.0)
    p.add_argument("--snr-high",        type=float, default=20.0)
    p.add_argument("--recon-weight",    type=float, default=0.1)

    # ── NEW ──────────────────────────────────────────────────────────────────
    p.add_argument("--freeze-finbert", action="store_true",
                   help="Keep FinBERT 100%% frozen for entire training (Task 4). "
                        "No Phase B, no unfreeze. Tests if high accuracy comes "
                        "purely from pretrained FinBERT representations.")
    p.add_argument("--weighted-loss",  action="store_true",
                   help="Inverse-frequency class weights for FLS and ESG. "
                        "Fixes the 100%% FLS accuracy caused by class imbalance.")
    # ─────────────────────────────────────────────────────────────────────────

    p.add_argument("--checkpoint-dir", default="checkpoints/finbert-deepsc-multitask")
    p.add_argument("--checkpoint",     default=None)
    p.add_argument("--eval-only",      action="store_true")
    p.add_argument("--seed",           type=int, default=42)
    p.add_argument("--device",
                   default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Task 3 diagnostic — print class distribution
# ─────────────────────────────────────────────────────────────────────────────

def print_label_distribution(dataset):
    df = dataset.df
    print("── Label Distribution (Task 3 Diagnostic) ──────────────")

    if "sentiment_label" in df.columns:
        counts = df["sentiment_label"].value_counts().sort_index()
        names  = {0:"Negative", 1:"Neutral", 2:"Positive"}
        print("\n  Sentiment:")
        for idx, cnt in counts.items():
            print(f"    {names.get(int(idx), str(idx)):12s}: {cnt:5d}  ({100*cnt/len(df):.1f}%)")

    if "fls_label" in df.columns and (df["fls_label"] != -1).any():
        counts = df["fls_label"].value_counts().sort_index()
        names  = {0:"Not FLS", 1:"Non-specific FLS", 2:"Specific FLS"}
        print("\n  FLS:")
        for idx, cnt in counts.items():
            bar = "█" * int(50 * cnt / len(df))
            print(f"    {names.get(int(idx), str(idx)):20s}: {cnt:5d}  ({100*cnt/len(df):.1f}%)  {bar}")
        count_dict = {int(k): v for k, v in counts.items()}
        fls_total  = sum(count_dict.get(i, 0) for i in [1, 2])
        if fls_total > 0:
            ratio = count_dict.get(0, 0) / fls_total
            flag  = " ← SEVERE imbalance, use --weighted-loss" if ratio > 5 else ""
            print(f"    Not-FLS : FLS ratio = {ratio:.1f}:1{flag}")

    if "esg_label" in df.columns and (df["esg_label"] != -1).any():
        counts = df["esg_label"].value_counts().sort_index()
        names  = {0:"Environmental", 1:"Social", 2:"Governance", 3:"None"}
        print("\n  ESG:")
        for idx, cnt in counts.items():
            bar = "█" * int(50 * cnt / len(df))
            print(f"    {names.get(int(idx), str(idx)):16s}: {cnt:5d}  ({100*cnt/len(df):.1f}%)  {bar}")
        count_dict = {int(k): v for k, v in counts.items()}
        minority   = min(count_dict.get(0,1), count_dict.get(1,1), count_dict.get(2,1))
        if minority > 0 and 3 in count_dict:
            ratio = count_dict[3] / minority
            flag  = " ← SEVERE imbalance, use --weighted-loss" if ratio > 5 else ""
            print(f"    None : minority ratio = {ratio:.1f}:1{flag}")

    print("─"*55 + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# Class weights helper
# ─────────────────────────────────────────────────────────────────────────────

def compute_class_weights(label_series, num_classes, device):
    counts  = torch.zeros(num_classes)
    for lbl in label_series:
        if lbl != -1:
            counts[int(lbl)] += 1
    counts  = counts.clamp(min=1)
    weights = 1.0 / counts
    weights = weights / weights.sum() * num_classes
    return weights.to(device)


# ─────────────────────────────────────────────────────────────────────────────
# Optimizer
# ─────────────────────────────────────────────────────────────────────────────

def make_optimizer(model, lr, finbert_lr=None):
    finbert_params = list(model.semantic_encoder.parameters())
    other_params   = [p for p in model.parameters()
                      if not any(p is fp for fp in finbert_params)]
    if finbert_lr is None:
        trainable = [p for p in model.parameters() if p.requires_grad]
        return torch.optim.Adam(trainable, lr=lr)
    return torch.optim.Adam([
        {"params": [p for p in finbert_params if p.requires_grad], "lr": finbert_lr},
        {"params": other_params, "lr": lr},
    ])


# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────

def _log(epoch, phase, train_losses, train_accs, val_losses, val_accs):
    tl = train_losses.get("L_total", 0)
    vl = val_losses.get("L_total",   0)
    parts = []
    for key in ("acc_sentiment","acc_fls","acc_esg"):
        va = val_accs.get(key)
        if va is not None:
            parts.append(f"{key.replace('acc_','')}={va*100:.1f}%")
    print(f"  Ep {epoch:>3} [{phase}]  "
          f"train={tl:.4f}  val={vl:.4f}  acc: {' | '.join(parts)}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args   = get_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = args.device
    tasks  = tuple(args.tasks)

    freeze_tag = "frozen"   if args.freeze_finbert else "finetune"
    weight_tag = "weighted" if args.weighted_loss  else "uniform"
    run_tag    = f"{args.channel}_{freeze_tag}_{weight_tag}"

    print(f"\n{'='*60}")
    print(f"  FinBERT-DeepSC Multi-Task Training")
    print(f"  Channel  : {args.channel}")
    print(f"  Tasks    : {tasks}")
    print(f"  ch_dim   : {args.ch_dim}  ({768//args.ch_dim}× compression)")
    print(f"  FinBERT  : {'FULLY FROZEN (Task 4)' if args.freeze_finbert else 'last 2 layers fine-tuned'}")
    print(f"  Weights  : {'inverse-freq (fix imbalance)' if args.weighted_loss else 'uniform'}")
    print(f"  Run tag  : {run_tag}")
    print(f"  Device   : {device}")
    print(f"{'='*60}\n")

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # ── Dataset ──────────────────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(args.finbert)
    full_ds   = PhraseBankTextDataset(args.data, tasks=tasks)
    print_label_distribution(full_ds)          # ← Task 3 diagnostic

    n_train  = int(len(full_ds) * args.train_split)
    n_val    = len(full_ds) - n_train
    train_ds, val_ds = random_split(
        full_ds, [n_train, n_val],
        generator=torch.Generator().manual_seed(args.seed),
    )
    collate_fn   = make_collate_fn(tokenizer)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True,  collate_fn=collate_fn)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
                              shuffle=False, collate_fn=collate_fn)
    print(f"Dataset: {len(train_ds)} train / {len(val_ds)} val\n")

    # ── Class weights (optional) ─────────────────────────────────────────────
    fls_weights = esg_weights = None
    if args.weighted_loss:
        fls_weights = compute_class_weights(
            full_ds.df["fls_label"], num_classes=3, device=device)
        esg_weights = compute_class_weights(
            full_ds.df["esg_label"], num_classes=4, device=device)
        print(f"FLS weights: {[f'{w:.3f}' for w in fls_weights.tolist()]}")
        print(f"ESG weights: {[f'{w:.3f}' for w in esg_weights.tolist()]}\n")

    # ── Model ────────────────────────────────────────────────────────────────
    model = DeepSCFinDecision(
        finbert_name=args.finbert,
        ch_dim=args.ch_dim,
        tasks=tasks,
    ).to(device)

    # ── Eval only ────────────────────────────────────────────────────────────
    if args.eval_only:
        ckpt = args.checkpoint or \
               os.path.join(args.checkpoint_dir, f"best_{run_tag}.pth")
        print(f"Loading: {ckpt}")
        model.load_state_dict(torch.load(ckpt, map_location=device))
        results  = snr_sweep_multitask(model, val_loader, args.channel, device)
        out_path = os.path.join(args.checkpoint_dir, f"snr_sweep_{run_tag}.json")
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Saved → {out_path}")
        return

    # ─────────────────────────────────────────────────────────────────────────
    # STAGE 1
    # ─────────────────────────────────────────────────────────────────────────
    print("─"*50)
    print("STAGE 1 — Near-zero noise  (n_var = 1e-6)")
    print("─"*50)

    near_zero = 1e-6
    best_s1   = float("inf")
    ckpt_s1   = os.path.join(args.checkpoint_dir, f"stage1_{run_tag}.pth")

    model.freeze_finbert()   # always start frozen

    if args.freeze_finbert:
        # ── Task 4: no unfreeze ever ──────────────────────────────────────
        print(f"\nFinBERT FULLY FROZEN — lr={args.lr_init}\n")
        opt       = make_optimizer(model, lr=args.lr_init)
        no_imp    = 0
        for ep in range(1, args.stage1_epochs + 1):
            tl, ta = run_epoch(model, train_loader, near_zero, args.channel,
                               optimizer=opt, recon_weight=args.recon_weight,
                               device=device,
                               fls_weights=fls_weights, esg_weights=esg_weights)
            vl, va = run_epoch(model, val_loader,   near_zero, args.channel,
                               optimizer=None, recon_weight=args.recon_weight,
                               device=device,
                               fls_weights=fls_weights, esg_weights=esg_weights)
            _log(ep, "FROZEN", tl, ta, vl, va)
            if vl["L_total"] < best_s1:
                best_s1 = vl["L_total"]
                no_imp  = 0
                torch.save(model.state_dict(), ckpt_s1)
            else:
                no_imp += 1
                if no_imp >= 3:
                    print(f"  Early stop (Stage 1) at epoch {ep}")
                    break
    else:
        # ── Phase A: frozen ───────────────────────────────────────────────
        print(f"\nPhase A — frozen  (ep 1–3)  lr={args.lr_init}")
        opt = make_optimizer(model, lr=args.lr_init)
        for ep in range(1, 4):
            tl, ta = run_epoch(model, train_loader, near_zero, args.channel,
                               optimizer=opt, recon_weight=args.recon_weight,
                               device=device,
                               fls_weights=fls_weights, esg_weights=esg_weights)
            vl, va = run_epoch(model, val_loader,   near_zero, args.channel,
                               optimizer=None, recon_weight=args.recon_weight,
                               device=device,
                               fls_weights=fls_weights, esg_weights=esg_weights)
            _log(ep, "A", tl, ta, vl, va)
            if vl["L_total"] < best_s1:
                best_s1 = vl["L_total"]
                torch.save(model.state_dict(), ckpt_s1)

        # ── Phase B: unfreeze last N ──────────────────────────────────────
        print(f"\nPhase B — unfreeze last {args.unfreeze_last_n}  "
              f"(ep 4–{args.stage1_epochs})  lr={args.lr_finetune}")
        model.unfreeze_finbert_last_n(args.unfreeze_last_n)
        opt    = make_optimizer(model, lr=args.lr_finetune,
                                finbert_lr=args.lr_finetune)
        no_imp = 0
        for ep in range(4, args.stage1_epochs + 1):
            tl, ta = run_epoch(model, train_loader, near_zero, args.channel,
                               optimizer=opt, recon_weight=args.recon_weight,
                               device=device,
                               fls_weights=fls_weights, esg_weights=esg_weights)
            vl, va = run_epoch(model, val_loader,   near_zero, args.channel,
                               optimizer=None, recon_weight=args.recon_weight,
                               device=device,
                               fls_weights=fls_weights, esg_weights=esg_weights)
            _log(ep, "B", tl, ta, vl, va)
            if vl["L_total"] < best_s1:
                best_s1 = vl["L_total"]
                no_imp  = 0
                torch.save(model.state_dict(), ckpt_s1)
            else:
                no_imp += 1
                if no_imp >= 3:
                    print(f"  Early stop (Stage 1) at epoch {ep}")
                    break

    print(f"\nStage 1 best val_loss = {best_s1:.4f}  →  {ckpt_s1}")

    # ─────────────────────────────────────────────────────────────────────────
    # STAGE 2
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "─"*50)
    print(f"STAGE 2 — {args.channel}  SNR ∈ [{args.snr_low}, {args.snr_high}] dB")
    print("─"*50)

    model.load_state_dict(torch.load(ckpt_s1, map_location=device))

    if args.freeze_finbert:
        model.freeze_finbert()
        opt = make_optimizer(model, lr=args.lr_init)
    else:
        model.unfreeze_finbert_last_n(args.unfreeze_last_n)
        opt = make_optimizer(model, lr=args.lr_finetune,
                             finbert_lr=args.lr_finetune)

    best_s2  = float("inf")
    ckpt_s2  = os.path.join(args.checkpoint_dir, f"best_{run_tag}.pth")
    no_imp   = 0

    for ep in range(1, args.epochs + 1):
        snr_db = float(np.random.uniform(args.snr_low, args.snr_high))
        n_var  = SNR_to_noise(snr_db)

        tl, ta = run_epoch(model, train_loader, n_var, args.channel,
                           optimizer=opt, recon_weight=args.recon_weight,
                           device=device,
                           fls_weights=fls_weights, esg_weights=esg_weights)
        vl, va = run_epoch(model, val_loader,   n_var, args.channel,
                           optimizer=None, recon_weight=args.recon_weight,
                           device=device,
                           fls_weights=fls_weights, esg_weights=esg_weights)
        _log(ep, f"S2 {snr_db:.1f}dB", tl, ta, vl, va)

        if vl["L_total"] < best_s2:
            best_s2 = vl["L_total"]
            no_imp  = 0
            torch.save(model.state_dict(), ckpt_s2)
        else:
            no_imp += 1
            if no_imp >= args.patience:
                print(f"\n  Early stop at epoch {ep}")
                break

    print(f"\nStage 2 best = {best_s2:.4f}  →  {ckpt_s2}")

    # ─────────────────────────────────────────────────────────────────────────
    # SNR sweep
    # ─────────────────────────────────────────────────────────────────────────
    print(f"\n{'─'*50}\nFINAL SNR SWEEP — {args.channel}\n{'─'*50}")
    model.load_state_dict(torch.load(ckpt_s2, map_location=device))
    results  = snr_sweep_multitask(model, val_loader, args.channel, device)
    out_path = os.path.join(args.checkpoint_dir, f"snr_sweep_{run_tag}.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved → {out_path}")


if __name__ == "__main__":
    main()