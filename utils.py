"""
utils.py  (updated for multi-task + class weights)
────────────────────────────────────────────────────
Changes from previous version:
  • multitask_loss() now accepts fls_weights and esg_weights tensors
    → passed to F.cross_entropy(weight=...) to fix class imbalance
  • run_epoch(), train_step_multitask(), val_step_multitask() all
    accept and forward fls_weights / esg_weights
  • Everything else unchanged
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models.fin_transceiver import SNR_to_noise


# ─────────────────────────────────────────────────────────────────────────────
# Loss function
# ─────────────────────────────────────────────────────────────────────────────

def multitask_loss(
    sentiment_logits, sentiment_labels,
    fls_logits,       fls_labels,
    esg_logits,       esg_labels,
    H_hat, H,
    recon_weight: float = 0.1,
    fls_weights=None,   # ← NEW: [2]   tensor of class weights for FLS
    esg_weights=None,   # ← NEW: [4]   tensor of class weights for ESG
):
    """
    Combined multi-task loss:
        L_total = L_sentiment + L_fls + L_esg + recon_weight * L_recon

    fls_weights / esg_weights: if provided, passed to F.cross_entropy as
    inverse-frequency class weights to handle imbalanced labels.
    H.detach() stops reconstruction gradients from flowing into FinBERT.
    """
    loss_dict  = {}
    total_loss = torch.tensor(0.0, device=H_hat.device, requires_grad=True)

    # ── Task 1: Sentiment ──────────────────────────────────────────────────
    if sentiment_logits is not None and (sentiment_labels != -1).any():
        mask = sentiment_labels != -1
        L_s  = F.cross_entropy(sentiment_logits[mask], sentiment_labels[mask])
        loss_dict["L_sentiment"] = L_s.item()
        total_loss = total_loss + L_s

    # ── Task 2: FLS ────────────────────────────────────────────────────────
    if fls_logits is not None and (fls_labels != -1).any():
        mask = fls_labels != -1
        L_f  = F.cross_entropy(
            fls_logits[mask],
            fls_labels[mask],
            weight=fls_weights,   # None → uniform, tensor → weighted
        )
        loss_dict["L_fls"] = L_f.item()
        total_loss = total_loss + L_f

    # ── Task 3: ESG ────────────────────────────────────────────────────────
    if esg_logits is not None and (esg_labels != -1).any():
        mask = esg_labels != -1
        L_e  = F.cross_entropy(
            esg_logits[mask],
            esg_labels[mask],
            weight=esg_weights,   # None → uniform, tensor → weighted
        )
        loss_dict["L_esg"] = L_e.item()
        total_loss = total_loss + L_e

    # ── Reconstruction loss (channel stabiliser) ───────────────────────────
    n_active = sum([
        sentiment_logits is not None,
        fls_logits       is not None,
        esg_logits       is not None,
    ])
    if n_active > 1:
        L_r = F.mse_loss(H_hat, H.detach())
        loss_dict["L_recon"] = L_r.item()
        total_loss = total_loss + recon_weight * L_r

    loss_dict["L_total"] = total_loss.item()
    return total_loss, loss_dict


# ─────────────────────────────────────────────────────────────────────────────
# Accuracy
# ─────────────────────────────────────────────────────────────────────────────

def accuracy(logits, labels):
    """Overall accuracy, ignores -1 labels."""
    if logits is None:
        return None
    mask = labels != -1
    if mask.sum() == 0:
        return None
    preds   = logits[mask].argmax(dim=-1)
    correct = (preds == labels[mask]).float().sum()
    return (correct / mask.sum()).item()


# ─────────────────────────────────────────────────────────────────────────────
# Train step
# ─────────────────────────────────────────────────────────────────────────────

def train_step_multitask(
    model, optimizer,
    input_ids, attention_mask,
    sentiment_labels, fls_labels, esg_labels,
    n_var, channel,
    recon_weight: float = 0.1,
    fls_weights=None,
    esg_weights=None,
):
    model.train()
    optimizer.zero_grad()

    sent_logits, fls_logits, esg_logits, H_hat, H = model(
        input_ids, attention_mask, n_var=n_var, channel=channel
    )

    total_loss, loss_dict = multitask_loss(
        sent_logits, sentiment_labels,
        fls_logits,  fls_labels,
        esg_logits,  esg_labels,
        H_hat, H,
        recon_weight=recon_weight,
        fls_weights=fls_weights,
        esg_weights=esg_weights,
    )

    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()

    acc_dict = {
        "acc_sentiment": accuracy(sent_logits, sentiment_labels),
        "acc_fls":       accuracy(fls_logits,  fls_labels),
        "acc_esg":       accuracy(esg_logits,  esg_labels),
    }
    return loss_dict, acc_dict


# ─────────────────────────────────────────────────────────────────────────────
# Validation step
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def val_step_multitask(
    model,
    input_ids, attention_mask,
    sentiment_labels, fls_labels, esg_labels,
    n_var, channel,
    recon_weight: float = 0.1,
    fls_weights=None,
    esg_weights=None,
):
    model.eval()

    sent_logits, fls_logits, esg_logits, H_hat, H = model(
        input_ids, attention_mask, n_var=n_var, channel=channel
    )

    total_loss, loss_dict = multitask_loss(
        sent_logits, sentiment_labels,
        fls_logits,  fls_labels,
        esg_logits,  esg_labels,
        H_hat, H,
        recon_weight=recon_weight,
        fls_weights=fls_weights,
        esg_weights=esg_weights,
    )

    acc_dict = {
        "acc_sentiment": accuracy(sent_logits, sentiment_labels),
        "acc_fls":       accuracy(fls_logits,  fls_labels),
        "acc_esg":       accuracy(esg_logits,  esg_labels),
    }
    return loss_dict, acc_dict


# ─────────────────────────────────────────────────────────────────────────────
# Epoch runner
# ─────────────────────────────────────────────────────────────────────────────

def run_epoch(
    model, loader, n_var, channel,
    optimizer=None,
    recon_weight: float = 0.1,
    device="cuda",
    fls_weights=None,   # ← NEW
    esg_weights=None,   # ← NEW
):
    """
    One full epoch. Train if optimizer given, else validate.
    Returns averaged loss_dict and acc_dict over all batches.
    """
    is_train  = optimizer is not None
    all_losses = {}
    all_accs   = {}
    n_batches  = 0

    for batch in loader:
        input_ids, attention_mask, sent_labels, fls_labels, esg_labels = batch
        input_ids      = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        sent_labels    = sent_labels.to(device)
        fls_labels     = fls_labels.to(device)
        esg_labels     = esg_labels.to(device)

        if is_train:
            loss_dict, acc_dict = train_step_multitask(
                model, optimizer,
                input_ids, attention_mask,
                sent_labels, fls_labels, esg_labels,
                n_var, channel, recon_weight,
                fls_weights=fls_weights,
                esg_weights=esg_weights,
            )
        else:
            loss_dict, acc_dict = val_step_multitask(
                model,
                input_ids, attention_mask,
                sent_labels, fls_labels, esg_labels,
                n_var, channel, recon_weight,
                fls_weights=fls_weights,
                esg_weights=esg_weights,
            )

        for k, v in loss_dict.items():
            all_losses[k] = all_losses.get(k, 0.0) + v
        for k, v in acc_dict.items():
            if v is not None:
                all_accs[k] = all_accs.get(k, 0.0) + v

        n_batches += 1

    avg_losses = {k: v / n_batches for k, v in all_losses.items()}
    avg_accs   = {k: v / n_batches for k, v in all_accs.items()}
    return avg_losses, avg_accs


# ─────────────────────────────────────────────────────────────────────────────
# SNR sweep
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def snr_sweep_multitask(
    model, loader, channel, device="cuda",
    snrs=(-15, -10, -5, -2, 0, 2, 5, 10, 15, 20),
):
    model.eval()
    results = {}

    for snr_db in snrs:
        n_var = SNR_to_noise(snr_db)
        accs  = {"acc_sentiment": [], "acc_fls": [], "acc_esg": []}

        for batch in loader:
            input_ids, attention_mask, sent_labels, fls_labels, esg_labels = batch
            input_ids      = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            sent_labels    = sent_labels.to(device)
            fls_labels     = fls_labels.to(device)
            esg_labels     = esg_labels.to(device)

            sent_logits, fls_logits, esg_logits, H_hat, H = model(
                input_ids, attention_mask, n_var=n_var, channel=channel
            )

            for key, logits, labels in [
                ("acc_sentiment", sent_logits, sent_labels),
                ("acc_fls",       fls_logits,  fls_labels),
                ("acc_esg",       esg_logits,  esg_labels),
            ]:
                a = accuracy(logits, labels)
                if a is not None:
                    accs[key].append(a)

        results[snr_db] = {
            k: float(np.mean(v)) if v else None
            for k, v in accs.items()
        }
        active = {k: f"{v:.4f}" for k, v in results[snr_db].items() if v is not None}
        print(f"  SNR {snr_db:>4} dB | {active}")

    return results