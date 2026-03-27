# eval_tools.py
import os
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from utils import Channels, PowerNormalize, SNR_to_noise


def masked_mean_pool(memory, src_mask):
    """
    memory: [B, L, d_model]
    src_mask: [B, 1, L] where 1 means PAD
    returns pooled: [B, d_model]
    """
    valid = (1.0 - src_mask).transpose(1, 2)  # [B, L, 1]
    denom = valid.sum(dim=1).clamp(min=1.0)   # [B, 1]
    pooled = (memory * valid).sum(dim=1) / denom
    return pooled


def forward_decision_logits(model, sents, n_var, channel_name, pad_idx, no_noise=False):
    """
    returns logits [B, 3]
    """
    src_mask = (sents == pad_idx).unsqueeze(-2).float().to(sents.device)

    enc_output = model.encoder(sents, src_mask)
    tx = model.channel_encoder(enc_output)
    tx = PowerNormalize(tx)

    if no_noise:
        rx = tx
    else:
        channels = Channels()
        if channel_name == "AWGN":
            rx = channels.AWGN(tx, n_var)
        elif channel_name == "Rayleigh":
            rx = channels.Rayleigh(tx, n_var)
        elif channel_name == "Rician":
            rx = channels.Rician(tx, n_var)
        else:
            raise ValueError("channel must be AWGN, Rayleigh, or Rician")

    memory = model.channel_decoder(rx)
    pooled = masked_mean_pool(memory, src_mask)
    logits = model.decision_head(pooled)
    return logits


def evaluate_decision(model, loader, pad_idx, channel_name, n_var, device, no_noise=False):
    criterion = nn.CrossEntropyLoss()
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_count = 0

    with torch.no_grad():
        for sents, labels in loader:
            sents = sents.to(device)
            labels = labels.to(device)

            logits = forward_decision_logits(model, sents, n_var, channel_name, pad_idx, no_noise=no_noise)
            loss = criterion(logits, labels)

            preds = logits.argmax(dim=1)
            total_correct += (preds == labels).sum().item()
            total_loss += loss.item() * labels.size(0)
            total_count += labels.size(0)

    avg_loss = total_loss / max(total_count, 1)
    acc = total_correct / max(total_count, 1)
    return avg_loss, acc


def train_with_early_stopping(model, train_loader, val_loader, args, pad_idx, device):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    best_val_loss = float("inf")
    patience = 10
    bad = 0

    os.makedirs(args.checkpoint_path, exist_ok=True)
    best_path = os.path.join(args.checkpoint_path, "best_decision.pth")

    for epoch in range(args.epochs):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")

        # train with random SNR in a range (like comms training)
        n_var_train = np.random.uniform(SNR_to_noise(5), SNR_to_noise(10), size=(1))[0]

        total_correct, total_count, total_loss = 0, 0, 0.0

        for sents, labels in pbar:
            sents = sents.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits = forward_decision_logits(model, sents, n_var_train, args.channel, pad_idx, no_noise=False)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            preds = logits.argmax(dim=1)
            total_correct += (preds == labels).sum().item()
            total_loss += loss.item() * labels.size(0)
            total_count += labels.size(0)

            pbar.set_postfix(loss=loss.item(), acc=(total_correct / max(total_count, 1)))

        # validate at fixed SNR (10 dB)
        n_var_val = SNR_to_noise(10)
        val_loss, val_acc = evaluate_decision(model, val_loader, pad_idx, args.channel, n_var_val, device, no_noise=False)
        print(f"[VAL] epoch={epoch+1} val_loss={val_loss:.4f} val_acc={val_acc:.4f}")

        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            bad = 0
            torch.save(model.state_dict(), best_path)
            print(f"✅ saved best -> {best_path}")
        else:
            bad += 1
            if bad >= patience:
                print("⛔ early stopping")
                break

    return best_path


def snr_sweep_results(model, test_loader, args, pad_idx, device):
    best_path = os.path.join(args.checkpoint_path, "best_decision.pth")
    if os.path.exists(best_path):
        model.load_state_dict(torch.load(best_path, map_location=device))
        model.to(device)
        print("Loaded best checkpoint:", best_path)

    # baseline with no noise
    base_loss, base_acc = evaluate_decision(model, test_loader, pad_idx, args.channel, n_var=0.0, device=device, no_noise=True)
    print(f"[BASELINE no-noise] loss={base_loss:.4f} acc={base_acc:.4f}")

    snrs = [-15, -10, -5, -2, 0, 2, 5, 10, 15, 20]
    out = []
    for snr in snrs:
        n_var = SNR_to_noise(snr)
        loss, acc = evaluate_decision(model, test_loader, pad_idx, args.channel, n_var, device, no_noise=False)
        out.append((snr, loss, acc))
        print(f"[SNR {snr:>3} dB] loss={loss:.4f} acc={acc:.4f}")

    return base_loss, base_acc, out