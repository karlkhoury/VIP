"""
fin_transceiver.py  (updated for multi-task)
─────────────────────────────────────────────
Changes from original:
  • Added fls_head  (768 → 256 → 2)
  • Added esg_head  (768 → 256 → 4)
  • forward() now returns all three head outputs + H_hat + H
  • tasks argument controls which heads are active
    → inactive heads still exist but their outputs are None
    → this lets you choose at runtime which tasks to train/eval
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel
import numpy as np


# ── Channel simulation ────────────────────────────────────────────────────────

class Channels:
    """AWGN, Rayleigh, and Rician channel models."""

    @staticmethod
    def AWGN(Tx_sig, n_var):
        device = Tx_sig.device
        noise  = torch.randn_like(Tx_sig) * (n_var ** 0.5)
        return Tx_sig + noise

    @staticmethod
    def Rayleigh(Tx_sig, n_var):
        device = Tx_sig.device
        shape  = Tx_sig.shape
        # Complex fading: real + imag parts
        h_real = torch.randn(*shape, device=device) / (2 ** 0.5)
        h_imag = torch.randn(*shape, device=device) / (2 ** 0.5)
        noise  = torch.randn_like(Tx_sig) * (n_var ** 0.5)
        # Apply fading and compensate (channel estimation)
        Rx_sig = h_real * Tx_sig + noise
        # Simple channel equalisation: divide by h_real
        h_real = torch.clamp(h_real, min=1e-6)
        return Rx_sig / h_real

    @staticmethod
    def Rician(Tx_sig, n_var, K=1.0):
        device = Tx_sig.device
        shape  = Tx_sig.shape
        # LoS component
        los    = torch.ones(*shape, device=device) * (K / (K + 1)) ** 0.5
        # Scattered component
        scatter_real = torch.randn(*shape, device=device) * (1 / (2*(K+1))) ** 0.5
        scatter_imag = torch.randn(*shape, device=device) * (1 / (2*(K+1))) ** 0.5
        h_real = los + scatter_real
        noise  = torch.randn_like(Tx_sig) * (n_var ** 0.5)
        Rx_sig = h_real * Tx_sig + noise
        h_real = torch.clamp(h_real, min=1e-6)
        return Rx_sig / h_real


def SNR_to_noise(snr_db):
    """Convert SNR in dB to noise variance (signal power assumed = 1)."""
    snr_linear = 10 ** (snr_db / 10.0)
    return 1.0 / snr_linear


def PowerNormalize(x):
    """Normalise x to unit average power per batch."""
    power = (x ** 2).mean()
    return x / (power ** 0.5 + 1e-8)


# ── Channel Decoder ───────────────────────────────────────────────────────────

class ChannelDecoder(nn.Module):
    """
    Two-block residual MLP.
    32 → 256 → 768 with skip connections and LayerNorm.
    """

    def __init__(self, in_dim: int, mid_dim: int, out_dim: int):
        super().__init__()
        # Block 1
        self.fc1  = nn.Linear(in_dim,  mid_dim)
        self.fc2  = nn.Linear(mid_dim, out_dim)
        self.fc3  = nn.Linear(out_dim, mid_dim)
        self.ln1  = nn.LayerNorm(mid_dim)
        # Block 2
        self.fc4  = nn.Linear(mid_dim, out_dim)
        self.fc5  = nn.Linear(out_dim, mid_dim)
        self.ln2  = nn.LayerNorm(mid_dim)
        self.drop = nn.Dropout(0.1)
        self.proj = nn.Linear(mid_dim, out_dim)
    def forward(self, x):
        x1   = self.fc1(x)
        out1 = self.ln1(x1 + self.fc3(F.gelu(self.fc2(F.gelu(x1)))))
        out1 = self.drop(out1)
        out2 = self.ln2(out1 + self.fc5(F.gelu(self.fc4(F.gelu(out1)))))
        return self.proj(out2)   # ← now outputs 768
'''
class ChannelDecoder(nn.Module):
    def __init__(self, in_dim, mid_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, mid_dim),
            nn.GELU(),
            nn.Linear(mid_dim, out_dim),
        )
    def forward(self, x):
        return self.net(x)
'''
# ── Main Model ────────────────────────────────────────────────────────────────

class DeepSCFinDecision(nn.Module):
    """
    FinBERT-DeepSC Multi-Task Receiver.

    Transmitter:
        FinBERT → Channel Encoder (768→ch_dim) → Power Norm

    Wireless channel:
        AWGN / Rician / Rayleigh

    Receiver (shared):
        Channel Decoder (ch_dim→768) → Masked Mean Pool → shared vector [768]

    Task heads (each independent, trained from scratch):
        Task 1 – Sentiment : 768 → 256 → 3   (neg / neu / pos)
        Task 2 – FLS       : 768 → 256 → 2   (not-FLS / forward-looking)
        Task 3 – ESG       : 768 → 256 → 4   (E / S / G / None)

    Args:
        finbert_name : HuggingFace model name for the semantic encoder
        d_model      : FinBERT hidden size (768 for bert-base)
        ch_dim       : channel bottleneck dimension (32 recommended)
        tasks        : tuple of task names to activate.
                       Choose any subset of ("sentiment", "fls", "esg").
                       Inactive tasks still have heads but return None outputs.
    """

    def __init__(
        self,
        finbert_name: str = "ProsusAI/finbert",
        d_model: int = 768,
        ch_dim: int = 32,
        tasks: tuple = ("sentiment", "fls", "esg"),
    ):
        super().__init__()

        self.d_model  = d_model
        self.ch_dim   = ch_dim
        self.tasks    = set(tasks)   # e.g. {"sentiment", "fls", "esg"}
        self.channels = Channels()

        # ── Semantic encoder (FinBERT) ─────────────────────────────────────
        self.semantic_encoder = AutoModel.from_pretrained(finbert_name)

        # ── Channel encoder: 768 → ch_dim ─────────────────────────────────
        self.channel_encoder = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.GELU(),
            nn.Linear(256, ch_dim),
        )

        # ── Channel decoder: ch_dim → 768 ─────────────────────────────────
        self.channel_decoder = ChannelDecoder(ch_dim, 256, d_model)

        # ── Task Head 1: Sentiment (3 classes) ────────────────────────────
        self.sentiment_head = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, 3),
        )

        # ── Task Head 2: FLS (2 classes) ──────────────────────────────────
        self.fls_head = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, 3),
        )

        # ── Task Head 3: ESG (4 classes) ──────────────────────────────────
        self.esg_head = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, 4),
        )

    # ── Freeze / unfreeze helpers ──────────────────────────────────────────

    def freeze_finbert(self):
        """Freeze ALL FinBERT layers."""
        for p in self.semantic_encoder.parameters():
            p.requires_grad = False

    def unfreeze_finbert_last_n(self, n: int = 2):
        """
        Unfreeze the last N encoder layers of FinBERT (layers 12-N to 12).
        Everything else stays frozen.
        """
        # First freeze everything
        self.freeze_finbert()
        # Then unfreeze last n transformer layers
        encoder_layers = self.semantic_encoder.encoder.layer
        for layer in encoder_layers[-n:]:
            for p in layer.parameters():
                p.requires_grad = True

    # ── Masked mean pool ───────────────────────────────────────────────────

    def masked_mean_pool(
        self,
        hidden_states: torch.Tensor,   # [B, L, 768]
        attention_mask: torch.Tensor,  # [B, L]
    ) -> torch.Tensor:                 # [B, 768]
        mask = attention_mask.unsqueeze(-1).float()          # [B, L, 1]
        sum_hidden = (hidden_states * mask).sum(dim=1)       # [B, 768]
        sum_mask   = mask.sum(dim=1).clamp(min=1e-9)         # [B, 1]
        return sum_hidden / sum_mask                         # [B, 768]

    # ── Forward pass ──────────────────────────────────────────────────────

    def forward(
        self,
        input_ids: torch.Tensor,        # [B, L]
        attention_mask: torch.Tensor,   # [B, L]
        n_var: float,
        channel: str = "AWGN",
    ):
        """
        Returns:
            sentiment_logits : [B, 3]  or None if "sentiment" not in self.tasks
            fls_logits       : [B, 2]  or None if "fls" not in self.tasks
            esg_logits       : [B, 4]  or None if "esg" not in self.tasks
            H_hat            : [B, L, 768]  channel decoder output
            H                : [B, L, 768]  FinBERT output (for recon loss)
        """
        # ── Transmitter ───────────────────────────────────────────────────
        # FinBERT encoding
        bert_out = self.semantic_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        H = bert_out.last_hidden_state          # [B, L, 768]  ← save for recon loss

        # Channel encoder + power normalisation
        Tx_sig = self.channel_encoder(H)        # [B, L, ch_dim]
        Tx_sig = PowerNormalize(Tx_sig)

        # ── Wireless channel ──────────────────────────────────────────────
        if channel == "AWGN":
            Rx_sig = self.channels.AWGN(Tx_sig, n_var)
        elif channel == "Rayleigh":
            Rx_sig = self.channels.Rayleigh(Tx_sig, n_var)
        elif channel == "Rician":
            Rx_sig = self.channels.Rician(Tx_sig, n_var)
        else:
            raise ValueError(f"Unknown channel: {channel}. "
                             f"Choose from AWGN, Rayleigh, Rician.")

        # ── Receiver: shared path ─────────────────────────────────────────
        H_hat  = self.channel_decoder(Rx_sig)   # [B, L, 768]
        pooled = self.masked_mean_pool(H_hat, attention_mask)  # [B, 768]

        # ── Task heads (only active tasks run) ────────────────────────────
        sentiment_logits = self.sentiment_head(pooled) if "sentiment" in self.tasks else None
        fls_logits       = self.fls_head(pooled)       if "fls"       in self.tasks else None
        esg_logits       = self.esg_head(pooled)       if "esg"       in self.tasks else None

        return sentiment_logits, fls_logits, esg_logits, H_hat, H