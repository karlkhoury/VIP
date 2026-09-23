"""
legacy/model.py
───────────────
The prior paper's FinBERT-DeepSC testbed (TP-SRA, Fig. 1), kept only as a baseline:

    BERT -> channel encoder (768 -> 256 -> ch_dim reals) -> power norm -> channel
         -> channel decoder (ch_dim -> 768) -> masked mean pool -> readout -> 3 linear heads

Readouts (paper §III-D/E):
    shared      B0  symmetric sharing, m_t = 1
    equal_split B1  disjoint d/T indicator masks
    softmask    B3  SoftMaskGate: MLP(SNR) -> delta in R^{T x d}, m_t = clip[0,2](1 + delta_t),
                    output layer zero-initialised (identity at init), ~150K params

Rebuilt from the old models/fin_transceiver.py (codec, decoder) and the paper text
(SoftMaskGate, linear heads). The SoftMaskGate/B1 code of the paper was not in the
repo, so B1/B3 follow the paper's description.

Changes vs. the old code (bug fixes only):
  * channel: complex symbols, i.i.d. per-symbol fading, Rician K=4, full complex ZF
    (common.channel). The old code used real h with clamp(min=1e-6).
  * power normalisation per sentence over real tokens; padding is not transmitted.
  * encoder checkpoint: ProsusAI/finbert is refused (PhraseBank leakage).

tx_mode:
    per_token  every word token is sent: ch_dim/2 complex symbols per word token (as
               the old code did; ~16 x sentence length symbols per sentence)
    pooled     BERT output is mean-pooled first and ONE vector of ch_dim/2 symbols is
               sent (ch_dim = 64 gives exactly 32 symbols per sentence, the budget of
               the new pipeline at cap 8)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from common.channel import complex_to_real, normalize_energy, real_to_complex
from common.data import NUM_CLASSES, TASKS


class ChannelDecoder(nn.Module):
    """Two-block residual MLP, ch_dim -> 768 (unchanged from the old code)."""

    def __init__(self, in_dim: int, mid_dim: int, out_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, mid_dim)
        self.fc2 = nn.Linear(mid_dim, out_dim)
        self.fc3 = nn.Linear(out_dim, mid_dim)
        self.ln1 = nn.LayerNorm(mid_dim)
        self.fc4 = nn.Linear(mid_dim, out_dim)
        self.fc5 = nn.Linear(out_dim, mid_dim)
        self.ln2 = nn.LayerNorm(mid_dim)
        self.drop = nn.Dropout(0.1)
        self.proj = nn.Linear(mid_dim, out_dim)

    def forward(self, x):
        x1 = self.fc1(x)
        out1 = self.drop(self.ln1(x1 + self.fc3(F.gelu(self.fc2(F.gelu(x1))))))
        out2 = self.ln2(out1 + self.fc5(F.gelu(self.fc4(F.gelu(out1)))))
        return self.proj(out2)


class SoftMaskGate(nn.Module):
    """B3: m_t(gamma) = clip[0, M](1 + delta_t(gamma)), zero-init output (identity at init)."""

    def __init__(self, d: int, n_tasks: int = 3, hidden: int = 64, max_gain: float = 2.0):
        super().__init__()
        self.n_tasks, self.d, self.max_gain = n_tasks, d, max_gain
        self.net = nn.Sequential(nn.Linear(1, hidden), nn.ReLU(), nn.Linear(hidden, n_tasks * d))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, snr_db):                                  # [B] -> [B, T, d]
        delta = self.net((snr_db / 10.0).unsqueeze(-1)).view(-1, self.n_tasks, self.d)
        return torch.clamp(1.0 + delta, 0.0, self.max_gain)


class FinDeepSC(nn.Module):
    def __init__(self, bert, ch_dim: int = 32, readout: str = "shared", tx_mode: str = "per_token"):
        super().__init__()
        if ch_dim % 2:
            raise ValueError("ch_dim must be even (pairs of reals form complex symbols)")
        if readout not in ("shared", "equal_split", "softmask"):
            raise ValueError(f"unknown readout {readout!r}")
        if tx_mode not in ("per_token", "pooled"):
            raise ValueError(f"unknown tx_mode {tx_mode!r}")
        self.bert = bert
        d = bert.config.hidden_size
        self.d, self.ch_dim, self.readout, self.tx_mode = d, ch_dim, readout, tx_mode
        self.T = len(TASKS)
        self.channel_encoder = nn.Sequential(nn.Linear(d, 256), nn.GELU(), nn.Linear(256, ch_dim))
        self.channel_decoder = ChannelDecoder(ch_dim, 256, d)
        self.heads = nn.ModuleList(nn.Linear(d, NUM_CLASSES[t]) for t in TASKS)
        self.gate = SoftMaskGate(d, self.T) if readout == "softmask" else None
        if readout == "equal_split":
            split = torch.zeros(self.T, d)
            for t, chunk in enumerate(torch.arange(d).chunk(self.T)):
                split[t, chunk] = 1.0
            self.register_buffer("split_masks", split, persistent=False)

    @staticmethod
    def masked_mean(x, mask):
        m = mask.unsqueeze(-1).float()
        return (x * m).sum(1) / m.sum(1).clamp(min=1e-9)

    def symbol_shape(self, input_ids):
        B, L = input_ids.shape
        return (B, L, self.ch_dim // 2) if self.tx_mode == "per_token" else (B, 1, self.ch_dim // 2)

    def symbols_per_sentence(self, attention_mask):
        n = attention_mask.sum(-1) if self.tx_mode == "per_token" else torch.ones_like(attention_mask[:, 0])
        return n * (self.ch_dim // 2)

    def forward(self, input_ids, attention_mask, snr_db, realization):
        H = self.bert(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        if self.tx_mode == "per_token":
            src, tx_mask = H, attention_mask
        else:
            src, tx_mask = self.masked_mean(H, attention_mask).unsqueeze(1), attention_mask[:, :1]
        x = real_to_complex(self.channel_encoder(src))                    # [B, L', ch_dim/2]
        x = normalize_energy(x, mask=tx_mask) * tx_mask.unsqueeze(-1)     # pads are not sent
        x_hat = realization.apply(x, snr_db) * tx_mask.unsqueeze(-1)
        H_hat = self.channel_decoder(complex_to_real(x_hat))              # [B, L', d]
        z = self.masked_mean(H_hat, tx_mask)                              # [B, d]
        if self.readout == "shared":
            zt = z.unsqueeze(1).expand(-1, self.T, -1)
        elif self.readout == "equal_split":
            zt = z.unsqueeze(1) * self.split_masks
        else:
            zt = z.unsqueeze(1) * self.gate(snr_db)
        logits = [head(zt[:, t]) for t, head in enumerate(self.heads)]
        recon = (H_hat, H) if self.tx_mode == "per_token" else None
        return {"logits": logits, "recon": recon,
                "symbols": self.symbols_per_sentence(attention_mask)}
