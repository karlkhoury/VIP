"""
tokenalloc/allocator.py
───────────────────────
Transmitter step 3 (CLAUDE.md §4): one network decides the task-token counts k_t
(and optionally power shares p_t) for all three tasks jointly.

Inputs: CLS (768), SNR (dB/10 -> 32), channel one-hot (3 -> 32), priority w (3),
optional per-task confidence (3, detached).

Priority path (monotonicity). A plain w -> 32 embedding inside the trunk cannot
guarantee that a task's score is non-decreasing in its own priority. So w enters
through a gain that is non-negative by construction:

    score_t = base_t(ctx) + gain_t(ctx) * w_t,     gain_t = softplus(.) >= 0

which is monotone in w_t for every sentence, SNR and channel, while the context
still decides HOW MUCH priority matters (e.g. little at high SNR).

Init: output layers zero => equal soft counts (token version of B1). The gain bias
starts at -5 (gain ~ 0.007), so the start is equal split for any w.

priority_mode="embed" is the plain design of the architecture figure, kept as a
comparison (Exp. 3): w -> 32 (ReLU) is concatenated into the trunk with CLS, SNR and
channel, and the count head reads the score directly. No monotonicity guarantee.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class Allocator(nn.Module):
    def __init__(self, hidden: int = 768, n_tasks: int = 3, max_tokens: int = 4,
                 use_sentence=True, use_snr=True, use_channel=True, use_confidence=False,
                 use_power=False, emb=32, width=128, priority_mode="gain", use_priority=True):
        super().__init__()
        if priority_mode not in ("gain", "embed"):
            raise ValueError(f"priority_mode must be 'gain' or 'embed', got {priority_mode!r}")
        self.n_tasks, self.max_tokens = n_tasks, max_tokens
        self.use_sentence, self.use_snr, self.use_channel = use_sentence, use_snr, use_channel
        self.use_confidence, self.use_power = use_confidence, use_power
        self.priority_mode = priority_mode
        self.use_priority = use_priority        # False: the allocator always sees equal priority
        self.snr_emb = nn.Sequential(nn.Linear(1, emb), nn.ReLU())
        self.ch_emb = nn.Sequential(nn.Linear(3, emb), nn.ReLU())
        self.prio_emb = nn.Sequential(nn.Linear(n_tasks, emb), nn.ReLU()) if priority_mode == "embed" else None
        in_dim = hidden + (3 if priority_mode == "embed" else 2) * emb + (n_tasks if use_confidence else 0)
        self.trunk = nn.Sequential(nn.Linear(in_dim, width), nn.ReLU())
        self.count_head = nn.Linear(width, n_tasks)
        self.gain_head = nn.Linear(width, n_tasks) if priority_mode == "gain" else None
        self.power_head = nn.Linear(width, n_tasks) if use_power else None
        for head in (self.count_head, self.gain_head, self.power_head):
            if head is not None:
                nn.init.zeros_(head.weight)
                nn.init.zeros_(head.bias)
        if self.gain_head is not None:
            nn.init.constant_(self.gain_head.bias, -5.0)

    def forward(self, cls, snr_db, channel_onehot, w, confidence=None):
        """Returns count scores [B, T] and power scores [B, T] (or None)."""
        B = cls.shape[0]
        if not self.use_priority:
            w = torch.full_like(w, 1.0 / w.shape[-1])
        parts = [
            cls if self.use_sentence else torch.zeros_like(cls),
            self.snr_emb((snr_db / 10.0).view(B, 1)) if self.use_snr
            else torch.zeros(B, self.snr_emb[0].out_features, device=cls.device),
            self.ch_emb(channel_onehot) if self.use_channel
            else torch.zeros(B, self.ch_emb[0].out_features, device=cls.device),
        ]
        if self.prio_emb is not None:
            parts.append(self.prio_emb(w))
        if self.use_confidence:
            parts.append(confidence.detach())
        h = self.trunk(torch.cat(parts, dim=-1))
        if self.priority_mode == "gain":
            scores = self.count_head(h) + F.softplus(self.gain_head(h)) * w
        else:
            scores = self.count_head(h)
        power = self.power_head(h) if self.use_power else None
        return scores, power


# ── Count decode (fixed math, no weights) ────────────────────────────────────

def soft_counts(scores: torch.Tensor, cap: int, max_tokens: int = 4) -> torch.Tensor:
    """c_t = 1 + 3 sigmoid(score); if sum c > cap, shrink the extras above the floor."""
    T = scores.shape[-1]
    c = 1.0 + (max_tokens - 1) * torch.sigmoid(scores)
    total = c.sum(-1, keepdim=True)
    shrink = (cap - T) / (total - T).clamp(min=1e-6)
    return torch.where(total > cap, 1.0 + (c - 1.0) * shrink, c)


def hard_counts(c: torch.Tensor, cap: int, w: torch.Tensor, max_tokens: int = 4) -> torch.Tensor:
    """
    Round to nearest; if that exceeds the cap use largest remainder (floor all,
    hand out the rest by largest fractional part, ties broken by priority).
    """
    c = c.detach()
    k = torch.round(c)
    over = k.sum(-1) > cap
    if over.any():
        fl = torch.floor(c[over])
        frac = c[over] - fl
        rest = (cap - fl.sum(-1)).long()
        # rank by fractional part, priority as a tiny tie-breaker
        order = torch.argsort(frac + 1e-6 * w[over], dim=-1, descending=True)
        rank = torch.argsort(order, dim=-1)
        fl = fl + (rank < rest.unsqueeze(-1)).float()
        k[over] = fl
    return k.clamp(1, c.new_tensor(float(max_tokens)))


def straight_through(c_soft: torch.Tensor, k_hard: torch.Tensor) -> torch.Tensor:
    return c_soft + (k_hard - c_soft).detach()


def token_masks(c_soft: torch.Tensor, k_hard: torch.Tensor, max_tokens: int = 4, tau: float = 0.3):
    """Keep the first k_t tokens. Forward value is the hard mask, gradient is the soft one."""
    j = torch.arange(1, max_tokens + 1, device=c_soft.device, dtype=c_soft.dtype)
    m_soft = torch.sigmoid((c_soft.unsqueeze(-1) - j + 0.5) / tau)            # [B, T, J]
    m_hard = (j <= k_hard.unsqueeze(-1)).to(c_soft.dtype)
    return m_soft + (m_hard - m_soft).detach()


def power_shares(power_scores, k):
    """p_t = K_used exp(b_t) / sum_j k_j exp(b_j)  =>  sum_t k_t p_t = K_used."""
    if power_scores is None:
        return torch.ones_like(k)
    k_used = k.sum(-1, keepdim=True)
    e = torch.exp(power_scores - power_scores.max(-1, keepdim=True).values)
    return k_used * e / (k * e).sum(-1, keepdim=True)


def normalized_entropy(logits: torch.Tensor) -> torch.Tensor:
    p = F.softmax(logits, dim=-1)
    return -(p * torch.log(p.clamp_min(1e-12))).sum(-1) / math.log(logits.shape[-1])
