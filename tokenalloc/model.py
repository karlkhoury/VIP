"""
tokenalloc/model.py
───────────────────
The full encoder-side task-token allocation system (CLAUDE.md §4, steps 1-12).

    encode()              steps 1-2   BERT + task tokens            (once per batch)
    allocate()            step 3      allocator -> counts, power
    transmit_decode()     steps 4-12  select, per-task channel enc, channel, ZF,
                                      per-task channel dec, masked attention pooling,
                                      linear heads

encode() does not depend on SNR, channel or policy, so evaluation runs it once per
batch and then tries every (channel, SNR, policy) on the cached outputs. Baselines
swap only the counts that go into transmit_decode(), never the pipeline.

Symbol grid: every sentence has T x J = 3 x 4 task-token slots x L = 4 complex
symbols. A channel Realization is drawn for the full grid, so a slot sees the same
fade and noise whichever method sends it. Unsent slots are masked out: they reach
nothing at the receiver (the header says they were not sent).
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from common.channel import CHANNELS, Realization, complex_to_real, normalize_energy, real_to_complex
from common.data import NUM_CLASSES, TASKS
from tokenalloc.allocator import (Allocator, hard_counts, normalized_entropy, power_shares,
                                  soft_counts, straight_through, token_masks)
from tokenalloc.encoder import TaskTokenEncoder

SYMBOLS_PER_TOKEN = 4          # L: complex symbols per task token (= 8 reals)


class PerTaskMLP(nn.Module):
    """One small network per task, shared by all tokens of that task."""

    def __init__(self, n_tasks, d_in, d_mid, d_out):
        super().__init__()
        self.nets = nn.ModuleList(
            nn.Sequential(nn.Linear(d_in, d_mid), nn.ReLU(), nn.Linear(d_mid, d_out))
            for _ in range(n_tasks))

    def forward(self, x):                                  # x [B, T, J, d_in]
        return torch.stack([net(x[:, t]) for t, net in enumerate(self.nets)], dim=1)


class MaskedAttentionPool(nn.Module):
    """
    Step 11: one learned query per task; score_j = q_t . r_j / sqrt(H); unsent
    slots excluded. Written as weights ∝ m_j exp(score_j) so the forward pass equals
    -inf masking for a hard 0/1 mask while d/dm_j stays informative for training.
    """

    def __init__(self, n_tasks, hidden):
        super().__init__()
        self.query = nn.Parameter(torch.randn(n_tasks, hidden) * 0.02)
        self.scale = 1.0 / math.sqrt(hidden)

    def forward(self, r, m):                               # r [B, T, J, H], m [B, T, J]
        s = torch.einsum("btjh,th->btj", r, self.query) * self.scale
        # max over SENT slots only; an unsent slot (possibly in a deep fade, huge after
        # ZF) must not shift the softmax. Clamp keeps its soft-gradient term finite.
        s_max = s.masked_fill(m <= 0.5, float("-inf")).max(-1, keepdim=True).values
        e = torch.exp((s - s_max).clamp(max=20.0)) * m
        a = e / e.sum(-1, keepdim=True).clamp_min(1e-12)
        return torch.einsum("btj,btjh->bth", a, r)


class TokenAllocSystem(nn.Module):
    def __init__(self, bert, alloc_cfg: dict, tokens_per_task: int = 4, max_tokens_per_task: int = None):
        super().__init__()
        self.T = len(TASKS)
        self.J = tokens_per_task                    # task tokens each task OWNS (slots)
        # most tokens one task may SEND (e.g. cap/2); defaults to all it owns
        self.Jmax = min(max_tokens_per_task or tokens_per_task, tokens_per_task)
        self.encoder = TaskTokenEncoder(bert, self.T, self.J)
        H = self.encoder.hidden
        self.allocator = Allocator(
            hidden=H, n_tasks=self.T, max_tokens=self.Jmax,
            use_sentence=alloc_cfg.get("use_sentence", True),
            use_snr=alloc_cfg.get("use_snr", True),
            use_channel=alloc_cfg.get("use_channel", True),
            use_confidence=alloc_cfg.get("use_confidence", False),
            use_power=alloc_cfg.get("use_power", False),
            priority_mode=alloc_cfg.get("priority_mode", "gain"),
            use_priority=alloc_cfg.get("use_priority", True))
        self.tau = alloc_cfg.get("tau", 0.3)
        self.ch_encoder = PerTaskMLP(self.T, H, 256, 2 * SYMBOLS_PER_TOKEN)   # 768 -> 256 -> 8
        self.ch_decoder = PerTaskMLP(self.T, 2 * SYMBOLS_PER_TOKEN, 256, H)   # 8 -> 256 -> 768
        self.pool = MaskedAttentionPool(self.T, H)
        self.heads = nn.ModuleList(nn.Linear(H, NUM_CLASSES[t]) for t in TASKS)

    # ── steps 1-2 ────────────────────────────────────────────────────────────
    def encode(self, input_ids, attention_mask):
        return self.encoder(input_ids, attention_mask)

    def task_logits(self, pooled):                          # pooled [B, T, H]
        return [head(pooled[:, t]) for t, head in enumerate(self.heads)]

    @torch.no_grad()
    def confidence(self, task_out):
        """
        Normalized entropy of each head on the CLEAN task tokens (all 4 sent, no channel):
        channel encoder -> unit energy -> channel decoder -> pooling -> head, i.e. exactly
        what the receiver would see over a perfect channel. The heads were trained on
        decoder outputs, so they are applied to decoder outputs here too. Detached.
        """
        m = torch.ones(task_out.shape[:3], device=task_out.device)
        x = normalize_energy(real_to_complex(self.ch_encoder(task_out)))
        r = self.ch_decoder(complex_to_real(x))
        logits = self.task_logits(self.pool(r, m))
        return torch.stack([normalized_entropy(l) for l in logits], dim=-1)

    # ── step 3 ───────────────────────────────────────────────────────────────
    def allocate(self, cls, task_out, snr_db, channel: str, w, cap: int):
        B = cls.shape[0]
        onehot = F.one_hot(torch.full((B,), CHANNELS.index(channel), device=cls.device),
                           len(CHANNELS)).float()
        conf = self.confidence(task_out) if self.allocator.use_confidence else None
        scores, power_scores = self.allocator(cls, snr_db, onehot, w, conf)
        c = soft_counts(scores, cap, self.Jmax)
        k = hard_counts(c, cap, w, self.Jmax)
        return {"c_soft": c, "k_hard": k, "k_st": straight_through(c, k), "power_scores": power_scores}

    # ── steps 4-12 ───────────────────────────────────────────────────────────
    def transmit_decode(self, task_out, snr_db, realization: Realization, k_hard,
                        c_soft=None, power_scores=None):
        """
        k_hard [B, T] integer counts. With c_soft given, masks and power carry
        straight-through gradients to the allocator; without it (baselines,
        nested dropout) the masks are plain 0/1.
        """
        if c_soft is None:
            j = torch.arange(1, self.J + 1, device=task_out.device)
            m = (j <= k_hard.unsqueeze(-1)).float()
            k_for_power = k_hard
        else:
            m = token_masks(c_soft, k_hard, self.J, self.tau)
            k_for_power = straight_through(c_soft, k_hard)
        p = power_shares(power_scores, k_for_power)                           # [B, T]

        x = real_to_complex(self.ch_encoder(task_out))                        # [B, T, J, 4] complex
        x = normalize_energy(x) * torch.sqrt(p)[..., None, None]
        x_hat = realization.apply(x, snr_db)                                  # ZF output
        r = self.ch_decoder(complex_to_real(x_hat))                           # [B, T, J, H]
        pooled = self.pool(r, m)
        return {"logits": self.task_logits(pooled), "k": k_hard, "power": p,
                "symbols": k_hard.sum(-1) * SYMBOLS_PER_TOKEN}

    def realization(self, batch_size, channel, generator=None, device="cpu"):
        return Realization((batch_size, self.T, self.J, SYMBOLS_PER_TOKEN), channel, generator, device)
