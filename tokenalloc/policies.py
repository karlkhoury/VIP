"""
tokenalloc/policies.py
──────────────────────
Allocation baselines (CLAUDE.md §7). Each returns hard counts k [B, 3] in {1..4}.
They are plugged into the SAME trained TokenAllocSystem; only the counts change.

  equal          fixed equal split at total K: 6 -> 2/2/2, 8 -> 3/3/2, 10 -> 4/3/3, 12 -> 4/4/4
  fixed:a-b-c    any fixed split
  proportional   round(w * K) by largest remainder, each task in [1, 4]
  snr_only       U-DeepSC-style: greedy on MEAN validation loss curves at this SNR,
                 same counts for every sentence (no sentence input)
  random_matched random split with the allocator's per-sentence total
  oracle_greedy  Fox (1966) greedy marginal allocation on this sentence's measured
                 loss curves (uses test labels: an oracle, not a deployable method)
  oracle_exhaustive  exact minimum of sum_t w_t CE_t(k_t) + lambda * sum k over all
                 4^3 splits within the cap (checks how close greedy is)
"""

import itertools

import torch


def equal_split(cap: int, batch_size: int, n_tasks=3, max_tokens=4, device="cpu"):
    base, rem = divmod(min(cap, n_tasks * max_tokens), n_tasks)
    k = [base + (1 if t < rem else 0) for t in range(n_tasks)]
    return torch.tensor(k, dtype=torch.float32, device=device).expand(batch_size, n_tasks).clone()


def fixed_split(split, batch_size, device="cpu"):
    return torch.tensor(split, dtype=torch.float32, device=device).expand(batch_size, len(split)).clone()


def proportional(w: torch.Tensor, cap: int, max_tokens=4):
    """Start at 1 each, hand out the rest one by one to the largest w_t K - k_t."""
    B, T = w.shape
    k = torch.ones_like(w)
    target = w * cap
    for _ in range(min(cap, T * max_tokens) - T):
        deficit = (target - k).masked_fill(k >= max_tokens, float("-inf"))
        k[torch.arange(B), deficit.argmax(-1)] += 1
    return k


def greedy_from_curves(ce: torch.Tensor, w: torch.Tensor, cap: int, lam: float = 0.0):
    """
    ce [B, T, J]: loss of task t when it sends j+1 tokens. Start at 1 token each,
    give the next token to the task with the largest w_t * (loss reduction); stop
    when the best gain is not above the price lam, or the cap is reached.
    """
    B, T, J = ce.shape
    k = torch.ones(B, T, dtype=torch.long, device=ce.device)
    active = torch.ones(B, dtype=torch.bool, device=ce.device)
    ar = torch.arange(B, device=ce.device)
    for _ in range(min(cap, T * J) - T):
        cur = ce.gather(2, (k - 1).unsqueeze(-1)).squeeze(-1)
        nxt = ce.gather(2, k.clamp(max=J - 1).unsqueeze(-1)).squeeze(-1)
        gain = (w * (cur - nxt)).masked_fill(k >= J, float("-inf"))
        best, t = gain.max(-1)
        active &= best > lam
        k[ar[active], t[active]] += 1
    return k.float()


def exhaustive_from_curves(ce: torch.Tensor, w: torch.Tensor, cap: int, lam: float = 0.0):
    B, T, J = ce.shape
    combos = torch.tensor([c for c in itertools.product(range(1, J + 1), repeat=T) if sum(c) <= cap],
                          device=ce.device)                                          # [C, T]
    idx = (combos - 1).T.unsqueeze(0).expand(B, T, len(combos))                      # [B, T, C]
    loss = (ce.gather(2, idx) * w.unsqueeze(-1)).sum(1) + lam * combos.sum(-1).float()
    return combos[loss.argmin(-1)].float()


def random_matched(totals: torch.Tensor, generator: torch.Generator, n_tasks=3, max_tokens=4):
    """Random composition of each sentence's total into n_tasks parts in [1, max_tokens]."""
    B = totals.shape[0]
    k = torch.ones(B, n_tasks)
    for b in range(B):
        for _ in range(int(totals[b].item()) - n_tasks):
            free = (k[b] < max_tokens).nonzero().flatten()
            k[b, free[torch.randint(len(free), (1,), generator=generator)]] += 1
    return k.to(totals.device)


def parse_policy(name: str):
    """'fixed:3-3-2' -> ('fixed', (3,3,2)); other names -> (name, None)."""
    if name.startswith("fixed:"):
        return "fixed", tuple(int(x) for x in name.split(":")[1].split("-"))
    return name, None
