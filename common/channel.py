"""
common/channel.py
─────────────────
Physical layer shared by both pipelines (CLAUDE.md §4 steps 7-8).

  y = h * x + n,   n ~ CN(0, sigma^2),   sigma^2 = 10^(-SNR/10)

  AWGN      h = 1
  Rayleigh  h ~ CN(0, 1)
  Rician    h = sqrt(K/(K+1)) e^{j theta} + sqrt(1/(K+1)) CN(0, 1),  K = 4

All channels have E|h|^2 = 1. Fading is i.i.d. per complex symbol (as in the prior
paper, §II-A). The receiver knows h perfectly and applies zero forcing, x_hat = y / h.

Symbols are complex tensors. A transmitter is expected to send average energy 1 per
symbol, so SNR is the per-symbol SNR.

Bug fixed vs. the old models/fin_transceiver.py: the old Rayleigh/Rician used a
REAL-valued h per real dimension and divided by clamp(h, min=1e-6). Every negative h
(50% for Rayleigh, 8% for Rician K=1) became 1e-6 and amplified that entry by ~1e6.
The old Rician also used K=1 instead of K=4.
"""

import hashlib
import math

import torch

CHANNELS = ("AWGN", "Rayleigh", "Rician")
RICIAN_K = 4.0


def channel_index(name: str) -> int:
    if name not in CHANNELS:
        raise ValueError(f"Unknown channel {name!r}; choose from {CHANNELS}")
    return CHANNELS.index(name)


def snr_db_to_noise_var(snr_db: torch.Tensor) -> torch.Tensor:
    """Complex noise variance sigma^2 for unit-energy symbols."""
    return torch.pow(10.0, -snr_db / 10.0)


# ── Real <-> complex packing ─────────────────────────────────────────────────

def real_to_complex(x: torch.Tensor) -> torch.Tensor:
    """[..., 2m] reals -> [..., m] complex (first half real, second half imag)."""
    m = x.shape[-1] // 2
    return torch.complex(x[..., :m], x[..., m:])


def complex_to_real(z: torch.Tensor) -> torch.Tensor:
    """[..., m] complex -> [..., 2m] reals. Inverse of real_to_complex."""
    return torch.cat([z.real, z.imag], dim=-1)


def normalize_energy(z: torch.Tensor, mask: torch.Tensor = None, eps: float = 1e-8) -> torch.Tensor:
    """
    Scale each block (last dim) to average energy 1 per symbol.
    If `mask` [...] is given (1 = valid), the average is taken over valid blocks
    along dim -2 as well, i.e. each sentence gets energy 1 per sent symbol.
    """
    energy = z.real ** 2 + z.imag ** 2
    if mask is None:
        return z / torch.sqrt(energy.mean(dim=-1, keepdim=True) + eps)
    m = mask.unsqueeze(-1).to(energy.dtype)
    per_sent = (energy * m).sum(dim=(-2, -1), keepdim=True) / (
        m.sum(dim=(-2, -1), keepdim=True) * energy.shape[-1] + eps)
    return z / torch.sqrt(per_sent + eps)


# ── Random draws ─────────────────────────────────────────────────────────────

def _cn(shape, generator=None) -> torch.Tensor:
    """CN(0, 1) samples on CPU (deterministic under a CPU generator)."""
    re = torch.randn(shape, generator=generator)
    im = torch.randn(shape, generator=generator)
    return torch.complex(re, im) / math.sqrt(2.0)


def sample_fading(shape, channel: str, generator=None, rician_k: float = RICIAN_K) -> torch.Tensor:
    """Per-symbol fading coefficients h, complex, E|h|^2 = 1."""
    if channel == "AWGN":
        return torch.ones(shape, dtype=torch.complex64)
    if channel == "Rayleigh":
        return _cn(shape, generator)
    if channel == "Rician":
        theta = torch.rand(shape, generator=generator) * (2 * math.pi)
        los = math.sqrt(rician_k / (rician_k + 1)) * torch.exp(torch.complex(torch.zeros(shape), theta))
        return los + math.sqrt(1.0 / (rician_k + 1)) * _cn(shape, generator)
    raise ValueError(f"Unknown channel {channel!r}; choose from {CHANNELS}")


class Realization:
    """
    One draw of fading h and unit-variance noise n0 for a fixed symbol grid.
    Drawing both up front lets every method see the SAME h and noise on the same
    symbol slot (required for paired bootstrap, CLAUDE.md §6).
    """

    def __init__(self, shape, channel: str, generator=None, device="cpu"):
        self.channel = channel
        self.h = sample_fading(shape, channel, generator).to(device)
        self.n0 = _cn(shape, generator).to(device)

    def apply(self, x: torch.Tensor, snr_db: torch.Tensor) -> torch.Tensor:
        """
        Send x through the channel and zero-force it. snr_db is [B] (per sentence);
        it is broadcast over the remaining dims of x. Returns x_hat = y / h.
        """
        sigma = torch.sqrt(snr_db_to_noise_var(snr_db)).to(x.device)
        sigma = sigma.view(-1, *([1] * (x.dim() - 1)))
        if self.h.shape != x.shape:
            raise ValueError(f"Realization shape {tuple(self.h.shape)} != signal shape {tuple(x.shape)}")
        y = self.h * x + sigma * self.n0
        return zero_forcing(y, self.h)


def zero_forcing(y: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
    """ZF equalizer with perfect CSI at the receiver."""
    return y / h


def transmit(x: torch.Tensor, channel: str, snr_db: torch.Tensor, generator=None) -> torch.Tensor:
    """Convenience: fresh realization for x's shape, then y / h."""
    return Realization(tuple(x.shape), channel, generator, device=x.device).apply(x, snr_db)


# ── Fixed evaluation seeds ───────────────────────────────────────────────────

def eval_generator(noise_seed: int, channel: str, snr_db: float, batch_idx: int) -> torch.Generator:
    """
    Deterministic CPU generator for one (channel, SNR, batch) cell. Identical for
    every method and every training seed, so all methods face the same channel.
    """
    key = f"{noise_seed}|{channel}|{float(snr_db):.3f}|{batch_idx}".encode()
    seed = int.from_bytes(hashlib.sha256(key).digest()[:8], "little") & ((1 << 63) - 1)
    g = torch.Generator()
    g.manual_seed(seed)
    return g
