"""
tests/test_all.py — unit and smoke tests (offline, CPU, tiny random BERT).

    python -m pytest tests -q          or          python -m tests.test_all

The smoke tests print the tensor shapes of one forward + backward pass, as
CLAUDE.md §10 asks after every step.
"""


import torch

from common.channel import (Realization, complex_to_real, eval_generator, normalize_energy,
                            real_to_complex, sample_fading, snr_db_to_noise_var)
from common.tiny import tiny_bert_and_tokenizer
from tokenalloc.allocator import (Allocator, hard_counts, power_shares, soft_counts,
                                  straight_through, token_masks)
from tokenalloc.policies import (equal_split, exhaustive_from_curves, greedy_from_curves,
                                 proportional)

N = 400_000


# ── Channel (task 2: Rician bug check) ───────────────────────────────────────

def test_fading_unit_power():
    for ch in ("AWGN", "Rayleigh", "Rician"):
        h = sample_fading((N,), ch, torch.Generator().manual_seed(0))
        assert abs((h.abs() ** 2).mean().item() - 1.0) < 0.01, ch


def _post_zf_mse(ch, snr_db):
    g = torch.Generator().manual_seed(1)
    x = normalize_energy(torch.complex(torch.randn(N // 4, 4, generator=g), torch.randn(N // 4, 4, generator=g)))
    real = Realization(tuple(x.shape), ch, g)
    x_hat = real.apply(x, torch.full((x.shape[0],), float(snr_db)))
    return ((x_hat - x).abs() ** 2).median().item()


def test_rician_beats_rayleigh_and_awgn_exact():
    for snr in (0.0, 10.0):
        awgn, ric, ray = (_post_zf_mse(c, snr) for c in ("AWGN", "Rician", "Rayleigh"))
        assert awgn < ric < ray, (snr, awgn, ric, ray)       # physics: LOS helps
    # AWGN: E|n|^2 = sigma^2 exactly
    g = torch.Generator().manual_seed(2)
    x = torch.zeros(1, N, dtype=torch.complex64)
    y = Realization((1, N), "AWGN", g).apply(x, torch.tensor([5.0]))
    assert abs((y.abs() ** 2).mean().item() / snr_db_to_noise_var(torch.tensor(5.0)).item() - 1) < 0.02


def test_fixed_noise_is_shared():
    a = Realization((4, 3, 4, 4), "Rician", eval_generator(1234, "Rician", 5, 7))
    b = Realization((4, 3, 4, 4), "Rician", eval_generator(1234, "Rician", 5, 7))
    c = Realization((4, 3, 4, 4), "Rician", eval_generator(1234, "Rician", 5, 8))
    assert torch.equal(a.h, b.h) and torch.equal(a.n0, b.n0) and not torch.equal(a.h, c.h)


def test_complex_packing_roundtrip():
    x = torch.randn(2, 5, 8)
    assert torch.allclose(complex_to_real(real_to_complex(x)), x)
    z = normalize_energy(real_to_complex(x))
    assert torch.allclose((z.abs() ** 2).mean(-1), torch.ones(2, 5), atol=1e-5)


# ── Allocator decode ─────────────────────────────────────────────────────────

def test_count_decode_rules():
    torch.manual_seed(0)
    w = torch.softmax(torch.randn(512, 3), -1)
    for cap in (6, 8, 10, 12):
        c = soft_counts(torch.randn(512, 3) * 3, cap)
        k = hard_counts(c, cap, w)
        assert (c >= 1).all() and (c <= 4).all() and (c.sum(-1) <= cap + 1e-4).all()
        assert (k >= 1).all() and (k <= 4).all() and (k.sum(-1) <= cap).all()
        assert torch.equal(k, k.round())


def test_straight_through_and_masks():
    scores = torch.zeros(2, 3, requires_grad=True)
    c = soft_counts(scores, 8)
    k = hard_counts(c, 8, torch.full((2, 3), 1 / 3))
    m = token_masks(c, k)
    assert torch.equal(m, (torch.arange(1, 5) <= k.unsqueeze(-1)).float())   # forward = hard
    (m * torch.arange(4.0)).sum().backward()
    assert scores.grad.abs().sum() > 0                                          # backward = soft
    ks = straight_through(c, k)
    assert torch.equal(ks, k)


def test_allocator_init_equal_split_and_priority_monotone():
    torch.manual_seed(0)
    alloc = Allocator(hidden=32)
    cls, snr, oh = torch.randn(4, 32), torch.zeros(4), torch.eye(3)[[0, 1, 2, 0]]
    s0, _ = alloc(cls, snr, oh, torch.full((4, 3), 1 / 3))
    assert torch.allclose(s0, s0[:, :1].expand_as(s0))                          # equal at init
    # after random training-like perturbation, score_t is non-decreasing in w_t
    for p in alloc.parameters():
        p.data += torch.randn_like(p) * 0.5
    w_lo, w_hi = torch.tensor([[0.2, 0.4, 0.4]] * 4), torch.tensor([[0.8, 0.1, 0.1]] * 4)
    assert (alloc(cls, snr, oh, w_hi)[0][:, 0] >= alloc(cls, snr, oh, w_lo)[0][:, 0]).all()


def test_allocator_embed_mode():
    """The figure's plain design (w -> 32 inside the trunk): equal at init, w changes scores."""
    torch.manual_seed(0)
    alloc = Allocator(hidden=32, priority_mode="embed")
    assert alloc.gain_head is None and alloc.trunk[0].in_features == 32 + 3 * 32
    cls, snr, oh = torch.randn(4, 32), torch.zeros(4), torch.eye(3)[[0, 1, 2, 0]]
    s0, _ = alloc(cls, snr, oh, torch.full((4, 3), 1 / 3))
    assert torch.allclose(s0, torch.zeros_like(s0))                              # equal at init
    for p in alloc.parameters():
        p.data += torch.randn_like(p) * 0.5
    w_lo, w_hi = torch.tensor([[0.2, 0.4, 0.4]] * 4), torch.tensor([[0.8, 0.1, 0.1]] * 4)
    assert not torch.allclose(alloc(cls, snr, oh, w_hi)[0], alloc(cls, snr, oh, w_lo)[0])


def test_power_budget():
    k = torch.tensor([[2.0, 3.0, 3.0], [1.0, 1.0, 4.0]])
    p = power_shares(torch.randn(2, 3), k)
    assert torch.allclose((k * p).sum(-1), k.sum(-1))


# ── Baselines ────────────────────────────────────────────────────────────────

def test_baselines():
    assert equal_split(6, 1).tolist() == [[2, 2, 2]]
    assert equal_split(8, 1).tolist() == [[3, 3, 2]]
    assert equal_split(10, 1).tolist() == [[4, 3, 3]]
    assert equal_split(12, 1).tolist() == [[4, 4, 4]]
    k = proportional(torch.tensor([[0.1, 0.1, 0.8]]), 8)
    assert k.tolist() == [[2, 2, 4]] and k.sum() == 8
    torch.manual_seed(0)
    ce = torch.sort(torch.rand(64, 3, 4), dim=-1, descending=True).values      # decreasing curves
    w = torch.softmax(torch.randn(64, 3), -1)
    for cap in (6, 8, 12):
        kg, ke = greedy_from_curves(ce, w, cap), exhaustive_from_curves(ce, w, cap)
        loss = lambda k: (ce.gather(2, (k.long() - 1).unsqueeze(-1)).squeeze(-1) * w).sum(-1)
        assert (loss(ke) <= loss(kg) + 1e-6).all() and (kg.sum(-1) <= cap).all()


# ── Smoke: 1 batch forward + backward, print shapes ──────────────────────────

def _batch(tok, n=4):
    texts = ["the company expects sales growth next year", "net loss rose", "board governance risk",
             "emissions fell this quarter and employees shares rose"][:n]
    enc = tok(texts, padding=True, return_tensors="pt")
    return enc["input_ids"], enc["attention_mask"], torch.tensor([[0, 1, 2], [1, 0, 3], [2, 2, 1], [0, 1, 0]])[:n]


def test_smoke_tokenalloc():
    from tokenalloc.encoder import check_attention_rules
    from tokenalloc.model import TokenAllocSystem
    torch.manual_seed(0)
    bert, tok = tiny_bert_and_tokenizer()
    model = TokenAllocSystem(bert, {"use_confidence": True, "use_power": True})
    ids, am, y = _batch(tok)
    ok, diffs = check_attention_rules(model.encoder, ids, am)
    assert ok, diffs
    cls, task_out = model.encode(ids, am)
    snr = torch.tensor([-5.0, 0.0, 5.0, 10.0])
    w = torch.full((4, 3), 1 / 3)
    a = model.allocate(cls, task_out, snr, "Rician", w, cap=8)
    out = model.transmit_decode(task_out, snr, model.realization(4, "Rician"), a["k_hard"],
                                a["c_soft"], a["power_scores"])
    loss = sum(torch.nn.functional.cross_entropy(l, y[:, t]) for t, l in enumerate(out["logits"]))
    loss = loss + 0.01 * a["c_soft"].sum()
    loss.backward()
    assert model.allocator.count_head.weight.grad.abs().sum() > 0
    assert model.encoder.task_tokens.grad.abs().sum() > 0
    print(f"\n[tokenalloc] input_ids {tuple(ids.shape)} -> CLS {tuple(cls.shape)}, task tokens "
          f"{tuple(task_out.shape)} -> counts {a['k_hard'].tolist()} -> symbols "
          f"{out['symbols'].tolist()} -> logits {[tuple(l.shape) for l in out['logits']]}, "
          f"loss {loss.item():.3f}")


def test_smoke_legacy():
    from legacy.model import FinDeepSC
    from legacy.train import legacy_loss, run_model
    torch.manual_seed(0)
    ids, am, y = None, None, None
    for readout in ("shared", "equal_split", "softmask"):
        for tx in ("per_token", "pooled"):
            bert, tok = tiny_bert_and_tokenizer()
            ids, am, y = _batch(tok)
            model = FinDeepSC(bert, ch_dim=32 if tx == "per_token" else 64, readout=readout, tx_mode=tx)
            batch = {"input_ids": ids, "attention_mask": am}
            out = run_model(model, batch, "Rayleigh", torch.zeros(4))
            loss = legacy_loss(out, y, {"sentiment": None, "fls": None, "esg": None}, 0.1, am)
            loss.backward()
            print(f"\n[legacy {readout}/{tx}] symbols/sentence {out['symbols'].tolist()}, "
                  f"logits {[tuple(l.shape) for l in out['logits']]}, loss {loss.item():.3f}")


if __name__ == "__main__":
    import inspect, sys
    fns = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and inspect.isfunction(f)]
    failed = 0
    for name, fn in fns:
        try:
            fn()
            print(f"PASS {name}")
        except Exception as e:                                   # noqa: BLE001
            failed += 1
            print(f"FAIL {name}: {e!r}")
    sys.exit(1 if failed else 0)
