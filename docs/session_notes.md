# Notes from the first build session

Read with `CLAUDE.md` (the design contract), `docs/architecture.png` (the pipeline
figure, steps 1-12) and `docs/prior_paper_TPSRA.pdf` (the prior paper).

## State
- Tasks 1-7 of CLAUDE.md §11 are implemented and tested offline (`python -m tests.test_all`,
  11 tests; `python run.py --config runs/configs/smoke.yaml` runs end to end on synthetic data).
- Tasks 8-9 (real runs, per-sentence CSVs) have NOT been run: they need a GPU and the data file.
- Old files still to delete (the owner approved; the cloud session was not permitted to):
  `main.py utils.py dataset.py generate_labels.py performance.py SBERT.py eval_tools.py
  preprocess_text.py models/` (replaced by `common/`, `legacy/`, `tokenalloc/`).

## Findings
1. Rician bug (task 2): the old code used a REAL fading coefficient per real number and divided
   by `clamp(h, min=1e-6)`; every negative h became 1e-6 (50% of entries on Rayleigh, 8% on
   Rician K=1) and amplified that entry by ~1e6. It also used K=1, not K=4. Not the LOS-phase
   issue CLAUDE.md suspected. Fixed in `common/channel.py`; a test checks AWGN < Rician < Rayleigh
   post-ZF error.
2. The repo did not contain the prior paper's code (no SoftMaskGate, B0-B6). The old repo code
   also differs from the paper (Adam, SNR 5-20 dB per epoch, MLP heads). `legacy/` rebuilds
   B0/B1/B3 from the old code + paper text (§III-D/E, §IV-B).
3. Budget mismatch: the old model sent every word token at ch_dim=32 (16 complex symbols per
   word token, ~150-600 per sentence), not 32 symbols per sentence. `legacy_B*_32sym` configs
   send one pooled vector as 32 symbols for a fair comparison with cap 8.
4. The paper says ESG labels come from FinBERT-tone; the label script used FinBERT-ESG
   (`yiyanghkust/finbert-esg`). Fix the sentence in the new paper.
5. The paper (§II-A) defines fading as i.i.d. per symbol; that is what is implemented.

## Interpretations to confirm with the authors
- Priority enters the allocator as `score_t = base_t(ctx) + softplus(gain_t(ctx)) * w_t`
  instead of a 3->32 embedding in the trunk: this guarantees monotonicity in the task's own
  priority, which an embedding cannot.
- CLS and [SEP] follow the word rule (they do not attend to task tokens), so the allocator's
  sentence input is task-agnostic. Task tokens use position id 0; words keep positions 1..n.
- `random_matched` matches the allocator's total per sentence (not just on average).
- Legacy lr: 2e-5 for all parameters (the paper reports only that); confirm with the old notebook.
- Class-weighted CE on ESG only (CLAUDE.md §5); change `train.class_weighted` to add FLS.

## Run order (GPU)
1. `python -m common.prepare_data`
2. `exp1_saturation`, `exp2_main_cap8`, `legacy_B1_32sym`, `legacy_B3_32sym`
3. `exp3_priority` (needs exp2 cap 8 checkpoints), `exp3_fixed_priority`, `exp4_no_*`
4. other caps, lambda sweep, remaining legacy configs
5. `python -m common.analyze runs/<...> --ref allocator --out runs/report_<name>`
