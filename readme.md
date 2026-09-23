# Encoder-side task-token allocation for multi-task semantic communication

Karl Khouri, Maria Slim (AUB). Follow-up to *Mapping the Design Space of Task-Priority
Resource Allocation in Multi-Task Wireless Semantic Communication* (TP-SRA).
The design contract is [`CLAUDE.md`](CLAUDE.md); read it first.

One financial sentence, three tasks (Sentiment, FLS, ESG). Each task owns 4 learnable
**task tokens** inside the BERT encoder. A small **allocator** decides, per sentence,
how many task tokens each task sends (1 to 4, at most `K_max` in total) from the sentence
(CLS), the SNR (CQI), the channel type and the operator priority `w`. Each task token
becomes 4 complex symbols. The receiver zero-forces, decodes each task's tokens and
answers the three questions.

## Layout

```
common/       shared by both pipelines
  channel.py      AWGN / Rayleigh / Rician K=4 (complex, per-symbol fading, E|h|^2=1), ZF, fixed seeds
  data.py         PhraseBank dataset, fixed train/val/test split, class weights
  prepare_data.py builds data/phrasebank_multitask.parquet (PhraseBank + FinBERT-FLS/ESG labels)
  stats.py        paired bootstrap over (seed, channel, SNR, sentence)
  analyze.py      tables, bootstrap CIs and figures from the per-sentence logs
  utils.py        configs, seeding, encoder loading, optimizer
  tiny.py         tiny random BERT + synthetic data (smoke tests only)
tokenalloc/   the new pipeline (CLAUDE.md §4)
  encoder.py      task tokens + attention masks inside BERT
  allocator.py    allocator network, count decode, straight-through masks, power shares
  model.py        per-task channel encoder/decoder, channel, ZF, masked attention pooling, heads
  policies.py     baselines: equal / fixed / proportional / SNR-only / random / greedy + exact oracle
  train.py        nested-dropout warm-up, then allocator phase
  evaluate.py     SNR sweep with shared channel realizations, per-sentence logs
legacy/       prior paper (B0 shared, B1 equal split, B3 SoftMaskGate), channel bug fixed
runs/configs/ one YAML per experiment (pipeline, cap, channels, seeds, SNR list)
tests/        unit + smoke tests (offline, CPU)
run.py        launches any config
```

`tokenalloc/` and `legacy/` never import each other; both only use `common/`.

## Setup

```bash
pip install -r requirements.txt
python -m tests.test_all                           # all tests, CPU, ~1 min, no downloads
python run.py --config runs/configs/smoke.yaml     # end-to-end smoke run on synthetic data
```

## Data (once, needs internet; a GPU makes the labelling fast)

```bash
python -m common.prepare_data --out data/phrasebank_multitask.parquet
# or, with the original zip extracted:
python -m common.prepare_data --phrasebank-txt FinancialPhraseBank-v1.0/Sentences_50Agree.txt
```

Sentiment labels are human (PhraseBank). FLS labels come from `yiyanghkust/finbert-fls` and
ESG labels from `yiyanghkust/finbert-esg` (machine labels: a label-noise floor).
The encoder is `yiyanghkust/finbert-pretrain`. `ProsusAI/finbert` is refused because it was
fine-tuned on PhraseBank (test-set leakage).

## Experiments (run in this order, GPU)

| Config | What it gives |
|---|---|
| `exp1_saturation` | accuracy vs task tokens per task (1..4) at -5/5/15 dB |
| `exp2_main_cap{6,8,10,12}` | allocator vs every baseline, accuracy and symbols vs SNR |
| `legacy_B{0,1,3}` and `legacy_B{0,1,3}_32sym` | prior-paper baselines (as before, and at 32 symbols) |
| `exp3_priority` (after exp2 cap 8), `exp3_fixed_priority` | does priority steer? |
| `exp4_no_{sentence,snr,channel}` | which allocator inputs matter |
| `exp2_lambda{0.001,0.01,0.1}_cap12` | variable rate: accuracy vs symbols |

```bash
python run.py --config runs/configs/exp2_main_cap8.yaml            # 3 seeds
python -m common.analyze runs/exp2_main_cap8 runs/legacy_B1_32sym runs/legacy_B3_32sym \
       --ref allocator --out runs/report_cap8
```

Every run writes `runs/<name>/seed<k>/per_sentence.csv.gz`: one row per method, channel, SNR,
priority and test sentence, with predictions, correctness, CE, counts `k_S,k_F,k_E`, symbols
and power shares. All methods and both pipelines see the same channel realization for the
same (channel, SNR, batch), so comparisons are paired.

## Google Colab

Open `colab.ipynb`. It mounts Google Drive, keeps `data/` and `runs/` there (so a disconnect
loses nothing), runs the tests, prepares the data and launches configs one by one.
