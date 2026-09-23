# CLAUDE.md: Encoder-Side Task-Token Allocation for Multi-Task Semantic Communication

This file is the design contract for the project. It captures every decision made
during the design discussion. Read it fully before touching code. Do not silently
deviate from it; if something here conflicts with the existing repo, flag it and ask.

Authors: Karl Khouri, Maria Slim (AUB). Prior paper: "Mapping the Design Space of
Task-Priority Resource Allocation in Multi-Task Wireless Semantic Communication"
(TP-SRA, FinBERT-DeepSC testbed). This project is the follow-up.

---

## 1. Why we are doing this (motivation)

The prior paper allocated the shared decoded representation AFTER the channel using
soft multiplicative masks (SoftMaskGate). Results: no soft-mask variant reached
statistical significance on any channel; only a hard equal split (B1) won everywhere;
doubling the channel rate (N=32 to 64) HURT the baseline (-3.92 pp); the joint
rate-mask solver (B6) learned how much to prune but not what to prune (cross-seed
Spearman ~0).

Lesson: allocation must happen BEFORE the channel, in separate task-owned units, and
more symbols is not automatically better. The new question: how many symbols does
each task actually need, for this sentence, on this channel, given this priority?

## 2. Problem formulation

Setting: one financial sentence x, three tasks answered simultaneously at the
receiver: Sentiment (3 classes), FLS (3 classes), ESG (4 classes). Transmitter knows
the average SNR via CQI feedback and the channel type. Operator sets priority
w = (w_S, w_F, w_E), sum 1.

Decide per sentence: token counts k_t in {1,2,3,4} per task, optional power share
p_t per task.

Constraints:
- sum_t k_t <= K_max (the cap; K_max is an external resource grant, NOT what is used)
- sum_t k_t * p_t = sum_t k_t (average power per sent symbol stays 1)

Objective (weighted form):
  minimize E[ sum_t w_t * CE_t + lambda * (symbols sent) ]
Objective (target form, later): minimize symbols s.t. accuracy_t >= a_t. lambda is
then the Lagrange multiplier.

Theory anchor: under diminishing returns, the optimum follows greedy marginal
allocation (Fox 1966): give tokens one at a time to the task with the largest
w_t * (loss reduction), stop when gain < price or cap reached. This is the discrete
twin of Theorem 1 in the prior paper. It also defines the greedy oracle baseline.

Claims to test:
- H1: task-owned tokens before the channel beat post-channel and shared-pool allocation
- H2: learned per-sentence, per-SNR allocation beats fixed splits (higher accuracy at
  equal symbols, fewer symbols at equal accuracy)
- H3: learned allocator is close to the greedy oracle
- H4: priority actually steers (prior paper's runtime lever collapsed to ~2.4 pp)
- H5: symbols sent fall as SNR rises

## 3. Terminology (use these words in code and paper)

- word token: a tokenizer word piece. Input only. Never sent.
- task token: one of 12 learnable vectors prepended to the encoder input (4 per task:
  S1-S4, F1-F4, E1-E4). Each is wired to exactly one task. Its 768-dim output is the
  unit of allocation. (Earlier nickname: "envelope". Do not use that word.)
- symbol: one complex channel use. L = 4 symbols per task token (encoder output = 8
  reals). N = K * L channel uses per sentence.
- cap K_max: max task tokens per sentence. Default 8 (32 symbols, matches prior
  paper). Also run 6, 10, 12.
- slots: 4 per task, 12 total. Slots != cap.

## 4. Architecture, end to end

### Transmitter
1. Tokenizer: sentence -> word tokens. Build input:
   [CLS] S1..S4 F1..F4 E1..E4 <words> [SEP]
2. Encoder (BERT-family, see Section 8 on checkpoint). Attention masks:
   - task tokens attend to words and to their own task's tokens
   - words do NOT attend to task tokens (keeps word processing pretraining-faithful,
     prevents leakage across tasks)
   - task tokens of different tasks do NOT attend to each other
   Outputs kept: CLS (768) and 12 task tokens (12 x 768). Word outputs discarded.
3. Allocator (one network for all tasks, joint decision):
   Inputs:
   - CLS: 768
   - SNR: dB/10, expanded by a small linear+ReLU to 32
   - channel type: one-hot(3) [AWGN, Rayleigh, Rician], expanded to 32
   - priority w: 3, expanded to 32; constrain weights so a task's score is
     monotone non-decreasing in its own priority (positive weights on that path)
   - optional (ablation): per-task confidence = normalized entropy of each task head
     run on the CLEAN pooled task tokens at the transmitter, 3 values, detached
   Network: concat -> 128 (ReLU) -> 3 scores (count head) [+ 3 scores (power head)]
   Count decode:
   - soft count c_t = 1 + 3 * sigmoid(score_t)   (floor 1, max 4)
   - if sum_t c_t > K_max: shrink extras: c_t = 1 + (c_t - 1) * (K_max - 3)/(sum(c)-3)
   - round to nearest int; if rounding exceeds K_max use largest-remainder rule
     (floor all, hand out remaining by largest fractional part, ties by priority)
   - straight-through: k = c_soft + stop_grad(k_hard - c_soft)
   Power decode (optional): p_t = K_used * exp(b_t) / sum_j k_j exp(b_j), so
   sum_t k_t p_t = K_used.
   Init: output layers zero -> starts at equal split, equal power (warm start = token
   version of prior paper's B1).
4. Token selection: keep the FIRST k_t tokens of each task. Mask:
   m_soft = sigmoid((c_soft - j + 0.5)/tau), tau=0.3; m_hard = 1[j <= k_hard];
   m = m_soft + stop_grad(m_hard - m_soft). Dropped tokens not encoded at test time.
5. Per-task channel encoder (3 separate nets, one per task, reused for each token of
   that task): 768 -> 256 (ReLU) -> 8 reals -> 4 complex symbols. Normalize each
   token block to average energy 1 per symbol, then multiply by sqrt(p_t).
   No mixing across tokens. (Steps 1-3 are DeepSC-standard; per-task power is ours.)
6. Frame: [pilots (not simulated)] [header: counts, 6 bits, assumed error-free]
   [payload: 4 symbols per kept token]. Interleaving: skip (no effect under i.i.d.
   per-symbol fading).

### Channel
7. y = h * x + n, n ~ CN(0, sigma^2), sigma^2 = 10^(-SNR/10).
   AWGN: h=1. Rayleigh: h ~ CN(0,1). Rician K=4: h = sqrt(4/5) e^{j theta} +
   sqrt(1/5) CN(0,1). All have E|h|^2 = 1.
   Training SNR: uniform in [-10, 10] dB per sentence. Channel type: random per
   batch during training.
   BUG CHECK FIRST: prior paper's Rician results were WORSE than Rayleigh, which
   physics forbids. Likely cause: equalizer not using the full complex h (LOS
   phase). Verify before any new runs.

### Receiver
8. Equalization: ZERO FORCING (x_hat = y / h), perfect h known at receiver, same as
   DeepSC. Decision: keep ZF for all methods for now (not MMSE). State in paper.
   Note h known at receiver is the standard perfect-CSI assumption.
9. Header read: counts -> split payload into 4-symbol blocks per task.
10. Per-task channel decoder: 8 reals -> 256 -> 768, one per task.
11. Masked attention pooling per task: learned query q_t (768); score_j =
    q_t . r_j / sqrt(768); unsent slots get -inf; softmax; weighted sum -> one
    768 vector per task.
12. Task heads: one linear layer per task (768 -> #classes). Softmax, argmax.

### CQI
- Real system: receiver measures SNR from pilots, reports it back (rounded, slightly
  late). Transmitter only ever knows the AVERAGE SNR, never per-symbol h.
- Simulation: we choose the SNR, set the channel noise to it, and hand the same
  value to the allocator (perfect CQI). Allocator may receive SNR at test time; this
  is legitimate. Never feed the allocator per-symbol h.
- Robustness test (later): allocator gets SNR rounded to 16 levels or +/-2 dB off.
- ZF vs MMSE does not affect CQI (measured before equalization).

## 5. Training

Loss: sum_t w_t * CE_t (class-weighted CE for ESG imbalance) + lambda * sum_t c_soft_t.
lambda = 0 when only redistributing under a hard cap; lambda > 0 for variable rate.
Sweep lambda in {0, 0.001, 0.01, 0.1} to trace accuracy vs symbols.

Phases:
1. Warm-up (no allocator): nested dropout. Per sentence, per task, sample a random
   cutoff j in {1..4}, keep first j tokens. Trains ordering (token 1 most important).
2. Allocator on, warm-started at equal split. Keep nested dropout on 20% of batches.
   Priority w sampled ~ Dirichlet per batch. SNR uniform [-10,10] dB. Channel type
   random. Also train one fixed-priority model (w_E = 0.8) as a check that the
   Dirichlet model matches it.
Confidence input (if used): turn on only after warm-up; .detach() so heads cannot
learn to look unsure to get more tokens.

Optimizer: AdamW, lr 2e-5 for encoder, larger (1e-3) for allocator/encoders/heads,
batch 16, ~12 epochs (match prior paper).

## 6. Testing

- Frozen weights. Allocator still runs per sentence and adapts to sentence, SNR,
  channel type, priority. Hard rounding only.
- SNR sweep: {-15,-10,-5,-2,0,2,5,10,15,20} dB. Three channels.
- FIXED noise/fading seeds shared across ALL methods (required for paired bootstrap).
- Log per sentence: predictions, counts (k_S,k_F,k_E), symbols used, power shares.
- Report: accuracy per task vs SNR; average symbols vs SNR; allocation histogram per
  SNR; priority sweep response.

## 7. Experiments and baselines (minimal set first)

Minimal set (run these first):
| # | Experiment | Baselines | Proves |
|---|---|---|---|
| 1 | Saturation map: accuracy vs tokens/task (1..4), SNR {-5,5,15} | none | picks L, builds oracle |
| 2 | Main sweep: acc vs SNR and avg symbols vs SNR, caps K in {6,8,10,12} | fixed equal split (2/2/2, 3/3/2, 4/3/3, 4/4/4), SNR-only rule, greedy oracle | bandwidth + accuracy |
| 3 | Priority sweep: w_E 0.1..0.9 at 0 dB | fixed split | priority steers |
| 4 | Ablation: remove sentence input; remove SNR input; remove channel-type input | full allocator | inputs matter |

Baselines share the EXACT pipeline; only the allocator is swapped:
- Fixed equal split at total K
- Fixed proportional split: round(w * K)
- SNR-only rule (U-DeepSC-style): count from SNR lookup, same for every sentence
- Random split matched to allocator's average count
- Prior paper B1 and B3 at 32 symbols
- Greedy oracle from measured loss curves

Statistics: n >= 3 seeds, paired-bootstrap 95% CIs (10,000 resamples) over
(seed, SNR, sentence). Rule for "better": higher accuracy at same average symbols,
or same accuracy with fewer symbols, CI excluding 0.

Later (only if time): per-task power ablation, confidence input ablation, imperfect
CQI, cap 12 with lambda sweep, second dataset.

## 8. Dataset and encoder (must fix before new results)

Problems with current setup:
1. LEAKAGE: ProsusAI/finbert was fine-tuned on Financial PhraseBank, our test set.
   Switch to a checkpoint NOT fine-tuned on PhraseBank: bert-base-uncased or a
   FinBERT pretrain-only checkpoint (e.g., yiyanghkust/finbert-pretrain).
2. FLS and ESG labels are machine-generated by other FinBERT classifiers (label
   noise floor). Verify which model produced ESG labels (FinBERT-ESG, not
   FinBERT-tone).
3. Sentiment ~97% leaves no room for allocation gains.
4. ~4,800 sentences, small.

Second dataset if PhraseBank results are flat: SemEval-2016 Task 6 (tweets with
stance + sentiment + target, all human labels, 2,914 train / 1,249 test; stance is
hard so tasks compete). Optional: FiQA 2018 Task 1 (sentiment + 2-level aspect,
~1,174 examples, use 5-fold CV).

## 9. Related work: what NOT to claim as new

Already exists: adaptive symbol count driven by noise (U-DeepSC, GlobeCom 2022);
task embedding tokens (U-DeepSC); nested dropout / ordered features in SemCom;
per-token unequal error protection (TONIC, TokenComSR); shared+private feature
splits (U-DeepSC, CMT-SemCom, CRU); Shannon-formula counts with importance ranking
(SMSC-FIR, Take-What-You-Need). U-DeepSC serves ONE task per transmission.

Our claim: several tasks from the same sentence COMPETE for one per-sentence budget,
each with its own task tokens; the allocator learns the minimum from sentence + SNR +
channel type + priority; priority is a runtime input. Must include U-DeepSC-style
and SMSC-FIR-style baselines.

Target venue: IEEE GlobeCom 2027 (ICC 2027 deadline too close).

## 10. How to treat the existing codebase: two SEPARATE pipelines

The repo contains the prior paper's pipeline (FinBERT-DeepSC, SoftMaskGate, B0-B6).
The new work is a SEPARATE pipeline that lives next to it. Layout:

  legacy/     the prior paper's code, frozen. Only bug fixes (Rician equalizer,
              encoder checkpoint). B0, B1, B3 must still run exactly as before.
  tokenalloc/ the new pipeline from this document (task tokens, allocator,
              per-task encoders/decoders, pooling, baselines).
  common/     shared by both: data loading, channel simulation, ZF equalizer,
              SNR sweep, seeding, bootstrap CIs, plotting.
  runs/       configs and outputs, one folder per experiment.

Rules:
- Never import legacy model code into tokenalloc/, and vice versa. They share
  only common/.
- Move shared utilities into common/ once, then both pipelines call them, so
  every comparison uses identical data, noise seeds, and evaluation.
- Fix bugs in place; do not rewrite legacy/.
- Before any edit: list every file, what it does, and every mismatch with this
  document. Ask before deleting anything.
- After each step: run a smoke test (1 batch forward + backward), report shapes.
- Every experiment is launched by a config file that names the pipeline
  (legacy or tokenalloc), the cap, the channel, the seeds, and the SNR list.

## 11. Immediate task list (in order)

1. Read the repo, summarize the current pipeline, report any mismatch with this file.
2. Fix the Rician equalization bug check.
3. Switch encoder checkpoint (Section 8).
4. Implement task tokens + attention masks + nested dropout warm-up.
5. Implement per-task channel encoder/decoder, ZF equalizer, masked attention pooling.
6. Implement allocator with CLS + SNR + channel type + priority inputs.
7. Implement fixed-split baselines and SNR-only rule.
8. Run Experiment 1 (saturation map), then 2, 3, 4 with 3 seeds and shared noise seeds.
9. Save all per-sentence logs to CSV for plotting.
