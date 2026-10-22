# Technical brief: task-token allocator vs information-bottleneck baseline

Purpose: an independent check of (1) whether the comparison between our method and the
information-bottleneck (IB) baseline is fair, and (2) whether the conclusions drawn from the
numbers are correct. Everything below is what the code actually does (repository
`karlkhoury/VIP`, branch `prior-paper-work`, commit `0f69bba`). Numbers are AWGN, test set,
3 seeds; Rayleigh has not been run for the baseline yet.

---

## 1. Task and data

- One English financial sentence, three classification tasks answered at the receiver from one
  transmission: Sentiment (3 classes, human labels), forward-looking statement FLS (3 classes) and
  ESG (4 classes: environmental, social, governance, none). FLS and ESG labels come from the
  FinBERT-FLS and FinBERT-ESG classifiers (label noise).
- Financial PhraseBank, 50 % agreement subset, cleaned to 4,825 sentences. Split 70/10/20
  (3,379 / 480 / 966), stratified on the joint label; sentences differing only in numbers are kept
  in the same split. Max 128 word pieces.
- Each channel (AWGN, Rayleigh) is a separate study: trained and tested only on that channel.

## 2. Shared architecture (our system)

1. **Encoder**: `yiyanghkust/finbert-pretrain` (BERT-base, 110M, d = 768). Input:
   `[CLS]`, then T·J learnable task-token vectors (T = 3 tasks, J = 32 per task, 96 total), then the
   word pieces, then `[SEP]`. Attention mask: a task token attends to the words and to the tokens of
   its own task; words, `[CLS]`, `[SEP]` attend only to words. Outputs used: CLS (768) and the 96
   task-token outputs (each 768). Word outputs discarded.
2. **Allocator** (ours, ~0.1M parameters): input = CLS (768) ‖ SNR embedding (γ/10 → Linear → ReLU,
   32) ‖ a constant zero 32-vector (unused channel-type slot). Hidden layer 832 → 128, ReLU → u.
   Two heads 128 → 3: base score b_t(u) and raw gain g_t(u). Score
   s_t = b_t(u) + softplus(g_t(u)) · w_t, where w is the operator priority (sums to 1).
   Soft count c_t = 1 + (J_max − 1)·σ(s_t); if Σ c_t > K, the part above 1 is scaled down so the
   total equals K; round; if the rounded total exceeds K, floor and give the rest by largest
   fractional part (ties by priority). Straight-through estimator: forward uses hard counts and hard
   masks, backward uses soft masks m_{t,j} = σ((c_t − j + 0.5)/τ), τ = 0.3. Heads initialised to
   zero, gain bias −5 (starts at the equal split). Cap K = 16 task tokens, J_max = K/2 = 8 per task.
   Task t sends its first k_t tokens; the others are never encoded.
3. **Channel encoder** (one per task, shared by that task's tokens): 768 → 256 (ReLU) → 8 reals →
   4 complex symbols. Each token block is normalised to average energy 1 per symbol.
   Symbols per sentence N = 4 Σ_t k_t (max 64 at K = 16). A header carries k (assumed error-free).
4. **Channel**: y = h ⊙ x + n, n ~ CN(0, σ²I), σ² = 10^(−SNR/10). AWGN h = 1; Rayleigh h ~ CN(0,1)
   i.i.d. per symbol. Receiver knows h, zero forcing x̂ = y / h.
5. **Receiver**: per-task channel decoder 8 → 256 (ReLU) → 768 per token; per-task attention
   pooling with a learned query (score q·r/√768, softmax over sent tokens only); per-task linear
   head 768 → 3 / 3 / 4.

## 3. Training of our system (main results in the paper)

- **Stage 1 (pipeline, joint)**: encoder + task tokens + channel encoders + decoders + pooling +
  heads, 12 epochs. Nested dropout: per sentence and task, k ~ U{1..32}, first k tokens sent.
  SNR ~ U[−10, 10] dB per sentence. Loss: (1/3) Σ_t CE_t, inverse-frequency class weights for FLS
  and ESG (mean 1). AdamW, lr 2e-5 (encoder) / 1e-3 (rest), batch 16, grad clip 1.0. Best epoch by
  fixed-noise validation loss.
- **Stage 2 (allocator only)**: everything from stage 1 frozen, encoder outputs cached. 80 epochs,
  AdamW lr 1e-3 (wd 0.01), batch 16. Per batch: SNR ~ U[−10,10], priority w ~ Dirichlet(1,1,1).
  Loss: Σ_t w_t CE_t (class-weighted) + λ Σ_t c_soft,t. Kept epoch: lowest validation objective
  (unweighted CE at equal priority + λ·tokens), each validation sentence scored at 5 fixed SNR/noise
  draws; epoch 0 (= equal split) is a candidate.
- **Choice of λ**: grid {0.001, 0.002, 0.003, 0.005, 0.01, 0.02} at K = 16; on VALIDATION, keep the
  λ with the fewest symbols whose mean accuracy is within ε = 0.25 points of the equal split.
  Selected λ = 0.005 (AWGN and Rayleigh). The test set checks non-inferiority (CI lower end > −ε).

## 4. IB baseline (Shao, Mao, Zhang, IEEE JSAC 2022, "VL-VFE"), adapted to our system

Original: single task, images, variable-length variational feature encoding. Reimplemented from
their public code (github.com/shaojiawei07/VL-VFE) inside our pipeline:

- **Tokens fixed**: equal split at K = 16 → k = (6, 5, 5) for (Sentiment, FLS, ESG). No priority input.
- **Decision = symbols per token, from the noise level only**: per task a "gamma" network
  (1 → 16 → 16 → 16 → 4, weights |W|, tanh) maps σ_r = sqrt(σ²/2) (per-real noise std) to 4 values;
  an upper-triangular cumulative sum makes them non-increasing (μ_1 ≥ … ≥ μ_4, clamp ≥ 1e-4).
  Encoded feature x = tanh(μ ⊙ u), u = copied channel-encoder output (μ shared by the real and
  imaginary part of each complex symbol). Hard mask μ > 0.01 keeps a prefix of the 4 symbols
  (first symbol forced on); straight-through in training. Sent symbols of each token normalised to
  energy 1 per sent symbol; unsent symbols carry nothing. Symbols = Σ_t k_t · m_t (min 16).
- **Decoder** sees the noise level: input = 8 reals ‖ 16-value noise embedding
  (MLP 1 → 16 → 16 → 16, ReLU, on σ_r); initialised from the stage-1 decoder, noise weights zero.
- **Loss**: (1/3) Σ_t class-weighted CE_t + β · a(epoch) · KL, KL = log-uniform-prior approximation
  (constants k1 = 0.63576, k2 = 1.8732, k3 = 1.48695) on α² = σ_r² / (x² + 1e-4), summed over the
  real dimensions of the sent tokens, × 0.1/σ_r, batch mean; a = 0 for epochs ≤ 5, linear to 1 over
  10 epochs.
- **Training**: FinBERT and task tokens frozen (cached). Trained jointly: copies of the stage-1
  channel encoders, pooling and heads + the new decoder + gamma nets. 80 epochs, AdamW lr 1e-3,
  wd 5e-5, batch 16, clip 1.0, SNR ~ U[−10,10]. Kept epoch: lowest validation (CE + β·KL at the
  equal split, 5 draws), from epoch 15 on.
- **Choice of β**: grid {3e-4, 1e-3, 3e-3, 1e-2, 3e-2}, same ε rule as λ. **Reference used by the
  code: the equal split on the STAGE-1 receiver** (see section 8, issue 1). Selected β = 0.001.

## 5. Same-receiver control (section 8c of the notebook)

Motivation: the IB baseline retrains its receiver for 80 epochs (with an SNR-aware decoder); our
allocator runs on the stage-1 receiver. Control: give our side the same receiver training.

- **Stage 1b (receiver, ours)**: FinBERT and task tokens frozen. Copies of the stage-1 channel
  encoders, pooling and heads + the SAME SNR-aware decoder (same function, same initialisation),
  trained jointly, 80 epochs, AdamW lr 1e-3, wd 5e-5, batch 16, clip 1.0, SNR ~ U[−10,10].
  Instead of symbol trimming: per sentence and task a random number of whole tokens
  k ~ U{1..14} (14 = K − 2, the most one task can send under K = 16), all 4 symbols each.
  Loss (1/3) Σ class-weighted CE. Kept epoch: lowest validation CE (random k and SNR from fixed seeds,
  5 draws). No allocator involved.
- **Stage 2 on it**: receiver frozen, allocator trained with exactly the stage-2 recipe above,
  λ = 0.005 (NOT re-selected on this receiver), J_max = 8.
- Variant B: λ = 0, J_max = 14 (total cap still 16).
- Summary of training per side: ours = stage 1 (shared) → stage 1b receiver (80 ep) → stage 2
  allocator (80 ep). IB = stage 1 (shared) → one joint stage (receiver + cut-off nets, 80 ep).

## 6. Evaluation protocol (all methods)

- Test: 966 sentences × 10 SNRs {−15, −10, −5, −2, 0, 2, 5, 10, 15, 20} dB × 3 seeds = 28,980 cases.
- Fixed channel draws: for each (channel, SNR, batch index) a generator seeded from
  noise_seed = 1234 produces h and n for a grid [batch, 3 tasks, 32 token slots, 4 symbols]; every
  method uses the same grid, so slot (task, token j, symbol l) sees the same h and n whatever the
  method (paired comparisons).
- Metrics: average symbols per sentence; mean accuracy = unweighted mean of the 3 task accuracies;
  weighted accuracy Σ w_t Acc_t when priorities are unequal.
- Statistics: paired bootstrap of the mean difference, 10,000 resamples, percentile 95 % CI.
  **Resampling unit = individual case (seed, SNR, sentence), i.i.d.** (see section 8, issue 4).
- Priority test: ESG priority w_E ∈ {0.1 … 0.9}, the other two tasks (1 − w_E)/2 each, at −15, −10,
  −5, 0 dB; compared on ESG accuracy (and weighted accuracy) with the IB baseline (no priority).

## 7. Results (AWGN, test, 3 seeds)

### 7a. Stage-1 receiver for ours, retrained receiver for IB (unequal receivers)
| Method | Symbols | Mean acc. |
|---|---|---|
| All tokens, no cap | 384 | 87.62 |
| Equal split, K = 16 | 64 | 84.86 |
| Allocator, λ = 0.005 | 35.13 | 84.81 (−0.05 [−0.16, +0.06] vs equal) |
| IB, β = 0.001 | 47.40 | 85.91 (+1.05 [+0.89, +1.20] vs equal) |

### 7b. Same receiver training (control)
| Method | Symbols | Mean acc. |
|---|---|---|
| Equal split, retrained receiver | 64 | 86.41 |
| Allocator, retrained receiver, λ = 0.005 | 35.16 | 86.39 (−0.03 [−0.13, +0.07] vs equal) |
| IB, β = 0.001 | 47.40 | 85.91 (−0.51 [−0.65, −0.37] vs retrained equal) |
| IB, β = 0.0003 | 53.57 | 86.88 (+0.47 [+0.33, +0.60] vs retrained equal) |

Allocator − IB(β = 0.001): accuracy +0.48 [+0.33, +0.64], symbols −12.24.
Retraining alone lifts the equal split by +1.55 points (84.86 → 86.41).

Validation (used for the ε rule): retrained equal 87.65; IB β = 0.0003: 87.75 (+0.10 [−0.08, +0.29]
vs retrained equal); IB β = 0.001: 86.83 (−0.82 [−1.03, −0.62]); IB β = 0.003: 34.4 symbols, 82.73.
Stage-1 equal (validation) 85.84; allocator λ = 0.005 (stage-1 receiver, validation): 35.0 symbols,
85.88.

### 7c. Priority test (ESG = high-priority task), same receiver, allocator λ = 0.005, J_max = 8
ESG accuracy, ours (w_E = 0.9) vs IB β = 0.001 (paired, 95 % CI):
| SNR | Ours | IB | Diff [CI] | Symbols ours / IB |
|---|---|---|---|---|
| −15 | 70.22 | 62.46 | +7.76 [+5.97, +9.52] | 56.8 / 64.0 |
| −10 | 84.89 | 78.47 | +6.42 [+5.14, +7.73] | 57.1 / 62.3 |
| −5 | 87.51 | 87.20 | +0.31 [−0.41, +1.04] | 53.7 / 62.3 |
| 0 | 88.47 | 89.34 | −0.86 [−1.59, −0.14] | 35.7 / 54.0 |
| all four | 82.77 | 79.37 | +3.41 [+2.80, +4.01] | 50.8 / 60.7 |

Against IB β = 0.0003 (the β the ε rule selects against the retrained equal split), paired:
| SNR | Ours | IB | ESG diff [CI] | Weighted-acc diff [CI] | Symbols ours / IB |
|---|---|---|---|---|---|
| −15 | 70.22 | 66.05 | +4.18 [+2.38, +5.94] | +2.94 [+1.31, +4.52] | 56.8 / 64.0 |
| −10 | 84.89 | 79.71 | +5.18 [+3.90, +6.42] | +4.01 [+2.86, +5.16] | 57.1 / 62.3 |
| −5 | 87.51 | 87.65 | −0.14 [−0.93, +0.62] | −0.39 [−1.10, +0.29] | 53.7 / 62.3 |
| 0 | 88.47 | 89.51 | −1.04 [−1.66, −0.45] | −1.00 [−1.56, −0.48] | 35.7 / 59.0 |
| all four | 82.77 | 80.73 | +2.04 [+1.45, +2.63] | +1.39 [+0.85, +1.92] | 50.8 / 61.9 |

Without the same receiver (stage-1 receiver for ours), the same test gave ours − IB(β = 0.001) =
−7.0 at −15 dB and +1.2 (n.s.) at −10 dB.

Variant B (λ = 0, J_max = 14): ESG +5.8 / +5.1 at −15 / −10 dB vs IB β = 0.001; ESG received only
~7 tokens on average, so raising J_max did not help.

## 8. Known issues / asymmetries to check

1. **β reference**: the IB's β was selected with the ε rule against the stage-1 equal split
   (85.84 % on validation), but the IB has a retrained receiver. Against the retrained equal split
   (87.65 %), β = 0.001 is infeasible (−0.82) and the rule would select β = 0.0003
   (53.6 symbols, test 86.88 %, +0.47 vs retrained equal). With that choice the IB is more accurate
   than our allocator (+~0.5) but uses 18 more symbols (53.6 vs 35.2): a trade-off, not a dominance.
   At equal symbols (~35) the allocator is far more accurate (IB β = 0.003: 82.7 % on validation vs
   ~86–87 % for the allocator).
2. λ was not re-selected on the retrained receiver (the stage-1 choice was reused).
3. Our receiver was trained with up to 14 tokens per task; the IB's with its fixed 6/5/5 and symbol
   trimming. Each receiver is trained for its own transmission scheme.
4. The bootstrap resamples individual cases i.i.d.; the same sentence appears at 10 SNRs × 3 seeds,
   so cases are correlated and the CIs are likely too narrow. A sentence-level (cluster) bootstrap
   would be more conservative.
5. The IB is a single-task method adapted to three tasks (one cut-off network per task, fixed equal
   token split); its minimum is 16 symbols (1 per token) vs 12 for ours.
6. Rayleigh not yet run for the baseline and the control.

## 9. Claims to verify

- (a) With equal receivers, the allocator keeps the accuracy of the equal split with 45 % fewer
  symbols; the IB, with the β the ε rule selects against the same receiver, is ~0.5 points more
  accurate with 16 % fewer symbols.
- (b) At the same number of symbols, the allocator is more accurate than the IB, especially at
  small budgets.
- (c) When ESG is given high priority, the allocator gives ESG higher accuracy than the IB at −15
  and −10 dB (+4 to +8 points depending on β), with fewer symbols; at 0 dB the IB is slightly better.
- (d) Is the comparison fair after the same-receiver control, and are issues 1–6 handled adequately?
