# Interval feasibility and trading economics

Measured 2026-10-03 against the frozen OOS artifacts
(`data/artifacts/preds_lstm_1h.parquet`, 180,975 rows, 2025-12-02 → 2026-09-30,
7,239 bars × 25 symbols). Nothing was retrained, refitted or re-tuned to produce
these numbers; they are read-only measurements of the existing model.
Reproduce with:
```
py -u tools/horizon_decay.py        # does any horizon carry information
py -u tools/cross_section_edge.py   # cross-sectional rank correlation and spread
py -u tools/edge_economics.py       # edge rate, cost structure, break-even hold
py -u tools/horizon_spread.py       # edge per trade at every holding horizon
```

## 1. Data availability, verified against the live Gate API

| Interval | Status | Retention start | Days | Bars/symbol |
|---|---|---|---|---|
| 15m | absent locally, fetchable | ~2026-06-22 | 103 | 9,888 |
| 30m | absent locally, fetchable | ~2026-03-09 | 208 | 9,984 |
| 1h | present, 216 symbols | ~2025-09-29 | 369 | 8,856 |
| 2h | derivable losslessly from 1h | ~2025-09-29 | 369 | 4,428 |
| 4h | derivable losslessly from 1h | ~2025-09-29 | 369 | 2,214 |
Local Parquet store also holds `1d` (216 symbols, 2013 → 2026) and a vestigial
`4h` slice covering only 2 symbols, which is not usable as a cross-section.
**Retention is the binding constraint on interval coverage.** 15m and 30m have
roughly the same *bar count* as 1h but cover 3.6× and 1.8× less calendar time, so
a 15% walk-forward holdout yields only 15 and 31 days of test data respectively.
At the product's 5 trades/day that is 77 and 156 test trades.

## 2. Measured edge

Cross-sectional long-top3 / short-bottom3, ranked by model score:

| Quantity | Value |
|---|---|
| Gross spread, 1-bar hold | +1.13 bps/bar |
| Cross-sectional rank ρ, within timestamp | +0.0037 |
| Permutation test vs. random assignment | **p = 0.578** |
| corr(predicted_return, target_return) | **+0.0001** |
**The model's ranking is not distinguishable from a random assignment of the same
score values.** Three independent measures agree: the permutation test above, the
within-timestamp rank correlation, and the raw correlation between the predicted
and realised return.
Trailing volatility alone ranks better than the model (+2.69 bps/bar, permutation
p = 0.078) but does not survive at the 5% level either.
Per-bar AUC by horizon: 1h `0.5036`, 2h `0.5049`, 4h `0.5035` — a spread of
0.0014 across horizons, i.e. flat. **There is no horizon at which this model
becomes meaningfully better.**
> **Correction.** An earlier version of this document reported the edge as real
> on the strength of a day-block bootstrap CI of [+0.24, +2.03] bps. That test
> asks whether the mean spread is non-zero, which is the wrong null: a small
> positive mean arises from the score's dispersion interacting with the return
> distribution even when the ordering carries no information. The permutation
> test controls for this and returns p = 0.578. The edge is **not** established.
> **Second correction.** An even earlier run reported "AUC CI excludes 0.5" as a
> passing release gate. That used an i.i.d. row bootstrap on 7,239 heavily
> overlapping hourly bars and is invalid. The gate has been replaced with a
> day-block bootstrap.

## 2b. Root cause: directional supervision is disabled

The defect audit (`tools/model_defect_audit.py`,
`tools/ranking_permutation_test.py`, and the training-call inspection below)
traced the absent signal to a specific, fixable bug.
`tools/one_year_sim.py:356` calls
```python
model, info = trainer.train(
    train, val, columns, target_column="target_return",
    seq_len=seq_len, horizon=horizon, kind=kind, epochs=epochs, seed=seed,
)
```
It passes neither `direction_column` nor `aux_direction_weight`. Both default to
`None` and `0.0`, and `src/cqip/forecaster.py:574` then evaluates
`use_direction = direction_column is not None and aux_direction_weight > 0.0`,
which is `False`. The guard at `forecaster.py:650` therefore never fires and the
auxiliary BCE loss is **never applied during training**.
The only loss is the heteroscedastic Gaussian NLL
`0.5 * (log_var + (target - mean)**2 / exp(log_var))`. When returns are close to
unpredictable this objective is fully satisfied by a constant `mean` plus a good
`log_var`: the variance head carries all the achievable loss reduction and the
mean head has no pressure to become informative. Checkpoint selection compounds
this, because `forecaster.py:677` selects on `val_metrics["loss"]`, which a
collapsed mean head still scores well.
The forecast that emerges is a drifting, wrong-signed drift estimate:

| Period | Model's predicted mean | Realised mean |
|---|---|---|
| 2025-12 | −1.84 bps/bar | +0.06 bps/bar |
| 2026-02 | −11.08 bps/bar | " |
| 2026-03 | −18.93 bps/bar | " |
| 2026-06 | −15.84 bps/bar | " |
| 2026-09 | −4.59 bps/bar | " |
Pooled, the model forecasts −7.93 bps/bar while the test window realised
+0.06 bps/bar. The forecast magnitude is not noise around zero; it is a confident,
systematically wrong drift term inherited from a −2.84 bps/bar training window
(2025-10-01 → 2025-12-02).
So the traded signal is, in substance, the residual noise of a mean head that was
never given a reason to learn direction.

## 2c. Secondary defects

- **The score cancels the only plausible factor.**
  `score = predicted_return / (target_volatility * vol_scale)`. Within-timestamp
  rank correlation between `score` and trailing volatility is −0.0138, i.e. zero
  by construction. Volatility is the one factor showing a suggestive effect, and
  normalising it away makes the strategy structurally incapable of expressing it.
- **The trained variance head is discarded.** The forecaster learns `log_var`, but
  the prediction artifact persists only `score`, `predicted_return`,
  `target_return`, `target_volatility`, `vol_scale`. There is no predicted
  volatility, and `one_year_sim.py:397` substitutes trailing realised volatility
  in its place.
- **The probability layer is a plug-in, not a calibration.** ECE is 0.2187. The
  bottom decile claims 4.4% and realises 48.4%; the top decile claims 94.9% and
  realises 49.6%. The module docstring's claim that `vol_scale` "makes Φ(score)
  an honest probability" is false; no isotonic or Platt map exists anywhere in the
  pipeline.
Confirmed *not* defective: the target itself
(`log(close.shift(-h)/close)`, no leakage), `target_volatility` (trailing-only; a
previous look-ahead was already fixed here), the per-symbol fold row slicing
(`prepare_symbol` reads one symbol sorted by ts, so `iloc[:val_start]` is
correct), and the purge/embargo geometry.

## 3. Edge decays faster than it accrues

Efficiency = (gross bps at k bars) / (k × gross bps at 1 bar). 1.0 means the edge
accrues linearly with holding time.

| Hold | Gross bps | Net bps (after 36 bps round trip) | Hit rate |
|---|---|---|---|
| 1h | +1.13 | −34.87 | 51.5% |
| 2h | +1.97 | −34.03 | 51.9% |
| 4h | +3.33 | −32.67 | 52.0% |
| 8h | +6.15 | −29.85 | 53.0% |
| 24h | +14.27 | −21.73 | 54.8% |
| 96h | +42.06 | +6.06 | 54.1% |
Efficiency falls to 0.74 at 4 bars and 0.39 at 96 bars. The model has a
next-bar tilt and little beyond it, so longer holds are progressively dead
weight — yet the edge is too small to pay for even one bar of cost.
Every horizon from 1h to 48h is net-negative. The first horizon to clear costs is
**96 hours**, and its CI includes zero, so it is not a reliable result.

## 4. The cost arithmetic that decides the product

Cost is a fixed ~36 bps per round trip (10 bps fee + 5 bps slippage + 3 bps
half-spread, per fill, ×2). The edge per bar is ~1.13 bps and decaying.
- Break-even hold at the point estimate: **~32 hours**.
- At 4h holds, gross edge is 3.33 bps against 36 bps of cost → **−32.7 bps/trade**.
- To break even at a 4h hold, round-trip cost would have to fall below **3.3 bps**,
  which is below any realistic crypto execution. **Cost reduction alone cannot
  rescue the requested holding range.**
Trade frequency is bounded by 6 slots × 24h = 144 position-hours per day:

| Target trades/day | Max hold that fits | Net bps/trade | Verdict |
|---|---|---|---|
| 1 | 144.0h | +126.6 | viable |
| 2 | 72.0h | +45.3 | viable |
| 3 | 48.0h | +18.2 | viable |
| 4 | 36.0h | +4.6 | marginal |
| 5 | 28.8h | −3.5 | loss-making |
| 6 | 24.0h | −8.9 | loss-making |
| 8 | 18.0h | −15.7 | loss-making |
| 10 | 14.4h | −19.7 | loss-making |

This is the theoretical best case: all 6 slots held continuously and rotated at
the maximum rate that still fits the requested frequency. **At 5 trades/day and
above the book cannot pay for itself at any holding period.**

## 5. A/B result: directional supervision is wired correctly and learns the null

§2b identified a specific defect — the auxiliary direction head is never trained —
and proposed a one-fold A/B as the test. That A/B has now been run. Harness:
`tools/ab_direction_head.py`; raw predictions:
`data/artifacts/reports/ab/ab_{A_current,B_directional}.parquet`; metrics:
`data/artifacts/reports/ab_direction_head.json`.
**Design.** One training event per variant on a single cutoff (bar 5100, seed 7, 12
epochs, `seq_len` 64, `val_ratio` 0.25, `aux_direction_weight` 1.0). Both variants
get byte-identical data, split, seed and evaluation; the only difference is
`direction_column="target_direction"` plus the weight. The universe is pinned to
the same 25 symbols the production hourly run traded. This is a single-cutoff A/B,
**not** a reproduction of the production 720-bar refit loop, and is reported as
such.
Train `2025-10-01 → 2026-03-09`, validation to `2026-05-01`, OOS
`2026-05-01 11:00 → 2026-09-30 02:00` — **151 days, 3,640 timestamps, 91,000
symbol-bars per variant.** The single cutoff is deliberate: a 720-bar fold yields
~30 independent days, which cannot resolve an effect worth ~1 bps/bar.
Weight selection. `--mode probe` trained BTC across {0, 0.25, 1, 4} and found no
weight producing a useful correlation (−0.11, −0.06, +0.02, +0.01). 1.0 was taken
as a round, non-cherry-picked value. Note the probe also showed reported `val_loss`
becoming *more negative* as weight rises (−4.32 → −4.65), purely because a BCE term
added to a negative NLL lowers the total regardless of skill — which is why loss is
excluded from the decision rule.

**The defect is confirmed fixed at the mechanism level.** All 25 B models report
`has_direction_head=True`. The auxiliary head trains and produces a real
probability.

### Ordering skill: still zero

| metric | A (current) | B (directional) | Δ |
|---|---|---|---|
| `corr(pred, target)` | −0.0077 | −0.0009 | +0.0068 |
| within-ts rank corr (score) | +0.0016 (se 0.0026) | −0.0034 (se 0.0029) | −0.0050 |
| permutation p (score) | 0.405 | **0.697** | — |
| permutation p (pred return) | **0.0145** (negative) | 0.551 | — |
| direction AUC | 0.5005 | 0.5023 | +0.0018 |
| balanced accuracy | 0.50015 | 0.50102 | +0.0009 |
B's ordering is indistinguishable from random, and slightly *worse* than A on the
score column, well inside one standard error. The permutation test on B fails
(p = 0.697).
One genuine improvement: A's raw `predicted_return` ranking was **significantly
anti-correlated** with realised returns (diff −6.18 bps, CI [−12.81, −1.01],
p = 0.0145). Direction supervision removed that pathology — it fell to p = 0.551.
The head eliminated a defect; it did not create signal.

### B's direction head learned the null, honestly

This is the most informative output of the A/B.

| bucket | claimed | observed | n |
|---|---|---|---|
| 0 | 0.4593 | 0.4863 | 9,100 |
| 4 | 0.4872 | 0.4976 | 9,100 |
| 9 | 0.5224 | 0.4946 | 9,100 |
**ECE 0.0140, BCE 0.69366 ≈ ln 2 = 0.69315, AUC 0.50225.** The head's claimed
probability spans only 0.459–0.522 while the observed rate is flat at ~0.49–0.50
in every decile. It learned precisely the correct null: *"I cannot tell the
direction; this is a coin flip at ~49.6%."* That is the model correctly reporting
that the supplied target and features carry no directional information.
For contrast, A's traded `Φ(score)` claims 0.0798 in its bottom decile and 0.9134
in its top while every observed bucket sits at ~0.49 — **ECE 0.2204, confidently
wrong at both ends.**

### Loss, drift and execution

| | A | B |
|---|---|---|
| val NLL (shared σ) | −3.0593 | −2.3511 |
| OOS NLL (shared σ) | −2.9510 | −2.0459 |
| pred mean bps/bar | −2.84 | −18.07 |
| realised mean bps/bar | +0.94 | +0.94 |
| drift bps/bar | −3.78 | −19.01 |
| execution net | **−72.61%** | **−19.93%** |
| execution gross (0 cost) | −17.61% | +60.62% |
| cost drag | 55.0 pp | 80.6 pp |
| round trips | 2,147 (14.2/day) | 1,349 (8.9/day) |
| Sharpe | −13.95 | −2.01 |
| max drawdown | −73.1% | −35.4% |
Drift did **not** improve at full scale: B drifts −19.0 bps/bar against A's −3.8.
(A 6-symbol, 2-epoch smoke had suggested drift improved; that was an artifact of
the tiny universe and an untrained model and is retracted.)
Both variants remain net-negative. B's superior net is not evidence of skill: its
gross +60.6% sits on 1,349 round trips whose 80.6 pp cost drag consumes all of it,
and the ordering test that would justify the gross figure returns p = 0.697. A
gross number with p = 0.697 behind it is the OOS window's drift, not alpha.

### Decision

The stated rule was: *if corrected ordering is approximately zero or fails the
permutation test, conclude that the present target/features/model formulation does
not extract usable directional signal.*
**B's rank correlation is −0.0034 and its permutation p is 0.697. The rule fires.**
Conclusion: **the current 1h target, feature set and LSTM formulation do not
extract usable directional signal on this universe.** The direction head was a
real defect and is now correctly wired, but enabling it changes the model's
*confidence*, not its *accuracy*. Calibration of B's probability layer is
therefore pointless — there is nothing to calibrate toward, and its ECE is already
0.0140 against a coin flip.
Per the rule, **calibration and the interval/trade-frequency sweep are not
proceeded with.** No full retraining was performed and no interval sweep was run.
If directional signal is to be pursued further, the change must be to the
*inputs*, not the loss: the label (e.g. volatility-adjusted or triple-barrier
forward return rather than raw next-bar return), the feature set, or the
target interval itself. Those are new experiments requiring fresh justification,
not a continuation of this one.

### Harness bugs found and fixed while building this

Recorded because each would have silently corrupted the A/B:
1. `realised` was computed as `log(close.shift(-1)/close - 1.0)` — log of a
   *simple* return, which is undefined (NaN) on every down-bar. This silently
   deleted ~50% of the sample and left a frame containing only up-moves, inflating
   apparent realised drift to 34 bps/bar and wrecking every downstream metric.
   Corrected to `log(close.shift(-1)/close)`.
2. The universe resolved to all 211 USDT pairs instead of the production 25,
   changing cross-sectional difficulty and making the run ~6 hours. Now pinned.
3. `_top_minus_bottom` returned exactly 0.0 when a cross-section had ≤ 2·TOP_N
   names (head and tail became the same rows). Now returns NaN, so the permutation
   test cannot report a fake zero.
4. `trainer.predict` discards `log_var` and the direction logit, so the actual
   training objective and the auxiliary probability were unmeasurable. A local
   `forward_all` helper now retains all three heads. Instrumentation only — no
   library code was modified.
5. Invariant checks were themselves wrong (`train_end_ts` pointed at the first OOS
   bar, making the no-lookahead check compare a value with itself; `score`
   consistency omitted the per-symbol `vol_scale`). Both fixed and now pass:
   alignment gap 1.2e-07, `score_consistent=True`, `oos_starts_after_train=True`,
   0 duplicate keys, 0 ties, 3,640 rows/symbol for all 25 symbols.

## 6. Conclusion

The requested product — 5–10 trades/day, 15m–4h holds, net after realistic costs
— is **arithmetically incompatible with the edge this model produces.** The
binding constraint is a signal worth ~1.1 bps/bar against a ~36 bps round trip,
and that signal is in any case indistinguishable from random ordering.
Corroboration: the full execution replay found cost coverage peaking at 0.90 with
0 of 24 configurations profitable, positive-Sharpe, or beating equal weight. The
best configuration's 32.6h average / 22h median hold independently arrived at the
same ~32h break-even the arithmetic predicts, and its headline +33.55% gross is a
low-turnover path on a diffuse ~1 bps/bar spread, not evidence of alpha.
**The audit found a specific and fixable root cause for the absent
signal**, documented in §2b: the auxiliary direction head is never trained because
`one_year_sim.py` omits `direction_column` and `aux_direction_weight`, and the
model is scored and selected purely on Gaussian NLL, which a collapsed mean head
satisfies.
**That A/B has now been run (§5). The defect was real and is fixed, and it was
not the reason the signal was missing.** With directional supervision enabled, all
25 models train the auxiliary head, the head reports the base rate almost exactly
(BCE 0.6937 ≈ ln 2, ECE 0.0140), and cross-sectional ordering remains
indistinguishable from random: rank correlation −0.0034, permutation p = 0.697.
So the binding constraint is not a disabled loss term. It is that **this target,
these features and this architecture do not extract directional information from
1h crypto returns at all.** The model now says so honestly and in calibrated form,
which is a better system than one that claimed 0.91 probabilities on 0.49
outcomes, but it is not a profitable one.
Retraining at 15m and 30m was deliberately **not** performed. Reasons:
1. It would not address the binding constraint, which is edge-per-bar, and
   horizon AUC is flat (0.5035–0.5049) across every measurable horizon.
2. Retention caps those intervals at 15 and 31 days of test data — windows too
   short to resolve an effect of this size, at a cost of hours of training.
3. The user asked that retraining happen only where necessary, with justification.
4. After the A/B in §5 there is no directional skill at 1h to carry into another
   interval; every interval would inherit the same empty result.
Status of the original priority list: step 1 (enable directional supervision and
re-validate) is **done — it did not produce signal**. Steps 2 and 3 (permutation
gate, then intervals) are **not proceeded with**, because the gate failed.
Recommended next step, if work continues: change the *inputs* rather than the
loss. The label is the first candidate — a volatility-adjusted or triple-barrier
forward return instead of the raw next-bar return — then the feature set, then the
target interval. Each needs its own justification and its own single-fold test
before any sweep.
