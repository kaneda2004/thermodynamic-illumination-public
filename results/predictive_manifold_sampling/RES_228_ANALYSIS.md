# RES-228: Order-Predictive Sampling - Detailed Analysis

## Experiment Overview

**Hypothesis**: Using effective dimensionality (eff_dim) measured during nested sampling every 20 iterations as feedback to dynamically adjust n_live achieves ≥2× speedup to reach order > 0.5 while maintaining success rate ≥90%.

**Status**: REFUTED (speedup 1.00x, success 26.7%)

## Results Summary

| Metric | Baseline | Adaptive | Target | Status |
|--------|----------|----------|--------|--------|
| Avg Samples | 247.5 | 247.5 | N/A | — |
| Speedup | — | 1.00x | ≥2.0x | ✗ FAILED |
| Success Rate | 26.7% | 26.7% | ≥90% | ✗ FAILED |
| N=15 CPPNs | 4 succeeded | 4 succeeded | — | — |

## Key Finding: Backward Adaptive Logic

The adaptive algorithm measured eff_dim values during sampling and made adjustment decisions. Analysis of the eff_dim histories reveals a critical issue:

### Observed Eff_Dim Behavior

Successful CPPN (order 0.5198):
- **Only 100 samples needed** (below target 300-iteration max)
- **Zero eff_dim measurements** (succeeded too quickly for feedback loop)
- **Zero n_live adjustments**

Failed CPPNs (order ~0.45, below 0.5 target):
- **All had high eff_dim** (3.5-4.5, well above the 3.0 threshold)
- **Adaptive logic decreased n_live** (from 100 → 81 → 80)
- **This is the opposite of what helps** (high eff_dim suggests still exploring, not collapsing)

Example from CPPN 1 (failed):
```
Check 0: eff_dim=3.99, n_live=100 → adjusted to 90 (wrong direction!)
Check 1: eff_dim=4.15, n_live=90  → adjusted to 81 (wrong direction!)
Check 2: eff_dim=3.86, n_live=81  → kept at 80  (still high eff_dim)
Check 3: eff_dim=4.59, n_live=80  → kept at 80  (high eff_dim!)
Check 4: eff_dim=3.53, n_live=80  → kept at 80  (still high!)
```

## Analysis: Why This Didn't Work

### 1. **Inverted Signal Logic**
The adaptation rules were:
- High eff_dim (>3.5) → DECREASE n_live (exploit)
- Low eff_dim (<2.0) → INCREASE n_live (explore)

But the evidence shows:
- **High eff_dim** = weight space is diffuse/random, not well-structured
- **Low eff_dim** = weight space has collapsed to a low-D manifold

The relationship to order discovery is unclear. During sampling, if eff_dim stays high, it might mean:
- The posterior isn't collapsing (bad sign for exploitation)
- We're still in random exploration (need more samples)
- The adjustment should be neutral or maintain current n_live

### 2. **Measurement Window Too Small**
The 20-sample window for eff_dim computation is insufficient:
- Only ~6-10 eff_dim measurements per run (max 300 iterations)
- High noise in 20-sample weight statistics
- Insufficient history to detect meaningful trends

### 3. **Success Rate Problem**
The fundamental issue: **26.7% baseline success rate** is too low
- Only 4 out of 15 random CPPNs reached order 0.5
- This suggests order 0.5 is difficult to achieve without guidance
- Adaptive adjustments are meaningless when 73% fail regardless

### 4. **No Correlation Between Signal and Outcome**
- Successful CPPN: Found order quickly (100 samples), zero feedback loop
- Failed CPPNs: Had eff_dim measurements and adjustments, still failed
- This suggests eff_dim during sampling is decoupled from order discovery

## Why RES-221 and RES-218 Don't Predict This Result

**RES-221** (Zero-shot prediction): Pre-sampling eff_dim didn't predict order (r < 0.4)
- This already suggested eff_dim is not causally related to order

**RES-218** (Eff_dim collapse): Eff_dim drops during sampling for high-order CPPNs
- But this is a correlation, not causation
- The drop happens after order is achieved, not before
- Can't use it as a predictive signal

**RES-220** (Manifold sampling): 3-5× speedup from constrained PCA basis
- This works because it uses PRE-COMPUTED structure from successful CPPNs
- Not reactive feedback during sampling

## Implications

The experiment reveals that:

1. **Eff_dim is not a real-time diagnostic signal** for guiding adaptive sampling
2. **Reactive feedback during sampling** may be ineffective because:
   - The signal (eff_dim) lags the outcome (order achievement)
   - The weight space dynamics don't directly control order
3. **Order discovery is driven by something else** that eff_dim doesn't capture
4. **Static strategies** (like RES-220's pre-computed manifold basis) outperform **dynamic feedback**

## Recommendation

Rather than using eff_dim as a feedback signal, consider:

1. **Direct order measurement feedback** (instead of eff_dim proxy)
   - Every N samples, measure order of current best
   - Use order change rate as signal

2. **Variance-based feedback** (instead of absolute eff_dim)
   - Track eff_dim slope, not absolute value
   - Increase n_live if eff_dim is rapidly decreasing

3. **Multi-metric feedback**
   - Combine eff_dim with weight magnitude, connection count changes
   - Ensemble signal might be more predictive

4. **Focus on the 73% failure rate problem**
   - Order 0.5 is very difficult to achieve
   - Consider easier target (order 0.3-0.4) or larger n_live baseline
   - Investigate why RES-220 needs pre-computed manifolds to work well

## Conclusion

RES-228 tested whether effective dimensionality measured during sampling could serve as a feedback signal to guide adaptive n_live adjustments. The hypothesis was refuted: adaptive and baseline strategies were identical (1.00x speedup, 26.7% success rate).

Root cause: **Eff_dim is not a predictive signal for order discovery during sampling**. The high eff_dim values observed in failed runs suggest the weight posterior remains in high-dimensional random exploration space, unable to converge to the low-order manifold.

This suggests order discovery requires non-adaptive mechanisms (like pre-computed constraints from RES-220) or direct order measurement feedback, not indirect dimensionality signals.
