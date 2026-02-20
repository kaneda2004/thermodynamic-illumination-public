# Dual-Channel Architecture Exploration Summary

## Status: REFUTED (RES-240)
**Date:** 2025-12-19
**Domain:** dual_channel_architecture
**Related:** RES-239 (radial_basis - refuted), RES-235 (hierarchical_composition - refuted)

## Hypothesis
Using alternative coordinate transforms [x+y, x-y] instead of standard [x, y, r] reduces effective dimensionality (≤3.0D) and improves sampling efficiency (≥2× speedup).

## Method (RES-240)
Tested 4 coordinate variants with 40 CPPNs each:
1. Standard baseline: [x, y, r, bias]
2. Dual sum-diff: [x+y, x-y, bias]
3. Dual abs-diff: [x+y, |x-y|, bias]
4. Dual interaction: [x+y, x-y, x*y, bias]

Measured: initial effective dimensionality, sampling efficiency, order quality via nested sampling.

## Results

### Baseline (Standard [x,y,r])
- Effective dimensionality: 3.76D
- Mean samples to order ≥0.5: 37,812
- Success rate: 10%
- Mean order: 0.444

### Dual Sum-Diff [x+y, x-y]
- Effective dimensionality: 3.75D (no reduction)
- Mean samples: 38,853 (0.97× baseline)
- Success rate: 10% (no improvement)
- Mean order: 0.437

### Dual Abs-Diff [x+y, |x-y|]
- Effective dimensionality: 3.40D (slight reduction)
- Mean samples: 39,253 (0.96× baseline)
- Success rate: 2.5% (degraded)
- Mean order: 0.462

### Dual Interaction [x+y, x-y, x*y]
- Effective dimensionality: 4.01D (increased)
- Mean samples: 37,740 (1.00× baseline)
- Success rate: 12.5% (marginal improvement)
- Mean order: 0.440

## Conclusion: REFUTED

**Key Finding:** Coordinate transformation alone does NOT reduce effective dimensionality.

### Why Dual-Channel Failed
1. **Passive transformation limitation:** [x+y, x-y] are linear combinations of [x, y]; they don't fundamentally reduce information content
2. **Redundancy not removed:** Both dual channels correlate with original coordinates; no independent dimension reduction
3. **Sampling efficiency unchanged:** ~0.95-1.00× speedup indicates no real sampling advantage
4. **Success rate plateau:** All variants cluster around 10-12% success rate (no diversity benefit)

### Interpretation
Effective dimensionality is determined by **network architecture and nonlinearity**, not coordinate parameterization. Simply rewriting coordinates cannot overcome the fundamental constraint that CPPN parameter space is ~3-4 dimensional.

## Implications

### What This Teaches Us
- **Architecture matters more than coordinates:** Swapping [x,y,r] for [x+y, x-y] is equivalent to a linear preprocessing step; learned networks can absorb this transformation
- **Dimensionality is parameter-bound, not coordinate-bound:** A 4-input CPPN inherently samples ~3-4 dimensional manifolds regardless of input encoding
- **Necessary conditions for speedup:**
  - Must reduce actual parameter degrees of freedom, OR
  - Must increase manifold structure efficiency (e.g., gating, hierarchical composition)
  - Simple coordinate transformation insufficient

## Next Steps (Pursued Elsewhere)

### Alternative Approaches Tested/In-Progress
- **RES-241:** Full interaction composition set [x, y, r, x*y, x², y²] - VALIDATED (2.48× improvement)
- **RES-239:** Radial basis architecture [r, θ] - REFUTED (no speedup)
- **RES-254:** Hierarchical multi-scale [x, x/2, x/4, y, y/2, y/4] - VALIDATED
- **RES-246:** Nonlinear interaction transfer across architectures - RUNNING
- **RES-250:** Information-theoretic bounds on speedup - VALIDATED

### Recommended Research Directions
1. **Learned coordinate systems:** Networks that discover optimal input encodings (AutoML for CPPN inputs)
2. **Manifold-aware sampling:** Structure exploitation via active learning on the order manifold
3. **Hierarchical decomposition:** Multi-scale basis functions that naturally capture order structure
4. **Spectral methods:** Frequency-domain analysis of order landscape (RES-249 in progress)

## Metrics Summary

| Metric | Standard | Dual-Sum | Dual-Abs | Dual-Int | Target | Status |
|--------|----------|----------|----------|----------|--------|--------|
| Eff Dim | 3.76 | 3.75 | 3.40 | 4.01 | ≤3.0 | MISS |
| Speedup | 1.00× | 0.97× | 0.96× | 1.00× | ≥2.0× | MISS |
| Success % | 10% | 10% | 2.5% | 12.5% | ≥60% | MISS |
| Order μ | 0.444 | 0.437 | 0.462 | 0.440 | ≥0.50 | MISS |

**Verdict:** All metrics fail to meet targets. Hypothesis REFUTED.

## Code Reference
- Experiment: `experiments/res_230_results.json` (RES-240)
- Implementation: Standard CPPN with coordinate swapping in `core/thermo_sampler_v3.py`

---

**This analysis closes the dual-channel architecture thread. Recommend focusing on interaction-enriched or hierarchical approaches that showed promise in RES-241 and RES-254.**
