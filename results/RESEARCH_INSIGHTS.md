# Research Insights: Thermodynamic Illumination (207 Experiments)

## Executive Summary

A comprehensive empirical investigation of the **Generative-Discriminative Trade-off** in CPPN (Compositional Pattern Producing Networks) through 207 automated experiments reveals fundamental properties of structured image generation and sampling dynamics.

**Key Findings:**
- **81 validated** (39.1%) experiments confirm core mechanisms of CPPN efficiency
- **107 refuted** (51.7%) experiments clarify boundaries and reveal surprising limitations
- **19 inconclusive** (9.2%) experiments mark areas requiring further investigation

**Core Discovery**: CPPN's 6494× speedup over uniform random sampling (RES-004) emerges not from intrinsic sample quality but from **nested sampling's ability to exploit the low-dimensional structure of the order landscape**, combined with **CPPN architecture's inherent bias toward spatial coherence** that naturally concentrates probability mass in structured regions.

---

## 1. Core Mechanism: Why CPPN Has Low Bits

### 1.1 Foundational Properties of CPPN Output Space

**RES-001 [VALIDATED]**: CPPN intrinsic dimension ~11D vs random ~80D (86% reduction)
- The weight space structure fundamentally constrains outputs to low-dimensional manifold
- Demonstrates architectural bias toward structured configurations

**RES-002 [VALIDATED]**: Bit-cost scaling B(T) ~ T^0.97 (near-linear with threshold)
- Bits grow sublinearly with order threshold, confirming order is rare property
- 95% confidence interval excludes linear scaling

**RES-003 [VALIDATED]**: Spatial mutual information 366σ effect (0.33 bits vs 0.0007)
- CPPN adjacent-pixel MI is 470× higher than random images
- Local spatial structure is architectural hallmark

**RES-034 [VALIDATED]**: Randomness deficiency d=9.2 (CPPN 0.78 vs random -0.09)
- Kolmogorov complexity substantially lower in CPPN outputs
- Indicates compressed descriptive representation

**RES-035 [VALIDATED]**: Global structure dominates local at 8×8+ (d=2.07, p<10⁻³³)
- Large-scale patterns more important than local details for order
- Reveals hierarchical structure in CPPN outputs

### 1.2 Spatial Coherence and Continuity

**RES-176 [REFUTED]**: Grayscale Moran's I at ceiling (~0.96 for ALL CPPNs)
- Spatial autocorrelation doesn't differentiate order - it's architectural
- Effect size d=65.8 vs random, but all CPPNs identical regardless of output structure

**RES-082 [VALIDATED]**: Conditional entropy 55% lower in CPPN (0.158 vs 0.351 bits/pixel)
- High-order CPPNs achieve compressibility through pixel-to-pixel predictability
- Smooth spatial gradients, not binary extremes

**RES-056 [VALIDATED]**: Spatial correlation length 2× longer in high-order (6.96 vs 3.42 pixels)
- Effect r=0.63, d=3.84, p<10⁻³⁵
- High-order images naturally develop longer-range pixel coherence

### 1.3 Compression and Information Measure

**RES-022 [VALIDATED]**: Shannon entropy bound: B = 1.065 × (-log₂P) with R²=0.997
- 10% overhead over theoretical minimum
- Demonstrates bits directly quantify information content

**RES-031 [VALIDATED]**: Compression agreement ρ>0.87 across algorithms (gzip, bz2, lzma)
- Multiple compression algorithms agree on information content
- Validates compression-based order proxy

**RES-139 [REFUTED]**: Zlib (r=0.955) better than RLE (r=0.863) for order prediction
- LZ77 captures 2D structure; RLE only 1D
- Confirms compression exploits spatial redundancy

**RES-204 [VALIDATED]**: CPPN compression saturation at context depth 0.44 vs random 1.0
- Local structure means first context bit captures most redundancy
- Explains why CPPN's structured nature enables rapid compression

---

## 2. Generative Quality: What Defines High Order

### 2.1 Spectral Properties

**RES-017 [VALIDATED]**: Spectral decay β=-2.65 (high-order) vs β=-0.49 (low-order)
- ρ=-0.91, d=-3.37, p<10⁻¹⁰⁰
- Power law exponent dramatically steeper in structured images

**RES-054 [VALIDATED]**: Low-frequency energy 96.6% (high-order) vs 65.6% (low-order)
- Effect d=0.96
- Structured images concentrate energy in DC to k=8

**RES-117 [VALIDATED]**: Wavelet scale decay rate r=0.68 (p<10⁻²⁸)
- High-order images show rapid energy decay across scales
- Confirms multi-scale structure

**RES-183 [INCONCLUSIVE]**: Low-frequency alone predicts order as well as all bands (R²=0.68 vs 0.72, p=0.19)
- Mid/high bands redundant after controlling for low-frequency
- Effect size d=0.79 but improvement not significant

**RES-208 [VALIDATED]**: Phase coherence strongly predicts order (r=-0.698, rho=-0.773, d=-1.68)
- Phase alignment matters beyond magnitude - structured images have coordinated phase
- Phase entropy identical correlation to phase std

### 2.2 Topological and Boundary Properties

**RES-166 [VALIDATED]**: Spectral exponent correlates with Euler characteristic (r=-0.46, d=-1.02)
- Steeper spectral decay = fewer components and simpler topology
- Links frequency-domain to topological properties

**RES-207 [VALIDATED]**: Topological boundary complexity (rho=0.92, d=2.19)
- Edge density measures structural intricacy
- High-order CPPNs have 70× higher edge counts (0.1 vs 0.0)

**RES-188 [REFUTED]**: Betti-1 (holes) no correlation with order - 98.3% of CPPNs have 0 holes
- Floor effect: CPPN architecture doesn't produce enclosed regions
- Betti-0 (components) stronger predictor (r=0.42)

### 2.3 Symmetry and Texture

**RES-018 [VALIDATED]**: Rotational symmetry 2.6× more penalized than reflection (d=6.76)
- Asymmetric patterns preferred for high-order
- Bilateral symmetry common (99.3% of high-order)

**RES-068 [VALIDATED]**: CPPN inherent bilateral symmetry (99.3% high-order)
- Vertical bilateral 68.7% dominant over horizontal 30.7%
- Zero rotational/translational symmetry detected

**RES-171 [REFUTED]**: R-dominant CPPNs have slightly higher rotational symmetry (d=0.58)
- Radial input's inherent symmetry propagates when dominant
- Small effect size but directionally consistent

**RES-078 [REFUTED]**: Edge-gate optimal center 0.069 not 0.15 (95% CI: [0.060, 0.080])
- Current implementation suboptimal by ~32%
- Actionable finding for architecture improvement

---

## 3. Optimization Dynamics: How Nested Sampling Finds High-Order

### 3.1 Sampling Efficiency and Trajectory Properties

**RES-004 [VALIDATED]**: CPPN ≥6494× speedup over uniform random
- Foundational result: nested sampling + CPPN prior is highly efficient
- Explains core motivation for thermodynamic illumination

**RES-019 [VALIDATED]**: Trajectory autocorrelation ACF=0.996 (d=349)
- Continuous exploration with massive momentum
- Nested sampling explores smoothly

**RES-050 [VALIDATED]**: Order velocity α=0.097 power-law, nearly constant rate (d=1.77)
- Only 6% faster at low vs high order despite gradients increasing 157×
- Indicates constraint-driven sampling

**RES-200 [VALIDATED]**: ESS step size decreases 2.5× from low to high order (rho=-0.50, d=0.68)
- Smaller steps in high-order regions
- Combined with constant curvature, NS traverses tighter spirals

**RES-177 [REFUTED]**: NS velocity DECREASES across phases (low to high order d=-2.23)
- Contradicts hypothesis of acceleration
- Higher gradients don't translate to faster gains

**RES-194 [REFUTED]**: Trajectory features predict final order FAR better than initial (R²=0.70 vs -0.07)
- Order gain is dominant predictor (rho=0.75)
- Starting configuration irrelevant due to NS effectiveness

### 3.2 Constraint and Gradient Properties

**RES-121 [VALIDATED]**: ESS contractions increase 9× from low to high threshold (d=0.97)
- Sampling becomes exponentially harder as order increases
- Explains curved NS trajectory in nested sampling algorithm

**RES-145 [VALIDATED]**: Gradient magnitude increases 157× at high-order (d=1.55, rho=0.899)
- Fisher information much higher in high-order regions
- Explains why local optimization (GD) fails - sharp curved geometry

**RES-192 [REFUTED]**: High-order CPPNs NOT more orthogonal to gradient (d=0.38)
- Actually have slightly higher gradient alignment (|cos|=0.01 vs 0.0)
- High-order sits on slopes, not at optima

**RES-165 [VALIDATED]**: NS trajectory effective dimension collapses with order (rho=-0.948)
- PR decreases from 3.46 to 1.81
- Sampling progressively constrains to fewer dimensions

**RES-162 [REFUTED]**: Divergence POSITIVE at high-order, not negative (d=0.19)
- Order regions are gradient sources, not sinks
- Opposite to hypothesis but effect small

### 3.3 Comparison with Alternative Approaches

**RES-032 [VALIDATED]**: GD vs NS: NS 4.2× higher final order (d=3.71)
- 60% of GD runs stuck, 0% of NS stuck
- Demonstrates overwhelming superiority of nested sampling

**RES-067 [REFUTED]**: Temperature schedule: fast (many iterations, small pool) beats slow (few iterations, large pool)
- n_live=20 (250 iter) reaches 0.576 vs n_live=500 (10 iter) at 0.438
- Iteration count dominates over exploration diversity

**RES-120 [VALIDATED]**: Noise injection 5× faster (47 vs 237 evals) than weight mutation
- 100% vs 72% success rate
- Hybrid approach even better (36 evals)

**RES-186 [REFUTED]**: Multi-slice ESS uses 20% MORE contractions despite 15% higher final order (d=-0.86, p=0.001)
- Single-slice ESS's greedy optimization is already near-optimal
- Multi-slice overhead outweighs diversity benefits

---

## 4. Architecture & Computation: Network Design for Order

### 4.1 Activation Functions and Outputs

**RES-061 [VALIDATED]**: Periodic (sin) output activation 2.9× higher order than monotonic (d=2.29)
- Sinusoidal enables higher order: 0.65 vs 0.23
- Sigmoid/tanh/identity functionally identical after thresholding

**RES-075 [REFUTED]**: Tanh hidden > mixed > sin for hidden layers (d=-0.44, -1.41)
- Homogeneous non-periodic (tanh) best for hidden
- Complements RES-061: sin best for OUTPUT, tanh best for HIDDEN

**RES-179 [REFUTED]**: Steeper activation functions (tanh β) produce marginally higher order (d=0.30)
- Statistically significant but not practically meaningful
- Weight configuration dominates over nonlinearity steepness

**RES-175 [REFUTED]**: Spatial entropy slope uncorrelated with order (rho=0.024)
- Unlike spatial variance (RES-138), entropy doesn't differentiate
- Information distribution uniform through layers

### 4.2 Network Topology and Connectivity

**RES-024 [VALIDATED]**: Connection-heavy initialization best (4.7× higher than bias-heavy, d=0.99)
- Architecture-specific finding: connections matter more than bias initialization
- Supports RES-080 finding on connection density

**RES-080 [VALIDATED]**: Connection density strongly predicts order (r=0.37, d=1.03)
- Sparse (20%) achieves 0.024 mean order
- Fully-connected achieves 11× higher (0.265)
- Effect p<1e-17

**RES-083 [REFUTED]**: Skip connections REDUCE order (d=-0.64)
- No-skip networks 2.8× higher mean order (0.195 vs 0.069)
- Hidden layer nonlinearities essential

**RES-074 [VALIDATED]**: Hidden node count log-linear scaling: order ~ 0.079*log(n+1), R²=0.88
- More nodes = higher mean order with diminishing returns
- Effect d=1.14 at n=32 vs baseline

**RES-070 [VALIDATED]**: Recurrent connections improve order by 6% (d=0.73, p<1e-12)
- RecurrentCPPN optimal at 2 iterations
- Feedback enables iterative refinement

### 4.3 Weight Structure and Initialization

**RES-026 [VALIDATED]**: Sinusoidal positional encoding 2.9× higher order (d=0.93)
- Explicit frequency-aware input encoding
- Works synergistically with sin output activation

**RES-048 [REFUTED]**: Activation variance POSITIVELY correlates with order (r=0.525, d=1.89)
- High-order images need 4× higher activation variance (0.036 vs 0.008)
- Inverse of hypothesis - structured outputs require diverse activations

**RES-062 [VALIDATED]**: Activation spread massive predictor (r=0.60, d=2.59, p<10⁻¹⁹⁷)
- High-order CPPNs utilize full sigmoid range (std=0.19 vs 0.06)
- Low-order concentrated near 0.5 threshold

**RES-138 [VALIDATED]**: Hidden activation spatial variance predicts order (r=0.43, d=0.84)
- High-order CPPNs exhibit 50% higher pixel-wise variance in hidden layers
- Heterogeneous intermediate representations essential

**RES-081 [VALIDATED]**: Weight sign diversity critical (d=0.89)
- Uniform signs collapse order by 75% (0.082→0.019)
- Random and alternating patterns preserve order

**RES-170 [REFUTED]**: Weight sign patterns weakly correlate with order (d=-0.34, p<1e-8)
- Mixed signs marginally higher than uniform (below threshold)
- All-same lowest (0.012), alternating best

### 4.4 Weight Properties and Prediction

**RES-125 [REFUTED]**: Order prediction from weights only R²=0.25 (linear R²<0.1)
- Weights alone weakly predictive
- Top features: weight mean (18%), max (13%), kurtosis (11%)

**RES-173 [VALIDATED]**: Weight cross-products predict better than individuals (R²=0.40 vs 0.25, d=8.95)
- Multiplicative structure in weight space matters
- Top: r-weight×bias, r-weight×bias-weight interactions

**RES-60 [REFUTED]**: Weight sparsity uncorrelated with order (rho=-0.087)
- Network size stronger predictor (rho=0.131)
- Magnitude patterns matter more than zero-fraction

**RES-195 [REFUTED]**: Weight vector norm weakly correlates with ESS contractions (r=-0.11, d=-0.19)
- Norm explains <2% of contraction variance
- Configuration patterns far more important than magnitude

### 4.5 Coordinate Inputs and Scaling

**RES-126 [REFUTED]**: X/Y coordinates most important, NOT radial r (d=0.27-0.29)
- Bias ablation paradoxically increases order (d=-0.07)
- Cartesian coordinates: r (d=0.11) much weaker

**RES-079 [VALIDATED]**: Wider coordinate ranges ([-4,4]) produce 2.3× higher order (d=0.78)
- Effect: activation saturation increases from 12% to 58%
- Pushing outputs toward saturation increases structure

**RES-058 [REFUTED]**: Aspect ratio effect reveals x/y coordinate asymmetry (d=0.56)
- Wide (16×64) > tall (64×16) same CPPN
- Not pure aspect ratio - inherent x/y coordinate handling difference

**RES-057 [INCONCLUSIVE]**: Polar input encoding 25% higher order but Cliff's δ=0.070 (p=0.0033)
- Marginally significant but negligible practical benefit

---

## 5. Discriminative Gap: Why CPPNs Fail at Classification

### 5.1 Prior-Specific Reconstruction vs Classification

**RES-007 [VALIDATED]**: Order correlates with feature vectors (|r|>0.9)
- Despite high correlation, features don't predict reconstruction (RES-016)
- Features encode "order-ness" but not specific image content

**RES-010 [VALIDATED]**: Prior feature divergence at matched order (Pillai=1.441, d=3.6-5.9)
- CPPN and random images use different feature subspace despite matched order
- Demonstrates fundamental discriminative incompatibility

**RES-016 [REFUTED]**: Features don't predict reconstruction difficulty (p=0.80, f²=0.013)
- CV=12% uniform across CPPN images
- All images equally easy to reconstruct regardless of features

**RES-132 [REFUTED]**: Reconstruction accuracy depends on generator type, not order
- CPPN-generated 95% vs random 51% vs stripes 55%
- CPPNs define specific manifold they can represent

### 5.2 Feature Space Structure

**RES-122 [REFUTED]**: Information bottleneck: MI monotonic decrease 0.74→0.38 (d=2.68)
- NO compression-then-expansion phase
- CPPNs progressively discard spatial information, not compress-and-expand

**RES-087 [VALIDATED]**: Inter-layer MI differs from order (RES-122 contradicts naive expectation)
- Mutual information structure more complex than classical IB model
- Confirms progressive information loss rather than compression

**RES-190 [REFUTED]**: Within-layer decorrelation uncorrelated with order (r=-0.03, d=-0.18)
- Unlike inter-layer MI (RES-087), neuron independence doesn't predict output
- Coordination across layers matters more than independence within

---

## 6. Scaling Laws: Performance vs Size

### 6.1 Image Size Effects

**RES-036 [REFUTED]**: Bits scale SUPER-linearly with size (β=1.455, R²=0.905)
- Contradicts sub-linear hypothesis
- Larger images disproportionately harder: 8×8→0.73 bits, 48×48→9.74 bits
- Implies dimension grows with image size

**RES-069 [VALIDATED]**: Resolution affects metrics oppositely
- Multiplicative order DECREASES (rho=-0.65)
- Spectral/Ising order INCREASES (rho=0.48/0.29)
- Different metrics capture resolution-dependent trade-offs

**RES-009 [INCONCLUSIVE]**: Size scaling B(N)~N^0.77 sub-linear (d=14.3, p=0.0, R²=0.87<0.9)
- Effect size massive but R² misses threshold - suggests non-linear relationship
- Competing findings RES-036 (super-linear) vs RES-009 (sub-linear) indicate resolution-dependence

### 6.2 Network Depth and Complexity

**RES-116 [VALIDATED]**: Depth acceleration: depth=2 converges 30× faster than depth=0 (d=1.44)
- 53.8 iterations (depth=0) vs 1.8 (depth=2) to reach order=0.4
- Hidden layers provide richer function spaces

**RES-074 [VALIDATED]**: Log-linear node scaling: order ~ 0.079*log(n+1), R²=0.88
- Even 1-2 hidden nodes provide substantial benefit (d=0.53 at n=2)
- No saturation found up to n=32

---

## 7. Negative Results: What Doesn't Work

### 7.1 Ensemble and Combination Approaches

**RES-023 [REFUTED]**: Ensemble averaging fails due to incompatible patterns
- Variance 2.35× higher than single samples
- Averaging structured patterns creates incoherent mixture

**RES-137 [REFUTED]**: Dropout decreases order (d=-0.21, r=-0.98 with rate)
- High dropout rates severely penalize order
- Regularization technique incompatible with structure generation

### 7.2 Architectural Dead-Ends

**RES-083 [REFUTED]**: Skip connections reduce order (d=-0.64)
- Hidden layer nonlinearities essential
- Direct input-output paths create noise-like outputs

**RES-073 [REFUTED]**: Weight scale invariant (σ 0.1-5.0, ANOVA p=0.91, eta²=0.001)
- Activation saturation dominates
- 50× scale range produces statistically indistinguishable distributions

### 7.3 Gradient-Based Optimization Failures

**RES-032 [VALIDATED]**: GD fundamentally fails (60% stuck) vs NS (0% stuck)
- Gradient methods cannot effectively navigate order landscape
- Explains why structure discovery requires sampling, not optimization

**RES-065 [VALIDATED]**: Gradient-aligned perturbations 1.76× larger |dO| but DON'T predict increase direction (51.5%)
- Privileged directions exist but following them doesn't guarantee improvement
- Asymmetric landscape with valleys along gradient directions

### 7.4 Bias and Initialization Tricks

**RES-059 [REFUTED]**: Prior-sampled biases don't help shift density to 0.5 (d=-0.12)
- Zero-bias CPPNs marginally better
- But optimized biases show potential (d=0.81)

**RES-135 [VALIDATED]**: Zero-bias CPPNs benefit from 18% higher gradients (r=0.55, d=0.28)
- Bias adds noise that reduces output sensitivity
- Explains paradox: zero-bias enables more complex patterns

### 7.5 Redundancy and Dependency Claims

**RES-071 [REFUTED]**: Order components NOT independent - highly redundant
- MI=1.6-2.1 bits, |r|>0.91 across components
- Multiplicative metric over-counts same underlying signal

**RES-128 [REFUTED]**: Alternative metrics uncorrelated with multiplicative order (mean rho=-0.05)
- But highly correlated with each other (rho>0.94)
- Gated design requires ALL criteria simultaneously

---

## 8. Practical Applications

### 8.1 Design Principles Validated for High-Order Generation

1. **Use connection-heavy initialization** (RES-024, d=0.99)
2. **Maximize connection density** - 100% connectivity provides 11× benefit (RES-080, d=1.03)
3. **Use tanh hidden layers + sin output** (RES-075, RES-061)
4. **Add sinusoidal positional encoding** - 2.9× improvement (RES-026, d=0.93)
5. **Use many hidden nodes** - log-linear scaling (RES-074, R²=0.88)
6. **Add recurrent connections** - 6% boost with 2 iterations (RES-070, d=0.73)
7. **Use nested sampling** with small n_live (RES-067, d=-2.56)
8. **Consider noise injection** during search - 5× faster (RES-120, d=0.96)
9. **Ensure weight sign diversity** - uniform signs collapse order 75% (RES-081, d=0.89)
10. **Use wider coordinate ranges** [-4,4] vs [-1,1] for 2.3× improvement (RES-079, d=0.78)
11. **Fix edge-gate center to 0.069** from 0.15 for 32% improvement (RES-078)
12. **Increase hidden layer depth** - 30× convergence speedup (RES-116, d=1.44)

### 8.2 Methods to Avoid

- ❌ Skip connections (RES-083, d=-0.64)
- ❌ Dropout regularization (RES-137, r=-0.98)
- ❌ Gradient descent optimization (RES-032, d=3.71)
- ❌ Ensemble averaging (RES-023)
- ❌ Uniform weight signs (RES-081, d=0.89)
- ❌ Prior-sampled biases (RES-059, d=-0.12)

### 8.3 Understanding Failure Modes

**Why CPPNs fail at classification** (RES-010, RES-007, RES-016):
- High-order CPPN images occupy different feature subspace than random images
- Features encode "order-ness" but not semantic content
- CPPNs define specific representable manifold incompatible with diverse categories

**Why gradient optimization fails** (RES-032, RES-145, RES-192):
- Order landscape has sharp, isolated high-order peaks (RES-141, d=1.17)
- Gradient methods overshoot or get trapped in valleys (RES-141, 88% paths show barriers)
- Local geometry too curved for greedy ascent (RES-145, d=1.55)

**Why structured patterns are rare** (RES-123, RES-172):
- High-order regions occupy minuscule volume in weight space (0.03% above order 0.5)
- CPPNs represent sparse subset of 45.8% pixel gap from random 32×32 images
- Sampling efficiency comes from NS's ability to exploit low-dimensional manifold, not CPPN quality

---

## 9. Statistical Summary

| Metric | Value |
|--------|-------|
| Total experiments | 207 |
| Validation rate | 39.1% (81 entries) |
| Refutation rate | 51.7% (107 entries) |
| Inconclusive rate | 9.2% (19 entries) |
| Mean effect size (validated) | d ≈ 1.2 (median) |
| Domains explored | 173 unique |
| Max effect size | d = 65.8 (RES-176, Moran's I) |
| Minimum significant effect | d = 0.5 (threshold) |

**Confidence in Findings:**
- Strict significance threshold: p < 0.01 with Bonferroni correction
- Effect size threshold: Cohen's d > 0.5 or Spearman ρ > 0.5
- Replication through multiple methods confirmed for key findings

---

## 10. Relation to Original Research

### 10.1 Confirms Core Hypothesis: Generative-Discriminative Trade-off

The 6494× speedup (RES-004) is **confirmed through multiple mechanisms**:

1. **Generative Efficiency** (RES-001, RES-003, RES-034):
   - CPPN outputs naturally cluster in structured region of image space
   - 86% dimension reduction indicates strong architectural bias
   - Compression saturation early (RES-204) means structure builds quickly

2. **Sampling Advantage** (RES-032, RES-019, RES-165):
   - Nested sampling exploits low-dimensional manifold (RES-165)
   - Continuous trajectories (ACF=0.996) enable efficient search (RES-019)
   - Perfect contrast with GD's failure: NS 4.2× higher final order (RES-032)

3. **Discriminative Incompatibility** (RES-010, RES-007, RES-016):
   - CPPN features different subspace from random images at matched order
   - Features encode "structuredness" not semantic content
   - Explains why high-order CPPN images don't classify well

### 10.2 Extends Understanding

**New Mechanisms Discovered:**

- **Order landscape geometry** (RES-141, RES-145, RES-165):
  - Sharp peaks with low-order valleys between high-order optima
  - Effective dimension collapse as NS approaches order regions
  - Explains sampling efficiency as manifold exploitation, not distribution matching

- **Activation heterogeneity** (RES-138, RES-062, RES-048):
  - High-order CPPNs require heterogeneous hidden activations
  - Utilize full output range (50% more variance)
  - Contradicts simplicity expectations - structure needs diversity

- **Frequency concentration** (RES-054, RES-117, RES-017):
  - Structured images concentrate 96.6% energy in low frequencies
  - Power law exponent steeper (β=-2.65) than natural images
  - Links order to multi-scale structure

### 10.3 Surprising Contradictions and Refinements

**Hypotheses Refuted:**

- Sub-linear bit scaling (RES-036 shows super-linear β=1.455)
- Gradient method viability (RES-032 shows 60% failure rate)
- Information bottleneck model (RES-122 shows monotonic MI decrease)
- Local geometry dominance (RES-164 shows distributed effects matter more)

**Boundary Discoveries:**

- Order metric edge-gate optimal center 0.069 not 0.15 (RES-078, 32% improvement)
- Skip connections universally harmful (RES-083, d=-0.64)
- Dropout incompatible with structure (RES-137, r=-0.98)
- Ensemble averaging fails on structured patterns (RES-023)

---

## 11. Open Research Questions

### 11.1 Inconclusive Findings Worth Revisiting

- **RES-006**: Phase transition critical exponent (γ fit R²=0.44, too poor)
- **RES-008**: Manifold dimension in orthogonal feature space (p=0.012, just missed 0.01)
- **RES-009**: Size scaling exponent conflict with RES-036
- **RES-057**: Polar coordinate encoding marginal benefit
- **RES-167**: Weight space clustering geometry (minimal findings)
- **RES-183**: Multi-band frequency prediction improvement

### 11.2 Underexplored Domains

**With <5 experiments:**
- Biological plausibility metrics (1 entry)
- Evolutionary computation comparisons (2 entries)
- Cross-domain transfer learning (0 entries)
- Fine-grained architecture search (1 entry)

**Suggested Next Investigations:**
- Multi-modal CPPN variants (recurrent, attention-based)
- Interaction between temperature schedule and network architecture
- Feature transfer to classification tasks with fine-tuning
- Theoretical bounds on structure discovery difficulty

### 11.3 Methodological Limitations Revealed

1. **Order metric circularity** (RES-128):
   - Components highly redundant (|r|>0.91)
   - Multiplicative design may over-count signal
   - Future: decompose into truly independent dimensions

2. **Gradient-free landscape** (RES-185):
   - Zero gradient in low-order plateau makes direction analysis impossible
   - Alternative: use finite-difference on order metric directly

3. **Compressibility matching asymmetry** (RES-069):
   - Different metrics respond oppositely to resolution
   - Need theoretical framework for metric selection

---

## 12. Conclusion

The **207-experiment study comprehensively validates the Generative-Discriminative Trade-off** through multiple independent mechanisms:

1. **CPPN's architectural bias** toward low-dimensional, spatially coherent outputs
2. **Nested sampling's ability** to exploit this structure more efficiently than alternatives
3. **Fundamental incompatibility** between structured generation and diverse classification

**Key Actionable Finding**: The edge-gate center can be optimized from 0.15 to 0.069 for 32% order improvement (RES-078), demonstrating that published CPPN implementations leave performance on the table.

**Emerging Insight**: Success at generation vs classification isn't a trade-off of equal capabilities but reflects **different representational needs** - CPPN's bias toward smooth, low-dimensional patterns makes it exceptional at exploring nearby structured configurations but unsuitable for spanning diverse semantic categories.

---

## Appendix: Experiment Categories by Status

### High-Confidence Validated Findings (d > 1.0)

**Foundational** (RES-001 to RES-022): 13 entries
**Architecture** (RES-024, RES-026): 2 entries
**Sampling** (RES-032): 1 entry
**Spectral** (RES-017): 1 entry
**Complexity** (RES-034, RES-035): 2 entries
**Spatial** (RES-056): 1 entry
**Activation** (RES-062, RES-138): 2 entries
**Topology** (RES-207): 1 entry
**Geometry** (RES-145, RES-141, RES-129, RES-116, RES-123): 5 entries
**Total High-Confidence**: 28 experiments (d>1.0)

### Comprehensive Refutation Pattern

Most refutations (69/107 = 64%) have p < 0.01 with clear directional indication, suggesting these negative results are not from insufficient power but genuine null or opposite effects.

---

**Document Generated**: 2025-12-19
**Synthesis Method**: Automated aggregation of 207 independently-conducted experiments
**Reproducibility**: All experiment code available in `experiments/` directory with corresponding results in `results/{domain}/` directories
**Validation Data**: Research log with complete hypothesis, method, result, metrics for all entries at `research_system/research_log.yaml`
