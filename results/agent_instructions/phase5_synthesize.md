# Phase 5: Synthesize Findings into Insights Document

**Purpose**: Generate comprehensive insights document relating all 207 findings to the original Generative-Discriminative trade-off discovery.

## Environment Setup

- **Working directory**: `/Users/matt/Development/monochrome_noise_converger`
- **Input file**: `results/audit/research_index.json`
- **Output file**: `results/RESEARCH_INSIGHTS.md`

## Input

All 207 research entries are complete with:
- Hypothesis
- Code location
- Results JSON file
- Metrics (p_value, effect_size, etc.)
- Status (validated, refuted, inconclusive)

## Process

### Step 1: Load research_index.json

```python
import json
with open('results/audit/research_index.json') as f:
    all_entries = json.load(f)
```

### Step 2: Group findings by theme

Create 6 major sections based on research themes:

1. **Core Mechanism**: Why CPPN has low bits
   - RES entries about architecture, weight structure, dimensionality

2. **Generative Quality**: What defines high order
   - RES entries about order metrics, MI, spectral properties, compression

3. **Discriminative Gap**: Why CPPNs fail at classification
   - RES entries about feature space, linear separability, bottleneck

4. **Optimization Dynamics**: How NS finds high-order images
   - RES entries about NS vs GD, ESS, trajectories, curvature

5. **Scaling Laws**: Performance vs size
   - RES entries about image size, network depth, sample efficiency

6. **Negative Results**: What doesn't work
   - Refuted hypotheses showing boundaries and limitations

### Step 3: Extract top findings per theme

For each theme:
- Filter entries by domain or keywords
- Sort by effect size (validated first)
- Select top 5-10 validated findings
- Note key refuted findings that define boundaries

### Step 4: Generate markdown document

Use template below. Fill in:
- Key findings with RES-ID and effect sizes
- Connections between findings
- Practical implications

### Step 5: Save to results/RESEARCH_INSIGHTS.md

```python
with open('results/RESEARCH_INSIGHTS.md', 'w') as f:
    f.write(markdown_content)
```

## Document Template

```markdown
# Research Insights: Thermodynamic Illumination
## 207 Experiments on CPPN Priors

### Executive Summary

This document synthesizes 207 experiments exploring the Generative-Discriminative trade-off discovered in CPPNs (Compositional Pattern Producing Networks).

**Original Finding**: CPPNs efficiently find structured images despite requiring fewer bits than random sampling—they generate high-order images while failing at classification tasks.

**This Research**: 207 systematic experiments validate, extend, and refute specific hypotheses about this phenomenon.

**Results**:
- 81 validated findings (39%)
- 107 refuted findings (52%)
- 19 inconclusive findings (9%)
- 173 unique research domains explored

---

## 1. Core Mechanism: Why CPPN Has Low Bits

[Summary of 5-10 validated findings showing why CPPNs find low-bit solutions]

**Key Validated Findings:**
- RES-001: CPPN ~11D vs random ~80D (86% dimension reduction)
- RES-074: Network depth follows log-linear scaling (R²=0.88)
- RES-080: Connection density strongly predicts order (r=0.37)
- RES-061: Periodic activations 2.9× more effective (d=2.29)
- RES-024: Connection-heavy initialization best (d=0.99)

**Mechanism Summary**: [1-2 paragraphs connecting findings]

---

## 2. Generative Quality: What Defines High Order

[Summary of validated findings about what makes images "ordered"]

**Key Validated Findings:**
- RES-003: Spatial MI 366σ higher (0.33 vs 0.0007 bits)
- RES-056: Correlation length 2× longer (r=0.63, d=3.84)
- RES-017: Spectral decay β=-2.65 (ρ=-0.91)
- RES-204: Compression saturation at 0.44 depth vs 1.0 random

**Metric Composition**: [Explain what order_multiplicative captures]

---

## 3. Discriminative Gap: Why CPPNs Fail at Classification

[Summary of findings explaining the trade-off]

**Key Findings:**
- RES-007: Feature correlation r>0.9 (VALIDATED)
- RES-010: Feature divergence Pillai=1.441 (VALIDATED)
- RES-122: Information bottleneck μ=0.74→0.38 (REFUTED)

**Why Structure ≠ Linear Separability**: [Explanation]

---

## 4. Optimization Dynamics: How NS Works

[Summary of findings about sampling and optimization]

**Key Validated Findings:**
- RES-032: NS 4.2× higher order than GD (d=3.71)
- RES-141: Order barriers between optima (d=1.17)
- RES-049: Curvature 157× higher in high-order (ρ=0.899)
- RES-019: Trajectory ACF=0.996 (smooth exploration)

**Why Nested Sampling Succeeds**: [Explanation]

---

## 5. Scaling Laws: Size and Complexity

[Summary of findings about how properties scale]

**Key Findings:**
- RES-009: Bits scale super-linearly β=1.455 (REFUTED sub-linear claim)
- RES-004: 6494× speedup over uniform (VALIDATED)
- RES-116: Network depth 30× speedup convergence (d=1.44)

**Practical Scaling Constraints**: [Discussion]

---

## 6. Negative Results: What Doesn't Work

[Summary of refuted hypotheses—important for understanding boundaries]

**Key Refuted Findings:**
- RES-023: Ensemble averaging fails (incompatible patterns)
- RES-137: Dropout decreases order (d=-0.21)
- RES-083: Skip connections reduce order (d=-0.64)
- RES-076: Local/global decomposition invalid
- RES-119: CPPN lowest order among generators

**Methodological Lessons**: [What we learned from refutations]

---

## 7. Practical Design Recipe

Combining all validated findings, here's the optimal configuration:

1. **Architecture**:
   - Connection-heavy (100% density preferred)
   - 2+ hidden layers (log-linear improvement)
   - Tanh hidden activations, sin output

2. **Input Encoding**:
   - Sinusoidal positional encoding
   - Wide coordinate ranges [-4, 4]

3. **Weight Initialization**:
   - Connection-biased (not bias-biased)
   - Sign diversity (no uniform signs)

4. **Search Strategy**:
   - Nested sampling with small n_live
   - Many iterations > fewer with large pools

**Expected Performance**: ~0.2-0.3 order at threshold 0.1 for 32×32 images

---

## 8. Open Questions

Inconclusive findings and unexplored domains:

1. [List 5-10 inconclusive RES-IDs]
2. [List underexplored domains]
3. [Methodological improvements needed]

---

## 9. Conclusion

These 207 experiments demonstrate that the Generative-Discriminative trade-off is not accidental but arises from fundamental properties of CPPN priors. The low intrinsic dimensionality that enables efficient search for structured images inherently limits their ability to learn diverse, linearly separable features.

**Key Takeaway**: Architecture, not optimization, determines generative vs discriminative capability.

---

## Appendix: Statistical Summary

- **Total experiments**: 207
- **Validated**: 81 (39.1%)
- **Refuted**: 107 (51.7%)
- **Inconclusive**: 19 (9.2%)
- **Domains**: 173 unique
- **Mean effect size (validated)**: [compute]
- **Largest effect**: [find max]
- **Most consistent finding**: [most validated in category]

```

## Output Format (CONCISE - 2-3 lines max)

**Success**:
```
✓ Generated RESEARCH_INSIGHTS.md
  7 sections, 207 findings synthesized
  Validated: 81, Refuted: 107, Inconclusive: 19
```

**Errors**:
```
✗ research_index.json not found
✗ Missing entries prevent synthesis
```

## Key Notes

- Do NOT include raw data or JSON dumps
- Do NOT explain process—just deliver document
- Return ONLY: confirmation that document was generated
- Focus on narrative and connections, not lists
- Use actual RES-IDs and effect sizes from data
- Keep sections ~400-600 words each
- Make it publication-ready
