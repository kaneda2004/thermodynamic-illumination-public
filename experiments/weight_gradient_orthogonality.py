"""
RES-192: Weight vectors of high-order CPPNs are more orthogonal to local order gradient

Hypothesis: High-order CPPNs sit near local maxima where the weight vector orientation
is orthogonal to the order gradient. This would manifest as lower |cos(w, grad)|
for high-order CPPNs compared to low-order ones.

Rationale:
- RES-129: High-order regions have higher curvature (sharper peaks)
- RES-145: Gradient magnitude increases with order
- If high-order CPPNs are at/near maxima, gradient should be orthogonal to position
"""

import numpy as np
import sys
sys.path.insert(0, '.')

from core.thermo_sampler_v3 import CPPN, order_multiplicative
from scipy import stats

def compute_order_gradient(cppn, epsilon=1e-4):
    """Compute numerical gradient of order with respect to weights."""
    weights = cppn.get_weights()
    n_weights = len(weights)
    gradient = np.zeros(n_weights)

    for i in range(n_weights):
        # Forward difference
        w_plus = weights.copy()
        w_plus[i] += epsilon
        cppn.set_weights(w_plus)
        img_plus = cppn.render(32)
        order_plus = order_multiplicative(img_plus)

        # Backward difference
        w_minus = weights.copy()
        w_minus[i] -= epsilon
        cppn.set_weights(w_minus)
        img_minus = cppn.render(32)
        order_minus = order_multiplicative(img_minus)

        gradient[i] = (order_plus - order_minus) / (2 * epsilon)

    # Restore original weights
    cppn.set_weights(weights)
    return gradient

def cosine_similarity(a, b):
    """Compute cosine similarity between two vectors."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a < 1e-10 or norm_b < 1e-10:
        return 0.0
    return np.dot(a, b) / (norm_a * norm_b)

def main():
    np.random.seed(42)
    n_samples = 500

    print("Generating CPPNs and computing weight-gradient angles...")

    results = []
    for i in range(n_samples):
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{n_samples}")

        # Generate random CPPN
        cppn = CPPN()
        weights = cppn.get_weights()
        img = cppn.render(32)
        order = order_multiplicative(img)

        # Compute gradient
        gradient = compute_order_gradient(cppn)
        grad_norm = np.linalg.norm(gradient)

        # Compute angle measures
        cos_sim = cosine_similarity(weights, gradient)
        abs_cos_sim = abs(cos_sim)  # Absolute value (orthogonality regardless of direction)

        results.append({
            'order': order,
            'cos_sim': cos_sim,
            'abs_cos_sim': abs_cos_sim,
            'grad_norm': grad_norm,
            'weight_norm': np.linalg.norm(weights)
        })

    # Convert to arrays
    orders = np.array([r['order'] for r in results])
    cos_sims = np.array([r['cos_sim'] for r in results])
    abs_cos_sims = np.array([r['abs_cos_sim'] for r in results])
    grad_norms = np.array([r['grad_norm'] for r in results])

    # Quartile analysis
    q25, q75 = np.percentile(orders, [25, 75])
    low_order_mask = orders <= q25
    high_order_mask = orders >= q75

    low_abs_cos = abs_cos_sims[low_order_mask]
    high_abs_cos = abs_cos_sims[high_order_mask]

    # Statistical tests
    # Correlation between order and |cos(w, grad)|
    r_abs, p_abs = stats.pearsonr(orders, abs_cos_sims)
    rho_abs, p_rho_abs = stats.spearmanr(orders, abs_cos_sims)

    # Signed cosine correlation (does gradient point toward or away from origin?)
    r_signed, p_signed = stats.pearsonr(orders, cos_sims)

    # T-test: low vs high order
    t_stat, p_ttest = stats.ttest_ind(high_abs_cos, low_abs_cos)

    # Effect size (Cohen's d)
    pooled_std = np.sqrt((np.std(high_abs_cos)**2 + np.std(low_abs_cos)**2) / 2)
    cohens_d = (np.mean(high_abs_cos) - np.mean(low_abs_cos)) / pooled_std

    print("\n" + "="*60)
    print("RESULTS: Weight-Gradient Orthogonality Analysis")
    print("="*60)

    print(f"\nSample: n={n_samples}")
    print(f"Order range: {orders.min():.4f} to {orders.max():.4f}")
    print(f"Low-order quartile: order <= {q25:.4f} (n={low_order_mask.sum()})")
    print(f"High-order quartile: order >= {q75:.4f} (n={high_order_mask.sum()})")

    print(f"\n--- Absolute Cosine Similarity |cos(w, grad)| ---")
    print(f"Low-order mean:  {np.mean(low_abs_cos):.4f} +/- {np.std(low_abs_cos):.4f}")
    print(f"High-order mean: {np.mean(high_abs_cos):.4f} +/- {np.std(high_abs_cos):.4f}")
    print(f"Cohen's d: {cohens_d:.3f}")
    print(f"T-test: t={t_stat:.3f}, p={p_ttest:.2e}")

    print(f"\n--- Correlations ---")
    print(f"Order vs |cos(w,grad)|: r={r_abs:.4f}, p={p_abs:.2e}")
    print(f"Order vs |cos(w,grad)|: rho={rho_abs:.4f}, p={p_rho_abs:.2e}")
    print(f"Order vs cos(w,grad) (signed): r={r_signed:.4f}, p={p_signed:.2e}")

    print(f"\n--- Gradient Norm Analysis ---")
    r_grad, p_grad = stats.pearsonr(orders, grad_norms)
    print(f"Order vs grad_norm: r={r_grad:.4f}, p={p_grad:.2e}")
    print(f"Low-order grad norm: {np.mean(grad_norms[low_order_mask]):.4f}")
    print(f"High-order grad norm: {np.mean(grad_norms[high_order_mask]):.4f}")

    # Interpretation
    print("\n" + "="*60)
    print("INTERPRETATION")
    print("="*60)

    if cohens_d < -0.5 and p_ttest < 0.01:
        print("VALIDATED: High-order CPPNs have MORE orthogonal weight vectors")
        print(f"  - High-order |cos|={np.mean(high_abs_cos):.3f} < Low-order |cos|={np.mean(low_abs_cos):.3f}")
        print("  - Consistent with high-order CPPNs being near local maxima")
        status = "validated"
    elif cohens_d > 0.5 and p_ttest < 0.01:
        print("REFUTED: High-order CPPNs have LESS orthogonal weight vectors")
        print(f"  - High-order |cos|={np.mean(high_abs_cos):.3f} > Low-order |cos|={np.mean(low_abs_cos):.3f}")
        print("  - Weight vectors are MORE aligned with gradient at high order")
        status = "refuted"
    elif abs(cohens_d) < 0.5:
        print("INCONCLUSIVE: Effect size too small to determine relationship")
        print(f"  - Cohen's d = {cohens_d:.3f} (threshold: |d| > 0.5)")
        status = "inconclusive"
    else:
        print("INCONCLUSIVE: p-value not significant")
        print(f"  - p = {p_ttest:.4f} (threshold: p < 0.01)")
        status = "inconclusive"

    print(f"\nFinal status: {status.upper()}")
    print(f"Effect size: d={cohens_d:.2f}")
    print(f"P-value: {p_ttest:.2e}")

    return {
        'status': status,
        'effect_size': cohens_d,
        'p_value': p_ttest,
        'r_abs': r_abs,
        'rho_abs': rho_abs,
        'low_mean': float(np.mean(low_abs_cos)),
        'high_mean': float(np.mean(high_abs_cos))
    }

if __name__ == "__main__":
    result = main()
