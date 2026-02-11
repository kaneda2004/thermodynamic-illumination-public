"""
RES-134: Test if local curvature directly predicts ESS difficulty.

Hypothesis: Sharper peaks (higher Hessian eigenvalues from RES-129)
require more ESS contractions to propose valid moves.

Building on:
- RES-129: High-order regions have 1.86d higher curvature (r=0.80, p<1e-22)
- RES-121: ESS contractions increase 9x from low (8.4) to high (78.0) thresholds

This experiment measures BOTH curvature and ESS contractions at the SAME
sample points to test direct causation.
"""

import numpy as np
from scipy import stats
import sys
sys.path.insert(0, '/Users/matt/Development/monochrome_noise_converger')
from core.thermo_sampler_v3 import (
    CPPN, order_multiplicative, elliptical_slice_sample, PRIOR_SIGMA
)

def compute_local_curvature(cppn: CPPN, order_fn, image_size: int = 32, eps: float = 0.01) -> float:
    """
    Compute local curvature via numerical Hessian diagonal.
    Returns mean absolute curvature (larger = sharper peak).
    """
    weights = cppn.get_weights()
    n_params = len(weights)

    # Current order value
    current_img = cppn.render(image_size)
    current_order = order_fn(current_img)

    # Compute diagonal Hessian elements using finite differences
    # f''(x) ≈ (f(x+h) - 2f(x) + f(x-h)) / h^2
    curvatures = []
    for i in range(n_params):
        w_plus = weights.copy()
        w_minus = weights.copy()
        w_plus[i] += eps
        w_minus[i] -= eps

        cppn.set_weights(w_plus)
        order_plus = order_fn(cppn.render(image_size))

        cppn.set_weights(w_minus)
        order_minus = order_fn(cppn.render(image_size))

        # Second derivative
        curvature = (order_plus - 2*current_order + order_minus) / (eps**2)
        curvatures.append(abs(curvature))

    # Restore original weights
    cppn.set_weights(weights)

    return np.mean(curvatures)


def measure_ess_contractions(cppn: CPPN, threshold: float, order_fn, image_size: int = 32) -> int:
    """
    Measure how many ESS contractions needed to find valid sample.
    Returns contraction count.
    """
    _, _, _, _, n_contractions, _ = elliptical_slice_sample(
        cppn, threshold, image_size, order_fn, max_contractions=200
    )
    return n_contractions


def run_experiment(n_samples: int = 500, seed: int = 42):
    """
    Sample CPPNs across order spectrum, measure curvature + ESS contractions.
    """
    np.random.seed(seed)

    print("RES-134: Testing curvature -> ESS difficulty correlation")
    print("=" * 60)

    # Generate diverse CPPNs and measure at their natural order levels
    results = []

    for i in range(n_samples):
        if (i + 1) % 20 == 0:
            print(f"  Processed {i+1}/{n_samples} samples...")

        # Create random CPPN
        cppn = CPPN()
        img = cppn.render(32)
        order = order_multiplicative(img)

        # Skip very low order samples (ESS trivially succeeds)
        if order < 0.05:
            continue

        # Measure local curvature at this point
        curvature = compute_local_curvature(cppn, order_multiplicative)

        # Use current order as threshold (sample just above current level)
        # This tests how hard it is to move within the order landscape
        threshold = order - 0.05  # Slightly below to allow some movement
        if threshold < 0.1:
            threshold = 0.1

        # Measure ESS difficulty
        contractions = measure_ess_contractions(cppn.copy(), threshold, order_multiplicative)

        results.append({
            'order': order,
            'curvature': curvature,
            'contractions': contractions
        })

    # Convert to arrays
    orders = np.array([r['order'] for r in results])
    curvatures = np.array([r['curvature'] for r in results])
    contractions = np.array([r['contractions'] for r in results])

    print(f"\nCollected {len(results)} samples")
    print(f"Order range: [{orders.min():.3f}, {orders.max():.3f}]")
    print(f"Curvature range: [{curvatures.min():.2f}, {curvatures.max():.2f}]")
    print(f"Contractions range: [{contractions.min()}, {contractions.max()}]")

    # PRIMARY TEST: Curvature -> Contractions correlation
    r_curv_contract, p_curv_contract = stats.pearsonr(curvatures, contractions)

    # CONTROL: Order -> Curvature (should replicate RES-129)
    r_order_curv, p_order_curv = stats.pearsonr(orders, curvatures)

    # CONTROL: Order -> Contractions (should replicate RES-121)
    r_order_contract, p_order_contract = stats.pearsonr(orders, contractions)

    # Partial correlation: Curvature -> Contractions controlling for Order
    # If curvature is the mechanism, this should remain significant
    # Using residual approach
    slope, intercept, _, _, _ = stats.linregress(orders, curvatures)
    curv_resid = curvatures - (slope * orders + intercept)

    slope, intercept, _, _, _ = stats.linregress(orders, contractions)
    contract_resid = contractions - (slope * orders + intercept)

    r_partial, p_partial = stats.pearsonr(curv_resid, contract_resid)

    # Effect sizes (Cohen's d)
    # Split by median curvature
    median_curv = np.median(curvatures)
    high_curv_contracts = contractions[curvatures >= median_curv]
    low_curv_contracts = contractions[curvatures < median_curv]

    pooled_std = np.sqrt(
        ((len(high_curv_contracts)-1)*np.std(high_curv_contracts, ddof=1)**2 +
         (len(low_curv_contracts)-1)*np.std(low_curv_contracts, ddof=1)**2) /
        (len(high_curv_contracts) + len(low_curv_contracts) - 2)
    )
    effect_size = (np.mean(high_curv_contracts) - np.mean(low_curv_contracts)) / pooled_std

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)

    print(f"\nPRIMARY: Curvature -> Contractions")
    print(f"  r = {r_curv_contract:.3f}, p = {p_curv_contract:.2e}")
    print(f"  High curvature contractions: {np.mean(high_curv_contracts):.1f} ± {np.std(high_curv_contracts):.1f}")
    print(f"  Low curvature contractions: {np.mean(low_curv_contracts):.1f} ± {np.std(low_curv_contracts):.1f}")
    print(f"  Cohen's d = {effect_size:.2f}")

    print(f"\nCONTROL: Order -> Curvature (should replicate RES-129)")
    print(f"  r = {r_order_curv:.3f}, p = {p_order_curv:.2e}")

    print(f"\nCONTROL: Order -> Contractions (should replicate RES-121)")
    print(f"  r = {r_order_contract:.3f}, p = {p_order_contract:.2e}")

    print(f"\nPARTIAL CORRELATION: Curvature -> Contractions | Order")
    print(f"  r = {r_partial:.3f}, p = {p_partial:.2e}")
    print("  (Tests if curvature explains variance beyond order)")

    # Validation criteria
    validated = (
        p_curv_contract < 0.01 and
        abs(effect_size) > 0.5 and
        r_curv_contract > 0.3  # Meaningful positive correlation
    )

    print("\n" + "=" * 60)
    if validated:
        print("STATUS: VALIDATED")
        print("Curvature directly predicts ESS difficulty with large effect size.")
    else:
        print("STATUS: REFUTED or INCONCLUSIVE")
        if p_curv_contract >= 0.01:
            print(f"  - p-value too high: {p_curv_contract:.3f} >= 0.01")
        if abs(effect_size) <= 0.5:
            print(f"  - Effect size too small: |{effect_size:.2f}| <= 0.5")
        if r_curv_contract <= 0.3:
            print(f"  - Correlation too weak: {r_curv_contract:.3f} <= 0.3")

    return {
        'status': 'validated' if validated else ('refuted' if p_curv_contract > 0.1 else 'inconclusive'),
        'r_curv_contract': float(r_curv_contract),
        'p_curv_contract': float(p_curv_contract),
        'effect_size': float(effect_size),
        'r_partial': float(r_partial),
        'p_partial': float(p_partial),
        'n_samples': len(results),
        'r_order_curv': float(r_order_curv),
        'r_order_contract': float(r_order_contract)
    }


if __name__ == "__main__":
    results = run_experiment()
    print(f"\nFinal metrics: {results}")
