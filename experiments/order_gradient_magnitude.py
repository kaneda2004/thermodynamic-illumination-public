"""
RES-145: Order gradient magnitude increases during nested sampling progression.

HYPOTHESIS: As nested sampling progresses to higher order thresholds, the
local gradient of order with respect to weights increases. This would explain
why ESS gets harder (RES-121) - the landscape becomes steeper.

APPROACH:
1. Run nested sampling and record live points at regular intervals
2. For each live point, compute |dOrder/dWeights| via finite differences
3. Track how mean gradient magnitude changes with iteration/threshold
4. Statistical test: correlation between iteration and gradient magnitude
"""

import numpy as np
import sys
sys.path.insert(0, '/Users/matt/Development/monochrome_noise_converger')
from core.thermo_sampler_v3 import CPPN, order_multiplicative
from scipy import stats
import json
import os

def compute_order_gradient(cppn: CPPN, image_size: int = 32, eps: float = 1e-4) -> float:
    """
    Compute ||dOrder/dWeights|| via central finite differences.
    Returns the L2 norm of the gradient vector.
    """
    weights = cppn.get_weights()
    n_weights = len(weights)
    gradient = np.zeros(n_weights)

    for i in range(n_weights):
        # Perturb +eps
        w_plus = weights.copy()
        w_plus[i] += eps
        cppn.set_weights(w_plus)
        order_plus = order_multiplicative(cppn.render(image_size))

        # Perturb -eps
        w_minus = weights.copy()
        w_minus[i] -= eps
        cppn.set_weights(w_minus)
        order_minus = order_multiplicative(cppn.render(image_size))

        # Central difference
        gradient[i] = (order_plus - order_minus) / (2 * eps)

    # Restore original weights
    cppn.set_weights(weights)

    return np.linalg.norm(gradient)


def run_nested_sampling_with_gradient_tracking(
    n_live: int = 50,
    n_iterations: int = 200,
    image_size: int = 32,
    sample_interval: int = 20,
    seed: int = 42
) -> dict:
    """
    Run NS and track gradient magnitude at regular intervals.
    """
    np.random.seed(seed)

    # Initialize live points
    live_points = []
    for _ in range(n_live):
        cppn = CPPN()
        img = cppn.render(image_size)
        order = order_multiplicative(img)
        live_points.append({'cppn': cppn, 'order': order})

    # Track gradient magnitude over iterations
    records = []

    for iteration in range(n_iterations):
        # Find worst
        worst_idx = min(range(n_live), key=lambda i: live_points[i]['order'])
        threshold = live_points[worst_idx]['order']

        # Sample gradient magnitude at intervals
        if iteration % sample_interval == 0:
            # Sample gradients from a subset of live points
            n_sample = min(10, n_live)
            sample_indices = np.random.choice(n_live, n_sample, replace=False)

            gradients = []
            orders = []
            for idx in sample_indices:
                cppn = live_points[idx]['cppn']
                grad_mag = compute_order_gradient(cppn, image_size)
                gradients.append(grad_mag)
                orders.append(live_points[idx]['order'])

            records.append({
                'iteration': iteration,
                'threshold': threshold,
                'mean_gradient': np.mean(gradients),
                'std_gradient': np.std(gradients),
                'mean_order': np.mean(orders),
                'gradients': gradients,
                'orders': orders
            })
            print(f"Iter {iteration}: threshold={threshold:.4f}, mean_grad={np.mean(gradients):.4f}")

        # Replace worst with new sample above threshold
        # Simple rejection sampling (not full ESS for speed)
        success = False
        for _ in range(100):
            # Clone and mutate a random live point
            donor_idx = np.random.randint(n_live)
            new_cppn = live_points[donor_idx]['cppn'].copy()

            # Mutate weights
            old_weights = new_cppn.get_weights()
            new_weights = old_weights + np.random.randn(len(old_weights)) * 0.3
            new_cppn.set_weights(new_weights)

            new_img = new_cppn.render(image_size)
            new_order = order_multiplicative(new_img)

            if new_order >= threshold:
                live_points[worst_idx] = {'cppn': new_cppn, 'order': new_order}
                success = True
                break

        if not success:
            # Terminate early if stuck
            print(f"Stuck at iteration {iteration}")
            break

    return {'records': records, 'n_iterations': iteration + 1}


def main():
    print("RES-145: Order gradient magnitude during nested sampling")
    print("=" * 60)

    # Run multiple seeds
    n_seeds = 8
    all_records = []

    for seed in range(n_seeds):
        print(f"\n--- Seed {seed} ---")
        result = run_nested_sampling_with_gradient_tracking(
            n_live=50,
            n_iterations=200,
            image_size=32,
            sample_interval=20,
            seed=seed
        )
        all_records.extend(result['records'])

    # Aggregate by iteration
    iterations = sorted(set(r['iteration'] for r in all_records))

    iteration_means = []
    iteration_gradients = []

    for it in iterations:
        grads = [r['mean_gradient'] for r in all_records if r['iteration'] == it]
        iteration_means.append(np.mean(grads))
        iteration_gradients.extend([(it, g) for g in grads])

    # Statistical tests
    its = [pair[0] for pair in iteration_gradients]
    grads = [pair[1] for pair in iteration_gradients]

    # Pearson correlation
    corr_r, corr_p = stats.pearsonr(its, grads)

    # Effect size: compare early vs late gradients
    early = [g for it, g in iteration_gradients if it < 50]
    late = [g for it, g in iteration_gradients if it >= 150]

    if len(early) > 0 and len(late) > 0:
        pooled_std = np.sqrt((np.std(early)**2 + np.std(late)**2) / 2)
        cohens_d = (np.mean(late) - np.mean(early)) / (pooled_std + 1e-10)
    else:
        cohens_d = 0

    # Linear regression slope
    slope, intercept, _, _, _ = stats.linregress(its, grads)

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Correlation (iteration vs gradient): r={corr_r:.3f}, p={corr_p:.2e}")
    print(f"Linear slope: {slope:.6f} per iteration")
    print(f"Early gradient (iter<50): {np.mean(early):.4f} +/- {np.std(early):.4f}")
    print(f"Late gradient (iter>=150): {np.mean(late):.4f} +/- {np.std(late):.4f}")
    print(f"Cohen's d (late - early): {cohens_d:.2f}")

    # Additional: correlation between order and gradient magnitude
    all_orders = []
    all_grads = []
    for rec in all_records:
        all_orders.extend(rec['orders'])
        all_grads.extend(rec['gradients'])

    r_order, p_order = stats.pearsonr(all_orders, all_grads)
    print(f"\nCorrelation (order vs gradient): r={r_order:.3f}, p={p_order:.2e}")

    # Verdict
    print("\n" + "=" * 60)
    early_mean = float(np.mean(early))
    late_mean = float(np.mean(late))
    r_val = float(corr_r)
    p_val = float(corr_p)
    d_val = float(cohens_d)

    if p_val < 0.01 and d_val > 0.5:
        verdict = "VALIDATED"
        summary = (f"Order gradient magnitude increases during NS (r={r_val:.2f}, p={p_val:.2e}, cohen_d={d_val:.2f}). "
                   f"Late gradients {late_mean:.3f} vs early {early_mean:.3f}. "
                   f"Steeper landscape explains harder sampling at high thresholds.")
    elif p_val < 0.01 and d_val < -0.5:
        verdict = "REFUTED"
        summary = (f"Order gradient magnitude DECREASES during NS (r={r_val:.2f}, p={p_val:.2e}, cohen_d={d_val:.2f}). "
                   f"Late gradients {late_mean:.3f} vs early {early_mean:.3f}. "
                   f"Landscape becomes flatter at high order, contradicting hypothesis.")
    else:
        verdict = "INCONCLUSIVE"
        summary = (f"Weak/no relationship between iteration and gradient magnitude "
                   f"(r={r_val:.2f}, p={p_val:.2e}, cohen_d={d_val:.2f}). "
                   f"Gradient magnitude may not explain ESS difficulty progression.")

    print(f"VERDICT: {verdict}")
    print(f"SUMMARY: {summary}")

    # Save results
    os.makedirs('results/gradient_magnitude', exist_ok=True)
    results = {
        'hypothesis': 'Order gradient magnitude increases during nested sampling progression',
        'verdict': verdict,
        'metrics': {
            'correlation_r': r_val,
            'p_value': p_val,
            'cohens_d': d_val,
            'early_mean': early_mean,
            'late_mean': late_mean,
            'slope': float(slope),
            'order_gradient_r': float(r_order)
        },
        'summary': summary
    }

    with open('results/gradient_magnitude/res145_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to results/gradient_magnitude/res145_results.json")

    return results


if __name__ == '__main__':
    main()
