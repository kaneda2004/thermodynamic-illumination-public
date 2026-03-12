#!/usr/bin/env python3
"""
RES-132: Test if CPPNs can more accurately reconstruct high-order structured
images than low-order random images.

Building on RES-123 which showed CPPNs can only match 54% of random image pixels,
we hypothesize that reconstruction accuracy depends on target structure - CPPNs
should better match structured (high-order) targets.

Method:
1. Generate target images across a range of order values
2. For each target, optimize CPPN weights to minimize MSE reconstruction loss
3. Measure reconstruction accuracy vs target order
4. Test correlation and effect size
"""

import numpy as np
from scipy import stats
from scipy.optimize import minimize
import sys
sys.path.insert(0, '/Users/matt/Development/monochrome_noise_converger')
from core.thermo_sampler_v3 import CPPN, order_multiplicative


def generate_target_with_order(target_order: str, size: int = 32) -> np.ndarray:
    """Generate target images with controlled order levels."""
    if target_order == 'random':
        # Pure random binary image - should have low order
        return (np.random.rand(size, size) > 0.5).astype(np.uint8)

    elif target_order == 'noise_smooth':
        # Smoothed random - moderate order
        from scipy.ndimage import gaussian_filter
        noise = np.random.rand(size, size)
        smoothed = gaussian_filter(noise, sigma=2.0)
        return (smoothed > 0.5).astype(np.uint8)

    elif target_order == 'cppn_low':
        # CPPN-generated - sample some and pick low order one
        worst_img = None
        worst_order = np.inf
        for _ in range(30):
            cppn = CPPN()  # Default constructor
            img = cppn.render(size)
            order = order_multiplicative(img)
            if order < worst_order:
                worst_order = order
                worst_img = img
        return worst_img

    elif target_order == 'cppn_high':
        # CPPN-generated high order (search for high order)
        best_img = None
        best_order = -np.inf
        for _ in range(50):
            cppn = CPPN()
            img = cppn.render(size)
            order = order_multiplicative(img)
            if order > best_order:
                best_order = order
                best_img = img
        return best_img

    elif target_order == 'stripes':
        # Horizontal stripes - very high order
        img = np.zeros((size, size), dtype=np.uint8)
        for i in range(size):
            if (i // 4) % 2 == 0:
                img[i, :] = 1
        return img

    elif target_order == 'checkerboard':
        # Checkerboard pattern - high order
        img = np.zeros((size, size), dtype=np.uint8)
        for i in range(size):
            for j in range(size):
                if (i + j) % 2 == 0:
                    img[i, j] = 1
        return img


def optimize_cppn_to_target(target: np.ndarray, max_evals: int = 500) -> tuple:
    """Optimize CPPN weights to reconstruct target image."""
    size = target.shape[0]
    cppn = CPPN()

    def loss_fn(weights):
        cppn.set_weights(weights)
        coords = np.linspace(-1, 1, size)
        x, y = np.meshgrid(coords, coords)
        rendered = cppn.activate(x, y)
        binary = (rendered > 0.5).astype(np.uint8)
        mse = np.mean((binary.astype(float) - target.astype(float)) ** 2)
        return mse

    # Use multiple random restarts
    best_loss = np.inf
    best_accuracy = 0

    for restart in range(5):
        # Create fresh CPPN to get new random weights
        cppn = CPPN()
        initial_weights = cppn.get_weights()

        # Optimize using L-BFGS-B
        result = minimize(
            loss_fn,
            initial_weights,
            method='L-BFGS-B',
            options={'maxiter': max_evals // 5}
        )

        # Compute accuracy
        cppn.set_weights(result.x)
        coords = np.linspace(-1, 1, size)
        x, y = np.meshgrid(coords, coords)
        rendered = cppn.activate(x, y)
        binary = (rendered > 0.5).astype(np.uint8)
        accuracy = np.mean(binary == target)

        if result.fun < best_loss:
            best_loss = result.fun
            best_accuracy = accuracy

    return best_accuracy, 1.0 - best_loss  # accuracy and 1-MSE


def main():
    np.random.seed(42)

    print("RES-132: Testing CPPN reconstruction accuracy vs target structure")
    print("=" * 70)

    # Generate diverse targets at different order levels
    target_types = ['random', 'noise_smooth', 'cppn_low', 'cppn_high', 'stripes', 'checkerboard']
    n_samples_per_type = 12

    results = []

    for target_type in target_types:
        print(f"\nProcessing {target_type} targets...")
        for i in range(n_samples_per_type):
            # Generate target
            target = generate_target_with_order(target_type)
            target_order = order_multiplicative(target)

            # Optimize CPPN to match target
            accuracy, one_minus_mse = optimize_cppn_to_target(target, max_evals=200)

            results.append({
                'type': target_type,
                'target_order': target_order,
                'accuracy': accuracy,
                'one_minus_mse': one_minus_mse
            })

            print(f"  Sample {i+1}: order={target_order:.3f}, accuracy={accuracy:.3f}")

    # Convert to arrays for analysis
    orders = np.array([r['target_order'] for r in results])
    accuracies = np.array([r['accuracy'] for r in results])

    print("\n" + "=" * 70)
    print("ANALYSIS")
    print("=" * 70)

    # Summary by type
    print("\nMean accuracy by target type:")
    for target_type in target_types:
        type_results = [r for r in results if r['type'] == target_type]
        type_orders = [r['target_order'] for r in type_results]
        type_accs = [r['accuracy'] for r in type_results]
        print(f"  {target_type:15s}: order={np.mean(type_orders):.3f} +/- {np.std(type_orders):.3f}, "
              f"accuracy={np.mean(type_accs):.3f} +/- {np.std(type_accs):.3f}")

    # Correlation between target order and reconstruction accuracy
    correlation, p_value = stats.pearsonr(orders, accuracies)
    spearman_corr, spearman_p = stats.spearmanr(orders, accuracies)

    print(f"\nPearson correlation (order vs accuracy): r={correlation:.3f}, p={p_value:.2e}")
    print(f"Spearman correlation (order vs accuracy): rho={spearman_corr:.3f}, p={spearman_p:.2e}")

    # Effect size: compare high-order vs low-order targets
    median_order = np.median(orders)
    low_order_acc = accuracies[orders < median_order]
    high_order_acc = accuracies[orders >= median_order]

    cohens_d = (np.mean(high_order_acc) - np.mean(low_order_acc)) / np.sqrt(
        (np.std(low_order_acc)**2 + np.std(high_order_acc)**2) / 2
    )

    t_stat, t_pval = stats.ttest_ind(high_order_acc, low_order_acc)

    print(f"\nHigh-order targets (n={len(high_order_acc)}): accuracy={np.mean(high_order_acc):.3f} +/- {np.std(high_order_acc):.3f}")
    print(f"Low-order targets (n={len(low_order_acc)}): accuracy={np.mean(low_order_acc):.3f} +/- {np.std(low_order_acc):.3f}")
    print(f"Cohen's d: {cohens_d:.3f}")
    print(f"t-test: t={t_stat:.3f}, p={t_pval:.2e}")

    # Validation criteria
    print("\n" + "=" * 70)
    print("VALIDATION")
    print("=" * 70)

    validated = (
        p_value < 0.01 and
        abs(cohens_d) > 0.5 and
        correlation > 0  # Positive correlation expected
    )

    print(f"p < 0.01: {p_value < 0.01} (p={p_value:.2e})")
    print(f"|Cohen's d| > 0.5: {abs(cohens_d) > 0.5} (d={cohens_d:.3f})")
    print(f"Positive correlation: {correlation > 0} (r={correlation:.3f})")
    print(f"\nSTATUS: {'VALIDATED' if validated else 'REFUTED'}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    if validated:
        print(f"CPPNs reconstruct high-order targets {np.mean(high_order_acc):.1%} vs "
              f"low-order targets {np.mean(low_order_acc):.1%} accuracy.")
        print(f"Strong positive correlation r={correlation:.2f} confirms structure-dependent expressivity.")
    else:
        print(f"Reconstruction accuracy ({np.mean(accuracies):.1%} overall) does not significantly "
              f"depend on target order (r={correlation:.2f}, d={cohens_d:.2f}).")

    return {
        'status': 'validated' if validated else 'refuted',
        'correlation': correlation,
        'p_value': p_value,
        'cohens_d': cohens_d,
        'high_order_acc': np.mean(high_order_acc),
        'low_order_acc': np.mean(low_order_acc)
    }


if __name__ == '__main__':
    main()
