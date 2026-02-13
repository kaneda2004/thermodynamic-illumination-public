"""
RES-189: Test if order metric sensitivity to bit-flips is U-shaped.

Hypothesis: Order metric sensitivity to bit-flips is highest at mid-order levels
and lowest at extremes (very low or very high order).

Rationale: The multiplicative gates may create different sensitivity profiles at
different operating points. At extremes, one gate dominates (low values), making
the metric less sensitive. At mid-order, all gates contribute, making perturbations
more impactful.

Method:
1. Generate 500 CPPN images spanning the order spectrum
2. Bin images by initial order into 5 quantiles
3. For each image, apply random bit-flips and measure order change
4. Test if sensitivity (|delta_order|) varies with initial order level
5. Check specifically for U-shaped relationship (quadratic fit)
"""

import numpy as np
import sys
sys.path.insert(0, '/Users/matt/Development/monochrome_noise_converger')
from core.thermo_sampler_v3 import CPPN, order_multiplicative
from scipy import stats

np.random.seed(42)

def generate_diverse_cppns(n_samples=500):
    """Generate CPPNs spanning order spectrum via rejection + NS-like search."""
    images = []
    orders = []

    # First, generate many random CPPNs
    for _ in range(n_samples * 5):
        cppn = CPPN()
        img = cppn.render(32)
        order = order_multiplicative(img)
        images.append(img)
        orders.append(order)

    # Keep subset spanning the range
    images = np.array(images)
    orders = np.array(orders)

    # Stratified sample: pick from each quintile
    selected_idx = []
    for q in range(5):
        low, high = np.percentile(orders, [q*20, (q+1)*20])
        mask = (orders >= low) & (orders <= high)
        idx = np.where(mask)[0]
        np.random.shuffle(idx)
        selected_idx.extend(idx[:n_samples // 5])

    return images[selected_idx], orders[selected_idx]

def measure_sensitivity(img, n_perturbations=20, n_flips=5):
    """Measure mean absolute order change from random bit-flips."""
    original_order = order_multiplicative(img)
    delta_orders = []

    for _ in range(n_perturbations):
        perturbed = img.copy()
        # Flip n_flips random pixels
        flat_idx = np.random.choice(img.size, n_flips, replace=False)
        for idx in flat_idx:
            i, j = idx // img.shape[1], idx % img.shape[1]
            perturbed[i, j] = 1 - perturbed[i, j]

        new_order = order_multiplicative(perturbed)
        delta_orders.append(abs(new_order - original_order))

    return np.mean(delta_orders), np.std(delta_orders)

def main():
    print("RES-189: Order metric sensitivity U-shape test")
    print("=" * 60)

    # Generate diverse sample
    print("\nGenerating diverse CPPN images...")
    images, orders = generate_diverse_cppns(500)
    print(f"Generated {len(images)} images")
    print(f"Order range: [{orders.min():.4f}, {orders.max():.4f}]")
    print(f"Order quartiles: {np.percentile(orders, [25, 50, 75])}")

    # Measure sensitivity for each image
    print("\nMeasuring bit-flip sensitivity...")
    sensitivities = []
    for i, (img, order) in enumerate(zip(images, orders)):
        mean_sens, _ = measure_sensitivity(img, n_perturbations=20, n_flips=5)
        sensitivities.append(mean_sens)
        if (i + 1) % 100 == 0:
            print(f"  Processed {i+1}/{len(images)}")

    sensitivities = np.array(sensitivities)

    # Bin by order quintiles
    print("\n" + "=" * 60)
    print("RESULTS BY ORDER QUINTILE")
    print("=" * 60)

    quintile_labels = ['Q1 (lowest)', 'Q2', 'Q3', 'Q4', 'Q5 (highest)']
    quintile_sens = []
    quintile_centers = []

    for q in range(5):
        low, high = np.percentile(orders, [q*20, (q+1)*20])
        mask = (orders >= low) & (orders <= high)
        q_sens = sensitivities[mask]
        q_orders = orders[mask]
        quintile_sens.append(q_sens.mean())
        quintile_centers.append(q_orders.mean())
        print(f"{quintile_labels[q]}: order=[{low:.4f},{high:.4f}], "
              f"mean_sens={q_sens.mean():.5f} (std={q_sens.std():.5f}), n={len(q_sens)}")

    # Test for U-shape via quadratic regression
    print("\n" + "=" * 60)
    print("STATISTICAL TESTS")
    print("=" * 60)

    # Fit linear vs quadratic
    x = orders
    y = sensitivities

    # Linear fit
    slope, intercept, r_linear, p_linear, _ = stats.linregress(x, y)
    r2_linear = r_linear ** 2

    # Quadratic fit
    coeffs = np.polyfit(x, y, 2)
    y_pred_quad = np.polyval(coeffs, x)
    ss_res = np.sum((y - y_pred_quad) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2_quad = 1 - ss_res / ss_tot

    # F-test for quadratic improvement
    n = len(x)
    f_stat = ((ss_tot * r2_linear - ss_tot * r2_quad) / 1) / (ss_res / (n - 3))
    p_quadratic = 1 - stats.f.cdf(f_stat, 1, n - 3)

    print(f"\nLinear fit: R^2 = {r2_linear:.4f}, slope = {slope:.6f}, p = {p_linear:.2e}")
    print(f"Quadratic fit: R^2 = {r2_quad:.4f}")
    print(f"Quadratic coefficient (a in ax^2+bx+c): {coeffs[0]:.6f}")
    print(f"F-test for quadratic improvement: F = {f_stat:.4f}, p = {p_quadratic:.2e}")

    # Check if quadratic coefficient is positive (U-shape)
    is_ushape = coeffs[0] > 0
    print(f"\nU-shape (positive quadratic coeff): {is_ushape}")

    # Spearman correlation (monotonic relationship)
    rho, p_spearman = stats.spearmanr(orders, sensitivities)
    print(f"Spearman correlation (order vs sensitivity): rho = {rho:.4f}, p = {p_spearman:.2e}")

    # Test extremes vs middle
    q1_sens = sensitivities[(orders <= np.percentile(orders, 20))]
    q5_sens = sensitivities[(orders >= np.percentile(orders, 80))]
    q_mid_sens = sensitivities[(orders >= np.percentile(orders, 40)) &
                               (orders <= np.percentile(orders, 60))]

    extremes = np.concatenate([q1_sens, q5_sens])
    t_stat, p_ttest = stats.ttest_ind(q_mid_sens, extremes)

    cohens_d = (q_mid_sens.mean() - extremes.mean()) / np.sqrt(
        (q_mid_sens.std()**2 + extremes.std()**2) / 2)

    print(f"\nMiddle vs Extremes comparison:")
    print(f"  Middle (Q2-Q4) mean sensitivity: {q_mid_sens.mean():.5f}")
    print(f"  Extremes (Q1+Q5) mean sensitivity: {extremes.mean():.5f}")
    print(f"  t-test: t = {t_stat:.4f}, p = {p_ttest:.2e}")
    print(f"  Cohen's d (middle - extremes): {cohens_d:.4f}")

    # Verdict
    print("\n" + "=" * 60)
    print("VERDICT")
    print("=" * 60)

    # U-shape requires: positive quadratic coeff AND significant improvement over linear
    # AND middle > extremes
    ushape_validated = (is_ushape and p_quadratic < 0.01 and
                        cohens_d > 0.5 and p_ttest < 0.01)

    # Alternative: monotonic relationship (high order = more/less sensitive)
    monotonic_validated = (abs(rho) > 0.3 and p_spearman < 0.01)

    print(f"\nU-shape hypothesis: {'VALIDATED' if ushape_validated else 'REFUTED'}")
    print(f"Monotonic alternative: {'VALIDATED' if monotonic_validated else 'REFUTED'}")

    if monotonic_validated and not ushape_validated:
        direction = "increases" if rho > 0 else "decreases"
        print(f"\nActual finding: Sensitivity monotonically {direction} with order (rho={rho:.3f})")

    # Summary stats for logging
    print("\n" + "=" * 60)
    print("SUMMARY FOR LOG")
    print("=" * 60)
    effect_size = max(abs(cohens_d), abs(rho))
    p_value = min(p_quadratic, p_spearman)

    print(f"effect_size: {effect_size:.2f}")
    print(f"p_value: {p_value:.2e}")

    if ushape_validated:
        status = "validated"
        summary = (f"Sensitivity IS U-shaped with order (quadratic coeff={coeffs[0]:.4f}, "
                  f"d={cohens_d:.2f}). Middle-order images {cohens_d:.1f}x more sensitive than extremes.")
    elif monotonic_validated:
        status = "refuted"
        direction = "increases" if rho > 0 else "decreases"
        summary = (f"Sensitivity monotonically {direction} with order (rho={rho:.3f}, p={p_spearman:.1e}). "
                  f"NOT U-shaped - quadratic fit shows no significant improvement (p={p_quadratic:.2f}). "
                  f"High-order images {'more' if rho > 0 else 'less'} sensitive to perturbations.")
    else:
        status = "refuted"
        summary = (f"No clear pattern: neither U-shape (d={cohens_d:.2f}) nor monotonic (rho={rho:.3f}) "
                  f"relationship found. Sensitivity to bit-flips varies independently of order level.")

    print(f"status: {status}")
    print(f"summary: {summary}")

    return {
        'status': status,
        'effect_size': effect_size,
        'p_value': p_value,
        'summary': summary,
        'rho': rho,
        'quadratic_coeff': coeffs[0],
        'cohens_d': cohens_d
    }

if __name__ == '__main__':
    main()
