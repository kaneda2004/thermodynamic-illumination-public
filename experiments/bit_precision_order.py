"""
RES-106: Bit Precision Effect on CPPN Order

Hypothesis: Quantizing CPPN weights to lower bit precision (8-bit and 4-bit)
significantly affects achievable order scores compared to full 32-bit precision.

Method:
1. Generate 100 CPPNs with float32 weights
2. For each, quantize weights to 8-bit and 4-bit
3. Compare order scores across precision levels
4. Statistical tests: paired t-test + effect size
"""

import numpy as np
import sys
sys.path.insert(0, '/Users/matt/Development/monochrome_noise_converger')
from core.thermo_sampler_v3 import CPPN, order_multiplicative
from scipy import stats

def quantize_weights(weights, n_bits):
    """Quantize weights to n-bit fixed point in [-4, 4] range."""
    scale = 4.0  # Weights typically in [-4, 4] range
    n_levels = 2 ** n_bits
    # Clip and normalize to [0, 1]
    normalized = (weights + scale) / (2 * scale)
    normalized = np.clip(normalized, 0, 1)
    # Quantize
    quantized_int = np.round(normalized * (n_levels - 1))
    # Dequantize back
    return (quantized_int / (n_levels - 1)) * (2 * scale) - scale

def measure_order_at_precision(cppn, precision_bits, image_size=32):
    """Get order score at specified bit precision."""
    original_weights = cppn.get_weights()

    if precision_bits == 32:
        # No quantization
        quantized = original_weights
    else:
        quantized = quantize_weights(original_weights, precision_bits)

    # Apply quantized weights
    cppn_copy = cppn.copy()
    cppn_copy.set_weights(quantized)

    img = cppn_copy.render(image_size)
    order = order_multiplicative(img)

    return order, quantized

def main():
    np.random.seed(42)
    n_samples = 100

    print("=" * 60)
    print("RES-106: Bit Precision Effect on CPPN Order")
    print("=" * 60)
    print(f"\nGenerating {n_samples} CPPNs and measuring order at different precisions...")

    orders_32bit = []
    orders_8bit = []
    orders_4bit = []

    for i in range(n_samples):
        cppn = CPPN()

        o32, _ = measure_order_at_precision(cppn, 32)
        o8, _ = measure_order_at_precision(cppn, 8)
        o4, _ = measure_order_at_precision(cppn, 4)

        orders_32bit.append(o32)
        orders_8bit.append(o8)
        orders_4bit.append(o4)

        if (i + 1) % 25 == 0:
            print(f"  Processed {i+1}/{n_samples}")

    orders_32bit = np.array(orders_32bit)
    orders_8bit = np.array(orders_8bit)
    orders_4bit = np.array(orders_4bit)

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)

    # Summary statistics
    print("\nOrder Score Statistics:")
    print(f"  32-bit: mean={orders_32bit.mean():.4f}, std={orders_32bit.std():.4f}")
    print(f"  8-bit:  mean={orders_8bit.mean():.4f}, std={orders_8bit.std():.4f}")
    print(f"  4-bit:  mean={orders_4bit.mean():.4f}, std={orders_4bit.std():.4f}")

    # Differences
    diff_8bit = orders_32bit - orders_8bit
    diff_4bit = orders_32bit - orders_4bit

    print(f"\nMean Difference (32bit - Xbit):")
    print(f"  32bit - 8bit:  {diff_8bit.mean():.4f} (std={diff_8bit.std():.4f})")
    print(f"  32bit - 4bit:  {diff_4bit.mean():.4f} (std={diff_4bit.std():.4f})")

    # Paired t-tests
    t_stat_8bit, p_value_8bit = stats.ttest_rel(orders_32bit, orders_8bit)
    t_stat_4bit, p_value_4bit = stats.ttest_rel(orders_32bit, orders_4bit)

    # Effect sizes (Cohen's d for paired samples)
    cohens_d_8bit = diff_8bit.mean() / diff_8bit.std()
    cohens_d_4bit = diff_4bit.mean() / diff_4bit.std()

    print("\n" + "-" * 60)
    print("Statistical Tests (paired t-test):")
    print("-" * 60)
    print(f"\n32-bit vs 8-bit:")
    print(f"  t-statistic: {t_stat_8bit:.4f}")
    print(f"  p-value: {p_value_8bit:.4e}")
    print(f"  Cohen's d: {cohens_d_8bit:.4f}")

    print(f"\n32-bit vs 4-bit:")
    print(f"  t-statistic: {t_stat_4bit:.4f}")
    print(f"  p-value: {p_value_4bit:.4e}")
    print(f"  Cohen's d: {cohens_d_4bit:.4f}")

    # Exact match rates
    exact_8bit = np.sum(orders_32bit == orders_8bit) / n_samples
    exact_4bit = np.sum(orders_32bit == orders_4bit) / n_samples

    print(f"\nExact match rates (same order after quantization):")
    print(f"  8-bit:  {exact_8bit*100:.1f}%")
    print(f"  4-bit:  {exact_4bit*100:.1f}%")

    # Correlation
    corr_8bit = np.corrcoef(orders_32bit, orders_8bit)[0,1]
    corr_4bit = np.corrcoef(orders_32bit, orders_4bit)[0,1]

    print(f"\nCorrelation with 32-bit order:")
    print(f"  8-bit:  r={corr_8bit:.4f}")
    print(f"  4-bit:  r={corr_4bit:.4f}")

    # Summary
    print("\n" + "=" * 60)
    print("CONCLUSION")
    print("=" * 60)

    # Validation criteria: p < 0.01 and |Cohen's d| > 0.5
    significant_8bit = p_value_8bit < 0.01 and abs(cohens_d_8bit) > 0.5
    significant_4bit = p_value_4bit < 0.01 and abs(cohens_d_4bit) > 0.5

    if significant_4bit or significant_8bit:
        status = "VALIDATED"
        print(f"Status: {status}")
        if significant_8bit:
            print(f"  8-bit quantization has significant effect (d={cohens_d_8bit:.3f}, p={p_value_8bit:.2e})")
        if significant_4bit:
            print(f"  4-bit quantization has significant effect (d={cohens_d_4bit:.3f}, p={p_value_4bit:.2e})")
    else:
        status = "REFUTED"
        print(f"Status: {status}")
        print("  Weight quantization does not significantly affect order scores")
        print(f"  8-bit: d={cohens_d_8bit:.3f}, p={p_value_8bit:.2e}")
        print(f"  4-bit: d={cohens_d_4bit:.3f}, p={p_value_4bit:.2e}")

    # Return results for logging
    return {
        'status': status,
        'effect_size_8bit': cohens_d_8bit,
        'effect_size_4bit': cohens_d_4bit,
        'p_value_8bit': p_value_8bit,
        'p_value_4bit': p_value_4bit,
        'correlation_8bit': corr_8bit,
        'correlation_4bit': corr_4bit,
        'exact_match_8bit': exact_8bit,
        'exact_match_4bit': exact_4bit,
    }

if __name__ == "__main__":
    results = main()
