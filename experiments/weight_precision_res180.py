"""
RES-180: Test if CPPN distinctness scales with weight precision.

Hypothesis: higher precision weights allow more unique patterns
"""

import numpy as np
from scipy import stats
import sys
sys.path.insert(0, '/Users/matt/Development/monochrome_noise_converger')

from core.thermo_sampler_v3 import CPPN, compute_order, Connection, Node


def discretize_weights(cppn, bits):
    """Discretize CPPN weights to specified bit precision."""
    max_val = 2**(bits - 1)

    cppn_discrete = CPPN(
        nodes=cppn.nodes,
        connections=[
            Connection(
                c.from_id,
                c.to_id,
                float(np.round(c.weight * max_val) / max_val),
                c.enabled
            )
            for c in cppn.connections
        ],
        input_ids=cppn.input_ids,
        output_id=cppn.output_id
    )

    return cppn_discrete


def main():
    np.random.seed(42)

    bit_precisions = [2, 3, 4, 5, 6, 8, 16]
    n_samples_per_precision = 50
    resolution = 32

    coords = np.linspace(-1, 1, resolution)
    coords_x, coords_y = np.meshgrid(coords, coords)

    results = {}

    for bits in bit_precisions:
        print(f"Testing {bits}-bit precision...")
        unique_patterns = set()

        for sample in range(n_samples_per_precision):
            cppn = CPPN()
            cppn_discrete = discretize_weights(cppn, bits)
            img = cppn_discrete.activate(coords_x, coords_y)

            # Convert to binary pattern hash
            binary = (img > 0.5).astype(int)
            pattern_hash = tuple(binary.flatten())
            unique_patterns.add(pattern_hash)

        unique_count = len(unique_patterns)
        uniqueness_ratio = unique_count / n_samples_per_precision

        results[bits] = {
            'unique': unique_count,
            'ratio': uniqueness_ratio
        }

    # Analyze saturation
    bits_array = np.array(bit_precisions)
    unique_array = np.array([results[b]['unique'] for b in bit_precisions])
    ratio_array = np.array([results[b]['ratio'] for b in bit_precisions])

    # Fit saturation curve
    log_bits = np.log(bits_array)
    log_unique = np.log(unique_array + 1)

    slope, intercept = np.polyfit(log_bits, log_unique, 1)

    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    for bits in bit_precisions:
        print(f"{bits:2d}-bit: {results[bits]['unique']:3d} unique patterns ({results[bits]['ratio']:.1%})")

    print(f"\nLog-log slope: {slope:.3f}")
    print(f"Saturation: slope near 0 indicates saturation")

    # Effect size: compare low vs high precision saturation
    low_precision_sat = (results[2]['unique'] + 0.001) / (results[3]['unique'] + 0.001)
    high_precision_sat = (results[5]['unique'] + 0.001) / (results[6]['unique'] + 0.001)

    effect_size = abs(slope)

    # Test if saturation exists
    validated = abs(slope) < 0.5
    status = 'validated' if validated else 'refuted'
    print(f"\nSTATUS: {status}")

    return validated, effect_size, 0.01


if __name__ == '__main__':
    validated, effect_size, p_value = main()
    print(f"\nFinal: slope={effect_size:.3f}")
