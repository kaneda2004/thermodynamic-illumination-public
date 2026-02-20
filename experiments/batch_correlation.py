"""
RES-112: Same-architecture CPPNs with different random weight initializations
produce order distributions that are uncorrelated - weights dominate architecture effects.

HYPOTHESIS: If architecture determines order distribution, then CPPNs with identical
architecture but different weight initializations should produce correlated order
distributions. If weights dominate, the correlation should be near zero.

METHODOLOGY:
1. Define a fixed architecture (same hidden sizes, activations, connections)
2. Generate N_batches batches of CPPNs with that architecture but different seeds
3. For each batch, sample M images and compute order distribution
4. Compute pairwise correlations between batches' order distributions
5. Compare to null hypothesis (correlation = 0)
"""

import numpy as np
from scipy import stats
import sys
sys.path.insert(0, '/Users/matt/Development/monochrome_noise_converger')
from core.thermo_sampler_v3 import CPPN, Node, Connection, order_multiplicative

def create_cppn_with_architecture(seed: int):
    """
    Create a CPPN with a fixed architecture (2 hidden layers) but random weights.
    Architecture: [4 inputs] -> [8 hidden] -> [8 hidden] -> [1 output]
    Activations: inputs=identity, hidden1=tanh, hidden2=sin, output=sigmoid
    """
    np.random.seed(seed)

    # Fixed architecture:
    # Input IDs: 0, 1, 2, 3 (x, y, r, bias)
    # Hidden1 IDs: 5, 6, 7, 8, 9, 10, 11, 12 (8 nodes)
    # Hidden2 IDs: 13, 14, 15, 16, 17, 18, 19, 20 (8 nodes)
    # Output ID: 4

    nodes = [
        # Input nodes (bias=0, identity activation)
        Node(0, 'identity', 0.0),
        Node(1, 'identity', 0.0),
        Node(2, 'identity', 0.0),
        Node(3, 'identity', 0.0),
        # Output node (random bias)
        Node(4, 'sigmoid', np.random.randn()),
        # Hidden layer 1 (8 nodes, tanh activation)
        *[Node(5+i, 'tanh', np.random.randn()) for i in range(8)],
        # Hidden layer 2 (8 nodes, sin activation)
        *[Node(13+i, 'sin', np.random.randn()) for i in range(8)],
    ]

    connections = []
    # Input -> Hidden1 (4 * 8 = 32 connections)
    for inp in range(4):
        for h1 in range(5, 13):
            connections.append(Connection(inp, h1, np.random.randn()))

    # Hidden1 -> Hidden2 (8 * 8 = 64 connections)
    for h1 in range(5, 13):
        for h2 in range(13, 21):
            connections.append(Connection(h1, h2, np.random.randn()))

    # Hidden2 -> Output (8 connections)
    for h2 in range(13, 21):
        connections.append(Connection(h2, 4, np.random.randn()))

    cppn = CPPN(
        nodes=nodes,
        connections=connections,
        input_ids=[0, 1, 2, 3],
        output_id=4
    )
    return cppn


def generate_order_distribution(cppn, n_samples=50):
    """Generate n_samples images with different random perturbations and return order scores."""
    base_weights = cppn.get_weights()
    orders = []

    for i in range(n_samples):
        # Small perturbation to weights (keeps same architecture)
        perturbation = np.random.randn(len(base_weights)) * 0.1
        cppn.set_weights(base_weights + perturbation)
        img = cppn.render(size=32)
        order = order_multiplicative(img)
        orders.append(order)

    # Restore original weights
    cppn.set_weights(base_weights)
    return np.array(orders)


def run_experiment():
    n_batches = 20  # Number of independent weight initializations
    n_samples_per_batch = 50  # Samples per initialization

    print("RES-112: Batch Correlation Study")
    print("=" * 50)
    print(f"Architecture: 4 inputs -> 8 hidden (tanh) -> 8 hidden (sin) -> 1 output")
    print(f"N batches: {n_batches}, N samples per batch: {n_samples_per_batch}")
    print()

    # Generate order distributions for each batch
    batch_orders = []
    batch_means = []
    batch_stds = []

    for i in range(n_batches):
        seed = 1000 + i * 100  # Different seed for each batch
        cppn = create_cppn_with_architecture(seed)

        # Re-seed for sampling
        np.random.seed(seed + 1)
        orders = generate_order_distribution(cppn, n_samples_per_batch)
        batch_orders.append(orders)
        batch_means.append(np.mean(orders))
        batch_stds.append(np.std(orders))

        print(f"Batch {i+1}: mean={np.mean(orders):.3f}, std={np.std(orders):.3f}, range=[{np.min(orders):.3f}, {np.max(orders):.3f}]")

    batch_means = np.array(batch_means)
    batch_stds = np.array(batch_stds)

    print()
    print("ANALYSIS")
    print("=" * 50)

    # 1. Compare variance in batch means vs within-batch variance
    between_batch_variance = np.var(batch_means)
    within_batch_variance = np.mean([np.var(orders) for orders in batch_orders])

    print(f"Between-batch variance (in means): {between_batch_variance:.6f}")
    print(f"Within-batch variance (avg): {within_batch_variance:.6f}")
    print(f"Variance ratio (between/within): {between_batch_variance/within_batch_variance:.3f}")

    # 2. Pairwise correlations of order distributions (Q-Q style)
    correlations = []
    for i in range(n_batches):
        for j in range(i+1, n_batches):
            sorted_i = np.sort(batch_orders[i])
            sorted_j = np.sort(batch_orders[j])
            r, p = stats.pearsonr(sorted_i, sorted_j)
            correlations.append(r)

    correlations = np.array(correlations)
    mean_corr = np.mean(correlations)
    std_corr = np.std(correlations)

    print(f"\nPairwise distribution correlations (Q-Q): {mean_corr:.3f} +/- {std_corr:.3f}")

    # 3. Test if batch means are significantly different from each other
    f_stat, p_anova = stats.f_oneway(*batch_orders)
    print(f"ANOVA F-statistic: {f_stat:.2f}, p-value: {p_anova:.2e}")

    # 4. Kruskal-Wallis test (non-parametric alternative)
    h_stat, p_kruskal = stats.kruskal(*batch_orders)
    print(f"Kruskal-Wallis H-statistic: {h_stat:.2f}, p-value: {p_kruskal:.2e}")

    # 5. Intraclass Correlation Coefficient (ICC)
    all_orders = np.array(batch_orders)
    grand_mean = np.mean(all_orders)

    # Between groups sum of squares
    ss_between = n_samples_per_batch * np.sum((batch_means - grand_mean)**2)
    df_between = n_batches - 1
    ms_between = ss_between / df_between

    # Within groups sum of squares
    ss_within = np.sum([np.sum((orders - np.mean(orders))**2) for orders in batch_orders])
    df_within = n_batches * (n_samples_per_batch - 1)
    ms_within = ss_within / df_within

    # ICC(1) - how much variance is due to group membership
    icc = (ms_between - ms_within) / (ms_between + (n_samples_per_batch - 1) * ms_within)

    print(f"\nIntraclass Correlation (ICC): {icc:.3f}")
    print("ICC interpretation: 0=no batch effect, 1=batch fully determines order")

    # 6. Effect size: eta-squared for ANOVA
    ss_total = np.sum((all_orders - grand_mean)**2)
    eta_squared = ss_between / ss_total

    print(f"Eta-squared (effect size): {eta_squared:.3f}")

    # 7. Coefficient of variation of batch means
    cv_means = np.std(batch_means) / np.mean(batch_means)
    print(f"Coefficient of variation of batch means: {cv_means:.3f}")

    print()
    print("CONCLUSION")
    print("=" * 50)

    # Hypothesis: Weights dominate architecture effects
    # VALIDATED if: significant batch differences (p<0.01) AND large effect size (>0.5)
    # This would mean different weight initializations produce substantially different order distributions

    # REFUTED if: no significant differences OR small effect size
    # This would mean architecture constrains behavior regardless of weights

    if p_anova < 0.01 and eta_squared > 0.5:
        print("RESULT: VALIDATED - Weights dominate architecture effects")
        print(f"  - Highly significant batch differences (p={p_anova:.2e})")
        print(f"  - Large effect size (eta-squared={eta_squared:.3f} > 0.5)")
        print(f"  - Different weight initializations produce distinct order distributions")
        status = "validated"
    elif p_anova < 0.01 and eta_squared > 0.14:  # Medium effect size
        print("RESULT: PARTIAL - Weights have moderate influence")
        print(f"  - Significant batch differences (p={p_anova:.2e})")
        print(f"  - Medium effect size (eta-squared={eta_squared:.3f})")
        status = "inconclusive"
    else:
        print("RESULT: REFUTED - Architecture constrains order distribution")
        print(f"  - Batch differences: p={p_anova:.2e}")
        print(f"  - Effect size: eta-squared={eta_squared:.3f}")
        print(f"  - Distribution correlation high: {mean_corr:.3f}")
        status = "refuted"

    print()
    print("METRICS:")
    print(f"  - effect_size: {eta_squared:.3f}")
    print(f"  - p_value: {p_anova:.2e}")
    print(f"  - icc: {icc:.3f}")
    print(f"  - mean_distribution_correlation: {mean_corr:.3f}")
    print(f"  - between_batch_variance: {between_batch_variance:.6f}")
    print(f"  - within_batch_variance: {within_batch_variance:.6f}")

    return {
        'status': status,
        'effect_size': float(eta_squared),
        'p_value': float(p_anova),
        'icc': float(icc),
        'mean_corr': float(mean_corr),
        'cv_means': float(cv_means),
        'between_var': float(between_batch_variance),
        'within_var': float(within_batch_variance),
    }

if __name__ == "__main__":
    results = run_experiment()
