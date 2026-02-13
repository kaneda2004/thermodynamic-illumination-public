"""
Ensemble Averaging Experiment (RES-023)

Hypothesis: Averaging multiple CPPN outputs increases order by smoothing out
individual variations while preserving shared spatial structure.

Null hypothesis: Order of averaged ensemble is equal to mean order of individuals.

Novelty: Prior work (RES-003, RES-007) analyzed individual CPPN images. This tests
whether ensemble methods can systematically increase order through smoothing.
RES-017 showed high-order images have steeper spectral decay (more low-frequency).
Averaging should concentrate power in low frequencies where signals align.

Method:
1. Generate N_ensembles ensembles of K CPPNs each
2. For each ensemble: compute mean output (pre-threshold) then threshold to binary
3. Compare order of ensemble average vs mean order of individuals
4. Statistical tests: paired t-test, Wilcoxon signed-rank, Cohen's d

Key insight: Averaging K independent CPPN continuous outputs should:
- Preserve low-frequency components (where CPPNs share coordinate structure)
- Average out high-frequency differences (noise-like variations)
- Result in smoother images with higher spectral coherence
"""

import numpy as np
import sys
from pathlib import Path
from scipy import stats
import json

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from core.thermo_sampler_v3 import CPPN, order_multiplicative, set_global_seed
from core.thermo_sampler_v3 import compute_compressibility, compute_edge_density
from core.thermo_sampler_v3 import compute_spectral_coherence, compute_symmetry


def generate_cppn_continuous(cppn: CPPN, size: int = 32) -> np.ndarray:
    """Get continuous output from CPPN (pre-threshold)."""
    coords = np.linspace(-1, 1, size)
    x, y = np.meshgrid(coords, coords)
    return cppn.activate(x, y)


def ensemble_average(cpnn_list: list, size: int = 32, threshold: float = 0.5) -> np.ndarray:
    """
    Average continuous outputs from multiple CPPNs, then threshold to binary.

    This should preserve shared low-frequency structure while averaging out
    high-frequency individual variations.
    """
    continuous_outputs = []
    for cppn in cpnn_list:
        continuous_outputs.append(generate_cppn_continuous(cppn, size))

    # Average the continuous outputs
    mean_output = np.mean(continuous_outputs, axis=0)

    # Threshold to binary
    return (mean_output > threshold).astype(np.uint8)


def run_experiment(
    n_ensembles: int = 200,
    ensemble_sizes: list = [2, 3, 5, 8, 12],
    image_size: int = 32,
    seed: int = 42
):
    """
    Run the ensemble averaging experiment.

    Tests whether averaging K CPPNs produces higher order than individuals.
    """
    set_global_seed(seed)

    print(f"=== Ensemble Averaging Experiment ===")
    print(f"N ensembles per size: {n_ensembles}")
    print(f"Ensemble sizes tested: {ensemble_sizes}")
    print(f"Image size: {image_size}")
    print()

    results_by_size = {}

    for K in ensemble_sizes:
        print(f"\n--- Testing ensemble size K={K} ---")

        individual_orders = []  # Mean order of K individuals
        ensemble_orders = []    # Order of ensemble average

        # Track features for analysis
        individual_spectral = []
        ensemble_spectral = []
        individual_edge = []
        ensemble_edge = []

        for i in range(n_ensembles):
            # Generate K CPPNs
            cpnn_list = [CPPN() for _ in range(K)]

            # Compute individual orders
            ind_orders = []
            ind_spectral = []
            ind_edge = []
            for cppn in cpnn_list:
                img = cppn.render(image_size)
                ind_orders.append(order_multiplicative(img))
                ind_spectral.append(compute_spectral_coherence(img))
                ind_edge.append(compute_edge_density(img))

            mean_ind_order = np.mean(ind_orders)
            individual_orders.append(mean_ind_order)
            individual_spectral.append(np.mean(ind_spectral))
            individual_edge.append(np.mean(ind_edge))

            # Compute ensemble average
            ensemble_img = ensemble_average(cpnn_list, image_size)
            ens_order = order_multiplicative(ensemble_img)
            ensemble_orders.append(ens_order)
            ensemble_spectral.append(compute_spectral_coherence(ensemble_img))
            ensemble_edge.append(compute_edge_density(ensemble_img))

            if (i + 1) % 50 == 0:
                print(f"  Processed {i + 1}/{n_ensembles}")

        individual_orders = np.array(individual_orders)
        ensemble_orders = np.array(ensemble_orders)
        individual_spectral = np.array(individual_spectral)
        ensemble_spectral = np.array(ensemble_spectral)
        individual_edge = np.array(individual_edge)
        ensemble_edge = np.array(ensemble_edge)

        # Paired difference analysis
        order_diff = ensemble_orders - individual_orders

        # Statistical tests
        # 1. Paired t-test
        t_stat, p_ttest = stats.ttest_rel(ensemble_orders, individual_orders)

        # 2. Wilcoxon signed-rank (non-parametric)
        w_stat, p_wilcox = stats.wilcoxon(ensemble_orders, individual_orders, alternative='two-sided')

        # 3. Cohen's d for paired samples
        d = np.mean(order_diff) / np.std(order_diff) if np.std(order_diff) > 0 else 0

        # 4. Fraction where ensemble beats individuals
        frac_ensemble_better = np.mean(ensemble_orders > individual_orders)

        # Feature analysis
        spectral_diff = ensemble_spectral - individual_spectral
        edge_diff = ensemble_edge - individual_edge

        print(f"\n  Individual mean order: {individual_orders.mean():.4f} +/- {individual_orders.std():.4f}")
        print(f"  Ensemble mean order:   {ensemble_orders.mean():.4f} +/- {ensemble_orders.std():.4f}")
        print(f"  Mean difference:       {order_diff.mean():.4f} +/- {order_diff.std():.4f}")
        print(f"  Paired t-test:         t={t_stat:.3f}, p={p_ttest:.2e}")
        print(f"  Wilcoxon:              W={w_stat:.1f}, p={p_wilcox:.2e}")
        print(f"  Cohen's d:             {d:.3f}")
        print(f"  Fraction ensemble > individual: {frac_ensemble_better:.1%}")
        print(f"\n  Spectral coherence diff: {spectral_diff.mean():.4f} +/- {spectral_diff.std():.4f}")
        print(f"  Edge density diff:       {edge_diff.mean():.4f} +/- {edge_diff.std():.4f}")

        results_by_size[K] = {
            'n_ensembles': n_ensembles,
            'individual_order_mean': float(individual_orders.mean()),
            'individual_order_std': float(individual_orders.std()),
            'ensemble_order_mean': float(ensemble_orders.mean()),
            'ensemble_order_std': float(ensemble_orders.std()),
            'diff_mean': float(order_diff.mean()),
            'diff_std': float(order_diff.std()),
            't_stat': float(t_stat),
            'p_ttest': float(p_ttest),
            'w_stat': float(w_stat),
            'p_wilcox': float(p_wilcox),
            'cohens_d': float(d),
            'frac_ensemble_better': float(frac_ensemble_better),
            'spectral_diff_mean': float(spectral_diff.mean()),
            'spectral_diff_std': float(spectral_diff.std()),
            'edge_diff_mean': float(edge_diff.mean()),
            'edge_diff_std': float(edge_diff.std()),
        }

    # Cross-size analysis: does effect increase with K?
    print("\n=== Cross-Size Analysis ===")
    Ks = sorted(results_by_size.keys())
    effects = [results_by_size[k]['cohens_d'] for k in Ks]
    p_values = [results_by_size[k]['p_wilcox'] for k in Ks]

    print(f"{'K':>4} | {'Cohen d':>10} | {'p-value':>12} | {'Ensemble better':>15}")
    print("-" * 50)
    for k in Ks:
        r = results_by_size[k]
        print(f"{k:>4} | {r['cohens_d']:>10.3f} | {r['p_wilcox']:>12.2e} | {r['frac_ensemble_better']:>14.1%}")

    # Spearman correlation: K vs effect size
    if len(Ks) >= 3:
        rho_k_effect, p_k_effect = stats.spearmanr(Ks, effects)
        print(f"\nCorrelation K vs Cohen's d: rho={rho_k_effect:.3f}, p={p_k_effect:.3f}")
    else:
        rho_k_effect, p_k_effect = 0.0, 1.0

    # Determine overall status
    # Use K=5 as primary test case (moderate ensemble size)
    primary_k = 5 if 5 in results_by_size else Ks[len(Ks)//2]
    primary = results_by_size[primary_k]

    # Criteria: p < 0.01, |d| > 0.5
    is_significant = primary['p_wilcox'] < 0.01 and np.abs(primary['cohens_d']) > 0.5
    direction_positive = primary['cohens_d'] > 0  # Ensemble order > individual

    print(f"\n=== CONCLUSION (based on K={primary_k}) ===")
    if is_significant:
        if direction_positive:
            print(f"VALIDATED: Ensemble averaging increases order (d={primary['cohens_d']:.2f}, p={primary['p_wilcox']:.2e})")
            status = "validated"
        else:
            print(f"REFUTED: Ensemble averaging DECREASES order (d={primary['cohens_d']:.2f}, p={primary['p_wilcox']:.2e})")
            status = "refuted"
    else:
        print(f"INCONCLUSIVE: Effect not significant (d={primary['cohens_d']:.2f}, p={primary['p_wilcox']:.2e})")
        status = "inconclusive"

    # Summary metrics for log
    summary_metrics = {
        'primary_k': primary_k,
        'primary_cohens_d': primary['cohens_d'],
        'primary_p_value': primary['p_wilcox'],
        'primary_frac_better': primary['frac_ensemble_better'],
        'k_effect_correlation': float(rho_k_effect) if len(Ks) >= 3 else None,
        'all_sizes_significant': all(results_by_size[k]['p_wilcox'] < 0.01 for k in Ks),
        'status': status,
    }

    # Full results
    full_results = {
        'experiment': 'ensemble_averaging',
        'parameters': {
            'n_ensembles': n_ensembles,
            'ensemble_sizes': ensemble_sizes,
            'image_size': image_size,
            'seed': seed
        },
        'results_by_size': results_by_size,
        'summary': summary_metrics
    }

    # Save results
    results_dir = Path(__file__).parent.parent / "results" / "ensemble_averaging"
    results_dir.mkdir(parents=True, exist_ok=True)

    with open(results_dir / "ensemble_results.json", 'w') as f:
        json.dump(full_results, f, indent=2)

    print(f"\nResults saved to {results_dir / 'ensemble_results.json'}")

    return full_results


if __name__ == "__main__":
    results = run_experiment()
