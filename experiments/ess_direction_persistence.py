"""
RES-198: ESS contraction direction persistence increases with order threshold during NS

Hypothesis: At higher order thresholds, ESS contractions occur in more consistent
directions (nu vector) across consecutive samples.

Method:
1. Run NS tracking the nu vector and final phi angle for each ESS call
2. Compute direction persistence: cosine similarity between consecutive nu vectors
   and between consecutive accepted directions (w' - w)
3. Compare persistence at low vs high order phases

Rationale:
- RES-165 shows trajectory dimension collapses at high order
- RES-191 shows order gain per contraction decreases
- If the acceptable weight space narrows, ESS may find valid proposals
  by searching in similar directions repeatedly
"""

import numpy as np
import sys
sys.path.insert(0, '/Users/matt/Development/monochrome_noise_converger')

from core.thermo_sampler_v3 import CPPN, order_multiplicative, log_prior, PRIOR_SIGMA
from scipy.stats import spearmanr, mannwhitneyu

def ess_with_tracking(cppn, threshold, image_size, order_fn, max_contractions=100, max_restarts=5):
    """ESS that returns the nu vector and final phi for direction analysis."""
    current_w = cppn.get_weights()
    n_params = len(current_w)
    total_contractions = 0

    for restart in range(max_restarts):
        nu = np.random.randn(n_params) * PRIOR_SIGMA

        phi = np.random.uniform(0, 2 * np.pi)
        phi_min = phi - 2 * np.pi
        phi_max = phi

        n_contractions = 0

        while n_contractions < max_contractions:
            proposal_w = current_w * np.cos(phi) + nu * np.sin(phi)

            proposal_cppn = cppn.copy()
            proposal_cppn.set_weights(proposal_w)
            proposal_img = proposal_cppn.render(image_size)
            proposal_order = order_fn(proposal_img)

            if proposal_order >= threshold:
                # Return extra info: nu vector, final phi, and weight displacement
                displacement = proposal_w - current_w
                return (proposal_cppn, proposal_img, proposal_order,
                        log_prior(proposal_cppn), total_contractions + n_contractions, True,
                        nu, phi, displacement)

            if phi < 0:
                phi_min = phi
            else:
                phi_max = phi

            phi = np.random.uniform(phi_min, phi_max)
            n_contractions += 1

            if phi_max - phi_min < 1e-10:
                break

        total_contractions += n_contractions

    # Failed - return current point
    current_img = cppn.render(image_size)
    displacement = np.zeros(n_params)
    return (cppn, current_img, order_fn(current_img), log_prior(cppn),
            total_contractions, False, np.zeros(n_params), 0.0, displacement)


def cosine_similarity(v1, v2):
    """Cosine similarity between two vectors."""
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 < 1e-10 or norm2 < 1e-10:
        return 0.0
    return np.dot(v1, v2) / (norm1 * norm2)


def run_ns_with_direction_tracking(n_live=20, n_iterations=500, image_size=32, seed=None):
    """Run NS tracking ESS direction information at each step."""
    if seed is not None:
        np.random.seed(seed)

    order_fn = order_multiplicative

    # Initialize live points
    live_points = []
    for _ in range(n_live):
        cppn = CPPN()
        img = cppn.render(image_size)
        order = order_fn(img)
        live_points.append({
            'cppn': cppn,
            'image': img,
            'order': order,
            'log_prior': log_prior(cppn)
        })

    # Track ESS direction info for each iteration
    direction_data = []
    prev_nu = None
    prev_displacement = None

    for iteration in range(n_iterations):
        # Find worst point
        orders = [lp['order'] for lp in live_points]
        worst_idx = np.argmin(orders)
        threshold = orders[worst_idx]

        # Select seed (exclude worst)
        valid_seeds = [i for i in range(n_live) if i != worst_idx]
        seed_idx = np.random.choice(valid_seeds)

        # Run ESS with tracking
        result = ess_with_tracking(
            live_points[seed_idx]['cppn'], threshold, image_size, order_fn
        )
        (new_cppn, new_img, new_order, new_logp,
         n_contractions, success, nu, phi, displacement) = result

        # Compute direction persistence
        nu_persistence = 0.0
        disp_persistence = 0.0

        if prev_nu is not None and np.linalg.norm(nu) > 1e-10:
            nu_persistence = abs(cosine_similarity(nu, prev_nu))

        if prev_displacement is not None and np.linalg.norm(displacement) > 1e-10:
            disp_persistence = abs(cosine_similarity(displacement, prev_displacement))

        direction_data.append({
            'iteration': iteration,
            'threshold': threshold,
            'order': new_order,
            'n_contractions': n_contractions,
            'success': success,
            'nu_persistence': nu_persistence,
            'disp_persistence': disp_persistence,
            'nu_norm': np.linalg.norm(nu),
            'disp_norm': np.linalg.norm(displacement),
            'final_phi': phi
        })

        # Update for next iteration
        prev_nu = nu.copy() if np.linalg.norm(nu) > 1e-10 else prev_nu
        prev_displacement = displacement.copy() if np.linalg.norm(displacement) > 1e-10 else prev_displacement

        # Replace worst with new
        live_points[worst_idx] = {
            'cppn': new_cppn,
            'image': new_img,
            'order': new_order,
            'log_prior': new_logp
        }

    return direction_data


def main():
    print("RES-198: ESS contraction direction persistence increases with order threshold")
    print("=" * 80)

    # Run multiple NS chains
    n_chains = 10
    n_iterations = 600
    all_data = []

    print(f"\nRunning {n_chains} NS chains with {n_iterations} iterations each...")

    for chain_idx in range(n_chains):
        print(f"  Chain {chain_idx + 1}/{n_chains}...", end=" ", flush=True)
        data = run_ns_with_direction_tracking(
            n_live=20, n_iterations=n_iterations, seed=42 + chain_idx
        )
        for d in data:
            d['chain'] = chain_idx
        all_data.extend(data)
        print(f"done (final order: {data[-1]['order']:.4f})")

    # Convert to arrays for analysis
    thresholds = np.array([d['threshold'] for d in all_data])
    nu_persistence = np.array([d['nu_persistence'] for d in all_data])
    disp_persistence = np.array([d['disp_persistence'] for d in all_data])
    orders = np.array([d['order'] for d in all_data])

    # Filter out first iterations (no prev data) and failed ESS
    valid_mask = (nu_persistence > 0) | (disp_persistence > 0)
    thresholds = thresholds[valid_mask]
    nu_persistence = nu_persistence[valid_mask]
    disp_persistence = disp_persistence[valid_mask]
    orders = orders[valid_mask]

    print(f"\n{len(thresholds)} valid data points after filtering")

    # Split into low/mid/high order phases
    t_low = np.percentile(thresholds, 33)
    t_high = np.percentile(thresholds, 67)

    low_mask = thresholds < t_low
    mid_mask = (thresholds >= t_low) & (thresholds < t_high)
    high_mask = thresholds >= t_high

    print(f"\nPhase thresholds: low < {t_low:.4f}, mid [{t_low:.4f}, {t_high:.4f}), high >= {t_high:.4f}")
    print(f"Counts: low={low_mask.sum()}, mid={mid_mask.sum()}, high={high_mask.sum()}")

    # Nu direction persistence by phase
    print("\n--- Nu Vector Persistence (|cos(nu_t, nu_{t-1})|) ---")
    nu_low = nu_persistence[low_mask]
    nu_mid = nu_persistence[mid_mask]
    nu_high = nu_persistence[high_mask]

    print(f"Low-order:  mean={nu_low.mean():.4f}, std={nu_low.std():.4f}")
    print(f"Mid-order:  mean={nu_mid.mean():.4f}, std={nu_mid.std():.4f}")
    print(f"High-order: mean={nu_high.mean():.4f}, std={nu_high.std():.4f}")

    # Displacement persistence by phase
    print("\n--- Displacement Persistence (|cos(d_t, d_{t-1})|) ---")
    disp_low = disp_persistence[low_mask]
    disp_mid = disp_persistence[mid_mask]
    disp_high = disp_persistence[high_mask]

    print(f"Low-order:  mean={disp_low.mean():.4f}, std={disp_low.std():.4f}")
    print(f"Mid-order:  mean={disp_mid.mean():.4f}, std={disp_mid.std():.4f}")
    print(f"High-order: mean={disp_high.mean():.4f}, std={disp_high.std():.4f}")

    # Statistical tests
    print("\n--- Statistical Analysis ---")

    # Spearman correlation with threshold
    rho_nu, p_nu = spearmanr(thresholds, nu_persistence)
    rho_disp, p_disp = spearmanr(thresholds, disp_persistence)

    print(f"Nu persistence vs threshold:   rho={rho_nu:.4f}, p={p_nu:.2e}")
    print(f"Disp persistence vs threshold: rho={rho_disp:.4f}, p={p_disp:.2e}")

    # Mann-Whitney U: low vs high
    stat_nu, p_mw_nu = mannwhitneyu(nu_low, nu_high, alternative='less')
    stat_disp, p_mw_disp = mannwhitneyu(disp_low, disp_high, alternative='less')

    print(f"\nMann-Whitney (low < high hypothesis):")
    print(f"  Nu persistence:   p={p_mw_nu:.2e}")
    print(f"  Disp persistence: p={p_mw_disp:.2e}")

    # Effect size (Cohen's d)
    def cohens_d(group1, group2):
        n1, n2 = len(group1), len(group2)
        var1, var2 = group1.var(), group2.var()
        pooled_std = np.sqrt(((n1-1)*var1 + (n2-1)*var2) / (n1 + n2 - 2))
        if pooled_std < 1e-10:
            return 0.0
        return (group2.mean() - group1.mean()) / pooled_std

    d_nu = cohens_d(nu_low, nu_high)
    d_disp = cohens_d(disp_low, disp_high)

    print(f"\nEffect size (Cohen's d, high vs low):")
    print(f"  Nu persistence:   d={d_nu:.2f}")
    print(f"  Disp persistence: d={d_disp:.2f}")

    # Determine verdict
    print("\n" + "=" * 80)
    print("VERDICT:")

    # Hypothesis: persistence increases with threshold
    # Success criteria: d > 0.5, p < 0.01, positive rho

    primary_metric = "displacement" if abs(d_disp) > abs(d_nu) else "nu"
    primary_d = d_disp if primary_metric == "displacement" else d_nu
    primary_p = p_mw_disp if primary_metric == "displacement" else p_mw_nu
    primary_rho = rho_disp if primary_metric == "displacement" else rho_nu

    if primary_d > 0.5 and primary_p < 0.01 and primary_rho > 0:
        status = "validated"
        print(f"VALIDATED: Direction persistence increases with order threshold")
    elif primary_d < -0.5 and primary_p < 0.01 and primary_rho < 0:
        status = "refuted"
        print(f"REFUTED: Direction persistence DECREASES with order threshold (opposite)")
    else:
        status = "refuted"
        print(f"REFUTED: No significant relationship (d={primary_d:.2f}, p={primary_p:.2e})")

    print(f"\nPrimary metric: {primary_metric} persistence")
    print(f"  Effect size d={primary_d:.2f}")
    print(f"  p-value={primary_p:.2e}")
    print(f"  Spearman rho={primary_rho:.4f}")

    # Additional analysis: is nu persistence just random?
    # Random vectors in 5D have expected |cos| ~ 0.3
    print("\n--- Baseline Check ---")
    expected_random = 0.3  # approximate for 5D
    print(f"Expected persistence for random 5D vectors: ~{expected_random:.2f}")
    print(f"Observed mean nu persistence: {nu_persistence.mean():.4f}")

    return {
        'status': status,
        'd_nu': d_nu,
        'd_disp': d_disp,
        'rho_nu': rho_nu,
        'rho_disp': rho_disp,
        'p_nu': p_mw_nu,
        'p_disp': p_mw_disp,
        'nu_low_mean': nu_low.mean(),
        'nu_high_mean': nu_high.mean(),
        'disp_low_mean': disp_low.mean(),
        'disp_high_mean': disp_high.mean()
    }


if __name__ == "__main__":
    results = main()
