"""
ESS Mixing Diagnostics Experiment
==================================

Goal: Prove constrained sampling is correct by reporting:
1. Acceptance rate vs iteration
2. Bracket shrink statistics
3. Weight correlation between consecutive samples
4. Brute-force cross-check at low τ

This addresses reviewer concern #2 about mixing credibility.
"""

import torch
import torch.nn as nn
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import json
from pathlib import Path
import zlib


torch.manual_seed(42)
np.random.seed(42)


# ============================================================================
# SIMPLE ARCHITECTURE FOR TESTING
# ============================================================================

class SimpleConvGen(nn.Module):
    """Simple convolutional generator for ESS testing."""
    def __init__(self, latent_dim=64, out_channels=1, img_size=32):
        super().__init__()
        self.img_size = img_size
        self.fc = nn.Linear(latent_dim, 64 * 4 * 4)
        self.net = nn.Sequential(
            nn.Upsample(scale_factor=2),  # 8
            nn.Conv2d(64, 32, 3, padding=1), nn.ReLU(),
            nn.Upsample(scale_factor=2),  # 16
            nn.Conv2d(32, 16, 3, padding=1), nn.ReLU(),
            nn.Upsample(scale_factor=2),  # 32
            nn.Conv2d(16, out_channels, 3, padding=1), nn.Sigmoid(),
        )

    def forward(self, z):
        return self.net(self.fc(z).view(-1, 64, 4, 4))


# ============================================================================
# ORDER METRIC
# ============================================================================

def order_metric(img: torch.Tensor) -> float:
    """Compute order metric."""
    if img.dim() == 4:
        img = img[0]
    img_np = (img.detach().cpu().numpy() * 255).astype(np.uint8)
    comp_ratio = 1 - len(zlib.compress(img_np.tobytes(), 9)) / img_np.nbytes
    img_f = img.detach().cpu().float()
    tv = (torch.abs(img_f[:, 1:, :] - img_f[:, :-1, :]).mean() +
          torch.abs(img_f[:, :, 1:] - img_f[:, :, :-1]).mean()).item()
    return comp_ratio * np.exp(-10 * tv)


# ============================================================================
# ESS WITH DIAGNOSTICS
# ============================================================================

def ellipsoidal_slice_step(model, z_current, score_current, threshold, latent_dim, device,
                           max_steps=100):
    """
    Single ESS step with diagnostic info.
    Returns: new_z, new_score, diagnostics
    """
    # Sample direction from unit sphere
    direction = torch.randn(latent_dim, device=device)
    direction = direction / direction.norm()

    # Initial bracket
    theta_min, theta_max = 0.0, 2 * np.pi
    theta = np.random.uniform(theta_min, theta_max)

    bracket_shrinks = 0
    attempts = 0

    for _ in range(max_steps):
        attempts += 1

        # Propose new point on ellipse
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        z_new = cos_t * z_current + sin_t * direction.unsqueeze(0)

        with torch.no_grad():
            img = model(z_new)
            score_new = order_metric(img)

        if score_new > threshold:
            # Accept
            return z_new, score_new, {
                'accepted': True,
                'attempts': attempts,
                'bracket_shrinks': bracket_shrinks,
                'final_bracket': theta_max - theta_min,
            }

        # Shrink bracket
        bracket_shrinks += 1
        if theta < 0:
            theta_min = theta
        else:
            theta_max = theta

        # New theta within bracket
        theta = np.random.uniform(theta_min, theta_max)

    # Failed - return current
    return z_current, score_current, {
        'accepted': False,
        'attempts': attempts,
        'bracket_shrinks': bracket_shrinks,
        'final_bracket': theta_max - theta_min,
    }


def nested_sampling_with_diagnostics(model, latent_dim, n_live=50, max_iters=300,
                                      device='cpu'):
    """Run nested sampling with detailed diagnostics."""
    model.eval()

    # Initialize
    live_z = torch.randn(n_live, latent_dim, device=device)
    with torch.no_grad():
        live_scores = np.array([order_metric(model(live_z[i:i+1]))
                                for i in range(n_live)])

    diagnostics = {
        'iterations': [],
        'thresholds': [],
        'acceptance_rates': [],
        'bracket_shrinks': [],
        'weight_correlations': [],
        'bits': [],
    }

    log_X = 0.0
    prev_z = None

    for iteration in range(max_iters):
        # Find worst point
        worst_idx = np.argmin(live_scores)
        threshold = live_scores[worst_idx]

        # Record threshold and bits
        diagnostics['iterations'].append(iteration)
        diagnostics['thresholds'].append(threshold)
        diagnostics['bits'].append(-log_X / np.log(2))

        # Contract
        log_X -= 1.0 / n_live

        # Replace via ESS
        z_new, score_new, step_diag = ellipsoidal_slice_step(
            model, live_z[worst_idx:worst_idx+1], threshold, threshold,
            latent_dim, device
        )

        diagnostics['acceptance_rates'].append(1.0 if step_diag['accepted'] else 0.0)
        diagnostics['bracket_shrinks'].append(step_diag['bracket_shrinks'])

        # Weight correlation with previous replacement
        if prev_z is not None:
            corr = torch.nn.functional.cosine_similarity(
                z_new.flatten().unsqueeze(0),
                prev_z.flatten().unsqueeze(0)
            ).item()
            diagnostics['weight_correlations'].append(corr)
        else:
            diagnostics['weight_correlations'].append(0.0)

        prev_z = z_new.clone()
        live_z[worst_idx] = z_new
        live_scores[worst_idx] = score_new

        # Early termination if all scores high
        if live_scores.min() > 0.5:
            break

    return diagnostics


def brute_force_validation(model, latent_dim, threshold, n_samples=10000, device='cpu'):
    """Brute-force sampling to validate NS at low threshold."""
    model.eval()

    scores = []
    with torch.no_grad():
        for _ in range(n_samples):
            z = torch.randn(1, latent_dim, device=device)
            img = model(z)
            scores.append(order_metric(img))

    scores = np.array(scores)

    # Count samples above threshold
    n_above = (scores > threshold).sum()
    fraction = n_above / n_samples

    # Convert to bits
    if fraction > 0:
        bits_bf = -np.log2(fraction)
    else:
        bits_bf = float('inf')

    return {
        'threshold': threshold,
        'n_samples': n_samples,
        'n_above': int(n_above),
        'fraction': fraction,
        'bits_brute_force': bits_bf,
        'score_mean': scores.mean(),
        'score_std': scores.std(),
    }


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    latent_dim = 64
    model = SimpleConvGen(latent_dim=latent_dim).to(device)

    print("\n" + "="*70)
    print("ESS MIXING DIAGNOSTICS EXPERIMENT")
    print("="*70)

    # Run NS with diagnostics
    print("\n--- Running Nested Sampling with Diagnostics ---")
    diag = nested_sampling_with_diagnostics(model, latent_dim, n_live=50,
                                             max_iters=200, device=device)

    # Aggregate statistics
    n_iters = len(diag['iterations'])
    print(f"\nCompleted {n_iters} iterations")

    # Acceptance rate over windows
    window = 20
    windowed_acc = []
    for i in range(0, n_iters - window, window):
        acc = np.mean(diag['acceptance_rates'][i:i+window])
        windowed_acc.append(acc)
        bits = diag['bits'][i]
        print(f"  Iters {i:3d}-{i+window:3d}: Acceptance={acc:.2f}, Bits={bits:.1f}")

    # Bracket shrink statistics
    shrinks = np.array(diag['bracket_shrinks'])
    print(f"\nBracket shrinks: mean={shrinks.mean():.1f}, max={shrinks.max()}")

    # Weight correlations
    corrs = np.array(diag['weight_correlations'][1:])  # Skip first
    print(f"Weight correlations: mean={corrs.mean():.3f}, std={corrs.std():.3f}")

    # Brute-force validation at low thresholds
    print("\n--- Brute-Force Cross-Check ---")
    bf_results = []
    for tau in [0.01, 0.02, 0.05, 0.1]:
        bf = brute_force_validation(model, latent_dim, tau, n_samples=5000, device=device)
        bf_results.append(bf)

        # Find NS estimate at same threshold
        ns_bits = None
        for i, (t, b) in enumerate(zip(diag['thresholds'], diag['bits'])):
            if t >= tau:
                ns_bits = b
                break

        if ns_bits is not None:
            diff = abs(bf['bits_brute_force'] - ns_bits)
            print(f"  τ={tau}: BF={bf['bits_brute_force']:.2f} bits, "
                  f"NS={ns_bits:.2f} bits, Δ={diff:.2f}")
        else:
            print(f"  τ={tau}: BF={bf['bits_brute_force']:.2f} bits (NS didn't reach)")

    # Save results
    results_dir = Path('results/ess_diagnostics')
    results_dir.mkdir(parents=True, exist_ok=True)

    results = {
        'nested_sampling': {
            'n_iterations': n_iters,
            'acceptance_rate_mean': np.mean(diag['acceptance_rates']),
            'acceptance_rate_final': np.mean(diag['acceptance_rates'][-20:]) if n_iters >= 20 else np.mean(diag['acceptance_rates']),
            'bracket_shrink_mean': shrinks.mean(),
            'bracket_shrink_max': int(shrinks.max()),
            'weight_correlation_mean': corrs.mean() if len(corrs) > 0 else 0,
            'final_bits': diag['bits'][-1] if diag['bits'] else 0,
        },
        'brute_force': bf_results,
        'diagnostics': {
            'iterations': diag['iterations'],
            'thresholds': diag['thresholds'],
            'bits': diag['bits'],
            'acceptance_rates': diag['acceptance_rates'],
            'bracket_shrinks': [int(x) for x in diag['bracket_shrinks']],
            'weight_correlations': diag['weight_correlations'],
        }
    }

    with open(results_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)

    # Generate figure
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # 1. Acceptance rate vs iteration
    ax = axes[0, 0]
    window = 10
    rolling_acc = [np.mean(diag['acceptance_rates'][max(0,i-window):i+1])
                   for i in range(len(diag['acceptance_rates']))]
    ax.plot(diag['iterations'], rolling_acc, 'b-', lw=1.5)
    ax.axhline(y=0.5, color='r', linestyle='--', alpha=0.5, label='50% threshold')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Acceptance Rate (rolling avg)')
    ax.set_title('ESS Acceptance Rate Over Time')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. Bracket shrink histogram
    ax = axes[0, 1]
    ax.hist(shrinks, bins=30, edgecolor='black', alpha=0.7)
    ax.axvline(x=shrinks.mean(), color='r', linestyle='--', label=f'Mean={shrinks.mean():.1f}')
    ax.set_xlabel('Bracket Shrinks per Step')
    ax.set_ylabel('Count')
    ax.set_title('Bracket Shrink Distribution')
    ax.legend()

    # 3. Weight correlation over time
    ax = axes[1, 0]
    if len(corrs) > 0:
        ax.plot(diag['iterations'][1:], corrs, 'g-', lw=1, alpha=0.7)
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax.axhline(y=corrs.mean(), color='r', linestyle='--',
                   label=f'Mean={corrs.mean():.3f}')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Cosine Similarity')
    ax.set_title('Weight Correlation Between Consecutive Samples')
    ax.set_ylim(-1, 1)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 4. NS vs Brute-Force comparison
    ax = axes[1, 1]
    taus = [bf['threshold'] for bf in bf_results]
    bf_bits = [bf['bits_brute_force'] for bf in bf_results]

    # Get NS bits at same thresholds
    ns_bits_at_tau = []
    for tau in taus:
        ns_b = None
        for t, b in zip(diag['thresholds'], diag['bits']):
            if t >= tau:
                ns_b = b
                break
        ns_bits_at_tau.append(ns_b if ns_b is not None else np.nan)

    x = np.arange(len(taus))
    width = 0.35
    ax.bar(x - width/2, bf_bits, width, label='Brute Force', color='steelblue')
    ax.bar(x + width/2, ns_bits_at_tau, width, label='Nested Sampling', color='coral')
    ax.set_xticks(x)
    ax.set_xticklabels([f'τ={t}' for t in taus])
    ax.set_ylabel('Bits')
    ax.set_title('NS vs Brute-Force Validation')
    ax.legend()

    plt.tight_layout()
    plt.savefig('figures/ess_diagnostics.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('figures/ess_diagnostics.png', dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\nFigure saved to figures/ess_diagnostics.pdf")
    print(f"Results saved to {results_dir}/results.json")

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Acceptance rate (overall): {np.mean(diag['acceptance_rates']):.2f}")
    print(f"Acceptance rate (final 20 iters): {np.mean(diag['acceptance_rates'][-20:]):.2f}")
    print(f"Bracket shrinks (mean): {shrinks.mean():.1f}")
    print(f"Weight correlation (mean): {corrs.mean():.3f}")

    # Check BF agreement
    valid_comparisons = [(bf['threshold'], bf['bits_brute_force'], ns)
                         for bf, ns in zip(bf_results, ns_bits_at_tau)
                         if not np.isnan(ns) and bf['bits_brute_force'] < float('inf')]

    if valid_comparisons:
        avg_diff = np.mean([abs(bf - ns) for _, bf, ns in valid_comparisons])
        print(f"NS-BF agreement (avg |Δ|): {avg_diff:.2f} bits")

        if avg_diff < 0.5:
            print("EXCELLENT: NS and BF agree within 0.5 bits")
        elif avg_diff < 1.0:
            print("GOOD: NS and BF agree within 1 bit")
        else:
            print("WARNING: NS-BF disagreement > 1 bit")
    print("="*70)

    return results


if __name__ == '__main__':
    main()
