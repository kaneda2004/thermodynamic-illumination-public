"""
ESS Mixing Diagnostics v2
==========================

Uses the MLP generator (harder case) with higher threshold (0.5)
to actually exercise the nested sampling algorithm.

Reports:
1. Acceptance rate over iterations
2. Bracket shrink statistics
3. Weight correlation between consecutive samples
4. Brute-force cross-validation at multiple thresholds
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import json
from pathlib import Path
import zlib


torch.manual_seed(42)
np.random.seed(42)


class MLPGen(nn.Module):
    """MLP generator - moderate structure difficulty."""
    def __init__(self, latent_dim=32, out_channels=1, img_size=32):
        super().__init__()
        self.img_size, self.out_channels = img_size, out_channels
        out_dim = out_channels * img_size * img_size
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 128), nn.ReLU(),
            nn.Linear(128, 256), nn.ReLU(),
            nn.Linear(256, out_dim), nn.Sigmoid(),
        )

    def forward(self, z):
        return self.net(z).view(-1, self.out_channels, self.img_size, self.img_size)


def order_metric(img: torch.Tensor) -> float:
    """Compute structure score."""
    if img.dim() == 4:
        img = img[0]
    img_np = (img.detach().cpu().numpy() * 255).astype(np.uint8)
    comp_ratio = 1 - len(zlib.compress(img_np.tobytes(), 9)) / img_np.nbytes
    img_f = img.detach().cpu().float()
    tv = (torch.abs(img_f[:, 1:, :] - img_f[:, :-1, :]).mean() +
          torch.abs(img_f[:, :, 1:] - img_f[:, :, :-1]).mean()).item()
    return comp_ratio * np.exp(-5 * tv)  # Less aggressive TV penalty


def ellipsoidal_slice_step(model, z_current, threshold, latent_dim, device, max_steps=100):
    """Single ESS step with diagnostics."""
    direction = torch.randn(latent_dim, device=device)
    direction = direction / direction.norm()

    theta_min, theta_max = 0.0, 2 * np.pi
    theta = np.random.uniform(theta_min, theta_max)

    bracket_shrinks = 0
    initial_bracket = theta_max - theta_min

    for attempt in range(max_steps):
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        z_new = cos_t * z_current + sin_t * direction.unsqueeze(0)

        with torch.no_grad():
            score_new = order_metric(model(z_new))

        if score_new > threshold:
            return z_new, score_new, {
                'accepted': True,
                'attempts': attempt + 1,
                'bracket_shrinks': bracket_shrinks,
                'initial_bracket': initial_bracket,
                'final_bracket': theta_max - theta_min,
            }

        bracket_shrinks += 1
        if theta < 0:
            theta_min = theta
        else:
            theta_max = theta

        if theta_max - theta_min < 1e-10:
            break

        theta = np.random.uniform(theta_min, theta_max)

    # Rejection - keep current
    with torch.no_grad():
        current_score = order_metric(model(z_current))
    return z_current, current_score, {
        'accepted': False,
        'attempts': max_steps,
        'bracket_shrinks': bracket_shrinks,
        'initial_bracket': initial_bracket,
        'final_bracket': theta_max - theta_min,
    }


def nested_sampling_with_diagnostics(model, latent_dim, n_live=100, max_iters=500,
                                      device='cpu'):
    """Full nested sampling with detailed diagnostics."""
    model.eval()

    # Initialize live points
    live_z = torch.randn(n_live, latent_dim, device=device)
    with torch.no_grad():
        live_scores = np.array([order_metric(model(live_z[i:i+1]))
                                for i in range(n_live)])

    diagnostics = {
        'iterations': [],
        'thresholds': [],
        'log_X': [],
        'bits': [],
        'acceptance_rates': [],
        'bracket_shrinks': [],
        'weight_norms': [],
        'weight_correlations': [],
    }

    log_X = 0.0
    prev_z = None
    window_accepts = []

    for iteration in range(max_iters):
        worst_idx = np.argmin(live_scores)
        threshold = live_scores[worst_idx]

        # Check termination
        if threshold > 0.9:
            print(f"  Terminated at iteration {iteration}: threshold={threshold:.3f}")
            break

        diagnostics['iterations'].append(iteration)
        diagnostics['thresholds'].append(threshold)
        diagnostics['log_X'].append(log_X)
        diagnostics['bits'].append(-log_X / np.log(2))

        # Contract volume
        log_X -= 1.0 / n_live

        # ESS replacement
        z_new, score_new, step_diag = ellipsoidal_slice_step(
            model, live_z[worst_idx:worst_idx+1], threshold, latent_dim, device
        )

        diagnostics['acceptance_rates'].append(1.0 if step_diag['accepted'] else 0.0)
        diagnostics['bracket_shrinks'].append(step_diag['bracket_shrinks'])
        diagnostics['weight_norms'].append(z_new.norm().item())

        # Weight correlation with previous
        if prev_z is not None:
            corr = torch.nn.functional.cosine_similarity(
                z_new.flatten().unsqueeze(0),
                prev_z.flatten().unsqueeze(0)
            ).item()
            diagnostics['weight_correlations'].append(corr)

        prev_z = z_new.clone()
        live_z[worst_idx] = z_new
        live_scores[worst_idx] = score_new

        # Print progress
        if (iteration + 1) % 50 == 0:
            recent_acc = np.mean(diagnostics['acceptance_rates'][-50:])
            print(f"  Iter {iteration+1}: threshold={threshold:.3f}, bits={-log_X/np.log(2):.2f}, acc={recent_acc:.2f}")

    return diagnostics


def brute_force_validation(model, latent_dim, thresholds, n_samples=10000, device='cpu'):
    """Brute-force sampling to validate NS estimates."""
    model.eval()

    scores = []
    with torch.no_grad():
        for _ in range(n_samples):
            z = torch.randn(1, latent_dim, device=device)
            scores.append(order_metric(model(z)))
    scores = np.array(scores)

    results = []
    for tau in thresholds:
        n_above = (scores > tau).sum()
        fraction = n_above / n_samples

        if fraction > 0:
            bits_bf = -np.log2(fraction)
        else:
            bits_bf = float('inf')

        results.append({
            'threshold': tau,
            'n_above': int(n_above),
            'fraction': fraction,
            'bits_brute_force': bits_bf,
        })

    return results, scores


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    latent_dim = 32
    model = MLPGen(latent_dim=latent_dim, out_channels=1, img_size=32).to(device)

    print("\n" + "="*70)
    print("ESS MIXING DIAGNOSTICS v2")
    print("="*70)

    # First, characterize the score distribution
    print("\n--- Score Distribution ---")
    test_scores = []
    with torch.no_grad():
        for _ in range(1000):
            z = torch.randn(1, latent_dim, device=device)
            test_scores.append(order_metric(model(z)))
    test_scores = np.array(test_scores)
    print(f"Random samples: mean={test_scores.mean():.3f}, std={test_scores.std():.3f}")
    print(f"Range: [{test_scores.min():.3f}, {test_scores.max():.3f}]")
    print(f"Fraction > 0.3: {(test_scores > 0.3).mean():.4f}")
    print(f"Fraction > 0.5: {(test_scores > 0.5).mean():.4f}")
    print(f"Fraction > 0.7: {(test_scores > 0.7).mean():.4f}")

    # Run nested sampling
    print("\n--- Nested Sampling ---")
    diag = nested_sampling_with_diagnostics(model, latent_dim, n_live=100,
                                             max_iters=300, device=device)

    n_iters = len(diag['iterations'])
    print(f"\nCompleted {n_iters} iterations")

    if n_iters > 0:
        # Aggregate statistics
        acc_rates = diag['acceptance_rates']
        shrinks = np.array(diag['bracket_shrinks'])
        corrs = np.array(diag['weight_correlations']) if diag['weight_correlations'] else np.array([0])

        print(f"Acceptance rate (overall): {np.mean(acc_rates):.3f}")
        print(f"Bracket shrinks (mean): {shrinks.mean():.1f}, (max): {shrinks.max()}")
        print(f"Weight correlations: mean={corrs.mean():.3f}, std={corrs.std():.3f}")

        # Brute-force validation
        print("\n--- Brute-Force Validation ---")
        bf_thresholds = [0.2, 0.3, 0.4, 0.5, 0.6]
        bf_results, bf_scores = brute_force_validation(model, latent_dim, bf_thresholds,
                                                        n_samples=5000, device=device)

        print(f"{'Threshold':>10} {'BF Bits':>10} {'NS Bits':>10} {'Δ':>8}")
        print("-"*40)

        comparisons = []
        for bf in bf_results:
            tau = bf['threshold']
            # Find NS estimate
            ns_bits = None
            for t, b in zip(diag['thresholds'], diag['bits']):
                if t >= tau:
                    ns_bits = b
                    break

            if ns_bits is not None and bf['bits_brute_force'] < float('inf'):
                delta = abs(bf['bits_brute_force'] - ns_bits)
                comparisons.append(delta)
                print(f"{tau:>10.2f} {bf['bits_brute_force']:>10.2f} {ns_bits:>10.2f} {delta:>8.2f}")
            else:
                print(f"{tau:>10.2f} {bf['bits_brute_force']:>10.2f} {'N/A':>10} {'N/A':>8}")

        if comparisons:
            avg_delta = np.mean(comparisons)
            print(f"\nAverage |Δ|: {avg_delta:.3f} bits")

    else:
        print("No iterations completed - all samples already above threshold")
        acc_rates = [1.0]
        shrinks = np.array([0])
        corrs = np.array([0])
        bf_results = []
        comparisons = []
        avg_delta = 0

    # Save results
    results_dir = Path('results/ess_diagnostics_v2')
    results_dir.mkdir(parents=True, exist_ok=True)

    results = {
        'n_iterations': n_iters,
        'acceptance_rate_mean': float(np.mean(acc_rates)),
        'bracket_shrink_mean': float(shrinks.mean()),
        'bracket_shrink_max': int(shrinks.max()) if len(shrinks) > 0 else 0,
        'weight_correlation_mean': float(corrs.mean()),
        'avg_bf_delta': float(avg_delta) if comparisons else None,
        'diagnostics': diag,
        'brute_force': bf_results,
    }

    with open(results_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2, default=float)

    # Plot
    if n_iters > 10:
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # Acceptance rate
        ax = axes[0, 0]
        window = max(10, n_iters // 20)
        rolling_acc = [np.mean(acc_rates[max(0,i-window):i+1]) for i in range(len(acc_rates))]
        ax.plot(diag['iterations'], rolling_acc, 'b-', lw=1.5)
        ax.axhline(y=0.5, color='r', linestyle='--', alpha=0.5, label='50%')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Acceptance Rate')
        ax.set_title('ESS Acceptance Rate Over Time')
        ax.grid(True, alpha=0.3)
        ax.legend()

        # Threshold evolution
        ax = axes[0, 1]
        ax.plot(diag['iterations'], diag['thresholds'], 'g-', lw=1.5)
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Threshold (Order Score)')
        ax.set_title('Threshold Evolution')
        ax.grid(True, alpha=0.3)

        # Bracket shrinks
        ax = axes[1, 0]
        ax.hist(shrinks, bins=30, edgecolor='black', alpha=0.7)
        ax.axvline(x=shrinks.mean(), color='r', linestyle='--', label=f'Mean={shrinks.mean():.1f}')
        ax.set_xlabel('Bracket Shrinks per Step')
        ax.set_ylabel('Count')
        ax.set_title('ESS Bracket Shrink Distribution')
        ax.legend()

        # NS vs BF comparison
        ax = axes[1, 1]
        if bf_results and n_iters > 0:
            taus = [bf['threshold'] for bf in bf_results if bf['bits_brute_force'] < float('inf')]
            bf_bits = [bf['bits_brute_force'] for bf in bf_results if bf['bits_brute_force'] < float('inf')]

            ns_bits_list = []
            for tau in taus:
                ns_b = None
                for t, b in zip(diag['thresholds'], diag['bits']):
                    if t >= tau:
                        ns_b = b
                        break
                ns_bits_list.append(ns_b if ns_b else 0)

            x = np.arange(len(taus))
            width = 0.35
            ax.bar(x - width/2, bf_bits, width, label='Brute Force', color='steelblue')
            ax.bar(x + width/2, ns_bits_list, width, label='Nested Sampling', color='coral')
            ax.set_xticks(x)
            ax.set_xticklabels([f'τ={t}' for t in taus])
            ax.set_ylabel('Bits')
            ax.set_title('NS vs Brute-Force Validation')
            ax.legend()
        else:
            ax.text(0.5, 0.5, 'No comparisons available', ha='center', va='center',
                   transform=ax.transAxes)

        plt.tight_layout()
        plt.savefig('figures/ess_diagnostics_v2.pdf', dpi=300, bbox_inches='tight')
        plt.savefig('figures/ess_diagnostics_v2.png', dpi=150, bbox_inches='tight')
        plt.close()

        print(f"\nFigure saved to figures/ess_diagnostics_v2.pdf")

    print(f"Results saved to {results_dir}/results.json")

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Iterations completed: {n_iters}")
    print(f"Acceptance rate: {np.mean(acc_rates):.3f}")
    print(f"Bracket shrinks (mean): {shrinks.mean():.1f}")

    if comparisons:
        print(f"NS-BF agreement: {avg_delta:.3f} bits average |Δ|")
        if avg_delta < 0.5:
            print("EXCELLENT: NS agrees with brute-force within 0.5 bits")
        elif avg_delta < 1.0:
            print("GOOD: NS agrees with brute-force within 1 bit")
    print("="*70)

    return results


if __name__ == '__main__':
    main()
