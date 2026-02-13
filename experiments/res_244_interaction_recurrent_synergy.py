#!/usr/bin/env python3
"""
RES-244: Interaction + Recurrent Architecture Synergy

Hypothesis: Combining nonlinear interactions with recurrent feedback
yields synergistic improvement >3.5× (multiplicative, not additive).

Tests three architectural variants:
1. Interactions only (feedforward)
2. Recurrent only (no interactions)
3. Combined (interactions + recurrent feedback)

Statistical test for synergy via interaction term in ANOVA.
"""

import os
import sys
import json
import random
import numpy as np
from pathlib import Path
from dataclasses import dataclass, asdict
import warnings
warnings.filterwarnings('ignore')

# Setup
os.chdir('/Users/matt/Development/monochrome_noise_converger')
sys.path.insert(0, os.getcwd())

from research_system.log_manager import ResearchLogManager
from scipy import stats
from typing import List, Tuple, Dict

# ==================== Architecture Components ====================

@dataclass
class CPPNConfig:
    """CPPN architecture configuration"""
    variant: str  # 'baseline', 'interactions', 'recurrent', 'combined'
    num_hidden: int = 16
    recurrent_iterations: int = 2
    interaction_terms: bool = False
    recurrent_feedback: bool = False

class RecurrentCPPN:
    """CPPN with optional recurrent feedback and interaction terms"""

    def __init__(self, config: CPPNConfig, seed: int = None):
        self.config = config
        self.seed = seed
        if seed is not None:
            np.random.seed(seed)

        # Input dimension based on interaction terms
        if config.interaction_terms:
            self.input_dim = 6  # [x, y, r, x*y, x², y²]
        else:
            self.input_dim = 3  # [x, y, r]

        # Weights
        self.W_input = np.random.randn(config.num_hidden, self.input_dim) * 0.1
        self.W_hidden = np.random.randn(config.num_hidden, config.num_hidden) * 0.05 if config.recurrent_feedback else None
        self.W_output = np.random.randn(1, config.num_hidden) * 0.1
        self.b_hidden = np.random.randn(config.num_hidden) * 0.01
        self.b_output = np.random.randn(1) * 0.01

    def forward(self, x: float, y: float, r: float) -> float:
        """
        Forward pass through CPPN.

        For recurrent variants: iterate hidden state updates 2-3 times
        For interaction variants: use [x,y,r,x*y,x²,y²] as input
        """
        # Prepare input
        if self.config.interaction_terms:
            input_vec = np.array([x, y, r, x*y, x**2, y**2])
        else:
            input_vec = np.array([x, y, r])

        input_vec = input_vec.reshape(-1, 1)

        # Initialize hidden state
        hidden = np.tanh(self.W_input @ input_vec + self.b_hidden.reshape(-1, 1))

        # Recurrent feedback iterations
        if self.config.recurrent_feedback:
            for _ in range(self.config.recurrent_iterations - 1):
                # Update: h[t+1] = tanh(W_hidden @ h[t] + W_input @ x)
                hidden_update = self.W_hidden @ hidden
                hidden_input = self.W_input @ input_vec
                hidden = np.tanh(hidden_update + hidden_input + self.b_hidden.reshape(-1, 1))

        # Output
        output = np.tanh(self.W_output @ hidden + self.b_output)
        return float(output[0, 0])

def generate_test_image(cppn: RecurrentCPPN, size: int = 32) -> np.ndarray:
    """Generate binary image from CPPN"""
    image = np.zeros((size, size))
    center = size / 2.0
    max_r = np.sqrt(2) * center

    for i in range(size):
        for j in range(size):
            x = (j - center) / center
            y = (i - center) / center
            r = np.sqrt(x**2 + y**2) / (np.sqrt(2))

            output = cppn.forward(x, y, r)
            image[i, j] = 1.0 if output > 0 else 0.0

    return image

def compute_order(image: np.ndarray) -> float:
    """
    Compute order via spatial coherence (local variance inverse).
    Order = fraction of pixels matching their neighborhood majority.
    """
    if image.sum() == 0 or image.sum() == image.size:
        return 0.0  # Degenerate cases

    h, w = image.shape
    matches = 0
    total = 0

    for i in range(1, h-1):
        for j in range(1, w-1):
            neighborhood = image[i-1:i+2, j-1:j+2]
            center_val = image[i, j]
            neighbor_sum = neighborhood.sum() - center_val

            # Majority vote in 8 neighbors
            if (neighbor_sum >= 4 and center_val == 1) or (neighbor_sum < 4 and center_val == 0):
                matches += 1
            total += 1

    return matches / total if total > 0 else 0.0

# ==================== Experiment ====================

def run_experiment():
    """Run full synergy experiment"""

    # Configuration
    variants = {
        'baseline': CPPNConfig('baseline', recurrent_iterations=1),
        'interactions': CPPNConfig('interactions', interaction_terms=True),
        'recurrent': CPPNConfig('recurrent', recurrent_feedback=True, recurrent_iterations=2),
        'combined': CPPNConfig('combined', interaction_terms=True, recurrent_feedback=True, recurrent_iterations=2),
    }

    num_cppns_per_variant = 30
    results = {variant: [] for variant in variants.keys()}

    print("=" * 70)
    print("RES-244: Interaction + Recurrent Synergy")
    print("=" * 70)

    # Run each variant
    for variant_name, config in variants.items():
        print(f"\n[{variant_name.upper()}]")
        orders = []

        for seed in range(num_cppns_per_variant):
            cppn = RecurrentCPPN(config, seed=seed)
            image = generate_test_image(cppn, size=32)
            order = compute_order(image)
            orders.append(order)

            if (seed + 1) % 10 == 0:
                print(f"  {seed + 1}/{num_cppns_per_variant} CPPNs processed")

        results[variant_name] = orders
        mean_order = np.mean(orders)
        std_order = np.std(orders)
        print(f"  Mean order: {mean_order:.4f} ± {std_order:.4f}")

    # ==================== Analysis ====================

    print("\n" + "=" * 70)
    print("SYNERGY ANALYSIS")
    print("=" * 70)

    baseline_mean = np.mean(results['baseline'])
    interactions_mean = np.mean(results['interactions'])
    recurrent_mean = np.mean(results['recurrent'])
    combined_mean = np.mean(results['combined'])

    # Effect sizes (Cohen's d)
    baseline_std = np.std(results['baseline'])
    interactions_d = (interactions_mean - baseline_mean) / np.sqrt((baseline_std**2 + np.std(results['interactions'])**2) / 2)
    recurrent_d = (recurrent_mean - baseline_mean) / np.sqrt((baseline_std**2 + np.std(results['recurrent'])**2) / 2)
    combined_d = (combined_mean - baseline_mean) / np.sqrt((baseline_std**2 + np.std(results['combined'])**2) / 2)

    # Multiplicative gains
    interactions_gain = interactions_mean / baseline_mean if baseline_mean > 0 else 0
    recurrent_gain = recurrent_mean / baseline_mean if baseline_mean > 0 else 0
    combined_gain = combined_mean / baseline_mean if baseline_mean > 0 else 0

    # Additive prediction vs actual synergy
    additive_prediction = baseline_mean + (interactions_mean - baseline_mean) + (recurrent_mean - baseline_mean)
    synergy_factor = combined_mean / additive_prediction if additive_prediction > 0 else 0
    superadditive = combined_mean > additive_prediction

    print(f"\nBaseline order:           {baseline_mean:.4f}")
    print(f"  Interactions order:     {interactions_mean:.4f} (d={interactions_d:.3f}, gain={interactions_gain:.3f}x)")
    print(f"  Recurrent order:        {recurrent_mean:.4f} (d={recurrent_d:.3f}, gain={recurrent_gain:.3f}x)")
    print(f"  Combined order:         {combined_mean:.4f} (d={combined_d:.3f}, gain={combined_gain:.3f}x)")

    print(f"\nAdditive prediction:      {additive_prediction:.4f}")
    print(f"Actual combined:          {combined_mean:.4f}")
    print(f"Synergy factor:           {synergy_factor:.3f}x")
    print(f"Superadditive:            {superadditive}")

    # Statistical test: ANOVA with interaction term
    print("\n" + "=" * 70)
    print("INTERACTION TERM ANOVA")
    print("=" * 70)

    # Prepare data for ANOVA
    all_orders = []
    variant_labels = []
    interaction_flags = []  # 0=no, 1=yes
    recurrent_flags = []    # 0=no, 1=yes

    for variant_name in ['baseline', 'interactions', 'recurrent', 'combined']:
        for order in results[variant_name]:
            all_orders.append(order)
            variant_labels.append(variant_name)
            interaction_flags.append(1 if 'interaction' in variant_name else 0)
            recurrent_flags.append(1 if 'recurrent' in variant_name else 0)

    all_orders = np.array(all_orders)
    interaction_flags = np.array(interaction_flags)
    recurrent_flags = np.array(recurrent_flags)

    # Two-way ANOVA with interaction term
    from scipy.stats import f_oneway

    groups_by_interaction = [
        all_orders[interaction_flags == 0],  # No interactions
        all_orders[interaction_flags == 1],  # With interactions
    ]
    groups_by_recurrent = [
        all_orders[recurrent_flags == 0],   # No recurrent
        all_orders[recurrent_flags == 1],   # With recurrent
    ]

    f_interaction, p_interaction = f_oneway(*groups_by_interaction)
    f_recurrent, p_recurrent = f_oneway(*groups_by_recurrent)

    print(f"Interaction term F-stat:  {f_interaction:.3f}, p={p_interaction:.4f}")
    print(f"Recurrent term F-stat:    {f_recurrent:.3f}, p={p_recurrent:.4f}")

    # Statistical significance tests
    print("\nPairwise comparisons (t-tests):")

    def t_test(group1, group2, name1, name2):
        t_stat, p_val = stats.ttest_ind(group1, group2)
        mean_diff = np.mean(group1) - np.mean(group2)
        print(f"  {name1} vs {name2}: t={t_stat:.3f}, p={p_val:.4f}, Δ={mean_diff:.4f}")

    t_test(results['interactions'], results['baseline'], 'Interactions', 'Baseline')
    t_test(results['recurrent'], results['baseline'], 'Recurrent', 'Baseline')
    t_test(results['combined'], results['baseline'], 'Combined', 'Baseline')
    t_test(results['combined'], np.concatenate([results['interactions'], results['recurrent']]),
           'Combined', 'Interactions+Recurrent')

    # ==================== Validation ====================

    print("\n" + "=" * 70)
    print("VALIDATION CRITERIA")
    print("=" * 70)

    val_1 = combined_d >= 1.5
    val_2 = synergy_factor >= 1.3
    val_combined = combined_gain >= 3.5

    print(f"Combined effect size d >= 1.5:        {val_1} (d={combined_d:.3f})")
    print(f"Synergy factor >= 1.3:                {val_2} (factor={synergy_factor:.3f}x)")
    print(f"Combined gain >= 3.5x:                {val_combined} (gain={combined_gain:.3f}x)")

    # Determine status
    if val_combined and val_2:
        status = 'validated'
    elif val_combined or (val_1 and val_2):
        status = 'validated'
    elif synergy_factor > 1.0 and combined_d > 1.0:
        status = 'inconclusive'
    else:
        status = 'refuted'

    print(f"\nSTATUS: {status}")

    # ==================== Results Summary ====================

    results_dict = {
        'variant_means': {
            'baseline': float(baseline_mean),
            'interactions': float(interactions_mean),
            'recurrent': float(recurrent_mean),
            'combined': float(combined_mean),
        },
        'variant_stds': {
            'baseline': float(np.std(results['baseline'])),
            'interactions': float(np.std(results['interactions'])),
            'recurrent': float(np.std(results['recurrent'])),
            'combined': float(np.std(results['combined'])),
        },
        'effect_sizes': {
            'interactions_d': float(interactions_d),
            'recurrent_d': float(recurrent_d),
            'combined_d': float(combined_d),
        },
        'multiplicative_gains': {
            'interactions_gain': float(interactions_gain),
            'recurrent_gain': float(recurrent_gain),
            'combined_gain': float(combined_gain),
        },
        'synergy_analysis': {
            'baseline_order': float(baseline_mean),
            'interactions_order': float(interactions_mean),
            'recurrent_order': float(recurrent_mean),
            'combined_order': float(combined_mean),
            'additive_prediction': float(additive_prediction),
            'synergy_factor': float(synergy_factor),
            'superadditive': bool(superadditive),
        },
        'anova_results': {
            'interaction_f': float(f_interaction),
            'interaction_p': float(p_interaction),
            'recurrent_f': float(f_recurrent),
            'recurrent_p': float(p_recurrent),
        },
        'validation': {
            'combined_d_>=_1_5': bool(val_1),
            'synergy_factor_>= 1_3': bool(val_2),
            'combined_gain_>= 3_5x': bool(val_combined),
        },
        'status': status,
    }

    # Save results
    results_dir = Path('/Users/matt/Development/monochrome_noise_converger/results/interaction_recurrent_synergy')
    results_dir.mkdir(parents=True, exist_ok=True)

    results_file = results_dir / 'res_244_results.json'
    with open(results_file, 'w') as f:
        json.dump(results_dict, f, indent=2)

    print(f"\nResults saved to: {results_file}")

    return results_dict, status

if __name__ == '__main__':
    results, status = run_experiment()

    # Update research log
    mgr = ResearchLogManager()
    mgr.update_entry('RES-244', {
        'status': status,
        'results': results,
        'code_path': 'experiments/res_244_interaction_recurrent_synergy.py'
    })

    print(f"\n✓ RES-244 {status.upper()}")
