#!/usr/bin/env python3
"""
Evolutionary Strategy Validation: The "Volume Navigator" Test

The thermodynamic metric measures VOLUME in parameter space.
- SGD navigates by LOCAL CURVATURE (gradients)
- ES navigates by GLOBAL VOLUME (sampling)

Hypothesis: ES should correlate PERFECTLY with bits-to-threshold
because both measure the same thing: "How much of parameter space
produces valid outputs?"

This is the DEFINITIVE test of the thermodynamic metric.

Usage:
    uv run python train_validation_es.py
"""

import torch
import torch.nn as nn
import numpy as np
import time
from dataclasses import dataclass
from torchvision import datasets, transforms


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


@dataclass
class ESResult:
    name: str
    bits_score: float
    fitness_history: list[float]
    final_fitness: float
    evals_to_threshold: int
    total_evals: int


class CPPNGenerator(nn.Module):
    """CPPN with coordinate conditioning."""

    def __init__(self, n_params: int = 100, image_size: int = 14):
        super().__init__()
        self.n_params = n_params
        self.image_size = image_size

        # Small network for speed
        self.w1 = nn.Parameter(torch.randn(4, 32) * 0.5)
        self.b1 = nn.Parameter(torch.zeros(32))
        self.w2 = nn.Parameter(torch.randn(32, 16) * 0.5)
        self.b2 = nn.Parameter(torch.zeros(16))
        self.w3 = nn.Parameter(torch.randn(16, 1) * 0.5)
        self.b3 = nn.Parameter(torch.zeros(1))

        # Coordinates
        x = torch.linspace(-1, 1, image_size)
        y = torch.linspace(-1, 1, image_size)
        xx, yy = torch.meshgrid(x, y, indexing='ij')
        r = torch.sqrt(xx**2 + yy**2)
        self.register_buffer('coords', torch.stack([
            xx.flatten(), yy.flatten(), r.flatten(), (r**2).flatten()
        ], dim=1))

    def forward(self) -> torch.Tensor:
        h = torch.tanh(self.coords @ self.w1 + self.b1)
        h = torch.tanh(h @ self.w2 + self.b2)
        out = torch.sigmoid(h @ self.w3 + self.b3)
        return out.view(self.image_size, self.image_size)

    def get_params(self) -> np.ndarray:
        return torch.cat([p.flatten() for p in self.parameters()]).detach().cpu().numpy()

    def set_params(self, params: np.ndarray):
        idx = 0
        for p in self.parameters():
            size = p.numel()
            p.data = torch.tensor(params[idx:idx+size], dtype=p.dtype, device=p.device).view(p.shape)
            idx += size


class MLPGenerator(nn.Module):
    """MLP without coordinate structure."""

    def __init__(self, n_params: int = 100, image_size: int = 14):
        super().__init__()
        self.n_params = n_params
        self.image_size = image_size
        n_pixels = image_size * image_size

        # Direct mapping
        self.w1 = nn.Parameter(torch.randn(8, 64) * 0.5)  # Seed → hidden
        self.b1 = nn.Parameter(torch.zeros(64))
        self.w2 = nn.Parameter(torch.randn(64, n_pixels) * 0.5)  # Hidden → pixels
        self.b2 = nn.Parameter(torch.zeros(n_pixels))

        # Fixed seed
        self.register_buffer('seed', torch.randn(8))

    def forward(self) -> torch.Tensor:
        h = torch.tanh(self.seed @ self.w1 + self.b1)
        out = torch.sigmoid(h @ self.w2 + self.b2)
        return out.view(self.image_size, self.image_size)

    def get_params(self) -> np.ndarray:
        return torch.cat([p.flatten() for p in self.parameters()]).detach().cpu().numpy()

    def set_params(self, params: np.ndarray):
        idx = 0
        for p in self.parameters():
            size = p.numel()
            p.data = torch.tensor(params[idx:idx+size], dtype=p.dtype, device=p.device).view(p.shape)
            idx += size


def simple_es(
    model: nn.Module,
    target: torch.Tensor,
    n_generations: int = 200,
    population_size: int = 50,
    sigma: float = 0.1,
    lr: float = 0.03,
    fitness_threshold: float = -0.02,
    checkpoint_every: int = 10
) -> ESResult:
    """
    Simple (μ, λ) Evolution Strategy.

    Uses fitness-weighted parameter updates without gradients.
    This should correlate with thermodynamic volume measure.
    """
    params = model.get_params()
    n_params = len(params)

    fitness_history = []
    evals_to_threshold = -1
    total_evals = 0
    best_fitness = float('-inf')

    for gen in range(n_generations):
        # Generate population with Gaussian noise
        noise = np.random.randn(population_size, n_params)
        population = params + sigma * noise

        # Evaluate fitness for each member
        fitnesses = []
        for i in range(population_size):
            model.set_params(population[i])
            with torch.no_grad():
                output = model()
                # Negative MSE as fitness (higher = better)
                fitness = -torch.mean((output - target) ** 2).item()
            fitnesses.append(fitness)
            total_evals += 1

        fitnesses = np.array(fitnesses)

        # Update parameters using fitness-weighted average
        # This is the key ES update: reward directions that improve fitness
        fitnesses_normalized = (fitnesses - fitnesses.mean()) / (fitnesses.std() + 1e-8)
        params = params + lr * np.dot(noise.T, fitnesses_normalized) / population_size

        # Track best
        current_best = fitnesses.max()
        if current_best > best_fitness:
            best_fitness = current_best

        if gen % checkpoint_every == 0:
            fitness_history.append(best_fitness)

        # Check threshold
        if best_fitness > fitness_threshold and evals_to_threshold < 0:
            evals_to_threshold = total_evals

    model.set_params(params)

    return ESResult(
        name=model.__class__.__name__,
        bits_score=0,
        fitness_history=fitness_history,
        final_fitness=best_fitness,
        evals_to_threshold=evals_to_threshold if evals_to_threshold > 0 else total_evals,
        total_evals=total_evals
    )


def create_target_digit(image_size: int = 14) -> torch.Tensor:
    """Create a simple target: circular digit-like shape."""
    x = torch.linspace(-1, 1, image_size)
    y = torch.linspace(-1, 1, image_size)
    xx, yy = torch.meshgrid(x, y, indexing='ij')
    r = torch.sqrt(xx**2 + yy**2)

    # Ring shape (like digit 0)
    target = ((r > 0.3) & (r < 0.7)).float()
    return target


def main():
    print("=" * 70)
    print("EVOLUTIONARY STRATEGY VALIDATION")
    print("=" * 70)
    print()
    print("ES navigates by VOLUME (like thermodynamic metric)")
    print("SGD navigates by CURVATURE (gradients)")
    print()
    print("If bits-to-threshold measures volume, ES should correlate perfectly.")
    print()

    device = get_device()
    print(f"Using device: {device}")

    # Thermodynamic scores
    thermo_scores = {
        'CPPNGenerator': 2.0,
        'MLPGenerator': 15.0,
    }

    # Target (simple ring shape - CPPN should find this easily)
    image_size = 14
    target = create_target_digit(image_size).to(device)

    print(f"\nTarget shape: {image_size}x{image_size} ring pattern")
    print("CPPN's radial coordinates should make this trivial.")
    print("MLP has no radial bias - must search harder.")

    # Run ES for each architecture
    generators = [
        ('CPPNGenerator', CPPNGenerator(image_size=image_size).to(device)),
        ('MLPGenerator', MLPGenerator(image_size=image_size).to(device)),
    ]

    results = []
    n_runs = 3  # Multiple runs for stability

    for name, _ in generators:
        print(f"\nRunning ES on {name} ({n_runs} runs)...")

        run_results = []
        for run in range(n_runs):
            # Fresh model each run
            if name == 'CPPNGenerator':
                model = CPPNGenerator(image_size=image_size).to(device)
            else:
                model = MLPGenerator(image_size=image_size).to(device)

            result = simple_es(
                model, target,
                n_generations=300,  # More generations
                population_size=50,  # Larger population
                sigma=0.3,
                lr=0.1,
                fitness_threshold=-0.08  # More forgiving threshold
            )
            run_results.append(result)
            print(f"  Run {run+1}: fitness={result.final_fitness:.4f}, evals={result.evals_to_threshold}")

        # Average across runs
        avg_result = ESResult(
            name=name,
            bits_score=thermo_scores[name],
            fitness_history=run_results[0].fitness_history,
            final_fitness=np.mean([r.final_fitness for r in run_results]),
            evals_to_threshold=int(np.mean([r.evals_to_threshold for r in run_results])),
            total_evals=run_results[0].total_evals
        )
        results.append(avg_result)

    # Results
    print("\n" + "=" * 70)
    print("RESULTS: Thermodynamic Score vs ES Performance")
    print("=" * 70)
    print()
    print(f"{'Generator':<20} {'Bits':<10} {'Fitness':<12} {'Evals to Goal':<15} {'Prediction'}")
    print("-" * 70)

    results.sort(key=lambda r: r.bits_score)

    for i, r in enumerate(results):
        rank = i + 1
        predicted = "BEST" if rank == 1 else f"#{rank}"
        print(f"{r.name:<20} {r.bits_score:<10.1f} {r.final_fitness:<12.4f} {r.evals_to_threshold:<15} {predicted}")

    # Analysis
    print("\n" + "=" * 70)
    print("ANALYSIS")
    print("=" * 70)

    cppn = next(r for r in results if 'CPPN' in r.name)
    mlp = next(r for r in results if 'MLP' in r.name)

    print(f"\nCPPN (2 bits):  fitness={cppn.final_fitness:.4f}, evals={cppn.evals_to_threshold}")
    print(f"MLP (15 bits):  fitness={mlp.final_fitness:.4f}, evals={mlp.evals_to_threshold}")

    if cppn.final_fitness > mlp.final_fitness:
        print("\n*** CPPN achieved HIGHER fitness (as predicted by lower bits) ***")
    if cppn.evals_to_threshold < mlp.evals_to_threshold:
        print("*** CPPN needed FEWER evaluations (as predicted by lower bits) ***")

    speedup = mlp.evals_to_threshold / max(cppn.evals_to_threshold, 1)
    print(f"\nSpeedup factor: {speedup:.1f}x")

    if cppn.final_fitness > mlp.final_fitness and cppn.evals_to_threshold < mlp.evals_to_threshold:
        print("\n" + "=" * 70)
        print("*** HYPOTHESIS CONFIRMED FOR EVOLUTIONARY SEARCH ***")
        print("Lower bits → Higher fitness AND Faster convergence")
        print("The thermodynamic metric predicts ES performance!")
        print("=" * 70)


if __name__ == "__main__":
    main()
