"""
RES-113: Topological Analysis of CPPN Images

Hypothesis: High-order CPPN images have simpler topology (fewer connected
components and Betti numbers) than low-order images.

Uses persistent homology-inspired analysis:
- Betti0 = number of connected components (foreground)
- Betti1 = number of holes (1-cycles)
"""

import numpy as np
import sys
sys.path.insert(0, '/Users/matt/Development/monochrome_noise_converger')

from core.thermo_sampler_v3 import CPPN, order_multiplicative, set_global_seed
from scipy import stats


def count_connected_components(img: np.ndarray, value: int = 1) -> int:
    """Count connected components for pixels with given value."""
    mask = (img == value)
    visited = np.zeros_like(mask, dtype=bool)
    count = 0

    def flood_fill(i, j):
        stack = [(i, j)]
        while stack:
            ci, cj = stack.pop()
            if ci < 0 or ci >= mask.shape[0] or cj < 0 or cj >= mask.shape[1]:
                continue
            if visited[ci, cj] or not mask[ci, cj]:
                continue
            visited[ci, cj] = True
            stack.extend([(ci+1, cj), (ci-1, cj), (ci, cj+1), (ci, cj-1)])

    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            if not visited[i, j] and mask[i, j]:
                flood_fill(i, j)
                count += 1
    return count


def compute_euler_characteristic(img: np.ndarray) -> int:
    """
    Compute Euler characteristic using pixel-based formula.
    chi = V - E + F for the image considered as a cell complex.

    For binary images: chi = #foreground_components - #holes
    Using: chi = B0 - B1
    """
    # Count vertices, edges, faces for foreground pixels
    h, w = img.shape

    # Vertices: corners of pixels (foreground pixel contributes 4 corners)
    # Edges: boundaries of pixels
    # Faces: foreground pixels themselves

    # Simpler: use Euler formula for 4-connected pixels
    # Each foreground pixel = 1 face
    # Edges between adjacent foreground pixels
    # Vertices at corners

    vertices = 0
    edges = 0
    faces = np.sum(img)

    # Count unique vertices and edges in the foreground
    for i in range(h):
        for j in range(w):
            if img[i, j] == 1:
                # Count edges (shared boundaries)
                # Right edge
                if j < w - 1 and img[i, j+1] == 1:
                    edges += 1
                # Down edge
                if i < h - 1 and img[i+1, j] == 1:
                    edges += 1

    # For vertices, count corners
    # A vertex is counted for each foreground pixel it touches
    vertex_count = np.zeros((h+1, w+1), dtype=int)
    for i in range(h):
        for j in range(w):
            if img[i, j] == 1:
                vertex_count[i, j] += 1
                vertex_count[i, j+1] += 1
                vertex_count[i+1, j] += 1
                vertex_count[i+1, j+1] += 1
    vertices = np.sum(vertex_count > 0)

    return vertices - edges + faces


def count_holes(img: np.ndarray) -> int:
    """
    Count holes (Betti1) in binary image.

    A hole is a connected component of background that is completely
    surrounded by foreground (not touching the border).
    """
    h, w = img.shape
    background = (img == 0)

    # Mark background connected to border
    border_connected = np.zeros_like(background, dtype=bool)

    def flood_fill_bg(i, j):
        stack = [(i, j)]
        while stack:
            ci, cj = stack.pop()
            if ci < 0 or ci >= h or cj < 0 or cj >= w:
                continue
            if border_connected[ci, cj] or not background[ci, cj]:
                continue
            border_connected[ci, cj] = True
            stack.extend([(ci+1, cj), (ci-1, cj), (ci, cj+1), (ci, cj-1)])

    # Start from all border pixels
    for i in range(h):
        if background[i, 0]:
            flood_fill_bg(i, 0)
        if background[i, w-1]:
            flood_fill_bg(i, w-1)
    for j in range(w):
        if background[0, j]:
            flood_fill_bg(0, j)
        if background[h-1, j]:
            flood_fill_bg(h-1, j)

    # Holes are background components not connected to border
    interior_background = background & ~border_connected

    # Count connected components of interior background
    return count_connected_components(interior_background.astype(int), value=1)


def compute_topological_features(img: np.ndarray) -> dict:
    """Compute all topological features of a binary image."""
    betti0_fg = count_connected_components(img, value=1)  # Foreground components
    betti0_bg = count_connected_components(img, value=0)  # Background components
    betti1 = count_holes(img)  # Holes in foreground
    euler = betti0_fg - betti1  # Euler characteristic

    return {
        'betti0_foreground': betti0_fg,
        'betti0_background': betti0_bg,
        'betti1_holes': betti1,
        'euler_characteristic': euler,
        'total_components': betti0_fg + betti0_bg,
        'topology_complexity': betti0_fg + betti1  # B0 + B1
    }


def main():
    set_global_seed(42)

    n_samples = 500

    print("Generating CPPN samples and computing topological features...")

    results = []
    for i in range(n_samples):
        cppn = CPPN()
        img = cppn.render(size=32)
        order = order_multiplicative(img)
        topo = compute_topological_features(img)

        results.append({
            'order': order,
            **topo
        })

        if (i + 1) % 100 == 0:
            print(f"  Processed {i+1}/{n_samples}")

    # Convert to arrays
    orders = np.array([r['order'] for r in results])
    betti0 = np.array([r['betti0_foreground'] for r in results])
    betti1 = np.array([r['betti1_holes'] for r in results])
    complexity = np.array([r['topology_complexity'] for r in results])

    # Split into high/low order groups (median split)
    median_order = np.median(orders)
    high_mask = orders > median_order
    low_mask = orders <= median_order

    print(f"\n=== Results ===")
    print(f"Order range: {orders.min():.4f} to {orders.max():.4f}")
    print(f"Median order: {median_order:.4f}")
    print(f"High-order samples: {high_mask.sum()}, Low-order: {low_mask.sum()}")

    # Compare topological features between groups
    metrics = [
        ('Betti0 (foreground components)', betti0),
        ('Betti1 (holes)', betti1),
        ('Topology complexity (B0+B1)', complexity),
    ]

    print(f"\n=== Topological Feature Comparison ===")

    all_results = {}
    for name, values in metrics:
        high_vals = values[high_mask]
        low_vals = values[low_mask]

        # Mann-Whitney U test (non-parametric)
        stat, p_value = stats.mannwhitneyu(high_vals, low_vals, alternative='less')

        # Effect size (Cohen's d)
        pooled_std = np.sqrt((np.var(high_vals) + np.var(low_vals)) / 2)
        if pooled_std > 0:
            cohens_d = (np.mean(low_vals) - np.mean(high_vals)) / pooled_std
        else:
            cohens_d = 0

        print(f"\n{name}:")
        print(f"  High-order: mean={np.mean(high_vals):.2f}, std={np.std(high_vals):.2f}")
        print(f"  Low-order:  mean={np.mean(low_vals):.2f}, std={np.std(low_vals):.2f}")
        print(f"  Mann-Whitney U p-value (high < low): {p_value:.6f}")
        print(f"  Cohen's d: {cohens_d:.3f}")

        all_results[name] = {
            'high_mean': np.mean(high_vals),
            'low_mean': np.mean(low_vals),
            'p_value': p_value,
            'cohens_d': cohens_d
        }

    # Correlation analysis
    print(f"\n=== Correlation with Order ===")

    for name, values in metrics:
        r, p = stats.pearsonr(orders, values)
        print(f"{name}: r={r:.4f}, p={p:.6f}")

    # Final verdict
    print(f"\n=== VERDICT ===")
    complexity_result = all_results['Topology complexity (B0+B1)']

    validated = (complexity_result['p_value'] < 0.01 and
                 complexity_result['cohens_d'] > 0.5)

    if validated:
        print("VALIDATED: High-order CPPN images have simpler topology")
    else:
        print(f"NOT VALIDATED: p={complexity_result['p_value']:.6f}, d={complexity_result['cohens_d']:.3f}")
        print("Requirements: p < 0.01, Cohen's d > 0.5")

    return validated, complexity_result


if __name__ == '__main__':
    validated, result = main()
