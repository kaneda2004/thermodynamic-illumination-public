"""
Fast Uniform Prior Sampler

Optimized for exploring uniform prior at high bit depths.
Uses batch rejection sampling with early filtering and optional GPU acceleration.

Speedup strategies:
1. Batch generation (1000+ images at once)
2. Vectorized metric computation
3. Early rejection (cheap metrics first)
4. Optional GPU via CuPy
5. Numba JIT for connected components
"""

import numpy as np
import zlib
from typing import Callable, Optional
import time

# Try to import optional accelerators
try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False
    cp = None

try:
    from numba import jit, prange
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False

print(f"Fast sampler initialized: CuPy={'YES' if HAS_CUPY else 'NO'}, Numba={'YES' if HAS_NUMBA else 'NO'}")


# =============================================================================
# VECTORIZED METRIC COMPUTATION
# =============================================================================

def batch_compute_density(images: np.ndarray) -> np.ndarray:
    """Compute density for batch of images. Shape: (N, H, W) -> (N,)"""
    return images.mean(axis=(1, 2))


def batch_compute_symmetry(images: np.ndarray) -> np.ndarray:
    """Compute symmetry for batch of images. Shape: (N, H, W) -> (N,)"""
    h_sym = (images == np.flip(images, axis=2)).mean(axis=(1, 2))
    v_sym = (images == np.flip(images, axis=1)).mean(axis=(1, 2))
    return (h_sym + v_sym) / 2


def batch_compute_edge_density(images: np.ndarray) -> np.ndarray:
    """Compute edge density for batch of images. Shape: (N, H, W) -> (N,)"""
    # Horizontal edges
    h_edges = (images[:, :, :-1] != images[:, :, 1:]).sum(axis=(1, 2))
    # Vertical edges
    v_edges = (images[:, :-1, :] != images[:, 1:, :]).sum(axis=(1, 2))
    total_possible = 2 * images.shape[1] * images.shape[2]
    return (h_edges + v_edges) / total_possible


def batch_compute_spectral_coherence(images: np.ndarray) -> np.ndarray:
    """Compute spectral coherence for batch. Shape: (N, H, W) -> (N,)"""
    # FFT on batch
    f = np.fft.fft2(images.astype(float) - 0.5, axes=(1, 2))
    f_shifted = np.fft.fftshift(f, axes=(1, 2))
    power = np.abs(f_shifted) ** 2

    n, h, w = images.shape
    cy, cx = h // 2, w // 2
    y, x = np.ogrid[:h, :w]
    r = np.sqrt((x - cx)**2 + (y - cy)**2)
    low_mask = r < (min(h, w) / 4)

    low_power = power[:, low_mask].sum(axis=1)
    total_power = power.sum(axis=(1, 2)) + 1e-10

    return low_power / total_power


def batch_compute_compressibility(images: np.ndarray) -> np.ndarray:
    """Compute compressibility for batch. Must be sequential due to zlib."""
    results = np.zeros(len(images))
    for i, img in enumerate(images):
        tiled = np.tile(img, (2, 2))
        packed = np.packbits(tiled.flatten())
        compressed = zlib.compress(bytes(packed), level=9)
        raw_bits = tiled.size
        compressed_bits = len(compressed) * 8
        results[i] = max(0, 1 - (compressed_bits / raw_bits))
    return results


# Connected components - use Numba if available
if HAS_NUMBA:
    @jit(nopython=True, cache=True)
    def _count_components_single(img):
        """Count connected components in a single image."""
        h, w = img.shape
        visited = np.zeros((h, w), dtype=np.bool_)
        count = 0

        for start_i in range(h):
            for start_j in range(w):
                if not visited[start_i, start_j] and img[start_i, start_j] == 1:
                    # BFS using a simple array as queue
                    queue_i = np.zeros(h * w, dtype=np.int32)
                    queue_j = np.zeros(h * w, dtype=np.int32)
                    queue_i[0] = start_i
                    queue_j[0] = start_j
                    head = 0
                    tail = 1
                    visited[start_i, start_j] = True

                    while head < tail:
                        ci, cj = queue_i[head], queue_j[head]
                        head += 1

                        # Check 4-neighbors
                        for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                            ni, nj = ci + di, cj + dj
                            if 0 <= ni < h and 0 <= nj < w:
                                if not visited[ni, nj] and img[ni, nj] == 1:
                                    visited[ni, nj] = True
                                    queue_i[tail] = ni
                                    queue_j[tail] = nj
                                    tail += 1
                    count += 1
        return count

    @jit(nopython=True, parallel=True, cache=True)
    def batch_compute_components_numba(images):
        """Parallel connected components using Numba."""
        n = len(images)
        results = np.zeros(n, dtype=np.int32)
        for i in prange(n):
            results[i] = _count_components_single(images[i])
        return results

    def batch_compute_components(images: np.ndarray) -> np.ndarray:
        return batch_compute_components_numba(images.astype(np.uint8))
else:
    def batch_compute_components(images: np.ndarray) -> np.ndarray:
        """Fallback: sequential connected components."""
        from core.thermo_sampler_v3 import compute_connected_components
        return np.array([compute_connected_components(img) for img in images])


# =============================================================================
# GPU-ACCELERATED BATCH GENERATION (if CuPy available)
# =============================================================================

def generate_batch_gpu(n_images: int, size: int) -> np.ndarray:
    """Generate batch of random binary images on GPU."""
    if HAS_CUPY:
        random_gpu = cp.random.random((n_images, size, size)) > 0.5
        return cp.asnumpy(random_gpu).astype(np.uint8)
    else:
        return (np.random.random((n_images, size, size)) > 0.5).astype(np.uint8)


def batch_spectral_coherence_gpu(images: np.ndarray) -> np.ndarray:
    """GPU-accelerated spectral coherence."""
    if not HAS_CUPY:
        return batch_compute_spectral_coherence(images)

    images_gpu = cp.asarray(images.astype(np.float32) - 0.5)
    f = cp.fft.fft2(images_gpu, axes=(1, 2))
    f_shifted = cp.fft.fftshift(f, axes=(1, 2))
    power = cp.abs(f_shifted) ** 2

    n, h, w = images.shape
    cy, cx = h // 2, w // 2
    y, x = cp.ogrid[:h, :w]
    r = cp.sqrt((x - cx)**2 + (y - cy)**2)
    low_mask = r < (min(h, w) / 4)

    low_power = power[:, low_mask].sum(axis=1)
    total_power = power.sum(axis=(1, 2)) + 1e-10

    return cp.asnumpy(low_power / total_power)


# =============================================================================
# FAST ORDER METRIC WITH EARLY REJECTION
# =============================================================================

def gaussian_gate(x: np.ndarray, center: float, sigma: float) -> np.ndarray:
    return np.exp(-((x - center) ** 2) / (2 * sigma ** 2))


def batch_order_multiplicative_fast(
    images: np.ndarray,
    threshold: float = 0.0,
    use_gpu: bool = True
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute order metric for batch with early rejection.

    Returns: (orders, mask) where mask indicates which images were fully evaluated.
    Images that fail early gates get order=0 without computing expensive metrics.
    """
    n = len(images)
    orders = np.zeros(n)
    mask = np.ones(n, dtype=bool)

    # Stage 1: Density gate (very fast)
    density = batch_compute_density(images)
    density_gate = gaussian_gate(density, 0.5, 0.25)

    # Early reject if density gate < threshold (can't possibly reach threshold)
    if threshold > 0:
        mask &= (density_gate >= threshold * 0.5)  # Conservative early rejection

    if not mask.any():
        return orders, mask

    # Stage 2: Edge density (fast)
    edge_density = np.zeros(n)
    edge_density[mask] = batch_compute_edge_density(images[mask])
    edge_gate = gaussian_gate(edge_density, 0.15, 0.08)

    # Stage 3: Spectral coherence (medium - uses FFT)
    coherence = np.zeros(n)
    if use_gpu and HAS_CUPY:
        coherence[mask] = batch_spectral_coherence_gpu(images[mask])
    else:
        coherence[mask] = batch_compute_spectral_coherence(images[mask])
    coherence_gate = 1 / (1 + np.exp(-20 * (coherence - 0.3)))

    # Early reject based on first three gates
    base_so_far = density_gate * edge_gate * coherence_gate
    if threshold > 0:
        mask &= (base_so_far >= threshold * 0.3)

    if not mask.any():
        return orders, mask

    # Stage 4: Compressibility (slow - sequential zlib)
    compressibility = np.zeros(n)
    compressibility[mask] = batch_compute_compressibility(images[mask])

    compress_gate = np.zeros(n)
    low_mask = compressibility < 0.2
    mid_mask = (compressibility >= 0.2) & (compressibility < 0.8)
    high_mask = compressibility >= 0.8
    compress_gate[low_mask] = compressibility[low_mask] / 0.2
    compress_gate[mid_mask] = 1.0
    compress_gate[high_mask] = np.maximum(0, 1 - (compressibility[high_mask] - 0.8) / 0.2)

    # Stage 5: Symmetry bonus (fast)
    symmetry = batch_compute_symmetry(images)
    symmetry_bonus = 0.3 * symmetry

    # Stage 6: Connected components (slow if no Numba)
    components = np.zeros(n, dtype=int)
    components[mask] = batch_compute_components(images[mask])

    component_bonus = np.zeros(n)
    comp_low = (components > 0) & (components <= 5)
    comp_high = components > 5
    component_bonus[comp_low] = 0.2 * (components[comp_low] / 5)
    component_bonus[comp_high] = np.maximum(0, 0.2 * (1 - (components[comp_high] - 5) / 20))

    # Final score
    base_score = density_gate * edge_gate * coherence_gate * compress_gate
    orders = np.minimum(1.0, base_score * (1 + symmetry_bonus + component_bonus))

    return orders, mask


# =============================================================================
# FAST NESTED SAMPLING FOR UNIFORM PRIOR
# =============================================================================

def fast_uniform_nested_sampling(
    n_live: int = 50,
    n_iterations: int = 2500,
    image_size: int = 32,
    batch_size: int = 1000,
    use_gpu: bool = True,
    seed: int = None,
    verbose: bool = True,
    progress_callback: Callable = None
) -> tuple[list, list]:
    """
    Fast nested sampling for uniform prior using batch rejection sampling.

    Args:
        n_live: Number of live points
        n_iterations: Number of iterations (determines max bits explored)
        image_size: Size of square images
        batch_size: Number of images to generate per batch
        use_gpu: Whether to use GPU acceleration (if available)
        seed: Random seed
        verbose: Print progress
        progress_callback: Optional callback(iteration, total, threshold, bits)

    Returns:
        (dead_points, live_points)
    """
    if seed is not None:
        np.random.seed(seed)
        if HAS_CUPY:
            cp.random.seed(seed)

    if verbose:
        print(f"\n{'='*60}")
        print(f"FAST UNIFORM NESTED SAMPLING")
        print(f"{'='*60}")
        print(f"  n_live: {n_live}")
        print(f"  n_iterations: {n_iterations}")
        print(f"  batch_size: {batch_size}")
        print(f"  GPU: {'YES' if (use_gpu and HAS_CUPY) else 'NO'}")
        print(f"  Numba: {'YES' if HAS_NUMBA else 'NO'}")
        max_bits = n_iterations / (n_live * np.log(2))
        print(f"  Max bits: {max_bits:.1f}")
        print()

    # Initialize live points
    live_images = generate_batch_gpu(n_live, image_size) if use_gpu else \
                  (np.random.random((n_live, image_size, image_size)) > 0.5).astype(np.uint8)
    live_orders, _ = batch_order_multiplicative_fast(live_images, use_gpu=use_gpu)

    dead_points = []
    log_X = 0.0

    total_samples = 0
    total_batches = 0
    start_time = time.time()

    for iteration in range(n_iterations):
        # Find worst point
        worst_idx = np.argmin(live_orders)
        threshold = live_orders[worst_idx]

        # Record dead point
        dead_points.append({
            'iteration': iteration,
            'order': float(threshold),
            'log_X': log_X,
            'image': live_images[worst_idx].copy()
        })

        # Shrink volume
        log_X -= 1.0 / n_live

        # Batch rejection sampling to find replacement
        found = False
        attempts = 0
        max_attempts = 100  # Max batches before giving up

        while not found and attempts < max_attempts:
            # Generate batch
            batch = generate_batch_gpu(batch_size, image_size) if use_gpu else \
                    (np.random.random((batch_size, image_size, image_size)) > 0.5).astype(np.uint8)

            # Compute orders with early rejection
            batch_orders, _ = batch_order_multiplicative_fast(batch, threshold, use_gpu)

            # Find candidates above threshold
            valid_idx = np.where(batch_orders >= threshold)[0]

            total_samples += batch_size
            total_batches += 1
            attempts += 1

            if len(valid_idx) > 0:
                # Pick random valid sample
                chosen = valid_idx[np.random.randint(len(valid_idx))]
                live_images[worst_idx] = batch[chosen]
                live_orders[worst_idx] = batch_orders[chosen]
                found = True

        if not found:
            # Clone random live point (fallback)
            clone_idx = np.random.randint(n_live)
            while clone_idx == worst_idx:
                clone_idx = np.random.randint(n_live)
            live_images[worst_idx] = live_images[clone_idx].copy()
            live_orders[worst_idx] = live_orders[clone_idx]

        # Progress reporting
        if verbose and (iteration + 1) % 100 == 0:
            bits = -log_X / np.log(2)
            elapsed = time.time() - start_time
            rate = (iteration + 1) / elapsed
            eta = (n_iterations - iteration - 1) / rate
            samples_per_iter = total_samples / (iteration + 1)
            print(f"  Iter {iteration+1}/{n_iterations}: τ={threshold:.4f}, "
                  f"bits={bits:.1f}, {samples_per_iter:.0f} samples/iter, "
                  f"ETA: {eta/60:.1f}min")

        if progress_callback:
            progress_callback(iteration, n_iterations, threshold, -log_X / np.log(2))

    elapsed = time.time() - start_time
    if verbose:
        print(f"\nCompleted in {elapsed/60:.1f} minutes")
        print(f"Total samples: {total_samples:,}")
        print(f"Avg samples/iteration: {total_samples/n_iterations:.0f}")

    # Convert live points to list format
    live_points = [
        {'image': live_images[i], 'order': float(live_orders[i])}
        for i in range(n_live)
    ]

    return dead_points, live_points


# =============================================================================
# BENCHMARK
# =============================================================================

def benchmark_batch_size(image_size: int = 32, n_batches: int = 10):
    """Find optimal batch size for current hardware."""
    print("\nBenchmarking batch sizes...")

    for batch_size in [100, 500, 1000, 2000, 5000, 10000]:
        start = time.time()
        for _ in range(n_batches):
            images = generate_batch_gpu(batch_size, image_size)
            orders, _ = batch_order_multiplicative_fast(images, threshold=0.0, use_gpu=True)
        elapsed = time.time() - start

        images_per_sec = (batch_size * n_batches) / elapsed
        print(f"  Batch {batch_size:5d}: {images_per_sec:,.0f} images/sec")


# =============================================================================
# PARALLEL MULTI-SEED RUNNER
# =============================================================================

def _run_single_seed(args):
    """Worker function for parallel execution."""
    seed, n_live, n_iterations, image_size, batch_size = args
    dead, live = fast_uniform_nested_sampling(
        n_live=n_live,
        n_iterations=n_iterations,
        image_size=image_size,
        batch_size=batch_size,
        use_gpu=False,  # GPU not safe across processes
        seed=seed,
        verbose=False
    )
    return {
        'seed': seed,
        'dead': dead,
        'live': live,
        'final_order': np.mean([p['order'] for p in live]),
        'final_bits': -dead[-1]['log_X'] / np.log(2)
    }


def parallel_uniform_experiment(
    n_runs: int = 10,
    n_live: int = 50,
    n_iterations: int = 2500,
    image_size: int = 32,
    batch_size: int = 2000,
    base_seed: int = 42,
    n_workers: int = None,
    verbose: bool = True
) -> list:
    """
    Run multiple uniform prior experiments in parallel.

    Args:
        n_runs: Number of independent runs
        n_live: Live points per run
        n_iterations: Iterations per run
        image_size: Image size
        batch_size: Batch size for rejection sampling
        base_seed: Starting seed (runs use base_seed + i)
        n_workers: Number of parallel workers (default: CPU count)
        verbose: Print progress

    Returns:
        List of result dicts with 'seed', 'dead', 'live', 'final_order', 'final_bits'
    """
    import multiprocessing as mp

    if n_workers is None:
        n_workers = mp.cpu_count()

    seeds = [base_seed + i for i in range(n_runs)]
    args_list = [(seed, n_live, n_iterations, image_size, batch_size) for seed in seeds]

    max_bits = n_iterations / (n_live * np.log(2))

    if verbose:
        print(f"\n{'='*60}")
        print(f"PARALLEL UNIFORM EXPERIMENT")
        print(f"{'='*60}")
        print(f"  Runs: {n_runs}")
        print(f"  Workers: {n_workers}")
        print(f"  Iterations/run: {n_iterations}")
        print(f"  Max bits: {max_bits:.1f}")
        print(f"  Batch size: {batch_size}")
        print()

    start_time = time.time()

    # Use spawn to avoid issues with Numba
    ctx = mp.get_context('spawn')
    with ctx.Pool(n_workers) as pool:
        if verbose:
            print("Running experiments in parallel...")
            results = []
            for i, result in enumerate(pool.imap_unordered(_run_single_seed, args_list)):
                results.append(result)
                elapsed = time.time() - start_time
                print(f"  Completed {i+1}/{n_runs} (seed={result['seed']}, "
                      f"order={result['final_order']:.4f}, bits={result['final_bits']:.1f}) "
                      f"[{elapsed/60:.1f}min elapsed]")
        else:
            results = pool.map(_run_single_seed, args_list)

    total_time = time.time() - start_time

    if verbose:
        print(f"\n{'='*60}")
        print(f"COMPLETED in {total_time/60:.1f} minutes")
        print(f"{'='*60}")

        orders = [r['final_order'] for r in results]
        bits = [r['final_bits'] for r in results]
        print(f"  Final order: {np.mean(orders):.4f} ± {np.std(orders):.4f}")
        print(f"  Final bits:  {np.mean(bits):.1f} ± {np.std(bits):.1f}")

    return results


def save_results_csv(results: list, output_path: str):
    """Save experiment results to CSV."""
    import csv

    # Aggregate all dead points
    rows = []
    for r in results:
        for d in r['dead']:
            rows.append({
                'seed': r['seed'],
                'iteration': d['iteration'],
                'order': d['order'],
                'log_X': d['log_X'],
                'bits': -d['log_X'] / np.log(2)
            })

    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['seed', 'iteration', 'order', 'log_X', 'bits'])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved {len(rows)} rows to {output_path}")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Fast uniform prior sampler')
    parser.add_argument('--benchmark', action='store_true', help='Run benchmark')
    parser.add_argument('--test', action='store_true', help='Run short test')
    parser.add_argument('--full', action='store_true', help='Run full experiment')
    parser.add_argument('--iterations', type=int, default=6000, help='Iterations for full run')
    parser.add_argument('--runs', type=int, default=10, help='Number of runs')
    parser.add_argument('--workers', type=int, default=None, help='Number of workers')
    parser.add_argument('--output', type=str, default='uniform_extended.csv', help='Output CSV path')
    args = parser.parse_args()

    if args.benchmark:
        print("\n" + "="*60)
        print("BENCHMARKING")
        print("="*60)
        benchmark_batch_size()

    if args.test:
        print("\n" + "="*60)
        print("SHORT TEST (500 iterations, 3 runs)")
        print("="*60)
        results = parallel_uniform_experiment(
            n_runs=3,
            n_iterations=500,
            batch_size=2000,
            verbose=True
        )

    if args.full:
        print("\n" + "="*60)
        print(f"FULL EXPERIMENT ({args.iterations} iterations, {args.runs} runs)")
        print("="*60)
        results = parallel_uniform_experiment(
            n_runs=args.runs,
            n_iterations=args.iterations,
            batch_size=2000,
            n_workers=args.workers,
            verbose=True
        )
        save_results_csv(results, args.output)
