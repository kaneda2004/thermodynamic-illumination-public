"""
Thermodynamic Illumination v3: Combined Best Practices

Combines:
1. Multiplicative (gated) order metric
2. Elliptical slice sampling (ESS)
3. Diversity-aware replacement
4. Multi-metric tracking for phase transitions
5. Prior comparison framework (ready for extension)
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Callable
import zlib
import random
import os
import math
import csv

# ============================================================================
# PRIOR DEFINITION (must be at top for use in CPPN initialization)
# ============================================================================

PRIOR_SIGMA = 1.0


def set_global_seed(seed: Optional[int]) -> None:
    """Set numpy + python RNG seeds for reproducible runs."""
    if seed is None:
        return
    np.random.seed(seed)
    random.seed(seed)

# ============================================================================
# CPPN (same as v2 - evaluation order fixed)
# ============================================================================

ACTIVATIONS = {
    'sin': lambda x: np.sin(x * np.pi),
    'cos': lambda x: np.cos(x * np.pi),
    'gauss': lambda x: np.exp(-x**2 * 2),
    'sigmoid': lambda x: 1 / (1 + np.exp(-np.clip(x, -10, 10))),
    'tanh': np.tanh,
    'abs': np.abs,
    'relu': lambda x: np.maximum(0, x),
    'identity': lambda x: x,
    'sin2': lambda x: np.sin(x * np.pi * 2),
    'ring': lambda x: np.sin(np.abs(x) * np.pi * 3),
}

@dataclass
class Node:
    id: int
    activation: str
    bias: float = 0.0  # Input nodes get 0; non-input nodes set in CPPN.__post_init__

@dataclass
class Connection:
    from_id: int
    to_id: int
    weight: float
    enabled: bool = True

@dataclass
class CPPN:
    nodes: list = field(default_factory=list)
    connections: list = field(default_factory=list)
    input_ids: list = field(default_factory=lambda: [0, 1, 2, 3])
    output_id: int = 4

    def __post_init__(self):
        if not self.nodes:
            # Input nodes have bias=0 (their values come from coordinates)
            # Non-input nodes get biases sampled from the prior
            self.nodes = [
                Node(0, 'identity', 0.0), Node(1, 'identity', 0.0),
                Node(2, 'identity', 0.0), Node(3, 'identity', 0.0),
                Node(4, 'sigmoid', np.random.randn() * PRIOR_SIGMA),  # Output node bias from prior
            ]
            for inp in self.input_ids:
                # Weights sampled from prior
                self.connections.append(Connection(inp, self.output_id, np.random.randn() * PRIOR_SIGMA))

    def _get_eval_order(self) -> list[int]:
        hidden_ids = sorted([n.id for n in self.nodes
                            if n.id not in self.input_ids and n.id != self.output_id])
        return hidden_ids + [self.output_id]

    def activate(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        r = np.sqrt(x**2 + y**2)
        bias = np.ones_like(x)
        values = {0: x, 1: y, 2: r, 3: bias}
        for nid in self._get_eval_order():
            node = next(n for n in self.nodes if n.id == nid)
            total = np.zeros_like(x) + node.bias
            for conn in self.connections:
                if conn.to_id == nid and conn.enabled and conn.from_id in values:
                    total += values[conn.from_id] * conn.weight
            values[nid] = ACTIVATIONS[node.activation](total)
        return values[self.output_id]

    def render(self, size: int = 32) -> np.ndarray:
        coords = np.linspace(-1, 1, size)
        x, y = np.meshgrid(coords, coords)
        return (self.activate(x, y) > 0.5).astype(np.uint8)

    def copy(self) -> 'CPPN':
        return CPPN(
            nodes=[Node(n.id, n.activation, n.bias) for n in self.nodes],
            connections=[Connection(c.from_id, c.to_id, c.weight, c.enabled) for c in self.connections],
            input_ids=self.input_ids.copy(),
            output_id=self.output_id
        )

    def get_weights(self) -> np.ndarray:
        weights = [c.weight for c in self.connections if c.enabled]
        biases = [n.bias for n in self.nodes if n.id not in self.input_ids]
        return np.array(weights + biases)

    def set_weights(self, w: np.ndarray):
        idx = 0
        for c in self.connections:
            if c.enabled:
                c.weight = w[idx]
                idx += 1
        for n in self.nodes:
            if n.id not in self.input_ids:
                n.bias = w[idx]
                idx += 1


# ============================================================================
# FEATURE COMPUTATIONS (shared by all metrics)
# ============================================================================

def compute_compressibility(img: np.ndarray) -> float:
    """
    Bit-level compressibility with tiling to overcome zlib header overhead.

    At 32x32, zlib header (~10 bytes) dominates the 128-byte payload,
    making random noise compress to ~0. Tiling 2x2 makes the data payload
    larger than the header, giving meaningful compressibility values.
    """
    # Tile 2x2 to overcome header overhead
    tiled = np.tile(img, (2, 2))
    packed = np.packbits(tiled.flatten())
    compressed = zlib.compress(bytes(packed), level=9)
    raw_bits = tiled.size
    compressed_bits = len(compressed) * 8
    return max(0, 1 - (compressed_bits / raw_bits))


def compute_edge_density(img: np.ndarray) -> float:
    """Fraction of pixels adjacent to different-valued pixel."""
    padded = np.pad(img, 1, mode='edge')
    edges = 0
    for di, dj in [(0, 1), (1, 0)]:
        shifted = padded[1+di:1+di+img.shape[0], 1+dj:1+dj+img.shape[1]]
        edges += np.sum(img != shifted)
    return edges / (2 * img.size)


def compute_spectral_coherence(img: np.ndarray) -> float:
    """
    FFT-based coherence: ratio of low-frequency power to total power.

    This is more robust than downsampling ratio for distinguishing noise
    from structure. Noise has flat spectrum (~0.1 coherence), while
    coherent structure has concentrated low-frequency power (>0.5).
    """
    f = np.fft.fft2(img.astype(float) - 0.5)
    f_shifted = np.fft.fftshift(f)
    power = np.abs(f_shifted) ** 2

    h, w = img.shape
    cy, cx = h // 2, w // 2
    y, x = np.ogrid[:h, :w]
    r = np.sqrt((x - cx)**2 + (y - cy)**2)

    # Low frequency = inner quarter of radius
    low_mask = r < (min(h, w) / 4)
    low_power = np.sum(power[low_mask])
    total_power = np.sum(power) + 1e-10

    return low_power / total_power


def compute_symmetry(img: np.ndarray) -> float:
    """Average of horizontal and vertical reflection symmetry."""
    h_sym = np.mean(img == np.fliplr(img))
    v_sym = np.mean(img == np.flipud(img))
    return (h_sym + v_sym) / 2


def compute_connected_components(img: np.ndarray) -> int:
    """Count foreground connected components."""
    visited = np.zeros_like(img, dtype=bool)
    count = 0
    def flood_fill(i, j):
        stack = [(i, j)]
        while stack:
            ci, cj = stack.pop()
            if ci < 0 or ci >= img.shape[0] or cj < 0 or cj >= img.shape[1]:
                continue
            if visited[ci, cj] or img[ci, cj] != 1:
                continue
            visited[ci, cj] = True
            stack.extend([(ci+1, cj), (ci-1, cj), (ci, cj+1), (ci, cj-1)])
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if not visited[i, j] and img[i, j] == 1:
                flood_fill(i, j)
                count += 1
    return count


# ============================================================================
# MULTIPLICATIVE (GATED) ORDER METRIC
# ============================================================================

def gaussian_gate(x: float, center: float, sigma: float) -> float:
    """Bell curve gate: peaks at center, falls off with sigma."""
    return np.exp(-((x - center) ** 2) / (2 * sigma ** 2))


def order_multiplicative(img: np.ndarray) -> float:
    """
    MULTIPLICATIVE order metric.

    Logic: Image must satisfy ALL of:
    - Non-trivial density (not empty/full)
    - Some edges (not solid block)
    - Spectral coherence (not noise - uses FFT low-frequency dominance)
    - Reasonable compressibility (structured, not trivial)

    If ANY gate is ~0, the whole score collapses.
    """
    density = np.mean(img)
    compressibility = compute_compressibility(img)
    edge_density = compute_edge_density(img)
    coherence = compute_spectral_coherence(img)  # Use FFT-based coherence
    symmetry = compute_symmetry(img)
    components = compute_connected_components(img)

    # Gate 1: Density (bell curve centered at 0.5, sigma=0.25)
    density_gate = gaussian_gate(density, center=0.5, sigma=0.25)

    # Gate 2: Edge density (bell curve centered at 0.15, sigma=0.08)
    edge_gate = gaussian_gate(edge_density, center=0.15, sigma=0.08)

    # Gate 3: Coherence (spectral, sigmoid centered at 0.3)
    # Noise ~0.1, structure >0.5. Steeper sigmoid (20x) for cleaner separation.
    coherence_gate = 1 / (1 + np.exp(-20 * (coherence - 0.3)))

    # Gate 4: Compressibility (tiled, so values are more meaningful)
    # With tiling: noise ~0.05, structure ~0.3-0.7, solids ~0.9
    if compressibility < 0.2:
        compress_gate = compressibility / 0.2
    elif compressibility < 0.8:
        compress_gate = 1.0
    else:
        compress_gate = max(0, 1 - (compressibility - 0.8) / 0.2)

    # Bonus: Symmetry (additive bonus, not a gate)
    symmetry_bonus = 0.3 * symmetry

    # Bonus: Component count (prefer 1-5 components)
    if components == 0:
        component_bonus = 0
    elif components <= 5:
        component_bonus = 0.2 * (components / 5)
    else:
        component_bonus = max(0, 0.2 * (1 - (components - 5) / 20))

    # Final score: product of gates, plus bonuses scaled by base
    base_score = density_gate * edge_gate * coherence_gate * compress_gate
    raw_score = base_score * (1 + symmetry_bonus + component_bonus)
    # Clamp to [0, 1] to match documented semantics
    return min(1.0, raw_score)


def order_multiplicative_v2(img: np.ndarray, resolution_ref: int = 32) -> float:
    """
    SCALE-INVARIANT multiplicative order metric.

    Normalizes gate centers to match reference resolution.
    Ensures same CPPN produces same order across resolutions.

    For smooth CPPN outputs: edge_density ~ 1/resolution (perimeter/area ratio)
    We scale gate centers to compensate: center(N) = center_32 * (32/N)

    Args:
        img: Binary image (H×W)
        resolution_ref: Reference resolution for gate calibration (default 32)

    Returns:
        Order score ∈ [0, 1]
    """
    resolution = img.shape[0]
    scale_factor = resolution_ref / resolution

    # Compute features (unchanged)
    density = np.mean(img)
    compressibility = compute_compressibility(img)
    edge_density = compute_edge_density(img)
    coherence = compute_spectral_coherence(img)
    symmetry = compute_symmetry(img)
    components = compute_connected_components(img)

    # Gate 1: Density (unchanged, already scale-invariant)
    density_gate = gaussian_gate(density, center=0.5, sigma=0.25)

    # Gate 2: Edge density (ADAPTIVE - scale with resolution)
    edge_gate = gaussian_gate(
        edge_density,
        center=0.15 * scale_factor,  # ADAPTIVE
        sigma=0.08 * scale_factor    # ADAPTIVE
    )

    # Gate 3: Coherence (unchanged, scale-invariant)
    coherence_gate = 1 / (1 + np.exp(-20 * (coherence - 0.3)))

    # Gate 4: Compressibility (unchanged)
    if compressibility < 0.2:
        compress_gate = compressibility / 0.2
    elif compressibility < 0.8:
        compress_gate = 1.0
    else:
        compress_gate = max(0, 1 - (compressibility - 0.8) / 0.2)

    # Bonuses (unchanged)
    symmetry_bonus = 0.3 * symmetry
    if components == 0:
        component_bonus = 0
    elif components <= 5:
        component_bonus = 0.2 * (components / 5)
    else:
        component_bonus = max(0, 0.2 * (1 - (components - 5) / 20))

    # Final score: product of gates, plus bonuses scaled by base
    base_score = density_gate * edge_gate * coherence_gate * compress_gate
    raw_score = base_score * (1 + symmetry_bonus + component_bonus)
    return min(1.0, raw_score)


# ============================================================================
# ALTERNATIVE METRICS (for phase transition analysis)
# ============================================================================

def order_symmetry(img: np.ndarray) -> float:
    """Pure symmetry-based order."""
    h_sym = np.mean(img == np.fliplr(img))
    v_sym = np.mean(img == np.flipud(img))
    rot180_sym = np.mean(img == np.rot90(img, 2))
    if img.shape[0] == img.shape[1]:
        rot90_sym = np.mean(img == np.rot90(img))
        diag_sym = np.mean(img == img.T)
    else:
        rot90_sym = 0
        diag_sym = 0
    return (h_sym + v_sym + rot90_sym + rot180_sym + diag_sym) / 5


def order_spectral(img: np.ndarray) -> float:
    """Fourier low-frequency dominance."""
    f = np.fft.fft2(img.astype(float) - 0.5)
    f_shifted = np.fft.fftshift(f)
    power = np.abs(f_shifted) ** 2
    h, w = img.shape
    cy, cx = h // 2, w // 2
    y, x = np.ogrid[:h, :w]
    r = np.sqrt((x - cx)**2 + (y - cy)**2)
    low_mask = r < (h / 4)
    low_power = np.sum(power[low_mask])
    total_power = np.sum(power) + 1e-10
    return low_power / total_power


def order_ising(img: np.ndarray) -> float:
    """Ising model alignment (physics-inspired)."""
    spins = 2 * img.astype(float) - 1
    h_align = np.sum(spins[:, :-1] * spins[:, 1:])
    v_align = np.sum(spins[:-1, :] * spins[1:, :])
    alignment = h_align + v_align
    max_align = 2 * img.size - img.shape[0] - img.shape[1]
    return (alignment / max_align + 1) / 2


# ============================================================================
# MAZE ORDER METRIC (verifiable, anti-trivial)
# ============================================================================

from collections import deque

def bfs_dist(open_mask: np.ndarray, start: tuple[int, int]) -> np.ndarray:
    """BFS to compute distances from start on open cells."""
    n, m = open_mask.shape
    dist = -np.ones((n, m), dtype=np.int32)
    si, sj = start
    if not open_mask[si, sj]:
        return dist
    q = deque()
    q.append((si, sj))
    dist[si, sj] = 0
    while q:
        i, j = q.popleft()
        d = dist[i, j] + 1
        if i > 0 and open_mask[i-1, j] and dist[i-1, j] < 0:
            dist[i-1, j] = d
            q.append((i-1, j))
        if i+1 < n and open_mask[i+1, j] and dist[i+1, j] < 0:
            dist[i+1, j] = d
            q.append((i+1, j))
        if j > 0 and open_mask[i, j-1] and dist[i, j-1] < 0:
            dist[i, j-1] = d
            q.append((i, j-1))
        if j+1 < m and open_mask[i, j+1] and dist[i, j+1] < 0:
            dist[i, j+1] = d
            q.append((i, j+1))
    return dist


def order_maze(img: np.ndarray) -> float:
    """
    Maze order: solvable + nontrivial + structured.

    Interpret img==1 as OPEN (walkable), img==0 as WALL.

    Gates:
    1. Solvability: path exists from (0,0) to (N-1,N-1) AND (0,N-1) to (N-1,0)
    2. Density: open cells in healthy band (~0.35-0.65)
    3. Reachability: most open cells connected (single component)
    4. Path length: requires nontrivial detours (not straight line)
    5. Junctions: enough branching (not one snake corridor)
    6. Dead ends: moderate (too few = blob, too many = noise)
    """
    n = img.shape[0]
    open_mask = (img.astype(np.uint8) == 1)

    # Force corners open - if not, maze is invalid
    if not open_mask[0, 0] or not open_mask[n-1, n-1]:
        return 0.0
    if not open_mask[0, n-1] or not open_mask[n-1, 0]:
        return 0.0

    # Distances from two corners
    dist_a = bfs_dist(open_mask, (0, 0))
    d1 = int(dist_a[n-1, n-1])
    if d1 < 0:  # Not solvable
        return 0.0

    dist_b = bfs_dist(open_mask, (0, n-1))
    d2 = int(dist_b[n-1, 0])
    if d2 < 0:  # Not solvable
        return 0.0

    # Reachability from (0,0)
    reachable = (dist_a >= 0)
    reachable_nodes = int(reachable.sum())
    open_nodes = int(open_mask.sum())
    if open_nodes == 0:
        return 0.0

    reachable_frac = reachable_nodes / open_nodes
    open_density = open_nodes / (n * n)

    # Degree statistics on reachable subgraph
    deg = np.zeros((n, n), dtype=np.uint8)
    deg[:, :-1] += (reachable[:, :-1] & reachable[:, 1:]).astype(np.uint8)
    deg[:, 1:] += (reachable[:, :-1] & reachable[:, 1:]).astype(np.uint8)
    deg[:-1, :] += (reachable[:-1, :] & reachable[1:, :]).astype(np.uint8)
    deg[1:, :] += (reachable[:-1, :] & reachable[1:, :]).astype(np.uint8)

    junctions = int(np.sum(reachable & (deg >= 3)))
    dead_ends = int(np.sum(reachable & (deg == 1)))

    junction_frac = junctions / max(1, reachable_nodes)
    deadend_frac = dead_ends / max(1, reachable_nodes)

    # Path length normalized by Manhattan straight-line distance
    manhattan_min = 2 * (n - 1)
    path_norm = (0.5 * (d1 + d2)) / max(1, manhattan_min)

    # ---- Gates ----
    # 1) Density: mazes like mid-density
    density_gate = gaussian_gate(open_density, center=0.50, sigma=0.12)

    # 2) Reachability: prefer single connected open region
    reach_gate = 1.0 / (1.0 + np.exp(-25.0 * (reachable_frac - 0.95)))

    # 3) Path length: require nontrivial detours (path_norm=1 is straight line)
    path_gate = 1.0 / (1.0 + np.exp(-3.0 * (path_norm - 1.5)))

    # 4) Junctions: avoid "one snake corridor"
    junction_gate = 1.0 / (1.0 + np.exp(-40.0 * (junction_frac - 0.03)))

    # 5) Dead ends: too few => open blob; too many => noisy junk
    deadend_gate = gaussian_gate(deadend_frac, center=0.12, sigma=0.08)

    base = density_gate * reach_gate * path_gate * junction_gate * deadend_gate

    # Bonus: reward longer paths gently
    length_bonus = 1.0 + 0.2 * np.tanh((path_norm - 1.5) / 2.0)

    return float(base * length_bonus)


def order_maze_v2(img: np.ndarray) -> float:
    """
    Stronger maze metric with anti-room and anti-blob gates.

    Core (verifiable):
    - Two corner-to-corner paths must exist

    Anti-triviality:
    - Anti-room: penalize 2x2 open blocks (bowls/caves)
    - Anti-blob: penalize cycles (blobs create many cycles)
    - Anti-room: penalize degree-4 cells (wide open areas)

    Band-pass:
    - Junction fraction in target range (not too few, not too many)
    """
    n = img.shape[0]
    open_mask = (img.astype(np.uint8) == 1)

    # Corner checks (hard constraint)
    if not (open_mask[0, 0] and open_mask[n-1, n-1] and
            open_mask[0, n-1] and open_mask[n-1, 0]):
        return 0.0

    # Two paths must exist
    dist_a = bfs_dist(open_mask, (0, 0))
    d1 = int(dist_a[n-1, n-1])
    if d1 < 0:
        return 0.0

    dist_b = bfs_dist(open_mask, (0, n-1))
    d2 = int(dist_b[n-1, 0])
    if d2 < 0:
        return 0.0

    reachable = (dist_a >= 0)
    reachable_nodes = int(reachable.sum())
    open_nodes = int(open_mask.sum())
    if open_nodes == 0 or reachable_nodes < 10:
        return 0.0

    open_density = open_nodes / (n * n)
    reachable_frac = reachable_nodes / open_nodes

    # Degree on reachable subgraph
    deg = np.zeros((n, n), dtype=np.uint8)
    deg[:, :-1] += (reachable[:, :-1] & reachable[:, 1:]).astype(np.uint8)
    deg[:, 1:] += (reachable[:, :-1] & reachable[:, 1:]).astype(np.uint8)
    deg[:-1, :] += (reachable[:-1, :] & reachable[1:, :]).astype(np.uint8)
    deg[1:, :] += (reachable[:-1, :] & reachable[1:, :]).astype(np.uint8)

    deg1 = int(np.sum(reachable & (deg == 1)))
    deg3 = int(np.sum(reachable & (deg == 3)))
    deg4 = int(np.sum(reachable & (deg == 4)))

    deadend_frac = deg1 / max(1, reachable_nodes)
    junction_frac = deg3 / max(1, reachable_nodes)
    deg4_frac = deg4 / max(1, reachable_nodes)

    # Path length normalized
    manhattan_min = 2 * (n - 1)
    path_norm = (0.5 * (d1 + d2)) / max(1, manhattan_min)

    # ---- Anti-room: 2x2 open blocks ----
    o = open_mask.astype(np.uint8)
    r = reachable.astype(np.uint8)
    block = (o[:-1, :-1] & o[1:, :-1] & o[:-1, 1:] & o[1:, 1:]).astype(np.uint8)
    block_reach = (r[:-1, :-1] & r[1:, :-1] & r[:-1, 1:] & r[1:, 1:]).astype(np.uint8)
    room_blocks = int(np.sum(block & block_reach))
    room_frac = room_blocks / max(1, reachable_nodes)

    # ---- Anti-blob: cycles per node (cyclomatic number) ----
    horiz = int(np.sum(reachable[:, :-1] & reachable[:, 1:]))
    vert = int(np.sum(reachable[:-1, :] & reachable[1:, :]))
    edges = horiz + vert
    nodes = reachable_nodes
    cycles = max(0, edges - nodes + 1)
    cycles_per_node = cycles / max(1, nodes)

    # ---- Gates ----
    # Density: broader range to allow different maze styles
    density_gate = gaussian_gate(open_density, center=0.45, sigma=0.15)

    # Connectedness: want one big open component
    reach_gate = 1.0 / (1.0 + np.exp(-25.0 * (reachable_frac - 0.90)))

    # Nontrivial detours (relaxed from 2.0)
    path_gate = 1.0 / (1.0 + np.exp(-3.0 * (path_norm - 1.2)))

    # Junctions: broader band (random walks have more junctions)
    junction_gate = gaussian_gate(junction_frac, center=0.15, sigma=0.10)

    # Dead ends: moderate amount
    deadend_gate = gaussian_gate(deadend_frac, center=0.10, sigma=0.10)

    # Penalize big rooms - relaxed threshold
    room_gate = 1.0 / (1.0 + np.exp(20.0 * (room_frac - 0.15)))

    # Penalize too many degree-4 cells - relaxed
    deg4_gate = 1.0 / (1.0 + np.exp(20.0 * (deg4_frac - 0.15)))

    # Penalize cycles - relaxed
    cycles_gate = 1.0 / (1.0 + np.exp(20.0 * (cycles_per_node - 0.20)))

    base = (density_gate * reach_gate * path_gate *
            junction_gate * deadend_gate *
            room_gate * deg4_gate * cycles_gate)

    # Gentle reward for longer paths
    length_bonus = 1.0 + 0.25 * np.tanh((path_norm - 2.0) / 2.0)

    return float(base * length_bonus)


# ============================================================================
# SAT ORDER METRIC (verifiable, computational complexity)
# ============================================================================

def get_hard_sat_instance(n_vars: int = 64, seed: int = 42) -> tuple:
    """
    Generates a fixed, hard 3-SAT instance.
    Ratio 4.26 is the phase transition (hardest region).
    64 vars * 4.26 ≈ 272 clauses.
    """
    rng = np.random.RandomState(seed)
    n_clauses = int(n_vars * 4.26)
    clauses = []

    for _ in range(n_clauses):
        # Pick 3 distinct variables
        vars_idx = rng.choice(n_vars, 3, replace=False)
        # Random negation for each
        lits = []
        for v in vars_idx:
            if rng.rand() < 0.5:
                lits.append(int(v + 1))   # Positive literal (1-based)
            else:
                lits.append(int(-(v + 1)))  # Negative literal
        clauses.append(lits)

    return n_vars, clauses


# Global SAT instance (lazy init for consistency across evaluations)
SAT_INSTANCE = None


def order_sat(img: np.ndarray) -> float:
    """
    3-SAT Order: Fraction of satisfied clauses.

    Interprets the image as a bitstream of boolean variables.
    Uses a fixed hard 3-SAT instance (ratio 4.26 = phase transition).

    This metric tests whether structure helps or hurts on uncorrelated
    logical problems. Hypothesis: Uniform prior should win because
    SAT variables have no spatial correlation.
    """
    global SAT_INSTANCE

    # Flatten image to get variables
    bits = (img.flatten() > 0.5)

    if SAT_INSTANCE is None:
        # Use 64 variables (first 8x8 block of 32x32 image)
        SAT_INSTANCE = get_hard_sat_instance(n_vars=64, seed=42)

    n_vars, clauses = SAT_INSTANCE

    # Extract variables from image (first 64 pixels)
    assignment = bits[:n_vars]

    satisfied_count = 0
    for clause in clauses:
        clause_sat = False
        for lit in clause:
            var_idx = abs(lit) - 1
            val = assignment[var_idx]

            # If lit>0, we need val=True. If lit<0, we need val=False.
            if (lit > 0 and val) or (lit < 0 and not val):
                clause_sat = True
                break

        if clause_sat:
            satisfied_count += 1

    return satisfied_count / len(clauses)


# ============================================================================
# KOLMOGOROV COMPLEXITY PROXY
# ============================================================================

def order_kolmogorov_proxy(img: np.ndarray) -> float:
    """
    Direct Kolmogorov complexity proxy: compressibility without tiling.

    Connects to theoretical bound from algorithmic information theory:
    - For N-bit strings, at most 2^K have Kolmogorov complexity < K bits
    - Fraction of images with K(x) < K is at most 2^(K-N)

    For 64x64 binary images (N=4096 bits):
    - 10% compression (K < 3686) → fraction < 2^(-410) ≈ 10^(-123)
    - Even 1% compression → astronomically rare

    This metric returns compressibility [0,1] where:
    - 0 = incompressible (appears random, K(x) ≈ N)
    - 1 = maximally compressible (highly structured, K(x) << N)

    Higher values = more structured = RARER under uniform distribution.
    """
    packed = np.packbits(img.flatten())
    compressed = zlib.compress(bytes(packed), level=9)
    raw_bits = img.size
    compressed_bits = len(compressed) * 8
    return max(0, 1 - (compressed_bits / raw_bits))


# ============================================================================
# METRIC REGISTRY
# ============================================================================

ORDER_METRICS = {
    'multiplicative': order_multiplicative,
    'maze': order_maze,
    'maze_v2': order_maze_v2,
    'sat': order_sat,
    'symmetry': order_symmetry,
    'spectral': order_spectral,
    'ising': order_ising,
    'kolmogorov': order_kolmogorov_proxy,
}


# ============================================================================
# PRIOR LOG-PROBABILITY
# ============================================================================

def log_prior(cppn: CPPN) -> float:
    w = cppn.get_weights()
    return -0.5 * np.sum(w**2) / (PRIOR_SIGMA**2)


# ============================================================================
# ELLIPTICAL SLICE SAMPLING
# ============================================================================

def elliptical_slice_sample(
    cppn: CPPN,
    threshold: float,
    image_size: int,
    order_fn: Callable,
    max_contractions: int = 100,
    max_restarts: int = 5
) -> tuple[CPPN, np.ndarray, float, float, int, bool]:
    """
    Elliptical slice sampling for prior-preserving constrained sampling.

    Key insight: Steps along ellipse defined by current point and
    a random draw from the prior. Every point on the ellipse has
    the same prior probability, so we only check the constraint.

    Returns: (cppn, image, order, log_prior, n_contractions, success)
    The success flag indicates whether a valid point was found.
    """
    current_w = cppn.get_weights()
    n_params = len(current_w)
    total_contractions = 0

    for restart in range(max_restarts):
        # Draw auxiliary vector from prior (defines the ellipse)
        nu = np.random.randn(n_params) * PRIOR_SIGMA

        # Initial angle and bracket
        phi = np.random.uniform(0, 2 * np.pi)
        phi_min = phi - 2 * np.pi
        phi_max = phi

        n_contractions = 0

        while n_contractions < max_contractions:
            # Proposal on ellipse: w' = w*cos(φ) + ν*sin(φ)
            proposal_w = current_w * np.cos(phi) + nu * np.sin(phi)

            proposal_cppn = cppn.copy()
            proposal_cppn.set_weights(proposal_w)
            proposal_img = proposal_cppn.render(image_size)
            proposal_order = order_fn(proposal_img)

            # Use >= to handle ties at threshold
            if proposal_order >= threshold:
                return proposal_cppn, proposal_img, proposal_order, log_prior(proposal_cppn), total_contractions + n_contractions, True

            # Shrink bracket
            if phi < 0:
                phi_min = phi
            else:
                phi_max = phi

            phi = np.random.uniform(phi_min, phi_max)
            n_contractions += 1

            if phi_max - phi_min < 1e-10:
                break

        total_contractions += n_contractions

    # All restarts failed - return current point but flag as failure
    current_img = cppn.render(image_size)
    return cppn, current_img, order_fn(current_img), log_prior(cppn), total_contractions, False


# ============================================================================
# DIVERSITY-AWARE REPLACEMENT
# ============================================================================

def compute_image_distance(img1: np.ndarray, img2: np.ndarray) -> float:
    """Hamming distance normalized to [0, 1]."""
    return np.mean(img1 != img2)


def select_diverse_seed(
    live_points: list,
    exclude_idx: int,
    n_candidates: int = 5
) -> int:
    """
    Select seed that maximizes minimum distance to other live points.
    This encourages exploration of different modes.
    """
    candidates = [i for i in range(len(live_points)) if i != exclude_idx]
    if len(candidates) <= n_candidates:
        selected = candidates
    else:
        selected = random.sample(candidates, n_candidates)

    best_idx = selected[0]
    best_min_dist = 0

    for idx in selected:
        min_dist = float('inf')
        for other_idx in range(len(live_points)):
            if other_idx != idx:
                dist = compute_image_distance(
                    live_points[idx].image,
                    live_points[other_idx].image
                )
                min_dist = min(min_dist, dist)

        if min_dist > best_min_dist:
            best_min_dist = min_dist
            best_idx = idx

    return best_idx


def select_diverse_seed_from_candidates(
    live_points: list,
    valid_indices: list,
    n_candidates: int = 5
) -> int:
    """
    Select seed from valid_indices that maximizes minimum distance to other live points.
    """
    if len(valid_indices) <= n_candidates:
        selected = valid_indices
    else:
        selected = random.sample(valid_indices, n_candidates)

    best_idx = selected[0]
    best_min_dist = 0

    for idx in selected:
        min_dist = float('inf')
        for other_idx in range(len(live_points)):
            if other_idx != idx:
                dist = compute_image_distance(
                    live_points[idx].image,
                    live_points[other_idx].image
                )
                min_dist = min(min_dist, dist)

        if min_dist > best_min_dist:
            best_min_dist = min_dist
            best_idx = idx

    return best_idx


# ============================================================================
# NESTED SAMPLING
# ============================================================================

@dataclass
class LivePoint:
    cppn: CPPN
    image: np.ndarray
    order_value: float
    log_prior: float
    metrics: dict = field(default_factory=dict)

@dataclass
class DeadPoint:
    image: np.ndarray
    order_value: float
    log_X: float
    iteration: int
    metrics: dict = field(default_factory=dict)


@dataclass
class LiveSnapshot:
    """Snapshot of live point statistics for phase transition analysis."""
    iteration: int
    log_X: float
    mean_order: float
    var_order: float
    min_order: float
    max_order: float
    mean_diversity: float  # Mean pairwise Hamming distance
    metric_means: dict = field(default_factory=dict)
    metric_vars: dict = field(default_factory=dict)


def compute_live_snapshot(
    live_points: list,
    iteration: int,
    log_X: float,
    n_diversity_samples: int = 20
) -> LiveSnapshot:
    """Compute statistics over live points for phase transition detection."""
    order_vals = [lp.order_value for lp in live_points]

    # Compute mean pairwise diversity (sample for efficiency)
    if len(live_points) > n_diversity_samples:
        sample_indices = random.sample(range(len(live_points)), n_diversity_samples)
    else:
        sample_indices = list(range(len(live_points)))

    diversity_sum = 0
    diversity_count = 0
    for i in range(len(sample_indices)):
        for j in range(i + 1, len(sample_indices)):
            idx_i, idx_j = sample_indices[i], sample_indices[j]
            diversity_sum += np.mean(live_points[idx_i].image != live_points[idx_j].image)
            diversity_count += 1
    mean_diversity = diversity_sum / max(1, diversity_count)

    # Compute metric statistics
    metric_means = {}
    metric_vars = {}
    if live_points[0].metrics:
        for metric_name in live_points[0].metrics.keys():
            vals = [lp.metrics.get(metric_name, 0) for lp in live_points]
            metric_means[metric_name] = np.mean(vals)
            metric_vars[metric_name] = np.var(vals)

    return LiveSnapshot(
        iteration=iteration,
        log_X=log_X,
        mean_order=np.mean(order_vals),
        var_order=np.var(order_vals),
        min_order=min(order_vals),
        max_order=max(order_vals),
        mean_diversity=mean_diversity,
        metric_means=metric_means,
        metric_vars=metric_vars
    )


def nested_sampling_v3(
    n_live: int = 100,
    n_iterations: int = 1000,
    image_size: int = 32,
    order_fn: Callable = order_multiplicative,
    sampling_mode: str = "measure",  # "measure" or "illumination"
    track_metrics: bool = True,
    output_dir: str = "thermo_output_v3",
    seed: Optional[int] = None
):
    """
    CPPN-specific nested sampling with illumination mode support.

    NOTE: For multi-prior comparison (cppn vs uniform vs walk etc.),
    use nested_sampling_with_prior() instead. This function is specialized
    for CPPN exploration with detailed metric tracking and illumination modes.

    Features:
    - Multiplicative order metric with spectral coherence
    - Elliptical slice sampling with safety checks
    - Dual modes: measure (calibrated) vs illumination (diverse)
    - Multi-metric tracking with DeadPoint dataclass

    Modes:
    - "measure": Random seed selection. Produces calibrated rarity curves
                 suitable for scientific publication.
    - "illumination": Diversity-aware seed selection. Explores more modes
                      but biases the live set away from the true prior.
                      Use for discovery, not for quantitative claims.
    """
    set_global_seed(seed)
    use_diversity = (sampling_mode == "illumination")
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 70)
    print("THERMODYNAMIC ILLUMINATION v3")
    print("=" * 70)
    print(f"Live points: {n_live}")
    print(f"Iterations: {n_iterations}")
    print(f"Image size: {image_size}×{image_size}")
    print(f"Order metric: {order_fn.__name__}")
    print(f"Sampling mode: {sampling_mode.upper()}")
    if sampling_mode == "measure":
        print("  → Calibrated rarity curve (random seeds)")
    else:
        print("  → Discovery mode (diversity-biased seeds)")
    print(f"Multi-metric tracking: {track_metrics}")
    print()

    # Sanity check
    print("Sanity check - random noise scores:")
    noise_scores = [order_fn((np.random.rand(image_size, image_size) > 0.5).astype(np.uint8))
                    for _ in range(100)]
    print(f"  Noise: mean={np.mean(noise_scores):.4f}, std={np.std(noise_scores):.4f}")
    print(f"  Range: [{min(noise_scores):.4f}, {max(noise_scores):.4f}]")
    print()

    # Initialize
    print("Initializing live points...")
    live_points: list[LivePoint] = []

    for _ in range(n_live):
        cppn = CPPN()
        img = cppn.render(image_size)
        o = order_fn(img)
        lp = log_prior(cppn)

        metrics = {}
        if track_metrics:
            metrics = {
                'multiplicative': order_multiplicative(img),
                'symmetry': order_symmetry(img),
                'spectral': order_spectral(img),
                'ising': order_ising(img),
            }

        live_points.append(LivePoint(cppn, img, o, lp, metrics))

    order_vals = [lp.order_value for lp in live_points]
    print(f"Initial order: [{min(order_vals):.4f}, {max(order_vals):.4f}], mean={np.mean(order_vals):.4f}")
    print()

    dead_points: list[DeadPoint] = []
    snapshots: list[LiveSnapshot] = []  # For phase transition analysis

    print("Starting nested sampling with ESS...")
    print("-" * 70)

    total_contractions = 0
    report_interval = max(1, n_iterations // 10)
    snapshot_interval = max(1, n_iterations // 50)  # More frequent snapshots

    for iteration in range(n_iterations):
        # Find worst
        worst_idx = min(range(n_live), key=lambda i: live_points[i].order_value)
        worst = live_points[worst_idx]
        threshold = worst.order_value

        # Standard nested sampling: after k removals, log_X ≈ -(k+1)/n_live
        log_X = -(iteration + 1) / n_live

        dead_points.append(DeadPoint(
            image=worst.image.copy(),
            order_value=worst.order_value,
            log_X=log_X,
            iteration=iteration,
            metrics=worst.metrics.copy()
        ))

        # Select seed that satisfies the constraint (order >= threshold)
        # This is critical for ESS correctness
        valid_seeds = [i for i in range(n_live)
                       if i != worst_idx and live_points[i].order_value >= threshold]

        if not valid_seeds:
            # Fallback: use any non-worst point (shouldn't happen with >= comparison)
            valid_seeds = [i for i in range(n_live) if i != worst_idx]

        if use_diversity:
            # Select diverse seed from valid candidates only
            seed_idx = select_diverse_seed_from_candidates(live_points, valid_seeds)
        else:
            seed_idx = random.choice(valid_seeds)

        # Replace using ESS
        new_cppn, new_img, new_order, new_logp, n_contract, success = elliptical_slice_sample(
            live_points[seed_idx].cppn, threshold, image_size, order_fn
        )

        # If ESS failed, try another seed
        if not success:
            for retry in range(3):
                alt_seed = random.choice(valid_seeds)
                new_cppn, new_img, new_order, new_logp, extra_contract, success = elliptical_slice_sample(
                    live_points[alt_seed].cppn, threshold, image_size, order_fn
                )
                n_contract += extra_contract
                if success:
                    break

        total_contractions += n_contract

        metrics = {}
        if track_metrics:
            metrics = {
                'multiplicative': order_multiplicative(new_img),
                'symmetry': order_symmetry(new_img),
                'spectral': order_spectral(new_img),
                'ising': order_ising(new_img),
            }

        live_points[worst_idx] = LivePoint(new_cppn, new_img, new_order, new_logp, metrics)

        # Collect snapshots for phase transition analysis
        if track_metrics and (iteration % snapshot_interval == 0 or iteration == n_iterations - 1):
            snapshot = compute_live_snapshot(live_points, iteration, log_X)
            snapshots.append(snapshot)

        if (iteration + 1) % report_interval == 0 or iteration == 0:
            order_vals = [lp.order_value for lp in live_points]
            avg_contract = total_contractions / report_interval if iteration > 0 else n_contract

            # Diversity metric
            unique = len(set(tuple(lp.image.flatten()) for lp in live_points))

            print(f"Iter {iteration+1:4d} | Thresh: {threshold:.4f} | "
                  f"Live: [{min(order_vals):.3f}, {max(order_vals):.3f}] | "
                  f"Contractions: {avg_contract:.1f} | Unique: {unique}")

            total_contractions = 0

    print("-" * 70)
    print("Complete!")
    print()

    # Results
    print_rarity_curve(dead_points, output_dir)
    print_difficulty_ladder(dead_points, output_dir)
    print_final_live_points(live_points, output_dir)

    if track_metrics:
        analyze_phase_transitions_v2(snapshots, output_dir)

    return dead_points, live_points, snapshots


# ============================================================================
# OUTPUT FUNCTIONS
# ============================================================================

def print_rarity_curve(dead_points, output_dir):
    print("=" * 70)
    print("RARITY CURVE")
    print("=" * 70)

    log_X_targets = [0, -1, -2, -3, -5, -7, -10, -15, -20]

    print(f"{'log(X)':>10} | {'Order':>10} | {'P(order>t)':>12} | {'~1 in N':>15}")
    print("-" * 60)

    for target in log_X_targets:
        idx = min(range(len(dead_points)), key=lambda i: abs(dead_points[i].log_X - target))
        dp = dead_points[idx]

        if dp.log_X > -50:
            prob = math.exp(dp.log_X)
            rarity = 1 / prob if prob > 1e-15 else float('inf')
            rarity_str = f"1 in {rarity:.1e}" if rarity < 1e15 else "astronomical"
            prob_str = f"{prob:.2e}"
        else:
            prob_str = "~0"
            rarity_str = "astronomical"

        print(f"{dp.log_X:>10.1f} | {dp.order_value:>10.4f} | {prob_str:>12} | {rarity_str:>15}")
        save_image_pbm(dp.image, f"{output_dir}/logX_{int(abs(target)):02d}.pbm")

    print()


def print_difficulty_ladder(dead_points, output_dir):
    print("=" * 70)
    print("DIFFICULTY LADDER")
    print("=" * 70)

    for target in [0, -2, -5, -10]:
        idx = min(range(len(dead_points)), key=lambda i: abs(dead_points[i].log_X - target))
        dp = dead_points[idx]
        print(f"\nlog(X) ≈ {dp.log_X:.1f} | Order: {dp.order_value:.4f}")
        print(img_to_ascii(dp.image, width=32))

    print()


def print_final_live_points(live_points, output_dir):
    print("=" * 70)
    print("FINAL LIVE POINTS")
    print("=" * 70)

    live_points_sorted = sorted(live_points, key=lambda lp: lp.order_value, reverse=True)

    for i, lp in enumerate(live_points_sorted[:5]):
        print(f"\n#{i+1} Order: {lp.order_value:.4f}")
        print(img_to_ascii(lp.image, width=32))
        save_image_pbm(lp.image, f"{output_dir}/final_{i+1}.pbm")

    unique = len(set(tuple(lp.image.flatten()) for lp in live_points_sorted[:10]))
    print(f"\nUnique patterns in top 10: {unique}")
    print()


def analyze_phase_transitions(dead_points, output_dir):
    """Look for discontinuities in the rarity curve across metrics (legacy)."""
    print("=" * 70)
    print("PHASE TRANSITION ANALYSIS (dead points)")
    print("=" * 70)

    metrics = ['multiplicative', 'symmetry', 'spectral', 'ising']

    for metric in metrics:
        values = [dp.metrics.get(metric, 0) for dp in dead_points]
        log_Xs = [dp.log_X for dp in dead_points]

        if len(values) > 10:
            # Compute slope
            slopes = np.gradient(values) / (np.gradient(log_Xs) + 1e-10)
            max_slope_idx = np.argmax(np.abs(slopes[5:-5])) + 5

            print(f"\n{metric}:")
            print(f"  Range: [{min(values):.4f}, {max(values):.4f}]")
            print(f"  Steepest change at log(X) ≈ {log_Xs[max_slope_idx]:.1f}")

    print()


def analyze_phase_transitions_v2(snapshots: list, output_dir: str):
    """
    Improved phase transition analysis using live point statistics.

    Looks for:
    - Sharp changes in mean order (first derivative)
    - Peaks in variance (susceptibility in physics terms)
    - Changes in diversity (mode collapse detection)
    """
    if len(snapshots) < 5:
        print("Not enough snapshots for phase transition analysis")
        return

    print("=" * 70)
    print("PHASE TRANSITION ANALYSIS (live point statistics)")
    print("=" * 70)

    log_Xs = [s.log_X for s in snapshots]
    mean_orders = [s.mean_order for s in snapshots]
    var_orders = [s.var_order for s in snapshots]
    diversities = [s.mean_diversity for s in snapshots]

    # 1. Find variance peak (susceptibility maximum)
    if len(var_orders) > 3:
        # Smooth variance to reduce noise
        window = min(5, len(var_orders) // 3)
        if window >= 3:
            smoothed_var = np.convolve(var_orders, np.ones(window)/window, mode='valid')
            peak_idx = np.argmax(smoothed_var) + window // 2
        else:
            peak_idx = np.argmax(var_orders)

        print("\n1. VARIANCE PEAK (susceptibility)")
        print("-" * 50)
        print(f"  Peak variance at log(X) ≈ {log_Xs[peak_idx]:.2f}")
        print(f"  Variance at peak: {var_orders[peak_idx]:.6f}")
        print(f"  Mean order at peak: {mean_orders[peak_idx]:.4f}")

    # 2. Find steepest mean change (order parameter jump)
    if len(mean_orders) > 5:
        d_mean = np.gradient(mean_orders)
        d_log_X = np.gradient(log_Xs)
        slopes = d_mean / (np.abs(d_log_X) + 1e-10)

        # Smooth slopes
        window = min(5, len(slopes) // 3)
        if window >= 3:
            smoothed_slopes = np.convolve(np.abs(slopes), np.ones(window)/window, mode='valid')
            steep_idx = np.argmax(smoothed_slopes) + window // 2
        else:
            steep_idx = np.argmax(np.abs(slopes))

        print("\n2. STEEPEST MEAN CHANGE")
        print("-" * 50)
        print(f"  Steepest change at log(X) ≈ {log_Xs[steep_idx]:.2f}")
        print(f"  Mean order: {mean_orders[steep_idx]:.4f}")
        print(f"  Slope magnitude: {abs(slopes[steep_idx]):.4f}")

    # 3. Diversity changes (mode collapse detection)
    if len(diversities) > 3:
        d_div = np.gradient(diversities)
        # Find biggest diversity drop
        min_d_div_idx = np.argmin(d_div)

        print("\n3. DIVERSITY CHANGES")
        print("-" * 50)
        print(f"  Initial diversity: {diversities[0]:.4f}")
        print(f"  Final diversity: {diversities[-1]:.4f}")
        print(f"  Largest drop at log(X) ≈ {log_Xs[min_d_div_idx]:.2f}")

    # 4. Per-metric analysis
    print("\n4. PER-METRIC VARIANCE PEAKS")
    print("-" * 50)

    if snapshots[0].metric_vars:
        for metric_name in snapshots[0].metric_vars.keys():
            metric_vars = [s.metric_vars.get(metric_name, 0) for s in snapshots]
            if len(metric_vars) > 3 and max(metric_vars) > 0:
                peak_idx = np.argmax(metric_vars)
                print(f"  {metric_name}: peak variance at log(X) ≈ {log_Xs[peak_idx]:.2f} "
                      f"(var={metric_vars[peak_idx]:.6f})")

    # 5. Save data
    data_file = f"{output_dir}/phase_stats.csv"
    with open(data_file, 'w') as f:
        headers = ['iteration', 'log_X', 'mean_order', 'var_order', 'diversity']
        if snapshots[0].metric_means:
            for name in snapshots[0].metric_means.keys():
                headers.extend([f'{name}_mean', f'{name}_var'])
        f.write(','.join(headers) + '\n')

        for s in snapshots:
            row = [str(s.iteration), f'{s.log_X:.6f}', f'{s.mean_order:.6f}',
                   f'{s.var_order:.8f}', f'{s.mean_diversity:.6f}']
            if s.metric_means:
                for name in s.metric_means.keys():
                    row.append(f'{s.metric_means.get(name, 0):.6f}')
                    row.append(f'{s.metric_vars.get(name, 0):.8f}')
            f.write(','.join(row) + '\n')

    print(f"\nSaved statistics to {data_file}")
    print()


# ============================================================================
# UTILITIES
# ============================================================================

def img_to_ascii(img: np.ndarray, width: int = 64) -> str:
    h, w = img.shape
    scale = max(1, w // width)
    lines = []
    for i in range(0, h, scale * 2):
        line = ""
        for j in range(0, w, scale):
            region = img[i:i+scale*2, j:j+scale]
            val = np.mean(region) if region.size > 0 else 0
            if val > 0.75:
                line += "█"
            elif val > 0.5:
                line += "▓"
            elif val > 0.25:
                line += "░"
            else:
                line += " "
        lines.append(line)
    return "\n".join(lines)


def save_image_pbm(img: np.ndarray, filename: str):
    h, w = img.shape
    with open(filename, 'w') as f:
        f.write(f"P1\n{w} {h}\n")
        for row in img:
            f.write(" ".join(str(int(p)) for p in row) + "\n")


# ============================================================================
# CLUSTERING FOR MULTI-MODAL SAMPLING
# ============================================================================

def cluster_live_points(live_points: list, n_clusters: int = 5) -> list[list[int]]:
    """
    Simple k-means-style clustering on image space.
    Returns list of cluster assignments.
    """
    n = len(live_points)
    if n <= n_clusters:
        return [[i] for i in range(n)]

    # Initialize centroids randomly
    centroid_idxs = random.sample(range(n), n_clusters)
    centroids = [live_points[i].image.flatten().astype(float) for i in centroid_idxs]

    # Iterate k-means
    for _ in range(10):
        # Assign points to nearest centroid
        clusters = [[] for _ in range(n_clusters)]
        for i, lp in enumerate(live_points):
            img_flat = lp.image.flatten().astype(float)
            distances = [np.sum((img_flat - c)**2) for c in centroids]
            cluster_idx = np.argmin(distances)
            clusters[cluster_idx].append(i)

        # Update centroids
        new_centroids = []
        for cluster in clusters:
            if cluster:
                cluster_imgs = [live_points[i].image.flatten().astype(float) for i in cluster]
                new_centroids.append(np.mean(cluster_imgs, axis=0))
            else:
                new_centroids.append(centroids[len(new_centroids)])
        centroids = new_centroids

    return clusters


def select_seed_from_cluster(
    live_points: list,
    clusters: list[list[int]],
    worst_idx: int
) -> int:
    """
    Select seed from a different cluster than the worst point,
    preferring larger clusters.
    """
    # Find which cluster the worst point is in
    worst_cluster = -1
    for i, cluster in enumerate(clusters):
        if worst_idx in cluster:
            worst_cluster = i
            break

    # Select from a different cluster (weighted by size)
    other_clusters = [(i, cluster) for i, cluster in enumerate(clusters)
                      if i != worst_cluster and len(cluster) > 0]

    if not other_clusters:
        # Fall back to same cluster
        other_clusters = [(worst_cluster, clusters[worst_cluster])]

    # Weight by cluster size
    weights = [len(c) for _, c in other_clusters]
    total = sum(weights)
    r = random.random() * total
    cumsum = 0
    for (_, cluster), w in zip(other_clusters, weights):
        cumsum += w
        if r < cumsum:
            return random.choice([i for i in cluster if i != worst_idx] or cluster)

    return random.choice(other_clusters[-1][1])


# ============================================================================
# UNIFORM PRIOR (bit-flip MCMC)
# ============================================================================

def uniform_constrained_mcmc(
    seed_img: np.ndarray,
    threshold: float,
    order_fn: Callable,
    n_steps: int = 200,
    flip_prob: float = 0.02,
    p_global: float = 0.10
) -> tuple[np.ndarray, float, int]:
    """
    MCMC sampling under uniform prior on images.

    Proposals:
    - With prob p_global: entirely fresh uniform random grid (global refresh)
    - Otherwise: flip each bit with probability flip_prob (local)

    Global refresh helps escape local minima when the target set is
    far (in Hamming distance) from typical random grids.

    Prior is uniform, so accept iff constraint is satisfied.
    """
    current = seed_img.copy()
    current_order = order_fn(current)
    accepted = 0
    h, w = current.shape

    for _ in range(n_steps):
        if np.random.rand() < p_global:
            # Global independence proposal (still uniform prior!)
            proposal = (np.random.rand(h, w) > 0.5).astype(np.uint8)
        else:
            # Local bit-flip proposal
            flip_mask = np.random.rand(h, w) < flip_prob
            proposal = current.copy()
            proposal[flip_mask] = 1 - proposal[flip_mask]

        proposal_order = order_fn(proposal)

        if proposal_order >= threshold:
            current = proposal
            current_order = proposal_order
            accepted += 1

    return current, current_order, accepted


# ============================================================================
# RANDOM WALK CARVER PRIOR (for maze generation)
# ============================================================================

WALK_N_WALKERS = 4  # Fewer walkers = narrower corridors
WALK_STEPS_PER_WALKER = 150  # Enough to connect corners but not flood

@dataclass
class WalkCarver:
    """
    Random walk carver for maze generation.

    Latent representation: Gaussian parameters that control walker paths.
    Each walker has:
      - 2 params for start position (sigmoid → grid coords)
      - steps_per_walker params for direction angles

    Total params = n_walkers * (2 + steps_per_walker)
    """
    params: np.ndarray = None
    n_walkers: int = WALK_N_WALKERS
    steps_per_walker: int = WALK_STEPS_PER_WALKER

    def __post_init__(self):
        n_params = self.n_walkers * (2 + self.steps_per_walker)
        if self.params is None:
            self.params = np.random.randn(n_params) * PRIOR_SIGMA

    def copy(self) -> 'WalkCarver':
        return WalkCarver(
            params=self.params.copy(),
            n_walkers=self.n_walkers,
            steps_per_walker=self.steps_per_walker
        )

    def get_params(self) -> np.ndarray:
        return self.params

    def set_params(self, params: np.ndarray):
        self.params = params

    def render(self, size: int = 32) -> np.ndarray:
        """Generate maze by carving random walk paths.

        Convention: 0 = wall, 1 = passage (matches maze metric)

        First 4 walkers start from the 4 corners (required for maze solvability).
        Additional walkers start at parameterized positions.
        """
        # Start with all walls (0), carve passages (1)
        grid = np.zeros((size, size), dtype=np.uint8)

        # Corner positions (required for maze metric)
        corners = [(0, 0), (0, size-1), (size-1, 0), (size-1, size-1)]

        params_per_walker = 2 + self.steps_per_walker

        for w in range(self.n_walkers):
            base = w * params_per_walker

            # First 4 walkers start from corners, rest from parameterized positions
            if w < 4:
                sx, sy = corners[w]
            else:
                # Decode start position via sigmoid
                sx = int(1 / (1 + np.exp(-self.params[base])) * (size - 1))
                sy = int(1 / (1 + np.exp(-self.params[base + 1])) * (size - 1))

            x, y = sx, sy
            grid[y, x] = 1  # Carve passage at starting cell

            for s in range(self.steps_per_walker):
                # Direction from angle param
                angle = self.params[base + 2 + s]

                # Quantize angle to 4 cardinal directions
                # angle in [-inf, inf], use modular mapping
                dir_idx = int(np.floor((angle + 0.5) * 2)) % 4

                # Directions: 0=right, 1=down, 2=left, 3=up
                directions = [(1, 0), (0, 1), (-1, 0), (0, -1)]
                dx, dy = directions[dir_idx]

                # Move (stay in bounds)
                nx = np.clip(x + dx, 0, size - 1)
                ny = np.clip(y + dy, 0, size - 1)

                x, y = nx, ny
                grid[y, x] = 1  # Carve passage

        return grid


def walk_log_prior(walker: WalkCarver) -> float:
    """Log probability under Gaussian prior."""
    return -0.5 * np.sum(walker.params ** 2) / (PRIOR_SIGMA ** 2)


def walk_elliptical_slice_sample(
    walker: WalkCarver,
    threshold: float,
    image_size: int,
    order_fn: Callable,
    max_contractions: int = 100,
    max_restarts: int = 5
) -> tuple[WalkCarver, np.ndarray, float, float, int, bool]:
    """
    Elliptical slice sampling for WalkCarver prior.
    Same algorithm as CPPN ESS, adapted for walk params.
    """
    current_params = walker.get_params()

    for restart in range(max_restarts):
        # Sample from prior (the "ellipse")
        nu = np.random.randn(len(current_params)) * PRIOR_SIGMA

        # Initial angle
        u = np.random.rand()
        theta = 2 * np.pi * u
        theta_min, theta_max = theta - 2 * np.pi, theta

        contractions = 0

        for _ in range(max_contractions):
            # Propose on ellipse
            new_params = current_params * np.cos(theta) + nu * np.sin(theta)

            # Create new walker and evaluate
            new_walker = walker.copy()
            new_walker.set_params(new_params)
            new_img = new_walker.render(image_size)
            new_order = order_fn(new_img)

            if new_order >= threshold:
                new_logp = walk_log_prior(new_walker)
                return new_walker, new_img, new_order, new_logp, contractions, True

            # Shrink bracket
            contractions += 1
            if theta < 0:
                theta_min = theta
            else:
                theta_max = theta
            theta = np.random.uniform(theta_min, theta_max)

    # Failed - return original
    orig_img = walker.render(image_size)
    orig_order = order_fn(orig_img)
    orig_logp = walk_log_prior(walker)
    return walker, orig_img, orig_order, orig_logp, max_contractions, False


# ============================================================================
# SPANNING TREE MAZE PRIOR (guaranteed perfect mazes)
# ============================================================================

@dataclass
class SpanningTreeMaze:
    """
    Spanning tree maze generator using Kruskal's algorithm.

    Prior: Gaussian weights on grid edges.
    Maze: Minimum spanning tree defines passages.

    This guarantees:
    - All cells reachable (spanning tree property)
    - No loops (tree structure)
    - Narrow 1-cell corridors in the interior

    For a 32×32 output grid:
    - Use 16×16 "room" cells at even coordinates (0,2,4,...,30)
    - MST edges become 1-cell-wide corridors between rooms
    - Corner cells are extended to reach actual grid corners for maze solvability
      (this creates small 2×2 open areas at corners, which is intentional)
    """
    edge_weights: np.ndarray = None
    n_cells: int = 16  # Cells per dimension (16×16 = 256 rooms)

    def __post_init__(self):
        # Number of edges in grid graph:
        # Horizontal: n_cells * (n_cells - 1)
        # Vertical: (n_cells - 1) * n_cells
        n_edges = 2 * self.n_cells * (self.n_cells - 1)
        if self.edge_weights is None:
            self.edge_weights = np.random.randn(n_edges) * PRIOR_SIGMA

    def copy(self) -> 'SpanningTreeMaze':
        return SpanningTreeMaze(
            edge_weights=self.edge_weights.copy(),
            n_cells=self.n_cells
        )

    def get_params(self) -> np.ndarray:
        return self.edge_weights

    def set_params(self, params: np.ndarray):
        self.edge_weights = params

    def _build_edges(self) -> list:
        """Build list of (edge_idx, cell1, cell2) for grid graph."""
        edges = []
        n = self.n_cells
        idx = 0

        # Horizontal edges: (i, j) -- (i, j+1)
        for i in range(n):
            for j in range(n - 1):
                c1 = i * n + j
                c2 = i * n + j + 1
                edges.append((idx, c1, c2))
                idx += 1

        # Vertical edges: (i, j) -- (i+1, j)
        for i in range(n - 1):
            for j in range(n):
                c1 = i * n + j
                c2 = (i + 1) * n + j
                edges.append((idx, c1, c2))
                idx += 1

        return edges

    def _kruskal_mst(self, edges: list) -> set:
        """Compute MST using Kruskal's algorithm with Union-Find."""
        n_total = self.n_cells * self.n_cells

        # Sort edges by weight
        weighted = [(self.edge_weights[idx], c1, c2) for idx, c1, c2 in edges]
        weighted.sort()

        # Union-Find with path compression and union by rank
        parent = list(range(n_total))
        rank = [0] * n_total

        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]

        def union(x, y):
            px, py = find(x), find(y)
            if px == py:
                return False
            if rank[px] < rank[py]:
                px, py = py, px
            parent[py] = px
            if rank[px] == rank[py]:
                rank[px] += 1
            return True

        mst = set()
        for _, c1, c2 in weighted:
            if union(c1, c2):
                mst.add((min(c1, c2), max(c1, c2)))
                if len(mst) == n_total - 1:
                    break

        return mst

    def render(self, size: int = 32) -> np.ndarray:
        """Generate maze image from spanning tree.

        Output: size×size binary image (0=wall, 1=passage)
        Room cells at even coordinates, corridors connect them.
        Corner cells extended to reach actual corners for solvability.
        """
        grid = np.zeros((size, size), dtype=np.uint8)
        n = self.n_cells

        # Cell spacing in output grid
        spacing = size // n  # 2 for 32/16

        # Build MST
        edges = self._build_edges()
        mst = self._kruskal_mst(edges)

        # Draw room cells (all rooms are passages)
        for i in range(n):
            for j in range(n):
                r, c = i * spacing, j * spacing
                if r < size and c < size:
                    grid[r, c] = 1

        # Draw corridors for MST edges
        for c1, c2 in mst:
            i1, j1 = c1 // n, c1 % n
            i2, j2 = c2 // n, c2 % n

            r1, c1_pos = i1 * spacing, j1 * spacing
            r2, c2_pos = i2 * spacing, j2 * spacing

            # Draw line between (r1, c1_pos) and (r2, c2_pos)
            if r1 == r2:  # Horizontal corridor
                for c in range(min(c1_pos, c2_pos), max(c1_pos, c2_pos) + 1):
                    if c < size:
                        grid[r1, c] = 1
            else:  # Vertical corridor
                for r in range(min(r1, r2), max(r1, r2) + 1):
                    if r < size:
                        grid[r, c1_pos] = 1

        # Extend corner rooms to actual corners for maze solvability
        # Rooms are at (0,0), (0,30), (30,0), (30,30) for 16x16 cells
        # Corners need to be at (0,0), (0,31), (31,0), (31,31)
        last_room = (n - 1) * spacing  # e.g., 30 for 16 cells with spacing 2

        # Top-left corner (0,0) already a room - OK
        # Top-right: extend from (0, last_room) to (0, size-1)
        for c in range(last_room, size):
            grid[0, c] = 1
        # Bottom-left: extend from (last_room, 0) to (size-1, 0)
        for r in range(last_room, size):
            grid[r, 0] = 1
        # Bottom-right: extend from (last_room, last_room) to (size-1, size-1)
        for r in range(last_room, size):
            grid[r, last_room] = 1
        for c in range(last_room, size):
            grid[last_room, c] = 1
        # Fill the corner pixel explicitly
        grid[size-1, size-1] = 1

        return grid


def tree_log_prior(maze: SpanningTreeMaze) -> float:
    """Log probability under Gaussian prior on edge weights."""
    return -0.5 * np.sum(maze.edge_weights ** 2) / (PRIOR_SIGMA ** 2)


def tree_elliptical_slice_sample(
    maze: SpanningTreeMaze,
    threshold: float,
    image_size: int,
    order_fn: Callable,
    max_contractions: int = 100,
    max_restarts: int = 5
) -> tuple[SpanningTreeMaze, np.ndarray, float, float, int, bool]:
    """
    Elliptical slice sampling for SpanningTreeMaze prior.
    """
    current_params = maze.get_params()

    for restart in range(max_restarts):
        nu = np.random.randn(len(current_params)) * PRIOR_SIGMA
        u = np.random.rand()
        theta = 2 * np.pi * u
        theta_min, theta_max = theta - 2 * np.pi, theta

        contractions = 0

        for _ in range(max_contractions):
            new_params = current_params * np.cos(theta) + nu * np.sin(theta)

            new_maze = maze.copy()
            new_maze.set_params(new_params)
            new_img = new_maze.render(image_size)
            new_order = order_fn(new_img)

            if new_order >= threshold:
                new_logp = tree_log_prior(new_maze)
                return new_maze, new_img, new_order, new_logp, contractions, True

            contractions += 1
            if theta < 0:
                theta_min = theta
            else:
                theta_max = theta
            theta = np.random.uniform(theta_min, theta_max)

    # Failed - return original
    orig_img = maze.render(image_size)
    orig_order = order_fn(orig_img)
    orig_logp = tree_log_prior(maze)
    return maze, orig_img, orig_order, orig_logp, max_contractions, False


# ============================================================================
# DSL PRIOR (expression tree)
# ============================================================================

DSL_OPS = ['add', 'mul', 'sin', 'cos', 'gauss', 'abs', 'neg', 'threshold']
DSL_LEAVES = ['x', 'y', 'r', 'const']

@dataclass
class DSLExpr:
    op: str
    args: list = field(default_factory=list)
    value: float = 0.0  # For constants

    def eval(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        r = np.sqrt(x**2 + y**2)

        if self.op == 'x':
            return x
        elif self.op == 'y':
            return y
        elif self.op == 'r':
            return r
        elif self.op == 'const':
            return np.full_like(x, self.value)
        elif self.op == 'add':
            return self.args[0].eval(x, y) + self.args[1].eval(x, y)
        elif self.op == 'mul':
            return self.args[0].eval(x, y) * self.args[1].eval(x, y)
        elif self.op == 'sin':
            return np.sin(self.args[0].eval(x, y) * np.pi)
        elif self.op == 'cos':
            return np.cos(self.args[0].eval(x, y) * np.pi)
        elif self.op == 'gauss':
            v = self.args[0].eval(x, y)
            return np.exp(-v**2 * 2)
        elif self.op == 'abs':
            return np.abs(self.args[0].eval(x, y))
        elif self.op == 'neg':
            return -self.args[0].eval(x, y)
        elif self.op == 'threshold':
            return (self.args[0].eval(x, y) > 0).astype(float)
        else:
            return np.zeros_like(x)

    def render(self, size: int = 32) -> np.ndarray:
        coords = np.linspace(-1, 1, size)
        x, y = np.meshgrid(coords, coords)
        output = self.eval(x, y)
        return (output > 0).astype(np.uint8)

    def copy(self) -> 'DSLExpr':
        return DSLExpr(
            op=self.op,
            args=[a.copy() for a in self.args],
            value=self.value
        )

    def size(self) -> int:
        """Count nodes in expression tree."""
        return 1 + sum(a.size() for a in self.args)


def random_dsl_expr(max_depth: int = 3) -> DSLExpr:
    """Generate random DSL expression."""
    if max_depth <= 0 or random.random() < 0.3:
        # Leaf
        leaf = random.choice(DSL_LEAVES)
        if leaf == 'const':
            return DSLExpr('const', value=np.random.randn())
        return DSLExpr(leaf)

    # Internal node
    op = random.choice(DSL_OPS)
    if op in ['add', 'mul']:
        return DSLExpr(op, [random_dsl_expr(max_depth-1), random_dsl_expr(max_depth-1)])
    else:
        return DSLExpr(op, [random_dsl_expr(max_depth-1)])


def mutate_dsl_expr(expr: DSLExpr, mutation_rate: float = 0.2) -> DSLExpr:
    """Mutate DSL expression tree."""
    result = expr.copy()

    if random.random() < mutation_rate:
        # Replace subtree
        if result.op in DSL_LEAVES:
            new_leaf = random.choice(DSL_LEAVES)
            result.op = new_leaf
            if new_leaf == 'const':
                result.value = np.random.randn()
        else:
            # Mutate arguments
            for i in range(len(result.args)):
                if random.random() < 0.3:
                    result.args[i] = random_dsl_expr(2)
                else:
                    result.args[i] = mutate_dsl_expr(result.args[i], mutation_rate)

    if result.op == 'const' and random.random() < 0.5:
        result.value += np.random.randn() * 0.3

    return result


def dsl_log_prior(expr: DSLExpr, size_penalty: float = 0.5) -> float:
    """
    Log prior on DSL expressions.
    Favor shorter expressions (MDL-style).
    """
    return -size_penalty * expr.size()


def dsl_constrained_mcmc(
    seed_expr: DSLExpr,
    threshold: float,
    order_fn: Callable,
    image_size: int,
    n_steps: int = 50
) -> tuple[DSLExpr, np.ndarray, float, float, int]:
    """
    MCMC sampling on DSL expressions with MH acceptance.

    WARNING: This is APPROXIMATE sampling. The tree mutation proposals
    (subtree replacement, constant perturbation) are not symmetric, and
    we do not compute the Hastings correction q(x'|x)/q(x|x'). This means
    samples may be biased away from the true DSL prior. For rigorous
    results, use CPPN prior with Elliptical Slice Sampling instead.
    """
    current = seed_expr.copy()
    current_img = current.render(image_size)
    current_order = order_fn(current_img)
    current_logp = dsl_log_prior(current)
    accepted = 0

    for _ in range(n_steps):
        proposal = mutate_dsl_expr(current)
        proposal_img = proposal.render(image_size)
        proposal_order = order_fn(proposal_img)

        if proposal_order <= threshold:
            continue

        proposal_logp = dsl_log_prior(proposal)
        log_alpha = proposal_logp - current_logp

        if np.log(random.random()) < log_alpha:
            current = proposal
            current_img = proposal_img
            current_order = proposal_order
            current_logp = proposal_logp
            accepted += 1

    return current, current_img, current_order, current_logp, accepted


# ============================================================================
# PRIOR COMPARISON FRAMEWORK
# ============================================================================

def nested_sampling_with_prior(
    prior_type: str = "cppn",  # "cppn", "uniform", "dsl", "walk", "tree"
    n_live: int = 100,
    n_iterations: int = 500,
    image_size: int = 32,
    order_fn: Callable = order_multiplicative,
    use_clustering: bool = False,
    output_dir: str = "prior_comparison",
    seed: Optional[int] = None,
    verbose: bool = True
) -> tuple[list, list, str]:
    """
    General-purpose nested sampling for comparing different priors.

    This is the recommended entry point for prior comparison experiments.
    Supports: cppn, uniform, dsl, walk, tree priors.

    For CPPN-specific exploration with illumination modes and detailed
    metric tracking, use nested_sampling_v3() instead.

    Returns:
        (dead_points, live_points, prior_type) - dead points have 'order', 'log_X', etc.
    """
    set_global_seed(seed)
    os.makedirs(output_dir, exist_ok=True)

    if verbose:
        print(f"\n{'='*70}")
        print(f"PRIOR: {prior_type.upper()}")
        print(f"{'='*70}")

    # Initialize based on prior type
    if prior_type == "cppn":
        live_points = []
        for _ in range(n_live):
            cppn = CPPN()
            img = cppn.render(image_size)
            o = order_fn(img)
            lp = log_prior(cppn)
            live_points.append({
                'generator': cppn,
                'image': img,
                'order': o,
                'log_prior': lp
            })

    elif prior_type == "uniform":
        live_points = []
        for _ in range(n_live):
            img = (np.random.rand(image_size, image_size) > 0.5).astype(np.uint8)
            o = order_fn(img)
            live_points.append({
                'generator': None,
                'image': img,
                'order': o,
                'log_prior': 0.0  # Uniform prior
            })

    elif prior_type == "dsl":
        live_points = []
        for _ in range(n_live):
            expr = random_dsl_expr(3)
            img = expr.render(image_size)
            o = order_fn(img)
            lp = dsl_log_prior(expr)
            live_points.append({
                'generator': expr,
                'image': img,
                'order': o,
                'log_prior': lp
            })

    elif prior_type == "walk":
        live_points = []
        for _ in range(n_live):
            walker = WalkCarver()
            img = walker.render(image_size)
            o = order_fn(img)
            lp = walk_log_prior(walker)
            live_points.append({
                'generator': walker,
                'image': img,
                'order': o,
                'log_prior': lp
            })

    elif prior_type == "tree":
        live_points = []
        for _ in range(n_live):
            maze = SpanningTreeMaze()
            img = maze.render(image_size)
            o = order_fn(img)
            lp = tree_log_prior(maze)
            live_points.append({
                'generator': maze,
                'image': img,
                'order': o,
                'log_prior': lp
            })

    else:
        raise ValueError(f"Unknown prior type: {prior_type}")

    if verbose:
        order_vals = [lp['order'] for lp in live_points]
        print(f"Initial order: [{min(order_vals):.4f}, {max(order_vals):.4f}]")

    dead_points = []

    for iteration in range(n_iterations):
        # Find worst
        worst_idx = min(range(n_live), key=lambda i: live_points[i]['order'])
        worst = live_points[worst_idx]
        threshold = worst['order']
        # Standard nested sampling: after k removals, log_X ≈ -(k+1)/n_live
        log_X = -(iteration + 1) / n_live

        # Track secondary metrics for comparison
        ising_val = order_ising(worst['image'])
        symmetry_val = order_symmetry(worst['image'])

        dead_points.append({
            'image': worst['image'].copy(),
            'order': worst['order'],
            'log_X': log_X,
            'ising': ising_val,
            'symmetry': symmetry_val,
        })

        # Sample replacement based on prior
        if use_clustering and prior_type == "cppn":
            # Convert to LivePoint format for clustering
            lp_objs = [LivePoint(lp['generator'], lp['image'], lp['order'], lp['log_prior'])
                       for lp in live_points]
            clusters = cluster_live_points(lp_objs, n_clusters=5)
            seed_idx = select_seed_from_cluster(lp_objs, clusters, worst_idx)
        else:
            seed_idx = random.choice([i for i in range(n_live) if i != worst_idx])

        seed = live_points[seed_idx]

        if prior_type == "cppn":
            new_cppn, new_img, new_order, new_logp, _, success = elliptical_slice_sample(
                seed['generator'], threshold, image_size, order_fn
            )
            # If ESS failed, the returned point is the seed (still valid for >= threshold)
            live_points[worst_idx] = {
                'generator': new_cppn,
                'image': new_img,
                'order': new_order,
                'log_prior': new_logp
            }

        elif prior_type == "uniform":
            new_img, new_order, _ = uniform_constrained_mcmc(
                seed['image'], threshold, order_fn
            )
            live_points[worst_idx] = {
                'generator': None,
                'image': new_img,
                'order': new_order,
                'log_prior': 0.0
            }

        elif prior_type == "dsl":
            new_expr, new_img, new_order, new_logp, _ = dsl_constrained_mcmc(
                seed['generator'], threshold, order_fn, image_size
            )
            live_points[worst_idx] = {
                'generator': new_expr,
                'image': new_img,
                'order': new_order,
                'log_prior': new_logp
            }

        elif prior_type == "walk":
            new_walker, new_img, new_order, new_logp, _, success = walk_elliptical_slice_sample(
                seed['generator'], threshold, image_size, order_fn
            )
            live_points[worst_idx] = {
                'generator': new_walker,
                'image': new_img,
                'order': new_order,
                'log_prior': new_logp
            }

        elif prior_type == "tree":
            new_maze, new_img, new_order, new_logp, _, success = tree_elliptical_slice_sample(
                seed['generator'], threshold, image_size, order_fn
            )
            live_points[worst_idx] = {
                'generator': new_maze,
                'image': new_img,
                'order': new_order,
                'log_prior': new_logp
            }

        if verbose and (iteration + 1) % 100 == 0:
            order_vals = [lp['order'] for lp in live_points]
            print(f"Iter {iteration+1:4d} | Thresh: {threshold:.4f} | "
                  f"Range: [{min(order_vals):.3f}, {max(order_vals):.3f}]")

    return dead_points, live_points, prior_type


def compare_priors(
    priors: list[str] = ["cppn", "uniform", "dsl"],
    n_live: int = 50,
    n_iterations: int = 500,
    image_size: int = 32,
    output_dir: str = "prior_comparison",
    order_fn: Callable = None,
    n_runs: int = 1,
    base_seed: Optional[int] = None,
    save_csv: bool = True,
    verbose: bool = True
):
    """
    Run nested sampling with multiple priors and compare rarity curves.
    """
    if order_fn is None:
        order_fn = order_multiplicative

    os.makedirs(output_dir, exist_ok=True)

    print("=" * 70)
    print(f"PRIOR COMPARISON: {order_fn.__name__}")
    print("=" * 70)
    print(f"Priors: {', '.join(priors)}")
    print(f"Runs per prior: {n_runs}")
    if base_seed is not None:
        print(f"Base seed: {base_seed}")
    print()

    if n_runs < 1:
        raise ValueError("n_runs must be >= 1")

    if base_seed is None:
        base_seed = 0

    run_seeds = [base_seed + i for i in range(n_runs)]

    def _write_csv(path: str, fieldnames: list[str], rows: list[dict]) -> None:
        dir_name = os.path.dirname(path)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    results: dict[str, dict] = {}
    run_summary_rows: list[dict] = []

    for prior in priors:
        prior_runs: list[dict] = []

        for run_idx, seed in enumerate(run_seeds):
            run_output_dir = f"{output_dir}/{prior}/run_{run_idx:02d}"
            if verbose:
                print(f"[{prior}] run {run_idx + 1}/{n_runs} (seed={seed})")

            dead, live, _ = nested_sampling_with_prior(
                prior_type=prior,
                n_live=n_live,
                n_iterations=n_iterations,
                image_size=image_size,
                order_fn=order_fn,
                output_dir=run_output_dir,
                seed=seed,
                verbose=(verbose and n_runs == 1)
            )

            prior_runs.append({"seed": seed, "dead": dead, "live": live})

            # Per-run curve CSV (dead points)
            if save_csv:
                curve_rows = []
                for i, d in enumerate(dead):
                    curve_rows.append({
                        "prior": prior,
                        "run": run_idx,
                        "seed": seed,
                        "iteration": i,
                        "log_X": d.get("log_X", float("nan")),
                        "order": d.get("order", float("nan")),
                        "symmetry": d.get("symmetry", float("nan")),
                        "ising": d.get("ising", float("nan")),
                    })
                _write_csv(
                    f"{run_output_dir}/curve_dead.csv",
                    ["prior", "run", "seed", "iteration", "log_X", "order", "symmetry", "ising"],
                    curve_rows
                )

            # Per-run final live set summary
            orders = [p.get("order", 0.0) for p in live]
            imgs = [p.get("image") for p in live if p.get("image") is not None]
            unique = len(set(img.tobytes() for img in imgs)) if imgs else 0
            run_summary_rows.append({
                "prior": prior,
                "run": run_idx,
                "seed": seed,
                "final_order_mean": float(np.mean(orders)) if orders else float("nan"),
                "final_order_min": float(np.min(orders)) if orders else float("nan"),
                "final_order_max": float(np.max(orders)) if orders else float("nan"),
                "final_unique_patterns": unique,
                "final_live_n": len(live),
            })

        results[prior] = {"runs": prior_runs}

    if save_csv:
        _write_csv(
            f"{output_dir}/run_summary.csv",
            ["prior", "run", "seed", "final_order_mean", "final_order_min", "final_order_max",
             "final_unique_patterns", "final_live_n"],
            run_summary_rows
        )

    # Aggregate curves into uncertainty bands per prior
    summary_all_rows: list[dict] = []
    for prior in priors:
        runs = results[prior]["runs"]
        if not runs:
            continue

        # Align by iteration index (all runs share n_live/n_iterations).
        n_points = min(len(r["dead"]) for r in runs)
        log_X = np.array([runs[0]["dead"][i].get("log_X", float("nan")) for i in range(n_points)], dtype=float)

        order_runs = np.array([[r["dead"][i].get("order", float("nan")) for i in range(n_points)] for r in runs], dtype=float)
        sym_runs = np.array([[r["dead"][i].get("symmetry", float("nan")) for i in range(n_points)] for r in runs], dtype=float)
        ising_runs = np.array([[r["dead"][i].get("ising", float("nan")) for i in range(n_points)] for r in runs], dtype=float)

        def _summarize(arr: np.ndarray) -> dict[str, np.ndarray]:
            q16, q50, q84 = np.quantile(arr, [0.16, 0.50, 0.84], axis=0)
            std = np.std(arr, axis=0, ddof=1) if arr.shape[0] > 1 else np.zeros(arr.shape[1])
            return {
                "mean": np.mean(arr, axis=0),
                "std": std,
                "q16": q16,
                "q50": q50,
                "q84": q84,
            }

        order_stats = _summarize(order_runs)
        sym_stats = _summarize(sym_runs)
        ising_stats = _summarize(ising_runs)

        prior_summary_rows: list[dict] = []
        for i in range(n_points):
            row = {
                "prior": prior,
                "iteration": i,
                "log_X": float(log_X[i]),
                "order_mean": float(order_stats["mean"][i]),
                "order_std": float(order_stats["std"][i]),
                "order_q16": float(order_stats["q16"][i]),
                "order_q50": float(order_stats["q50"][i]),
                "order_q84": float(order_stats["q84"][i]),
                "symmetry_mean": float(sym_stats["mean"][i]),
                "symmetry_std": float(sym_stats["std"][i]),
                "symmetry_q16": float(sym_stats["q16"][i]),
                "symmetry_q50": float(sym_stats["q50"][i]),
                "symmetry_q84": float(sym_stats["q84"][i]),
                "ising_mean": float(ising_stats["mean"][i]),
                "ising_std": float(ising_stats["std"][i]),
                "ising_q16": float(ising_stats["q16"][i]),
                "ising_q50": float(ising_stats["q50"][i]),
                "ising_q84": float(ising_stats["q84"][i]),
                "n_runs": len(runs),
            }
            prior_summary_rows.append(row)
            summary_all_rows.append(row)

        if save_csv:
            _write_csv(
                f"{output_dir}/{prior}/curve_summary.csv",
                ["prior", "iteration", "log_X",
                 "order_mean", "order_std", "order_q16", "order_q50", "order_q84",
                 "symmetry_mean", "symmetry_std", "symmetry_q16", "symmetry_q50", "symmetry_q84",
                 "ising_mean", "ising_std", "ising_q16", "ising_q50", "ising_q84",
                 "n_runs"],
                prior_summary_rows
            )

    if save_csv:
        _write_csv(
            f"{output_dir}/curve_summary_all.csv",
            ["prior", "iteration", "log_X",
             "order_mean", "order_std", "order_q16", "order_q50", "order_q84",
             "symmetry_mean", "symmetry_std", "symmetry_q16", "symmetry_q50", "symmetry_q84",
             "ising_mean", "ising_std", "ising_q16", "ising_q50", "ising_q84",
             "n_runs"],
            summary_all_rows
        )

    # Compare rarity curves
    print("\n" + "=" * 70)
    print("COMPARISON (median ± 1σ band): Order at different log(X)")
    print("=" * 70)

    log_X_targets = [0, -1, -2, -4, -6, -8]

    header = f"{'log(X)':>8}"
    for prior in priors:
        header += f" | {prior:>10}"
    print(header)
    print("-" * (10 + 13 * len(priors)))

    for target in log_X_targets:
        row = f"{target:>8.1f}"
        for prior in priors:
            runs = results.get(prior, {}).get("runs", [])
            if not runs:
                row += f" | {'N/A':>10}"
                continue
            vals = []
            for r in runs:
                dead = r["dead"]
                idx = min(range(len(dead)), key=lambda i: abs(dead[i].get("log_X", 0) - target))
                vals.append(dead[idx].get("order", 0.0))
            q16, q50, q84 = np.quantile(vals, [0.16, 0.50, 0.84])
            row += f" | {q50:>6.3f}[{q16:>4.2f},{q84:>4.2f}]"
        print(row)

    # Compare Ising (secondary metric)
    print("\n" + "=" * 70)
    print("ISING (median ± 1σ band) at different log(X)")
    print("=" * 70)

    header = f"{'log(X)':>8}"
    for prior in priors:
        header += f" | {prior:>10}"
    print(header)
    print("-" * (10 + 13 * len(priors)))

    for target in log_X_targets:
        row = f"{target:>8.1f}"
        for prior in priors:
            runs = results.get(prior, {}).get("runs", [])
            if not runs:
                row += f" | {'N/A':>10}"
                continue
            vals = []
            for r in runs:
                dead = r["dead"]
                idx = min(range(len(dead)), key=lambda i: abs(dead[i].get("log_X", 0) - target))
                vals.append(dead[idx].get("ising", 0.0))
            q16, q50, q84 = np.quantile(vals, [0.16, 0.50, 0.84])
            row += f" | {q50:>6.3f}[{q16:>4.2f},{q84:>4.2f}]"
        print(row)

    # Symmetry comparison
    print("\n" + "=" * 70)
    print("SYMMETRY (median ± 1σ band) at different log(X)")
    print("=" * 70)

    header = f"{'log(X)':>8}"
    for prior in priors:
        header += f" | {prior:>10}"
    print(header)
    print("-" * (10 + 13 * len(priors)))

    for target in log_X_targets:
        row = f"{target:>8.1f}"
        for prior in priors:
            runs = results.get(prior, {}).get("runs", [])
            if not runs:
                row += f" | {'N/A':>10}"
                continue
            vals = []
            for r in runs:
                dead = r["dead"]
                idx = min(range(len(dead)), key=lambda i: abs(dead[i].get("log_X", 0) - target))
                vals.append(dead[idx].get("symmetry", 0.0))
            q16, q50, q84 = np.quantile(vals, [0.16, 0.50, 0.84])
            row += f" | {q50:>6.3f}[{q16:>4.2f},{q84:>4.2f}]"
        print(row)

    # Show best images from each prior
    print("\n" + "=" * 70)
    print("BEST IMAGES FROM EACH PRIOR")
    print("=" * 70)

    for prior in priors:
        runs = results.get(prior, {}).get("runs", [])
        if not runs:
            continue
        best = None
        for r in runs:
            live = r["live"]
            if not live:
                continue
            candidate = max(live, key=lambda x: x.get("order", float("-inf")))
            if best is None or candidate.get("order", float("-inf")) > best.get("order", float("-inf")):
                best = candidate
        if best is None:
            continue
        print(f"\n{prior.upper()} (order={best['order']:.4f}):")
        print(img_to_ascii(best['image'], width=32))
        save_image_pbm(best['image'], f"{output_dir}/{prior}_best.pbm")

    # Summary statistics
    print("\n" + "=" * 70)
    print("SUMMARY: Final live point statistics")
    print("=" * 70)

    for prior in priors:
        runs = results.get(prior, {}).get("runs", [])
        if not runs:
            continue
        final_means = []
        final_uniques = []
        final_maxes = []
        final_mins = []
        for r in runs:
            live = r["live"]
            if not live:
                continue
            orders = [lp.get("order", 0.0) for lp in live]
            imgs = [lp.get("image") for lp in live if lp.get("image") is not None]
            unique = len(set(img.tobytes() for img in imgs)) if imgs else 0
            final_means.append(float(np.mean(orders)))
            final_maxes.append(float(np.max(orders)))
            final_mins.append(float(np.min(orders)))
            final_uniques.append(float(unique))

        print(f"\n{prior.upper()}:")
        if final_means:
            m16, m50, m84 = np.quantile(final_means, [0.16, 0.50, 0.84])
            u16, u50, u84 = np.quantile(final_uniques, [0.16, 0.50, 0.84])
            print(f"  Final mean order: {m50:.4f} [{m16:.4f}, {m84:.4f}]")
            print(f"  Final order range (median of mins/maxes): "
                  f"[{np.median(final_mins):.4f}, {np.median(final_maxes):.4f}]")
            print(f"  Unique patterns (median): {int(u50)}/{n_live}")
        else:
            print("  (no results)")

    # Sampler mixing check: warn if variance across runs is high
    if n_runs > 1:
        print("\n" + "=" * 70)
        print("SAMPLER MIXING CHECK")
        print("=" * 70)
        threshold_check = 0.1  # Check bits to reach order=0.1
        mixing_ok = True

        for prior in priors:
            runs = results.get(prior, {}).get("runs", [])
            if len(runs) < 2:
                continue

            # Compute bits to threshold for each run
            bits_per_run = []
            for r in runs:
                dead = r["dead"]
                bits = None
                for d in dead:
                    if d.get("order", 0) >= threshold_check:
                        bits = -d.get("log_X", 0) / np.log(2)
                        break
                if bits is None:
                    # Threshold not reached - use lower bound
                    bits = len(dead) / (n_live * np.log(2))
                bits_per_run.append(bits)

            if len(bits_per_run) >= 2:
                bits_std = np.std(bits_per_run, ddof=1)
                bits_mean = np.mean(bits_per_run)

                if bits_std > 2.0:
                    mixing_ok = False
                    print(f"  WARNING: {prior} has high variance in bits@{threshold_check}")
                    print(f"           Mean={bits_mean:.1f}, Std={bits_std:.1f} bits")
                    print(f"           Sampler may not be mixing across modes!")
                else:
                    print(f"  {prior}: bits@{threshold_check} variance OK (std={bits_std:.2f})")

        if mixing_ok:
            print("  All priors show consistent results across runs.")

    return results


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    import sys

    mode = sys.argv[1] if len(sys.argv) > 1 else "measure"

    def _get_flag_value(prefix: str) -> Optional[str]:
        for arg in sys.argv:
            if arg.startswith(prefix):
                return arg.split("=", 1)[1]
        return None

    # Check for metric flag (--metric=maze or similar)
    metric_name = "multiplicative"
    for arg in sys.argv:
        if arg.startswith("--metric="):
            metric_name = arg.split("=")[1]
        elif arg in ORDER_METRICS:
            metric_name = arg

    order_fn = ORDER_METRICS.get(metric_name, order_multiplicative)

    if mode == "compare":
        # Compare priors (with optional multi-run uncertainty).
        # With n_live=50, n_iterations=600 gives log(X) = -12.
        n_runs = int(_get_flag_value("--runs=") or "1")
        base_seed_val = _get_flag_value("--seed=")
        base_seed = int(base_seed_val) if base_seed_val is not None else None
        n_live = int(_get_flag_value("--n-live=") or "50")
        n_iterations = int(_get_flag_value("--n-iter=") or "600")
        size = int(_get_flag_value("--size=") or "32")
        out_dir = _get_flag_value("--out=") or f"prior_comparison_{metric_name}"

        compare_priors(
            priors=["cppn", "uniform", "dsl"],
            n_live=n_live,
            n_iterations=n_iterations,
            image_size=size,
            output_dir=out_dir,
            order_fn=order_fn,
            n_runs=n_runs,
            base_seed=base_seed,
            save_csv=True,
            verbose=True
        )
    elif mode in ("measure", "illumination"):
        # Single run with specified sampling mode
        n_live = int(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[2].isdigit() else 100
        n_iter = int(sys.argv[3]) if len(sys.argv) > 3 and sys.argv[3].isdigit() else 1000
        size = int(sys.argv[4]) if len(sys.argv) > 4 and sys.argv[4].isdigit() else 32
        seed_val = _get_flag_value("--seed=")
        seed = int(seed_val) if seed_val is not None else None

        print(f"Using order metric: {metric_name}")

        nested_sampling_v3(
            n_live=n_live,
            n_iterations=n_iter,
            image_size=size,
            order_fn=order_fn,
            sampling_mode=mode,
            track_metrics=True,
            output_dir=f"thermo_output_{mode}_{metric_name}",
            seed=seed
        )
    else:
        print("Usage:")
        print("  python thermo_sampler_v3.py measure [n_live] [n_iter] [size] [--metric=NAME]")
        print("  python thermo_sampler_v3.py illumination [n_live] [n_iter] [size] [--metric=NAME]")
        print("  python thermo_sampler_v3.py compare [--metric=NAME] [--runs=N] [--seed=S] [--n-live=N] [--n-iter=N] [--size=N] [--out=DIR]")
        print()
        print("Modes:")
        print("  measure      - Calibrated rarity curves (random seed selection)")
        print("  illumination - Discovery mode (diversity-biased seed selection)")
        print("  compare      - Compare CPPN, uniform, and DSL priors")
        print()
        print("Reproducibility:")
        print("  --seed=S     - Sets the RNG seed (base seed for compare mode runs)")
        print("  --runs=N     - Number of independent runs per prior (compare mode)")
        print()
        print("Metrics:")
        for name in ORDER_METRICS:
            print(f"  {name}")
        print()
        print("Examples:")
        print("  python thermo_sampler_v3.py compare --metric=maze --runs=10 --seed=0")
        print("  python thermo_sampler_v3.py measure 100 1000 32 --metric=maze --seed=0")
