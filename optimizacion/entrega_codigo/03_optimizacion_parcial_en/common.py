"""
common.py
=========
Shared utilities for the exoskeleton optimization experiments (COMROB paper:
comparison of Genetic Algorithms vs Differential Evolution for kinematic
synthesis of a finger rehabilitation exoskeleton).

What's in here:
  1. MOCAP data loading and preprocessing (mocap_pinza_fina_120pts.csv).
  2. Shape metrics (bidirectional Chamfer distance) and monotonicity penalty.
  3. Optimal 2D rigid alignment (Kabsch/Procrustes, no scale).
  4. Basic kinematic solvers (5-bar, 4-bar, circle-circle intersection)
     used by both the simplified and full models.
  5. A self-contained real-coded GENETIC ALGORITHM (no external libs) with
     an interface similar to scipy.optimize.differential_evolution so we can
     compare both approaches on the exact same objective function.

Note: all lengths in meters, angles in radians unless stated otherwise.
"""
import os
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.signal import savgol_filter

# ==============================================================================
# --- 1. ANTHROPOMETRIC PARAMETERS (real finger phalanx lengths) ---
# ==============================================================================
FP_REAL = 0.049   # Proximal phalanx (m)
FM_REAL = 0.026   # Medial phalanx   (m)
FD_REAL = 0.024   # Distal phalanx   (m)


# ==============================================================================
# --- 2. MOCAP DATA LOADING AND PREPROCESSING ---
# ==============================================================================
def load_mocap(path="mocap_pinza_fina_120pts.csv", input_range_deg=85.0):
    """Load and preprocess motion capture data.

    Same pipeline as the original optimizer:
      (a) Reverse the trajectory (from max flexion -> extension to
          open -> close grasp motion).
      (b) Clip DIP to >= 0 degrees (no hyperextension in the exo).
      (c) Savitzky-Golay smoothing (window 15, order 2) to reduce noise.

    Returns a dict with:
      - 'ifp','ifd','tip' : Nx2 joint position arrays (m).
      - 'theta_fm','theta_fd' : medial and distal absolute angles (rad).
      - 'dip_rel' : relative DIP angle profile (rad).
      - 'theta_input' : main crank sweep (rad).
      - 'n' : number of data points.
    """
    data = pd.read_csv(path)

    # (a) Reverse: now goes from open -> close (grasping motion)
    mcp_raw = data['Theta_MCP'].values[::-1]
    pip_raw = data['Theta_PIP'].values[::-1]
    dip_raw = data['Theta_DIP'].values[::-1]
    n = len(mcp_raw)

    # (b) Clip DIP to 0 degrees minimum (no hyperextension)
    dip_raw = np.clip(dip_raw, 0.0, None)

    # (c) Savitzky-Golay smoothing to remove mocap noise
    win = 15
    if n >= win:
        mcp = savgol_filter(np.deg2rad(mcp_raw), window_length=win, polyorder=2)
        pip = savgol_filter(np.deg2rad(pip_raw), window_length=win, polyorder=2)
        dip = savgol_filter(np.deg2rad(dip_raw), window_length=win, polyorder=2)
        dip = np.clip(dip, 0.0, None)
    else:
        mcp = np.deg2rad(mcp_raw)
        pip = np.deg2rad(pip_raw)
        dip = np.deg2rad(dip_raw)

    # Forward kinematics of the finger (rigid body chain from the MCP joint)
    seg_prox = mcp
    seg_med = mcp + pip
    seg_dist = mcp + pip + dip

    pxIFP = FP_REAL * np.cos(seg_prox)
    pyIFP = FP_REAL * np.sin(seg_prox)
    pxIFD = pxIFP + FM_REAL * np.cos(seg_med)
    pyIFD = pyIFP + FM_REAL * np.sin(seg_med)
    pxPF = pxIFD + FD_REAL * np.cos(seg_dist)
    pyPF = pyIFD + FD_REAL * np.sin(seg_dist)

    return {
        'ifp': np.column_stack((pxIFP, pyIFP)),
        'ifd': np.column_stack((pxIFD, pyIFD)),
        'tip': np.column_stack((pxPF, pyPF)),
        'theta_fm': seg_med,
        'theta_fd': seg_dist,
        'dip_rel': dip - dip[0],
        'theta_input': np.linspace(0, np.deg2rad(input_range_deg), n),
        'n': n,
    }


# ==============================================================================
# --- 3. METRICS ---
# ==============================================================================
def chamfer_distance(curve_target, curve_sim):
    """Bidirectional Chamfer distance -- our shape error metric."""
    dists = cdist(curve_target, curve_sim)
    return np.mean(np.min(dists, axis=1)) + np.mean(np.min(dists, axis=0))


def optimal_rigid_transform(target, sim):
    """Optimal rigid transform (R + t) between two 2D point clouds."""
    c_t = np.mean(target, axis=0)
    c_s = np.mean(sim, axis=0)
    H = (sim - c_s).T @ (target - c_t)
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[1, :] *= -1
        R = Vt.T @ U.T
    t = c_t - R @ c_s
    return R, t


def apply_transform(points, R, t):
    return (R @ points.T).T + t


def monotonicity_penalty(curve):
    """Penalizes direction reversals in the simulated trajectory."""
    diffs = np.diff(curve, axis=0)
    arc = np.linalg.norm(diffs, axis=1)
    total = np.sum(arc)
    if total < 1e-9:
        return 0.0
    mean_dir = np.sum(diffs, axis=0) / (total + 1e-12)
    mean_dir /= (np.linalg.norm(mean_dir) + 1e-12)
    proj = diffs @ mean_dir
    return np.sum(np.clip(-proj, 0, None))


# ==============================================================================
# --- 4. BASIC KINEMATIC SOLVERS ---
# ==============================================================================
def solve_five_bar(r1, r2, r3, r4, r5, theta1, theta2):
    """Solve the coupler position of a 5-bar mechanism.
    Returns None if the mechanism cannot assemble at this configuration.
    """
    den = r4 * np.cos(theta2) - r1 * np.cos(theta1) + 2 * r3
    if np.abs(den) < 1e-4:
        return None
    e = (r1 * np.sin(theta1) - r4 * np.sin(theta2)) / den
    f = (2 * (r1 * r3 * np.cos(theta1) + r3 * r4 * np.cos(theta2))
         - r1**2 + r2**2 + r4**2 - r5**2) / (2 * den)
    d_ = e**2 + 1
    g = 2 * (e * f - e * r1 * np.cos(theta1) + e * r3 - r1 * np.sin(theta1))
    h = (f**2 - 2 * f * (r1 * np.cos(theta1) - r3)
         - 2 * r1 * r3 * np.cos(theta1) + r1**2 + r3**2 - r2**2)
    disc = g**2 - 4 * d_ * h
    if disc < 0:
        return None  # no assembly
    py = (-g + np.sqrt(disc)) / (2 * d_)
    px = e * py + f
    return px, py


def solve_four_bar(a, b, c, d, theta2, theta1):
    """Solve the rocker angle of a 4-bar mechanism (open branch).
    Returns None if no real solution exists.
    """
    k1 = a * np.cos(theta2) + d * np.cos(theta1)
    k2 = a * np.sin(theta2) + d * np.sin(theta1)
    k3 = k1**2 + k2**2 + c**2 - b**2
    A1 = -2 * k1 * c - k3
    B1 = 4 * k2 * c
    C1 = 2 * k1 * c - k3
    disc = B1**2 - 4 * A1 * C1
    if disc < 0:
        return None  # no real solution
    return 2 * np.arctan((-B1 - np.sqrt(disc)) / (2 * A1))


def circle_intersections(c0, r0, c1, r1):
    """Intersection of two circles. Returns None if they don't intersect."""
    c0 = np.asarray(c0, dtype=float)
    c1 = np.asarray(c1, dtype=float)
    dvec = c1 - c0
    dist = np.hypot(dvec[0], dvec[1])
    if dist > (r0 + r1) or dist < abs(r0 - r1) or dist == 0:
        return None  # no intersection
    aa = (r0**2 - r1**2 + dist**2) / (2 * dist)
    hh2 = r0**2 - aa**2
    if hh2 < 0:
        return None
    hh = np.sqrt(hh2)
    pm = c0 + aa * dvec / dist
    perp = np.array([-dvec[1], dvec[0]]) / dist
    return pm + hh * perp, pm - hh * perp


# ==============================================================================
# --- 5. REAL-CODED GENETIC ALGORITHM ---
# ==============================================================================
class OptimResult:
    """Result container compatible with scipy.optimize (has .x and .fun)."""

    def __init__(self, x, fun, nit, nfev, history):
        self.x = np.asarray(x)
        self.fun = float(fun)
        self.nit = int(nit)
        self.nfev = int(nfev)
        self.history = list(history)  # best fitness per generation

    def __repr__(self):
        return (f"OptimResult(fun={self.fun:.6f}, nit={self.nit}, "
                f"nfev={self.nfev})")


def _sbx_crossover_vectorized(parents1, parents2, lower, upper, eta_c, rng):
    """Vectorized Simulated Binary Crossover (SBX) over batches of pairs."""
    m, dim = parents1.shape
    h1 = parents1.copy()
    h2 = parents2.copy()

    # Crossover applied to 50% of genes where parents differ
    do_cross = (rng.random((m, dim)) <= 0.5) & (np.abs(parents1 - parents2) > 1e-14)

    x1 = np.minimum(parents1, parents2)
    x2 = np.maximum(parents1, parents2)
    dx = np.where(x2 - x1 > 1e-14, x2 - x1, 1e-14)
    u = rng.random((m, dim))
    pexp = 1.0 / (eta_c + 1.0)

    # Child 1 (lower side)
    beta1 = 1.0 + 2.0 * (x1 - lower) / dx
    alpha1 = 2.0 - beta1 ** (-(eta_c + 1.0))
    betaq1 = np.where(u <= 1.0 / alpha1,
                      (u * alpha1) ** pexp,
                      (1.0 / (2.0 - u * alpha1)) ** pexp)
    c1 = 0.5 * ((x1 + x2) - betaq1 * dx)

    # Child 2 (upper side)
    beta2 = 1.0 + 2.0 * (upper - x2) / dx
    alpha2 = 2.0 - beta2 ** (-(eta_c + 1.0))
    betaq2 = np.where(u <= 1.0 / alpha2,
                      (u * alpha2) ** pexp,
                      (1.0 / (2.0 - u * alpha2)) ** pexp)
    c2 = 0.5 * ((x1 + x2) + betaq2 * dx)

    c1 = np.clip(c1, lower, upper)
    c2 = np.clip(c2, lower, upper)
    h1 = np.where(do_cross, c1, h1)
    h2 = np.where(do_cross, c2, h2)
    return h1, h2


def _polynomial_mutation_vectorized(pop, lower, upper, span, eta_m, mutpb, rng):
    """Vectorized polynomial mutation over the entire population."""
    y = pop.copy()
    do_mutate = rng.random(pop.shape) <= mutpb
    span_safe = np.where(span > 0, span, 1.0)

    delta1 = (y - lower) / span_safe
    delta2 = (upper - y) / span_safe
    u = rng.random(pop.shape)
    mut_pow = 1.0 / (eta_m + 1.0)

    val_low = 2.0 * u + (1.0 - 2.0 * u) * ((1.0 - delta1) ** (eta_m + 1.0))
    deltaq_low = val_low ** mut_pow - 1.0
    val_high = 2.0 * (1.0 - u) + 2.0 * (u - 0.5) * ((1.0 - delta2) ** (eta_m + 1.0))
    deltaq_high = 1.0 - val_high ** mut_pow
    deltaq = np.where(u < 0.5, deltaq_low, deltaq_high)

    mutated = np.clip(y + deltaq * span, lower, upper)
    return np.where(do_mutate, mutated, y)


def genetic_algorithm(func, bounds, popsize=40, ngen=300,
                      cxpb=0.9, mutpb=0.15, eta_c=15.0, eta_m=20.0,
                      tournsize=3, n_elite=2, seed=None, disp=False,
                      tol=1e-8, patience=60, callback=None):
    """Real-coded Genetic Algorithm (vectorized).

    Standard operators for real-valued GA:
      - Tournament selection (tournsize competitors).
      - SBX crossover (Simulated Binary Crossover, distribution index eta_c).
      - Polynomial mutation (distribution index eta_m).
      - Elitism (top n_elite individuals pass unchanged to next generation).

    Interface is analogous to differential_evolution: takes func and bounds,
    returns an object with .x and .fun for a direct comparison.

    Parameters
    ----------
    popsize   : population size
    ngen      : maximum number of generations
    cxpb      : crossover probability per pair
    mutpb     : mutation probability per gene
    eta_c     : SBX distribution index (larger = offspring closer to parents)
    eta_m     : polynomial mutation index (larger = smaller perturbation)
    tournsize : number of competitors in tournament
    n_elite   : number of elite individuals preserved
    seed      : random seed for reproducibility
    patience  : generations without improvement before early stopping
    callback  : optional callback(gen, best_x, best_fun)
    """
    rng = np.random.default_rng(seed)
    bounds = np.asarray(bounds, dtype=float)
    lower = bounds[:, 0]
    upper = bounds[:, 1]
    dim = len(bounds)
    span = upper - lower

    # Uniform random initial population within bounds
    pop = lower + rng.random((popsize, dim)) * span
    fitness = np.array([func(ind) for ind in pop])
    nfev = popsize

    best_idx = int(np.argmin(fitness))
    best_x = pop[best_idx].copy()
    best_fun = float(fitness[best_idx])
    history = [best_fun]
    no_improve = 0

    def _tournament_selection(n):
        """Select n individuals via tournament (vectorized)."""
        candidates = rng.integers(0, popsize, size=(n, tournsize))
        fit_cand = fitness[candidates]
        winners = candidates[np.arange(n), np.argmin(fit_cand, axis=1)]
        return pop[winners]

    n_offspring = popsize - n_elite
    n_pairs = (n_offspring + 1) // 2
    gen = 0

    for gen in range(1, ngen + 1):
        # Elitism: best individuals pass unchanged
        order = np.argsort(fitness)
        elite = pop[order[:n_elite]].copy()

        # Tournament selection of parents
        parents1 = _tournament_selection(n_pairs)
        parents2 = _tournament_selection(n_pairs)

        # SBX crossover (with probability cxpb per pair)
        h1, h2 = _sbx_crossover_vectorized(parents1, parents2, lower, upper, eta_c, rng)
        no_cross = rng.random(n_pairs) > cxpb
        h1[no_cross] = parents1[no_cross]
        h2[no_cross] = parents2[no_cross]

        offspring = np.vstack([h1, h2])[:n_offspring]

        # Polynomial mutation
        offspring = _polynomial_mutation_vectorized(offspring, lower, upper, span,
                                                   eta_m, mutpb, rng)

        # New population = elite + offspring
        pop = np.vstack([elite, offspring])
        fitness = np.concatenate([
            fitness[order[:n_elite]],              # elite already evaluated
            np.array([func(ind) for ind in offspring])
        ])
        nfev += n_offspring

        gen_idx = int(np.argmin(fitness))
        gen_fun = float(fitness[gen_idx])
        if gen_fun < best_fun - tol:
            best_fun = gen_fun
            best_x = pop[gen_idx].copy()
            no_improve = 0
        else:
            no_improve += 1

        history.append(best_fun)
        if callback is not None:
            callback(gen, best_x, best_fun)
        if disp and (gen % 20 == 0 or gen == 1):
            print(f"   [GA] gen {gen:4d}/{ngen}  best fitness = {best_fun:.6f}")

        # Early stopping if no improvement
        if no_improve >= patience:
            if disp:
                print(f"   [GA] converged (no improvement in {patience} gen) "
                      f"at generation {gen}")
            break

    return OptimResult(best_x, best_fun, gen, nfev, history)
