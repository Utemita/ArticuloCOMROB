"""
simplified_model.py
===================
SIMPLIFIED kinematic model of the exoskeleton -- only computes the trajectory
up to the MEDIAL PHALANX (DIP joint inclusive). This validates that both
optimization algorithms (DE vs GA) work well on a smaller problem before
moving to the full 3-phalanx model.

Differences from the full model (exo_18_pinza_fina.py):
  * The distal phalanx and the entire third 4-bar mechanism
    (Link9_3, Link10_3 and the dorsal support back3_3/up3_3) are removed.
  * The distal offset angle theta_aux_fd is removed (no fingertip to place).
  * Objectives are reduced to two trajectories: PIP and DIP.
  * The design vector shrinks from 21 to 16 parameters.

Improvements introduced for the paper:
  * REDUCED DIMENSIONS: link lengths capped at max 60 mm (previously 80 mm)
    so the mechanism is more compact and manufacturable.
  * REDUCED MEDIAL AUXILIARY ANGLE (c2): theta_aux_fm now has an upper limit
    of 70 deg (previously 180 deg), with a soft regularization that favors
    small values WITHOUT degrading the shape fit.
  * KINEMATIC VALIDATION: validate_kinematics() checks that parameters are
    feasible and the mechanism assembles without NaN or reversals over the
    full crank sweep.

Note: the bounds and the fitness_function are IDENTICAL for DE and GA, so
the comparison measures the algorithm, not the problem formulation.
"""
import os
import numpy as np

import common
from common import (FP_REAL, FM_REAL, solve_five_bar, solve_four_bar,
                    chamfer_distance, optimal_rigid_transform, apply_transform,
                    monotonicity_penalty)

# --- MOCAP data (only PIP and DIP are used in this version) ---
_MOCAP = common.load_mocap(os.path.join(os.path.dirname(__file__),
                                        "mocap_pinza_fina_120pts.csv"))
mocap_pts = {'ifp': _MOCAP['ifp'], 'ifd': _MOCAP['ifd']}
theta_input = _MOCAP['theta_input']
N_POINTS = _MOCAP['n']

# ==============================================================================
# NAMES OF THE 16 DESIGN PARAMETERS (simplified version)
# ==============================================================================
NAMES = [
    "Ground1 (m)", "Ground2 (m)",
    "Link1 (m)", "Link2 (m)", "Link3 (m)", "Link4 (m)", "Link5 (m)",
    "Link6 (m)", "Link7 (m)", "Link8 (m)", "Link10 / c2 (m)",
    "hsp (m)", "dsp (m)",
    "Theta Aux FM (rad)", "Gear ratio", "Theta Offset (rad)",
]

# ==============================================================================
# --- BOUNDS with reduced dimensions and auxiliary angle ---
# Link lengths: max 60 mm (previously 80). theta_aux_fm: max 70 deg (previously 180).
# ==============================================================================
bounds = [
    (0.015, 0.060), (0.015, 0.060),                  # Ground1, Ground2
    (0.010, 0.060), (0.010, 0.060), (0.010, 0.060),  # Link1, Link2, Link3
    (0.010, 0.060), (0.010, 0.060),                  # Link4, Link5
    (0.010, 0.060), (0.010, 0.060), (0.010, 0.060),  # Link6, Link7, Link8
    (0.010, 0.055),                                  # Link10 / c2
    (0.005, 0.030), (0.005, 0.023),                  # hsp, dsp
    (0.0, np.deg2rad(70.0)),                         # theta_aux_fm (REDUCED)
    (1.0, 8.0),                                      # gear_ratio
    (-np.pi, np.pi),                                 # theta_offset
]

# Objective weights (shape + monotonicity + soft regularization)
W_IFP = 0.40
W_IFD = 0.60
W_MONO = 5.0
# Very soft regularization: pushes toward smaller dimensions and auxiliary angle
# without degrading the fit (chamfer ~ 3e-3 m; these terms ~ 1e-5).
W_REG_DIM = float(os.environ.get('W_REG_DIM', '5e-4'))   # length scale
W_REG_AUX = float(os.environ.get('W_REG_AUX', '5e-4'))   # auxiliary angle scale


# ==============================================================================
# --- SIMPLIFIED KINEMATIC MODEL (up to DIP / medial phalanx) ---
# ==============================================================================
def run_kinematics(p, th_input):
    """Forward kinematics of the exo up to the medial phalanx tip (DIP).

    Returns a dict with trajectories 'ifp' and 'ifd' (Nx2, m) and the
    angular profile 'theta_fm' (rad). If the mechanism fails at any point
    in the sweep (invalid lengths, non-assembly, or reversal), returns None.
    """
    (Bancada1, Bancada2, Link1, Link2, Link3, Link4, Link5,
     Link6, Link7, Link8, Link10, hsp, dsp,
     theta_aux_fm, gear_ratio, theta_offset) = p

    # Check that parameters make sense
    if gear_ratio <= 0:
        return None
    if min(p[:11]) <= 0.005:           # minimum link lengths
        return None
    if hsp <= 0 or dsp <= 0:
        return None
    if FP_REAL - 2.0 * dsp <= 0.001:
        return None

    r3_val = Bancada1 / 2.0
    theta14B = np.pi / 2.0
    c1 = np.sqrt(hsp**2 + dsp**2)
    rs2 = np.sqrt(hsp**2 + (FP_REAL - dsp)**2)
    theta_aux_s2 = np.arctan2(hsp, FP_REAL - dsp)

    PXifp, PYifp, PXifd, PYifd, TH_FM = [], [], [], [], []
    prev_theta_fm = None

    for th2 in th_input:
        th1 = (th2 / gear_ratio) + theta_offset

        # --- Mechanism 1: 5-bar ---
        res5 = solve_five_bar(Link4, Link3, r3_val, Link1, Link2, th1, th2)
        if res5 is None:
            return None
        pxP, pyP = res5

        # --- Mechanism 1: 4-bar ---
        theta4 = solve_four_bar(Link4, Link5, c1, Bancada2, th1, theta14B)
        if theta4 is None:
            return None

        theta_fp = theta4 + np.arctan2(hsp, dsp)
        px_ifp = FP_REAL * np.cos(theta_fp) - Bancada2 * np.cos(theta14B) - r3_val
        py_ifp = FP_REAL * np.sin(theta_fp) - Bancada2 * np.sin(theta14B)

        pxs1 = c1 * np.cos(theta4) - Bancada2 * np.cos(theta14B) - r3_val
        pys1 = c1 * np.sin(theta4) - Bancada2 * np.sin(theta14B)

        theta_ps2 = theta_fp - theta_aux_s2
        pxs2 = rs2 * np.cos(theta_ps2) - Bancada2 * np.cos(theta14B) - r3_val
        pys2 = rs2 * np.sin(theta_ps2) - Bancada2 * np.sin(theta14B)

        pxm4 = Link4 * np.cos(th2) - r3_val
        pym4 = Link4 * np.sin(th2)

        # --- Mechanism 2: 5-bar (secondary reference) ---
        theta_roll = np.arctan2(pym4 - pys1, pxm4 - pxs1)
        theta1m2 = np.arctan2(pys2 - pys1, pxs2 - pxs1) - theta_roll
        theta2m2 = np.arctan2(pyP - pym4, pxP - pxm4) - theta_roll

        res5_2 = solve_five_bar(FP_REAL - 2.0 * dsp, Link7, Link5 / 2.0, Link3, Link6,
                                theta1m2, theta2m2)
        if res5_2 is None:
            return None
        px_local, py_local = res5_2

        mag = np.sqrt(px_local**2 + py_local**2)
        theta_loc = np.arctan2(py_local, px_local)
        px_aux = (pxs1 + pxm4) / 2.0
        py_aux = (pys1 + pym4) / 2.0
        pxP2 = mag * np.cos(theta_loc + theta_roll) + px_aux
        pyP2 = mag * np.sin(theta_loc + theta_roll) + py_aux

        # --- Mechanism 2: 4-bar ---
        theta1m42 = np.arctan2(pys2 - py_ifp, pxs2 - px_ifp)
        theta2m42 = np.arctan2(pyP2 - pys2, pxP2 - pxs2)

        theta4m2 = solve_four_bar(Link7, Link8, Link10, c1, theta2m42, theta1m42)
        if theta4m2 is None:
            return None

        # Medial phalanx angle (Link10/c2 + theta_aux_fm)
        theta_fm = theta4m2 + theta_aux_fm

        # Anti-hook constraint: theta_fm must be monotonically increasing
        if prev_theta_fm is not None:
            delta_fm = (theta_fm - prev_theta_fm + np.pi) % (2 * np.pi) - np.pi
            if delta_fm < -np.deg2rad(0.5):
                return None  # reversing, not valid
        prev_theta_fm = theta_fm

        # DIP position (end of the medial phalanx)
        px_ifd = FM_REAL * np.cos(theta_fm) + px_ifp
        py_ifd = FM_REAL * np.sin(theta_fm) + py_ifp

        # Check for out-of-range or NaN values
        if (np.abs(px_ifd) > 0.3 or np.abs(py_ifd) > 0.3
                or not np.isfinite(px_ifd) or not np.isfinite(py_ifd)):
            return None

        PXifp.append(px_ifp); PYifp.append(py_ifp)
        PXifd.append(px_ifd); PYifd.append(py_ifd)
        TH_FM.append(theta_fm)

    return {
        'ifp': np.column_stack((PXifp, PYifp)),
        'ifd': np.column_stack((PXifd, PYifd)),
        'theta_fm': np.asarray(TH_FM),
    }


# ==============================================================================
# --- OBJECTIVE FUNCTION (identical for DE and GA) ---
# ==============================================================================
def fitness_function(p):
    sim = run_kinematics(p, theta_input)
    if sim is None:
        return 1000.0  # penalty for invalid mechanisms

    all_mocap = np.vstack([mocap_pts['ifp'], mocap_pts['ifd']])
    all_sim = np.vstack([sim['ifp'], sim['ifd']])
    R, t = optimal_rigid_transform(all_mocap, all_sim)
    aligned = {k: apply_transform(sim[k], R, t) for k in ('ifp', 'ifd')}

    err_ifp = chamfer_distance(mocap_pts['ifp'], aligned['ifp'])
    err_ifd = chamfer_distance(mocap_pts['ifd'], aligned['ifd'])
    mono_ifd = monotonicity_penalty(aligned['ifd'])

    shape_error = W_IFP * err_ifp + W_IFD * err_ifd
    mono_error = W_MONO * mono_ifd

    # Soft regularization: pushes toward smaller dimensions and auxiliary angle
    link_sum = float(np.sum(p[:11]))                  # 11 link lengths (m)
    reg_dim = W_REG_DIM * link_sum
    reg_aux = W_REG_AUX * (p[13] / np.pi)             # theta_aux_fm normalized

    return shape_error + mono_error + reg_dim + reg_aux


# ==============================================================================
# --- EVALUATION / METRICS IN mm (for reports) ---
# ==============================================================================
def evaluate(p):
    """Compute errors in mm and aligned data. Returns None if kinematics fail."""
    sim = run_kinematics(p, theta_input)
    if sim is None:
        return None
    all_mocap = np.vstack([mocap_pts['ifp'], mocap_pts['ifd']])
    all_sim = np.vstack([sim['ifp'], sim['ifd']])
    R, t = optimal_rigid_transform(all_mocap, all_sim)
    aligned = {k: apply_transform(sim[k], R, t) for k in ('ifp', 'ifd')}
    err_ifp = chamfer_distance(mocap_pts['ifp'], aligned['ifp']) * 1000
    err_ifd = chamfer_distance(mocap_pts['ifd'], aligned['ifd']) * 1000
    return {
        'err_ifp_mm': err_ifp,
        'err_ifd_mm': err_ifd,
        'err_global_mm': (err_ifp + err_ifd) / 2.0,
        'R': R, 't': t,
        'aligned': aligned,
        'sim': sim,
        'mounting_angle_deg': np.rad2deg(np.arctan2(R[1, 0], R[0, 0])),
    }


# ==============================================================================
# --- KINEMATIC VALIDATION (checks mechanism feasibility) ---
# ==============================================================================
def validate_kinematics(p, verbose=False):
    """Check that the parameters produce a feasible mechanism over the full
    crank sweep: assembles correctly, no NaN/Inf, and no medial phalanx
    reversal.

    Returns (ok: bool, info: dict).
    """
    info = {}
    sim = run_kinematics(p, theta_input)
    if sim is None:
        info['reason'] = ("Kinematics fail at some point in the sweep "
                          "(non-assembly, invalid length, or medial phalanx "
                          "reversal).")
        if verbose:
            print(">> KINEMATIC VALIDATION: FAILED ->", info['reason'])
        return False, info

    # Check for NaN or Inf
    for k in ('ifp', 'ifd'):
        if not np.all(np.isfinite(sim[k])):
            info['reason'] = f"Non-finite values in trajectory {k}."
            if verbose:
                print(">> KINEMATIC VALIDATION: FAILED ->", info['reason'])
            return False, info

    # Ensure the full sweep was solved
    n_ok = len(sim['ifp'])
    if n_ok != N_POINTS:
        info['reason'] = (f"Only {n_ok}/{N_POINTS} poses were solved in "
                          f"the sweep.")
        if verbose:
            print(">> KINEMATIC VALIDATION: FAILED ->", info['reason'])
        return False, info

    # Actual monotonicity of theta_fm (no reversals)
    th_fm = np.unwrap(sim['theta_fm'])
    delta = np.diff(th_fm)
    info['theta_fm_monotonic'] = bool(np.all(delta >= -np.deg2rad(0.5)))
    info['rom_medial_deg'] = float(np.rad2deg(th_fm[-1] - th_fm[0]))
    info['poses_solved'] = n_ok
    info['reason'] = "OK: mechanism is feasible over the full range of motion."
    if verbose:
        print(f">> KINEMATIC VALIDATION: OK  "
              f"({n_ok}/{N_POINTS} poses, medial ROM "
              f"{info['rom_medial_deg']:.1f} deg)")
    return True, info
