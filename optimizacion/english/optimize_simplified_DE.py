"""
optimize_simplified_DE.py
=========================
Optimization of the SIMPLIFIED model (up to the medial phalanx, 16 parameters)
using DIFFERENTIAL EVOLUTION (scipy.optimize.differential_evolution).

This is one of the two contenders in the COMROB paper. It uses the SAME
objective function and the SAME bounds as the GA version
(optimize_simplified_GA.py), so the comparison measures the algorithm, not
the problem formulation.

Output files:
  - results/parameters_simplified_DE.txt
  - results/trajectories_simplified_DE.csv
  - results/biofidelity_simplified_DE.png
  - results/convergence_simplified_DE.csv

Budget configurable via environment: OPT_POPSIZE (15), OPT_MAXITER (300),
OPT_SEED (42).
"""
import os
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import differential_evolution

import simplified_model as M

POPSIZE = int(os.environ.get('OPT_POPSIZE', '15'))
MAXITER = int(os.environ.get('OPT_MAXITER', '300'))
SEED = int(os.environ.get('OPT_SEED', '42'))
OUTDIR = os.path.join(os.path.dirname(__file__), 'results')


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    print('=' * 64)
    print(' SIMPLIFIED OPTIMIZATION (MEDIAL PHALANX) - DIFFERENTIAL EVOLUTION')
    print('=' * 64)
    print(f'   parameters : {len(M.bounds)}   popsize: {POPSIZE}   '
          f'maxiter: {MAXITER}   seed: {SEED}')

    history = []

    def cb(xk, convergence=None):
        history.append(M.fitness_function(xk))

    t0 = time.time()
    res = differential_evolution(
        M.fitness_function, M.bounds,
        strategy='best1bin', popsize=POPSIZE,
        mutation=(0.5, 1.0), recombination=0.7,
        maxiter=MAXITER, tol=1e-7,
        polish=True, disp=True,
        updating='deferred', workers=-1, seed=SEED,
        callback=cb,
    )
    dt = time.time() - t0
    print(f'\n>> DE finished in {dt:.1f}s  (fitness={res.fun:.6f}, '
          f'nfev={res.nfev})')

    _report(res.x, dt, res.fun, history, res.nfev)


def _report(p_opt, dt, fitness, history, nfev):
    # --- Check kinematic feasibility ---
    ok, info = M.validate_kinematics(p_opt, verbose=True)

    metr = M.evaluate(p_opt)
    if metr is None:
        print('>> WARNING: the result does not produce valid kinematics.')
        return

    print('\n>> RESULTS (Differential Evolution)')
    print(f'   Global Error : {metr["err_global_mm"]:.3f} mm')
    print(f'   PIP Error    : {metr["err_ifp_mm"]:.3f} mm')
    print(f'   DIP Error    : {metr["err_ifd_mm"]:.3f} mm')
    print(f'   Medial ROM   : {info.get("rom_medial_deg", float("nan")):.2f} deg')
    print(f'   Theta Aux FM : {np.rad2deg(p_opt[13]):.2f} deg')
    link_sum_mm = float(np.sum(p_opt[:11])) * 1000
    print(f'   Total link length : {link_sum_mm:.1f} mm')

    print('\n>> OPTIMIZED PARAMETERS')
    for name, value in zip(M.NAMES, p_opt):
        print(f'   {name:26s}: {value:.6f}')

    # --- Save parameters ---
    header = ('Optimized parameters (16) - SIMPLIFIED model (medial phalanx) '
              '- DIFFERENTIAL EVOLUTION\n'
              f'fitness={fitness:.6f}  err_global_mm={metr["err_global_mm"]:.4f}  '
              f'err_ifp_mm={metr["err_ifp_mm"]:.4f}  err_ifd_mm={metr["err_ifd_mm"]:.4f}  '
              f'time_s={dt:.1f}  nfev={nfev}\n'
              + ' | '.join(M.NAMES))
    np.savetxt(os.path.join(OUTDIR, 'parameters_simplified_DE.txt'),
               p_opt, header=header)

    # --- Aligned trajectories ---
    al = metr['aligned']
    pd.DataFrame({
        'x_ifp_mocap': M.mocap_pts['ifp'][:, 0], 'y_ifp_mocap': M.mocap_pts['ifp'][:, 1],
        'x_ifp_exo': al['ifp'][:, 0], 'y_ifp_exo': al['ifp'][:, 1],
        'x_ifd_mocap': M.mocap_pts['ifd'][:, 0], 'y_ifd_mocap': M.mocap_pts['ifd'][:, 1],
        'x_ifd_exo': al['ifd'][:, 0], 'y_ifd_exo': al['ifd'][:, 1],
    }).to_csv(os.path.join(OUTDIR, 'trajectories_simplified_DE.csv'), index=False)

    # --- Convergence curve ---
    pd.DataFrame({'iteration': np.arange(len(history)),
                  'best_fitness': history}).to_csv(
        os.path.join(OUTDIR, 'convergence_simplified_DE.csv'), index=False)

    # --- Biofidelity plot ---
    fig, ax = plt.subplots(figsize=(10, 7))
    mp = M.mocap_pts
    ax.plot(mp['ifp'][:, 0]*1000, mp['ifp'][:, 1]*1000, 'r--', lw=2.5, alpha=0.65, label='MOCAP PIP')
    ax.plot(mp['ifd'][:, 0]*1000, mp['ifd'][:, 1]*1000, 'g--', lw=2.5, alpha=0.65, label='MOCAP DIP')
    ax.plot(al['ifp'][:, 0]*1000, al['ifp'][:, 1]*1000, 'r-', lw=2.5, label='EXO PIP')
    ax.plot(al['ifd'][:, 0]*1000, al['ifd'][:, 1]*1000, 'g-', lw=2.5, label='EXO DIP')
    ax.set_aspect('equal'); ax.grid(True, ls=':', alpha=0.7); ax.legend(fontsize=10)
    ax.set_title('Biofidelity - Simplified model (medial phalanx)\n'
                 f'Differential Evolution - Global error {metr["err_global_mm"]:.3f} mm',
                 fontsize=13, fontweight='bold')
    ax.set_xlabel('X axis (mm)'); ax.set_ylabel('Y axis (mm)')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, 'biofidelity_simplified_DE.png'), dpi=150)
    plt.close()

    print(f'\n>> All results saved to {OUTDIR}/')


if __name__ == '__main__':
    main()
