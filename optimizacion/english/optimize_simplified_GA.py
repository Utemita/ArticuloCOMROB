"""
optimize_simplified_GA.py
=========================
Optimization of the SIMPLIFIED model (up to the medial phalanx, 16 parameters)
using a real-coded GENETIC ALGORITHM (from common.py: tournament selection +
SBX crossover + polynomial mutation + elitism).

This is the second contender in the COMROB paper. It uses the SAME objective
function and the SAME bounds as the DE version (optimize_simplified_DE.py),
so the comparison is fair.

Output files:
  - results/parameters_simplified_GA.txt
  - results/trajectories_simplified_GA.csv
  - results/biofidelity_simplified_GA.png
  - results/convergence_simplified_GA.csv

Budget configurable via environment: GA_POPSIZE (70), GA_NGEN (300),
GA_SEED (42).
"""
import os
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

import simplified_model as M
from common import genetic_algorithm

POPSIZE = int(os.environ.get('GA_POPSIZE', '70'))
NGEN = int(os.environ.get('GA_NGEN', '300'))
SEED = int(os.environ.get('GA_SEED', '42'))
OUTDIR = os.path.join(os.path.dirname(__file__), 'results')


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    print('=' * 64)
    print(' SIMPLIFIED OPTIMIZATION (MEDIAL PHALANX) - GENETIC ALGORITHM')
    print('=' * 64)
    print(f'   parameters : {len(M.bounds)}   popsize: {POPSIZE}   '
          f'ngen: {NGEN}   seed: {SEED}')

    t0 = time.time()
    res = genetic_algorithm(
        M.fitness_function, M.bounds,
        popsize=POPSIZE, ngen=NGEN,
        cxpb=0.9, mutpb=0.15, eta_c=15.0, eta_m=20.0,
        tournsize=3, n_elite=2, seed=SEED, disp=True, patience=100,
    )
    dt = time.time() - t0
    print(f'\n>> GA finished in {dt:.1f}s  (fitness={res.fun:.6f}, '
          f'generations={res.nit}, nfev={res.nfev})')

    _report(res.x, dt, res.fun, res.history, res.nfev)


def _report(p_opt, dt, fitness, history, nfev):
    ok, info = M.validate_kinematics(p_opt, verbose=True)

    metr = M.evaluate(p_opt)
    if metr is None:
        print('>> WARNING: the result does not produce valid kinematics.')
        return

    print('\n>> RESULTS (Genetic Algorithm)')
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

    header = ('Optimized parameters (16) - SIMPLIFIED model (medial phalanx) '
              '- GENETIC ALGORITHM\n'
              f'fitness={fitness:.6f}  err_global_mm={metr["err_global_mm"]:.4f}  '
              f'err_ifp_mm={metr["err_ifp_mm"]:.4f}  err_ifd_mm={metr["err_ifd_mm"]:.4f}  '
              f'time_s={dt:.1f}  nfev={nfev}\n'
              + ' | '.join(M.NAMES))
    np.savetxt(os.path.join(OUTDIR, 'parameters_simplified_GA.txt'),
               p_opt, header=header)

    al = metr['aligned']
    pd.DataFrame({
        'x_ifp_mocap': M.mocap_pts['ifp'][:, 0], 'y_ifp_mocap': M.mocap_pts['ifp'][:, 1],
        'x_ifp_exo': al['ifp'][:, 0], 'y_ifp_exo': al['ifp'][:, 1],
        'x_ifd_mocap': M.mocap_pts['ifd'][:, 0], 'y_ifd_mocap': M.mocap_pts['ifd'][:, 1],
        'x_ifd_exo': al['ifd'][:, 0], 'y_ifd_exo': al['ifd'][:, 1],
    }).to_csv(os.path.join(OUTDIR, 'trajectories_simplified_GA.csv'), index=False)

    pd.DataFrame({'generation': np.arange(len(history)),
                  'best_fitness': history}).to_csv(
        os.path.join(OUTDIR, 'convergence_simplified_GA.csv'), index=False)

    fig, ax = plt.subplots(figsize=(10, 7))
    mp = M.mocap_pts
    ax.plot(mp['ifp'][:, 0]*1000, mp['ifp'][:, 1]*1000, 'r--', lw=2.5, alpha=0.65, label='MOCAP PIP')
    ax.plot(mp['ifd'][:, 0]*1000, mp['ifd'][:, 1]*1000, 'g--', lw=2.5, alpha=0.65, label='MOCAP DIP')
    ax.plot(al['ifp'][:, 0]*1000, al['ifp'][:, 1]*1000, 'r-', lw=2.5, label='EXO PIP')
    ax.plot(al['ifd'][:, 0]*1000, al['ifd'][:, 1]*1000, 'g-', lw=2.5, label='EXO DIP')
    ax.set_aspect('equal'); ax.grid(True, ls=':', alpha=0.7); ax.legend(fontsize=10)
    ax.set_title('Biofidelity - Simplified model (medial phalanx)\n'
                 f'Genetic Algorithm - Global error {metr["err_global_mm"]:.3f} mm',
                 fontsize=13, fontweight='bold')
    ax.set_xlabel('X axis (mm)'); ax.set_ylabel('Y axis (mm)')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, 'biofidelity_simplified_GA.png'), dpi=150)
    plt.close()

    print(f'\n>> All results saved to {OUTDIR}/')


if __name__ == '__main__':
    main()
