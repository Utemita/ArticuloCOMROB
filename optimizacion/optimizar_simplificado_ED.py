"""
optimizar_simplificado_ED.py
============================
Aqui corremos la optimizacion del modelo SIMPLIFICADO (hasta la falange medial,
16 parametros) con EVOLUCION DIFERENCIAL (scipy.optimize.differential_evolution).

Este es uno de los dos contendientes del articulo COMROB. Usa la MISMA funcion
objetivo y los MISMOS limites que la version de GA (optimizar_simplificado_GA.py),
asi que la comparacion mide el algoritmo, no el problema.

Lo que genera:
  - resultados/parametros_simplificado_ED.txt
  - resultados/trayectorias_simplificado_ED.csv
  - resultados/biofidelidad_simplificado_ED.png
  - resultados/convergencia_simplificado_ED.csv

Presupuesto configurable por entorno: OPT_POPSIZE (15), OPT_MAXITER (300),
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

import modelo_simplificado as M

POPSIZE = int(os.environ.get('OPT_POPSIZE', '15'))
MAXITER = int(os.environ.get('OPT_MAXITER', '300'))
SEED = int(os.environ.get('OPT_SEED', '42'))
OUTDIR = os.path.join(os.path.dirname(__file__), 'resultados')


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    print('=' * 64)
    print(' OPTIMIZACION SIMPLIFICADA (FALANGE MEDIAL) - EVOLUCION DIFERENCIAL')
    print('=' * 64)
    print(f'   parametros : {len(M.bounds)}   popsize: {POPSIZE}   '
          f'maxiter: {MAXITER}   seed: {SEED}')

    historial = []

    def cb(xk, convergence=None):
        historial.append(M.fitness_function(xk))

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
    print(f'\n>> ED listo en {dt:.1f}s  (fitness={res.fun:.6f}, '
          f'nfev={res.nfev})')

    _reportar(res.x, dt, res.fun, historial, res.nfev)


def _reportar(p_opt, dt, fitness, historial, nfev):
    # --- Checamos viabilidad cinematica ---
    ok, info = M.validar_cinematica(p_opt, verbose=True)

    metr = M.evaluar(p_opt)
    if metr is None:
        print('>> OJO: el resultado no produce cinematica valida.')
        return

    print('\n>> RESULTADOS (Evolucion Diferencial)')
    print(f'   Error Global : {metr["err_global_mm"]:.3f} mm')
    print(f'   Error IFP    : {metr["err_ifp_mm"]:.3f} mm')
    print(f'   Error IFD    : {metr["err_ifd_mm"]:.3f} mm')
    print(f'   ROM medial   : {info.get("rom_medial_deg", float("nan")):.2f} deg')
    print(f'   Theta Aux FM : {np.rad2deg(p_opt[13]):.2f} deg')
    suma_links_mm = float(np.sum(p_opt[:11])) * 1000
    print(f'   Long. total eslabones : {suma_links_mm:.1f} mm')

    print('\n>> PARAMETROS OPTIMIZADOS')
    for nombre, valor in zip(M.NOMBRES, p_opt):
        print(f'   {nombre:26s}: {valor:.6f}')

    # --- Guardado de parametros ---
    cab = ('Parametros optimizados (16) - Modelo SIMPLIFICADO (falange medial) '
           '- EVOLUCION DIFERENCIAL\n'
           f'fitness={fitness:.6f}  err_global_mm={metr["err_global_mm"]:.4f}  '
           f'err_ifp_mm={metr["err_ifp_mm"]:.4f}  err_ifd_mm={metr["err_ifd_mm"]:.4f}  '
           f'tiempo_s={dt:.1f}  nfev={nfev}\n'
           + ' | '.join(M.NOMBRES))
    np.savetxt(os.path.join(OUTDIR, 'parametros_simplificado_ED.txt'),
               p_opt, header=cab)

    # --- Trayectorias alineadas ---
    al = metr['aligned']
    pd.DataFrame({
        'x_ifp_mocap': M.mocap_pts['ifp'][:, 0], 'y_ifp_mocap': M.mocap_pts['ifp'][:, 1],
        'x_ifp_exo': al['ifp'][:, 0], 'y_ifp_exo': al['ifp'][:, 1],
        'x_ifd_mocap': M.mocap_pts['ifd'][:, 0], 'y_ifd_mocap': M.mocap_pts['ifd'][:, 1],
        'x_ifd_exo': al['ifd'][:, 0], 'y_ifd_exo': al['ifd'][:, 1],
    }).to_csv(os.path.join(OUTDIR, 'trayectorias_simplificado_ED.csv'), index=False)

    # --- Curva de convergencia ---
    pd.DataFrame({'iteracion': np.arange(len(historial)),
                  'mejor_fitness': historial}).to_csv(
        os.path.join(OUTDIR, 'convergencia_simplificado_ED.csv'), index=False)

    # --- Grafica de biofidelidad ---
    fig, ax = plt.subplots(figsize=(10, 7))
    mp = M.mocap_pts
    ax.plot(mp['ifp'][:, 0]*1000, mp['ifp'][:, 1]*1000, 'r--', lw=2.5, alpha=0.65, label='MOCAP IFP')
    ax.plot(mp['ifd'][:, 0]*1000, mp['ifd'][:, 1]*1000, 'g--', lw=2.5, alpha=0.65, label='MOCAP IFD')
    ax.plot(al['ifp'][:, 0]*1000, al['ifp'][:, 1]*1000, 'r-', lw=2.5, label='EXO IFP')
    ax.plot(al['ifd'][:, 0]*1000, al['ifd'][:, 1]*1000, 'g-', lw=2.5, label='EXO IFD')
    ax.set_aspect('equal'); ax.grid(True, ls=':', alpha=0.7); ax.legend(fontsize=10)
    ax.set_title('Biofidelidad - Modelo simplificado (falange medial)\n'
                 f'Evolucion Diferencial - Error global {metr["err_global_mm"]:.3f} mm',
                 fontsize=13, fontweight='bold')
    ax.set_xlabel('Eje X (mm)'); ax.set_ylabel('Eje Y (mm)')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, 'biofidelidad_simplificado_ED.png'), dpi=150)
    plt.close()

    print(f'\n>> Todo guardado en {OUTDIR}/')


if __name__ == '__main__':
    main()
