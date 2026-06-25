"""
optimizar_simplificado_GA.py
============================
Optimizacion del modelo SIMPLIFICADO (hasta la falange medial, 16 parametros)
mediante un ALGORITMO GENETICO de codificacion real (definido en `comun.py`:
seleccion por torneo + cruce SBX + mutacion polinomial + elitismo).

Es el segundo contendiente del articulo COMROB. Usa EXACTAMENTE la misma
funcion objetivo y los mismos limites que la version de Evolucion Diferencial
(`optimizar_simplificado_ED.py`).

Genera:
  - resultados/parametros_simplificado_GA.txt
  - resultados/trayectorias_simplificado_GA.csv
  - resultados/biofidelidad_simplificado_GA.png
  - resultados/convergencia_simplificado_GA.csv

Presupuesto configurable por entorno: GA_POPSIZE (70), GA_NGEN (300),
GA_SEED (42).
"""
import os
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

import modelo_simplificado as M
from comun import algoritmo_genetico

POPSIZE = int(os.environ.get('GA_POPSIZE', '70'))
NGEN = int(os.environ.get('GA_NGEN', '300'))
SEED = int(os.environ.get('GA_SEED', '42'))
OUTDIR = os.path.join(os.path.dirname(__file__), 'resultados')


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    print('=' * 64)
    print(' OPTIMIZACION SIMPLIFICADA (FALANGE MEDIAL) - ALGORITMO GENETICO')
    print('=' * 64)
    print(f'   parametros : {len(M.bounds)}   popsize: {POPSIZE}   '
          f'ngen: {NGEN}   seed: {SEED}')

    t0 = time.time()
    res = algoritmo_genetico(
        M.fitness_function, M.bounds,
        popsize=POPSIZE, ngen=NGEN,
        cxpb=0.9, mutpb=0.15, eta_c=15.0, eta_m=20.0,
        tournsize=3, n_elite=2, seed=SEED, disp=True, paciencia=100,
    )
    dt = time.time() - t0
    print(f'\n>> GA finalizado en {dt:.1f}s  (fitness={res.fun:.6f}, '
          f'generaciones={res.nit}, nfev={res.nfev})')

    _reportar(res.x, dt, res.fun, res.historial, res.nfev)


def _reportar(p_opt, dt, fitness, historial, nfev):
    ok, info = M.validar_cinematica(p_opt, verbose=True)

    metr = M.evaluar(p_opt)
    if metr is None:
        print('>> ADVERTENCIA: el resultado no produce cinematica valida.')
        return

    print('\n>> RESULTADOS (Algoritmo Genetico)')
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

    cab = ('Parametros optimizados (16) - Modelo SIMPLIFICADO (falange medial) '
           '- ALGORITMO GENETICO\n'
           f'fitness={fitness:.6f}  err_global_mm={metr["err_global_mm"]:.4f}  '
           f'err_ifp_mm={metr["err_ifp_mm"]:.4f}  err_ifd_mm={metr["err_ifd_mm"]:.4f}  '
           f'tiempo_s={dt:.1f}  nfev={nfev}\n'
           + ' | '.join(M.NOMBRES))
    np.savetxt(os.path.join(OUTDIR, 'parametros_simplificado_GA.txt'),
               p_opt, header=cab)

    al = metr['aligned']
    pd.DataFrame({
        'x_ifp_mocap': M.mocap_pts['ifp'][:, 0], 'y_ifp_mocap': M.mocap_pts['ifp'][:, 1],
        'x_ifp_exo': al['ifp'][:, 0], 'y_ifp_exo': al['ifp'][:, 1],
        'x_ifd_mocap': M.mocap_pts['ifd'][:, 0], 'y_ifd_mocap': M.mocap_pts['ifd'][:, 1],
        'x_ifd_exo': al['ifd'][:, 0], 'y_ifd_exo': al['ifd'][:, 1],
    }).to_csv(os.path.join(OUTDIR, 'trayectorias_simplificado_GA.csv'), index=False)

    pd.DataFrame({'generacion': np.arange(len(historial)),
                  'mejor_fitness': historial}).to_csv(
        os.path.join(OUTDIR, 'convergencia_simplificado_GA.csv'), index=False)

    fig, ax = plt.subplots(figsize=(10, 7))
    mp = M.mocap_pts
    ax.plot(mp['ifp'][:, 0]*1000, mp['ifp'][:, 1]*1000, 'r--', lw=2.5, alpha=0.65, label='MOCAP IFP')
    ax.plot(mp['ifd'][:, 0]*1000, mp['ifd'][:, 1]*1000, 'g--', lw=2.5, alpha=0.65, label='MOCAP IFD')
    ax.plot(al['ifp'][:, 0]*1000, al['ifp'][:, 1]*1000, 'r-', lw=2.5, label='EXO IFP')
    ax.plot(al['ifd'][:, 0]*1000, al['ifd'][:, 1]*1000, 'g-', lw=2.5, label='EXO IFD')
    ax.set_aspect('equal'); ax.grid(True, ls=':', alpha=0.7); ax.legend(fontsize=10)
    ax.set_title('Biofidelidad - Modelo simplificado (falange medial)\n'
                 f'Algoritmo Genetico - Error global {metr["err_global_mm"]:.3f} mm',
                 fontsize=13, fontweight='bold')
    ax.set_xlabel('Eje X (mm)'); ax.set_ylabel('Eje Y (mm)')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, 'biofidelidad_simplificado_GA.png'), dpi=150)
    plt.close()

    print(f'\n>> Archivos guardados en {OUTDIR}/ '
          '(parametros, trayectorias, convergencia, biofidelidad).')


if __name__ == '__main__':
    main()
