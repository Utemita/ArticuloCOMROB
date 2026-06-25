"""
optimizar_completo_ED.py
========================
Optimizacion del modelo COMPLETO del exoesqueleto (TRES falanges, CINCO
mecanismos, 21 parametros) mediante EVOLUCION DIFERENCIAL.

Se usa Evolucion Diferencial por ser el enfoque GANADOR del estudio comparativo
realizado sobre el modelo simplificado (ED 2.15 mm vs GA ~5.5 mm). Incorpora las
mejoras validadas alli: dimensiones acotadas, angulo auxiliar medial-c2
reducido, regularizacion suave y validacion cinematica de viabilidad.

Genera:
  - resultados/parametros_completo_ED.txt           (21 parametros)
  - resultados/trayectorias_completo_ED.csv
  - resultados/biofidelidad_completo_ED.png

Presupuesto configurable por entorno: OPT_POPSIZE (20), OPT_MAXITER (600),
OPT_SEED (100).
"""
import os
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import differential_evolution

import modelo_completo as M

POPSIZE = int(os.environ.get('OPT_POPSIZE', '20'))
MAXITER = int(os.environ.get('OPT_MAXITER', '600'))
SEED = int(os.environ.get('OPT_SEED', '100'))
OUTDIR = os.path.join(os.path.dirname(__file__), 'resultados')


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    print('=' * 64)
    print(' OPTIMIZACION COMPLETA (3 FALANGES, 5 MECANISMOS) - EVOL. DIFERENCIAL')
    print('=' * 64)
    print(f'   parametros : {len(M.bounds)}  (17 base + 4 del tercer 4 barras)')
    print(f'   popsize: {POPSIZE}   maxiter: {MAXITER}   seed: {SEED}')

    # Validacion rapida con parametros de referencia (MATLAB + tercer mec.)
    p_ref = [0.018, 0.020, 0.035, 0.049, 0.025, 0.020, 0.025,
             0.055, 0.035, 0.052, 0.04601, 0.017, 0.018,
             np.deg2rad(51.39), np.deg2rad(38.78), 2.0, np.deg2rad(109),
             0.025, 0.035, -0.012, 0.002]
    print(f'   fitness ref. : {M.fitness_function(p_ref):.6f}\n')

    t0 = time.time()
    res = differential_evolution(
        M.fitness_function, M.bounds,
        strategy='best1bin', popsize=POPSIZE,
        mutation=(0.5, 1.0), recombination=0.7,
        maxiter=MAXITER, tol=1e-6,
        polish=True, disp=True,
        updating='deferred', workers=-1, seed=SEED,
    )
    dt = time.time() - t0
    print(f'\n>> ED finalizada en {dt:.1f}s  (fitness={res.fun:.6f}, '
          f'nfev={res.nfev})')

    _reportar(res.x, dt, res.fun, res.nfev)


def _reportar(p_opt, dt, fitness, nfev):
    ok, info = M.validar_cinematica(p_opt, verbose=True)

    metr = M.evaluar(p_opt)
    if metr is None:
        print('>> ADVERTENCIA: el resultado no produce cinematica valida.')
        return

    print('\n>> RESULTADOS (Evolucion Diferencial - modelo completo)')
    print(f'   Error Global : {metr["err_global_mm"]:.3f} mm')
    print(f'   Error IFP    : {metr["err_ifp_mm"]:.3f} mm')
    print(f'   Error IFD    : {metr["err_ifd_mm"]:.3f} mm')
    print(f'   Error Punta  : {metr["err_tip_mm"]:.3f} mm')
    print(f'   ROM medial   : {info.get("rom_medial_deg", float("nan")):.2f} deg')
    print(f'   ROM DIP      : {metr["rom_dip_deg"]:.2f} deg')
    print(f'   Theta Aux FM : {np.rad2deg(p_opt[13]):.2f} deg')
    suma_links_mm = float(np.sum(p_opt[:11])) * 1000
    print(f'   Long. total eslabones (base) : {suma_links_mm:.1f} mm')

    print('\n>> PARAMETROS OPTIMIZADOS')
    for nombre, valor in zip(M.NOMBRES, p_opt):
        print(f'   {nombre:26s}: {valor:.6f}')

    print('\n>> TERCER MECANISMO DE 4 BARRAS (IFD/DIP) [mm]')
    print(f'   Link9_3  acoplador (D3->Pa)     : {p_opt[17]*1000:.2f} mm')
    print(f'   Link10_3 balancin  (IFD->D3)    : {p_opt[18]*1000:.2f} mm')
    print(f'   back3_3  soporte S3 (a lo largo) : {p_opt[19]*1000:.2f} mm')
    print(f'   up3_3    standoff dorsal         : {p_opt[20]*1000:.2f} mm')

    cab = ('Parametros optimizados (21) - Modelo COMPLETO (3 falanges, 5 mecanismos) '
           '- EVOLUCION DIFERENCIAL\n'
           'Indices 0-16: mecanismo base; 17=Link9_3, 18=Link10_3, 19=back3_3, '
           '20=up3_3 (tercer 4 barras IFD/DIP)\n'
           f'fitness={fitness:.6f}  err_global_mm={metr["err_global_mm"]:.4f}  '
           f'err_ifp_mm={metr["err_ifp_mm"]:.4f}  err_ifd_mm={metr["err_ifd_mm"]:.4f}  '
           f'err_tip_mm={metr["err_tip_mm"]:.4f}  rom_dip_deg={metr["rom_dip_deg"]:.2f}  '
           f'tiempo_s={dt:.1f}  nfev={nfev}\n'
           + ' | '.join(M.NOMBRES))
    np.savetxt(os.path.join(OUTDIR, 'parametros_completo_ED.txt'), p_opt, header=cab)

    al = metr['aligned']
    mp = M.mocap_pts
    pd.DataFrame({
        'x_ifp_mocap': mp['ifp'][:, 0], 'y_ifp_mocap': mp['ifp'][:, 1],
        'x_ifp_exo': al['ifp'][:, 0], 'y_ifp_exo': al['ifp'][:, 1],
        'x_ifd_mocap': mp['ifd'][:, 0], 'y_ifd_mocap': mp['ifd'][:, 1],
        'x_ifd_exo': al['ifd'][:, 0], 'y_ifd_exo': al['ifd'][:, 1],
        'x_tip_mocap': mp['tip'][:, 0], 'y_tip_mocap': mp['tip'][:, 1],
        'x_tip_exo': al['tip'][:, 0], 'y_tip_exo': al['tip'][:, 1],
    }).to_csv(os.path.join(OUTDIR, 'trayectorias_completo_ED.csv'), index=False)

    fig, ax = plt.subplots(figsize=(11, 8))
    lw = 2.5
    ax.plot(mp['ifp'][:, 0]*1000, mp['ifp'][:, 1]*1000, 'r--', lw=lw, alpha=0.65, label='MOCAP IFP')
    ax.plot(mp['ifd'][:, 0]*1000, mp['ifd'][:, 1]*1000, 'g--', lw=lw, alpha=0.65, label='MOCAP IFD')
    ax.plot(mp['tip'][:, 0]*1000, mp['tip'][:, 1]*1000, 'b--', lw=lw, alpha=0.65, label='MOCAP Punta')
    ax.plot(al['ifp'][:, 0]*1000, al['ifp'][:, 1]*1000, 'r-', lw=lw, label='EXO IFP')
    ax.plot(al['ifd'][:, 0]*1000, al['ifd'][:, 1]*1000, 'g-', lw=lw, label='EXO IFD')
    ax.plot(al['tip'][:, 0]*1000, al['tip'][:, 1]*1000, 'b-', lw=lw, label='EXO Punta')
    ax.set_aspect('equal'); ax.grid(True, ls=':', alpha=0.7); ax.legend(fontsize=9)
    ax.set_title('Biofidelidad - Modelo completo (3 falanges, 5 mecanismos)\n'
                 f'Evolucion Diferencial - Error global {metr["err_global_mm"]:.3f} mm',
                 fontsize=13, fontweight='bold')
    ax.set_xlabel('Eje X (mm)'); ax.set_ylabel('Eje Y (mm)')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, 'biofidelidad_completo_ED.png'), dpi=150)
    plt.close()

    print(f'\n>> Archivos guardados en {OUTDIR}/ '
          '(parametros, trayectorias, biofidelidad).')


if __name__ == '__main__':
    main()
