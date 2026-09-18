"""
optimizar_tercer_mecanismo.py
=============================
Runner PARALELO de la optimizacion del exoesqueleto CON el tercer mecanismo de
4 barras ya integrado (cinematica IFD/DIP real) definido en
`exo_18_pinza_fina.py`.

Por que un runner aparte:
  - Importa (sin re-ejecutar la optimizacion) el modelo cinematico, la funcion
    de fitness y los bounds de `exo_18_pinza_fina.py` -el bloque main de ese
    archivo esta protegido con `if __name__ == '__main__'`-, de modo que aqui se
    reutiliza EXACTAMENTE la misma cinematica (21 parametros, incluido el tercer
    4 barras).
  - Ejecuta `differential_evolution` con `workers=-1` (los 8 nucleos) y
    `updating='deferred'`, por lo que es ~6-8x mas rapido que la version
    secuencial y permite obtener parametros optimizados en una sola corrida.

Presupuesto configurable por variables de entorno:
  OPT_POPSIZE (def 22), OPT_MAXITER (def 1000), OPT_SEED (def 42)

Genera los MISMOS archivos de salida que el bloque main del optimizador:
  - Parametros_Optimizados_Pinza_Fina.txt   (21 parametros)
  - Resultados_Alineados_Pinza_Fina.csv
  - Resultados_Biofidelidad_Pinza_Fina.png
"""
import os
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import differential_evolution

# Importa el modelo ya modificado (no corre su optimizacion: main protegido)
import exo_18_pinza_fina as exo

POPSIZE = int(os.environ.get('OPT_POPSIZE', '22'))
MAXITER = int(os.environ.get('OPT_MAXITER', '1000'))
SEED    = int(os.environ.get('OPT_SEED', '42'))


def main():
    print('=' * 60)
    print(' OPTIMIZACION PARALELA - EXO CON TERCER MECANISMO (IFD/DIP)')
    print('=' * 60)
    print(f'   nucleos      : {os.cpu_count()}  (workers=-1)')
    print(f'   popsize      : {POPSIZE}   maxiter: {MAXITER}   seed: {SEED}')
    print(f'   parametros   : {len(exo.bounds)}  (17 base + 4 del tercer mecanismo)')

    # Validacion rapida con parametros de referencia (MATLAB + tercer mec.)
    p_ref = [0.018, 0.020, 0.035, 0.049, 0.025, 0.020, 0.025,
             0.055, 0.035, 0.052, 0.04601, 0.017, 0.018,
             np.deg2rad(51.39), np.deg2rad(38.78), 2.0, np.deg2rad(109),
             0.025, 0.035, -0.012, 0.002]
    print(f'   fitness ref. : {exo.fitness_function(p_ref):.6f}\n')

    t0 = time.time()
    resultado = differential_evolution(
        exo.fitness_function, exo.bounds,
        strategy='best1bin', popsize=POPSIZE,
        mutation=(0.5, 1.0), recombination=0.7,
        maxiter=MAXITER, tol=1e-6,
        polish=True, disp=True,
        updating='deferred', workers=-1, seed=SEED
    )
    print(f'\n>> Optimizacion finalizada en {time.time()-t0:.1f}s '
          f'(fitness={resultado.fun:.6f})')

    p_opt = resultado.x
    best_sim = exo.run_kinematics(p_opt, exo.theta_input)
    if best_sim is None:
        print('>> ADVERTENCIA: el resultado final no produce cinematica valida.')
        return

    # --- Alineacion rigida final (mocap <- exo) ---
    mp = exo.mocap_pts
    all_mocap = np.vstack([mp[k] for k in ('ifp', 'ifd', 'tip')])
    all_sim = np.vstack([best_sim[k] for k in ('ifp', 'ifd', 'tip')])
    R_opt, t_opt = exo.optimal_rigid_transform(all_mocap, all_sim)
    angulo_montaje = np.rad2deg(np.arctan2(R_opt[1, 0], R_opt[0, 0]))
    sim_aligned = {k: exo.apply_transform(best_sim[k], R_opt, t_opt)
                   for k in ('ifp', 'ifd', 'tip')}

    err_ifp_mm = exo.chamfer_distance(mp['ifp'], sim_aligned['ifp']) * 1000
    err_ifd_mm = exo.chamfer_distance(mp['ifd'], sim_aligned['ifd']) * 1000
    err_tip_mm = exo.chamfer_distance(mp['tip'], sim_aligned['tip']) * 1000
    error_global_mm = (err_ifp_mm + err_ifd_mm + err_tip_mm) / 3.0

    print('\n' + '=' * 60)
    print('>> RESULTADOS CINEMATICOS')
    print('=' * 60)
    print(f'   Error Global : {error_global_mm:.3f} mm')
    print(f'   Error IFP    : {err_ifp_mm:.3f} mm')
    print(f'   Error IFD    : {err_ifd_mm:.3f} mm')
    print(f'   Error Punta  : {err_tip_mm:.3f} mm\n')

    nombres = [
        "Bancada1 (m)", "Bancada2 (m)", "Link1 (m)", "Link2 (m)",
        "Link3 (m)", "Link4 (m)", "Link5 (m)", "Link6 (m)",
        "Link7 (m)", "Link8 (m)", "Link10 / c2 (m)", "hsp (m)", "dsp (m)",
        "Theta Aux FM (rad)", "Theta Aux FD (rad)",
        "Relacion de engranaje", "Theta Offset (rad)",
        "Link9_3  acoplador (m)", "Link10_3 balancin (m)",
        "back3_3  soporte (m)", "up3_3    standoff (m)"
    ]
    print('>> PARAMETROS DIMENSIONALES DEL MECANISMO')
    for nombre, valor in zip(nombres, p_opt):
        print(f'   {nombre:30s}: {valor:.6f}')

    print('\n>> PARAMETROS DE MONTAJE (transformacion rigida)')
    print(f'   Traslacion X : {t_opt[0]*1000:.2f} mm')
    print(f'   Traslacion Y : {t_opt[1]*1000:.2f} mm')
    print(f'   Rotacion Base: {angulo_montaje:.2f} deg')

    print('\n>> TERCER MECANISMO DE 4 BARRAS (IFD/DIP) [mm]')
    print(f'   Link9_3  acoplador (D3->Pa)    : {p_opt[17]*1000:.2f} mm')
    print(f'   Link10_3 balancin  (IFD->D3)   : {p_opt[18]*1000:.2f} mm')
    print(f'   back3_3  soporte S3 (a lo largo): {p_opt[19]*1000:.2f} mm')
    print(f'   up3_3    standoff dorsal        : {p_opt[20]*1000:.2f} mm')

    rel_dip = np.rad2deg(np.unwrap(best_sim['theta_fd'])
                         - np.unwrap(best_sim['theta_fm']))
    rel_dip -= rel_dip[0]
    mocap_dip_rom = np.rad2deg(exo.dip_rel_mocap.max() - exo.dip_rel_mocap.min())
    print(f'\n   Flexion distal relativa IFD (EXO)  : min={rel_dip.min():+.2f} deg, '
          f'max={rel_dip.max():+.2f} deg, ROM={rel_dip.max()-rel_dip.min():.2f} deg')
    print(f'   Flexion distal relativa DIP (MOCAP): ROM={mocap_dip_rom:.2f} deg '
          f'(objetivo fisiologico)')
    print('=' * 60 + '\n')

    # --- Grafica de biofidelidad ---
    fig, ax = plt.subplots(figsize=(11, 8))
    lw = 2.5
    ax.plot(mp['ifp'][:, 0]*1000, mp['ifp'][:, 1]*1000, 'r--', lw=lw, alpha=0.65, label='MOCAP IFP')
    ax.plot(mp['ifd'][:, 0]*1000, mp['ifd'][:, 1]*1000, 'g--', lw=lw, alpha=0.65, label='MOCAP IFD')
    ax.plot(mp['tip'][:, 0]*1000, mp['tip'][:, 1]*1000, 'b--', lw=lw, alpha=0.65, label='MOCAP Punta')
    ax.plot(sim_aligned['ifp'][:, 0]*1000, sim_aligned['ifp'][:, 1]*1000, 'r-', lw=lw, label='EXO IFP')
    ax.plot(sim_aligned['ifd'][:, 0]*1000, sim_aligned['ifd'][:, 1]*1000, 'g-', lw=lw, label='EXO IFD')
    ax.plot(sim_aligned['tip'][:, 0]*1000, sim_aligned['tip'][:, 1]*1000, 'b-', lw=lw, label='EXO Punta')
    ax.set_aspect('equal')
    ax.grid(True, linestyle=':', alpha=0.7)
    ax.legend(loc='upper right', fontsize=9)
    ax.set_title(f'Biofidelidad Pinza Fina (con 3er mecanismo IFD)\n'
                 f'Error Global: {error_global_mm:.3f} mm', fontsize=14, fontweight='bold')
    ax.set_xlabel('Eje X (mm)', fontsize=12)
    ax.set_ylabel('Eje Y (mm)', fontsize=12)
    plt.tight_layout()
    plt.savefig('Resultados_Biofidelidad_Pinza_Fina.png', dpi=150)
    plt.close()

    # --- Guardado de parametros y trayectorias ---
    np.savetxt('Parametros_Optimizados_Pinza_Fina.txt', p_opt,
               header='Eslabones y offsets optimizados (21 parametros) - Pinza Fina. '
                      'Indices 0-16: mecanismo original; 17=Link9_3, 18=Link10_3, '
                      '19=back3_3, 20=up3_3 (tercer mecanismo de 4 barras IFD/DIP)')

    if len(mp['ifp']) == len(sim_aligned['ifp']):
        pd.DataFrame({
            'x_ifp_mocap': mp['ifp'][:, 0], 'y_ifp_mocap': mp['ifp'][:, 1],
            'x_ifp_exo': sim_aligned['ifp'][:, 0], 'y_ifp_exo': sim_aligned['ifp'][:, 1],
            'x_ifd_mocap': mp['ifd'][:, 0], 'y_ifd_mocap': mp['ifd'][:, 1],
            'x_ifd_exo': sim_aligned['ifd'][:, 0], 'y_ifd_exo': sim_aligned['ifd'][:, 1],
            'x_tip_mocap': mp['tip'][:, 0], 'y_tip_mocap': mp['tip'][:, 1],
            'x_tip_exo': sim_aligned['tip'][:, 0], 'y_tip_exo': sim_aligned['tip'][:, 1],
        }).to_csv('Resultados_Alineados_Pinza_Fina.csv', index=False)

    print('>> Archivos guardados: Parametros_Optimizados_Pinza_Fina.txt, '
          'Resultados_Alineados_Pinza_Fina.csv, Resultados_Biofidelidad_Pinza_Fina.png')


if __name__ == '__main__':
    main()
