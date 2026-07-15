"""
optimizar_completo_ED.py
========================
Optimizacion del modelo COMPLETO del exo (TRES falanges, CINCO mecanismos,
21 parametros) con EVOLUCION DIFERENCIAL.

Usamos ED porque fue el ganador de la comparacion sobre el modelo simplificado.
Aqui ya van TODAS las mejoras validadas en la version parcial: dimensiones
acotadas, angulo auxiliar medial-c2 reducido, regularizacion suave, validacion
cinematica y --como en el modelo simplificado-- SINTONIZACION DE HIPERPARAMETROS
CON OPTUNA (muestreador TPE), en lugar de fijarlos a mano.

Flujo de dos etapas (identico al del modelo simplificado):
  1) BUSQUEDA (Optuna/TPE): prueba N_TRIALS_OPTUNA configuraciones de
     hiperparametros con corridas cortas (DE_MAXITER_OPTUNA generaciones).
  2) CORRIDA PROFUNDA: re-optimiza con los mejores hiperparametros y el
     presupuesto completo (OPT_MAXITER generaciones).

Lo que genera:
  - resultados/parametros_completo_ED.txt   (21 parametros + hiperparametros)
  - resultados/trayectorias_completo_ED.csv
  - resultados/biofidelidad_completo_ED.png
  - resultados/convergencia_completo_ED.csv

Presupuesto configurable por entorno:
  N_TRIALS_OPTUNA (25), DE_MAXITER_OPTUNA (60), OPT_MAXITER (600),
  OPT_SEED (42), OPTUNA_SEED (42).
"""
import os
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import optuna
from scipy.optimize import differential_evolution

import modelo_completo as M

optuna.logging.set_verbosity(optuna.logging.WARNING)

# --- Presupuesto (configurable por variables de entorno) ---
N_TRIALS_OPTUNA   = int(os.environ.get('N_TRIALS_OPTUNA', '25'))
DE_MAXITER_OPTUNA = int(os.environ.get('DE_MAXITER_OPTUNA', '60'))
MAXITER           = int(os.environ.get('OPT_MAXITER', '600'))
SEED              = int(os.environ.get('OPT_SEED', '42'))
OPTUNA_SEED       = int(os.environ.get('OPTUNA_SEED', '42'))
OUTDIR = os.path.join(os.path.dirname(__file__), 'resultados')


def _correr_ed(strategy, popsize, mut_low, mut_high, recombination,
               maxiter, seed, callback=None, polish=True):
    """Lanza differential_evolution con los hiperparametros dados.

    Durante la sintonizacion (Optuna) se usa polish=False para acelerar: el
    pulido local L-BFGS es costoso en el modelo completo (21 parametros) y solo
    hace falta en la corrida profunda final. La sintonizacion solo necesita
    comparar hiperparametros de forma relativa.
    """
    return differential_evolution(
        M.fitness_function, M.bounds,
        strategy=strategy, popsize=popsize,
        mutation=(mut_low, mut_high), recombination=recombination,
        maxiter=maxiter, tol=1e-6,
        polish=polish, disp=False,
        updating='deferred', workers=-1, seed=seed,
        callback=callback,
    )


# ==============================================================================
# --- ETAPA 1: SINTONIZACION BAYESIANA (Optuna) ---
# ==============================================================================
def objective_optuna(trial):
    """Cada trial prueba una configuracion de hiperparametros de la ED con una
    corrida corta y regresa el mejor fitness obtenido (a minimizar)."""
    strategy = trial.suggest_categorical(
        'strategy', ['best1bin', 'rand1bin', 'best1exp', 'currenttobest1bin'])
    popsize = trial.suggest_int('popsize', 15, 30)
    mut_low = trial.suggest_float('mut_low', 0.3, 0.7)
    mut_high = trial.suggest_float('mut_high', 0.8, 1.2)
    recombination = trial.suggest_float('recombination', 0.5, 0.95)

    res = _correr_ed(strategy, popsize, mut_low, mut_high, recombination,
                     maxiter=DE_MAXITER_OPTUNA, seed=SEED, polish=False)
    return float(res.fun)


def sintonizar():
    print('=' * 64)
    print(' ETAPA 1: SINTONIZACION DE HIPERPARAMETROS - ED (Optuna/TPE)')
    print('=' * 64)
    print(f'   trials: {N_TRIALS_OPTUNA}   maxiter por trial: {DE_MAXITER_OPTUNA}')
    sampler = optuna.samplers.TPESampler(seed=OPTUNA_SEED)
    study = optuna.create_study(direction='minimize', sampler=sampler)
    t0 = time.time()
    study.optimize(objective_optuna, n_trials=N_TRIALS_OPTUNA,
                   show_progress_bar=False)
    dt = time.time() - t0
    bp = study.best_params
    print(f'\n>> Mejor configuracion (fitness corto={study.best_value:.6f}, '
          f'{dt:.1f}s):')
    for k, v in bp.items():
        print(f'   {k:16s}: {v}')
    return bp, dt


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    print('=' * 64)
    print(' OPTIMIZACION COMPLETA (3 FALANGES, 5 MECANISMOS) - EVOL. DIFERENCIAL')
    print('=' * 64)
    print(f'   parametros : {len(M.bounds)}  (17 base + 4 del tercer 4 barras)')

    # Checamos rapido con parametros de referencia (los del MATLAB + tercer mec.)
    p_ref = [0.018, 0.020, 0.035, 0.049, 0.025, 0.020, 0.025,
             0.055, 0.035, 0.052, 0.04601, 0.017, 0.018,
             np.deg2rad(51.39), np.deg2rad(38.78), 2.0, np.deg2rad(109),
             0.025, 0.035, -0.012, 0.002]
    print(f'   fitness ref. : {M.fitness_function(p_ref):.6f}\n')

    # --- ETAPA 1: sintonizar con Optuna ---
    bp, t_tune = sintonizar()

    # --- ETAPA 2: corrida profunda con los mejores hiperparametros ---
    print('\n' + '=' * 64)
    print(' ETAPA 2: CORRIDA PROFUNDA - EVOLUCION DIFERENCIAL')
    print('=' * 64)
    print(f'   parametros : {len(M.bounds)}   popsize: {bp["popsize"]}   '
          f'maxiter: {MAXITER}   seed: {SEED}   strategy: {bp["strategy"]}')

    historial = []

    def cb(xk, convergence=None):
        historial.append(M.fitness_function(xk))

    t0 = time.time()
    res = _correr_ed(bp['strategy'], bp['popsize'], bp['mut_low'],
                     bp['mut_high'], bp['recombination'],
                     maxiter=MAXITER, seed=SEED, callback=cb)
    dt = time.time() - t0
    print(f'\n>> ED listo en {dt:.1f}s  (fitness={res.fun:.6f}, '
          f'nfev={res.nfev})')

    hp = {
        'strategy': bp['strategy'],
        'popsize': bp['popsize'],
        'mutation_low': round(bp['mut_low'], 4),
        'mutation_high': round(bp['mut_high'], 4),
        'recombination': round(bp['recombination'], 4),
        'optuna_trials': N_TRIALS_OPTUNA,
        'tiempo_tune_s': round(t_tune, 1),
    }
    _reportar(res.x, dt, res.fun, historial, res.nfev, hp)


def _reportar(p_opt, dt, fitness, historial, nfev, hp):
    # Checamos viabilidad
    ok, info = M.validar_cinematica(p_opt, verbose=True)

    metr = M.evaluar(p_opt)
    if metr is None:
        print('>> OJO: el resultado no produce cinematica valida.')
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

    # --- Guardado (incluye hiperparametros elegidos por Optuna) ---
    hp_str = '  '.join(f'{k}={v}' for k, v in hp.items())
    cab = ('Parametros optimizados (21) - Modelo COMPLETO (3 falanges, 5 mecanismos) '
           '- EVOLUCION DIFERENCIAL\n'
           'Indices 0-16: mecanismo base; 17=Link9_3, 18=Link10_3, 19=back3_3, '
           '20=up3_3 (tercer 4 barras IFD/DIP)\n'
           f'fitness={fitness:.6f}  err_global_mm={metr["err_global_mm"]:.4f}  '
           f'err_ifp_mm={metr["err_ifp_mm"]:.4f}  err_ifd_mm={metr["err_ifd_mm"]:.4f}  '
           f'err_tip_mm={metr["err_tip_mm"]:.4f}  rom_dip_deg={metr["rom_dip_deg"]:.2f}  '
           f'tiempo_s={dt:.1f}  nfev={nfev}\n'
           f'HIPERPARAMETROS (Optuna): {hp_str}\n'
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

    # --- Curva de convergencia ---
    pd.DataFrame({'iteracion': np.arange(len(historial)),
                  'mejor_fitness': historial}).to_csv(
        os.path.join(OUTDIR, 'convergencia_completo_ED.csv'), index=False)

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

    print(f'\n>> Todo guardado en {OUTDIR}/')


if __name__ == '__main__':
    main()
