"""
optimizar_simplificado_ED.py
============================
Aqui corremos la optimizacion del modelo SIMPLIFICADO (hasta la falange medial,
16 parametros) con EVOLUCION DIFERENCIAL (scipy.optimize.differential_evolution).

Este es uno de los dos contendientes del articulo COMROB. Usa la MISMA funcion
objetivo y los MISMOS limites que la version de GA (optimizar_simplificado_GA.py),
asi que la comparacion mide el algoritmo, no el problema.

SINTONIZACION DE HIPERPARAMETROS (Optuna):
  Para que la comparacion ED vs GA sea SIMETRICA, los hiperparametros de la ED
  se sintonizan con optimizacion bayesiana (Optuna, muestreador TPE) en una
  etapa previa, igual que se hace con el GA. El flujo es de dos etapas:
    1) BUSQUEDA: Optuna prueba N_TRIALS_OPTUNA configuraciones con corridas
       cortas (DE_MAXITER_OPTUNA generaciones) y se queda con la mejor.
    2) CORRIDA PROFUNDA: se vuelve a optimizar con los mejores hiperparametros
       y el presupuesto completo (OPT_MAXITER generaciones).

Lo que genera:
  - resultados/parametros_simplificado_ED.txt   (incluye hiperparametros elegidos)
  - resultados/trayectorias_simplificado_ED.csv
  - resultados/biofidelidad_simplificado_ED.png
  - resultados/convergencia_simplificado_ED.csv

Presupuesto configurable por entorno:
  N_TRIALS_OPTUNA (25), DE_MAXITER_OPTUNA (40), OPT_MAXITER (300),
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

import modelo_simplificado as M

optuna.logging.set_verbosity(optuna.logging.WARNING)

# --- Presupuesto (configurable por variables de entorno) ---
N_TRIALS_OPTUNA   = int(os.environ.get('N_TRIALS_OPTUNA', '25'))
DE_MAXITER_OPTUNA = int(os.environ.get('DE_MAXITER_OPTUNA', '40'))
MAXITER           = int(os.environ.get('OPT_MAXITER', '300'))
SEED              = int(os.environ.get('OPT_SEED', '42'))
OPTUNA_SEED       = int(os.environ.get('OPTUNA_SEED', '42'))
OUTDIR = os.path.join(os.path.dirname(__file__), 'resultados')


def _correr_ed(strategy, popsize, mut_low, mut_high, recombination,
               maxiter, seed, callback=None):
    """Lanza differential_evolution con los hiperparametros dados."""
    return differential_evolution(
        M.fitness_function, M.bounds,
        strategy=strategy, popsize=popsize,
        mutation=(mut_low, mut_high), recombination=recombination,
        maxiter=maxiter, tol=1e-7,
        polish=True, disp=False,
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
                     maxiter=DE_MAXITER_OPTUNA, seed=SEED)
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

    # --- Guardado de parametros (incluye hiperparametros elegidos por Optuna) ---
    hp_str = '  '.join(f'{k}={v}' for k, v in hp.items())
    cab = ('Parametros optimizados (16) - Modelo SIMPLIFICADO (falange medial) '
           '- EVOLUCION DIFERENCIAL\n'
           f'fitness={fitness:.6f}  err_global_mm={metr["err_global_mm"]:.4f}  '
           f'err_ifp_mm={metr["err_ifp_mm"]:.4f}  err_ifd_mm={metr["err_ifd_mm"]:.4f}  '
           f'tiempo_s={dt:.1f}  nfev={nfev}\n'
           f'HIPERPARAMETROS (Optuna): {hp_str}\n'
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
