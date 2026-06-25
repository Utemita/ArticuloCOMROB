"""
optimizar_simplificado_GA.py
============================
Optimizacion del modelo SIMPLIFICADO (hasta la falange medial, 16 parametros)
con un ALGORITMO GENETICO de codificacion real (el que esta en comun.py:
torneo + cruce SBX + mutacion polinomial + elitismo).

Este es el segundo contendiente del articulo. Usa la MISMA funcion objetivo y
los MISMOS limites que la version de ED (optimizar_simplificado_ED.py), asi
que la comparacion es justa.

SINTONIZACION DE HIPERPARAMETROS (Optuna):
  Igual que la ED, los hiperparametros del GA se sintonizan con optimizacion
  bayesiana (Optuna, muestreador TPE) usando el MISMO numero de trials, para
  que la comparacion sea SIMETRICA. El flujo es de dos etapas:
    1) BUSQUEDA: Optuna prueba N_TRIALS_OPTUNA configuraciones con corridas
       cortas (GA_NGEN_OPTUNA generaciones) y se queda con la mejor.
    2) CORRIDA PROFUNDA: se vuelve a optimizar con los mejores hiperparametros
       y el presupuesto completo (GA_NGEN generaciones).

Lo que genera:
  - resultados/parametros_simplificado_GA.txt   (incluye hiperparametros elegidos)
  - resultados/trayectorias_simplificado_GA.csv
  - resultados/biofidelidad_simplificado_GA.png
  - resultados/convergencia_simplificado_GA.csv

Presupuesto configurable por entorno:
  N_TRIALS_OPTUNA (25), GA_NGEN_OPTUNA (60), GA_NGEN (300),
  GA_SEED (42), OPTUNA_SEED (42).
"""
import os
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import optuna

import modelo_simplificado as M
from comun import algoritmo_genetico

optuna.logging.set_verbosity(optuna.logging.WARNING)

# --- Presupuesto (configurable por variables de entorno) ---
N_TRIALS_OPTUNA = int(os.environ.get('N_TRIALS_OPTUNA', '25'))
NGEN_OPTUNA     = int(os.environ.get('GA_NGEN_OPTUNA', '60'))
NGEN            = int(os.environ.get('GA_NGEN', '300'))
SEED            = int(os.environ.get('GA_SEED', '42'))
OPTUNA_SEED     = int(os.environ.get('OPTUNA_SEED', '42'))
OUTDIR = os.path.join(os.path.dirname(__file__), 'resultados')


def _correr_ga(popsize, cxpb, mutpb, eta_c, eta_m, tournsize,
               ngen, seed, paciencia, disp=False, callback=None):
    """Lanza el algoritmo genetico con los hiperparametros dados."""
    return algoritmo_genetico(
        M.fitness_function, M.bounds,
        popsize=popsize, ngen=ngen,
        cxpb=cxpb, mutpb=mutpb, eta_c=eta_c, eta_m=eta_m,
        tournsize=tournsize, n_elite=2, seed=seed,
        disp=disp, paciencia=paciencia, callback=callback,
    )


# ==============================================================================
# --- ETAPA 1: SINTONIZACION BAYESIANA (Optuna) ---
# ==============================================================================
def objective_optuna(trial):
    """Cada trial prueba una configuracion de hiperparametros del GA con una
    corrida corta y regresa el mejor fitness obtenido (a minimizar)."""
    popsize = trial.suggest_int('popsize', 40, 80, step=2)
    cxpb = trial.suggest_float('cxpb', 0.6, 0.95)
    mutpb = trial.suggest_float('mutpb', 0.05, 0.30)
    eta_c = trial.suggest_float('eta_c', 5.0, 30.0)
    eta_m = trial.suggest_float('eta_m', 10.0, 40.0)
    tournsize = trial.suggest_int('tournsize', 2, 4)

    # En la busqueda corta no hacemos early-stop (paciencia alta) para que
    # todos los trials usen el mismo presupuesto efectivo.
    res = _correr_ga(popsize, cxpb, mutpb, eta_c, eta_m, tournsize,
                     ngen=NGEN_OPTUNA, seed=SEED, paciencia=NGEN_OPTUNA + 1)
    return float(res.fun)


def sintonizar():
    print('=' * 64)
    print(' ETAPA 1: SINTONIZACION DE HIPERPARAMETROS - GA (Optuna/TPE)')
    print('=' * 64)
    print(f'   trials: {N_TRIALS_OPTUNA}   ngen por trial: {NGEN_OPTUNA}')
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
    print(' ETAPA 2: CORRIDA PROFUNDA - ALGORITMO GENETICO')
    print('=' * 64)
    print(f'   parametros : {len(M.bounds)}   popsize: {bp["popsize"]}   '
          f'ngen: {NGEN}   seed: {SEED}')

    t0 = time.time()
    res = _correr_ga(bp['popsize'], bp['cxpb'], bp['mutpb'], bp['eta_c'],
                     bp['eta_m'], bp['tournsize'],
                     ngen=NGEN, seed=SEED, paciencia=100, disp=True)
    dt = time.time() - t0
    print(f'\n>> GA listo en {dt:.1f}s  (fitness={res.fun:.6f}, '
          f'generaciones={res.nit}, nfev={res.nfev})')

    hp = {
        'popsize': bp['popsize'],
        'cxpb': round(bp['cxpb'], 4),
        'mutpb': round(bp['mutpb'], 4),
        'eta_c': round(bp['eta_c'], 2),
        'eta_m': round(bp['eta_m'], 2),
        'tournsize': bp['tournsize'],
        'n_elite': 2,
        'optuna_trials': N_TRIALS_OPTUNA,
        'tiempo_tune_s': round(t_tune, 1),
    }
    _reportar(res.x, dt, res.fun, res.historial, res.nfev, hp)


def _reportar(p_opt, dt, fitness, historial, nfev, hp):
    ok, info = M.validar_cinematica(p_opt, verbose=True)

    metr = M.evaluar(p_opt)
    if metr is None:
        print('>> OJO: el resultado no produce cinematica valida.')
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

    hp_str = '  '.join(f'{k}={v}' for k, v in hp.items())
    cab = ('Parametros optimizados (16) - Modelo SIMPLIFICADO (falange medial) '
           '- ALGORITMO GENETICO\n'
           f'fitness={fitness:.6f}  err_global_mm={metr["err_global_mm"]:.4f}  '
           f'err_ifp_mm={metr["err_ifp_mm"]:.4f}  err_ifd_mm={metr["err_ifd_mm"]:.4f}  '
           f'tiempo_s={dt:.1f}  nfev={nfev}\n'
           f'HIPERPARAMETROS (Optuna): {hp_str}\n'
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

    print(f'\n>> Todo guardado en {OUTDIR}/')


if __name__ == '__main__':
    main()
