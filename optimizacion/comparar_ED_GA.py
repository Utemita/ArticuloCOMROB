"""
comparar_ED_GA.py
=================
Script que genera el material comparativo pa el articulo COMROB. Toma los
resultados del modelo SIMPLIFICADO (falange medial) que sacaron:
  - optimizar_simplificado_ED.py  (Evolucion Diferencial)
  - optimizar_simplificado_GA.py  (Algoritmo Genetico)

Lo que produce:
  - resultados/comparacion_convergencia.png : curvas de convergencia ED vs GA.
  - resultados/comparacion_resumen.csv      : tabla resumen (error, tiempo, nfev).
  - Resumen impreso en consola.

Ojo: hay que correr primero los dos optimizadores del simplificado, si no
este script truena porque no encuentra los archivos.
"""
import os
import re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

OUTDIR = os.path.join(os.path.dirname(__file__), 'resultados')


def _leer_cabecera(path):
    """Saca los pares clave=valor de la cabecera de un .txt de parametros."""
    datos = {}
    with open(path) as fh:
        for linea in fh:
            if not linea.startswith('#'):
                break
            for m in re.finditer(r'(\w+)=([-\d.eE]+)', linea):
                datos[m.group(1)] = float(m.group(2))
    return datos


def main():
    ed_txt = os.path.join(OUTDIR, 'parametros_simplificado_ED.txt')
    ga_txt = os.path.join(OUTDIR, 'parametros_simplificado_GA.txt')
    ed_conv = os.path.join(OUTDIR, 'convergencia_simplificado_ED.csv')
    ga_conv = os.path.join(OUTDIR, 'convergencia_simplificado_GA.csv')

    # Checamos que existan los archivos necesarios
    for f in (ed_txt, ga_txt, ed_conv, ga_conv):
        if not os.path.exists(f):
            raise SystemExit(f">> Falta {f}. Corre primero los optimizadores "
                             "simplificados ED y GA.")

    ed = _leer_cabecera(ed_txt)
    ga = _leer_cabecera(ga_txt)

    resumen = pd.DataFrame([
        {'metodo': 'Evolucion Diferencial',
         'error_global_mm': ed.get('err_global_mm'),
         'error_ifp_mm': ed.get('err_ifp_mm'),
         'error_ifd_mm': ed.get('err_ifd_mm'),
         'fitness': ed.get('fitness'),
         'tiempo_s': ed.get('tiempo_s'),
         'nfev': ed.get('nfev')},
        {'metodo': 'Algoritmo Genetico',
         'error_global_mm': ga.get('err_global_mm'),
         'error_ifp_mm': ga.get('err_ifp_mm'),
         'error_ifd_mm': ga.get('err_ifd_mm'),
         'fitness': ga.get('fitness'),
         'tiempo_s': ga.get('tiempo_s'),
         'nfev': ga.get('nfev')},
    ])
    resumen.to_csv(os.path.join(OUTDIR, 'comparacion_resumen.csv'), index=False)

    print('=' * 64)
    print(' COMPARACION ED vs GA - Modelo simplificado (falange medial)')
    print('=' * 64)
    print(resumen.to_string(index=False))

    # --- Curvas de convergencia ---
    ed_c = pd.read_csv(ed_conv)
    ga_c = pd.read_csv(ga_conv)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(ed_c.iloc[:, 0], ed_c['mejor_fitness'], 'b-', lw=2,
            label='Evolucion Diferencial')
    ax.plot(ga_c.iloc[:, 0], ga_c['mejor_fitness'], 'r-', lw=2,
            label='Algoritmo Genetico')
    ax.set_yscale('log')
    ax.set_xlabel('Iteracion / Generacion', fontsize=12)
    ax.set_ylabel('Mejor valor de la funcion objetivo (log)', fontsize=12)
    ax.set_title('Convergencia: Evolucion Diferencial vs Algoritmo Genetico\n'
                 'Sintesis del exoesqueleto (modelo simplificado)',
                 fontsize=13, fontweight='bold')
    ax.grid(True, which='both', ls=':', alpha=0.6)
    ax.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, 'comparacion_convergencia.png'), dpi=150)
    plt.close()

    print(f'\n>> Listo. Guardados: comparacion_resumen.csv y '
          f'comparacion_convergencia.png en {OUTDIR}/')


if __name__ == '__main__':
    main()
