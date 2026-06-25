"""
figuras_articulo.py
===================
Genera las figuras comparativas para el articulo COMROB (ED vs GA), modelo
simplificado (hasta la falange medial). Sale todo a resultados/.

Figuras:
  1. comparacion_dimensiones.png : longitudes de eslabon (13 params de longitud)
     para 3 disenos -> baseline del disenador (CAD original), ED y GA, en mm.
  2. comparacion_errores.png     : barras de error (global / IFP / IFD) ED vs GA
     + panel de presupuesto (evaluaciones y tiempo).

Las cifras vienen de:
  resultados/parametros_simplificado_ED.txt
  resultados/parametros_simplificado_GA.txt
  resultados/comparacion_resumen.csv
y el baseline del disenador del vector p_matlab de exo_18_pinza_fina.py
(mapeado a los 16 parametros del modelo simplificado).
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "resultados")


def _load_params(fname):
    vals = []
    with open(os.path.join(OUT, fname)) as fh:
        for line in fh:
            line = line.strip()
            if line and not line.startswith("#"):
                vals.append(float(line))
    return np.array(vals)


# 16 parametros: [Bancada1, Bancada2, Link1..Link8, c2, hsp, dsp,
#                 theta_aux_fm, gear_ratio, theta_offset]
P_ED = _load_params("parametros_simplificado_ED.txt")
P_GA = _load_params("parametros_simplificado_GA.txt")

# Baseline del disenador (p_matlab de exo_18_pinza_fina.py), mapeado a los
# 16 parametros simplificados (se omiten theta_aux_fd y el 3er mecanismo):
#   Bancada1, Bancada2, Link1..Link8, c2, hsp, dsp, theta_aux_fm, gear, offset
P_BASE = np.array([
    0.018, 0.020, 0.035, 0.049, 0.025, 0.020, 0.025, 0.055, 0.035, 0.052,
    0.04601, 0.017, 0.018, np.deg2rad(51.39), 2.0, np.deg2rad(109.0),
])

# Etiquetas de los 13 parametros de longitud (en mm)
LEN_LABELS = [r"$B_1$", r"$B_2$", r"$L_1$", r"$L_2$", r"$L_3$", r"$L_4$",
              r"$L_5$", r"$L_6$", r"$L_7$", r"$L_8$", r"$c_2$",
              r"$h_{sp}$", r"$d_{sp}$"]
LEN_IDX = list(range(0, 13))  # primeros 13 son longitudes


def fig_dimensiones():
    base = P_BASE[LEN_IDX] * 1000.0
    ed = P_ED[LEN_IDX] * 1000.0
    ga = P_GA[LEN_IDX] * 1000.0

    x = np.arange(len(LEN_LABELS))
    w = 0.27
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - w, base, w, label="Diseno CAD (sin optimizar)",
           color="#9e9e9e", edgecolor="black", linewidth=0.5)
    ax.bar(x, ed, w, label="Evolucion Diferencial (ED)",
           color="#1f77b4", edgecolor="black", linewidth=0.5)
    ax.bar(x + w, ga, w, label="Algoritmo Genetico (AG)",
           color="#ff7f0e", edgecolor="black", linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(LEN_LABELS, fontsize=11)
    ax.set_ylabel("Longitud (mm)", fontsize=11)
    ax.set_title("Dimensiones de los eslabones: diseno CAD vs. optimizado (ED y AG)",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=9, framealpha=0.9)
    ax.grid(True, axis="y", ls=":", alpha=0.6)
    ax.axhline(60, color="red", ls="--", lw=1.0, alpha=0.7)
    ax.text(len(x) - 0.5, 61, "limite superior 60 mm", color="red",
            fontsize=8, ha="right", va="bottom")
    ax.set_ylim(0, 66)

    # Suma total de longitudes (compacidad)
    sums = [base.sum(), ed.sum(), ga.sum()]
    txt = (f"Longitud total (13 param.):  CAD = {sums[0]:.0f} mm   |   "
           f"ED = {sums[1]:.0f} mm   |   AG = {sums[2]:.0f} mm")
    ax.text(0.5, -0.22, txt, transform=ax.transAxes, ha="center",
            fontsize=9, style="italic")

    plt.tight_layout()
    out = os.path.join(OUT, "comparacion_dimensiones.png")
    plt.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()
    print(">> guardado:", out)


def fig_errores():
    # Errores (mm) -> grafico unico de barras (sin panel de presupuesto)
    ed_err = [2.1523, 2.6564, 1.6482]   # global, ifp, ifd
    ga_err = [5.5486, 3.9250, 7.1721]
    cats = ["Global", "IFP", "IFD"]

    fig, ax1 = plt.subplots(figsize=(6.2, 4.0))

    x = np.arange(len(cats))
    w = 0.36
    b1 = ax1.bar(x - w / 2, ed_err, w, label="Evolucion Diferencial",
                 color="#1f77b4", edgecolor="black", linewidth=0.5)
    b2 = ax1.bar(x + w / 2, ga_err, w, label="Algoritmo Genetico",
                 color="#ff7f0e", edgecolor="black", linewidth=0.5)
    ax1.bar_label(b1, fmt="%.2f", fontsize=10, padding=2)
    ax1.bar_label(b2, fmt="%.2f", fontsize=10, padding=2)
    ax1.set_xticks(x)
    ax1.set_xticklabels(cats, fontsize=12)
    ax1.set_ylabel("Error de forma (Chamfer) [mm]", fontsize=11)
    ax1.legend(fontsize=10)
    ax1.grid(True, axis="y", ls=":", alpha=0.6)
    ax1.set_ylim(0, 8.2)

    plt.tight_layout()
    out = os.path.join(OUT, "comparacion_errores.png")
    plt.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()
    print(">> guardado:", out)


if __name__ == "__main__":
    fig_dimensiones()
    fig_errores()
