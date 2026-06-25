# -*- coding: utf-8 -*-
"""
Diagrama esquematico (idealizado) del mecanismo simplificado del exoesqueleto.

Esta figura NO se calcula a partir de la cinematica: es un esquema limpio,
trazado a mano, que reproduce la topologia y el etiquetado del mecanismo
(falange proximal Fp y medial Fm, cadena de eslabones L1-L8 por encima del
dorso, bancadas B1/B2 y los angulos de entrada). Se usa unicamente con fines
ilustrativos en el articulo.
"""
import os
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Arc, FancyArrowPatch

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_RES = os.path.join(HERE, "resultados", "diagrama_modelo_simplificado.png")

# ---------------------------------------------------------------------------
# Coordenadas de los nodos (x a la derecha, y hacia arriba).
# ---------------------------------------------------------------------------
P = {
    "A":  (1.80, 6.74),   # nodo extremo izquierdo (L8 izq / cuatro barras distal)
    "B":  (6.95, 9.39),   # nodo superior centro-izquierda (L6, L8, L7)
    "C":  (13.20, 9.52),  # nodo superior derecha (L6, L2, L3)
    "D":  (16.10, 5.94),  # nodo extremo derecho (L1, L2)
    "E":  (10.75, 8.12),  # nodo central (L3, L5, L4)
    "F":  (11.50, 6.04),  # pivote a tierra izq (theta1, L4) -- bancada
    "G":  (13.70, 6.02),  # pivote a tierra der (theta2, L1) -- bancada
    "H":  (8.05, 5.44),   # extremo inferior de L7
    "I":  (9.45, 5.54),   # extremo inferior de L5
    "J":  (4.05, 4.96),   # nodo cuatro barras distal
    "K":  (6.10, 3.46),   # union Fp-Fm (vertice de theta_aux_fm)
    "L":  (3.20, 2.69),   # extremo distal de Fm
    "M":  (11.50, 3.46),  # pivote a tierra inferior sobre Fp (d_sp) -- bancada
    "Hf": (8.05, 3.46),   # pie de la vertical h_sp sobre Fp
    "If": (9.45, 3.46),   # pie de la vertical de L5 sobre Fp
}


def pt(name):
    return np.array(P[name], dtype=float)


fig, ax = plt.subplots(figsize=(11.0, 6.6))
ax.set_aspect("equal")
ax.axis("off")

LW = 3.0          # grosor de los eslabones
RJ = 0.17         # radio de las juntas
COL = "black"


def link(a, b, lw=LW):
    pa, pb = pt(a), pt(b)
    ax.plot([pa[0], pb[0]], [pa[1], pb[1]], "-", color=COL, lw=lw,
            solid_capstyle="round", zorder=2)


def joint(name):
    p = pt(name)
    c = plt.Circle((p[0], p[1]), RJ, facecolor="white", edgecolor=COL,
                   lw=2.0, zorder=4)
    ax.add_patch(c)


def ground(name, size=0.42):
    """Dibuja el simbolo de bancada (triangulo + rayado) por debajo del pivote."""
    p = pt(name)
    bx, by = p[0], p[1] - RJ
    # triangulo
    ax.plot([bx, bx - size * 0.6, bx + size * 0.6, bx],
            [by, by - size, by - size, by], "-", color=COL, lw=1.6, zorder=3)
    # base
    base_y = by - size
    ax.plot([bx - size * 0.9, bx + size * 0.9], [base_y, base_y],
            "-", color=COL, lw=1.6, zorder=3)
    # rayado
    n = 6
    xs = np.linspace(bx - size * 0.75, bx + size * 0.75, n)
    for x in xs:
        ax.plot([x, x - size * 0.35], [base_y, base_y - size * 0.35],
                "-", color=COL, lw=1.1, zorder=3)


def lbl(x, y, s, fs=20, ha="center", va="center", style="italic"):
    ax.text(x, y, s, fontsize=fs, ha=ha, va=va, fontstyle=style,
            zorder=6)


# ---------------------------------------------------------------------------
# Eslabones de la cadena (cinco barras + cuatro barras de las dos etapas)
# ---------------------------------------------------------------------------
link("G", "D")           # L1
link("D", "C")           # L2
link("C", "E")           # L3
link("B", "C")           # L6
link("A", "B")           # L8
link("B", "H")           # L7
link("E", "I")           # L5
link("E", "F")           # L4
link("H", "Hf")          # vertical h_sp
link("I", "If")          # vertical (pie de L5)

# Falange proximal Fp (linea horizontal inferior)
link("K", "M", lw=LW + 0.4)

# Bancada B2 = d (vertical entre los dos pivotes a tierra F y M)
pf, pm = pt("F"), pt("M")
ax.plot([pf[0], pm[0]], [pf[1] - RJ, pm[1] + RJ], "-", color=COL, lw=1.6,
        zorder=1)

# Cuatro barras distal (Fm)
link("A", "J")           # lado superior izquierdo
link("J", "L")           # c2 (eslabon corto)
link("L", "K", lw=LW + 0.4)  # Fm (falange medial)
link("K", "A")           # acoplador (con theta_aux_fm en K)

# ---------------------------------------------------------------------------
# Juntas
# ---------------------------------------------------------------------------
for nm in ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M"]:
    joint(nm)

# Bancadas
ground("F")
ground("G")
ground("M")

# ---------------------------------------------------------------------------
# Etiquetas de eslabones
# ---------------------------------------------------------------------------
def mid(a, b):
    pa, pb = pt(a), pt(b)
    return (pa + pb) / 2.0

lbl(*(mid("G", "D") + [0.0, -0.55]), r"$L_1$")
lbl(*(mid("D", "C") + [0.55, 0.0]), r"$L_2$")
lbl(*(mid("C", "E") + [0.55, 0.25]), r"$L_3$")
lbl(*(mid("B", "C") + [0.0, 0.45]), r"$L_6$")
lbl(*(mid("A", "B") + [-0.2, 0.5]), r"$L_8$")
lbl(*(mid("B", "H") + [-0.45, 0.3]), r"$L_7$")
lbl(*(mid("E", "I") + [-0.45, 0.2]), r"$L_5$")
lbl(*(mid("E", "F") + [0.45, 0.1]), r"$L_4$")
lbl(*(mid("J", "L") + [-0.5, 0.05]), r"$c_2$")
lbl(*(mid("L", "K") + [0.15, -0.55]), r"$F_m$")

# Fp y d_sp sobre la falange proximal
lbl((pt("K")[0] + pt("Hf")[0]) / 2, pt("K")[1] - 0.55, r"$d_{sp}$")
lbl((pt("If")[0] + pt("M")[0]) / 2, pt("M")[1] - 0.55, r"$d_{sp}$")
lbl((pt("Hf")[0] + pt("If")[0]) / 2, pt("Hf")[1] - 0.58, r"$F_p$", fs=22)

# h_sp junto a la vertical
lbl(pt("H")[0] + 0.55, (pt("H")[1] + pt("Hf")[1]) / 2, r"$h_{sp}$")

# B2 = d
lbl(pt("F")[0] - 0.95, (pt("F")[1] + pt("M")[1]) / 2 + 0.1, r"$B_2=d$")

# ---------------------------------------------------------------------------
# Cotas B1 y r3 (flechas dobles horizontales entre pivotes a tierra)
# ---------------------------------------------------------------------------
def dim_arrow(x0, x1, y, label, dy=0.0):
    ax.annotate("", xy=(x1, y), xytext=(x0, y),
                arrowprops=dict(arrowstyle="<->", color=COL, lw=1.4))
    lbl((x0 + x1) / 2, y + dy, label)

yb1 = pt("F")[1] - 0.62
dim_arrow(pt("F")[0], pt("G")[0], yb1, r"$B_1$", dy=0.22)
dim_arrow(pt("F")[0], pt("F")[0] + 0.55 * (pt("G")[0] - pt("F")[0]),
          yb1 - 0.55, r"$r_3$", dy=-0.32)

# ---------------------------------------------------------------------------
# Angulos de entrada (arcos con flecha) y marco de referencia local
# ---------------------------------------------------------------------------
# theta_1_inicial (en el pivote F)
arc1 = Arc((pt("F")[0], pt("F")[1] + 0.05), 1.5, 1.5, angle=0,
           theta1=60, theta2=140, color=COL, lw=1.4)
ax.add_patch(arc1)
ax.annotate("", xy=(pt("F")[0] - 0.55, pt("F")[1] + 0.55),
            xytext=(pt("F")[0] - 0.30, pt("F")[1] + 0.70),
            arrowprops=dict(arrowstyle="-|>", color=COL, lw=1.4))
lbl(pt("F")[0] + 0.05, pt("F")[1] + 1.15, r"$\theta_{1_{inicial}}$", fs=17)

# marco de referencia local (x-y) junto a theta_1
fx, fy = pt("F")[0] + 0.65, pt("F")[1] + 0.35
ax.annotate("", xy=(fx, fy + 0.6), xytext=(fx, fy),
            arrowprops=dict(arrowstyle="-|>", color=COL, lw=1.3))
ax.annotate("", xy=(fx + 0.6, fy), xytext=(fx, fy),
            arrowprops=dict(arrowstyle="-|>", color=COL, lw=1.3))

# theta_2_inicial (en el pivote G)
lbl(pt("G")[0] + 0.35, pt("G")[1] + 0.7, r"$\theta_{2_{inicial}}$", fs=17,
    ha="left")
lbl(pt("G")[0] + 1.05, pt("G")[1] - 0.55, r"$B_1$", fs=18)

# theta_1m4B_inicial (en el pivote inferior M, con linea de referencia punteada)
ax.plot([pt("M")[0], pt("M")[0] + 2.2], [pt("M")[1], pt("M")[1]], "--",
        color=COL, lw=1.3, zorder=1)
arc3 = Arc((pt("M")[0], pt("M")[1]), 1.6, 1.6, angle=0,
           theta1=8, theta2=95, color=COL, lw=1.4)
ax.add_patch(arc3)
ax.annotate("", xy=(pt("M")[0] + 0.10, pt("M")[1] + 0.80),
            xytext=(pt("M")[0] + 0.35, pt("M")[1] + 0.72),
            arrowprops=dict(arrowstyle="-|>", color=COL, lw=1.4))
lbl(pt("M")[0] + 1.35, pt("M")[1] + 0.6, r"$\theta_{1m4B_{inicial}}$", fs=16,
    ha="left")

# theta_aux_fm (en la union K, entre Fm y el acoplador)
arc4 = Arc((pt("K")[0], pt("K")[1]), 1.25, 1.25, angle=0,
           theta1=152, theta2=212, color=COL, lw=1.4)
ax.add_patch(arc4)
# flecha doble del angulo
ax.annotate("", xy=(pt("K")[0] - 0.62, pt("K")[1] + 0.30),
            xytext=(pt("K")[0] - 0.66, pt("K")[1] - 0.28),
            arrowprops=dict(arrowstyle="<|-|>", color=COL, lw=1.3))
lbl(pt("K")[0] - 0.30, pt("K")[1] + 0.55, r"$\theta_{aux_{fm}}$", fs=16,
    ha="left")

# ---------------------------------------------------------------------------
# Margenes y guardado
# ---------------------------------------------------------------------------
ax.set_xlim(0.4, 17.4)
ax.set_ylim(1.4, 10.6)
plt.tight_layout(pad=0.4)
fig.savefig(OUT_RES, dpi=200, bbox_inches="tight", facecolor="white")
print(">> Diagrama esquematico guardado en:", OUT_RES)
