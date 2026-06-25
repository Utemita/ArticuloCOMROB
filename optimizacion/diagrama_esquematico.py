# -*- coding: utf-8 -*-
"""
Diagrama esquematico (idealizado) del mecanismo completo del exoesqueleto.

Esta figura NO se calcula a partir de la cinematica: es un esquema limpio,
vectorial, que reproduce la topologia y el etiquetado del mecanismo completo
(falanges proximal Fp, medial Fm y distal Fd, cadena de eslabones L1-L10 por
encima del dorso, bancadas B1/B2 y los angulos de entrada). Se usa con fines
ilustrativos en el articulo.

Se exporta en PDF (vectorial, para el documento) y en PNG de alta resolucion
(para previsualizacion).
"""
import os
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Arc

# Tipografia vectorial (texto editable en el PDF) y buena calidad
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
matplotlib.rcParams["svg.fonttype"] = "none"
matplotlib.rcParams["mathtext.fontset"] = "cm"

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_PDF = os.path.join(HERE, "resultados", "diagrama_modelo_simplificado.pdf")
OUT_PNG = os.path.join(HERE, "resultados", "diagrama_modelo_simplificado.png")

# ---------------------------------------------------------------------------
# Coordenadas de los nodos (x a la derecha, y hacia arriba).
# ---------------------------------------------------------------------------
P = {
    # Etapa de entrada y primeras barras (lado derecho)
    "B":  (7.00, 9.15),   # nodo superior centro-izquierda (L6, L8, L7)
    "C":  (13.20, 9.28),  # nodo superior derecha (L6, L2, L3)
    "D":  (16.20, 5.78),  # nodo extremo derecho (L1, L2)
    "E":  (10.75, 7.88),  # nodo central (L3, L5, L4)
    "F":  (11.50, 5.82),  # pivote a tierra izq (theta1, L4) -- bancada
    "G":  (13.70, 5.78),  # pivote a tierra der (theta2, L1) -- bancada
    "H":  (8.05, 5.20),   # extremo inferior de L7
    "I":  (9.45, 5.30),   # extremo inferior de L5
    "M":  (11.50, 3.25),  # pivote a tierra inferior sobre Fp (d_sp) -- bancada
    "Hf": (8.05, 3.25),   # pie de la vertical h_sp sobre Fp
    "If": (9.45, 3.25),   # pie de la vertical de L5 sobre Fp
    # Cadena distal (lado izquierdo)
    "A":  (1.80, 6.50),   # hub superior izq (L8, L10, L9, ref. punteada)
    "K":  (6.10, 3.25),   # union Fp-Fm (vertice de theta_aux_fm)
    "Pp": (0.85, 2.65),   # extremo inferior izq de L10
    "Q":  (3.20, 2.45),   # hub distal (Fm, Fd, c2, L10-base) -- theta_aux_fd
    "T":  (1.30, 0.50),   # extremo distal de Fd
    "R":  (4.05, 4.75),   # nodo c2 (sobre la linea de referencia)
}


def pt(name):
    return np.array(P[name], dtype=float)


fig, ax = plt.subplots(figsize=(13.0, 7.6))
ax.set_aspect("equal")
ax.axis("off")

LW = 3.0          # grosor de los eslabones
RJ = 0.16         # radio de las juntas
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
    """Simbolo de bancada (triangulo + rayado) por debajo del pivote."""
    p = pt(name)
    bx, by = p[0], p[1] - RJ
    ax.plot([bx, bx - size * 0.6, bx + size * 0.6, bx],
            [by, by - size, by - size, by], "-", color=COL, lw=1.6, zorder=3)
    base_y = by - size
    ax.plot([bx - size * 0.9, bx + size * 0.9], [base_y, base_y],
            "-", color=COL, lw=1.6, zorder=3)
    xs = np.linspace(bx - size * 0.75, bx + size * 0.75, 6)
    for x in xs:
        ax.plot([x, x - size * 0.35], [base_y, base_y - size * 0.35],
                "-", color=COL, lw=1.1, zorder=3)


def lbl(x, y, s, fs=21, ha="center", va="center", style="italic"):
    ax.text(x, y, s, fontsize=fs, ha=ha, va=va, fontstyle=style, zorder=6)


def mid(a, b):
    return (pt(a) + pt(b)) / 2.0


# ---------------------------------------------------------------------------
# Eslabones de la cadena
# ---------------------------------------------------------------------------
link("G", "D")           # L1
link("D", "C")           # L2
link("C", "E")           # L3
link("E", "F")           # L4
link("E", "I")           # L5
link("B", "C")           # L6
link("B", "H")           # L7
link("A", "B")           # L8
link("H", "Hf")          # vertical h_sp
link("I", "If")          # vertical (pie de L5)
link("K", "M", lw=LW + 0.4)   # falange proximal Fp

# Bancada B2 = d (vertical entre los pivotes F y M)
pf, pm = pt("F"), pt("M")
ax.plot([pf[0], pm[0]], [pf[1] - RJ, pm[1] + RJ], "-", color=COL, lw=1.6,
        zorder=1)

# Cadena distal (etapas que orientan Fm y Fd)
link("A", "Pp")          # L10
link("A", "R")           # L9
link("R", "Q")           # c2 (eslabon corto)
link("Pp", "Q")          # base de la etapa L10
link("K", "Q", lw=LW + 0.4)   # falange medial Fm
link("Q", "T", lw=LW + 0.4)   # falange distal Fd

# ---------------------------------------------------------------------------
# Lineas de referencia punteadas (para los angulos auxiliares)
# ---------------------------------------------------------------------------
# Referencia de theta_aux_fm: recta A -> K prolongada un poco
da = pt("K") - pt("A")
da = da / np.hypot(*da)
ref0 = pt("A")
ref1 = pt("K") + da * 0.5
ax.plot([ref0[0], ref1[0]], [ref0[1], ref1[1]], "--", color=COL, lw=1.4,
        zorder=1)

# Referencia de theta_aux_fd en Q (recta tendida hacia la izquierda)
ax.plot([pt("Q")[0] + 0.3, -0.4], [pt("Q")[1] + 0.05, pt("Q")[1] - 0.55],
        "--", color=COL, lw=1.4, zorder=1)

# ---------------------------------------------------------------------------
# Juntas y bancadas
# ---------------------------------------------------------------------------
for nm in ["A", "B", "C", "D", "E", "F", "G", "H", "I", "K", "M",
           "Pp", "Q", "T", "R"]:
    joint(nm)
for nm in ["F", "G", "M"]:
    ground(nm)

# ---------------------------------------------------------------------------
# Etiquetas de eslabones (reubicadas para no encimarse)
# ---------------------------------------------------------------------------
lbl(*(mid("G", "D") + [0.05, -0.55]), r"$L_1$")
lbl(*(mid("D", "C") + [0.55, 0.15]), r"$L_2$")
lbl(*(mid("C", "E") + [0.60, 0.30]), r"$L_3$")
lbl(*(mid("B", "C") + [0.00, 0.45]), r"$L_6$")
lbl(*(mid("A", "B") + [-0.10, 0.55]), r"$L_8$")
lbl(*(mid("B", "H") + [-0.45, 0.30]), r"$L_7$")
lbl(*(mid("E", "I") + [-0.50, 0.25]), r"$L_5$")
lbl(pt("E")[0] - 0.55, pt("E")[1] - 1.05, r"$L_4$")
lbl(*(mid("A", "Pp") + [-0.55, 0.10]), r"$L_{10}$")
lbl(*(mid("A", "R") + [-0.30, 0.45]), r"$L_9$")
lbl(pt("R")[0] + 0.55, pt("R")[1] + 0.20, r"$c_2$")
lbl(*(mid("K", "Q") + [0.10, -0.55]), r"$F_m$")
lbl(*(mid("Q", "T") + [0.95, 0.05]), r"$F_d$")

# Fp y d_sp sobre la falange proximal
lbl((pt("K")[0] + pt("Hf")[0]) / 2, pt("K")[1] - 0.55, r"$d_{sp}$")
lbl((pt("If")[0] + pt("M")[0]) / 2, pt("M")[1] - 0.55, r"$d_{sp}$")
lbl((pt("Hf")[0] + pt("If")[0]) / 2 - 0.1, pt("Hf")[1] - 0.60, r"$F_p$", fs=23)

# h_sp junto a la vertical de L7
lbl(pt("H")[0] + 0.55, (pt("H")[1] + pt("Hf")[1]) / 2, r"$h_{sp}$")

# B2 = d (cota vertical)
lbl(pt("F")[0] - 1.05, (pt("F")[1] + pt("M")[1]) / 2 + 0.05, r"$B_2=d$")

# ---------------------------------------------------------------------------
# Cotas B1 y r3 (flechas dobles horizontales entre pivotes a tierra)
# ---------------------------------------------------------------------------
def dim_arrow(x0, x1, y, label, dy=0.0):
    ax.annotate("", xy=(x1, y), xytext=(x0, y),
                arrowprops=dict(arrowstyle="<->", color=COL, lw=1.4))
    lbl((x0 + x1) / 2, y + dy, label)


yb1 = pt("F")[1] - 0.62
dim_arrow(pt("F")[0], pt("G")[0], yb1, r"$B_1$", dy=0.24)
dim_arrow(pt("F")[0], pt("F")[0] + 0.55 * (pt("G")[0] - pt("F")[0]),
          yb1 - 0.58, r"$r_3$", dy=-0.34)

# ---------------------------------------------------------------------------
# Angulos de entrada y marco de referencia local
# ---------------------------------------------------------------------------
# theta_1_inicial (en el pivote F)
arc1 = Arc((pt("F")[0], pt("F")[1] + 0.05), 1.5, 1.5, angle=0,
           theta1=60, theta2=140, color=COL, lw=1.4)
ax.add_patch(arc1)
ax.annotate("", xy=(pt("F")[0] - 0.55, pt("F")[1] + 0.55),
            xytext=(pt("F")[0] - 0.30, pt("F")[1] + 0.70),
            arrowprops=dict(arrowstyle="-|>", color=COL, lw=1.4))
lbl(pt("F")[0] + 0.10, pt("F")[1] + 1.20, r"$\theta_{1_{inicial}}$", fs=18)

# marco de referencia local (x-y) junto a theta_1
fx, fy = pt("F")[0] + 0.70, pt("F")[1] + 0.35
ax.annotate("", xy=(fx, fy + 0.60), xytext=(fx, fy),
            arrowprops=dict(arrowstyle="-|>", color=COL, lw=1.3))
ax.annotate("", xy=(fx + 0.60, fy), xytext=(fx, fy),
            arrowprops=dict(arrowstyle="-|>", color=COL, lw=1.3))

# theta_2_inicial (en el pivote G)
lbl(pt("G")[0] + 0.35, pt("G")[1] + 0.75, r"$\theta_{2_{inicial}}$", fs=18,
    ha="left")

# theta_1m4B_inicial (en el pivote inferior M, con referencia punteada)
ax.plot([pt("M")[0], pt("M")[0] + 2.3], [pt("M")[1], pt("M")[1]], "--",
        color=COL, lw=1.3, zorder=1)
arc3 = Arc((pt("M")[0], pt("M")[1]), 1.6, 1.6, angle=0,
           theta1=8, theta2=95, color=COL, lw=1.4)
ax.add_patch(arc3)
ax.annotate("", xy=(pt("M")[0] + 0.10, pt("M")[1] + 0.80),
            xytext=(pt("M")[0] + 0.35, pt("M")[1] + 0.72),
            arrowprops=dict(arrowstyle="-|>", color=COL, lw=1.4))
lbl(pt("M")[0] + 1.45, pt("M")[1] + 0.65, r"$\theta_{1m4B_{inicial}}$", fs=17,
    ha="left")

# theta_aux_fm (en la union K, entre la referencia y Fm)
arc4 = Arc((pt("K")[0], pt("K")[1]), 1.25, 1.25, angle=0,
           theta1=148, theta2=200, color=COL, lw=1.4)
ax.add_patch(arc4)
ax.annotate("", xy=(pt("K")[0] - 0.62, pt("K")[1] + 0.36),
            xytext=(pt("K")[0] - 0.66, pt("K")[1] - 0.22),
            arrowprops=dict(arrowstyle="<|-|>", color=COL, lw=1.3))
lbl(pt("K")[0] + 0.15, pt("K")[1] + 1.05, r"$\theta_{aux_{fm}}$", fs=17,
    ha="left")

# theta_aux_fd (en el hub distal Q, entre la referencia y Fd)
arc5 = Arc((pt("Q")[0], pt("Q")[1]), 1.05, 1.05, angle=0,
           theta1=188, theta2=232, color=COL, lw=1.4)
ax.add_patch(arc5)
ax.annotate("", xy=(pt("Q")[0] - 0.45, pt("Q")[1] - 0.30),
            xytext=(pt("Q")[0] - 0.52, pt("Q")[1] - 0.05),
            arrowprops=dict(arrowstyle="<|-|>", color=COL, lw=1.3))
lbl(pt("Q")[0] - 1.55, pt("Q")[1] - 0.95, r"$\theta_{aux_{fd}}$", fs=17,
    ha="left")

# ---------------------------------------------------------------------------
# Margenes y guardado (PDF vectorial + PNG de alta resolucion)
# ---------------------------------------------------------------------------
ax.set_xlim(-0.8, 17.4)
ax.set_ylim(0.0, 10.4)
plt.tight_layout(pad=0.3)
fig.savefig(OUT_PDF, bbox_inches="tight", facecolor="white")
fig.savefig(OUT_PNG, dpi=300, bbox_inches="tight", facecolor="white")
print(">> Diagrama (PDF vectorial) guardado en:", OUT_PDF)
print(">> Diagrama (PNG alta resolucion) guardado en:", OUT_PNG)
