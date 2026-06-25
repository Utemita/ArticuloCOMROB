"""
diagrama_modelo_simplificado.py
===============================
Diagrama cinematico del modelo SIMPLIFICADO del exo (hasta la falange medial).

Las posiciones de TODAS las juntas se obtienen de la CINEMATICA REAL del modelo
(`modelo_simplificado.pose_juntas`), la MISMA funcion que usa el optimizador
para evaluar el mecanismo. No se reimplementa ninguna ecuacion aqui: el diagrama
y la optimizacion comparten una unica fuente de verdad, de modo que el dibujo
refleja exactamente el mecanismo que se optimizo y la pose ensambla siempre.

Para que el dibujo sea legible y compacto en el articulo, el conjunto de puntos
se reorienta SOLO con ROTACIONES rigidas (nunca con reflexiones): primero se
gira para que el dedo quede horizontal (MCF a la izquierda, IFD a la derecha) y,
si la bancada queda arriba, se aplica una rotacion adicional de 180 grados para
dejar los soportes abajo. Reflejar invertiria la quiralidad del mecanismo y lo
dibujaria espejeado (un mecanismo distinto del real), por eso NO se usa.

Estilo: dibujo tecnico B/N. Eslabones = lineas delgadas; falanges = lineas
gruesas; juntas de revoluta = circulos blancos con borde negro; soportes fijos
(tierra) con achurado; c2 = linea de construccion punteada. El tercer mecanismo
(L9, L10) y la falange distal Fd NO se dibujan porque el modelo simplificado se
corta en la falange medial.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import modelo_simplificado as M

# =============================================================================
# 1) PARAMETROS OPTIMIZADOS (ED) Y CONSTANTES DE FALANGE (en mm para el dibujo)
# =============================================================================
HERE = os.path.dirname(os.path.abspath(__file__))
p = np.loadtxt(os.path.join(HERE, "resultados", "parametros_simplificado_ED.txt"))
fp = M.FP_REAL * 1000.0   # falange proximal (mm)
fm = M.FM_REAL * 1000.0   # falange medial   (mm)


# =============================================================================
# 2) ELEGIR UNA POSE INTERMEDIA USANDO LA CINEMATICA REAL (pose_juntas)
#    Barremos el mismo rango de la manivela del MOCAP y preferimos una pose a
#    la mitad del recorrido para que el dibujo sea representativo y legible.
# =============================================================================
_th_grid = M.theta_input
_obj = 0.5 * (_th_grid.min() + _th_grid.max())
_orden = np.argsort(np.abs(_th_grid - _obj))
est = None
for _i in _orden:
    cand = M.pose_juntas(p, float(_th_grid[_i]))
    if cand is not None:
        est = {k: np.asarray(v, dtype=float) * 1000.0      # m -> mm
               for k, v in cand.items() if isinstance(v, np.ndarray)}
        THETA2_diag = np.rad2deg(_th_grid[_i])
        break
if est is None:
    raise RuntimeError("Ninguna pose ensambla con estos parametros.")


# =============================================================================
# 2b) REORIENTACION RIGIDA (SOLO ROTACIONES, SIN REFLEXION)
# =============================================================================
def _rotacion(theta):
    c_, s_ = np.cos(theta), np.sin(theta)
    return np.array([[c_, -s_], [s_, c_]])


_PT_KEYS = ["MCF", "G1", "G2", "T2", "P", "M4", "S1", "S2",
            "IFP", "P2", "P3", "IFD"]

# Giramos para que el eje MCF->IFD quede horizontal (dedo apuntando a la derecha).
_v = est["IFD"] - est["MCF"]
_ang = -np.arctan2(_v[1], _v[0])
_R = _rotacion(_ang)
_origin = est["MCF"].copy()
for k in _PT_KEYS:
    est[k] = _R @ (est[k] - _origin) + _origin

# Si la cadena de eslabones (el mecanismo) quedo POR DEBAJO de las falanges,
# aplicamos una ROTACION rigida de 180 grados (NO una reflexion) para que la
# cadena quede POR ARRIBA del dorso del dedo, como en el dispositivo real.
# Reflejar invertiria la quiralidad del mecanismo, por eso NO se usa.
_finger_y = np.mean([est["MCF"][1], est["IFP"][1], est["IFD"][1]])
_chain_y = np.mean([est[k][1] for k in
                    ["G1", "G2", "T2", "P", "M4", "S1", "S2", "P2", "P3"]])
if _chain_y < _finger_y:
    _R180 = _rotacion(np.pi)
    for k in _PT_KEYS:
        est[k] = _R180 @ (est[k] - _origin) + _origin

MCF, IFP, IFD = est["MCF"], est["IFP"], est["IFD"]
P, P2, P3 = est["P"], est["P2"], est["P3"]
S1, S2, M4 = est["S1"], est["S2"], est["M4"]
G1, G2, T2 = est["G1"], est["G2"], est["T2"]

# Comprobacion rapida de longitudes (las del mecanismo primario y las falanges
# deben coincidir EXACTAMENTE con los parametros, porque vienen de la cinematica
# real; sirve de verificacion al regenerar la figura).
print(f">> Pose THETA2={THETA2_diag:.1f} deg")
print(f"   |MCF-IFP| = {np.linalg.norm(IFP - MCF):.1f} mm (fp={fp:.0f})")
print(f"   |IFP-IFD| = {np.linalg.norm(IFD - IFP):.1f} mm (fm={fm:.0f})")
print(f"   |G1-M4|=L4 = {np.linalg.norm(M4 - G1):.2f} mm (Link4={p[5]*1000:.2f})")
print(f"   |M4-P|=L3  = {np.linalg.norm(P - M4):.2f} mm (Link3={p[4]*1000:.2f})")


# =============================================================================
# 4) DIBUJO (estilo B/N de ingenieria, apaisado y compacto)
# =============================================================================
LW_LINK, LW_PHAL, LW_GROUND, R_JOINT = 1.4, 2.6, 1.3, 2.0
FS_LINK, FS_PT = 11, 9

pts = np.array([G1, G2, T2, P, M4, S1, S2, IFP, P2, P3, IFD, MCF])
_w = pts[:, 0].max() - pts[:, 0].min()
_h = pts[:, 1].max() - pts[:, 1].min()
_aspect = (_w + 40) / (_h + 40)
_fig_h = 3.4
fig, ax = plt.subplots(figsize=(_fig_h * _aspect, _fig_h))
fig.patch.set_facecolor("white")
ax.set_facecolor("white")


def link(p1, p2, lw=LW_LINK, ls="-", z=4):
    ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color="black",
            linewidth=lw, linestyle=ls, zorder=z, solid_capstyle="round")


def joint(pt, r=R_JOINT, z=10):
    ax.add_patch(plt.Circle((pt[0], pt[1]), r, facecolor="white",
                            edgecolor="black", linewidth=1.3, zorder=z))


def ground(pt, size=6):
    """Soporte fijo (triangulo + achurado), apuntando hacia abajo."""
    base_y = pt[1] - size
    b1 = np.array([pt[0] - size * 0.7, base_y])
    b2 = np.array([pt[0] + size * 0.7, base_y])
    link(pt, b1, lw=1.0, z=6)
    link(pt, b2, lw=1.0, z=6)
    ax.plot([b1[0], b2[0]], [b1[1], b2[1]], "k-", lw=1.0, zorder=6)
    for t in np.linspace(0, 1, 6):
        hx = b1[0] + (b2[0] - b1[0]) * t
        ax.plot([hx, hx - size * 0.35], [base_y, base_y - size * 0.45],
                "k-", lw=0.7, zorder=6)


def mid(a, b):
    return (np.asarray(a) + np.asarray(b)) / 2.0


def perp_off(p1, p2, dist):
    v = np.asarray(p2) - np.asarray(p1)
    n = np.array([-v[1], v[0]])
    n = n / (np.hypot(*n) + 1e-12)
    return n * dist


def lbl(p1, p2, text, dx=0.0, dy=0.0, fs=FS_LINK):
    ax.text((p1[0] + p2[0]) / 2 + dx, (p1[1] + p2[1]) / 2 + dy, text,
            fontsize=fs, style="italic", ha="center", va="center", zorder=20)


def lbl_at(pt, text, dx=0.0, dy=0.0, fs=FS_PT, ha="center", va="center"):
    ax.text(pt[0] + dx, pt[1] + dy, text, fontsize=fs, style="italic",
            ha=ha, va=va, zorder=20)


# --- Bancada (fija) ---
link(G1, G2, lw=LW_GROUND)                  # B1 (Bancada1)
link(G1, MCF, lw=LW_GROUND)                 # B2 = d (Bancada2)

# --- 1.er mecanismo de 5 barras ---
link(G2, T2)                                # L1
link(T2, P)                                 # L2
link(M4, P)                                 # L3
link(G1, M4)                                # L4

# --- 1.er mecanismo de 4 barras ---
link(M4, S1)                                # L5
link(MCF, S1)                               # c (rocker soporte S1)

# --- Soportes de la falange proximal (hsp / dsp) ---
link(IFP, S2)                               # strut hacia S2
link(S1, S2)                                # base de soportes

# --- 2.o mecanismo de 5 barras ---
link(P, P2)                                 # L6
link(S2, P2)                                # L7

# --- 2.o mecanismo de 4 barras ---
link(P2, P3)                                # L8
link(IFP, P3, ls=(0, (5, 3)), lw=1.0)       # c2 (linea de construccion)

# --- Falanges (hueso del dedo) ---
link(MCF, IFP, lw=LW_PHAL, z=3)             # Fp
link(IFP, IFD, lw=LW_PHAL, z=3)             # Fm

# --- Juntas de revoluta ---
for pt in [G1, G2, T2, P, M4, S1, S2, IFP, P2, P3, IFD, MCF]:
    joint(pt)

# --- Soportes fijos ---
ground(G1, size=6)
ground(G2, size=6)


# =============================================================================
# 5) ETIQUETAS DE LOS PARAMETROS OPTIMIZADOS
# =============================================================================
off = 6.0
lbl(G2, T2, r"$L_1$", *perp_off(G2, T2, off))
lbl(T2, P, r"$L_2$", *perp_off(T2, P, off))
lbl(M4, P, r"$L_3$", *perp_off(M4, P, off))
lbl(G1, M4, r"$L_4$", *perp_off(G1, M4, off))
lbl(M4, S1, r"$L_5$", *perp_off(M4, S1, -off))
lbl(P, P2, r"$L_6$", *perp_off(P, P2, off))
lbl(S2, P2, r"$L_7$", *perp_off(S2, P2, off))
lbl(P2, P3, r"$L_8$", *perp_off(P2, P3, off))
lbl(MCF, S1, r"$c$", *perp_off(MCF, S1, -off))
lbl(IFP, P3, r"$c_2$", *perp_off(IFP, P3, off))

# Bancadas
lbl(G1, G2, r"$B_1$", *perp_off(G1, G2, -off))
lbl(G1, MCF, r"$B_2$", *perp_off(G1, MCF, -off))

# Soportes de la falange proximal
lbl(S2, IFP, r"$h_{sp}$", *perp_off(S2, IFP, off))
lbl_at(mid(MCF, S1), r"$d_{sp}$", dy=-5, fs=FS_PT)

# Falanges
lbl_at(mid(MCF, IFP), r"$F_p$", *perp_off(MCF, IFP, off + 2), fs=FS_LINK)
lbl_at(mid(IFP, IFD), r"$F_m$", *perp_off(IFP, IFD, off + 2), fs=FS_LINK)

# Angulo auxiliar de la falange medial (entre c2 y Fm, en IFP)
lbl_at(IFP, r"$\theta_{aux_{fm}}$", dx=8, dy=8, fs=FS_PT)

# Articulaciones del dedo
lbl_at(MCF, "MCF", dx=-3, dy=-9, ha="right", fs=9)
lbl_at(IFP, "IFP", dx=0, dy=10, ha="center", fs=9)
lbl_at(IFD, "IFD", dx=5, dy=4, ha="left", fs=9)


# =============================================================================
# 6) AJUSTES Y GUARDADO
# =============================================================================
ax.set_aspect("equal", adjustable="datalim")
ax.axis("off")

mxr, myr = 16, 16
ax.set_xlim(pts[:, 0].min() - mxr, pts[:, 0].max() + mxr)
ax.set_ylim(pts[:, 1].min() - myr, pts[:, 1].max() + myr)

out = os.path.join(HERE, "resultados", "diagrama_modelo_simplificado.png")
plt.savefig(out, dpi=200, bbox_inches="tight", facecolor="white", pad_inches=0.08)
plt.close()
print(">> Diagrama guardado en:", out)
