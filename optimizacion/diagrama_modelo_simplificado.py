"""
diagrama_modelo_simplificado.py
===============================
Diagrama cinematico del modelo SIMPLIFICADO del exo (hasta la falange medial).

Las posiciones de TODAS las juntas se calculan con la cinematica REAL (las
mismas ecuaciones de CinematicaExoFinal.m / run_kinematics) usando las
DIMENSIONES OPTIMIZADAS por Evolucion Diferencial, asi que el diagrama refleja
la geometria que de verdad produce el mecanismo.

Estilo: dibujo tecnico B/N. Eslabones = lineas delgadas; falanges = lineas
gruesas; juntas de revoluta = circulos blancos con borde negro; soportes fijos
(tierra) con achurado; c2 = linea de construccion punteada (no es eslabon
fisico). El tercer mecanismo (L9, L10) y la falange distal Fd NO se dibujan
porque el modelo simplificado se corta en la falange medial.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# =============================================================================
# 1) PARAMETROS OPTIMIZADOS (16) -> CONSTANTES DE DISENO (en mm)
# =============================================================================
HERE = os.path.dirname(os.path.abspath(__file__))
_params = []
with open(os.path.join(HERE, "resultados", "parametros_simplificado_ED.txt")) as fh:
    for line in fh:
        line = line.strip()
        if line and not line.startswith("#"):
            _params.append(float(line))
p = np.array(_params)

Bancada1 = p[0] * 1000
Bancada2 = p[1] * 1000
Link1 = p[2] * 1000
Link2 = p[3] * 1000
Link3 = p[4] * 1000
Link4 = p[5] * 1000
Link5 = p[6] * 1000
Link6 = p[7] * 1000
Link7 = p[8] * 1000
Link8 = p[9] * 1000
c2 = p[10] * 1000               # "Link10 / c2": balancin del 2.o 4-barras
hsp = p[11] * 1000
dsp = p[12] * 1000
THETAauxfm = np.rad2deg(p[13])
reng = p[14]
TETHA1inicial = np.rad2deg(p[15])
fp = 49.0                       # falange proximal (mm)
fm = 26.0                       # falange medial   (mm)
THETA14B = 90.0

# Derivadas (identicas a la cinematica original)
r4, r5, r2, r1 = Link1, Link2, Link3, Link4
r3 = Bancada1 / 2.0
c = np.sqrt(hsp**2 + dsp**2)
d = Bancada2
r1m2 = fp - 2 * dsp
r2m2 = Link7
r3m2 = Link5 / 2.0
r4m2 = Link3
r5m2 = Link6
a2 = Link7
b2 = Link8
d2 = np.sqrt(hsp**2 + dsp**2)
theta14B = np.deg2rad(THETA14B)


# =============================================================================
# 2) CINEMATICA REAL DE UNA POSE (hasta la falange medial)
# =============================================================================
def cinematica_paso(THETA2_deg):
    theta2 = np.deg2rad(THETA2_deg)
    THETA1 = THETA2_deg / reng + TETHA1inicial
    theta1 = np.deg2rad(THETA1)

    # --- 1.er mecanismo de 5 barras ---
    den = r4 * np.cos(theta2) - r1 * np.cos(theta1) + 2 * r3
    e = (r1 * np.sin(theta1) - r4 * np.sin(theta2)) / den
    f = (2 * (r1 * r3 * np.cos(theta1) + r3 * r4 * np.cos(theta2))
         - r1**2 + r2**2 + r4**2 - r5**2) / (2 * den)
    daux = e**2 + 1
    g = 2 * (e * f - e * r1 * np.cos(theta1) + e * r3 - r1 * np.sin(theta1))
    h = (f**2 - 2 * f * (r1 * np.cos(theta1) - r3)
         - 2 * r1 * r3 * np.cos(theta1) + r1**2 + r3**2 - r2**2)
    disc = g**2 - 4 * daux * h
    if disc < 0:
        return None
    pyP = (-g + np.sqrt(disc)) / (2 * daux)
    pxP = e * pyP + f

    # --- 1.er mecanismo de 4 barras ---
    k1 = d * np.cos(theta14B) + a2 * 0 + Link4 * np.cos(theta1)
    k2 = d * np.sin(theta14B) + Link4 * np.sin(theta1)
    k3 = k1**2 + k2**2 + c**2 - Link5**2
    A1 = -k3 - 2 * k1 * c
    B1 = 4 * k2 * c
    C1 = 2 * k1 * c - k3
    discq = B1**2 - 4 * A1 * C1
    if discq < 0:
        return None
    theta4a = 2 * np.arctan((-B1 - np.sqrt(discq)) / (2 * A1))
    THETA4a = np.rad2deg(theta4a)
    if THETA4a < 0:
        THETA4a += 360

    THETAfp = THETA4a + np.rad2deg(np.arctan2(hsp, dsp))
    thetafp = np.deg2rad(THETAfp)
    pxIFP = fp * np.cos(thetafp) - d * np.cos(theta14B) - r3
    pyIFP = fp * np.sin(thetafp) - d * np.sin(theta14B)

    pxS1 = c * np.cos(theta4a) - d * np.cos(theta14B) - r3
    pyS1 = c * np.sin(theta4a) - d * np.sin(theta14B)
    THETAps2 = THETAfp - np.rad2deg(np.arctan2(hsp, (fp - dsp)))
    thetaps2 = np.deg2rad(THETAps2)
    rs2 = np.sqrt(hsp**2 + (fp - dsp)**2)
    pxS2 = rs2 * np.cos(thetaps2) - d * np.cos(theta14B) - r3
    pyS2 = rs2 * np.sin(thetaps2) - d * np.sin(theta14B)

    pxM4 = Link4 * np.cos(theta1) - r3
    pyM4 = Link4 * np.sin(theta1)

    # --- 2.o mecanismo de 5 barras (sistema secundario) ---
    THETAroll = np.rad2deg(np.arctan2(pyM4 - pyS1, pxM4 - pxS1))
    thetaroll = np.deg2rad(THETAroll)
    THETA1m2so = np.rad2deg(np.arctan2(pyS2 - pyS1, pxS2 - pxS1))
    if THETA1m2so < 0:
        THETA1m2so += 360
    theta1m2 = np.deg2rad(THETA1m2so - THETAroll)
    THETA2m2so = np.rad2deg(np.arctan2(pyP - pyM4, pxP - pxM4))
    if THETA2m2so < 0:
        THETA2m2so += 360
    theta2m2 = np.deg2rad(THETA2m2so - THETAroll)

    den2 = (r4m2 * np.cos(theta2m2) - r1m2 * np.cos(theta1m2) + 2 * r3m2)
    em2 = (r1m2 * np.sin(theta1m2) - r4m2 * np.sin(theta2m2)) / den2
    fm2 = (2 * (r1m2 * r3m2 * np.cos(theta1m2) + r3m2 * r4m2 * np.cos(theta2m2))
           - r1m2**2 + r2m2**2 + r4m2**2 - r5m2**2) / (2 * den2)
    dauxm2 = em2**2 + 1
    gm2 = 2 * (em2 * fm2 - em2 * r1m2 * np.cos(theta1m2)
               + em2 * r3m2 - r1m2 * np.sin(theta1m2))
    hm2 = (fm2**2 - 2 * fm2 * (r1m2 * np.cos(theta1m2) - r3m2)
           - 2 * r1m2 * r3m2 * np.cos(theta1m2)
           + r1m2**2 + r3m2**2 - r2m2**2)
    discm2 = gm2**2 - 4 * dauxm2 * hm2
    if discm2 < 0:
        return None
    pyP2 = (-gm2 + np.sqrt(discm2)) / (2 * dauxm2)
    pxP2 = em2 * pyP2 + fm2
    p2mag = np.sqrt(pxP2**2 + pyP2**2)
    theta2p2 = np.arctan2(pyP2, pxP2)
    pxAUX = (pxS1 + pxM4) / 2
    pyAUX = (pyS1 + pyM4) / 2
    pxP2so = p2mag * np.cos(theta2p2 + thetaroll) + pxAUX
    pyP2so = p2mag * np.sin(theta2p2 + thetaroll) + pyAUX

    # --- 2.o mecanismo de 4 barras ---
    THETA14B2 = np.rad2deg(np.arctan2(pyS2 - pyIFP, pxS2 - pxIFP))
    theta14B2 = np.deg2rad(THETA14B2)
    THETA24B2 = np.rad2deg(np.arctan2(pyP2so - pyS2, pxP2so - pxS2))
    if THETA24B2 < 0:
        THETA24B2 += 360
    theta24B2 = np.deg2rad(THETA24B2)
    k1m2 = d2 * np.cos(theta14B2) + a2 * np.cos(theta24B2)
    k2m2 = d2 * np.sin(theta14B2) + a2 * np.sin(theta24B2)
    k3m2 = k1m2**2 + k2m2**2 + c2**2 - b2**2
    A1m2 = -k3m2 - 2 * k1m2 * c2
    B1m2 = 4 * k2m2 * c2
    C1m2 = 2 * k1m2 * c2 - k3m2
    discm4 = B1m2**2 - 4 * A1m2 * C1m2
    if discm4 < 0:
        return None
    theta4am2 = 2 * np.arctan((-B1m2 - np.sqrt(discm4)) / (2 * A1m2))
    THETA4am2 = np.rad2deg(theta4am2)
    if THETA4am2 < 0:
        THETA4am2 += 360

    pxP3 = pxIFP + c2 * np.cos(np.deg2rad(THETA4am2))
    pyP3 = pyIFP + c2 * np.sin(np.deg2rad(THETA4am2))

    THETAfm = THETA4am2 + THETAauxfm
    thetafm = np.deg2rad(THETAfm)
    pxIFD = fm * np.cos(thetafm) + pxIFP
    pyIFD = fm * np.sin(thetafm) + pyIFP

    return {
        "THETA2": THETA2_deg, "THETA1": THETA1,
        "MCF": np.array([-r3, -d]),
        "G1": np.array([-r3, 0.0]), "G2": np.array([r3, 0.0]),
        "T2": np.array([r3, 0.0]) + Link1 * np.array([np.cos(theta2), np.sin(theta2)]),
        "P": np.array([pxP, pyP]),
        "M4": np.array([pxM4, pyM4]),
        "S1": np.array([pxS1, pyS1]),
        "S2": np.array([pxS2, pyS2]),
        "IFP": np.array([pxIFP, pyIFP]),
        "P2": np.array([pxP2so, pyP2so]),
        "P3": np.array([pxP3, pyP3]),
        "IFD": np.array([pxIFD, pyIFD]),
        "THETAfp": THETAfp, "THETAfm": THETAfm,
        "THETA1m4B": THETA4a,
    }


# =============================================================================
# 3) ELEGIR UNA POSE INTERMEDIA QUE ENSAMBLE
# =============================================================================
est = None
for th2_try in [40, 38, 42, 35, 45, 30, 48, 25, 50, 20, 55, 15, 60]:
    cand = cinematica_paso(float(th2_try))
    if cand is not None:
        est = cand
        THETA2_diag = th2_try
        break
if est is None:
    raise RuntimeError("Ninguna pose intermedia ensambla con estos parametros.")

MCF, IFP, IFD = est["MCF"], est["IFP"], est["IFD"]
P, P2, P3 = est["P"], est["P2"], est["P3"]
S1, S2, M4 = est["S1"], est["S2"], est["M4"]
G1, G2, T2 = est["G1"], est["G2"], est["T2"]

# Comprobacion rapida de longitudes de falange (deben dar ~fp y ~fm)
print(f">> Pose THETA2={THETA2_diag} deg | THETAfm={est['THETAfm']:.1f} deg")
print(f"   |MCF-IFP| = {np.linalg.norm(IFP - MCF):.1f} mm (fp={fp})")
print(f"   |IFP-IFD| = {np.linalg.norm(IFD - IFP):.1f} mm (fm={fm})")


# =============================================================================
# 4) DIBUJO (estilo B/N de ingenieria)
# =============================================================================
LW_LINK, LW_PHAL, LW_GROUND, R_JOINT = 1.6, 3.0, 1.6, 2.2
FS_LINK, FS_PT = 15, 12

# El tamano de la figura lo calculamos despues de saber el rango real de
# los puntos, asi mantenemos aspecto 1:1 sin que el dibujo quede como una
# tira chiquita perdida en una hoja en blanco.
pts_tmp = np.array([G1, G2, T2, P, M4, S1, S2, IFP, P2, P3, IFD, MCF])
_w = pts_tmp[:, 0].max() - pts_tmp[:, 0].min()
_h = pts_tmp[:, 1].max() - pts_tmp[:, 1].min()
_aspect = (_w + 60) / (_h + 60)
_fig_h = 9.0
fig, ax = plt.subplots(figsize=(_fig_h * _aspect, _fig_h))
fig.patch.set_facecolor("white")
ax.set_facecolor("white")


def link(p1, p2, lw=LW_LINK, ls="-", z=4):
    ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color="black",
            linewidth=lw, linestyle=ls, zorder=z, solid_capstyle="round")


def joint(pt, r=R_JOINT, z=10):
    ax.add_patch(plt.Circle((pt[0], pt[1]), r, facecolor="white",
                            edgecolor="black", linewidth=1.6, zorder=z))


def ground(pt, size=7):
    """Soporte fijo (triangulo + achurado), apuntando hacia abajo."""
    base_y = pt[1] - size
    b1 = np.array([pt[0] - size * 0.7, base_y])
    b2 = np.array([pt[0] + size * 0.7, base_y])
    link(pt, b1, lw=1.2, z=6)
    link(pt, b2, lw=1.2, z=6)
    ax.plot([b1[0], b2[0]], [b1[1], b2[1]], "k-", lw=1.2, zorder=6)
    for t in np.linspace(0, 1, 6):
        hx = b1[0] + (b2[0] - b1[0]) * t
        ax.plot([hx, hx - size * 0.35], [base_y, base_y - size * 0.45],
                "k-", lw=0.8, zorder=6)


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
link(IFP, P3, ls=(0, (6, 4)), lw=1.2)       # c2 (linea de construccion)

# --- Falanges (hueso del dedo) ---
link(MCF, IFP, lw=LW_PHAL, z=3)             # Fp
link(IFP, IFD, lw=LW_PHAL, z=3)             # Fm

# --- Juntas de revoluta ---
for pt in [G1, G2, T2, P, M4, S1, S2, IFP, P2, P3, IFD, MCF]:
    joint(pt)

# --- Soportes fijos ---
ground(G1, size=7)
ground(G2, size=7)


# =============================================================================
# 5) ETIQUETAS (al estilo de la figura de referencia)
# =============================================================================
def mid(a, b):
    return (np.asarray(a) + np.asarray(b)) / 2.0


def perp_off(p1, p2, dist):
    v = np.asarray(p2) - np.asarray(p1)
    n = np.array([-v[1], v[0]])
    n = n / (np.hypot(*n) + 1e-12)
    return n * dist


# Eslabones
lbl(G2, T2, r"$L_1$", dy=-7)
lbl(T2, P, r"$L_2$", *perp_off(T2, P, 8))
lbl(M4, P, r"$L_3$", *perp_off(M4, P, 8))
lbl(G1, M4, r"$L_4$", dx=-7)
lbl(M4, S1, r"$L_5$", *perp_off(M4, S1, -8))
lbl(P, P2, r"$L_6$", *perp_off(P, P2, 8))
lbl(S2, P2, r"$L_7$", *perp_off(S2, P2, 8))
lbl(P2, P3, r"$L_8$", *perp_off(P2, P3, 8))
lbl(MCF, S1, r"$c$", dx=-6)
lbl(IFP, P3, r"$c_2$", *perp_off(IFP, P3, 7))

# Bancadas y r3
lbl(G1, G2, r"$B_1$", dy=6)
lbl(G1, MCF, r"$B_2=d$", dx=-12)
ax.annotate("", xy=G2, xytext=(0, 0),
            arrowprops=dict(arrowstyle="<->", color="black", lw=1.0))
ax.text((G1[0] + G2[0]) / 2, G1[1] + 11, r"$r_3$", fontsize=FS_PT,
        style="italic", ha="center", va="bottom", zorder=20)

# hsp / dsp (sobre la falange proximal)
lbl_at(mid(MCF, IFP), r"$F_p$", dy=-9, fs=FS_LINK)
lbl(S2, IFP, r"$h_{sp}$", *perp_off(S2, IFP, -10))
lbl(MCF, S1, r"$d_{sp}$", dx=8, dy=-9, fs=FS_PT)

# theta_aux_fm (entre c2 y la falange medial)
lbl_at(IFP, r"$\theta_{aux_{fm}}$", dx=10, dy=10, fs=FS_PT)

# Falange medial
lbl_at(mid(IFP, IFD), r"$F_m$", *perp_off(IFP, IFD, -10), fs=FS_LINK)

# Angulos iniciales de entrada (cerca de las bancadas)
ax.annotate(r"$\theta_{1_{inicial}}$", xy=G1, xytext=(G1[0] + 18, G1[1] + 24),
            fontsize=FS_PT, style="italic",
            arrowprops=dict(arrowstyle="->", color="black", lw=0.9))
ax.annotate(r"$\theta_{2_{inicial}}$", xy=G2, xytext=(G2[0] + 22, G2[1] + 14),
            fontsize=FS_PT, style="italic",
            arrowprops=dict(arrowstyle="->", color="black", lw=0.9))
ax.annotate(r"$\theta_{1m4B_{inicial}}$", xy=MCF, xytext=(MCF[0] + 22, MCF[1] + 16),
            fontsize=FS_PT, style="italic",
            arrowprops=dict(arrowstyle="->", color="black", lw=0.9))

# Articulaciones del dedo
lbl_at(MCF, "MCF", dx=-4, dy=-10, ha="right", fs=11)
lbl_at(IFP, "IFP", dx=-4, dy=10, ha="right", fs=11)
lbl_at(IFD, "IFD", dx=6, dy=-2, ha="left", fs=11)


# =============================================================================
# 6) AJUSTES Y GUARDADO
# =============================================================================
ax.set_aspect("equal", adjustable="datalim")
ax.axis("off")

pts = np.array([G1, G2, T2, P, M4, S1, S2, IFP, P2, P3, IFD, MCF])
mxr, myr = 22, 20
ax.set_xlim(pts[:, 0].min() - mxr, pts[:, 0].max() + mxr)
ax.set_ylim(pts[:, 1].min() - myr, pts[:, 1].max() + myr)

out = os.path.join(HERE, "resultados", "diagrama_modelo_simplificado.png")
plt.savefig(out, dpi=200, bbox_inches="tight", facecolor="white", pad_inches=0.15)
plt.close()
print(">> Diagrama guardado en:", out)
