"""
gen_tikz_optimizado.py
======================
Genera (1) una PREVISUALIZACION matplotlib y (2) el cuerpo TikZ de un diagrama
de eslabones A ESCALA del mecanismo simplificado con las dimensiones OPTIMIZADAS
por la ED. Las juntas provienen de la cinematica real (pose_juntas); las
etiquetas se colocan en posiciones manuales (mm) con lineas guia para evitar
encimados. Solo se reorienta con ROTACIONES rigidas (sin reflexion).
Salidas: resultados/preview_tikz_opt.png  y  resultados/tikz_optimizado.tex
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import modelo_simplificado as M

HERE = os.path.dirname(os.path.abspath(__file__))
p = np.loadtxt(os.path.join(HERE, "resultados", "parametros_simplificado_ED.txt"))


def rot(t):
    c, s = np.cos(t), np.sin(t)
    return np.array([[c, -s], [s, c]])


KEYS = ["MCF", "G1", "G2", "T2", "P", "M4", "S1", "S2", "IFP", "P2", "P3", "IFD"]
th = M.theta_input
obj = 0.5 * (th.min() + th.max())
order = np.argsort(np.abs(th - obj))
est = None
for i in order:
    cand = M.pose_juntas(p, float(th[i]))
    if cand is not None:
        est = {k: np.asarray(v, float) * 1000.0
               for k, v in cand.items() if isinstance(v, np.ndarray)}
        break
v = est["IFD"] - est["MCF"]
R = rot(-np.arctan2(v[1], v[0]))
o = est["MCF"].copy()
for k in KEYS:
    est[k] = R @ (est[k] - o) + o
fy = np.mean([est["MCF"][1], est["IFP"][1], est["IFD"][1]])
cy = np.mean([est[k][1] for k in ["G1", "G2", "T2", "P", "M4", "S1", "S2", "P2", "P3"]])
if cy < fy:
    R2 = rot(np.pi)
    for k in KEYS:
        est[k] = R2 @ (est[k] - o) + o
mn = np.array([est[k] for k in KEYS]).min(axis=0)
for k in KEYS:
    est[k] = est[k] - mn
G = {k: est[k] for k in KEYS}
allp = np.array([G[k] for k in KEYS])


def mid(a, b):
    return (np.asarray(G[a]) + np.asarray(G[b])) / 2.0


# conectividad
GND = [("G1", "G2"), ("G1", "MCF")]
ESL = [("G2", "T2"), ("T2", "P"), ("M4", "P"), ("G1", "M4"), ("M4", "S1"),
       ("MCF", "S1"), ("IFP", "S2"), ("S1", "S2"), ("P", "P2"), ("S2", "P2"),
       ("P2", "P3")]

# --- ETIQUETAS: (texto, x, y, tipo, ancla_para_guia_o_None) ---
# tipo: 'lab' (eslabon), 'labp' (pequena), 'artl' (articulacion)
# El 4o campo es el punto (mm) al que apunta una linea guia, o None.
LAB = [
    (r"$L_1$", 131, 5, "lab", None),
    (r"$L_2$", 128, 21, "lab", None),
    (r"$L_3$", 104, 31, "lab", mid("M4", "P")),
    (r"$L_4$", 100, 51, "lab", mid("G1", "M4")),
    (r"$L_5$", 82, 40, "lab", mid("M4", "S1")),
    (r"$L_6$", 95, 57, "lab", mid("P", "P2")),
    (r"$L_7$", 50, 69, "lab", None),
    (r"$L_8$", 32, 66, "lab", None),
    (r"$c$", 74, 51, "labp", mid("MCF", "S1")),
    (r"$c_2$", 9, 54, "lab", mid("IFP", "P3")),
    (r"$B_1$", 124, 28, "lab", mid("G1", "G2")),
    (r"$B_2$", 88, 46, "lab", mid("G1", "MCF")),
    (r"$h_{sp}$", 24, 54, "labp", mid("S2", "IFP")),
    (r"$d_{sp}$", 60, 46, "labp", None),
    (r"$F_p$", 50, 33, "lab", None),
    (r"$F_m$", 13, 33, "lab", None),
    (r"$\theta_{aux_{fm}}$", 20, 51, "labp", G["IFP"]),
    ("MCF", 76, 32, "artl", None),
    ("IFP", 30, 39, "artl", None),
    ("IFD", -7, 36, "artl", None),
]

# =================== PREVIEW matplotlib ===================
fig, ax = plt.subplots(figsize=(11, 6))
def L(a, b, lw, ls="-", z=4):
    ax.plot([G[a][0], G[b][0]], [G[a][1], G[b][1]], "k-", lw=lw, ls=ls,
            zorder=z, solid_capstyle="round")
for a, b in GND:
    L(a, b, 1.3)
for a, b in ESL:
    L(a, b, 1.4)
ax.plot([G["IFP"][0], G["P3"][0]], [G["IFP"][1], G["P3"][1]], "k--", lw=1.0)
L("MCF", "IFP", 2.6, z=3); L("IFP", "IFD", 2.6, z=3)
for k in KEYS:
    ax.add_patch(plt.Circle(G[k], 1.7, fc="white", ec="black", lw=1.1, zorder=10))
for g in ["G1", "G2"]:
    x, y = G[g]; s = 6.0
    b1 = (x - .7*s, y-s); b2 = (x+.7*s, y-s)
    ax.plot([x, b1[0]], [y, b1[1]], "k-", lw=1); ax.plot([x, b2[0]], [y, b2[1]], "k-", lw=1)
    ax.plot([b1[0], b2[0]], [b1[1], b2[1]], "k-", lw=1)
    for t in np.linspace(0, 1, 6):
        hx = b1[0] + (b2[0]-b1[0])*t
        ax.plot([hx, hx-.35*s], [b1[1], b1[1]-.45*s], "k-", lw=.7)
fs = {"lab": 15, "labp": 11, "artl": 12}
for txt, x, y, typ, anchor in LAB:
    if anchor is not None:
        ax.plot([x, anchor[0]], [y, anchor[1]], "-", color="0.5", lw=0.6, zorder=2)
    ax.text(x, y, txt, fontsize=fs[typ], style="italic", ha="center", va="center", zorder=20)
ax.set_aspect("equal"); ax.axis("off")
ax.set_xlim(allp[:, 0].min()-16, allp[:, 0].max()+16)
ax.set_ylim(allp[:, 1].min()-16, allp[:, 1].max()+16)
plt.savefig(os.path.join(HERE, "resultados", "preview_tikz_opt.png"), dpi=130,
            bbox_inches="tight", facecolor="white")
plt.close()
print(">> preview escrito: resultados/preview_tikz_opt.png")

# =================== EMISION TikZ ===================
out = []
for k in KEYS:
    out.append(f"  \\coordinate ({k}) at ({G[k][0]:.2f},{G[k][1]:.2f});")
for a, b in GND:
    out.append(f"  \\draw[gnd] ({a}) -- ({b});")
for a, b in ESL:
    out.append(f"  \\draw[esl] ({a}) -- ({b});")
out.append("  \\draw[c2s] (IFP) -- (P3);")
out.append("  \\draw[fal] (MCF) -- (IFP);")
out.append("  \\draw[fal] (IFP) -- (IFD);")
out.append("  \\foreach \\n in {" + ",".join(KEYS) + "}{")
out.append("    \\filldraw[fill=white,draw=black,line width=0.8pt] (\\n) circle (1.7);")
out.append("  }")
for g in ["G1", "G2"]:
    x, y = G[g]; s = 6.0
    b1 = (x-.7*s, y-s); b2 = (x+.7*s, y-s)
    out.append(f"  \\draw[gnd] ({x:.2f},{y:.2f}) -- ({b1[0]:.2f},{b1[1]:.2f});")
    out.append(f"  \\draw[gnd] ({x:.2f},{y:.2f}) -- ({b2[0]:.2f},{b2[1]:.2f});")
    out.append(f"  \\draw[gnd] ({b1[0]:.2f},{b1[1]:.2f}) -- ({b2[0]:.2f},{b2[1]:.2f});")
    for t in np.linspace(0, 1, 6):
        hx = b1[0] + (b2[0]-b1[0])*t
        out.append(f"  \\draw[hatch] ({hx:.2f},{b1[1]:.2f}) -- ({hx-.35*s:.2f},{b1[1]-.45*s:.2f});")
for txt, x, y, typ, anchor in LAB:
    if anchor is not None:
        out.append(f"  \\draw[guia] ({x:.2f},{y:.2f}) -- ({anchor[0]:.2f},{anchor[1]:.2f});")
for txt, x, y, typ, anchor in LAB:
    out.append(f"  \\node[{typ}] at ({x:.2f},{y:.2f}) {{{txt}}};")
body = "\n".join(out)
with open(os.path.join(HERE, "resultados", "tikz_optimizado.tex"), "w") as f:
    f.write(body + "\n")
print(">> TikZ escrito: resultados/tikz_optimizado.tex")
