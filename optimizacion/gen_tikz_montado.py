"""
gen_tikz_montado.py
===================
Genera el cuerpo TikZ del mecanismo OPTIMIZADO (ED) dibujado en el estilo limpio
de la Figura 1 y ANCLADO sobre el mismo dedo (transformacion de semejanza, sin
reflexion, que lleva MCF e IFP a las posiciones fijas del dedo de la Fig. 1).
Asi el mecanismo optimizado queda montado sobre el dorso, a escala, con el mismo
aspecto que el esquema sin optimizar.

Uso: python3 gen_tikz_montado.py [idx_pose]
Salidas: resultados/tikz_montado.tex  y  (para preview) _preview_montado.pdf
"""
import os
import sys
import numpy as np
import modelo_simplificado as M

HERE = os.path.dirname(os.path.abspath(__file__))
p = np.loadtxt(os.path.join(HERE, "resultados", "parametros_simplificado_ED.txt"))

# Posiciones FIJAS del dedo en el marco local de la Fig. 1 (unidades de 0.62cm)
P_MCF = np.array([13.7, 3.8])
P_IFP = np.array([5.5, 3.8])

IDX = int(sys.argv[1]) if len(sys.argv) > 1 else 47
th = M.theta_input
est = M.pose_juntas(p, float(th[IDX]))
if est is None:
    raise SystemExit("pose no ensambla")
KEYS = ["MCF", "G1", "G2", "T2", "P", "M4", "S1", "S2", "IFP", "P2", "P3", "IFD"]
G = {k: np.asarray(est[k], float) * 1000.0 for k in KEYS}  # mm

# --- Semejanza (rot + escala + traslacion, SIN reflexion): MCF->P_MCF, IFP->P_IFP
a, b = G["MCF"], G["IFP"]
u = b - a
U = P_IFP - P_MCF
s = np.linalg.norm(U) / np.linalg.norm(u)
ang = np.arctan2(U[1], U[0]) - np.arctan2(u[1], u[0])
c_, s_ = np.cos(ang), np.sin(ang)
Rm = s * np.array([[c_, -s_], [s_, c_]])
for k in KEYS:
    G[k] = P_MCF + Rm @ (G[k] - a)

# Si la cadena quedo por DEBAJO del dedo, no se puede reflejar; avisamos.
finger_y = np.mean([G["MCF"][1], G["IFP"][1], G["IFD"][1]])
chain_y = np.mean([G[k][1] for k in ["G1", "G2", "P", "M4", "S1", "S2", "P2", "P3"]])
print(f"[idx {IDX}] chain_above={chain_y>finger_y}  "
      f"bbox x[{min(G[k][0] for k in KEYS):.1f},{max(G[k][0] for k in KEYS):.1f}] "
      f"y[{min(G[k][1] for k in KEYS):.1f},{max(G[k][1] for k in KEYS):.1f}]")


def perp(p1, p2, d):
    v = np.asarray(p2) - np.asarray(p1)
    n = np.array([-v[1], v[0]]); n = n / (np.hypot(*n) + 1e-12)
    return n * d


def mid(x, y):
    return (G[x] + G[y]) / 2.0


GND = [("G1", "G2"), ("G1", "MCF")]
ESL = [("G2", "T2"), ("T2", "P"), ("M4", "P"), ("G1", "M4"), ("M4", "S1"),
       ("MCF", "S1"), ("IFP", "S2"), ("S1", "S2"), ("P", "P2"), ("S2", "P2"),
       ("P2", "P3")]

o = []
for k in KEYS:
    o.append(f"  \\coordinate ({k}) at ({G[k][0]:.3f},{G[k][1]:.3f});")
for aa, bb in GND:
    o.append(f"  \\draw[gnd] ({aa}) -- ({bb});")
for aa, bb in ESL:
    o.append(f"  \\draw[esl] ({aa}) -- ({bb});")
o.append("  \\draw[c2s] (IFP) -- (P3);")
o.append("  \\draw[fal] (MCF) -- (IFP);")
o.append("  \\draw[fal] (IFP) -- (IFD);")
o.append("  \\foreach \\n in {" + ",".join(KEYS) + "}{")
o.append("    \\filldraw[fill=white,draw=black,line width=0.6pt] (\\n) circle (0.17);")
o.append("  }")
# soportes fijos (triangulo + achurado) apuntando hacia -y local
for g in ["G1", "G2", "MCF"]:
    x, y = G[g]; sz = 0.5
    b1 = (x - 0.7*sz, y - sz); b2 = (x + 0.7*sz, y - sz)
    o.append(f"  \\draw[gnd] ({x:.3f},{y:.3f}) -- ({b1[0]:.3f},{b1[1]:.3f});")
    o.append(f"  \\draw[gnd] ({x:.3f},{y:.3f}) -- ({b2[0]:.3f},{b2[1]:.3f});")
    o.append(f"  \\draw[gnd] ({b1[0]:.3f},{b1[1]:.3f}) -- ({b2[0]:.3f},{b2[1]:.3f});")
    for t in np.linspace(0, 1, 5):
        hx = b1[0] + (b2[0]-b1[0])*t
        o.append(f"  \\draw[hatch] ({hx:.3f},{b1[1]:.3f}) -- ({hx-0.28*sz:.3f},{b1[1]-0.36*sz:.3f});")

# Posiciones de etiqueta ajustadas manualmente para la pose IDX=47 (marco Fig.1)
labels = [
    (r"$L_1$", (23.4, -1.1), "lab"),
    (r"$L_2$", (22.5, 1.3), "lab"),
    (r"$L_3$", (19.2, 2.7), "labp"),
    (r"$L_4$", (17.6, 6.5), "labp"),
    (r"$L_5$", (13.4, 6.2), "lab"),
    (r"$L_6$", (15.6, 7.6), "lab"),
    (r"$L_7$", (8.9, 8.0), "lab"),
    (r"$L_8$", (5.6, 7.4), "lab"),
    (r"$c_2$", (2.7, 5.4), "lab"),
    (r"$B_1$", (21.8, 2.3), "labp"),
    (r"$B_2$", (15.2, 5.2), "labp"),
    (r"$h_{sp}$", (4.9, 5.2), "labp"),
    (r"$F_p$", (9.6, 3.05), "lab"),
    (r"$F_m$", (3.2, 2.4), "lab"),
]
for txt, pos, sty in labels:
    o.append(f"  \\node[{sty}] at ({pos[0]:.3f},{pos[1]:.3f}) {{{txt}}};")

body = "\n".join(o)
with open(os.path.join(HERE, "resultados", "tikz_montado.tex"), "w") as f:
    f.write(body + "\n")

# standalone para preview (con imagen del dedo y mismo scope que la Fig.1)
pre = r"""\documentclass[border=3pt]{standalone}
\usepackage{amsmath}\usepackage{graphicx}\usepackage{tikz}
\usetikzlibrary{arrows.meta,calc}
\graphicspath{{../../articulo_latex/figs/}}
\begin{document}
\begin{tikzpicture}[x=0.62cm,y=0.62cm,
  esl/.style={line width=1.3pt,line cap=round,black},
  fal/.style={line width=2.6pt,line cap=round,black},
  gnd/.style={line width=1.0pt,black},
  hatch/.style={line width=0.6pt,black},
  c2s/.style={line width=0.9pt,dash pattern=on 3pt off 2pt},
  lab/.style={font=\itshape\large,fill=white,fill opacity=0.62,text opacity=1,inner sep=1pt},
  labp/.style={font=\itshape\footnotesize,fill=white,fill opacity=0.7,text opacity=1,inner sep=0.6pt}]
  \node[anchor=center,inner sep=0pt] at (14.1,2.7){\includegraphics[width=18.6cm]{dedo_sin_fondo.png}};
  \begin{scope}[shift={($(13.0,6.7)-(13.7,3.8)$)},rotate around={17:(13.7,3.8)},scale around={0.72:(13.7,3.8)},transform shape]
  \input{tikz_montado.tex}
  \end{scope}
\end{tikzpicture}
\end{document}
"""
with open(os.path.join(HERE, "resultados", "_preview_montado.tex"), "w") as f:
    f.write(pre)
print(">> escrito tikz_montado.tex y _preview_montado.tex")
