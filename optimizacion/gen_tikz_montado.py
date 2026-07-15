"""
gen_tikz_montado.py
===================
Genera el cuerpo TikZ del mecanismo OPTIMIZADO (ED) en el estilo del diagrama de
referencia (diagrama_modelo_simplificado.png):
  - Bancada B1 HORIZONTAL (G1 y G2 alineados en la misma linea)
  - B2 VERTICAL (MCF directamente debajo de G1)
  - B1 y B2 NO se dibujan como eslabones fisicos, sino como cotas/distancias
    con achurado de tierra
  - Pose inicial (idx=0) para maxima legibilidad

Las posiciones de las juntas provienen de la cinematica real (pose_juntas).
Salida: resultados/tikz_montado.tex
"""
import os
import sys
import numpy as np
import modelo_simplificado as M

HERE = os.path.dirname(os.path.abspath(__file__))
p = np.loadtxt(os.path.join(HERE, "resultados", "parametros_simplificado_ED.txt"))

KEYS = ["MCF", "G1", "G2", "T2", "P", "M4", "S1", "S2", "IFP", "P2", "P3", "IFD"]

IDX = int(sys.argv[1]) if len(sys.argv) > 1 else 100
th = M.theta_input
est = M.pose_juntas(p, float(th[IDX]))
if est is None:
    raise SystemExit("pose no ensambla")
G = {k: np.asarray(est[k], float) * 1000.0 for k in KEYS}

# En el marco natural: G1-G2 horizontal, MCF debajo de G1.
# Solo trasladar para que min=0
allp = np.array([G[k] for k in KEYS])
mn = allp.min(axis=0)
for k in KEYS:
    G[k] = G[k] - mn

# --- EMITIR TikZ ---
o = []

# Coordenadas
for k in KEYS:
    o.append(f"  \\coordinate ({k}) at ({G[k][0]:.2f},{G[k][1]:.2f});")

# ESLABONES FISICOS (sin B1 ni B2)
ESL = [("G2", "T2"), ("T2", "P"), ("M4", "P"), ("G1", "M4"), ("M4", "S1"),
       ("MCF", "S1"), ("IFP", "S2"), ("S1", "S2"), ("P", "P2"), ("S2", "P2"),
       ("P2", "P3")]
for a, b in ESL:
    o.append(f"  \\draw[esl] ({a}) -- ({b});")
o.append("  \\draw[c2s] (IFP) -- (P3);")

# FALANGES
o.append("  \\draw[fal] (MCF) -- (IFP);")
o.append("  \\draw[fal] (IFP) -- (IFD);")

# JUNTAS
o.append("  \\foreach \\n in {" + ",".join(KEYS) + "}{")
o.append("    \\filldraw[fill=white,draw=black,line width=0.8pt] (\\n) circle (1.4);")
o.append("  }")

# BANCADA (linea horizontal con achurado) - B1 entre G1 y G2
# y soporte en MCF (base del dedo)
g1 = G["G1"]; g2 = G["G2"]; mcf = G["MCF"]
# Linea base horizontal para G1 y G2
base_y = g1[1] - 2.5  # justo debajo de los pivotes
o.append(f"  \\draw[gnd] ({g1[0]:.2f},{base_y:.2f}) -- ({g2[0]:.2f},{base_y:.2f});")
# Hatching debajo
for t in np.linspace(0, 1, 8):
    hx = g1[0] + t * (g2[0] - g1[0])
    o.append(f"  \\draw[hatch] ({hx:.2f},{base_y:.2f}) -- ({hx - 1.5:.2f},{base_y - 2.0:.2f});")
# Triangulos de apoyo en G1 y G2
for pt in [g1, g2]:
    x, y = pt
    sz = 2.5
    o.append(f"  \\draw[gnd] ({x:.2f},{y:.2f}) -- ({x - 0.6*sz:.2f},{base_y:.2f});")
    o.append(f"  \\draw[gnd] ({x:.2f},{y:.2f}) -- ({x + 0.6*sz:.2f},{base_y:.2f});")

# Soporte en MCF (igual que en el diagrama de referencia: triangulo hacia abajo)
mcf_base_y = mcf[1] - 2.5
o.append(f"  \\draw[gnd] ({mcf[0]:.2f},{mcf[1]:.2f}) -- ({mcf[0] - 1.5:.2f},{mcf_base_y:.2f});")
o.append(f"  \\draw[gnd] ({mcf[0]:.2f},{mcf[1]:.2f}) -- ({mcf[0] + 1.5:.2f},{mcf_base_y:.2f});")
o.append(f"  \\draw[gnd] ({mcf[0] - 1.5:.2f},{mcf_base_y:.2f}) -- ({mcf[0] + 1.5:.2f},{mcf_base_y:.2f});")
for t in np.linspace(0, 1, 4):
    hx = (mcf[0] - 1.5) + t * 3.0
    o.append(f"  \\draw[hatch] ({hx:.2f},{mcf_base_y:.2f}) -- ({hx - 1.2:.2f},{mcf_base_y - 1.6:.2f});")


# --- ETIQUETAS ---
def mid(a, b):
    return (G[a] + G[b]) / 2.0

def perp(a, b, d):
    v = G[b] - G[a]
    n = np.array([-v[1], v[0]]); n = n / (np.hypot(*n) + 1e-12)
    return n * d

LB = 4.5  # offset para etiquetas
labels = [
    (r"$L_1$", mid("G2", "T2") + np.array([3, 4])),
    (r"$L_2$", mid("T2", "P") + np.array([6, 0])),
    (r"$L_3$", mid("M4", "P") + np.array([5, -3])),
    (r"$L_4$", mid("G1", "M4") + np.array([-6, 0])),
    (r"$L_5$", mid("M4", "S1") + np.array([-6, 0])),
    (r"$L_6$", mid("P", "P2") + np.array([-6, 3])),
    (r"$L_7$", mid("S2", "P2") + np.array([-6, 0])),
    (r"$L_8$", mid("P2", "P3") + np.array([-6, 0])),
    (r"$c_2$", mid("IFP", "P3") + np.array([-6, 0])),
    (r"$F_p$", mid("MCF", "IFP") + np.array([5, -5])),
    (r"$F_m$", mid("IFP", "IFD") + np.array([5, -4])),
    (r"$h_{sp}$", mid("S1", "S2") + np.array([6, 0])),
    (r"$d_{sp}$", G["MCF"] + np.array([7, -5])),
]
for txt, pos in labels:
    o.append(f"  \\node[lab] at ({pos[0]:.2f},{pos[1]:.2f}) {{{txt}}};")

# Cotas B1 y B2 (flechas de dimension, no eslabones)
# B1: flecha horizontal debajo de la bancada
b1_y = base_y - 5.0
o.append(f"  \\draw[cota] ({g1[0]:.2f},{b1_y:.2f}) -- ({g2[0]:.2f},{b1_y:.2f});")
o.append(f"  \\node[labp,below] at ({(g1[0]+g2[0])/2:.2f},{b1_y:.2f}) {{$B_1$}};")
# B2: flecha vertical a la izquierda
b2_x = g1[0] - 5.0
o.append(f"  \\draw[cota] ({b2_x:.2f},{g1[1]:.2f}) -- ({b2_x:.2f},{mcf[1]:.2f});")
o.append(f"  \\node[labp,left] at ({b2_x:.2f},{(g1[1]+mcf[1])/2:.2f}) {{$B_2$}};")

body = "\n".join(o)
outf = os.path.join(HERE, "resultados", "tikz_montado.tex")
with open(outf, "w") as f:
    f.write(body + "\n")
print(f">> escrito: {outf}")
print(f">> bbox: w={allp[:,0].max()-allp[:,0].min():.0f} x h={allp[:,1].max()-allp[:,1].min():.0f} mm")
