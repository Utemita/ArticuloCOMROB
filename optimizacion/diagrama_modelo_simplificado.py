"""
diagrama_modelo_simplificado.py
===============================
Genera un diagrama cinematico del modelo simplificado del exoesqueleto
usando los parametros optimizados por Evolucion Diferencial.
Muestra la configuracion en la pose ~60 de 120 (mitad del recorrido).

Estilo: dibujo tecnico en blanco y negro, lineas delgadas para eslabones,
circulos blancos con borde negro para juntas de revoluta, triangulos con
rayado para pivotes fijos (tierra), lineas punteadas gruesas para falanges.
"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, Polygon
from matplotlib.lines import Line2D

# --- Agregar el directorio actual al path para importar comun ---
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from comun import FP_REAL, FM_REAL, sol_5_barras, solve_four_bar

# ==============================================================================
# CARGAR PARAMETROS OPTIMIZADOS
# ==============================================================================
params_file = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "resultados", "parametros_simplificado_ED.txt")
params = []
with open(params_file, 'r') as f:
    for line in f:
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        params.append(float(line))

params = np.array(params)
(Bancada1, Bancada2, Link1, Link2, Link3, Link4, Link5,
 Link6, Link7, Link8, Link10, hsp, dsp,
 theta_aux_fm, gear_ratio, theta_offset) = params

# ==============================================================================
# CALCULAR GEOMETRIA EN POSE ~60
# ==============================================================================
n_poses = 120
n_grados_input = 85.0
theta_input = np.linspace(0, np.deg2rad(n_grados_input), n_poses)
pose_idx = 60  # mitad del recorrido
th2 = theta_input[pose_idx]
th1 = (th2 / gear_ratio) + theta_offset

r3_val = Bancada1 / 2.0
theta14B = np.pi / 2.0
c1 = np.sqrt(hsp**2 + dsp**2)
rs2 = np.sqrt(hsp**2 + (FP_REAL - dsp)**2)
theta_aux_s2 = np.arctan2(hsp, FP_REAL - dsp)

# --- Pivotes fijos (tierra) ---
# Pivot A: origen del mecanismo de 5 barras (manivela 1)
pivot_A = np.array([-r3_val, 0.0])
# Pivot B: segundo pivote fijo del 5 barras (manivela 2)
pivot_B = np.array([r3_val, 0.0])
# Pivot C: pivote fijo del 4 barras (Bancada2)
pivot_C = np.array([-r3_val, Bancada2])  # desplazado en theta14B = pi/2

# --- Mecanismo 1: 5 barras ---
# Extremo de Link1 (manivela 1 desde pivot_A)
pA_link1 = pivot_A + np.array([Link1 * np.cos(th1), Link1 * np.sin(th1)])
# Extremo de Link4 (manivela 2 desde pivot_B -> recordar que en el modelo r3=Bancada1/2)
# En el modelo: pxm4 = Link4*cos(th2) - r3_val, pym4 = Link4*sin(th2)
pB_link4 = np.array([Link4 * np.cos(th2) - r3_val, Link4 * np.sin(th2)])

# Punto P del 5 barras (interseccion)
res5 = sol_5_barras(Link4, Link3, r3_val, Link1, Link2, th1, th2)
if res5 is None:
    raise RuntimeError("El mecanismo de 5 barras no ensambla en la pose seleccionada")
pxP, pyP = res5
point_P = np.array([pxP, pyP])

# --- Mecanismo 1: 4 barras ---
theta4 = solve_four_bar(Link4, Link5, c1, Bancada2, th1, theta14B)
if theta4 is None:
    raise RuntimeError("El mecanismo de 4 barras (etapa 1) no ensambla")

# Punto donde Link5 conecta con c1 (extremo de Link5 desde pivot_A en theta4)
# En el modelo: el 4 barras usa Link4 como manivela con th1, Link5 como acoplador
# El balancin c1 esta anclado en pivot_C con angulo theta4
point_c1 = pivot_C + np.array([c1 * np.cos(theta4), c1 * np.sin(theta4)])

# Extremo de Link5 (conecta Link4-manivela con c1-balancin)
# Link4 sale del pivot con th1 (pero ojo, en solve_four_bar la entrada es theta2=th1)
# y Link5 es el acoplador. El pivot fijo del balancin es pivot_C.
# Link4 en el 4-barras sale de pivot_A con angulo th1
point_link4_end_4bar = pivot_A + np.array([Link4 * np.cos(th1), Link4 * np.sin(th1)])

# Angulo de la falange proximal
theta_fp = theta4 + np.arctan2(hsp, dsp)

# IFP (articulacion interfalangica proximal) - extremo de la falange proximal
px_ifp = FP_REAL * np.cos(theta_fp) - Bancada2 * np.cos(theta14B) - r3_val
py_ifp = FP_REAL * np.sin(theta_fp) - Bancada2 * np.sin(theta14B)
point_IFP = np.array([px_ifp, py_ifp])

# Punto s1 (conexion c1 en la falange proximal)
pxs1 = c1 * np.cos(theta4) - Bancada2 * np.cos(theta14B) - r3_val
pys1 = c1 * np.sin(theta4) - Bancada2 * np.sin(theta14B)
point_s1 = np.array([pxs1, pys1])

# Punto s2 (segundo punto en la falange proximal para etapa 2)
theta_ps2 = theta_fp - theta_aux_s2
pxs2 = rs2 * np.cos(theta_ps2) - Bancada2 * np.cos(theta14B) - r3_val
pys2 = rs2 * np.sin(theta_ps2) - Bancada2 * np.sin(theta14B)
point_s2 = np.array([pxs2, pys2])

# Punto m4 (extremo de Link4 desde pivot_B)
pxm4 = Link4 * np.cos(th2) - r3_val
pym4 = Link4 * np.sin(th2)
point_m4 = np.array([pxm4, pym4])

# --- Mecanismo 2: 5 barras (secundario, en marco local) ---
theta_roll = np.arctan2(pym4 - pys1, pxm4 - pxs1)
theta1m2 = np.arctan2(pys2 - pys1, pxs2 - pxs1) - theta_roll
theta2m2 = np.arctan2(pyP - pym4, pxP - pxm4) - theta_roll

res5_2 = sol_5_barras(FP_REAL - 2.0 * dsp, Link7, Link5 / 2.0, Link3, Link6,
                      theta1m2, theta2m2)
if res5_2 is None:
    raise RuntimeError("El mecanismo de 5 barras (etapa 2) no ensambla")
px_local, py_local = res5_2

mag = np.sqrt(px_local**2 + py_local**2)
theta_loc = np.arctan2(py_local, px_local)
px_aux = (pxs1 + pxm4) / 2.0
py_aux = (pys1 + pym4) / 2.0
pxP2 = mag * np.cos(theta_loc + theta_roll) + px_aux
pyP2 = mag * np.sin(theta_loc + theta_roll) + py_aux
point_P2 = np.array([pxP2, pyP2])

# --- Mecanismo 2: 4 barras ---
theta1m42 = np.arctan2(pys2 - py_ifp, pxs2 - px_ifp)
theta2m42 = np.arctan2(pyP2 - pys2, pxP2 - pxs2)

theta4m2 = solve_four_bar(Link7, Link8, Link10, c1, theta2m42, theta1m42)
if theta4m2 is None:
    raise RuntimeError("El mecanismo de 4 barras (etapa 2) no ensambla")

# Angulo de la falange medial
theta_fm = theta4m2 + theta_aux_fm

# IFD (articulacion interfalangica distal) - extremo de la falange medial
px_ifd = FM_REAL * np.cos(theta_fm) + px_ifp
py_ifd = FM_REAL * np.sin(theta_fm) + py_ifp
point_IFD = np.array([px_ifd, py_ifd])

# Origen de la falange proximal (MCF) = pivot_C
point_MCF = pivot_C.copy()

# Extremo de Link8 desde point_s2
point_link8_end = point_s2 + np.array([Link8 * np.cos(theta2m42 + theta_roll),
                                        Link8 * np.sin(theta2m42 + theta_roll)])

# Extremo de Link10/c2 desde point_IFP
point_link10_end = point_IFP + np.array([Link10 * np.cos(theta4m2 + theta_roll),
                                          Link10 * np.sin(theta4m2 + theta_roll)])


# ==============================================================================
# FUNCIONES DE DIBUJO
# ==============================================================================
def draw_ground_symbol(ax, center, size=0.003, angle=0):
    """Dibuja simbolo de tierra (triangulo con rayado) en la posicion dada."""
    cx, cy = center
    # Triangulo apuntando hacia abajo
    h = size * 1.2
    w = size * 1.4
    cos_a, sin_a = np.cos(angle), np.sin(angle)

    # Vertices del triangulo (apuntando abajo por defecto)
    verts_local = np.array([
        [-w/2, -h*0.3],
        [w/2, -h*0.3],
        [0, -h*1.1],
    ])
    # Rotar
    rot = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
    verts = (rot @ verts_local.T).T + np.array([cx, cy])

    tri = Polygon(verts, closed=True, fill=False, edgecolor='black',
                  linewidth=1.0, zorder=5)
    ax.add_patch(tri)

    # Rayado debajo del triangulo
    n_lines = 4
    base_y = -h * 1.1
    for i in range(n_lines):
        xi = -w/2 + (i + 0.5) * w / n_lines
        local_start = np.array([xi, base_y])
        local_end = np.array([xi - size*0.3, base_y - size*0.4])
        start = rot @ local_start + np.array([cx, cy])
        end = rot @ local_end + np.array([cx, cy])
        ax.plot([start[0], end[0]], [start[1], end[1]], 'k-', linewidth=0.6, zorder=5)


def draw_revolute_joint(ax, center, radius=0.0015):
    """Dibuja junta de revoluta (circulo blanco con borde negro)."""
    circle = plt.Circle(center, radius, fill=True, facecolor='white',
                        edgecolor='black', linewidth=1.2, zorder=10)
    ax.add_patch(circle)


def draw_link(ax, p1, p2, linewidth=1.0, color='black', linestyle='-'):
    """Dibuja un eslabon (linea entre dos puntos)."""
    ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color=color,
            linewidth=linewidth, linestyle=linestyle, zorder=3)


def draw_phalanx(ax, p1, p2, linewidth=3.0):
    """Dibuja una falange (linea punteada gruesa)."""
    ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color='black',
            linewidth=linewidth, linestyle='--', dashes=(4, 2), zorder=4)


def add_label(ax, text, pos, offset=(0, 0), fontsize=7, ha='center', va='bottom'):
    """Agrega etiqueta con desplazamiento."""
    ax.annotate(text, xy=pos, xytext=(pos[0] + offset[0], pos[1] + offset[1]),
                fontsize=fontsize, ha=ha, va=va, color='black',
                arrowprops=dict(arrowstyle='->', color='black', lw=0.5,
                                connectionstyle='arc3,rad=0.1'),
                zorder=20)


# ==============================================================================
# GENERAR DIAGRAMA
# ==============================================================================
fig, ax = plt.subplots(1, 1, figsize=(10, 8), dpi=300)
ax.set_aspect('equal')
ax.axis('off')
ax.set_facecolor('white')
fig.patch.set_facecolor('white')

# Escala para offsets de etiquetas
sc = 0.006

# --- ETAPA 1: Mecanismo de 5 barras ---
# Link1: pivot_A -> pA_link1
draw_link(ax, pivot_A, pA_link1, linewidth=1.0)
# Link2: pA_link1 -> point_P
draw_link(ax, pA_link1, point_P, linewidth=1.0)
# Link3: point_P -> pB_link4
draw_link(ax, point_P, pB_link4, linewidth=1.0)
# Link4: pivot_B -> pB_link4 (manivela 2)
draw_link(ax, pivot_B, pB_link4, linewidth=1.0)
# Link5: pivot_A -> point_c1 (a traves del 4 barras)
draw_link(ax, point_link4_end_4bar, point_c1, linewidth=1.0)

# --- ETAPA 1: Mecanismo de 4 barras ---
# Link4 (manivela del 4-barras): pivot_A -> point_link4_end_4bar
draw_link(ax, pivot_A, point_link4_end_4bar, linewidth=1.0, color='black')
# c1 (balancin): pivot_C -> point_c1
draw_link(ax, pivot_C, point_c1, linewidth=1.0)
# Bancada2: pivot_A -> pivot_C
draw_link(ax, pivot_A, pivot_C, linewidth=0.8, linestyle='-.')

# --- Falange proximal (dashed gruesa) ---
draw_phalanx(ax, point_MCF, point_IFP)

# --- ETAPA 2: Eslabones ---
# Link6: point_m4 -> point_P2 (via 5 barras secundario)
draw_link(ax, point_m4, point_P2, linewidth=1.0)
# Link7: point_s2 -> point_P2
draw_link(ax, point_s2, point_P2, linewidth=1.0)
# Link8: point_s2 -> point_link10_end (via 4 barras secundario)
draw_link(ax, point_s2, point_link8_end, linewidth=1.0)
# Link10/c2: point_IFP -> point_link10_end
draw_link(ax, point_IFP, point_link10_end, linewidth=1.0)

# Conectores de la etapa 2
draw_link(ax, point_s1, point_s2, linewidth=0.8, linestyle=':')
draw_link(ax, point_link8_end, point_link10_end, linewidth=1.0)

# --- Falange medial (dashed gruesa) ---
draw_phalanx(ax, point_IFP, point_IFD)

# --- Pivotes fijos (tierra) ---
draw_ground_symbol(ax, pivot_A, size=0.003, angle=0)
draw_ground_symbol(ax, pivot_B, size=0.003, angle=0)
draw_ground_symbol(ax, pivot_C, size=0.003, angle=0)

# --- Juntas de revoluta (circulos blancos) ---
joints = [pivot_A, pivot_B, pivot_C, pA_link1, pB_link4, point_P,
           point_link4_end_4bar, point_c1, point_s1, point_s2, point_m4,
           point_P2, point_IFP, point_IFD, point_link8_end, point_link10_end]
for j in joints:
    draw_revolute_joint(ax, j, radius=0.0012)

# ==============================================================================
# ETIQUETAS
# ==============================================================================
# Eslabones - posicion en el punto medio
def mid(p1, p2):
    return (p1 + p2) / 2.0

# Link1
add_label(ax, 'Link1', mid(pivot_A, pA_link1), offset=(-sc*1.5, -sc*0.5), fontsize=6)
# Link2
add_label(ax, 'Link2', mid(pA_link1, point_P), offset=(-sc*1.5, sc*0.3), fontsize=6)
# Link3
add_label(ax, 'Link3', mid(point_P, pB_link4), offset=(sc*1.0, sc*0.5), fontsize=6)
# Link4
add_label(ax, 'Link4', mid(pivot_B, pB_link4), offset=(sc*1.5, -sc*0.3), fontsize=6)
# Link5
add_label(ax, 'Link5', mid(point_link4_end_4bar, point_c1), offset=(sc*1.5, sc*0.3), fontsize=6)
# Link6
add_label(ax, 'Link6', mid(point_m4, point_P2), offset=(sc*1.2, sc*0.8), fontsize=6)
# Link7
add_label(ax, 'Link7', mid(point_s2, point_P2), offset=(-sc*1.5, sc*0.5), fontsize=6)
# Link8
add_label(ax, 'Link8', mid(point_s2, point_link8_end), offset=(sc*1.5, -sc*0.5), fontsize=6)
# Link10/c2
add_label(ax, 'Link10/c2', mid(point_IFP, point_link10_end), offset=(-sc*1.8, -sc*0.8), fontsize=6)

# Bancada1 (distancia entre pivot_A y pivot_B)
add_label(ax, 'Bancada1', mid(pivot_A, pivot_B), offset=(0, -sc*1.8), fontsize=6)
# Bancada2 (distancia pivot_A a pivot_C)
add_label(ax, 'Bancada2', mid(pivot_A, pivot_C), offset=(-sc*2.0, 0), fontsize=6)

# Parametros especiales
# hsp y dsp (en la falange proximal)
add_label(ax, 'hsp', mid(point_MCF, point_s1), offset=(-sc*1.5, sc*1.0), fontsize=6)
add_label(ax, 'dsp', mid(point_MCF, point_s1), offset=(sc*0.5, -sc*1.5), fontsize=6)

# theta_aux_fm
add_label(ax, r'$\theta_{aux,fm}$', point_IFP, offset=(sc*2.0, -sc*1.5), fontsize=6)

# gear_ratio y theta_offset (cerca de las manivelas)
add_label(ax, f'gear_ratio={gear_ratio:.2f}', pivot_A, offset=(-sc*3.5, -sc*2.5), fontsize=5.5)
add_label(ax, r'$\theta_{offset}$' + f'={np.rad2deg(theta_offset):.1f} deg',
          pivot_A, offset=(-sc*3.5, -sc*3.5), fontsize=5.5)

# Articulaciones principales
add_label(ax, 'MCF', point_MCF, offset=(-sc*2.0, sc*1.0), fontsize=7)
add_label(ax, 'IFP', point_IFP, offset=(sc*0.5, sc*1.5), fontsize=7)
add_label(ax, 'IFD', point_IFD, offset=(sc*1.5, sc*0.5), fontsize=7)

# Falanges
fp_mid = mid(point_MCF, point_IFP)
add_label(ax, 'Falange\nproximal', fp_mid, offset=(sc*2.5, sc*1.5), fontsize=6)
fm_mid = mid(point_IFP, point_IFD)
add_label(ax, 'Falange\nmedial', fm_mid, offset=(sc*2.0, sc*1.5), fontsize=6)

# c1 label
add_label(ax, 'c1', mid(pivot_C, point_c1), offset=(sc*1.2, sc*0.5), fontsize=6)

# --- Indicador de engranaje (arco con flecha entre manivelas) ---
arc_r = 0.008
arc_angles = np.linspace(np.rad2deg(th1), np.rad2deg(th2), 20)
arc_x = pivot_A[0] + arc_r * np.cos(np.deg2rad(arc_angles))
arc_y = pivot_A[1] + arc_r * np.sin(np.deg2rad(arc_angles))
ax.plot(arc_x, arc_y, 'k-', linewidth=0.5, zorder=2)

# --- Titulo ---
ax.set_title('Modelo Simplificado del Exoesqueleto - Diagrama Cinematico\n'
             '(Pose intermedia, parametros optimizados por ED)',
             fontsize=9, fontweight='bold', pad=10)

# --- Leyenda ---
legend_elements = [
    Line2D([0], [0], color='black', linewidth=1.0, linestyle='-', label='Eslabones'),
    Line2D([0], [0], color='black', linewidth=3.0, linestyle='--', label='Falanges del dedo'),
    Line2D([0], [0], color='black', linewidth=0.8, linestyle='-.', label='Bancada (fija)'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='white',
           markeredgecolor='black', markersize=8, label='Junta de revoluta'),
    Line2D([0], [0], marker='^', color='w', markerfacecolor='white',
           markeredgecolor='black', markersize=8, label='Pivote fijo (tierra)'),
]
ax.legend(handles=legend_elements, loc='lower right', fontsize=6,
          framealpha=0.9, edgecolor='black')

# --- Ajustar limites ---
all_pts = np.array(joints)
margin = 0.025
ax.set_xlim(all_pts[:, 0].min() - margin, all_pts[:, 0].max() + margin)
ax.set_ylim(all_pts[:, 1].min() - margin, all_pts[:, 1].max() + margin)

# --- Guardar ---
output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "resultados", "diagrama_modelo_simplificado.png")
plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white',
            edgecolor='none')
plt.close()
print(f"Diagrama guardado en: {output_path}")
