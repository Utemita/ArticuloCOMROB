import numpy as np
import matplotlib
matplotlib.use('Agg')   # backend headless (sin pantalla)
import matplotlib.pyplot as plt
import pandas as pd
import optuna
from scipy.optimize import differential_evolution
from scipy.spatial.distance import cdist
from scipy.signal import savgol_filter
import time
import os

optuna.logging.set_verbosity(optuna.logging.WARNING)

# Presupuesto de optimizacion (configurable por variables de entorno para poder
# hacer corridas rapidas de prueba sin tocar el codigo; los valores por defecto
# son los exhaustivos).
N_TRIALS_OPTUNA   = int(os.environ.get('OPT_TRIALS', '8'))
DE_MAXITER        = int(os.environ.get('OPT_MAXITER', '1500'))
DE_MAXITER_OPTUNA = int(os.environ.get('OPT_MAXITER_OPTUNA', '20'))

# Tope fisiologico de la flexion distal relativa (DIP) en grados. Las
# soluciones con un ROM de DIP mayor se descartan por irreales.
DIP_MAX_DEG = float(os.environ.get('DIP_MAX_DEG', '35.0'))

# ==============================================================================
# --- 1. PARAMETROS ANTROPOMETRICOS (longitudes de falanges reales) ---
# ==============================================================================
FP_REAL = 0.049   # Falange proximal (m)
FM_REAL = 0.026   # Falange medial (m)
FD_REAL = 0.024   # Falange distal (m)

# ==============================================================================
# --- 2. CARGA Y CORRECCION DE DATOS MOCAP ---
# ==============================================================================
# CORRECCIONES APLICADAS:
#   (a) Los datos del CSV vienen de flexion maxima -> extension; se invierten
#       para que representen el movimiento natural de apertura a cierre.
#   (b) theta_dip tiene valores negativos (~75/120 muestras) debidos a ruido
#       del sensor o ligera hiperextension fisiologica. Un angulo DIP negativo
#       no puede replicarse con el exoesqueleto (limite fisico del mecanismo),
#       por lo que se recorta a 0 grados como minimo.
#   (c) Se aplica suavizado Savitzky-Golay para eliminar oscilaciones de alta
#       frecuencia que generan el "gancho" en las trayectorias cartesianas de
#       IFD y Punta.
# ==============================================================================
try:
    print(">> Cargando base de datos MOCAP 'mocap_pinza_fina_120pts.csv'...")
    datos_mocap = pd.read_csv("mocap_pinza_fina_120pts.csv")

    # (a) Invertir: de apertura -> cierre (movimiento de agarre)
    mcp_raw = datos_mocap['Theta_MCP'].values[::-1]
    pip_raw = datos_mocap['Theta_PIP'].values[::-1]
    dip_raw = datos_mocap['Theta_DIP'].values[::-1]

    N_PUNTOS = len(mcp_raw)

    # (b) Recortar DIP a 0 grados minimo (no puede existir hiperextension en el exo)
    dip_raw = np.clip(dip_raw, 0.0, None)

    # (c) Suavizado Savitzky-Golay para eliminar oscilaciones del sensor
    WIN = 15  # ventana impar; ajustar si N_PUNTOS < 15
    if N_PUNTOS >= WIN:
        mcp_smooth = savgol_filter(np.deg2rad(mcp_raw), window_length=WIN, polyorder=2)
        pip_smooth = savgol_filter(np.deg2rad(pip_raw), window_length=WIN, polyorder=2)
        dip_smooth = savgol_filter(np.deg2rad(dip_raw), window_length=WIN, polyorder=2)
        # El suavizado puede introducir valores levemente negativos en DIP -> re-recortar
        dip_smooth = np.clip(dip_smooth, 0.0, None)
    else:
        mcp_smooth = np.deg2rad(mcp_raw)
        pip_smooth = np.deg2rad(pip_raw)
        dip_smooth = np.deg2rad(dip_raw)

except FileNotFoundError:
    print("\n>> ERROR CRITICO: No se encontro 'mocap_pinza_fina_120pts.csv'.")
    raise

# Cinematica directa del dedo (cadena de cuerpos rigidos desde la MCF)
# Los angulos son ACUMULADOS en el sistema global (igual que el codigo original,
# pero ahora los datos de entrada estan limpios).
seg_prox = mcp_smooth                          # orientacion falange proximal
seg_med  = mcp_smooth + pip_smooth             # orientacion falange medial
seg_dist = mcp_smooth + pip_smooth + dip_smooth  # orientacion falange distal

pxIFP_mocap = FP_REAL * np.cos(seg_prox)
pyIFP_mocap = FP_REAL * np.sin(seg_prox)
pxIFD_mocap = pxIFP_mocap + FM_REAL * np.cos(seg_med)
pyIFD_mocap = pyIFP_mocap + FM_REAL * np.sin(seg_med)
pxPF_mocap  = pxIFD_mocap + FD_REAL * np.cos(seg_dist)
pyPF_mocap  = pyIFD_mocap + FD_REAL * np.sin(seg_dist)

mocap_pts = {
    'ifp': np.column_stack((pxIFP_mocap, pyIFP_mocap)),
    'ifd': np.column_stack((pxIFD_mocap, pyIFD_mocap)),
    'tip': np.column_stack((pxPF_mocap,  pyPF_mocap))
}

# Perfil REAL del angulo articular DIP (variacion respecto a la pose inicial).
# dip_smooth es el angulo articular distal-medial del dedo real; su variacion
# se usa para que el tercer mecanismo reproduzca la flexion distal fisiologica
# (y NO una flexion exagerada que solo encaje la punta geometricamente).
dip_rel_mocap = dip_smooth - dip_smooth[0]

# Rango de entrada de la manivela principal (0 -> 85 grados), igual que antes
theta_input = np.linspace(0, np.deg2rad(85), N_PUNTOS)

print(f">> MOCAP cargado: {N_PUNTOS} puntos.")
print(f"   MCP: {np.rad2deg(mcp_smooth[0]):.1f} deg -> {np.rad2deg(mcp_smooth[-1]):.1f} deg")
print(f"   PIP: {np.rad2deg(pip_smooth[0]):.1f} deg -> {np.rad2deg(pip_smooth[-1]):.1f} deg")
print(f"   DIP: {np.rad2deg(dip_smooth[0]):.1f} deg -> {np.rad2deg(dip_smooth[-1]):.1f} deg")

# ==============================================================================
# --- 3. FUNCIONES DE EVALUACION ---
# ==============================================================================
def chamfer_distance(curve_target, curve_sim):
    """Distancia de Chamfer bidireccional (metrica de forma)."""
    dists = cdist(curve_target, curve_sim)
    return np.mean(np.min(dists, axis=1)) + np.mean(np.min(dists, axis=0))

def optimal_rigid_transform(target, sim):
    """Transformacion rigida optima (rotacion + traslacion) entre dos nubes de puntos."""
    c_t = np.mean(target, axis=0)
    c_s = np.mean(sim,    axis=0)
    H   = (sim - c_s).T @ (target - c_t)
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[1, :] *= -1
        R = Vt.T @ U.T
    t = c_t - R @ c_s
    return R, t

def apply_transform(points, R, t):
    return (R @ points.T).T + t

def monotonicity_penalty(curve):
    """
    Penaliza inversiones de direccion en la trayectoria simulada.
    Calcula la variacion total y la compara con la longitud de arco neta;
    si hay reversiones, la diferencia es positiva.
    
    Devuelve un valor >= 0 (0 = curva perfectamente monotona en avance).
    """
    diffs = np.diff(curve, axis=0)
    arc_increments = np.linalg.norm(diffs, axis=1)
    total_arc = np.sum(arc_increments)
    
    # Proyeccion acumulada en la direccion dominante (componente principal)
    if total_arc < 1e-9:
        return 0.0
    mean_dir = np.sum(diffs, axis=0) / (total_arc + 1e-12)
    mean_dir /= (np.linalg.norm(mean_dir) + 1e-12)
    projections = diffs @ mean_dir
    # Suma de retrocesos (proyecciones negativas)
    reversals = np.sum(np.clip(-projections, 0, None))
    return reversals

# ==============================================================================
# --- 4. SOLUCIONES MATEMATICAS ---
# ==============================================================================
def sol_5_barras(r1, r2, r3, r4, r5, theta1, theta2):
    den = r4*np.cos(theta2) - r1*np.cos(theta1) + 2*r3
    if np.abs(den) < 1e-4:
        return None
    e = (r1*np.sin(theta1) - r4*np.sin(theta2)) / den
    f = (2*(r1*r3*np.cos(theta1) + r3*r4*np.cos(theta2))
         - r1**2 + r2**2 + r4**2 - r5**2) / (2*den)
    d_ = e**2 + 1
    g  = 2*(e*f - e*r1*np.cos(theta1) + e*r3 - r1*np.sin(theta1))
    h  = (f**2 - 2*f*(r1*np.cos(theta1) - r3)
          - 2*r1*r3*np.cos(theta1) + r1**2 + r3**2 - r2**2)
    disc = g**2 - 4*d_*h
    if disc < 0:
        return None
    py = (-g + np.sqrt(disc)) / (2*d_)
    px = e*py + f
    return px, py

def solve_four_bar(a, b, c, d, theta2, theta1):
    k1 = a*np.cos(theta2) + d*np.cos(theta1)
    k2 = a*np.sin(theta2) + d*np.sin(theta1)
    k3 = k1**2 + k2**2 + c**2 - b**2
    A1 = -2*k1*c - k3
    B1 =  4*k2*c
    C1 =  2*k1*c - k3
    disc = B1**2 - 4*A1*C1
    if disc < 0:
        return None
    # Configuracion abierta (signo -)
    return 2*np.arctan((-B1 - np.sqrt(disc)) / (2*A1))

def _circle_intersections(c0, r0, c1, r1):
    """Interseccion de dos circulos (centros c0,c1 y radios r0,r1).

    Se usa para resolver el tercer mecanismo de 4 barras: el punto D3 esta a
    distancia Link9_3 del anclaje Pa y a distancia Link10_3 de la IFD, por lo
    que es una de las intersecciones de ambos circulos.
    Devuelve (sol_a, sol_b) con sol_b = rama '-perp' (lado dorsal), o None si
    los circulos no se cortan (el mecanismo no ensambla en esa pose).
    """
    c0 = np.asarray(c0, dtype=float)
    c1 = np.asarray(c1, dtype=float)
    dvec = c1 - c0
    dist = np.hypot(dvec[0], dvec[1])
    if dist > (r0 + r1) or dist < abs(r0 - r1) or dist == 0:
        return None
    aa = (r0**2 - r1**2 + dist**2) / (2 * dist)
    hh2 = r0**2 - aa**2
    if hh2 < 0:
        return None
    hh = np.sqrt(hh2)
    pm = c0 + aa * dvec / dist
    perp = np.array([-dvec[1], dvec[0]]) / dist
    return pm + hh * perp, pm - hh * perp

# ==============================================================================
# --- 5. MODELO CINEMATICO ---
# ==============================================================================
def run_kinematics(p, th_input):
    # 17 parametros originales (dos 5 barras + dos 4 barras) + 4 parametros
    # NUEVOS del tercer mecanismo de 4 barras que da movimiento real a la IFD:
    #   Link9_3  : acoplador (D3 -> Pa)
    #   Link10_3 : balancin  (IFD -> D3, rigido con la falange distal)
    #   back3_3  : desplazamiento del soporte S3 a lo largo de la falange
    #              proximal (NEGATIVO = el soporte se adelanta pasando la IFP)
    #   up3_3    : standoff DORSAL del soporte S3 (perfil bajo)
    (Bancada1, Bancada2, Link1, Link2, Link3, Link4, Link5,
     Link6, Link7, Link8, Link10, hsp, dsp,
     theta_aux_fm, theta_aux_fd, gear_ratio, theta_offset,
     Link9_3, Link10_3, back3_3, up3_3) = p

    # Validaciones basicas
    if gear_ratio <= 0:
        return None
    if min(p[:11]) <= 0.005:   # longitudes de eslabon minimas (11 primeros)
        return None
    if Link9_3 <= 0.005 or Link10_3 <= 0.005:  # eslabones del tercer mecanismo
        return None
    if up3_3 <= 0:             # el standoff dorsal debe ser positivo
        return None
    if hsp <= 0 or dsp <= 0:
        return None
    if FP_REAL - 2.0 * dsp <= 0.001:
        return None

    # Pre-calculos fijos (independientes del angulo de entrada)
    r3_val   = Bancada1 / 2.0
    theta14B = np.pi / 2.0
    c1       = np.sqrt(hsp**2 + dsp**2)

    # CORRECCION CLAVE: rs2 y theta_aux_s2 deben calcularse con FP_REAL
    # (longitud real de la falange proximal que el exoesqueleto debe seguir),
    # NO con el parametro Bancada2 ni con ningun Link optimizable.
    rs2          = np.sqrt(hsp**2 + (FP_REAL - dsp)**2)
    theta_aux_s2 = np.arctan2(hsp, FP_REAL - dsp)

    PXifp, PYifp = [], []
    PXifd, PYifd = [], []
    PXtip, PYtip = [], []
    TH_FM, TH_FD = [], []   # angulos de falange medial y distal (diagnostico/ROM)

    prev_theta_fm = None    # para detectar reversiones en theta_fm
    gamma_bracket3 = None   # offset del soporte S3 (se calibra en el 1er paso)
    dorsal_sign3   = None   # lado dorsal del tercer mecanismo (se fija 1 vez)

    for th2 in th_input:
        th1 = (th2 / gear_ratio) + theta_offset

        # --- Mecanismo 1: 5 barras ---
        res5 = sol_5_barras(Link4, Link3, r3_val, Link1, Link2, th1, th2)
        if res5 is None:
            return None
        pxP, pyP = res5

        # --- Mecanismo 1: 4 barras ---
        theta4 = solve_four_bar(Link4, Link5, c1, Bancada2, th1, theta14B)
        if theta4 is None:
            return None

        # Angulo y posicion de la articulacion IFP
        theta_fp = theta4 + np.arctan2(hsp, dsp)
        px_ifp   = FP_REAL * np.cos(theta_fp) - Bancada2*np.cos(theta14B) - r3_val
        py_ifp   = FP_REAL * np.sin(theta_fp) - Bancada2*np.sin(theta14B)

        # Soportes de la falange proximal
        pxs1 = c1 * np.cos(theta4) - Bancada2*np.cos(theta14B) - r3_val
        pys1 = c1 * np.sin(theta4) - Bancada2*np.sin(theta14B)

        theta_ps2 = theta_fp - theta_aux_s2
        pxs2 = rs2 * np.cos(theta_ps2) - Bancada2*np.cos(theta14B) - r3_val
        pys2 = rs2 * np.sin(theta_ps2) - Bancada2*np.sin(theta14B)

        pxm4 = Link4*np.cos(th2) - r3_val
        pym4 = Link4*np.sin(th2)

        # --- Mecanismo 2: 5 barras (sistema de referencia secundario) ---
        theta_roll = np.arctan2(pym4 - pys1, pxm4 - pxs1)
        theta1m2   = np.arctan2(pys2 - pys1, pxs2 - pxs1) - theta_roll
        theta2m2   = np.arctan2(pyP  - pym4, pxP  - pxm4) - theta_roll

        res5_2 = sol_5_barras(FP_REAL - 2.0 * dsp, Link7, Link5/2.0, Link3, Link6,
                              theta1m2, theta2m2)
        if res5_2 is None:
            return None
        px_local, py_local = res5_2

        mag        = np.sqrt(px_local**2 + py_local**2)
        theta_loc  = np.arctan2(py_local, px_local)
        px_aux     = (pxs1 + pxm4) / 2.0
        py_aux     = (pys1 + pym4) / 2.0
        pxP2 = mag * np.cos(theta_loc + theta_roll) + px_aux
        pyP2 = mag * np.sin(theta_loc + theta_roll) + py_aux

        # --- Mecanismo 2: 4 barras ---
        theta1m42 = np.arctan2(pys2 - py_ifp, pxs2 - px_ifp)
        theta2m42 = np.arctan2(pyP2 - pys2,   pxP2 - pxs2)

        theta4m2 = solve_four_bar(Link7, Link8, Link10, c1, theta2m42, theta1m42)
        if theta4m2 is None:
            return None

        # Angulo de la falange medial (igual que antes)
        theta_fm = theta4m2 + theta_aux_fm

        # -----------------------------------------------------------------
        # RESTRICCION FISICA ANTI-GANCHO:
        # En un dedo real realizando un agarre, el angulo de la falange medial
        # debe ser MONOTONO (nunca retrocede) a lo largo del cierre. Si theta_fm
        # retrocede respecto al paso anterior, la solucion no es fisicamente
        # realizable y se descarta.
        # -----------------------------------------------------------------
        if prev_theta_fm is not None:
            delta_fm = (theta_fm - prev_theta_fm + np.pi) % (2*np.pi) - np.pi
            # Toleramos pequenas oscilaciones numericas (< 0.5 grados)
            if delta_fm < -np.deg2rad(0.5):
                return None   # retroceso -> solucion no valida
        prev_theta_fm = theta_fm

        # Posicion de la articulacion IFD (extremo de la falange medial)
        px_ifd = FM_REAL * np.cos(theta_fm) + px_ifp
        py_ifd = FM_REAL * np.sin(theta_fm) + py_ifp

        # -----------------------------------------------------------------
        # TERCER MECANISMO DE 4 BARRAS (cinematica REAL de la IFD/DIP)
        # -----------------------------------------------------------------
        # Antes la distal usaba un offset CONSTANTE (theta_fd = theta_fm +
        # theta_aux_fd), por lo que se movia rigida con la medial. Ahora el
        # angulo distal lo entrega un 4 barras real:
        #   - Tierra:    Pa -> IFP   (soporte S3 rigido a la falange proximal)
        #   - Manivela:  IFP -> IFD  (la propia falange medial, FM_REAL)
        #   - Balancin:  IFD -> D3   (Link10_3, rigido con la falange distal)
        #   - Acoplador: D3  -> Pa   (Link9_3), por el lado DORSAL
        # El offset theta_aux_fd ahora solo fija la flexion natural en reposo
        # (via gamma_bracket3, calibrado en el 1er paso); la EVOLUCION del
        # angulo distal la determina la geometria del 4 barras -> flexion DIP
        # gradual, como un dedo real.
        prox_dir  = np.array([np.cos(theta_fp), np.sin(theta_fp)])
        prox_norm = np.array([-prox_dir[1], prox_dir[0]])
        IFP_pt = np.array([px_ifp, py_ifp])
        IFD_pt = np.array([px_ifd, py_ifd])

        # Lado dorsal: se fija UNA sola vez usando el lado de P3 (el resto del
        # exoesqueleto va por el dorso). P3 = extremo del balancin del 2do 4B.
        if dorsal_sign3 is None:
            P3_pt = IFP_pt + Link10 * np.array([np.cos(theta4m2), np.sin(theta4m2)])
            dorsal_sign3 = 1.0 if np.dot(P3_pt - IFP_pt, prox_norm) >= 0 else -1.0
        dorsal_norm = dorsal_sign3 * prox_norm

        # Anclaje del acoplador, fijo a la falange proximal (soporte S3).
        Pa = IFP_pt - back3_3 * prox_dir + up3_3 * dorsal_norm

        sols = _circle_intersections(Pa, Link9_3, IFD_pt, Link10_3)
        if sols is None:
            return None          # el tercer mecanismo no ensambla en esta pose
        D3 = sols[1]             # rama dorsal ('-perp')
        ang_rocker = np.arctan2(D3[1] - py_ifd, D3[0] - px_ifd)

        # Calibrar gamma_bracket3 en el 1er paso ensamblable para conservar la
        # flexion natural inicial (theta_fm + theta_aux_fd).
        if gamma_bracket3 is None:
            gamma_bracket3 = (theta_fm + theta_aux_fd) - ang_rocker

        theta_fd = ang_rocker + gamma_bracket3

        # Posicion de la punta (extremo de la falange distal)
        px_tip = FD_REAL * np.cos(theta_fd) + px_ifd
        py_tip = FD_REAL * np.sin(theta_fd) + py_ifd

        # Rechazar valores fuera del espacio de trabajo esperado
        if (np.abs(px_tip) > 0.3 or np.abs(py_tip) > 0.3
                or np.isnan(px_tip) or np.isnan(py_tip)):
            return None

        PXifp.append(px_ifp); PYifp.append(py_ifp)
        PXifd.append(px_ifd); PYifd.append(py_ifd)
        PXtip.append(px_tip); PYtip.append(py_tip)
        TH_FM.append(theta_fm); TH_FD.append(theta_fd)

    # -----------------------------------------------------------------
    # RESTRICCION FISIOLOGICA DE LA FLEXION DISTAL (DIP):
    # La flexion relativa de la IFD a lo largo del cierre no puede exceder un
    # maximo plausible (~35 deg). Evita soluciones que abusan del tercer
    # mecanismo con una flexion distal irreal solo para encajar la punta.
    # -----------------------------------------------------------------
    dip_rel = np.unwrap(np.asarray(TH_FD)) - np.unwrap(np.asarray(TH_FM))
    if np.ptp(dip_rel) > np.deg2rad(DIP_MAX_DEG):
        return None

    return {
        'ifp': np.column_stack((PXifp, PYifp)),
        'ifd': np.column_stack((PXifd, PYifd)),
        'tip': np.column_stack((PXtip, PYtip)),
        'theta_fm': np.asarray(TH_FM),
        'theta_fd': np.asarray(TH_FD)
    }

# ==============================================================================
# --- 6. FUNCION OBJETIVO ---
# ==============================================================================
# Peso extra para IFD y Punta donde el gancho era visible
W_IFP = 0.25
W_IFD = 0.375
W_TIP = 0.375

# Peso de la penalizacion de monotonicidad (escala en metros)
W_MONO = 5.0

# Peso de la penalizacion del PERFIL ANGULAR de la IFD/DIP.
# Obliga a que el angulo distal del exo (theta_fd - theta_fm) siga el perfil
# real de la DIP del mocap, evitando que el tercer mecanismo produzca una
# flexion distal exagerada (no fisiologica) con tal de encajar la punta.
# El error angular (rad) se convierte a metros multiplicando por FD_REAL
# (desplazamiento equivalente en la punta), para ser comparable al chamfer.
# Configurable por entorno para calibrar el balance forma/DIP.
W_DIP = float(os.environ.get('W_DIP', '0.3'))

def fitness_function(p):
    sim_data = run_kinematics(p, theta_input)
    if sim_data is None:
        return 1000.0

    all_mocap = np.vstack([mocap_pts[k] for k in ('ifp', 'ifd', 'tip')])
    all_sim   = np.vstack([sim_data[k]  for k in ('ifp', 'ifd', 'tip')])

    R, t = optimal_rigid_transform(all_mocap, all_sim)

    aligned = {k: apply_transform(sim_data[k], R, t) for k in ('ifp', 'ifd', 'tip')}

    err_ifp = chamfer_distance(mocap_pts['ifp'], aligned['ifp'])
    err_ifd = chamfer_distance(mocap_pts['ifd'], aligned['ifd'])
    err_tip = chamfer_distance(mocap_pts['tip'], aligned['tip'])

    # Penalizacion de monotonicidad sobre las trayectorias alineadas
    mono_ifd = monotonicity_penalty(aligned['ifd'])
    mono_tip = monotonicity_penalty(aligned['tip'])

    # Penalizacion del perfil angular de la IFD: el angulo distal del exo
    # (theta_fd - theta_fm) debe seguir la variacion real de la DIP del mocap.
    # La rotacion de alineacion rigida se cancela en la DIFERENCIA de angulos.
    exo_dip     = np.unwrap(sim_data['theta_fd']) - np.unwrap(sim_data['theta_fm'])
    exo_dip_rel = exo_dip - exo_dip[0]
    dip_error   = FD_REAL * np.mean(np.abs(exo_dip_rel - dip_rel_mocap))

    shape_error = W_IFP*err_ifp + W_IFD*err_ifd + W_TIP*err_tip
    mono_error  = W_MONO * (mono_ifd + mono_tip)
    dip_pen     = W_DIP * dip_error

    return shape_error + mono_error + dip_pen

# ==============================================================================
# --- 7. BOUNDS ---
# Bounds ajustados para dimensiones realistas de un exoesqueleto de mano
# (max ~80mm por eslabon). Se reduce el espacio de busqueda para evitar
# soluciones con longitudes mayores a 9cm que no son fabricables ni
# compatibles con el modelo CAD.
# ==============================================================================
bounds = [
    (0.015, 0.075), (0.015, 0.075),               # Bancada1, Bancada2 (max 75mm)
    (0.01,  0.08),  (0.01,  0.08),  (0.01, 0.08), # Link1, Link2, Link3 (max 80mm)
    (0.01,  0.08),  (0.01,  0.08),                 # Link4, Link5 (max 80mm)
    (0.01,  0.08),  (0.01,  0.08),  (0.01, 0.08), # Link6, Link7, Link8 (max 80mm)
    (0.01,  0.07),                                  # Link10 (max 70mm)
    (0.005, 0.035), (0.005, 0.023),                 # hsp, dsp  (dsp < 0.0245 para r1m2 > 0)
    (0.0,   np.pi), (0.0,  np.pi),                 # theta_aux_fm, theta_aux_fd
    (1.0,   8.0),                                   # gear_ratio
    (-np.pi, np.pi),                                # theta_offset
    # --- Tercer mecanismo de 4 barras (cinematica IFD/DIP) ---
    (0.010, 0.050),                                 # Link9_3  acoplador (max 50mm)
    (0.010, 0.050),                                 # Link10_3 balancin  (max 50mm)
    (-0.030, 0.010),                                # back3_3  (neg = soporte adelantado)
    (0.001, 0.015)                                  # up3_3    standoff dorsal (1-15mm)
]

# ==============================================================================
# --- 8. OPTIMIZACION BAYESIANA (Optuna) ---
# ==============================================================================
def objective_optuna(trial):
    strategy      = trial.suggest_categorical('strategy',
                        ['best1exp', 'rand1bin', 'best1bin'])
    popsize       = trial.suggest_int('popsize', 15, 30)
    mut_min       = trial.suggest_float('mut_min', 0.5, 0.9)
    mut_max       = trial.suggest_float('mut_max', mut_min + 0.1, 1.5)
    recombination = trial.suggest_float('recombination', 0.5, 0.9)

    res = differential_evolution(
        fitness_function, bounds,
        strategy=strategy, popsize=popsize,
        mutation=(mut_min, mut_max), recombination=recombination,
        maxiter=DE_MAXITER_OPTUNA, tol=1e-3, polish=False,
        updating='immediate', workers=1
    )
    return res.fun

# ==============================================================================
# --- 9. EJECUCION PRINCIPAL ---
# ==============================================================================
if __name__ == '__main__':
    # Quick validation with MATLAB-equivalent parameters (converted to meters)
    # Incluye los 4 parametros del tercer mecanismo con los valores ORIGINALES
    # del disenador (Link9=25mm, Link10=35mm, back3=-12mm, up3=2mm).
    p_matlab = [0.018, 0.020, 0.035, 0.049, 0.025, 0.020, 0.025,
                0.055, 0.035, 0.052, 0.04601,
                0.017, 0.018,
                np.deg2rad(51.39), np.deg2rad(38.78),
                2.0, np.deg2rad(109),
                0.025, 0.035, -0.012, 0.002]
    test_result = run_kinematics(p_matlab, theta_input)
    if test_result is not None:
        print(">> VALIDACION: Cinematica con parametros MATLAB - OK")
        test_fitness = fitness_function(p_matlab)
        print(f"   Fitness con params MATLAB: {test_fitness:.6f}")
    else:
        print(">> VALIDACION: Cinematica con parametros MATLAB - FALLO")
        print("   ADVERTENCIA: Los parametros de referencia no producen una solucion valida.")

    print('\n====================================================')
    print('>> ETAPA 1: Busqueda de Estrategia con Optuna')
    print('====================================================')

    study = optuna.create_study(direction='minimize')
    t0 = time.time()
    study.optimize(objective_optuna, n_trials=N_TRIALS_OPTUNA)
    best_hp = study.best_params
    print(f'\n>> Optuna finalizado en {time.time()-t0:.1f}s')
    print(f'   Mejor estrategia : {best_hp["strategy"]}')
    print(f'   Mejor popsize    : {best_hp["popsize"]}')

    print('\n====================================================')
    print('>> ETAPA 2: Optimizacion Cinematica Profunda (ED)')
    print('====================================================')

    resultado = differential_evolution(
        fitness_function, bounds,
        strategy=best_hp['strategy'],
        popsize=best_hp['popsize'],
        mutation=(best_hp['mut_min'], best_hp['mut_max']),
        recombination=best_hp['recombination'],
        maxiter=DE_MAXITER, tol=1e-5,
        polish=True, disp=True,
        updating='immediate', workers=1
    )

    p_opt    = resultado.x
    best_sim = run_kinematics(p_opt, theta_input)

    if best_sim is None:
        print("\n>> ADVERTENCIA: El resultado final no produce una cinematica valida.")
        print("   Considere ampliar los bounds o aumentar maxiter.")
        exit()

    # --- Transformacion de alineacion final ---
    all_mocap = np.vstack([mocap_pts[k] for k in ('ifp', 'ifd', 'tip')])
    all_sim   = np.vstack([best_sim[k]  for k in ('ifp', 'ifd', 'tip')])

    R_opt, t_opt = optimal_rigid_transform(all_mocap, all_sim)
    angulo_montaje = np.rad2deg(np.arctan2(R_opt[1, 0], R_opt[0, 0]))

    sim_aligned = {k: apply_transform(best_sim[k], R_opt, t_opt)
                   for k in ('ifp', 'ifd', 'tip')}

    err_ifp_mm = chamfer_distance(mocap_pts['ifp'], sim_aligned['ifp']) * 1000
    err_ifd_mm = chamfer_distance(mocap_pts['ifd'], sim_aligned['ifd']) * 1000
    err_tip_mm = chamfer_distance(mocap_pts['tip'], sim_aligned['tip']) * 1000
    error_global_mm = (err_ifp_mm + err_ifd_mm + err_tip_mm) / 3.0

    # --- Reporte de resultados ---
    print('\n====================================================')
    print('>> RESULTADOS CINEMATICOS')
    print('====================================================')
    print(f"   Error Global : {error_global_mm:.3f} mm")
    print(f"   Error IFP    : {err_ifp_mm:.3f} mm")
    print(f"   Error IFD    : {err_ifd_mm:.3f} mm")
    print(f"   Error Punta  : {err_tip_mm:.3f} mm\n")

    nombres = [
        "Bancada1 (m)",            "Bancada2 (m)",
        "Link1 (m)",               "Link2 (m)",
        "Link3 (m)",               "Link4 (m)",
        "Link5 (m)",               "Link6 (m)",
        "Link7 (m)",               "Link8 (m)",
        "Link10 / c2 (m)",
        "hsp (m)",                 "dsp (m)",
        "Theta Aux FM (rad)",      "Theta Aux FD (rad)",
        "Relacion de engranaje",   "Theta Offset (rad)",
        "Link9_3  acoplador (m)",  "Link10_3 balancin (m)",
        "back3_3  soporte (m)",    "up3_3    standoff (m)"
    ]

    print('>> PARAMETROS DIMENSIONALES DEL MECANISMO')
    for nombre, valor in zip(nombres, p_opt):
        print(f"   {nombre:30s}: {valor:.6f}")

    print('\n>> PARAMETROS DE MONTAJE (transformacion rigida)')
    print(f'   Traslacion X : {t_opt[0]*1000:.2f} mm')
    print(f'   Traslacion Y : {t_opt[1]*1000:.2f} mm')
    print(f'   Rotacion Base: {angulo_montaje:.2f} deg')

    # --- Tercer mecanismo de 4 barras (cinematica IFD/DIP) ---
    print('\n>> TERCER MECANISMO DE 4 BARRAS (IFD/DIP) [mm]')
    print(f'   Link9_3  acoplador (D3->Pa)   : {p_opt[17]*1000:.2f} mm')
    print(f'   Link10_3 balancin  (IFD->D3)  : {p_opt[18]*1000:.2f} mm')
    print(f'   back3_3  soporte S3 (a lo largo): {p_opt[19]*1000:.2f} mm')
    print(f'   up3_3    standoff dorsal       : {p_opt[20]*1000:.2f} mm')

    # Rango de movimiento RELATIVO de la IFD (flexion distal gradual real).
    # Se desenrollan los angulos (np.unwrap) para evitar saltos de +-360 deg.
    rel_dip_deg = np.rad2deg(np.unwrap(best_sim['theta_fd'])
                             - np.unwrap(best_sim['theta_fm']))
    rel_dip_rel = rel_dip_deg - rel_dip_deg[0]   # respecto a la pose inicial
    print(f'\n   Flexion distal relativa IFD: '
          f'min={rel_dip_rel.min():+.2f} deg, max={rel_dip_rel.max():+.2f} deg, '
          f'ROM={rel_dip_rel.max()-rel_dip_rel.min():.2f} deg (movimiento gradual)')
    print('====================================================\n')

    # --- Grafica de resultados ---
    fig, ax = plt.subplots(figsize=(11, 8))
    lw = 2.5

    ax.plot(mocap_pts['ifp'][:, 0]*1000, mocap_pts['ifp'][:, 1]*1000,
            'r--', lw=lw, alpha=0.65, label='MOCAP IFP')
    ax.plot(mocap_pts['ifd'][:, 0]*1000, mocap_pts['ifd'][:, 1]*1000,
            'g--', lw=lw, alpha=0.65, label='MOCAP IFD')
    ax.plot(mocap_pts['tip'][:, 0]*1000, mocap_pts['tip'][:, 1]*1000,
            'b--', lw=lw, alpha=0.65, label='MOCAP Punta')

    ax.plot(sim_aligned['ifp'][:, 0]*1000, sim_aligned['ifp'][:, 1]*1000,
            'r-', lw=lw, label='EXO IFP (Fisico)')
    ax.plot(sim_aligned['ifd'][:, 0]*1000, sim_aligned['ifd'][:, 1]*1000,
            'g-', lw=lw, label='EXO IFD (Fisico)')
    ax.plot(sim_aligned['tip'][:, 0]*1000, sim_aligned['tip'][:, 1]*1000,
            'b-', lw=lw, label='EXO Punta (Fisico)')

    ax.set_aspect('equal')
    ax.grid(True, linestyle=':', alpha=0.7)
    ax.legend(loc='upper right', fontsize=9)
    ax.set_title(
        f'Sintesis de Biofidelidad - Pinza Fina\nError Global: {error_global_mm:.3f} mm',
        fontsize=14, fontweight='bold'
    )
    ax.set_xlabel('Eje X (mm)', fontsize=12)
    ax.set_ylabel('Eje Y (mm)', fontsize=12)
    plt.tight_layout()
    plt.savefig('Resultados_Biofidelidad_Pinza_Fina.png', dpi=150)
    plt.close()

    # --- Guardado ---
    np.savetxt("Parametros_Optimizados_Pinza_Fina.txt", p_opt,
               header="Eslabones y offsets optimizados (21 parametros) - Pinza Fina. "
                      "Indices 0-16: mecanismo original; 17=Link9_3, 18=Link10_3, "
                      "19=back3_3, 20=up3_3 (tercer mecanismo de 4 barras IFD/DIP)")

    if len(mocap_pts['ifp']) == len(sim_aligned['ifp']):
        pd.DataFrame({
            'x_ifp_mocap': mocap_pts['ifp'][:, 0],
            'y_ifp_mocap': mocap_pts['ifp'][:, 1],
            'x_ifp_exo':   sim_aligned['ifp'][:, 0],
            'y_ifp_exo':   sim_aligned['ifp'][:, 1],
            'x_ifd_mocap': mocap_pts['ifd'][:, 0],
            'y_ifd_mocap': mocap_pts['ifd'][:, 1],
            'x_ifd_exo':   sim_aligned['ifd'][:, 0],
            'y_ifd_exo':   sim_aligned['ifd'][:, 1],
            'x_tip_mocap': mocap_pts['tip'][:, 0],
            'y_tip_mocap': mocap_pts['tip'][:, 1],
            'x_tip_exo':   sim_aligned['tip'][:, 0],
            'y_tip_exo':   sim_aligned['tip'][:, 1],
        }).to_csv('Resultados_Alineados_Pinza_Fina.csv', index=False)
        print(">> Archivos guardados: Parametros_Optimizados_Pinza_Fina.txt, "
              "Resultados_Alineados_Pinza_Fina.csv, Resultados_Biofidelidad_Pinza_Fina.png")
