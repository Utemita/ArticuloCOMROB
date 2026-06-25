"""
modelo_simplificado.py
======================
Modelo cinematico SIMPLIFICADO del exo -- solo llega hasta la trayectoria de la
FALANGE MEDIAL (articulacion IFD inclusive). Esto es pa validar que los
algoritmos de optimizacion (ED vs GA) funcionan bien sobre un problema mas
chiquito antes de meterle las 3 falanges completas.

Que le quitamos respecto al modelo completo (exo_18_pinza_fina.py):
  * Se elimina la falange distal y TODO el tercer mecanismo de 4 barras
    (Link9_3, Link10_3 y el soporte dorsal back3_3/up3_3).
  * Se quita el offset distal theta_aux_fd (ya no hay punta que ubicar).
  * Los objetivos quedan en dos trayectorias: IFP e IFD.
  * El vector de diseno pasa de 21 a 16 parametros.

Mejoras que le metimos pa el articulo:
  * DIMENSIONES REDUCIDAS: eslabones acotados a max 60 mm (antes era 80 mm)
    pa que el mecanismo sea mas compacto y fabricable.
  * ANGULO AUXILIAR MEDIAL-c2 REDUCIDO: theta_aux_fm ahora tiene limite
    superior de 70 deg (antes 180 deg), con una regularizacion suave que
    favorece valores chicos SIN echar a perder el ajuste de forma.
  * VALIDACION CINEMATICA: validar_cinematica() checa que los parametros son
    viables y que el mecanismo ensambla sin NaN ni retrocesos en todo el
    barrido de la manivela.

Ojo: los bounds y la fitness_function son IDENTICOS pa ED y GA, asi la
comparacion mide el algoritmo y no el planteamiento.
"""
import os
import numpy as np

import comun
from comun import (FP_REAL, FM_REAL, sol_5_barras, solve_four_bar,
                   chamfer_distance, optimal_rigid_transform, apply_transform,
                   monotonicity_penalty)

# --- Datos MOCAP (solo usamos IFP e IFD en esta version) ---
_MOCAP = comun.cargar_mocap(os.path.join(os.path.dirname(__file__),
                                         "mocap_pinza_fina_120pts.csv"))
mocap_pts = {'ifp': _MOCAP['ifp'], 'ifd': _MOCAP['ifd']}
theta_input = _MOCAP['theta_input']
N_PUNTOS = _MOCAP['n']

# ==============================================================================
# NOMBRES DE LOS 16 PARAMETROS DE DISENO (version simplificada)
# ==============================================================================
NOMBRES = [
    "Bancada1 (m)", "Bancada2 (m)",
    "Link1 (m)", "Link2 (m)", "Link3 (m)", "Link4 (m)", "Link5 (m)",
    "Link6 (m)", "Link7 (m)", "Link8 (m)", "Link10 / c2 (m)",
    "hsp (m)", "dsp (m)",
    "Theta Aux FM (rad)", "Relacion de engranaje", "Theta Offset (rad)",
]

# ==============================================================================
# --- LIMITES (BOUNDS) con dimensiones y angulo auxiliar reducidos ---
# Eslabones: max 60 mm (antes 80). theta_aux_fm: max 70 deg (antes 180).
# ==============================================================================
bounds = [
    (0.015, 0.060), (0.015, 0.060),                  # Bancada1, Bancada2
    (0.010, 0.060), (0.010, 0.060), (0.010, 0.060),  # Link1, Link2, Link3
    (0.010, 0.060), (0.010, 0.060),                  # Link4, Link5
    (0.010, 0.060), (0.010, 0.060), (0.010, 0.060),  # Link6, Link7, Link8
    (0.010, 0.055),                                  # Link10 / c2
    (0.005, 0.030), (0.005, 0.023),                  # hsp, dsp
    (0.0, np.deg2rad(70.0)),                         # theta_aux_fm (REDUCIDO)
    (1.0, 8.0),                                      # gear_ratio
    (-np.pi, np.pi),                                 # theta_offset
]

# Pesos del objetivo (forma + monotonicidad + regularizacion suave)
W_IFP = 0.40
W_IFD = 0.60
W_MONO = 5.0
# Regularizacion MUY suave: empuja a dimensiones y angulo aux chicos
# pero sin degradar el ajuste (chamfer ~ 3e-3 m; estos terminos ~ 1e-5).
W_REG_DIM = float(os.environ.get('W_REG_DIM', '5e-4'))   # escala de longitudes
W_REG_AUX = float(os.environ.get('W_REG_AUX', '5e-4'))   # escala del angulo aux


# ==============================================================================
# --- MODELO CINEMATICO SIMPLIFICADO (hasta la IFD / falange medial) ---
# ==============================================================================
def run_kinematics(p, th_input):
    """Cinematica directa del exo hasta el extremo de la falange medial (IFD).

    Regresa un dict con las trayectorias 'ifp' e 'ifd' (Nx2, m) y el perfil
    angular 'theta_fm' (rad). Si el mecanismo no jala en algun punto del
    barrido (longitudes invalidas, no-ensamble o retroceso), regresa None.
    """
    (Bancada1, Bancada2, Link1, Link2, Link3, Link4, Link5,
     Link6, Link7, Link8, Link10, hsp, dsp,
     theta_aux_fm, gear_ratio, theta_offset) = p

    # Checamos que los parametros tengan sentido
    if gear_ratio <= 0:
        return None
    if min(p[:11]) <= 0.005:           # longitudes minimas de eslabon
        return None
    if hsp <= 0 or dsp <= 0:
        return None
    if FP_REAL - 2.0 * dsp <= 0.001:
        return None

    r3_val = Bancada1 / 2.0
    theta14B = np.pi / 2.0
    c1 = np.sqrt(hsp**2 + dsp**2)
    rs2 = np.sqrt(hsp**2 + (FP_REAL - dsp)**2)
    theta_aux_s2 = np.arctan2(hsp, FP_REAL - dsp)

    PXifp, PYifp, PXifd, PYifd, TH_FM = [], [], [], [], []
    prev_theta_fm = None

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

        theta_fp = theta4 + np.arctan2(hsp, dsp)
        px_ifp = FP_REAL * np.cos(theta_fp) - Bancada2 * np.cos(theta14B) - r3_val
        py_ifp = FP_REAL * np.sin(theta_fp) - Bancada2 * np.sin(theta14B)

        pxs1 = c1 * np.cos(theta4) - Bancada2 * np.cos(theta14B) - r3_val
        pys1 = c1 * np.sin(theta4) - Bancada2 * np.sin(theta14B)

        theta_ps2 = theta_fp - theta_aux_s2
        pxs2 = rs2 * np.cos(theta_ps2) - Bancada2 * np.cos(theta14B) - r3_val
        pys2 = rs2 * np.sin(theta_ps2) - Bancada2 * np.sin(theta14B)

        pxm4 = Link4 * np.cos(th2) - r3_val
        pym4 = Link4 * np.sin(th2)

        # --- Mecanismo 2: 5 barras (referencia secundaria) ---
        theta_roll = np.arctan2(pym4 - pys1, pxm4 - pxs1)
        theta1m2 = np.arctan2(pys2 - pys1, pxs2 - pxs1) - theta_roll
        theta2m2 = np.arctan2(pyP - pym4, pxP - pxm4) - theta_roll

        res5_2 = sol_5_barras(FP_REAL - 2.0 * dsp, Link7, Link5 / 2.0, Link3, Link6,
                              theta1m2, theta2m2)
        if res5_2 is None:
            return None
        px_local, py_local = res5_2

        mag = np.sqrt(px_local**2 + py_local**2)
        theta_loc = np.arctan2(py_local, px_local)
        px_aux = (pxs1 + pxm4) / 2.0
        py_aux = (pys1 + pym4) / 2.0
        pxP2 = mag * np.cos(theta_loc + theta_roll) + px_aux
        pyP2 = mag * np.sin(theta_loc + theta_roll) + py_aux

        # --- Mecanismo 2: 4 barras ---
        theta1m42 = np.arctan2(pys2 - py_ifp, pxs2 - px_ifp)
        theta2m42 = np.arctan2(pyP2 - pys2, pxP2 - pxs2)

        theta4m2 = solve_four_bar(Link7, Link8, Link10, c1, theta2m42, theta1m42)
        if theta4m2 is None:
            return None

        # Angulo de la falange medial (Link10/c2 + theta_aux_fm)
        theta_fm = theta4m2 + theta_aux_fm

        # Restriccion anti-gancho: theta_fm debe ser monotono creciente
        if prev_theta_fm is not None:
            delta_fm = (theta_fm - prev_theta_fm + np.pi) % (2 * np.pi) - np.pi
            if delta_fm < -np.deg2rad(0.5):
                return None  # se esta regresando, no sirve
        prev_theta_fm = theta_fm

        # Posicion de la IFD (extremo de la falange medial)
        px_ifd = FM_REAL * np.cos(theta_fm) + px_ifp
        py_ifd = FM_REAL * np.sin(theta_fm) + py_ifp

        # Checamos que no truene (valores fuera de rango o NaN)
        if (np.abs(px_ifd) > 0.3 or np.abs(py_ifd) > 0.3
                or not np.isfinite(px_ifd) or not np.isfinite(py_ifd)):
            return None

        PXifp.append(px_ifp); PYifp.append(py_ifp)
        PXifd.append(px_ifd); PYifd.append(py_ifd)
        TH_FM.append(theta_fm)

    return {
        'ifp': np.column_stack((PXifp, PYifp)),
        'ifd': np.column_stack((PXifd, PYifd)),
        'theta_fm': np.asarray(TH_FM),
    }


# ==============================================================================
# --- FUNCION OBJETIVO (identica pa ED y GA) ---
# ==============================================================================
def fitness_function(p):
    sim = run_kinematics(p, theta_input)
    if sim is None:
        return 1000.0  # penalizacion pa mecanismos invalidos

    all_mocap = np.vstack([mocap_pts['ifp'], mocap_pts['ifd']])
    all_sim = np.vstack([sim['ifp'], sim['ifd']])
    R, t = optimal_rigid_transform(all_mocap, all_sim)
    aligned = {k: apply_transform(sim[k], R, t) for k in ('ifp', 'ifd')}

    err_ifp = chamfer_distance(mocap_pts['ifp'], aligned['ifp'])
    err_ifd = chamfer_distance(mocap_pts['ifd'], aligned['ifd'])
    mono_ifd = monotonicity_penalty(aligned['ifd'])

    shape_error = W_IFP * err_ifp + W_IFD * err_ifd
    mono_error = W_MONO * mono_ifd

    # Regularizacion suave: empuja a dimensiones y angulo aux chicos
    suma_links = float(np.sum(p[:11]))                # 11 longitudes (m)
    reg_dim = W_REG_DIM * suma_links
    reg_aux = W_REG_AUX * (p[13] / np.pi)             # theta_aux_fm normalizado

    return shape_error + mono_error + reg_dim + reg_aux


# ==============================================================================
# --- EVALUACION / METRICAS EN mm (pa los reportes) ---
# ==============================================================================
def evaluar(p):
    """Calcula errores en mm y datos alineados. Si no jala, regresa None."""
    sim = run_kinematics(p, theta_input)
    if sim is None:
        return None
    all_mocap = np.vstack([mocap_pts['ifp'], mocap_pts['ifd']])
    all_sim = np.vstack([sim['ifp'], sim['ifd']])
    R, t = optimal_rigid_transform(all_mocap, all_sim)
    aligned = {k: apply_transform(sim[k], R, t) for k in ('ifp', 'ifd')}
    err_ifp = chamfer_distance(mocap_pts['ifp'], aligned['ifp']) * 1000
    err_ifd = chamfer_distance(mocap_pts['ifd'], aligned['ifd']) * 1000
    return {
        'err_ifp_mm': err_ifp,
        'err_ifd_mm': err_ifd,
        'err_global_mm': (err_ifp + err_ifd) / 2.0,
        'R': R, 't': t,
        'aligned': aligned,
        'sim': sim,
        'angulo_montaje_deg': np.rad2deg(np.arctan2(R[1, 0], R[0, 0])),
    }


# ==============================================================================
# --- VALIDACION CINEMATICA (checa que el mecanismo sea viable) ---
# ==============================================================================
def validar_cinematica(p, verbose=False):
    """Checa que los parametros producen un mecanismo viable en todo el barrido
    de la manivela: que ensamble, sin NaN/Inf y sin que se regrese la falange
    medial.

    Regresa (ok: bool, info: dict).
    """
    info = {}
    sim = run_kinematics(p, theta_input)
    if sim is None:
        info['motivo'] = ("La cinematica no jala en algun punto del barrido "
                          "(no-ensamble, longitud invalida o retroceso de la "
                          "falange medial).")
        if verbose:
            print(">> VALIDACION CINEMATICA: FALLO ->", info['motivo'])
        return False, info

    # Checamos que no haya NaN ni Inf
    for k in ('ifp', 'ifd'):
        if not np.all(np.isfinite(sim[k])):
            info['motivo'] = f"Valores no finitos en la trayectoria {k}."
            if verbose:
                print(">> VALIDACION CINEMATICA: FALLO ->", info['motivo'])
            return False, info

    # Que se haya resuelto todo el barrido completo
    n_ok = len(sim['ifp'])
    if n_ok != N_PUNTOS:
        info['motivo'] = (f"Solo se resolvieron {n_ok}/{N_PUNTOS} poses del "
                          f"barrido.")
        if verbose:
            print(">> VALIDACION CINEMATICA: FALLO ->", info['motivo'])
        return False, info

    # Monotonicidad real de theta_fm (que no se regrese)
    th_fm = np.unwrap(sim['theta_fm'])
    delta = np.diff(th_fm)
    info['theta_fm_monotono'] = bool(np.all(delta >= -np.deg2rad(0.5)))
    info['rom_medial_deg'] = float(np.rad2deg(th_fm[-1] - th_fm[0]))
    info['poses_resueltas'] = n_ok
    info['motivo'] = "OK: mecanismo viable en todo el recorrido."
    if verbose:
        print(f">> VALIDACION CINEMATICA: OK  "
              f"({n_ok}/{N_PUNTOS} poses, ROM medial "
              f"{info['rom_medial_deg']:.1f} deg)")
    return True, info
