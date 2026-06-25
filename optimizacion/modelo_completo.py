"""
modelo_completo.py
==================
Modelo cinematico COMPLETO del exoesqueleto de rehabilitacion de dedo: las TRES
falanges (proximal, medial y distal) y los CINCO mecanismos que lo componen:

  ETAPA 1 (5 barras + 4 barras) -> falange proximal (articulacion MCP)
  ETAPA 2 (5 barras + 4 barras) -> falange medial   (articulacion IFP)
  ETAPA 3 (4 barras)            -> falange distal    (articulacion IFD/DIP)

Es la regeneracion del optimizador original (`exo_18_pinza_fina.py`) integrando
las MEJORAS validadas en el modelo simplificado:
  * Dimensiones acotadas (eslabones <= 60 mm; tercer mecanismo <= 45 mm).
  * Angulo auxiliar medial-c2 (`theta_aux_fm`) limitado a 70 deg.
  * Regularizacion suave que favorece dimensiones y angulo auxiliar pequenos
    sin degradar el ajuste.
  * Funcion `validar_cinematica()` que verifica la viabilidad del mecanismo en
    TODO el barrido de la manivela.

El enfoque de optimizacion asociado es EVOLUCION DIFERENCIAL
(`optimizar_completo_ED.py`), por ser el ganador del estudio comparativo
realizado sobre el modelo simplificado.

Vector de diseno: 21 parametros (17 del mecanismo base + 4 del tercer 4 barras).
Todas las longitudes en metros y los angulos en radianes.
"""
import os
import numpy as np

import comun
from comun import (FP_REAL, FM_REAL, FD_REAL, sol_5_barras, solve_four_bar,
                   circle_intersections, chamfer_distance,
                   optimal_rigid_transform, apply_transform,
                   monotonicity_penalty)

# Tope fisiologico de la flexion distal relativa (DIP) en grados.
DIP_MAX_DEG = float(os.environ.get('DIP_MAX_DEG', '35.0'))

# --- Datos MOCAP completos (IFP, IFD y punta) ---
_MOCAP = comun.cargar_mocap(os.path.join(os.path.dirname(__file__),
                                         "mocap_pinza_fina_120pts.csv"))
mocap_pts = {'ifp': _MOCAP['ifp'], 'ifd': _MOCAP['ifd'], 'tip': _MOCAP['tip']}
dip_rel_mocap = _MOCAP['dip_rel']
theta_input = _MOCAP['theta_input']
N_PUNTOS = _MOCAP['n']

# ==============================================================================
# NOMBRES DE LOS 21 PARAMETROS DE DISENO
# ==============================================================================
NOMBRES = [
    "Bancada1 (m)", "Bancada2 (m)",
    "Link1 (m)", "Link2 (m)", "Link3 (m)", "Link4 (m)", "Link5 (m)",
    "Link6 (m)", "Link7 (m)", "Link8 (m)", "Link10 / c2 (m)",
    "hsp (m)", "dsp (m)",
    "Theta Aux FM (rad)", "Theta Aux FD (rad)",
    "Relacion de engranaje", "Theta Offset (rad)",
    "Link9_3 acoplador (m)", "Link10_3 balancin (m)",
    "back3_3 soporte (m)", "up3_3 standoff (m)",
]

# ==============================================================================
# --- LIMITES (BOUNDS) con DIMENSIONES y ANGULO AUXILIAR REDUCIDOS ---
# ==============================================================================
bounds = [
    (0.015, 0.060), (0.015, 0.060),                  # Bancada1, Bancada2
    (0.010, 0.060), (0.010, 0.060), (0.010, 0.060),  # Link1, Link2, Link3
    (0.010, 0.060), (0.010, 0.060),                  # Link4, Link5
    (0.010, 0.060), (0.010, 0.060), (0.010, 0.060),  # Link6, Link7, Link8
    (0.010, 0.055),                                  # Link10 / c2
    (0.005, 0.030), (0.005, 0.023),                  # hsp, dsp
    (0.0, np.deg2rad(70.0)),                         # theta_aux_fm (REDUCIDO)
    (0.0, np.pi),                                    # theta_aux_fd
    (1.0, 8.0),                                      # gear_ratio
    (-np.pi, np.pi),                                 # theta_offset
    # --- Tercer mecanismo de 4 barras (cinematica IFD/DIP) ---
    (0.010, 0.045),                                  # Link9_3  acoplador
    (0.010, 0.045),                                  # Link10_3 balancin
    (-0.030, 0.010),                                 # back3_3  soporte
    (0.001, 0.015),                                  # up3_3    standoff dorsal
]

# Pesos de la funcion objetivo
W_IFP = 0.25
W_IFD = 0.375
W_TIP = 0.375
W_MONO = 5.0
W_DIP = float(os.environ.get('W_DIP', '0.3'))
# Regularizacion suave (misma filosofia que el modelo simplificado)
W_REG_DIM = float(os.environ.get('W_REG_DIM', '5e-4'))
W_REG_AUX = float(os.environ.get('W_REG_AUX', '5e-4'))


# ==============================================================================
# --- MODELO CINEMATICO COMPLETO (3 falanges, 5 mecanismos) ---
# ==============================================================================
def run_kinematics(p, th_input):
    (Bancada1, Bancada2, Link1, Link2, Link3, Link4, Link5,
     Link6, Link7, Link8, Link10, hsp, dsp,
     theta_aux_fm, theta_aux_fd, gear_ratio, theta_offset,
     Link9_3, Link10_3, back3_3, up3_3) = p

    if gear_ratio <= 0:
        return None
    if min(p[:11]) <= 0.005:
        return None
    if Link9_3 <= 0.005 or Link10_3 <= 0.005:
        return None
    if up3_3 <= 0:
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

    PXifp, PYifp, PXifd, PYifd, PXtip, PYtip = [], [], [], [], [], []
    TH_FM, TH_FD = [], []
    prev_theta_fm = None
    gamma_bracket3 = None
    dorsal_sign3 = None

    for th2 in th_input:
        th1 = (th2 / gear_ratio) + theta_offset

        res5 = sol_5_barras(Link4, Link3, r3_val, Link1, Link2, th1, th2)
        if res5 is None:
            return None
        pxP, pyP = res5

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

        theta1m42 = np.arctan2(pys2 - py_ifp, pxs2 - px_ifp)
        theta2m42 = np.arctan2(pyP2 - pys2, pxP2 - pxs2)

        theta4m2 = solve_four_bar(Link7, Link8, Link10, c1, theta2m42, theta1m42)
        if theta4m2 is None:
            return None

        theta_fm = theta4m2 + theta_aux_fm

        if prev_theta_fm is not None:
            delta_fm = (theta_fm - prev_theta_fm + np.pi) % (2 * np.pi) - np.pi
            if delta_fm < -np.deg2rad(0.5):
                return None
        prev_theta_fm = theta_fm

        px_ifd = FM_REAL * np.cos(theta_fm) + px_ifp
        py_ifd = FM_REAL * np.sin(theta_fm) + py_ifp

        # --- TERCER MECANISMO DE 4 BARRAS (cinematica real IFD/DIP) ---
        prox_dir = np.array([np.cos(theta_fp), np.sin(theta_fp)])
        prox_norm = np.array([-prox_dir[1], prox_dir[0]])
        IFP_pt = np.array([px_ifp, py_ifp])
        IFD_pt = np.array([px_ifd, py_ifd])

        if dorsal_sign3 is None:
            P3_pt = IFP_pt + Link10 * np.array([np.cos(theta4m2), np.sin(theta4m2)])
            dorsal_sign3 = 1.0 if np.dot(P3_pt - IFP_pt, prox_norm) >= 0 else -1.0
        dorsal_norm = dorsal_sign3 * prox_norm

        Pa = IFP_pt - back3_3 * prox_dir + up3_3 * dorsal_norm
        sols = circle_intersections(Pa, Link9_3, IFD_pt, Link10_3)
        if sols is None:
            return None
        D3 = sols[1]
        ang_rocker = np.arctan2(D3[1] - py_ifd, D3[0] - px_ifd)

        if gamma_bracket3 is None:
            gamma_bracket3 = (theta_fm + theta_aux_fd) - ang_rocker
        theta_fd = ang_rocker + gamma_bracket3

        px_tip = FD_REAL * np.cos(theta_fd) + px_ifd
        py_tip = FD_REAL * np.sin(theta_fd) + py_ifd

        if (np.abs(px_tip) > 0.3 or np.abs(py_tip) > 0.3
                or not np.isfinite(px_tip) or not np.isfinite(py_tip)):
            return None

        PXifp.append(px_ifp); PYifp.append(py_ifp)
        PXifd.append(px_ifd); PYifd.append(py_ifd)
        PXtip.append(px_tip); PYtip.append(py_tip)
        TH_FM.append(theta_fm); TH_FD.append(theta_fd)

    # Restriccion fisiologica de la flexion distal (DIP <= DIP_MAX_DEG)
    dip_rel = np.unwrap(np.asarray(TH_FD)) - np.unwrap(np.asarray(TH_FM))
    if np.ptp(dip_rel) > np.deg2rad(DIP_MAX_DEG):
        return None

    return {
        'ifp': np.column_stack((PXifp, PYifp)),
        'ifd': np.column_stack((PXifd, PYifd)),
        'tip': np.column_stack((PXtip, PYtip)),
        'theta_fm': np.asarray(TH_FM),
        'theta_fd': np.asarray(TH_FD),
    }


# ==============================================================================
# --- FUNCION OBJETIVO ---
# ==============================================================================
def fitness_function(p):
    sim = run_kinematics(p, theta_input)
    if sim is None:
        return 1000.0

    all_mocap = np.vstack([mocap_pts[k] for k in ('ifp', 'ifd', 'tip')])
    all_sim = np.vstack([sim[k] for k in ('ifp', 'ifd', 'tip')])
    R, t = optimal_rigid_transform(all_mocap, all_sim)
    aligned = {k: apply_transform(sim[k], R, t) for k in ('ifp', 'ifd', 'tip')}

    err_ifp = chamfer_distance(mocap_pts['ifp'], aligned['ifp'])
    err_ifd = chamfer_distance(mocap_pts['ifd'], aligned['ifd'])
    err_tip = chamfer_distance(mocap_pts['tip'], aligned['tip'])

    mono_ifd = monotonicity_penalty(aligned['ifd'])
    mono_tip = monotonicity_penalty(aligned['tip'])

    exo_dip = np.unwrap(sim['theta_fd']) - np.unwrap(sim['theta_fm'])
    exo_dip_rel = exo_dip - exo_dip[0]
    dip_error = FD_REAL * np.mean(np.abs(exo_dip_rel - dip_rel_mocap))

    shape_error = W_IFP * err_ifp + W_IFD * err_ifd + W_TIP * err_tip
    mono_error = W_MONO * (mono_ifd + mono_tip)
    dip_pen = W_DIP * dip_error

    suma_links = float(np.sum(p[:11]))
    reg_dim = W_REG_DIM * suma_links
    reg_aux = W_REG_AUX * (p[13] / np.pi)

    return shape_error + mono_error + dip_pen + reg_dim + reg_aux


# ==============================================================================
# --- EVALUACION / METRICAS EN mm ---
# ==============================================================================
def evaluar(p):
    sim = run_kinematics(p, theta_input)
    if sim is None:
        return None
    all_mocap = np.vstack([mocap_pts[k] for k in ('ifp', 'ifd', 'tip')])
    all_sim = np.vstack([sim[k] for k in ('ifp', 'ifd', 'tip')])
    R, t = optimal_rigid_transform(all_mocap, all_sim)
    aligned = {k: apply_transform(sim[k], R, t) for k in ('ifp', 'ifd', 'tip')}
    err_ifp = chamfer_distance(mocap_pts['ifp'], aligned['ifp']) * 1000
    err_ifd = chamfer_distance(mocap_pts['ifd'], aligned['ifd']) * 1000
    err_tip = chamfer_distance(mocap_pts['tip'], aligned['tip']) * 1000
    rel_dip = np.rad2deg(np.unwrap(sim['theta_fd']) - np.unwrap(sim['theta_fm']))
    rel_dip -= rel_dip[0]
    return {
        'err_ifp_mm': err_ifp,
        'err_ifd_mm': err_ifd,
        'err_tip_mm': err_tip,
        'err_global_mm': (err_ifp + err_ifd + err_tip) / 3.0,
        'R': R, 't': t,
        'aligned': aligned,
        'sim': sim,
        'rom_dip_deg': float(rel_dip.max() - rel_dip.min()),
        'angulo_montaje_deg': np.rad2deg(np.arctan2(R[1, 0], R[0, 0])),
    }


# ==============================================================================
# --- VALIDACION CINEMATICA (viabilidad en TODO el movimiento) ---
# ==============================================================================
def validar_cinematica(p, verbose=False):
    info = {}
    sim = run_kinematics(p, theta_input)
    if sim is None:
        info['motivo'] = ("Cinematica no valida en algun punto (no-ensamble, "
                          "longitud invalida, retroceso medial o DIP fuera de rango).")
        if verbose:
            print(">> VALIDACION CINEMATICA: FALLO ->", info['motivo'])
        return False, info

    for k in ('ifp', 'ifd', 'tip'):
        if not np.all(np.isfinite(sim[k])):
            info['motivo'] = f"Valores no finitos en la trayectoria {k}."
            if verbose:
                print(">> VALIDACION CINEMATICA: FALLO ->", info['motivo'])
            return False, info

    n_ok = len(sim['ifp'])
    if n_ok != N_PUNTOS:
        info['motivo'] = f"Solo se resolvieron {n_ok}/{N_PUNTOS} poses."
        if verbose:
            print(">> VALIDACION CINEMATICA: FALLO ->", info['motivo'])
        return False, info

    th_fm = np.unwrap(sim['theta_fm'])
    info['theta_fm_monotono'] = bool(np.all(np.diff(th_fm) >= -np.deg2rad(0.5)))
    info['rom_medial_deg'] = float(np.rad2deg(th_fm[-1] - th_fm[0]))
    rel_dip = np.rad2deg(np.unwrap(sim['theta_fd']) - np.unwrap(sim['theta_fm']))
    rel_dip -= rel_dip[0]
    info['rom_dip_deg'] = float(rel_dip.max() - rel_dip.min())
    info['poses_resueltas'] = n_ok
    info['motivo'] = "OK: mecanismo viable en todo el recorrido."
    if verbose:
        print(f">> VALIDACION CINEMATICA: OK  ({n_ok}/{N_PUNTOS} poses, "
              f"ROM medial {info['rom_medial_deg']:.1f} deg, "
              f"ROM DIP {info['rom_dip_deg']:.1f} deg)")
    return True, info
