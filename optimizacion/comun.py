"""
comun.py
========
Aqui estan todas las funciones que se reusan en los experimentos de optimizacion
del exo de rehabilitacion de dedo (lo del articulo COMROB, la comparacion de
Algoritmos Geneticos vs Evolucion Diferencial).

Basicamente lo que hay aqui:
  1. Carga del MOCAP (mocap_pinza_fina_120pts.csv) y su preprocesado.
  2. Metricas de forma (Chamfer distance) y penalizacion de monotonicidad.
  3. Alineacion rigida optima 2D (tipo Kabsch/Procrustes, sin escala).
  4. Solvers cinematicos basicos (5 barras, 4 barras, interseccion de circulos)
     que usan tanto el modelo simplificado como el completo.
  5. Un ALGORITMO GENETICO de codificacion real hecho a mano (sin librerias
     externas) con una interfaz parecida a scipy.optimize.differential_evolution
     pa poder comparar los dos enfoques sobre la misma funcion objetivo.

Ojo: todas las longitudes en metros y angulos en radianes a menos que se diga
otra cosa.
"""
import os
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.signal import savgol_filter

# ==============================================================================
# --- 1. PARAMETROS ANTROPOMETRICOS (longitudes de las falanges del dedo) ---
# ==============================================================================
FP_REAL = 0.049   # Falange proximal (m)
FM_REAL = 0.026   # Falange medial   (m)
FD_REAL = 0.024   # Falange distal   (m)


# ==============================================================================
# --- 2. CARGA Y CORRECCION DE DATOS MOCAP ---
# ==============================================================================
def cargar_mocap(ruta="mocap_pinza_fina_120pts.csv", n_grados_input=85.0):
    """Carga los datos de captura de movimiento y los preprocesa.

    Aqui le metemos el mismo preprocesado que usaba el optimizador original:
      (a) Se invierte el recorrido (de flexion max -> extension a apertura -> cierre).
      (b) Se recorta DIP a >= 0 grados (el exo no hace hiperextension).
      (c) Suavizado Savitzky-Golay (ventana 15, orden 2) pa quitar ruido.

    Regresa un dict con:
      - 'ifp','ifd','tip' : nubes de puntos Nx2 de las articulaciones (m).
      - 'theta_fm','theta_fd' : orientaciones medial y distal (rad).
      - 'dip_rel' : perfil relativo del angulo DIP (rad).
      - 'theta_input' : barrido de la manivela principal (rad).
      - 'n' : numero de puntos.
    """
    datos = pd.read_csv(ruta)

    # (a) Invertir: ahora va de apertura -> cierre (movimiento de agarre)
    mcp_raw = datos['Theta_MCP'].values[::-1]
    pip_raw = datos['Theta_PIP'].values[::-1]
    dip_raw = datos['Theta_DIP'].values[::-1]
    n = len(mcp_raw)

    # (b) Recortar DIP a 0 grados minimo (nada de hiperextension)
    dip_raw = np.clip(dip_raw, 0.0, None)

    # (c) Suavizado Savitzky-Golay pa quitar ruido del mocap
    win = 15
    if n >= win:
        mcp = savgol_filter(np.deg2rad(mcp_raw), window_length=win, polyorder=2)
        pip = savgol_filter(np.deg2rad(pip_raw), window_length=win, polyorder=2)
        dip = savgol_filter(np.deg2rad(dip_raw), window_length=win, polyorder=2)
        dip = np.clip(dip, 0.0, None)
    else:
        mcp = np.deg2rad(mcp_raw)
        pip = np.deg2rad(pip_raw)
        dip = np.deg2rad(dip_raw)

    # Cinematica directa del dedo (cadena de cuerpos rigidos desde la MCF)
    seg_prox = mcp
    seg_med = mcp + pip
    seg_dist = mcp + pip + dip

    pxIFP = FP_REAL * np.cos(seg_prox)
    pyIFP = FP_REAL * np.sin(seg_prox)
    pxIFD = pxIFP + FM_REAL * np.cos(seg_med)
    pyIFD = pyIFP + FM_REAL * np.sin(seg_med)
    pxPF = pxIFD + FD_REAL * np.cos(seg_dist)
    pyPF = pyIFD + FD_REAL * np.sin(seg_dist)

    return {
        'ifp': np.column_stack((pxIFP, pyIFP)),
        'ifd': np.column_stack((pxIFD, pyIFD)),
        'tip': np.column_stack((pxPF, pyPF)),
        'theta_fm': seg_med,
        'theta_fd': seg_dist,
        'dip_rel': dip - dip[0],
        'theta_input': np.linspace(0, np.deg2rad(n_grados_input), n),
        'n': n,
    }


# ==============================================================================
# --- 3. METRICAS ---
# ==============================================================================
def chamfer_distance(curve_target, curve_sim):
    """Distancia de Chamfer bidireccional -- la metrica de forma que usamos."""
    dists = cdist(curve_target, curve_sim)
    return np.mean(np.min(dists, axis=1)) + np.mean(np.min(dists, axis=0))


def optimal_rigid_transform(target, sim):
    """Transformacion rigida optima (R + t) entre dos nubes de puntos 2D."""
    c_t = np.mean(target, axis=0)
    c_s = np.mean(sim, axis=0)
    H = (sim - c_s).T @ (target - c_t)
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
    """Penaliza si la trayectoria se regresa (inversiones de direccion)."""
    diffs = np.diff(curve, axis=0)
    arc = np.linalg.norm(diffs, axis=1)
    total = np.sum(arc)
    if total < 1e-9:
        return 0.0
    mean_dir = np.sum(diffs, axis=0) / (total + 1e-12)
    mean_dir /= (np.linalg.norm(mean_dir) + 1e-12)
    proj = diffs @ mean_dir
    return np.sum(np.clip(-proj, 0, None))


# ==============================================================================
# --- 4. SOLUCIONADORES CINEMATICOS BASICOS ---
# ==============================================================================
def sol_5_barras(r1, r2, r3, r4, r5, theta1, theta2):
    """Resuelve la posicion del acoplador de un mecanismo de 5 barras.
    Si no jala (discriminante negativo o denominador ~0), regresa None.
    """
    den = r4 * np.cos(theta2) - r1 * np.cos(theta1) + 2 * r3
    if np.abs(den) < 1e-4:
        return None
    e = (r1 * np.sin(theta1) - r4 * np.sin(theta2)) / den
    f = (2 * (r1 * r3 * np.cos(theta1) + r3 * r4 * np.cos(theta2))
         - r1**2 + r2**2 + r4**2 - r5**2) / (2 * den)
    d_ = e**2 + 1
    g = 2 * (e * f - e * r1 * np.cos(theta1) + e * r3 - r1 * np.sin(theta1))
    h = (f**2 - 2 * f * (r1 * np.cos(theta1) - r3)
         - 2 * r1 * r3 * np.cos(theta1) + r1**2 + r3**2 - r2**2)
    disc = g**2 - 4 * d_ * h
    if disc < 0:
        return None  # no ensambla
    py = (-g + np.sqrt(disc)) / (2 * d_)
    px = e * py + f
    return px, py


def solve_four_bar(a, b, c, d, theta2, theta1):
    """Resuelve el angulo del balancin de un 4 barras (rama abierta).
    Si no tiene solucion real, regresa None.
    """
    k1 = a * np.cos(theta2) + d * np.cos(theta1)
    k2 = a * np.sin(theta2) + d * np.sin(theta1)
    k3 = k1**2 + k2**2 + c**2 - b**2
    A1 = -2 * k1 * c - k3
    B1 = 4 * k2 * c
    C1 = 2 * k1 * c - k3
    disc = B1**2 - 4 * A1 * C1
    if disc < 0:
        return None  # no hay solucion real
    return 2 * np.arctan((-B1 - np.sqrt(disc)) / (2 * A1))


def circle_intersections(c0, r0, c1, r1):
    """Interseccion de dos circulos. Si no se tocan, regresa None."""
    c0 = np.asarray(c0, dtype=float)
    c1 = np.asarray(c1, dtype=float)
    dvec = c1 - c0
    dist = np.hypot(dvec[0], dvec[1])
    if dist > (r0 + r1) or dist < abs(r0 - r1) or dist == 0:
        return None  # no se intersectan
    aa = (r0**2 - r1**2 + dist**2) / (2 * dist)
    hh2 = r0**2 - aa**2
    if hh2 < 0:
        return None
    hh = np.sqrt(hh2)
    pm = c0 + aa * dvec / dist
    perp = np.array([-dvec[1], dvec[0]]) / dist
    return pm + hh * perp, pm - hh * perp


# ==============================================================================
# --- 5. ALGORITMO GENETICO DE CODIFICACION REAL ---
# ==============================================================================
class ResultadoOptim:
    """Contenedor del resultado, compatible con scipy (tiene .x y .fun)."""

    def __init__(self, x, fun, nit, nfev, historial):
        self.x = np.asarray(x)
        self.fun = float(fun)
        self.nit = int(nit)
        self.nfev = int(nfev)
        self.historial = list(historial)   # mejor fitness por generacion

    def __repr__(self):
        return (f"ResultadoOptim(fun={self.fun:.6f}, nit={self.nit}, "
                f"nfev={self.nfev})")


def _sbx_vectorizado(padres1, padres2, lower, upper, eta_c, rng):
    """Cruce SBX vectorizado sobre lotes de parejas.
    Ojo: padres1, padres2 son (m, dim). Regresa dos arrays de hijos (m, dim).
    """
    m, dim = padres1.shape
    h1 = padres1.copy()
    h2 = padres2.copy()

    # Genes donde se aplica el cruce (50%) y donde los padres son distintos
    cruzar = (rng.random((m, dim)) <= 0.5) & (np.abs(padres1 - padres2) > 1e-14)

    x1 = np.minimum(padres1, padres2)
    x2 = np.maximum(padres1, padres2)
    dx = np.where(x2 - x1 > 1e-14, x2 - x1, 1e-14)
    u = rng.random((m, dim))
    pexp = 1.0 / (eta_c + 1.0)

    # Hijo 1 (lado inferior)
    beta1 = 1.0 + 2.0 * (x1 - lower) / dx
    alpha1 = 2.0 - beta1 ** (-(eta_c + 1.0))
    betaq1 = np.where(u <= 1.0 / alpha1,
                      (u * alpha1) ** pexp,
                      (1.0 / (2.0 - u * alpha1)) ** pexp)
    c1 = 0.5 * ((x1 + x2) - betaq1 * dx)

    # Hijo 2 (lado superior)
    beta2 = 1.0 + 2.0 * (upper - x2) / dx
    alpha2 = 2.0 - beta2 ** (-(eta_c + 1.0))
    betaq2 = np.where(u <= 1.0 / alpha2,
                      (u * alpha2) ** pexp,
                      (1.0 / (2.0 - u * alpha2)) ** pexp)
    c2 = 0.5 * ((x1 + x2) + betaq2 * dx)

    c1 = np.clip(c1, lower, upper)
    c2 = np.clip(c2, lower, upper)
    h1 = np.where(cruzar, c1, h1)
    h2 = np.where(cruzar, c2, h2)
    return h1, h2


def _mutacion_polinomial_vectorizada(pop, lower, upper, span, eta_m, mutpb, rng):
    """Mutacion polinomial vectorizada sobre toda la poblacion (m, dim)."""
    y = pop.copy()
    mutar = rng.random(pop.shape) <= mutpb
    span_safe = np.where(span > 0, span, 1.0)

    delta1 = (y - lower) / span_safe
    delta2 = (upper - y) / span_safe
    u = rng.random(pop.shape)
    mut_pow = 1.0 / (eta_m + 1.0)

    val_low = 2.0 * u + (1.0 - 2.0 * u) * ((1.0 - delta1) ** (eta_m + 1.0))
    deltaq_low = val_low ** mut_pow - 1.0
    val_high = 2.0 * (1.0 - u) + 2.0 * (u - 0.5) * ((1.0 - delta2) ** (eta_m + 1.0))
    deltaq_high = 1.0 - val_high ** mut_pow
    deltaq = np.where(u < 0.5, deltaq_low, deltaq_high)

    mutado = np.clip(y + deltaq * span, lower, upper)
    return np.where(mutar, mutado, y)


def algoritmo_genetico(func, bounds, popsize=40, ngen=300,
                       cxpb=0.9, mutpb=0.15, eta_c=15.0, eta_m=20.0,
                       tournsize=3, n_elite=2, seed=None, disp=False,
                       tol=1e-8, paciencia=60, callback=None):
    """Algoritmo genetico de codificacion real, vectorizado.

    Basicamente es un GA clasico con representacion real:
      - Seleccion por torneo (tournsize individuos compiten).
      - Cruce SBX (Simulated Binary Crossover, indice eta_c).
      - Mutacion polinomial (indice eta_m).
      - Elitismo (los n_elite mejores pasan directo a la siguiente gen).

    La interfaz es analoga a differential_evolution: le pasas func y bounds y
    te regresa un objeto con .x y .fun. Esto es pa que la comparacion sea
    directa.

    Parametros principales:
      popsize   - tamano de la poblacion
      ngen      - generaciones maximas
      cxpb      - prob de cruce por pareja
      mutpb     - prob de mutacion por gen
      eta_c     - indice del SBX (mas grande = hijos mas pegados a los padres)
      eta_m     - indice de la mutacion polinomial (mas grande = perturbacion menor)
      tournsize - cuantos compiten en el torneo
      n_elite   - cuantos pasan directo (elitismo)
      seed      - semilla pa reproducibilidad
      paciencia - gens sin mejora antes del early stop
      callback  - funcion opcional callback(gen, mejor_x, mejor_fun)
    """
    rng = np.random.default_rng(seed)
    bounds = np.asarray(bounds, dtype=float)
    lower = bounds[:, 0]
    upper = bounds[:, 1]
    dim = len(bounds)
    span = upper - lower

    # Poblacion inicial uniforme dentro de los limites
    pop = lower + rng.random((popsize, dim)) * span
    fitness = np.array([func(ind) for ind in pop])
    nfev = popsize

    mejor_idx = int(np.argmin(fitness))
    mejor_x = pop[mejor_idx].copy()
    mejor_fun = float(fitness[mejor_idx])
    historial = [mejor_fun]
    sin_mejora = 0

    def _seleccion_torneo(n):
        """Selecciona n individuos por torneo (vectorizado)."""
        aspirantes = rng.integers(0, popsize, size=(n, tournsize))
        fit_asp = fitness[aspirantes]
        ganadores = aspirantes[np.arange(n), np.argmin(fit_asp, axis=1)]
        return pop[ganadores]

    n_hijos = popsize - n_elite
    n_parejas = (n_hijos + 1) // 2
    gen = 0

    for gen in range(1, ngen + 1):
        # Elitismo: los mejores pasan intactos
        orden = np.argsort(fitness)
        elite = pop[orden[:n_elite]].copy()

        # Seleccion por torneo de padres
        padres1 = _seleccion_torneo(n_parejas)
        padres2 = _seleccion_torneo(n_parejas)

        # Cruce SBX (con probabilidad cxpb por pareja)
        h1, h2 = _sbx_vectorizado(padres1, padres2, lower, upper, eta_c, rng)
        sin_cruce = rng.random(n_parejas) > cxpb
        h1[sin_cruce] = padres1[sin_cruce]
        h2[sin_cruce] = padres2[sin_cruce]

        hijos = np.vstack([h1, h2])[:n_hijos]

        # Mutacion polinomial
        hijos = _mutacion_polinomial_vectorizada(hijos, lower, upper, span,
                                                 eta_m, mutpb, rng)

        # Nueva poblacion = elite + hijos
        pop = np.vstack([elite, hijos])
        fitness = np.concatenate([
            fitness[orden[:n_elite]],                       # elite ya evaluado
            np.array([func(ind) for ind in hijos])
        ])
        nfev += n_hijos

        gen_idx = int(np.argmin(fitness))
        gen_fun = float(fitness[gen_idx])
        if gen_fun < mejor_fun - tol:
            mejor_fun = gen_fun
            mejor_x = pop[gen_idx].copy()
            sin_mejora = 0
        else:
            sin_mejora += 1

        historial.append(mejor_fun)
        if callback is not None:
            callback(gen, mejor_x, mejor_fun)
        if disp and (gen % 20 == 0 or gen == 1):
            print(f"   [GA] gen {gen:4d}/{ngen}  mejor fitness = {mejor_fun:.6f}")

        # Si ya no mejora, paramos
        if sin_mejora >= paciencia:
            if disp:
                print(f"   [GA] ya convergio (sin mejora en {paciencia} gen) "
                      f"en la generacion {gen}")
            break

    return ResultadoOptim(mejor_x, mejor_fun, gen, nfev, historial)
