# Contexto completo — Optimización del exoesqueleto con tercer mecanismo de 4 barras (IFD/DIP)

## 1. Objetivo

Integrar el **tercer mecanismo de 4 barras** (que controla la flexión de la
articulación interfalángica distal, IFD/DIP) dentro del flujo de **optimización**
del exoesqueleto de rehabilitación de dedo, obtener los **parámetros optimizados**
que reproducen el agarre de pinza fina medido por captura de movimiento (mocap),
y dejar todo listo para **construir el CAD** y verificar que replica el agarre.

Rama de trabajo: `feat/optimizacion-tercer-mecanismo` (basada en
`feat/tercer-mecanismo-dip`).

---

## 2. Problema identificado

En el optimizador original (`exo_18_pinza_fina.py`) la falange distal se calculaba con
un offset **constante**:

```
theta_fd = theta_fm + theta_aux_fd   # offset fijo  ->  IFD rígida
```

Esto hacía que la falange distal se moviera como cuerpo rígido junto con la falange
medial, **sin flexión relativa real** en la IFD. El tercer mecanismo de 4 barras
(`cinematica_tercer_mecanismo.py`, eslabones `Link9`/`Link10`, soporte `Pa` con
desplazamientos `back3`/`up3`, montaje **dorsal**) existía pero **no estaba integrado**
en la optimización.

---

## 3. Solución implementada

### 3.1 Cambios en `exo_18_pinza_fina.py`
1. **Imports y configuración por variables de entorno**: `matplotlib.use('Agg')`,
   `import os`, y parámetros configurables: `N_TRIALS_OPTUNA` (def 8),
   `DE_MAXITER` (def 1500), `DE_MAXITER_OPTUNA` (def 20), `DIP_MAX_DEG` (def 35.0).
2. **`run_kinematics`** ahora desempaqueta **21 parámetros** (17 originales + 4 nuevos:
   `Link9_3`, `Link10_3`, `back3_3`, `up3_3`), con validaciones.
3. Nuevo helper `_circle_intersections()` (intersección de dos círculos).
4. **Núcleo cinemático**: se reemplazó el offset constante por el tercer mecanismo de
   4 barras real. Calcula el soporte `Pa`, el punto `D3` por intersección de círculos,
   el ángulo del balancín (`ang_rocker`) y **calibra `gamma_bracket3`** en el primer
   paso para conservar la flexión natural inicial:
   `theta_fd = ang_rocker + gamma_bracket3`.
5. **Tope físico**: si el rango de la IFD relativa supera `DIP_MAX_DEG`, se descarta la
   solución (devuelve `None`).
6. **`fitness_function`**: se añadió el término `W_DIP` (def 0.3) que ata el perfil
   angular de la IFD del exo al perfil DIP real del mocap:
   `dip_error = FD_REAL * mean(abs(exo_dip_rel - dip_rel_mocap))`.
7. **bounds**: 4 límites nuevos — `Link9_3` 0.010–0.050, `Link10_3` 0.010–0.050,
   `back3_3` −0.030–0.010, `up3_3` 0.001–0.015 (metros).
8. Reporte y `savetxt` ampliados a **21 parámetros**, con ROM de la IFD.

### 3.2 Runner paralelo nuevo: `optimizar_tercer_mecanismo.py`
- Importa `exo_18_pinza_fina` y corre `scipy.differential_evolution` con `workers=-1`
  (8 núcleos, `updating='deferred'`).
- Variables de entorno: `OPT_POPSIZE` (def 22), `OPT_MAXITER` (def 1000),
  `OPT_SEED` (def 42).
- Reporta ROM EXO vs ROM MOCAP (~22.19°).

### 3.3 Calibración del balance de objetivos
- `W_DIP=2.0` → DIP perfecto pero error global alto.
- `W_DIP=0` → error 2.27 mm pero DIP irreal (~90°).
- **Solución final**: tope duro DIP ≤ 35° + `W_DIP=0.3`.
- Probadas semillas 1, 11, 42, 100.
- **Mejor corrida: `SEED=100`, `maxiter=1500`, `popsize=25`** → error global **2.926 mm**,
  ROM IFD 29.13° (mocap 22.19°), fitness 0.003480, ~478 s.

---

## 4. Parámetros optimizados finales (SEED=100)

Todas las longitudes en **metros**; los ángulos en **radianes**.

| Idx | Nombre | Valor | Notas |
|----:|--------|-------|-------|
| 0  | Bancada1     | 0.018983 | mecanismo original |
| 1  | Bancada2     | 0.049628 | |
| 2  | Link1        | 0.056832 | |
| 3  | Link2        | 0.079029 | |
| 4  | Link3        | 0.032447 | |
| 5  | Link4        | 0.023598 | |
| 6  | Link5        | 0.043246 | |
| 7  | Link6        | 0.031388 | |
| 8  | Link7        | 0.051771 | |
| 9  | Link8        | 0.066191 | |
| 10 | Link10/c2    | 0.024191 | |
| 11 | hsp          | 0.027608 | |
| 12 | dsp          | 0.008922 | |
| 13 | Theta_Aux_FM | 1.743746 rad | |
| 14 | Theta_Aux_FD | 0.931951 rad | |
| 15 | gear_ratio   | 1.524732 | |
| 16 | theta_offset | −1.572638 rad | |
| **17** | **Link9_3**  | **0.035559 m (35.56 mm)** | tercer mecanismo |
| **18** | **Link10_3** | **0.013870 m (13.87 mm)** | tercer mecanismo |
| **19** | **back3_3**  | **−0.024022 m (−24.02 mm)** | soporte dorsal |
| **20** | **up3_3**    | **0.012201 m (12.20 mm)** | soporte dorsal |

### Errores de biofidelidad (exo vs mocap)
- **Global (RMS): 2.926 mm**
- IFP: 3.770 mm
- IFD: 1.828 mm
- Punta: 3.179 mm
- ROM IFD EXO: **29.13°** vs MOCAP: **22.19°** (flexión distal gradual y fisiológica)

---

## 5. Archivos del entregable

| Archivo | Descripción |
|---------|-------------|
| `exo_18_pinza_fina.py` | Optimizador modificado (17→21 parámetros, tercer 4-barras integrado) |
| `optimizar_tercer_mecanismo.py` | Runner paralelo (differential_evolution, workers=-1) |
| `cinematica_tercer_mecanismo.py` | Cinemática de referencia del tercer mecanismo |
| `mocap_pinza_fina_120pts.csv` | Datos de captura de movimiento (120 pts) |
| `Parametros_Optimizados_Pinza_Fina.txt` | 21 parámetros optimizados |
| `Resultados_Alineados_Pinza_Fina.csv` | Trayectorias exo alineadas vs mocap |
| `Resultados_Biofidelidad_Pinza_Fina.png` | Gráfica MOCAP (punteado) vs EXO (sólido) |

---

## 6. Próximos pasos

1. **Revisar** la gráfica de biofidelidad y los parámetros en el PR.
2. **Construir el CAD** (SolidWorks) usando los 21 parámetros optimizados — en
   particular los del tercer mecanismo dorsal: `Link9_3=35.56 mm`,
   `Link10_3=13.87 mm`, `back3_3=−24.02 mm`, `up3_3=12.20 mm`.
3. **Verificar el agarre** en el CAD y comparar contra `Resultados_Alineados_Pinza_Fina.csv`.
4. Ajustar dimensiones si aparecen interferencias mecánicas.

---

## 7. Notas técnicas
- Montaje **dorsal** (mismo lado que el resto del exo, para no chocar con la palma).
- La IFD entrega flexión **gradual y fisiológica** (~22–30°), no 90°.
- Cómo reproducir la mejor corrida:
  ```bash
  OPT_SEED=100 OPT_MAXITER=1500 OPT_POPSIZE=25 python3 optimizar_tercer_mecanismo.py
  ```
- Dependencias: `numpy`, `scipy`, `pandas`, `optuna`, `matplotlib`.
