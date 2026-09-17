# Auditoria: optimizacion COMPLETA vs PARCIAL

Comparacion a fondo del codigo de la optimizacion completa (`modelo_completo.py`,
`optimizar_completo_ED.py`, `optimizar_tercer_mecanismo.py`) contra la
optimizacion parcial (`modelo_simplificado.py`, `optimizar_simplificado_ED.py`,
`optimizar_simplificado_GA.py`), para verificar que la completa incorpora las
mismas mejoras que la parcial en cuatro ejes:

- (a) funcion objetivo Chamfer ponderada por articulacion
- (b) penalizacion de monotonicidad (W_MONO = 5.0)
- (c) sintonizacion de hiperparametros con Optuna/TPE (25 trials, semilla 42)
- (d) restricciones cinematicas + penalizacion fija de 1000

Todas las rutas citadas son relativas a `optimizacion/`. Los numeros de linea
corresponden al estado del codigo auditado en la rama `feat/optimizacion-ed-vs-ga`.

## Resumen ejecutivo

La optimizacion completa **ya esta alineada** con la parcial en los cuatro ejes.
La completa reutiliza exactamente el mismo planteamiento (misma estructura de
funcion objetivo, misma penalizacion de monotonicidad, mismo flujo Optuna/TPE de
dos etapas con las mismas semillas y rangos, y el mismo esquema de restricciones
con penalizacion 1000) y lo **extiende** de forma coherente para cubrir la
falange distal: agrega el termino de punta (W_TIP), el perfil angular DIP
(W_DIP), la monotonicidad de la punta, el tercer mecanismo de cuatro barras y el
tope fisiologico DIP.

**No se requirio ninguna correccion funcional.** El codigo es consistente entre
ambas versiones y las diferencias que existen son extensiones deliberadas y
documentadas, no inconsistencias. Se verifico por smoke que la
`fitness_function` de ambos modelos devuelve un float finito sobre sus
parametros de referencia (ver el detalle del vector `p_ref` en la seccion de
verificacion smoke, mas abajo).

Punto que requiere confirmacion del usuario: el usuario describio la tercera
etapa como "3 barras", pero el codigo la implementa de forma consistente como un
mecanismo de **cuatro barras**. Se detalla en la seccion correspondiente. No se
cambio el codigo por esta discrepancia.

## Tabla comparativa por criterio

| Criterio | Parcial (simplificado) | Completa | Veredicto |
|---|---|---|---|
| (a) Funcion objetivo | Chamfer ponderada IFP/IFD: `W_IFP=0.40`, `W_IFD=0.60`; reg suave `W_REG_DIM=W_REG_AUX=5e-4` | Chamfer ponderada IFP/IFD/punta: `W_IFP=0.25`, `W_IFD=0.375`, `W_TIP=0.375`; ademas perfil DIP `W_DIP=0.3`; misma reg suave | Alineado + extendido a la falange distal |
| (b) Monotonicidad | `monotonicity_penalty` con `W_MONO=5.0`, aplicada a IFD | `monotonicity_penalty` con `W_MONO=5.0`, aplicada a IFD y punta | Alineado + extendido |
| (c) Optuna/TPE | 2 etapas, `N_TRIALS_OPTUNA=25`, `OPTUNA_SEED=42`, `OPT_SEED=42`; mismos rangos (strategy, popsize 15-30, mut_low, mut_high, recombination) | Idem, mismos trials/semillas/rangos; unica diferencia deliberada: `polish=False` durante la busqueda corta | Alineado (diferencia documentada) |
| (d) Restricciones + penalizacion | Validaciones cinematicas + penalizacion fija 1000 + `validar_cinematica()` | Mismas validaciones + penalizacion 1000; agrega tercer 4 barras (`circle_intersections`->D3, calibracion `gamma_bracket3`) y tope `DIP_MAX_DEG=35` | Alineado + extendido |

## Hallazgos detallados

### (a) Funcion objetivo

La estructura de la funcion objetivo es la misma en ambos modelos:
Chamfer por articulacion + penalizacion de monotonicidad + regularizacion suave
+ penalizacion fija de 1000 si el mecanismo no ensambla.

Parcial (`modelo_simplificado.py`):

- Pesos: `W_IFP = 0.40`, `W_IFD = 0.60`, `W_MONO = 5.0` (lineas 72-74).
- Regularizacion suave: `W_REG_DIM = 5e-4`, `W_REG_AUX = 5e-4` (lineas 77-78).
- `fitness_function` (lineas 255-277): alinea rigidamente mocap contra sim
  (`optimal_rigid_transform`), calcula `err_ifp` y `err_ifd` con
  `chamfer_distance`, `mono_ifd` con `monotonicity_penalty`, y suma
  `shape_error = W_IFP*err_ifp + W_IFD*err_ifd`, `mono_error = W_MONO*mono_ifd`,
  mas `reg_dim` y `reg_aux`. Penaliza con `1000.0` si `run_kinematics` regresa
  `None` (linea 258).

Completa (`modelo_completo.py`):

- Pesos: `W_IFP = 0.25`, `W_IFD = 0.375`, `W_TIP = 0.375`, `W_MONO = 5.0`,
  `W_DIP = 0.3` (lineas 81-85).
- Misma regularizacion suave: `W_REG_DIM = 5e-4`, `W_REG_AUX = 5e-4`
  (lineas 87-88).
- `fitness_function` (lineas 250-282): misma alineacion rigida; calcula
  `err_ifp`, `err_ifd` y ademas `err_tip` (punta) con `chamfer_distance`;
  `mono_ifd` y `mono_tip`; un termino de error de perfil DIP `dip_error`
  (lineas 270-271) escalado por `W_DIP`; suma
  `shape_error = W_IFP*err_ifp + W_IFD*err_ifd + W_TIP*err_tip`,
  `mono_error = W_MONO*(mono_ifd + mono_tip)`, `dip_pen = W_DIP*dip_error`, mas
  la misma `reg_dim` y `reg_aux`. Penaliza con `1000.0` si no ensambla
  (linea 254).

**Veredicto:** consistente. La completa mantiene el planteamiento de la parcial
y lo extiende con el termino de punta y el perfil DIP para cubrir la falange
distal. Los pesos IFP/IFD/punta suman 1.0 en la completa (0.25 + 0.375 + 0.375)
y IFP/IFD suman 1.0 en la parcial (0.40 + 0.60), de modo que ambas reparten la
misma unidad de peso de forma coherente.

### (b) Penalizacion de monotonicidad

La funcion `monotonicity_penalty` esta definida una sola vez en `comun.py`
(lineas 127-137) y la usan ambos modelos con el mismo peso `W_MONO = 5.0`:

- Parcial: `mono_ifd = monotonicity_penalty(aligned['ifd'])` y
  `mono_error = W_MONO * mono_ifd` (`modelo_simplificado.py`, lineas 267 y 270).
- Completa: `mono_ifd = monotonicity_penalty(aligned['ifd'])`,
  `mono_tip = monotonicity_penalty(aligned['tip'])` y
  `mono_error = W_MONO * (mono_ifd + mono_tip)` (`modelo_completo.py`,
  lineas 265-266 y 274).

Ademas, ambos modelos aplican una restriccion anti-gancho dura sobre el angulo de
la falange medial durante el barrido (`delta_fm < -np.deg2rad(0.5)` -> `None`):
`modelo_simplificado.py` lineas 232-233 (dentro de `run_kinematics`) y
`modelo_completo.py` lineas 187-189.

**Veredicto:** consistente. La completa usa la misma penalizacion y peso, y la
extiende a la punta.

### (c) Sintonizacion con Optuna/TPE

El flujo de dos etapas (Optuna/TPE para sintonizar hiperparametros + corrida
profunda) es identico entre `optimizar_simplificado_ED.py` y
`optimizar_completo_ED.py`.

Presupuesto y semillas (identicos):

- Parcial (`optimizar_simplificado_ED.py`, lineas 45-49):
  `N_TRIALS_OPTUNA=25`, `DE_MAXITER_OPTUNA=40`, `OPT_MAXITER=300`, `OPT_SEED=42`,
  `OPTUNA_SEED=42`.
- Completa (`optimizar_completo_ED.py`, lineas 44-48): `N_TRIALS_OPTUNA=25`,
  `DE_MAXITER_OPTUNA=60`, `OPT_MAXITER=600`, `OPT_SEED=42`, `OPTUNA_SEED=42`.

Los trials y ambas semillas coinciden (25 trials, semilla 42 para el sampler TPE
y para la ED). El presupuesto por trial y el de la corrida profunda son mayores
en la completa (60 vs 40, 600 vs 300), lo cual es esperable por el mayor tamano
del problema (21 vs 16 parametros); no afecta la simetria del metodo.

Rangos de sugerencia de Optuna (identicos): en ambos `objective_optuna` se
sugieren los mismos hiperparametros con los mismos rangos:
`strategy` en `['best1bin','rand1bin','best1exp','currenttobest1bin']`,
`popsize` en `[15, 30]`, `mut_low` en `[0.3, 0.7]`, `mut_high` en `[0.8, 1.2]`,
`recombination` en `[0.5, 0.95]`
(`optimizar_simplificado_ED.py` lineas 73-78; `optimizar_completo_ED.py`
lineas 78-83).

Sampler y estudio (identicos): ambos usan
`optuna.samplers.TPESampler(seed=OPTUNA_SEED)` y
`optuna.create_study(direction='minimize', ...)`
(`optimizar_simplificado_ED.py` lineas 90-91; `optimizar_completo_ED.py`
lineas 95-96).

Diferencia deliberada y documentada: en la completa, la funcion `_correr_ed`
acepta un parametro `polish` (por defecto `True`) y `objective_optuna` lo llama
con `polish=False` durante la busqueda corta (`optimizar_completo_ED.py`
lineas 52-66 y 85-86). El docstring justifica que el pulido local L-BFGS es
costoso con 21 parametros y solo hace falta en la corrida profunda final; la
sintonizacion solo necesita comparar hiperparametros de forma relativa. La
corrida profunda si usa `polish=True` (valor por defecto al llamar `_correr_ed`
sin el argumento, `optimizar_completo_ED.py` linea 139). En la parcial, la
ED usa `polish=True` siempre (`optimizar_simplificado_ED.py` lineas 53-61). Esta
es una optimizacion de costo, no un cambio de metodo, y esta documentada en el
codigo.

Runner legacy `optimizar_tercer_mecanismo.py`: este script NO usa Optuna. Es un
runner heredado que fija los hiperparametros a mano: `strategy='best1bin'`,
`popsize=22`, `mutation=(0.5, 1.0)`, `recombination=0.7`
(`optimizar_tercer_mecanismo.py` lineas 38-40 para los defaults por entorno y
lineas 59-65 para la llamada a `differential_evolution`). Importa el modelo
heredado `exo_18_pinza_fina.py`. La via "alineada con la parcial" para la
optimizacion completa es la pareja `optimizar_completo_ED.py` +
`modelo_completo.py`; `optimizar_tercer_mecanismo.py` + `exo_18_pinza_fina.py`
es el camino legacy que se conserva como referencia historica.

**Veredicto:** consistente. La via canonica de la completa
(`optimizar_completo_ED.py`) usa el mismo Optuna/TPE con los mismos trials,
semillas y rangos que la parcial; la unica diferencia (`polish=False` en la
busqueda) es deliberada y esta documentada. El runner legacy sin Optuna existe
en paralelo y no compite con la via canonica.

### (d) Restricciones cinematicas + penalizacion fija 1000

Ambos modelos comparten el mismo esquema de restricciones dentro de
`run_kinematics` y `validar_cinematica`:

Validaciones basicas de parametros (regresan `None` -> penalizacion 1000):

- `gear_ratio <= 0`: `modelo_simplificado.py` linea 103; `modelo_completo.py`
  linea 104.
- longitudes minimas `min(p[:11]) <= 0.005`: `modelo_simplificado.py` linea 105;
  `modelo_completo.py` linea 106.
- `hsp <= 0 or dsp <= 0`: `modelo_simplificado.py` linea 107;
  `modelo_completo.py` linea 112.
- `FP_REAL - 2.0*dsp <= 0.001`: `modelo_simplificado.py` linea 109;
  `modelo_completo.py` linea 114.

Restriccion anti-gancho de la medial (`delta_fm >= -0.5 deg`):
`modelo_simplificado.py` lineas 231-233; `modelo_completo.py` lineas 187-189.

Chequeos de finitud/rango de la trayectoria de salida (`abs > 0.3` o no finito
-> `None`): `modelo_simplificado.py` lineas 237-239 (IFD);
`modelo_completo.py` lineas 224-226 (punta).

`validar_cinematica()` sobre todo el barrido: comprueba no-ensamble, no
finitud, que se resuelvan las `N_PUNTOS` poses y la monotonicidad de `theta_fm`
(`modelo_simplificado.py` lineas 308-353; `modelo_completo.py` lineas 318-358).

Penalizacion fija de 1000: `return 1000.0` cuando `run_kinematics` regresa
`None` en `fitness_function` de ambos (`modelo_simplificado.py` linea 258;
`modelo_completo.py` linea 254).

Extensiones de la completa (falange distal):

- Validaciones extra del tercer mecanismo: `Link9_3 <= 0.005 or Link10_3 <= 0.005`
  y `up3_3 <= 0` (`modelo_completo.py` lineas 108-110).
- Tercer mecanismo de cuatro barras: calcula el punto de anclaje `Pa`, resuelve
  `D3` por interseccion de circulos (`circle_intersections`) y calibra
  `gamma_bracket3` en la primera pose ensamblable (`modelo_completo.py`
  lineas 208-218). Si los circulos no intersectan, regresa `None`
  (linea 211), lo que dispara la penalizacion 1000.
- Tope fisiologico DIP: `DIP_MAX_DEG = 35.0` (linea 35) y la restriccion
  `if np.ptp(dip_rel) > np.deg2rad(DIP_MAX_DEG): return None`
  (lineas 234-236), que tambien deriva en penalizacion 1000.

**Veredicto:** consistente. La completa hereda todas las restricciones de la
parcial (con la misma penalizacion fija 1000) y agrega las propias del tercer
mecanismo y del tope DIP.

## Discrepancia destacada: tercera etapa "3 barras" (usuario) vs "4 barras" (codigo)

El usuario indico que al modelo original se le agrego una "tercera etapa de 3
barras" para dar movimiento a la falange distal respecto a la proximal, y que se
incorporo al analisis cinematico.

El codigo, de forma **consistente**, implementa esa tercera etapa como un
mecanismo de **cuatro barras**, no de tres:

- `modelo_completo.py` linea 9: comentario de encabezado
  `ETAPA 3 (4 barras) -> falange distal (articulacion IFD/DIP)`.
- `modelo_completo.py` linea 22: `Vector de diseno: 21 parametros (17 del
  mecanismo base + 4 del tercer 4 barras)`.
- `modelo_completo.py` linea 73: comentario
  `# --- Tercer mecanismo de 4 barras (cinematica IFD/DIP) ---`.
- La cinematica del tercer mecanismo usa cuatro parametros dimensionales
  (`Link9_3` acoplador, `Link10_3` balancin, `back3_3` soporte, `up3_3`
  standoff dorsal; ver bounds en `modelo_completo.py` lineas 74-77 y NOMBRES en
  lineas 55-56) y resuelve el lazo con `circle_intersections`
  (`modelo_completo.py` lineas 209-218).
- `optimizar_completo_ED.py` lineas 114 y 193 tambien lo describen como
  "tercer 4 barras".
- `optimizar_tercer_mecanismo.py` lineas 5, 13 y 156 lo describen como
  "4 barras" e "Indices ... tercer mecanismo de 4 barras IFD/DIP".
- `exo_18_pinza_fina.py` (modelo legacy) lo describe igual: lineas 195, 221,
  487, 618 y 670 ("tercer mecanismo de 4 barras"), con la misma
  parametrizacion `Link9_3`/`Link10_3`/`back3_3`/`up3_3` y el solver
  `_circle_intersections` (definido en linea 192, usado en linea 372) y la
  calibracion `gamma_bracket3` (lineas 380-383).

**Punto a confirmar por el usuario:** existe una diferencia de nomenclatura o de
conteo entre lo que el usuario describe (3 barras) y lo que el codigo implementa
(4 barras). Posibles lecturas, a confirmar por el usuario:

- Que el usuario cuente solo los eslabones moviles nuevos y no la bancada/soporte
  (un cuatro barras clasico tiene un eslabon fijo mas tres moviles, por lo que
  "3 barras moviles" y "4 barras" pueden referirse al mismo mecanismo).
- Que el mecanismo fisico real sea de tres barras y el codigo lo modele como un
  cuatro barras equivalente.

Esta auditoria documenta el hecho tal como esta en el codigo (cuatro barras) y
**no modifica el codigo** por esta discrepancia. Se recomienda que el usuario
confirme la nomenclatura deseada antes de fijar la redaccion en la tesis/articulo.

## Correcciones aplicadas

**Ninguna.** No se aplico ningun cambio funcional al codigo.

Justificacion con evidencia:

- (a) Funcion objetivo: los pesos y la estructura son consistentes
  (`modelo_simplificado.py` lineas 72-78 y 255-277; `modelo_completo.py`
  lineas 81-88 y 250-282). La completa extiende, no contradice.
- (b) Monotonicidad: misma funcion y peso `W_MONO=5.0`
  (`comun.py` lineas 127-137; `modelo_simplificado.py` lineas 267-270;
  `modelo_completo.py` lineas 265-274).
- (c) Optuna: mismos trials (25), mismas semillas (42, 42) y mismos rangos de
  sugerencia (`optimizar_simplificado_ED.py` lineas 45-49 y 73-91;
  `optimizar_completo_ED.py` lineas 44-48 y 78-96); la unica diferencia
  (`polish=False` en la busqueda) es deliberada y esta documentada en el
  docstring de `_correr_ed`.
- (d) Restricciones: mismo conjunto de validaciones y misma penalizacion 1000
  (referencias de linea en la seccion (d)).

Verificacion smoke (sin excepcion, float finito) sobre los parametros de
referencia, ejecutada tras la auditoria.

Modelo completo: se usa el vector `p_ref` del bloque `main` de
`optimizar_completo_ED.py` (lineas 117-120), es decir

```
p_ref = [0.018, 0.020, 0.035, 0.049, 0.025, 0.020, 0.025,
         0.055, 0.035, 0.052, 0.04601, 0.017, 0.018,
         np.deg2rad(51.39), np.deg2rad(38.78), 2.0, np.deg2rad(109),
         0.025, 0.035, -0.012, 0.002]
```

Con ese vector, `modelo_completo.fitness_function(p_ref)` = 0.036594 (float
finito). El valor es reproducible con solo copiar ese `p_ref`.

Modelo simplificado: `modelo_simplificado.fitness_function(p)` devuelve un float
finito (sin excepcion) para un vector de 16 parametros dentro de los bounds
declarados en `modelo_simplificado.py`; el valor numerico exacto depende del
vector elegido y no se fija aqui. Lo verificable y reproducible es que la
funcion devuelve un float finito. Un vector de prueba valido es el que se usa en
el smoke de la entrega (FEAT-001):
`[0.02, 0.04, 0.03, 0.05, 0.025, 0.02, 0.03, 0.05, 0.035, 0.05, 0.033, 0.017, 0.008, 0.7, 3.0, -1.4]`.

Como no se toco ningun `.py`, la `fitness_function` de ambos modelos sigue
devolviendo un float finito.

## Recomendaciones

1. Confirmar con el usuario la nomenclatura de la tercera etapa (3 barras vs 4
   barras). El codigo es internamente consistente con "4 barras"; si la tesis
   debe decir "3 barras", aclarar si se refiere a eslabones moviles o ajustar la
   redaccion. No cambiar el codigo hasta tener esa confirmacion.
2. Mantener `optimizar_completo_ED.py` como la via canonica de la optimizacion
   completa (es la alineada con la parcial via Optuna). Documentar en la entrega
   que `optimizar_tercer_mecanismo.py` + `exo_18_pinza_fina.py` es el camino
   legacy con hiperparametros fijos, conservado como referencia.
3. Dejar constancia en la redaccion de la tesis/articulo de la diferencia
   deliberada `polish=False` durante la sintonizacion de la completa, para que
   la comparacion metodologica quede documentada.
4. Si en algun momento se desea simetria total del pulido, se podria activar
   `polish=True` tambien en la busqueda de la completa, pero no es necesario para
   la validez de la comparacion y encarece la sintonizacion.
