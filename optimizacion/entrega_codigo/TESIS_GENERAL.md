# Documento general de la tesis

Sintesis dimensional cinematica de un exoesqueleto de rehabilitacion de dedo.

Este documento resume de que trata la tesis, cual es el objetivo, en que se
basa y que falta. Acompana a la entrega de codigo organizada en tres carpetas
descargables dentro de esta misma ubicacion (`entrega_codigo/`).

## De que trata la tesis

La tesis aborda la **sintesis dimensional cinematica** de un exoesqueleto de
rehabilitacion para un dedo de la mano. El objetivo del diseno es que el
exoesqueleto reproduzca el **agarre de pinza fina** que se midio por captura de
movimiento (mocap) en un dedo humano real. Es decir, dado un patron de
movimiento humano (la trayectoria de las falanges durante la pinza fina), se
buscan las dimensiones de los eslabones del mecanismo del exoesqueleto para que
sus falanges sigan ese mismo movimiento.

El problema se plantea como una optimizacion: se define una funcion objetivo
que mide cuanto se parece la trayectoria del exoesqueleto a la del mocap, y se
buscan los parametros dimensionales que la minimizan.

Se trabajan dos versiones del modelo:

- **Modelo simplificado (parcial), 16 parametros:** cubre la cinematica hasta la
  falange medial (articulaciones MCP e IFP). Es el banco de pruebas para
  comparar metaheuristicas de forma justa.
- **Modelo completo, 21 parametros:** cubre las tres falanges (proximal, medial
  y distal) con cinco mecanismos. Extiende el modelo simplificado para incluir
  la falange distal.

## Objetivo

El objetivo metodologico es **comparar dos metaheuristicas de forma justa y
simetrica**:

- Evolucion Diferencial (ED)
- Algoritmo Genetico (GA)

La comparacion es justa porque ambos algoritmos:

- Minimizan **la misma funcion objetivo** con los mismos pesos.
- Usan **los mismos limites** (bounds) de los parametros.
- **Sintonizan sus hiperparametros con Optuna** (muestreador TPE, 25 trials,
  semilla 42), de modo que ninguno queda en desventaja por una mala eleccion
  manual de hiperparametros.

Asi, la diferencia de resultados mide el ALGORITMO y no la formulacion del
problema. El ganador de la comparacion en el modelo simplificado (ED, con error
global 2.17 mm frente a 2.55 mm del GA) es la metaheuristica que despues se usa
para optimizar el **modelo completo**.

## En que se basa

- **Datos de mocap de pinza fina (120 puntos):** la trayectoria objetivo que el
  exoesqueleto debe reproducir (`mocap_pinza_fina_120pts.csv`).
- **Distancia de Chamfer** como metrica de forma: compara la nube de puntos de
  la trayectoria del exoesqueleto con la del mocap, ponderada por articulacion.
- **Alineacion rigida optima tipo Kabsch/Procrustes:** antes de medir el error
  se alinea rigidamente la trayectoria simulada con la del mocap, de modo que se
  compara la FORMA del movimiento y no su posicion absoluta.
- **Penalizacion de monotonicidad (W_MONO = 5.0):** obliga al movimiento a
  avanzar de forma coherente, sin retrocesos artificiales.
- **Marco de sintesis de mecanismos de 5 y 4 barras:** las etapas del
  exoesqueleto se modelan como combinaciones de mecanismos de cinco y cuatro
  barras, resueltos con los solucionadores de `comun.py` (`sol_5_barras`,
  `solve_four_bar`, `circle_intersections`).
- **Restricciones cinematicas + penalizacion fija de 1000:** los disenos que no
  ensamblan o violan restricciones fisiologicas reciben una penalizacion alta.
- **El articulo COMROB** como soporte y marco de publicacion de los resultados.

Detalle de la comparacion completa vs parcial (funcion objetivo, monotonicidad,
Optuna y restricciones) en el reporte
[`AUDITORIA_completa_vs_parcial.md`](AUDITORIA_completa_vs_parcial.md).

## Tercera etapa para la falange distal

Al modelo original se le **agrego una TERCERA ETAPA de mecanismo** para dar
movimiento a la **falange distal respecto de la proximal** (segun la indicacion
del usuario), y esa tercera etapa se **incorporo al analisis cinematico** (queda
dentro del barrido y de la funcion objetivo del modelo completo).

> **Aclaracion anatomica (no proviene del usuario):** en la anatomia del dedo, la
> falange distal articula directamente con la falange media (o medial) a traves
> de la articulacion interfalangica distal (DIP/IFD). Se deja constancia de esta
> precision como aclaracion separada, sin sustituir la frase del usuario
> ("respecto a la proximal"). Confirmar con el usuario la referencia anatomica que
> debe quedar en la redaccion final de la tesis.

En el codigo (`modelo_completo.py`, seccion "TERCER MECANISMO DE 4 BARRAS") esta
tercera etapa se implementa como un mecanismo de **cuatro barras**:

- Acoplador `Link9_3`.
- Balancin `Link10_3`.
- Soporte dorsal definido por `back3_3` (retroceso a lo largo del eje de la
  falange) y `up3_3` (standoff dorsal), que fija el punto de anclaje.
- El punto `D3` se obtiene por **interseccion de circulos**
  (`circle_intersections`), y de ahi se calcula el angulo de la falange distal.

Incluye ademas el tope fisiologico `DIP_MAX_DEG = 35` grados y un termino de
perfil angular DIP en la funcion objetivo.

## Que falta

- **Construir el CAD** del exoesqueleto con los 21 parametros optimizados del
  modelo completo.
- **Verificar el agarre** del CAD contra las trayectorias alineadas del mocap.
- **Ajustar por interferencias mecanicas** (colisiones y choques entre eslabones
  que la cinematica idealizada no captura).

## Estructura de la entrega

- `01_optimizacion_parcial_es/`: optimizacion parcial (modelo simplificado, 16
  parametros) en espanol, con comparacion ED vs GA, figuras y resultados.
- `02_optimizacion_completa_es/`: optimizacion completa (modelo de 3 falanges,
  21 parametros) en espanol, con la via canonica (`optimizar_completo_ED.py`) y
  la via legacy de referencia del tercer mecanismo.
- `03_optimizacion_parcial_en/`: version en ingles de la optimizacion parcial.
- `AUDITORIA_completa_vs_parcial.md`: reporte de auditoria que verifica que la
  optimizacion completa incorpora las mismas mejoras que la parcial.

Cada carpeta es autocontenida (incluye su CSV de mocap y sus scripts) y trae su
propio documento de texto (`LEEME.txt` / `README.txt`) con el detalle de
scripts, dependencias exactas, orden de ejecucion y resultados esperados.
