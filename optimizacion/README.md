# Optimizacion del exo - Algoritmos Geneticos vs Evolucion Diferencial

Codigo de soporte pa el articulo COMROB: la comparacion de tecnicas de
optimizacion (GA vs ED) aplicada a la sintesis cinematica del exo de
rehabilitacion de dedo.

El exo tiene **tres falanges** (proximal, medial, distal) movidas por
**cinco mecanismos**:

| Etapa | Mecanismo | Articulacion / Falange |
|-------|-----------|------------------------|
| 1 | 5 barras + 4 barras | MCP - falange proximal |
| 2 | 5 barras + 4 barras | IFP - falange medial |
| 3 | 4 barras (tercer mecanismo) | IFD/DIP - falange distal |

Lo que se busca es encontrar las dimensiones de los eslabones que hagan que
las trayectorias del exo reproduzcan el agarre de pinza fina medido con
captura de movimiento (`mocap_pinza_fina_120pts.csv`).

---

## Estructura del proyecto

```
optimizacion/
├── mocap_pinza_fina_120pts.csv     # datos de mocap (120 puntos)
├── comun.py                        # utilidades + el ALGORITMO GENETICO
│
├── modelo_simplificado.py          # cinematica hasta la falange medial (16 params)
├── optimizar_simplificado_ED.py    # -> Evolucion Diferencial   (contendiente 1)
├── optimizar_simplificado_GA.py    # -> Algoritmo Genetico      (contendiente 2)
├── comparar_ED_GA.py               # tabla + curvas de convergencia ED vs GA
│
├── modelo_completo.py              # cinematica COMPLETA, 3 falanges (21 params)
├── optimizar_completo_ED.py        # -> Evolucion Diferencial (modelo completo)
│
├── exo_18_pinza_fina.py            # CODIGO ORIGINAL de referencia (no tocar)
├── optimizar_tercer_mecanismo.py   # runner original de referencia (no tocar)
└── resultados/                     # salidas generadas (parametros, csv, png)
```

---

## Como va el estudio

1. **Modelo simplificado (pa validar el algoritmo).** Le recortamos el
   problema hasta la trayectoria de la falange medial: se elimina la falange
   distal y todo el tercer mecanismo. El vector de diseno baja de 21 a **16
   parametros** y los objetivos quedan en dos trayectorias (IFP e IFD). Sobre
   este problema mas chiquito comparamos los dos algoritmos con la **misma
   funcion objetivo, mismos limites y misma semilla (42)**:
   - `optimizar_simplificado_ED.py` - Evolucion Diferencial (scipy).
   - `optimizar_simplificado_GA.py` - Algoritmo Genetico (codificacion real:
     torneo + SBX + mutacion polinomial + elitismo).

   Para que la comparacion sea **justa**, los hiperparametros de *ambos*
   algoritmos se sintonizan automaticamente con **Optuna** (muestreador TPE,
   25 ensayos cada uno). Asi ninguno de los dos queda en desventaja por una
   eleccion manual de parametros.

2. **Seleccion del algoritmo.** El que gane se usa pa el modelo completo y
   pa construir un modelo CAD de validacion.

3. **Modelo completo.** Con el ganador (Evolucion Diferencial) se regenera el
   codigo de las **tres falanges y los cinco mecanismos**
   (`optimizar_completo_ED.py`), ya con las mejoras validadas.

### Mejoras que le metimos
- **Dimensiones reducidas**: eslabones acotados a max 60 mm (antes 80 mm) y
  los del tercer mecanismo a max 45 mm. Esto es pa que el mecanismo sea mas
  compacto y fabricable.
- **Angulo auxiliar medial-c2 (`theta_aux_fm`) reducido**: limite superior
  bajado de 180 a 70 deg, con regularizacion suave que favorece valores chicos
  **sin echar a perder el ajuste**.
- **Validacion cinematica** (`validar_cinematica`): checa que los parametros
  son viables y que el mecanismo ensambla sin NaN ni retrocesos en **todo** el
  barrido de la manivela.

---

## Resultados (modelo simplificado, semilla 42, hiperparametros sintonizados con Optuna)

| Metodo | Error global | Error IFP | Error IFD | Evaluaciones | Tiempo aprox. |
|--------|-------------:|----------:|----------:|-------------:|--------------:|
| **Evolucion Diferencial** | **2.17 mm** | 2.82 mm | 1.53 mm | 135 970 | ~70-90 s |
| Algoritmo Genetico | 2.55 mm | 3.68 mm | 1.43 mm | 15 052 | ~55-65 s |

Con **ambos** algoritmos sintonizados por Optuna (misma funcion objetivo, mismos
limites y misma semilla 42), la **Evolucion Diferencial** obtiene el menor error
global y ademas un mecanismo mas compacto (suma de eslabones 347.2 mm vs 457.1 mm
del GA). El GA queda competitivo e incluso mejora ligeramente en la IFD
(1.43 vs 1.53 mm), pero a costa de un mecanismo mas grande. Por eso adoptamos ED
pa el modelo completo.

**Modelo completo (ED, semilla 100):** error global **3.15 mm**
(IFP 4.01 / IFD 1.80 / Punta 3.64 mm), ROM DIP fisiologico de 30.7 deg y
viabilidad cinematica verificada en las 120 poses.

> El tiempo depende del hardware y del numero de nucleos. Las metricas de error,
> fitness y evaluaciones son reproducibles gracias a las semillas fijas, aunque
> pueden variar ligeramente segun la version de las bibliotecas.

---

## Como reproducir

```bash
pip install numpy scipy pandas matplotlib optuna

# Modelo simplificado: comparacion ED vs GA
python3 optimizar_simplificado_ED.py     # ED + sintonizacion Optuna
python3 optimizar_simplificado_GA.py     # GA + sintonizacion Optuna
python3 comparar_ED_GA.py                # tabla resumen + curva de convergencia
python3 figuras_articulo.py              # figuras de dimensiones y errores
python3 diagrama_modelo_simplificado.py  # diagrama esquematico del modelo

# Modelo completo (algoritmo ganador: Evolucion Diferencial)
python3 optimizar_completo_ED.py
```

> `optuna` es obligatorio: `optimizar_simplificado_ED.py` y
> `optimizar_simplificado_GA.py` lo usan para sintonizar los hiperparametros.
> El orden importa: `comparar_ED_GA.py` y `figuras_articulo.py` leen los `.txt`
> y `.csv` que generan los dos optimizadores.

Cada script simplificado hace dos etapas: (1) **Optuna** prueba
`N_TRIALS_OPTUNA` configuraciones con corridas cortas y se queda con la mejor;
(2) **corrida profunda** con esos hiperparametros y el presupuesto completo.

Presupuesto y semillas configurables por variables de entorno (con sus valores
por defecto, que reproducen los numeros del articulo):

| Script | Variables (valor por defecto) |
|--------|-------------------------------|
| `optimizar_simplificado_ED.py` | `N_TRIALS_OPTUNA` (25), `DE_MAXITER_OPTUNA` (40), `OPT_MAXITER` (300), `OPT_SEED` (42), `OPTUNA_SEED` (42) |
| `optimizar_simplificado_GA.py` | `N_TRIALS_OPTUNA` (25), `GA_NGEN_OPTUNA` (60), `GA_NGEN` (300), `GA_SEED` (42), `OPTUNA_SEED` (42) |
| `optimizar_completo_ED.py` | `OPT_POPSIZE` (20), `OPT_MAXITER` (600), `OPT_SEED` (100) |

Las salidas (parametros, trayectorias alineadas, curvas de convergencia y
graficas de biofidelidad) se escriben en `resultados/`. Las figuras que usa el
articulo (`biofidelidad_simplificado_ED/GA.png`, `comparacion_convergencia.png`,
`comparacion_errores.png`, `diagrama_modelo_simplificado.png/pdf`) deben copiarse
luego a `../articulo_latex/figs/`.
