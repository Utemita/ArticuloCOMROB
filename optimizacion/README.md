# Optimización del exoesqueleto — Algoritmos Genéticos vs Evolución Diferencial

Código de soporte para el artículo COMROB **"Comparación de técnicas de
optimización: Algoritmos Genéticos vs Evolución Diferencial"**, aplicado a la
síntesis cinemática de un exoesqueleto de rehabilitación de dedo.

El exoesqueleto consta de **tres falanges** (proximal, medial, distal)
accionadas por **cinco mecanismos**:

| Etapa | Mecanismo | Articulación / Falange |
|-------|-----------|------------------------|
| 1 | 5 barras + 4 barras | MCP — falange proximal |
| 2 | 5 barras + 4 barras | IFP — falange medial |
| 3 | 4 barras (tercer mecanismo) | IFD/DIP — falange distal |

El objetivo de la optimización es hallar las dimensiones de los eslabones que
hacen que las trayectorias del exo reproduzcan el agarre de pinza fina medido
por captura de movimiento (`mocap_pinza_fina_120pts.csv`).

---

## Estructura

```
optimizacion/
├── mocap_pinza_fina_120pts.csv     # datos de captura de movimiento (120 pts)
├── comun.py                        # utilidades + ALGORITMO GENETICO (real-coded)
│
├── modelo_simplificado.py          # cinematica HASTA la falange medial (16 params)
├── optimizar_simplificado_ED.py    # -> Evolucion Diferencial   (contendiente 1)
├── optimizar_simplificado_GA.py    # -> Algoritmo Genetico      (contendiente 2)
├── comparar_ED_GA.py               # tabla + curvas de convergencia ED vs GA
│
├── modelo_completo.py              # cinematica COMPLETA, 3 falanges (21 params)
├── optimizar_completo_ED.py        # -> Evolucion Diferencial (modelo completo)
│
├── exo_18_pinza_fina.py            # CODIGO ORIGINAL de referencia (21 params)
├── optimizar_tercer_mecanismo.py   # runner original de referencia
└── resultados/                     # salidas generadas (parametros, csv, png)
```

---

## Flujo del estudio

1. **Modelo simplificado (validación del algoritmo).** Se recorta el problema
   *hasta la trayectoria de la falange medial*: se elimina la falange distal y
   todo el tercer mecanismo de 4 barras. El vector de diseño pasa de 21 a **16
   parámetros** y los objetivos quedan en dos trayectorias (IFP e IFD). Sobre
   este problema más manejable se comparan los dos algoritmos con **idéntica
   función objetivo y límites**:
   - `optimizar_simplificado_ED.py` → Evolución Diferencial (`scipy`).
   - `optimizar_simplificado_GA.py` → Algoritmo Genético (codificación real:
     selección por torneo + cruce SBX + mutación polinomial + elitismo).

2. **Selección del algoritmo.** Se elige el de mejor desempeño para construir un
   modelo CAD sencillo de validación.

3. **Modelo completo.** Con el algoritmo ganador (Evolución Diferencial) se
   regenera el código de las **tres falanges y los cinco mecanismos**
   (`optimizar_completo_ED.py`), incorporando las mejoras validadas.

### Mejoras incorporadas
- **Dimensiones reducidas**: eslabones acotados a ≤ 60 mm (antes 80 mm) y los
  del tercer mecanismo a ≤ 45 mm → mecanismo más compacto y fabricable.
- **Ángulo auxiliar medial–c2 (`theta_aux_fm`) reducido**: límite superior
  bajado de 180° a 70°, con una regularización suave que favorece valores
  pequeños **sin degradar el ajuste**.
- **Validación cinemática** (`validar_cinematica`): comprueba que los
  parámetros son viables y que el mecanismo ensambla, sin `NaN` ni retrocesos,
  a lo largo de **todo** el barrido de la manivela.

---

## Resultados (modelo simplificado, semilla 42)

| Método | Error global | Error IFP | Error IFD | Evaluaciones | Tiempo |
|--------|-------------:|----------:|----------:|-------------:|-------:|
| **Evolución Diferencial** | **2.15 mm** | 2.66 mm | 1.65 mm | 72 597 | ~41 s |
| Algoritmo Genético | 5.55 mm | 3.93 mm | 7.17 mm | 20 470 | ~96 s |

La **Evolución Diferencial** alcanza un error sustancialmente menor y resulta
mucho más **estable** entre semillas, mientras que el Algoritmo Genético
presenta mayor varianza y convergencia prematura. Por ello se adopta la
Evolución Diferencial para el modelo completo.

**Modelo completo (ED, semilla 100):** error global **3.15 mm**
(IFP 4.01 / IFD 1.80 / Punta 3.64 mm), ROM DIP fisiológico de 30.7° y
viabilidad cinemática verificada en las 120 poses.

> Los valores exactos pueden variar levemente según la versión de las
> bibliotecas y el número de núcleos disponibles.

---

## Reproducción

```bash
pip install numpy scipy pandas matplotlib

# Modelo simplificado: comparacion ED vs GA
python3 optimizar_simplificado_ED.py
python3 optimizar_simplificado_GA.py
python3 comparar_ED_GA.py

# Modelo completo (algoritmo ganador: Evolucion Diferencial)
python3 optimizar_completo_ED.py
```

Presupuesto configurable por variables de entorno:
- ED: `OPT_POPSIZE`, `OPT_MAXITER`, `OPT_SEED`.
- GA: `GA_POPSIZE`, `GA_NGEN`, `GA_SEED`.

Las salidas (parámetros, trayectorias alineadas, curvas de convergencia y
gráficas de biofidelidad) se escriben en `resultados/`.
