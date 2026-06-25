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
   funcion objetivo y limites**:
   - `optimizar_simplificado_ED.py` - Evolucion Diferencial (scipy).
   - `optimizar_simplificado_GA.py` - Algoritmo Genetico (codificacion real:
     torneo + SBX + mutacion polinomial + elitismo).

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

## Resultados (modelo simplificado, semilla 42)

| Metodo | Error global | Error IFP | Error IFD | Evaluaciones | Tiempo |
|--------|-------------:|----------:|----------:|-------------:|-------:|
| **Evolucion Diferencial** | **2.15 mm** | 2.66 mm | 1.65 mm | 72 597 | ~41 s |
| Algoritmo Genetico | 5.55 mm | 3.93 mm | 7.17 mm | 20 470 | ~96 s |

La **Evolucion Diferencial** le gana por bastante y es mucho mas estable entre
semillas, mientras que el GA tiene mas varianza y convergencia prematura. Por
eso adoptamos ED pa el modelo completo.

**Modelo completo (ED, semilla 100):** error global **3.15 mm**
(IFP 4.01 / IFD 1.80 / Punta 3.64 mm), ROM DIP fisiologico de 30.7 deg y
viabilidad cinematica verificada en las 120 poses.

> Ojo: los valores exactos pueden variar un poco segun la version de las
> bibliotecas y cuantos nucleos haya disponibles.

---

## Como reproducir

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

Las salidas (parametros, trayectorias alineadas, curvas de convergencia y
graficas de biofidelidad) se escriben en `resultados/`.
