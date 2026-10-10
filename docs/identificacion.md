# Identificación: qué se puede y qué no se puede afirmar

Estado: **con los datos actuales el diseño NO identifica pass-through cambiario.**
Identifica (a lo sumo) un **diferencial relativo de precios** entre dos grupos de
capítulos del IPC INDEC alrededor de diciembre de 2023.

## 1. Qué estima el código

`log(IPC_it) = a_i + g_t + Σ_{τ≠-1} β_τ · 1[τ_t = τ] · D_i + e_it`

- Unidades: 12 capítulos del IPC nacional (INDEC/SSPM), mensual.
- `D_i` = 1 si el capítulo fue **clasificado a priori** como "transable" (5 capítulos);
  la clasificación es un juicio del autor, no una medida (no hay contenido
  importado, ni participación de bienes comercializables, ni exposición continua).
- `τ` = meses a dic-2023; base τ = -1 (nov-2023).
- Errores clusterizados por capítulo (G = 12, 5 tratados).

`β_τ` es el cambio en el diferencial de inflación acumulada entre grupos respecto de nov-2023.

## 2. Por qué hoy no hay pass-through

El pass-through es `∂ ln P / ∂ ln E` (elasticidad del precio al tipo de cambio). Para
identificarlo hace falta (a) una variable de TC o un shock de TC medido, y (b) variación
en la exposición al TC que sea independiente de otros determinantes del precio.

1. **No hay tipo de cambio en el modelo.** El "tratamiento" es un dummy de
   capítulo × tiempo. Nunca se mide cuánto se movió el TC, ni cuánto se transmitió.
   Un `β` no es una elasticidad; no tiene unidades de pass-through.
2. **El shock de dic-2023 es compuesto.** Simultáneamente: devaluación oficial (13-dic;
   magnitud a verificar en fuente oficial), desregulación/liberalización de precios (DNU 70/2023), quita de   subsidios a energía y transporte, tarifazos, fin de acuerdos de precios
   (Precios Justos), cambios de impuestos. Estos afectaron **diferencialmente** a los
   capítulos de la clasificación (p. ej. Vivienda/Servicios públicos y Transporte están
   en el grupo "no transable" y recibieron aumentos regulados fuertes). La fecha es
   común a todo; no hay un contraste que aísle el canal cambiario.
3. **Tratamiento no aleatorio y definido a priori con N=12.** Los grupos difieren en
   volatilidad, indexación, regulación y composición. Las tendencias paralelas no
   son un supuesto creíble por construcción, y no se puede testear la ausencia de
   confusores contemporáneos con datos pre-shock.
4. **Variable dependiente:** índice de capítulo agregado, no precios de producto.
   Mezcla bienes y servicios dentro de capítulos ("Alimentos" no es 100% transable;
   "Transporte" incluye combustibles y autos).
5. **Contaminación de la base y el post:** la inflación mensual cambia de régimen
   (aceleración en dic-23/ene-24 y desinflación posterior); la ventana post incluye
   la reversión, por lo que `β_τ` mezcla dinámica de TC real, atraso de tarifas y
   normalización. El efecto estático depende de la ventana (ver §5).

## 3. Confusores no resueltos (lista explícita)

- Quita de subsidios y tarifazos (energía, agua, gas, transporte).
- Liberalización de precios y salida de acuerdos de precios.
- Indexación/inercia diferencial y contratos en USD (alquileres, prepagas, educación privada).
- Cambios impositivos (p. ej. impuestos internos, PAIS) con incidencia por capítulo.
- Brecha cambiaria previa (dólar paralelo ya reflejado en precios de bienes transables
  antes de dic-23: puede producir "pre-tendencias" o un pass-through ya consumido).
- Estacionalidad diferencial por capítulo (Educación en marzo, Prendas por temporada, Turismo).
- Composición de los ponderadores del IPC y su actualización.

## 4. Limitaciones inferenciales con 12 clusters / 5 tratados

- La varianza clusterizada asintótica (CR) subestima la varianza con G pequeño y pocos
  tratados (Cameron-Gelbach-Miller 2008; MacKinnon-Webb 2017). Los IC "normales" del
  gráfico son **anticonservadores**.
- El test de pre-tendencias (Wald, q = nº de leads) tiene **potencia baja**: no rechazar
  no valida el supuesto (Roth 2022); además, condicionar el análisis en pasar el
  pre-test distorsiona la inferencia posterior.
- La estimación puntual de la matriz de covarianzas con G=12 y q=5 restricciones es
  inestable; se reporta además el p-valor con F(q, G-1) y el bootstrap.
- `econometria/robustez.py` provee: wild cluster bootstrap (WCR Rademacher; con G=12 se
  enumeran exactamente las 2^12 combinaciones de signos), randomization inference exacta
  (792 asignaciones de 5 tratados entre 12), placebo en fechas falsas, leave-one-out por
  capítulo y ventanas alternativas. Ninguno de estos arregla el sesgo de identificación;
  sólo corrigen/ilustran la incertidumbre muestral.
- **HonestDiD (Rambachan-Roth) no está implementado.** No hay port Python maduro. Lo que
  hay (`sensibilidad_magnitud_relativa`) son cotas heurísticas simples sin cobertura
  garantizada. Un análisis de sensibilidad formal queda **pendiente**.

## 5. Resultados de la corrida de referencia (datos INDEC descargados; rama mejoras-honestidad)

Corrida con ventana ±6 meses; esto es salida de `python -m econometria.robustez`, no un
resultado a citar como evidencia de pass-through:

- DiD estático (`D × 1[τ≥0]`): β = 0.027, EE clusterizado = 0.068, t = 0.40. G = 12, 5 tratados.
- Inferencia alternativa: WCR p = 0.717 (propio, 4096 signos exactos) y 0.716
  (`wildboottest`, WCR31); p t(G-1) = 0.697; randomization inference exacta p = 0.708.
- Pre-tendencias: Wald χ²(5) = 8.53 (EE CR1 propio, p asintótico 0.129; bootstrap p = 0.530).
  Con `linearmodels` el estadístico es 7.16 (p = 0.209) por la distinta corrección de muestra
  pequeña. Los coeficientes pre del event study son negativos y decrecientes en magnitud
  hacia τ = -1 (≈ -0.03 a -0.005): no se puede descartar una tendencia diferencial.
- El signo del DiD estático cambia al excluir Bienes y Serv. Varios o Educación, y con
  ventanas post de 9 y 12 meses. Placebos: ninguno rechaza al 5%, pero los de may–jun/2023
  tienen p ≈ 0.08–0.15 y signo opuesto al "efecto".
- Cota heurística de magnitud relativa: el intervalo incluye 0 incluso con M = 0.

Lectura honesta: no hay efecto estadísticamente distinguible de 0 y los intervalos son
muy anchos (poca potencia); tampoco se puede descartar un efecto relevante. **Ausencia de
evidencia, no evidencia de ausencia.**

## 6. Gate de identificabilidad

El texto "pass-through" (en README, app, informes de IA, títulos) sólo puede usarse si se
cumplen **todos** estos criterios; hoy se cumple **0 de 5**:

| # | Criterio | Hoy |
|---|----------|-----|
| G1 | Variable de TC (o shock de TC medido) con fuente verificada incorporada al modelo | No |
| G2 | Exposición medida (continua o con base documentada), no un dummy a priori | No |
| G3 | Control o estrategia frente a los confusores de dic-23 (subsidios/regulados excluidos o controlados; ver §3) | No |
| G4 | Inferencia válida con pocos clusters: WCR y RI concordantes, y ≥ 20–30 clusters o unidades más finas (precios de producto/IPC por región) | No |
| G5 | Pre-tendencias evaluadas con análisis de sensibilidad (HonestDiD o similar) y no sólo un Wald no rechazado | No |

Si G1–G3 no se cumplen: el alcance permitido es **"diferencial relativo de precios
(grupo clasificado transable vs. no transable) alrededor de dic-2023"**.

### Especificación propuesta (NO implementada; no hay datos verificados incorporados)

- Serie de TC oficial (y, en paralelo, de la brecha con el dólar financiero/paralelo) del
  BCRA u otra fuente oficial, **a verificar al momento de usarla**.
- Exposición continua `E_i` (por ejemplo, contenido importado o participación de bienes
  comercializables por capítulo, de la matriz insumo-producto INDEC o equivalente) en vez de
  `D_i`: `Δ ln P_it = α_i + γ_t + Σ_k θ_k · E_i · 1[τ_t=k] + ...`; pass-through = `θ`
  escalado por el movimiento del TC medido.
- Excluir/controlar regulados (energía, transporte público, agua) y, si es posible,
  usar IPC por regiones o precios de producto para aumentar el número de clusters.

## 7. Texto del repo que debería cambiar (no modificado por este agente)

- README / app / notebook / informe IA: reemplazar "pass-through cambiario", "efecto del salto
  cambiario", "Shock Devaluatorio" por "diferencial relativo de precios (transable vs. no
  transable, clasificación a priori) alrededor de dic-2023".
- Cualquier "Tendencias paralelas válidas / validadas" → "no se rechaza H0 con poca potencia;
  no valida el supuesto".
- `datos/api_indec.py`: las `justificacion` de cada capítulo afirman "alta exposición…" sin
  evidencia en el repo; deben presentarse como clasificación a priori no verificada.
- `notebook_principal.ipynb`: usa `wald_test.stat/.pval` de `linearmodels` y la celda de texto
  fija "chi2(5)"; debe leerse del dict de `wald_pretendencias`.
- `app/streamlit_app.py` (líneas ~140-141) todavía usa `wald_test.stat`/`pval`; ahora
  `estimar_event_study_twfe` devuelve un dict (`stat_chi2`, `df`, `pvalue_chi2`, `pvalue_F_G_menos_1`).

## Referencias

Cameron, Gelbach y Miller (2008); MacKinnon y Webb (2017); Roth (2022), "Pretest with
Caution"; Rambachan y Roth (2023), "A More Credible Approach to Parallel Trends".
