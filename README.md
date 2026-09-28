# 🏥 Predicción de riesgos en trabajadores informales

Aplicación en Streamlit que, a partir de la **edad, sexo, ocupación y agente de exposición** de un
trabajador informal, estima qué **tipo de lesión o sistema comprometido** es más probable, y ofrece
herramientas de vigilancia y priorización con los datos del observatorio de salud de Bogotá
(`obs_salud1.csv`, 22 438 registros, 22 167 tras depurar, 2017–2025).

```bash
pip install -r requirements.txt
streamlit run app.py
```

### Versión web (sin servidor)

`web/index.html` es la misma app en una sola página que corre en el navegador: incluye los
coeficientes del modelo y tablas **agregadas** (ningún registro individual). Se abre con doble clic
o se puede publicar en cualquier hosting estático (GitHub Pages, Netlify…). Para regenerarla
después de cambiar datos o modelo:

```bash
python exportar_web.py   # verifica que la predicción del navegador coincide con scikit-learn
```

La interfaz se edita en `web/plantilla.html`; `web/index.html` es el archivo generado.

## Pestañas

| Pestaña | Para qué sirve |
|---|---|
| 🔎 **Predicción** | Top-k lesiones para un perfil, con la **razón frente a la prevalencia** (cuántas veces más frecuente que en el total de casos) y los **casos reales** con la misma ocupación y agente como evidencia. Alerta si el trabajador es menor de edad o adulto mayor. |
| 📂 **Por lotes** | Sube un CSV (`Edad, Sexo, Ocupacion, Agente`) y descarga las 3 lesiones más probables por trabajador. Útil para tamizar censos de UTI o planear brigadas. |
| 📈 **Vigilancia** | Lesiones y agentes más frecuentes, tendencia anual y matriz ocupación × lesión, filtrable por años, localidad y sexo. |
| 🎯 **Priorización** | Carga de casos vs. vulnerabilidad social por ocupación (% no asegurado, % < 1 SMMLV, % ≥ 65 años, pago a destajo) y listado de casos en **menores de 18 años**. |
| 🧪 **Confiabilidad** | Métricas del modelo frente a una línea base, validación temporal y señales de cambios en la codificación. |

## Qué tan bien funciona (medido, no supuesto)

| Validación | Método | Acierto 1ª opción | Acierto en top 3 | Log-loss |
|---|---|---|---|---|
| Aleatoria 80/20 | Modelo actual | 43.1 % | 78.3 % | 1.63 |
| Aleatoria 80/20 | Modelo anterior (MLP + LabelEncoder, sin depurar) | 36.4 % | 72.0 % | 1.88 |
| Aleatoria 80/20 | Línea base (prevalencia) | 18.9 % | 46.0 % | 2.63 |
| Temporal (< 2024 → ≥ 2024) | Modelo actual | 29.2 % | 65.6 % | 2.23 |
| Temporal (< 2024 → ≥ 2024) | Línea base (prevalencia) | 18.2 % | 42.3 % | 2.79 |

La **validación temporal** es la cifra honesta: simula predecir casos futuros. La caída frente a la
aleatoria se explica por cambios en cómo se registran los casos entre años (p. ej. en 2023 el 44 % de
los casos se codificó como "Otros trastornos de tejidos blandos", frente a 5 % en 2017).

## Decisiones de modelado

- **One-hot + regresión logística** en lugar de `LabelEncoder` + red neuronal. Codificar ocupaciones o
  agentes como 0, 1, 2… les inventa un orden que no existe; el modelo nuevo es más preciso, más rápido
  (≈3 s vs ≈11 s), mejor calibrado e interpretable.
- **Se agrupan lesiones con < 30 casos** (la mayoría son registros con varias lesiones concatenadas)
  en una categoría "poco frecuentes / múltiples".
- **Se eliminan 264 duplicados exactos** (las 14 variables idénticas, probablemente notificaciones repetidas).
- **Limpieza de textos**: espacios sobrantes (p. ej. `"Afecciones de vía respiratoria baja "`) y valores
  basura (`"0"`).
- **No se usan `Año` ni `Localidad` como predictores**. `Localidad` sube el acierto temporal
  (≈37 %), pero probablemente refleja *quién registra* (equipos locales con hábitos de codificación
  distintos), no un riesgo del trabajador; incluirla haría al modelo aprender el sesgo del registro.
  `Año` empeora la validación temporal. Las variables socioeconómicas aportan < 2 puntos y se usan en
  la pestaña de priorización en vez del modelo.
- **`Síntoma` no se usa**: se conoce después de que aparece la lesión, así que usarlo sería fuga de
  información para un uso preventivo.

## Límites que hay que conocer

- Los datos contienen **solo casos notificados**. El modelo estima *qué lesión es más probable si el
  trabajador llega a ser un caso*, **no** la probabilidad de lesionarse (incidencia). Para eso haría
  falta el denominador (trabajadores expuestos sin lesión).
- Asociaciones, no causalidad: el "agente" es el *probablemente asociado* según quien registró.
- Combinaciones ocupación-agente sin casos se extrapolan; la app lo advierte.
- Es apoyo a la prevención, no un diagnóstico clínico.

## Usos adicionales que habilita

- **Tamizaje masivo** de bases de trabajadores para priorizar visitas (pestaña por lotes).
- **Detección de trabajo infantil/adolescente** (hay casos de 15–17 años) y de adultos mayores
  trabajando sin protección.
- **Auditoría de calidad del registro**: picos de categorías genéricas o diferencias entre localidades.
- **Focalización de intervenciones** combinando carga de casos y vulnerabilidad social.
- **Material de formación** para equipos de salud ocupacional: perfiles típicos de lesión por oficio.

## Informe de preparación de datos

`informe/Informe_preparacion_datos.docx` (y su versión PDF) documenta la depuración, codificación y
escalado del dataset con tablas antes/después y gráficos. Para regenerarlo:

```bash
python preparacion_datos.py                 # tablas, cifras y figuras en informe/
node informe/generar_informe.js --aprendiz "Nombre Apellido"   # requiere: npm install docx
```

Fuente de los datos: [Datos Abiertos Bogotá – Enfermedades derivadas de la ocupación en UTI](https://datosabiertos.bogota.gov.co/dataset/enfermedades-derivadas-de-la-ocupacion-en-unidades-de-trabajo-informal-uti-en-bogota-d-c)
(Secretaría Distrital de Salud, SIVISTRA).

## Estructura

```
app.py          # interfaz Streamlit
modelo.py       # carga, limpieza, entrenamiento, evaluación y predicción
exportar_web.py # genera web/index.html (versión web autónoma)
web/            # plantilla.html (fuente) e index.html (generado)
tests/          # pruebas (pip install -r requirements-dev.txt && pytest)
obs_salud1.csv  # datos (separador ";" y codificación latin-1)
```
