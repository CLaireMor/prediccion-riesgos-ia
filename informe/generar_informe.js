// Genera informe/Informe_preparacion_datos.docx a partir de resultados.json y figuras/.
// Requisitos: python preparacion_datos.py (antes) y el paquete npm "docx".
// Uso: node informe/generar_informe.js [--aprendiz "Nombre Apellido"]
const fs = require("fs");
const path = require("path");
const {
  Document, Packer, Paragraph, TextRun, ImageRun, Table, TableRow, TableCell, WidthType, ShadingType,
  AlignmentType, HeadingLevel, PageBreak, BorderStyle, LevelFormat, Footer, Header, PageNumber, ExternalHyperlink,
  TabStopType,
} = require("docx");

const DIR = __dirname;
const RAIZ = path.join(DIR, "..");
const R = JSON.parse(fs.readFileSync(path.join(DIR, "resultados.json"), "utf8"));
const arg = process.argv.indexOf("--aprendiz");
const APRENDIZ = arg > 0 ? process.argv[arg + 1] : null;
const FECHA = "28 de septiembre de 2026";

const TINTA = "15201B", TINTA2 = "4B5852", ACENTO = "0B6B5C", AZUL = "2A78D6", RAYA = "D5DDD8", FONDO = "EEF3F0";
const ANCHO = 9026; // A4 con márgenes de 1"
const fmt = (n) => Number(n).toLocaleString("es-CO");
const pct = (x) => (x * 100).toFixed(1).replace(".", ",") + " %";

// ---------- Ayudas ----------
const t = (text, o = {}) => new TextRun({ text, ...o });
const p = (runs, o = {}) => new Paragraph({ children: Array.isArray(runs) ? runs : [t(runs)], spacing: { after: 120, line: 300 }, ...o });
const h1 = (text) => new Paragraph({ heading: HeadingLevel.HEADING_1, children: [t(text)] });
const h2 = (text) => new Paragraph({ heading: HeadingLevel.HEADING_2, children: [t(text)] });
const vi = (text, nivel = 0) => new Paragraph({ numbering: { reference: "vinetas", level: nivel }, children: parseRuns(text), spacing: { after: 60, line: 290 } });
const salto = () => new Paragraph({ children: [new PageBreak()] });

// Negritas con **texto**
function parseRuns(s, base = {}) {
  return s.split(/(\*\*[^*]+\*\*)/).filter(Boolean).map((x) => x.startsWith("**") ? t(x.slice(2, -2), { ...base, bold: true }) : t(x, base));
}
const pr = (s, o = {}) => p(parseRuns(s), o);

function tabla(cab, filas, anchos, { num = [], destacar = null } = {}) {
  const total = anchos.reduce((a, b) => a + b, 0);
  const borde = { style: BorderStyle.SINGLE, size: 4, color: RAYA };
  const bordes = { top: borde, bottom: borde, left: borde, right: borde };
  const celda = (txt, i, esCab, fila) => new TableCell({
    width: { size: anchos[i], type: WidthType.DXA },
    borders: bordes,
    shading: esCab ? { type: ShadingType.CLEAR, fill: FONDO, color: "auto" } : (destacar && destacar(fila) ? { type: ShadingType.CLEAR, fill: "E3EEFB", color: "auto" } : undefined),
    margins: { top: 60, bottom: 60, left: 90, right: 90 },
    children: [new Paragraph({
      alignment: num.includes(i) && !esCab ? AlignmentType.RIGHT : AlignmentType.LEFT,
      children: parseRuns(String(txt), { size: 17, bold: esCab, color: esCab ? TINTA2 : TINTA }),
    })],
  });
  return new Table({
    width: { size: total, type: WidthType.DXA },
    columnWidths: anchos,
    rows: [
      new TableRow({ tableHeader: true, children: cab.map((c, i) => celda(c, i, true)) }),
      ...filas.map((f) => new TableRow({ children: f.map((c, i) => celda(c, i, false, f)) })),
    ],
  });
}
const leyenda = (s) => new Paragraph({ children: parseRuns(s, { size: 17, italics: true, color: TINTA2 }), spacing: { before: 80, after: 240 } });

function figura(archivo, anchoPx, titulo) {
  const buf = fs.readFileSync(path.join(DIR, "figuras", archivo));
  const w = buf.readUInt32BE(16), h = buf.readUInt32BE(20);
  const ancho = anchoPx, alto = Math.round((h / w) * ancho);
  return [
    new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 120 }, children: [new ImageRun({ type: "png", data: buf, transformation: { width: ancho, height: alto }, altText: { title: titulo, description: titulo, name: archivo } })] }),
    leyenda(titulo),
  ];
}

function codigo(archivo, desde = null, hasta = null) {
  let lineas = fs.readFileSync(path.join(RAIZ, archivo), "utf8").replace(/\r/g, "").split("\n");
  if (desde != null) lineas = lineas.slice(desde, hasta);
  return lineas.map((l) => new Paragraph({
    spacing: { after: 0, line: 228 },
    shading: { type: ShadingType.CLEAR, fill: "F4F6F4", color: "auto" },
    children: [t(l.length ? l : " ", { font: "Consolas", size: 14, color: TINTA })],
  }));
}
function lineasDe(archivo, inicio, fin) {
  const l = fs.readFileSync(path.join(RAIZ, archivo), "utf8").split("\n");
  const a = l.findIndex((x) => x.startsWith(inicio));
  const b = fin ? l.findIndex((x, i) => i > a && x.startsWith(fin)) : l.length;
  return [a, b];
}
const enlace = (texto, url) => new ExternalHyperlink({ link: url, children: [t(texto, { style: "Hyperlink" })] });

// ---------- Datos ----------
const pasos = Object.fromEntries(R.pasos.map((x) => [x.paso, x.detalle]));
const faltantes = pasos["Tratar faltantes"];
const imput = faltantes["imputación por moda"];
const obj = pasos["Agrupar clases raras del objetivo"];
const dup = pasos["Eliminar duplicados exactos (14 variables idénticas)"]["filas eliminadas"];
const [nA, cA] = R.dimension_antes, [nL] = R.dimension_limpio, [, cT] = R.dimension_transformado;
const imp = R.impacto_modelo;
const esc = R.escalado;

const NOMBRES = {
  Anio: "Año", Localidad: "Localidad de ocurrencia", Sexo: "Sexo", Edad: "Edad", Regimen: "Régimen de aseguramiento",
  Escolaridad: "Escolaridad", Ocupacion: "Ocupación", TipoUTI: "Tipo UTI", ClaseUTI: "Clase UTI", Ingresos: "Nivel de ingresos",
  FormaPago: "Forma de pago", Sintoma: "Síntoma", Agente: "Agente probablemente asociado", Lesion: "Tipo de lesión (objetivo)",
};
const ROL = {
  Anio: "Contexto temporal", Localidad: "Contexto geográfico", Sexo: "Predictor", Edad: "Predictor", Regimen: "Vulnerabilidad",
  Escolaridad: "Vulnerabilidad", Ocupacion: "Predictor", TipoUTI: "Contexto laboral", ClaseUTI: "Contexto laboral", Ingresos: "Vulnerabilidad",
  FormaPago: "Contexto laboral", Sintoma: "Excluida (posterior a la lesión)", Agente: "Predictor", Lesion: "Variable objetivo",
};
const TRATAMIENTO = {
  Anio: "Min-Max (0–1)", Localidad: "One-Hot", Sexo: "One-Hot", Edad: "Z-score", Regimen: "Limpieza + One-Hot",
  Escolaridad: "Ordinal + Min-Max", Ocupacion: "One-Hot (mín. 10 casos)", TipoUTI: "One-Hot", ClaseUTI: "One-Hot", Ingresos: "Ordinal + Min-Max",
  FormaPago: "One-Hot", Sintoma: "No se usa", Agente: "One-Hot (mín. 10 casos)", Lesion: "Agrupación + codificación de etiqueta",
};
const despuesPorVar = Object.fromEntries(R.estructura_despues.map((x) => [x.Variable, x]));

// ---------- Contenido ----------
const portada = [
  new Paragraph({ spacing: { before: 1800, after: 120 }, children: [t("INFORME TÉCNICO · PREPARACIÓN DE DATOS PARA INTELIGENCIA ARTIFICIAL", { size: 18, color: ACENTO, bold: true, characterSpacing: 20 })] }),
  new Paragraph({ spacing: { after: 240 }, border: { bottom: { style: BorderStyle.SINGLE, size: 12, color: ACENTO, space: 12 } },
    children: [t("Depuración y transformación del dataset de enfermedades derivadas de la ocupación en trabajadores informales de Bogotá", { size: 44, bold: true, color: TINTA, font: "Arial" })] }),
  p([t("Proyecto: ", { bold: true, color: TINTA2 }), t("Sistema de predicción de riesgos con IA", { color: TINTA2 })], { spacing: { before: 240, after: 1600 } }),
  p([t("Aprendiz", { size: 18, color: TINTA2, bold: true })], { spacing: { after: 40 } }),
  p([APRENDIZ ? t(APRENDIZ, { size: 26, bold: true }) : t("[Escribe aquí tu nombre completo]", { size: 26, bold: true, highlight: "yellow" })], { spacing: { after: 280 } }),
  p([t("Fecha", { size: 18, color: TINTA2, bold: true })], { spacing: { after: 40 } }),
  p([t(FECHA, { size: 26 })], { spacing: { after: 280 } }),
  p([t("Repositorio", { size: 18, color: TINTA2, bold: true })], { spacing: { after: 40 } }),
  p([enlace("github.com/CLaireMor/prediccion-riesgos-ia", "https://github.com/CLaireMor/prediccion-riesgos-ia")]),
  salto(),
];

const contenido = [
  h1("Contenido"),
  ...["1. Introducción", "2. Descripción del dataset base", "3. Resumen del proceso de depuración y transformación",
    "4. Tablas comparativas antes y después", "5. Gráficos de apoyo", "6. Impacto en el modelo de IA", "7. Conclusiones",
    "8. Verificación de la lista de chequeo", "Anexo A. Código de programación", "Anexo B. Enlaces y fuentes"].map((x) => p(x, { spacing: { after: 60 } })),
  salto(),
];

const intro = [
  h1("1. Introducción"),
  pr("Bogotá vigila la salud de los trabajadores de la economía informal a través del Subsistema de Vigilancia Epidemiológica Ocupacional de las y los Trabajadores de la Economía Informal (**SIVISTRA**) de la Secretaría Distrital de Salud. Cada notificación describe al trabajador (edad, sexo, escolaridad, aseguramiento, ingresos), su unidad de trabajo informal (UTI), la ocupación, el agente de riesgo al que se expone y el tipo de lesión o sistema comprometido."),
  pr("El proyecto usa estos datos para entrenar un modelo de inteligencia artificial que, a partir de la **edad, el sexo, la ocupación y el agente de riesgo**, estima qué tipo de lesión es más probable. El resultado se ofrece en una aplicación web que apoya la prevención: permite anticipar lesiones típicas por oficio, tamizar bases de trabajadores y priorizar intervenciones."),
  h2("1.1 Justificación técnica del análisis"),
  pr("Un modelo de IA solo puede ser tan bueno como los datos con los que aprende. El dataset original presenta problemas que, sin tratamiento, degradan el modelo: categorías duplicadas por espacios sobrantes, valores sin sentido como \"0\", registros repetidos, un objetivo con 60 clases de las cuales muchas tienen uno o dos casos, y variables categóricas que la versión inicial del proyecto convertía en números arbitrarios (0, 1, 2…), inventando un orden que no existe."),
  pr("Este informe documenta cada decisión de limpieza, codificación y escalado, explica por qué se eligió cada técnica según el tipo de dato y el modelo previsto (regresión logística multinomial con regularización), y mide su efecto con tablas y gráficos comparativos."),
  h2("1.2 Modelo de IA previsto"),
  pr("Se usa una **regresión logística multinomial** con regularización L2. Es un modelo lineal, interpretable y bien calibrado, adecuado para un objetivo categórico con muchas clases. Dos propiedades condicionan la preparación: necesita que las categóricas sin orden entren como variables indicadoras (One-Hot), y su regularización penaliza los coeficientes por igual, por lo que las variables numéricas deben estar en escalas comparables."),
];

const dataset = [
  h1("2. Descripción del dataset base"),
  h2("2.1 Dimensión"),
  pr(`El archivo **obs_salud1.csv** contiene **${fmt(nA)} registros** (filas) y **${cA} atributos** (columnas), con separador punto y coma y codificación Latin-1. Cubre notificaciones de ${esc["Año (original)"].min}–${esc["Año (original)"].max} en las 20 localidades de Bogotá. Supera ampliamente el mínimo requerido de 10 atributos y 100 registros.`),
  h2("2.2 Tipología de variables"),
  pr("El dataset combina **2 variables numéricas** (Año y Edad) y **12 categóricas**. Entre las categóricas, 2 tienen un orden explícito (Escolaridad y Nivel de ingresos) y 10 son nominales, sin orden."),
  tabla(["Variable", "Tipo", "Subtipo", "Distintos", "Rol en el proyecto"],
    R.estructura_antes.map((v) => [NOMBRES[v.Variable], v.Tipo, v.Tipo === "numérica" ? "Discreta" : (["Escolaridad", "Ingresos"].includes(v.Variable) ? "Ordinal" : "Nominal"), fmt(v.Distintos), ROL[v.Variable]]),
    [2500, 1150, 1050, 1000, 3326], { num: [3] }),
  leyenda("Tabla 1. Variables del dataset original, su tipo y su rol. \"Distintos\" = número de valores únicos."),
  h2("2.3 Origen documentado"),
  pr("**Fuente:** Secretaría Distrital de Salud de Bogotá, Observatorio de Salud de Bogotá (SaluData), subsistema SIVISTRA. Conjunto de datos \"Enfermedades derivadas de la ocupación en Unidades de Trabajo Informal (UTI) en Bogotá D.C.\", publicado en el portal de Datos Abiertos Bogotá. Los enlaces completos están en el Anexo B."),
  p([t("Portal de datos abiertos: ", { bold: true }), enlace("datosabiertos.bogota.gov.co/dataset/enfermedades-derivadas-de-la-ocupacion-en-unidades-de-trabajo-informal-uti-en-bogota-d-c", "https://datosabiertos.bogota.gov.co/dataset/enfermedades-derivadas-de-la-ocupacion-en-unidades-de-trabajo-informal-uti-en-bogota-d-c")]),
  p([t("Indicador en SaluData: ", { bold: true }), enlace("saludata.saludcapital.gov.co/osb/indicadores/enfermedades-derivadas-de-la-ocupacion", "https://saludata.saludcapital.gov.co/osb/en/indicadores/enfermedades-derivadas-de-la-ocupacion/")]),
];

const proceso = [
  h1("3. Resumen del proceso de depuración y transformación"),
  pr("El proceso completo está en el script **preparacion_datos.py** (Anexo A) y se ejecuta en este orden. Cada paso registra cuántos valores o filas afectó, y esas cifras son las que aparecen en este informe."),
  h2("3.1 Coherencia de tipos y valores"),
  vi("**Tipos por variable.** Año y Edad se leen como enteros; las otras 12 como texto. Se verificó que ninguna variable numérica viniera como texto ni al revés."),
  vi(`**Rangos.** Edad va de ${esc["Edad (original)"].min} a ${esc["Edad (original)"].max} años y Año de ${esc["Año (original)"].min} a ${esc["Año (original)"].max}. No hay valores imposibles (edades negativas o años futuros). Las edades de 15 a 17 años (${R.menores_18} casos) no se eliminan: son reales y señalan posible trabajo adolescente.`),
  vi(`**Espacios sobrantes.** \"Especial\" y \"Especial \" aparecían como dos regímenes distintos. Al normalizar espacios, Régimen pasa de ${pasos["Normalizar espacios en textos"].Regimen} y Síntoma de ${pasos["Normalizar espacios en textos"].Sintoma}. También se limpiaron etiquetas del objetivo como \"Afecciones de vía respiratoria baja \".`),
  vi(`**Valores sin sentido.** Un \"0\" en Forma de pago, un \"0\" en Tipo de lesión y ${pasos["Valores sin sentido ('0', 'NS / NR') marcados como faltantes"]["Ingresos (NS / NR)"]} respuestas \"NS / NR\" en Nivel de ingresos se convirtieron en faltantes, para tratarlos con una regla explícita en lugar de dejarlos como categorías falsas.`),
  h2("3.2 Duplicados"),
  pr(`Se encontraron **${dup} filas idénticas en las 14 variables**, incluidas la edad exacta y el síntoma. Sin un identificador de caso, la explicación más probable es una notificación repetida del mismo caso, así que se eliminaron conservando la primera aparición. Representan el ${pct(dup / nA)} de los registros, por lo que su eliminación no altera la distribución y evita que el modelo cuente dos veces el mismo caso (y que un caso quede a la vez en entrenamiento y en prueba).`),
  h2("3.3 Tratamiento de faltantes"),
  pr("Los faltantes son muy pocos, pero cada uno recibió una decisión justificada según el papel de la variable:"),
  tabla(["Variable", "Faltantes", "Decisión", "Justificación"], [
    ["Ocupación", String(faltantes["nulos detectados"].Ocupacion), "Eliminar fila", "Es predictor principal del modelo; imputarla inventaría una exposición laboral."],
    ["Tipo de lesión", String(faltantes["nulos detectados"].Lesion), "Eliminar fila", "Es la variable objetivo: un registro sin respuesta no sirve para entrenar ni evaluar."],
    ...Object.entries(imput).map(([k, v]) => [NOMBRES[k], String(v.n), `Imputar moda (\"${v.moda}\")`, "Variable secundaria categórica; con tan pocos casos la moda no distorsiona la distribución."]),
  ], [1700, 1000, 2400, 3926], { num: [1] }),
  leyenda(`Tabla 2. Faltantes detectados (tras eliminar duplicados) y su tratamiento. En total se eliminaron ${faltantes["filas eliminadas (variable clave u objetivo)"]} filas.`),
  pr("Se usó la **moda** y no la media o la mediana porque todas las variables con faltantes son categóricas: la media no existe para ellas. En Nivel de ingresos, aunque es ordinal, la moda (\"1 SMMLV\") coincide con la mediana, así que ambas opciones darían el mismo resultado."),
  h2("3.4 Variable objetivo"),
  pr(`El tipo de lesión tenía **${obj["clases antes"]} clases**. Muchas tienen menos de 30 casos y, al revisarlas, casi todas son registros con varias lesiones concatenadas en una sola celda (por ejemplo \"Artrosis Dorsolumbalgias Otros Trastornos De Tejidos Blandos\"). Con tan pocos ejemplos el modelo no puede aprenderlas, así que se agruparon en \"Otras lesiones / registros múltiples\". Quedan **${obj["clases después"]} clases** y se reagrupan solo ${fmt(obj["registros agrupados"])} registros (${pct(obj["registros agrupados"] / nL)}).`),
  h2("3.5 Codificación categórica"),
  vi(`**One-Hot Encoding** para las 8 nominales: Sexo, Localidad, Régimen, Ocupación, Tipo UTI, Clase UTI, Forma de pago y Agente. Cada categoría se vuelve una columna 0/1. Las categorías con menos de 10 casos se agrupan en una columna \"infrecuente\" para no crear cientos de columnas casi vacías; así Ocupación genera ${R.columnas_onehot.Ocupacion} columnas en lugar de 228 y Agente ${R.columnas_onehot.Agente} en lugar de 76.`),
  vi("**Ordinal Encoding** para las 2 variables con jerarquía, con el orden definido a mano (no alfabético):"),
  vi(`Escolaridad (0 a 11): ${R.orden_escolaridad.join(" < ")}.`, 1),
  vi(`Nivel de ingresos (0 a 3): ${R.orden_ingresos.join(" < ")}.`, 1),
  vi("**Síntoma** se excluye de la matriz: se registra cuando la lesión ya apareció, así que usarlo para predecirla sería fuga de información y el modelo no serviría para prevenir."),
  vi("**Objetivo:** las 26 clases se codifican como etiquetas 0–25, que es lo que espera el clasificador (no son variables de entrada, así que el número no introduce orden)."),
  h2("3.6 Escalado numérico"),
  vi("**Z-score** (media 0, desviación estándar 1) para **Edad**. Su distribución es aproximadamente simétrica y tiene algunos valores altos (hasta 95 años). El Z-score no comprime el resto de los datos por culpa de esos extremos, algo que sí hace Min-Max."),
  vi("**Min-Max** (0 a 1) para **Año** y para los códigos ordinales de **Escolaridad** e **Ingresos**. Son variables acotadas y sin atípicos, y Min-Max conserva el orden y los deja en el mismo rango que las columnas One-Hot (0/1)."),
  h2("3.7 Justificación técnica de cada técnica"),
  tabla(["Técnica", "Aplicada a", "Por qué", "Alternativa descartada"], [
    ["Normalizar texto", "Todas las categóricas", "Evita categorías falsas por espacios.", "Dejarlas: duplica categorías."],
    ["Eliminar duplicados", `${dup} filas`, "Evita contar dos veces un caso y la fuga entre entrenamiento y prueba.", "Conservarlos: sesga frecuencias."],
    ["Eliminar filas", "Ocupación, Lesión", "Variable clave u objetivo: imputar inventaría información.", "Imputar moda: crea exposiciones falsas."],
    ["Imputar moda", "5 variables secundarias", "Categóricas con 1–12 faltantes; no distorsiona.", "Media/mediana: no aplican a categorías."],
    ["One-Hot", "8 nominales", "Sin orden natural; la regresión logística necesita indicadoras.", "LabelEncoder: inventa un orden 0 < 1 < 2."],
    ["Ordinal", "Escolaridad, Ingresos", "Tienen jerarquía explícita que conviene conservar.", "One-Hot: pierde el orden."],
    ["Z-score", "Edad", "Simétrica, con atípicos; la regularización L2 exige escalas comparables.", "Min-Max: comprime por los extremos."],
    ["Min-Max", "Año, ordinales", "Acotadas, sin atípicos; quedan en 0–1 como las One-Hot.", "Z-score: igual de válido, menos interpretable."],
  ], [1600, 1750, 3200, 2476]),
  leyenda("Tabla 3. Resumen de técnicas, a qué se aplicaron y por qué."),
];

const comparativas = [
  h1("4. Tablas comparativas antes y después"),
  h2("4.1 Estructura general"),
  tabla(["Indicador", "Antes", "Después de depurar", "Después de transformar"], [
    ["Registros (filas)", fmt(nA), fmt(nL), fmt(nL)],
    ["Atributos (columnas)", String(cA), String(cA), `${cT - 1} + objetivo`],
    ["Valores faltantes o sin sentido", String(R.estructura_antes.reduce((a, v) => a + v.Nulos, 0) + 1 + 1 + pasos["Valores sin sentido ('0', 'NS / NR') marcados como faltantes"]["Ingresos (NS / NR)"]), "0", "0"],
    ["Filas duplicadas", String(dup), "0", "0"],
    ["Categorías de Régimen", "7", "6", `${R.columnas_onehot.Regimen} columnas 0/1`],
    ["Clases del objetivo", fmt(R.estructura_antes.find((v) => v.Variable === "Lesion").Distintos), String(obj["clases después"]), `${obj["clases después"]} etiquetas (0–25)`],
    ["Tipos de dato", "2 numéricas + 12 texto", "2 numéricas + 12 texto", `${cT - 1} numéricas`],
  ], [2700, 1900, 2100, 2326], { num: [1, 2] }),
  leyenda(`Tabla 4. Estructura del dataset en cada etapa. En \"Antes\", el objetivo cuenta también el valor \"0\"; los faltantes incluyen \"0\" y \"NS / NR\" y se cuentan antes de eliminar duplicados.`),
  h2("4.2 Por variable"),
  tabla(["Variable", "Faltantes antes", "Distintos antes", "Faltantes después", "Distintos después", "Tratamiento"],
    R.estructura_antes.map((v) => {
      const d = despuesPorVar[v.Variable];
      const falt = v.Nulos + ({ FormaPago: 1, Lesion: 1, Ingresos: 15 }[v.Variable] || 0);
      return [NOMBRES[v.Variable], String(falt), fmt(v.Distintos), String(d.Nulos), fmt(d.Distintos), TRATAMIENTO[v.Variable]];
    }), [2150, 1050, 1050, 1100, 1100, 2576], { num: [1, 2, 3, 4] }),
  leyenda("Tabla 5. Comparación por variable. Los faltantes \"antes\" incluyen valores sin sentido (\"0\", \"NS / NR\")."),
  h2("4.3 Estadísticos de las variables escaladas"),
  tabla(["Variable", "Media", "Desv. estándar", "Mínimo", "Máximo"],
    Object.entries(esc).map(([k, v]) => [k, v.media.toFixed(3).replace(".", ","), v.de.toFixed(3).replace(".", ","), v.min.toFixed(3).replace(".", ","), v.max.toFixed(3).replace(".", ",")]),
    [3226, 1450, 1450, 1450, 1450], { num: [1, 2, 3, 4] }),
  leyenda("Tabla 6. El Z-score deja la Edad con media 0 y desviación 1; Min-Max deja Año y los ordinales entre 0 y 1."),
  h2("4.4 Ejemplo de un registro antes y después"),
  tabla(["Campo original", "Valor original", "Columnas resultantes"], [
    ["Edad", "63", `zscore__Edad = ${((63 - esc["Edad (original)"].media) / esc["Edad (original)"].de).toFixed(3).replace(".", ",")}`],
    ["Sexo", "Femenino", "onehot__Sexo_Femenino = 1; demás columnas de Sexo = 0"],
    ["Escolaridad", "Secundaria Incompleta", "ordinal__Escolaridad = 3/11 = 0,273"],
    ["Nivel de ingresos", "1 SMMLV", "ordinal__Ingresos = 1/3 = 0,333"],
    ["Año", "2019", "minmax__Anio = (2019 − 2017)/8 = 0,250"],
    ["Ocupación", "Zapateros Y Afines", "onehot__Ocupacion_Zapateros Y Afines = 1; demás = 0"],
  ], [2000, 2400, 4626]),
  leyenda("Tabla 7. Cómo se transforma el primer registro del dataset (una zapatera de 63 años de Tunjuelito, 2019)."),
];

const graficos = [
  h1("5. Gráficos de apoyo"),
  pr("Los gráficos muestran el estado de los datos antes y después de cada transformación."),
  ...figura("01_nulos.png", 560, "Figura 1. Valores faltantes o sin sentido por variable. Después de la depuración no queda ninguno."),
  ...figura("02_histogramas_edad.png", 600, `Figura 2. Histogramas de Edad original, con Z-score y con Min-Max. La forma de la distribución es idéntica; solo cambia la escala. Con Z-score la media pasa de ${esc["Edad (original)"].media.toFixed(1).replace(".", ",")} años a 0 y la desviación de ${esc["Edad (original)"].de.toFixed(1).replace(".", ",")} a 1.`),
  ...figura("03_boxplots.png", 600, `Figura 3. Boxplots. A la izquierda, Edad en años: ${R.atipicos.atipicos_edad} valores superan el límite de ${String(R.atipicos.limite_superior_iqr).replace(".", ",")} años (Q3 + 1,5 × IQR). Se conservan porque son edades reales. A la derecha, las variables ya escaladas quedan en rangos comparables.`),
  ...figura("04_dispersion.png", 600, `Figura 4. Dispersión de Edad frente a Escolaridad, antes y después (muestra aleatoria de 3.000 registros). La correlación (r = ${String(R.correlacion_edad_escolaridad).replace(".", ",")}: a mayor edad, menor escolaridad) se conserva exactamente, porque ambas transformaciones son lineales y respetan el orden.`),
  ...figura("05_objetivo.png", 600, `Figura 5. Distribución del objetivo en escala logarítmica. Antes: ${obj["clases antes"]} clases, muchas con 1 o 2 casos. Después: ${obj["clases después"]} clases, todas con al menos 30 casos.`),
];

const impacto = [
  h1("6. Impacto en el modelo de IA"),
  pr("Para medir si la preparación mejora el modelo, se entrenaron dos versiones con la misma partición 80/20 y se compararon con una línea base que predice siempre la lesión más común, sin mirar al trabajador:"),
  tabla(["Tratamiento", "Acierto 1ª opción", "Acierto top 3", "Log-loss (menor es mejor)"],
    imp.map((r) => [r.Tratamiento, pct(r.top1), pct(r.top3), r.log_loss.toFixed(2).replace(".", ",")]),
    [4226, 1600, 1600, 1600], { num: [1, 2, 3], destacar: (f) => f[0].startsWith("Depurado") }),
  leyenda("Tabla 8. Desempeño con la misma partición de prueba. \"Top 3\" = la lesión real está entre las 3 que el modelo considera más probables."),
  ...figura("06_impacto_modelo.png", 580, "Figura 6. Comparación del acierto entre el tratamiento original, el depurado y la línea base."),
  pr(`El tratamiento depurado sube el acierto en la primera opción de **${pct(imp[0].top1)} a ${pct(imp[1].top1)}** y en el top 3 de **${pct(imp[0].top3)} a ${pct(imp[1].top3)}**, además de reducir el log-loss (mejor calibración). La mejora no viene de un modelo más complejo, sino de lo contrario: se pasó de una red neuronal a una regresión logística, más simple. El cambio que importa es la preparación, sobre todo One-Hot en lugar de números arbitrarios y la limpieza del objetivo.`),
  pr(`Esta cifra es optimista porque mezcla años. Si se entrena con datos anteriores a 2024 y se prueba con 2024–2025, que es como se usaría en la práctica, el modelo acierta la primera opción en cerca del **29 %** y el top 3 en cerca del **66 %**, frente a 18 % y 42 % de la línea base. La caída se debe a que la forma de registrar los casos cambia entre años (en 2023, el 44 % de los casos se codificó como \"Otros trastornos de tejidos blandos\").`),
];

const conclusiones = [
  h1("7. Conclusiones"),
  vi(`**La calidad de los datos pesa más que la complejidad del modelo.** Con los mismos datos de origen, una preparación correcta y un modelo más simple superan a la red neuronal original: +${((imp[1].top1 - imp[0].top1) * 100).toFixed(1).replace(".", ",")} puntos en la primera opción y +${((imp[1].top3 - imp[0].top3) * 100).toFixed(1).replace(".", ",")} en el top 3.`),
  vi("**Codificar según el tipo de dato es decisivo.** Tratar Ocupación o Agente como números (0, 1, 2…) le dice al modelo que \"peluquero < mecánico\", y eso es falso. One-Hot elimina ese sesgo, y la codificación ordinal conserva la jerarquía real de Escolaridad e Ingresos."),
  vi(`**La limpieza fue pequeña en volumen pero importante en efecto.** Se eliminó solo el ${pct((nA - nL) / nA)} de los registros (${fmt(nA - nL)} filas), pero quitar duplicados, categorías falsas y valores sin sentido evita que el modelo aprenda artefactos del registro.`),
  vi("**El escalado no cambia la información, cambia la escala.** Los histogramas y la dispersión lo confirman: la forma de las distribuciones y las correlaciones se conservan, y todas las variables quedan en rangos comparables. Eso es lo que necesita la regularización L2 para tratarlas con equidad."),
  vi("**Aplicabilidad en IA.** El dataset transformado (22.167 × 167 variables numéricas) queda listo para regresión logística y sirve también para otros modelos lineales, redes neuronales o máquinas de vectores de soporte. Los modelos de árboles no necesitarían el escalado, pero sí se benefician de la limpieza."),
  vi("**Límites.** Los datos contienen solo casos notificados: el modelo estima qué lesión es más probable si el trabajador llega a ser un caso, no la probabilidad de lesionarse. Además, el agente de riesgo es el \"probablemente asociado\" según quien registra. Por eso los resultados apoyan la prevención, pero no reemplazan una valoración clínica."),
];

const check = [
  h1("8. Verificación de la lista de chequeo"),
  tabla(["Requisito", "Dónde se cumple", "Estado"], [
    ["Dimensión mínima (≥10 atributos, ≥100 registros)", `Sección 2.1: ${cA} atributos, ${fmt(nA)} registros`, "Cumple"],
    ["Tipología mixta", "Sección 2.2, Tabla 1: 2 numéricas y 12 categóricas", "Cumple"],
    ["Origen documentado", "Sección 2.3 y Anexo B", "Cumple"],
    ["Tratamiento de nulos/faltantes", "Sección 3.3, Tabla 2: moda y eliminación justificada", "Cumple"],
    ["Coherencia y duplicados", `Secciones 3.1 y 3.2: tipos, rangos, texto y ${dup} duplicados`, "Cumple"],
    ["Codificación categórica", "Sección 3.5: One-Hot (8 nominales) y Ordinal (2 ordinales)", "Cumple"],
    ["Escalado numérico", "Sección 3.6: Z-score (Edad) y Min-Max (Año, ordinales)", "Cumple"],
    ["Justificación técnica", "Sección 3.7, Tabla 3", "Cumple"],
    ["Tablas antes y después", "Sección 4, Tablas 4 a 7", "Cumple"],
    ["Histogramas, dispersión y boxplots", "Sección 5, Figuras 2, 3 y 4", "Cumple"],
    ["Portada (título, aprendiz, fecha)", "Portada", APRENDIZ ? "Cumple" : "Falta el nombre"],
    ["Introducción", "Sección 1", "Cumple"],
    ["Resumen del proceso", "Sección 3", "Cumple"],
    ["Tablas y gráficos integrados", "Secciones 4, 5 y 6", "Cumple"],
    ["Conclusiones", "Sección 7", "Cumple"],
    ["Anexos: código y enlaces", "Anexos A y B", "Cumple"],
  ], [3300, 4326, 1400]),
  leyenda("Tabla 9. Cada requisito de la lista de chequeo y la sección donde se cumple."),
];

const [a1, b1] = lineasDe("modelo.py", "def cargar_datos", "def _metricas");
const anexos = [
  salto(),
  h1("Anexo A. Código de programación"),
  pr("Todo el código está en Python 3.11 con pandas, scikit-learn y matplotlib, y se encuentra en el repositorio del proyecto. Para reproducir los resultados de este informe:"),
  ...codigo_texto(["pip install -r requirements.txt matplotlib", "python preparacion_datos.py      # depuración, transformación, tablas y figuras", "python -m pytest tests            # pruebas automáticas del modelo"]),
  h2("A.1 preparacion_datos.py (depuración, codificación, escalado y gráficos)"),
  ...codigo("preparacion_datos.py"),
  h2("A.2 modelo.py (fragmento: carga, limpieza y pipeline del modelo)"),
  ...codigo("modelo.py", a1, b1),
  h1("Anexo B. Enlaces y fuentes"),
  vi("Datos Abiertos Bogotá. \"Enfermedades derivadas de la ocupación en Unidades de Trabajo Informal (UTI) en Bogotá D.C.\" Secretaría Distrital de Salud. https://datosabiertos.bogota.gov.co/dataset/enfermedades-derivadas-de-la-ocupacion-en-unidades-de-trabajo-informal-uti-en-bogota-d-c"),
  vi("Observatorio de Salud de Bogotá (SaluData). \"Enfermedades derivadas de la ocupación en Unidades de Trabajo Informal.\" https://saludata.saludcapital.gov.co/osb/en/indicadores/enfermedades-derivadas-de-la-ocupacion/"),
  vi("SaluData. Sección Salud Laboral. https://saludata.saludcapital.gov.co/osb/datos-de-salud/salud-laboral/"),
  vi("Repositorio del proyecto (código, datos y aplicación). https://github.com/CLaireMor/prediccion-riesgos-ia"),
  vi("Documentación de scikit-learn: OneHotEncoder, OrdinalEncoder, StandardScaler, MinMaxScaler y LogisticRegression. https://scikit-learn.org/stable/modules/preprocessing.html"),
];
function codigo_texto(lineas) {
  return lineas.map((l) => new Paragraph({ spacing: { after: 0, line: 240 }, shading: { type: ShadingType.CLEAR, fill: "F4F6F4", color: "auto" }, children: [t(l, { font: "Consolas", size: 16 })] }));
}

// ---------- Documento ----------
const doc = new Document({
  creator: APRENDIZ || "Aprendiz",
  title: "Depuración y transformación del dataset de enfermedades derivadas de la ocupación",
  styles: {
    default: { document: { run: { font: "Calibri", size: 21, color: TINTA } } },
    paragraphStyles: [
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { font: "Arial", size: 30, bold: true, color: ACENTO }, paragraph: { spacing: { before: 360, after: 160 }, outlineLevel: 0, keepNext: true } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { font: "Arial", size: 23, bold: true, color: TINTA }, paragraph: { spacing: { before: 240, after: 100 }, outlineLevel: 1, keepNext: true } },
    ],
  },
  numbering: { config: [{ reference: "vinetas", levels: [
    { level: 0, format: LevelFormat.BULLET, text: "•", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 440, hanging: 260 } } } },
    { level: 1, format: LevelFormat.BULLET, text: "–", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 880, hanging: 260 } } } },
  ] }] },
  sections: [
    { properties: { page: { margin: { top: 1440, bottom: 1440, left: 1440, right: 1440 } } }, children: portada },
    {
      properties: { page: { margin: { top: 1440, bottom: 1440, left: 1440, right: 1440 }, pageNumbers: { start: 2 } } },
      headers: { default: new Header({ children: [new Paragraph({ alignment: AlignmentType.RIGHT, children: [t("Preparación de datos · Riesgo laboral informal", { size: 16, color: TINTA2 })] })] }) },
      footers: { default: new Footer({ children: [new Paragraph({ alignment: AlignmentType.RIGHT, children: [t("Página ", { size: 16, color: TINTA2 }), new TextRun({ children: [PageNumber.CURRENT], size: 16, color: TINTA2 })] })] }) },
      children: [...contenido, ...intro, ...dataset, ...proceso, ...comparativas, ...graficos, ...impacto, ...conclusiones, ...check, ...anexos],
    },
  ],
});

const salida = path.join(DIR, "Informe_preparacion_datos.docx");
Packer.toBuffer(doc).then((b) => { fs.writeFileSync(salida, b); console.log(salida); });
