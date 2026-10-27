# Señal perdida · Dashboard Telco Customer Churn

Dashboard interactivo en español, construido con **Python Dash y Plotly**, para explorar el proyecto de abandono de clientes de **Laura Rivera y Natalý Cárdenas**, Universidad del Norte.

![Portada del dashboard](preview-desktop.png)

- Jupyter Book: https://nataly-cardenas.github.io/Telco_Churn_Entregable/
- Fuente de los datos y resultados: https://github.com/Nataly-Cardenas/Telco_Churn_Entregable
- Commit de los datos: `41191edb71ebf15a1543d538a158513db8781d2f` (recuperado el 23 de octubre de 2026).

## Ejecutar

Requiere Python 3.12. Desde esta carpeta:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python app.py
```

En Windows, activar con `.venv\Scripts\activate`.
Abrir http://localhost:8050. La variable `PORT` permite cambiar el puerto.

## Publicar en Render

El proyecto incluye `render.yaml` y expone `server = app.server`:

1. Subir estos archivos al repositorio de la aplicación.
2. En Render, crear un Blueprint desde ese repositorio usando `render.yaml`.
3. Render instala dependencias y ejecuta `gunicorn app:server --bind 0.0.0.0:$PORT --workers 2 --timeout 120`.

La app no requiere claves, base de datos ni acceso a GitHub al ejecutarse. No se ha creado un servicio remoto desde esta sesión.

## Recorrido

Las secciones siguen los capítulos del Jupyter Book, con las mismas gráficas y tablas (aquí animadas e interactivas), para poder explicar el proyecto desde el dash.

- **Portada interactiva:** una red de clientes conectados a un celular 3D. Los puntos resaltados son clientes a punto de irse y se pueden retener pasando el cursor (o el dedo).
- **Introducción:** contexto, objetivo general, objetivos específicos, marco teórico en tarjetas y descripción de variables.
- **Análisis exploratorio:** filtros globales y pestañas con la numeración del libro: ETL y partición, univariado, bivariado (pruebas, Cramér V y Holm), multivariado (Q-Q, Spearman, matriz de dispersión, KDE), valores faltantes, resumen y explorador con descarga CSV.
- **Preprocesamiento:** pasos del pipeline, validación cruzada, ablación y evaluación del modelo base en test (matriz de confusión y ROC animada).
- **Modelado avanzado:** podio; resultados por modelo (matriz de confusión, ROC y confiabilidad), calibración, optimizadores, comparación estadística (Friedman, Nemenyi, DeLong) e interpretabilidad (SHAP global, waterfall y LIME).
- **Simulador:** herramienta creada para el dash (perfil de riesgo, clientes reales del test y laboratorio de umbral).
- **Conclusiones:** hallazgos, limitaciones, próximos pasos y enlace al paper.
- **Autoras:** fotos, LinkedIn, GitHub y correo.

Cada capítulo tiene un único enlace a su versión en el Jupyter Book, debajo del título. El diseño se adapta a móvil y respeta `prefers-reduced-motion`.

## Idioma, tema y paleta

- **Español / inglés:** botones `ES | EN` arriba a la derecha. Todos los textos están en `i18n.py` y un test verifica que ninguno quede sin traducir.
- **Modo claro / oscuro:** botón de sol/luna. La elección se guarda en el navegador.
- **Íconos:** íconos de línea propios en `icons.py` (sin emojis).
- **Paleta:** las gráficas usan un solo color (`#2E86AB` azul) y cambian a `#E4572E` naranja solo para resaltar el mejor resultado o la clase de interés; `#9AA5B1` gris para referencias. Los textos usan blanco/gris y el mismo azul como acento. Un test verifica que ninguna gráfica use otros colores.

## Asistente

El botón flotante abre un asistente sin límite de preguntas que responde sobre el proyecto, sus datos y cada sección del dash. No responde copiando texto del Jupyter Book. Para preguntas sobre perfiles concretos, como «tasa de abandono de clientes con fibra y mes a mes», calcula la cifra sobre el conjunto de entrenamiento.

- **Sin configuración**, responde con la base de conocimiento local de `chatbot.py` (preguntas frecuentes en ES/EN sobre el dash).
- **Con la variable de entorno `ANTHROPIC_API_KEY`**, redacta las respuestas con Claude (`claude-opus-5-5`) usando ese mismo contexto, y vuelve a la respuesta local si la API falla. Cada pregunta consume créditos de la API de Anthropic.

```bash
# Windows PowerShell
$env:ANTHROPIC_API_KEY="tu-clave"
python app.py
```

En Render, la variable se agrega en *Environment* (ya está declarada en `render.yaml` sin valor).

## Contrato de datos

- Dataset real de 7.043 clientes, 1.869 abandonos.
- Índices históricos: 5.634 train y 1.409 test, sin solapamiento.
- `TotalCharges` se convierte a numérico para el EDA y conserva sus 11 ausencias. La explicación del pipeline documenta su imputación determinista con cero por `tenure = 0`.
- Los filtros superiores afectan al EDA y al explorador. **No recalculan ni reentrenan modelos publicados**.
- El simulador de perfil muestra la tasa de abandono **observada** en clientes de entrenamiento parecidos; no es una predicción del modelo.
- La tabla de 112 configuraciones representa media y desviación de cinco folds externos de la realización 0. `unidades.csv` contiene las 1.680 unidades (112 × 3 × 5).
- El panel de optimización promedia el mejor F1 **interno** acumulado, sin confundirlo con el externo.
- El laboratorio de umbral usa predicciones **nativas** guardadas de tres finalistas sobre el test histórico. Es didáctico: mover el umbral no valida una regla operativa.
- El test ya había sido expuesto en la primera entrega y es evidencia suplementaria. Los contrastes por folds son exploratorios, no confirmatorios.
- No se cargan pickles ni se entrena al iniciar. La selección de archivos está en `data/provenance.json`.
- Las gráficas y tablas presentan los resultados numéricos originales del libro (pruebas del EDA recalculadas con los mismos métodos; predicciones OOF, SHAP y LIME tomadas de los artefactos publicados).
- En el comparador, las barras de error son **desviaciones estándar**, no IC. El mapa familia × balanceo promedia optimizadores.
- La tabla del explorador filtra su vista localmente. El CSV exporta la selección completa de los filtros superiores y las columnas elegidas, tal como indica la interfaz.

## Archivos

- `app.py`: portada, secciones, callbacks, paleta y servidor.
- `i18n.py`: traducciones español → inglés.
- `icons.py`: íconos de línea.
- `chatbot.py`: asistente (base de conocimiento, búsqueda y conexión opcional con Claude).
- `eda.py`: pruebas estadísticas del análisis exploratorio.
- `analytics.py`: carga local, cohortes, tasas, clientes similares y errores por umbral.
- `assets/style.css`: diseño visual, animaciones y estilos de los controles de Dash 4.
- `assets/cover.js`: red de clientes, celular 3D y contadores animados (Dash lo carga automáticamente).
- `data/`: CSV, interpretabilidad y textos originales necesarios, unos 8 MB.
- `tests/test_dashboard.py`: integridad del split, métricas, filtros, simulador, páginas en ES/EN sin emojis, traducciones completas, descarga única del CSV, asistente y endpoints.
- `preview-desktop.png`, `preview-mobile.png`: capturas reales del dashboard.

## Verificar

```bash
pip install pytest
python -m pytest -q
```

Se comprobaron navegación, filtros, comparación de modelos, búsqueda, descarga CSV y visualización de escritorio/móvil con Chromium. `/healthz` entrega un estado simple para monitoreo. No se han reproducido los entrenamientos originales; se verifican los artefactos publicados.
