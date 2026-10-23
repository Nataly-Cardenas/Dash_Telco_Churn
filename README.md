# Telco Atlas · Dashboard del Jupyter Book

Dashboard interactivo en español, construido con **Python Dash y Plotly** para explicar el proyecto de abandono de clientes de **Laura Rivera y Natalý Cárdenas**, Universidad del Norte.

- Libro: https://nataly-cardenas.github.io/Telco_Churn_Entregable/
- Fuente: https://github.com/Nataly-Cardenas/Telco_Churn_Entregable
- Inspiración de navegación e interpretación: https://github.com/lauraformore/DASH_VIZ
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

1. Panorama: KPIs, composición de clases, abandono por contrato y antigüedad.
2. Proyecto y objetivos: problema, conceptos y alcance temporal.
3. Datos y calidad: partición, faltantes, duplicados y diccionario.
4. Análisis univariado: todas las variables candidatas, histograma, caja y conteos.
5. Abandono por segmento: tasas con denominadores y distribuciones por resultado.
6. Relaciones multivariadas: Spearman, redundancia y dispersión de dos variables.
7. Preprocesamiento: reglas, ingeniería, pipeline, ablación y baseline.
8. Comparador de modelos: las 112 configuraciones, filtros, métricas y desviación entre folds.
9. Optimización: convergencia interna y tiempos por familia, con presupuesto común.
10. Calibración y umbrales: Brier/ECE y simulación de errores sobre test histórico.
11. Evidencia estadística: Friedman, DeLong/Holm e incertidumbre condicional.
12. SHAP y LIME: importancia global, aciertos, errores y fidelidad del sustituto local.
13. Conclusiones: evidencia, limitaciones y pasos para evaluación prospectiva.
14. Explorador: columnas seleccionables, orden/filtro de tabla y exportación CSV.
15. Libro en detalle: explicaciones de todos los apartados de EDA, preprocesamiento y modelado, buscables por texto.
16. Autoras y fuentes: autoría y trazabilidad.

Las rutas usan fragmentos (`#models`, `#explain`, etc.), admiten enlaces directos y navegación del navegador. Incluye modo presentación y diseño móvil.

## Contrato de datos

- Dataset real de 7.043 clientes, 1.869 abandonos.
- Índices históricos: 5.634 train y 1.409 test, sin solapamiento.
- `TotalCharges` se convierte a numérico para el EDA y conserva sus 11 ausencias. La explicación del pipeline documenta su imputación determinista con cero por `tenure = 0`.
- Los filtros superiores afectan al EDA y al explorador. **No recalculan ni reentrenan modelos publicados**.
- La tabla de 112 configuraciones representa media y desviación de cinco folds externos de la realización 0. `unidades.csv` contiene las 1.680 unidades (112 × 3 × 5).
- El panel de optimización promedia el mejor F1 **interno** acumulado, sin confundirlo con el externo.
- El laboratorio de umbral usa predicciones **nativas** guardadas de tres finalistas sobre el test histórico. Es didáctico: mover el umbral no valida una regla operativa.
- El test ya había sido expuesto en la primera entrega y es evidencia suplementaria. Los contrastes por folds son exploratorios, no confirmatorios.
- No se cargan pickles ni se entrena al iniciar. La selección de archivos está en `data/provenance.json`.
- Los textos del lector y las figuras LIME proceden de los notebooks y artefactos originales, con atribución. El lector omite código y salidas de ejecución; las gráficas y tablas del dashboard presentan los resultados numéricos originales.
- En el comparador, las barras de error son **desviaciones estándar**, no IC. El mapa familia × balanceo promedia optimizadores.
- La tabla del explorador filtra su vista localmente. El CSV exporta la selección completa de los filtros superiores y las columnas elegidas, tal como indica la interfaz.

## Archivos

- `app.py`: páginas, callbacks y servidor.
- `analytics.py`: carga local, cohortes, tasas y errores por umbral.
- `assets/style.css`: interfaz responsive oscura y estilos de controles Dash 4.
- `data/`: CSV, interpretabilidad y textos originales necesarios, unos 8 MB.
- `tests/test_dashboard.py`: integridad del split, métricas, filtros, serialización de páginas, exportación y endpoints.
- `preview-desktop.png`, `preview-mobile.png`: capturas reales del dashboard.

## Verificar

```bash
pip install pytest
python -m pytest -q
```

Se comprobaron navegación, filtros, comparación de modelos, búsqueda, descarga CSV y visualización de escritorio/móvil con Chromium. `/healthz` entrega un estado simple para monitoreo. No se han reproducido los entrenamientos originales; se verifican los artefactos publicados.
