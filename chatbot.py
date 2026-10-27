"""Asistente del dashboard: responde preguntas sobre el proyecto, sus datos y cada sección del dash, sin límite de preguntas.

Funciona en dos modos:
- Con la variable de entorno ANTHROPIC_API_KEY, redacta la respuesta con Claude a partir del contexto recuperado.
- Sin ella (o si la API falla), responde con la entrada más parecida de la base de conocimiento local.
"""
import os
import re
from functools import lru_cache
import numpy as np
from dash import dcc, html, Input, Output, State, ALL, ctx, no_update
from sklearn.feature_extraction.text import TfidfVectorizer
import analytics as a
from i18n import EN, T, N, set_lang, get_lang
from icons import icon

MODEL = 'claude-opus-5-5'
BOOK = 'https://nataly-cardenas.github.io/Telco_Churn_Entregable/'

# ---------------------------------------------------------------- base de conocimiento
# Cada entrada: (preguntas de ejemplo, respuesta en español, respuesta en inglés).
FAQ = [
    (['qué es el churn', 'qué es el abandono de clientes', 'what is churn'],
     'El *churn* o abandono es la cancelación del servicio por parte de un cliente. En este proyecto la variable `Churn` indica si el cliente se fue durante el último mes observado.',
     'Churn is when a customer cancels the service. In this project the `Churn` column says whether the customer left during the last observed month.'),
    (['de qué trata el proyecto', 'cuál es el objetivo', 'what is the project about', 'goal of the project'],
     'El proyecto busca identificar a los clientes de una empresa de telecomunicaciones con mayor riesgo de abandonar el servicio y los factores asociados, comparando de forma reproducible 112 configuraciones de modelos de clasificación.',
     'The project aims to identify the telecom customers most likely to cancel and the factors associated with it, by reproducibly comparing 112 classification model configurations.'),
    (['qué datos se usaron', 'cuántos clientes hay en el dataset', 'de dónde salen los datos', 'cuántas variables hay', 'which data was used', 'how many customers are in the dataset', 'where does the data come from'],
     'Se usó el conjunto público **Telco Customer Churn** de IBM (Kaggle): 7,043 clientes y 21 columnas con servicios contratados, datos de la cuenta y perfil demográfico. Se dividió 80/20: 5,634 clientes de entrenamiento y 1,409 de test.',
     'The public IBM **Telco Customer Churn** dataset (Kaggle): 7,043 customers and 21 columns with services, account data and demographics. It was split 80/20: 5,634 training and 1,409 test customers.'),
    (['cuál es la tasa de abandono', 'cuántos clientes abandonan', 'qué porcentaje se va', 'what is the churn rate', 'how many customers churn', 'what percentage leaves'],
     'Abandonan 1,869 de 7,043 clientes, es decir **26.5 %**. Por eso es un problema desbalanceado: la clase de interés es minoritaria.',
     '1,869 of 7,043 customers churn, i.e. **26.5 %**. The problem is imbalanced: the class of interest is the minority.'),
    (['qué clientes abandonan más', 'quiénes se van más', 'cuáles son los factores de riesgo', 'qué variables influyen en el abandono', 'por qué se van los clientes', 'which customers churn the most', 'who leaves the most', 'what are the risk factors', 'which variables matter', 'why do customers leave'],
     'Las tasas más altas aparecen con **cheque electrónico (45.7 %)**, **contrato mes a mes (42.7 %)**, **fibra óptica (42.1 %)**, sin seguridad en línea (41.9 %) y sin soporte técnico (41.8 %). SHAP confirma que **contrato, servicio de internet y antigüedad** son las variables que más pesan en los modelos.',
     'The highest rates appear with **electronic check (45.7 %)**, **month-to-month contracts (42.7 %)**, **fiber optic (42.1 %)**, no online security (41.9 %) and no tech support (41.8 %). SHAP confirms that **contract, internet service and tenure** weigh the most in the models.'),
    (['cuál fue el mejor modelo', 'qué modelo ganó', 'qué modelo funcionó mejor', 'cuál es el modelo con mejor f1', 'which was the best model', 'which model won', 'what is the best model'],
     '**Random Forest con SMOTE** obtuvo el mayor F1 medio (0.638 ± 0.023), empatado con **XGBoost con SMOTE** (0.638) y seguido por la regresión logística (0.634). Las diferencias no son estadísticamente detectables, así que no hay un ganador demostrado. Míralo en *Arena de modelos*.',
     '**Random Forest with SMOTE** had the highest mean F1 (0.638 ± 0.023), tied with **XGBoost with SMOTE** (0.638) and followed by logistic regression (0.634). The differences are not statistically detectable, so there is no proven winner. See *Model arena*.'),
    (['qué es el f1', 'qué métrica se usó', 'por qué f1', 'what is f1', 'which metric'],
     'El **F1** de la clase de abandono es la media armónica de Precision y Recall: F1 = 2TP / (2TP + FP + FN). Se eligió porque equilibra detectar a quienes se van y no generar demasiadas falsas alertas; con clases desbalanceadas es más informativo que la exactitud.',
     'The **F1** of the churn class is the harmonic mean of precision and recall: F1 = 2TP / (2TP + FP + FN). It balances catching churners and avoiding false alarms, and is more informative than accuracy with imbalanced classes.'),
    (['qué es el auc', 'auc roc', 'what is auc'],
     'El **AUC-ROC** mide qué tan bien el modelo ordena a los clientes por riesgo: es la probabilidad de que un cliente que abandona reciba más puntaje que uno que se queda. Los finalistas tienen AUC ≈ 0.84 en el test.',
     '**AUC-ROC** measures how well the model ranks customers by risk: the probability that a churner gets a higher score than a non-churner. The finalists reach AUC ≈ 0.84 on the test set.'),
    (['qué es smote y por qué se usó', 'cómo se trató el desbalance', 'qué técnicas de balanceo se usaron', 'what is smote and why was it used', 'how was class imbalance handled'],
     '**SMOTE** crea clientes sintéticos de la clase minoritaria interpolando entre vecinos cercanos. Se comparó con ADASYN, pesos de clase y no balancear; SMOTE mejoró el F1 en 6 de las 7 familias. Siempre se aplica solo sobre los datos de entrenamiento.',
     '**SMOTE** creates synthetic minority-class customers by interpolating between near neighbors. It was compared with ADASYN, class weights and no balancing; SMOTE improved F1 in 6 of the 7 families. It is only ever applied to training data.'),
    (['qué es la validación cruzada anidada', 'cómo se validaron los modelos', 'nested cross validation'],
     'En la **validación cruzada anidada** un bucle interno (3 folds) elige los hiperparámetros y uno externo (5 folds) mide el desempeño con datos que no participaron en esa elección. Así se evita una estimación optimista.',
     'In **nested cross-validation** an inner loop (3 folds) picks the hyperparameters and an outer loop (5 folds) measures performance on data that played no part in that choice, avoiding optimistic estimates.'),
    (['qué es la fuga de información', 'data leakage'],
     'La **fuga de información** ocurre cuando datos de validación o test influyen en el entrenamiento. Para evitarla, la imputación, la codificación, el escalado y el balanceo se ajustan solo con el entrenamiento de cada partición, dentro de un pipeline.',
     '**Data leakage** happens when validation or test data influence training. To avoid it, imputation, encoding, scaling and balancing are fitted only on each split\'s training data, inside a pipeline.'),
    (['cuántas configuraciones se probaron', 'cuántos modelos se probaron', 'qué modelos se compararon', 'how many models were tested', 'which models were compared', 'how many configurations'],
     'Se compararon **112 configuraciones**: 7 familias (regresión logística, KNN, Naive Bayes, árbol de decisión, Random Forest, XGBoost y SVM) × 4 balanceos (ninguno, pesos, SMOTE, ADASYN) × 4 buscadores de hiperparámetros (Grid, Random, TPE y genético).',
     '**112 configurations** were compared: 7 families (logistic regression, KNN, Naive Bayes, decision tree, Random Forest, XGBoost and SVM) × 4 balancing strategies (none, weights, SMOTE, ADASYN) × 4 hyperparameter searches (Grid, Random, TPE and genetic).'),
    (['qué optimizador es mejor', 'grid random tpe genético', 'which optimizer'],
     'Con el mismo presupuesto de 24 evaluaciones, TPE, Random y el algoritmo genético solo mejoraron a Grid con presupuestos bajos; al final quedaron prácticamente empatados. El balanceo influyó mucho más que el buscador. Mira la *Carrera de buscadores*.',
     'With the same budget of 24 evaluations, TPE, Random and the genetic algorithm only beat Grid at low budgets; in the end they were practically tied. Balancing mattered far more than the search method. See the *Search race*.'),
    (['qué es la calibración', 'brier ece', 'what is calibration'],
     'La **calibración sigmoide** (Platt) ajusta las probabilidades para que reflejen la frecuencia real. Redujo a la vez el puntaje de Brier y el ECE en 80 de 96 configuraciones.',
     '**Sigmoid calibration** (Platt) adjusts probabilities so they match observed frequencies. It reduced both the Brier score and ECE in 80 of 96 configurations.'),
    (['hay diferencias significativas', 'pruebas estadísticas', 'delong friedman', 'statistical tests'],
     'Friedman detecta diferencias entre las 112 configuraciones, pero Nemenyi no separa a las mejores. DeLong con corrección de Holm da p = 1.0 entre los tres finalistas y los intervalos bootstrap del AUC incluyen el cero.',
     'Friedman detects differences among the 112 configurations, but Nemenyi does not separate the best ones. DeLong with Holm correction gives p = 1.0 among the three finalists and the bootstrap AUC intervals include zero.'),
    (['qué es shap', 'qué es lime', 'interpretabilidad', 'what is shap'],
     '**SHAP** reparte cada predicción entre las variables (valores de Shapley) y su promedio da la importancia global. **LIME** explica un cliente concreto con un modelo simple local. Describen al modelo, no causas del abandono. Ver *¿Por qué se van?*.',
     '**SHAP** splits each prediction among the variables (Shapley values) and its average gives global importance. **LIME** explains one customer with a simple local model. They describe the model, not causes of churn. See *Why do they leave?*.'),
    (['cómo uso el simulador', 'perfil de riesgo', 'how do i use the simulator'],
     'En *Simulador → Perfil de riesgo* eliges contrato, internet, método de pago, antigüedad y cargo mensual; el dash busca clientes de entrenamiento parecidos y muestra su tasa de abandono observada. En *Clientes reales* ves las probabilidades de los 3 finalistas para un cliente del test, y en *Umbral de alerta* mueves la alarma de retención.',
     'In *Simulator → Risk profile* you choose contract, internet, payment method, tenure and monthly charge; the dash finds similar training customers and shows their observed churn rate. *Real customers* shows the 3 finalists\' probabilities for a test customer, and *Alert threshold* lets you move the retention alarm.'),
    (['qué pasa si cambio el umbral', 'umbral de alerta', 'threshold'],
     'Bajar el umbral detecta más clientes que se van (más Recall) pero genera más falsas alertas (menos Precision). Elegirlo bien exige conocer el costo de contactar a un cliente y el de perderlo.',
     'Lowering the threshold catches more churners (higher recall) but raises more false alarms (lower precision). Choosing it well requires knowing the cost of contacting a customer and of losing one.'),
    (['cómo descargo los datos', 'cómo descargo el csv', 'puedo exportar la tabla', 'how do i download the data', 'download csv', 'export the table'],
     'Ve a *Datos → Explorador*, elige las columnas y pulsa **Descargar CSV**. Se descarga un solo archivo con la selección de los filtros superiores.',
     'Go to *Data → Explorer*, pick the columns and press **Download CSV**. You get a single file with the current selection of the top filters.'),
    (['cómo cambio el idioma', 'cómo cambio a modo oscuro', 'cómo pongo el modo claro', 'how do i change the language', 'how do i switch to dark mode', 'light mode'],
     'Arriba a la derecha están los botones **ES / EN** para el idioma y el botón de sol/luna para cambiar entre modo claro y oscuro. La elección se recuerda en tu navegador.',
     'At the top right you\'ll find the **ES / EN** buttons for the language and the sun/moon button to switch between light and dark mode. Your choice is remembered in your browser.'),
    (['qué secciones tiene el dash', 'cómo navego', 'what sections'],
     'El dash tiene 8 secciones: El problema, Datos, Preparación, Arena de modelos, Simulador, ¿Por qué se van?, Conclusiones y Autoras. La portada es interactiva: pasa el cursor sobre los puntos rojos para retener clientes.',
     'The dash has 8 sections: The problem, Data, Preparation, Model arena, Simulator, Why do they leave?, Conclusions and Authors. The cover is interactive: hover over the red dots to retain customers.'),
    (['quiénes son las autoras', 'quién hizo el proyecto', 'who are the authors'],
     'Las autoras son **Laura S. Rivera Figueredo** y **Natalý A. Cárdenas Izaquita**, del Departamento de Matemáticas, Física y Ciencia de Datos de la Universidad del Norte (Barranquilla, Colombia).',
     'The authors are **Laura S. Rivera Figueredo** and **Natalý A. Cárdenas Izaquita**, from the Department of Mathematics, Physics and Data Science at Universidad del Norte (Barranquilla, Colombia).'),
    (['dónde está el jupyter book', 'dónde leo el paper', 'jupyter book', 'paper'],
     f'Todo el proyecto está en el [Jupyter Book]({BOOK}), incluido el [paper]({BOOK}paper.html). También hay un botón en la barra superior.',
     f'The whole project is in the [Jupyter Book]({BOOK}), including the [paper]({BOOK}paper.html). There is also a button in the top bar.'),
    (['cuáles son las limitaciones', 'limitations'],
     'Se usó un único conjunto de datos, el test ya había sido expuesto (es evidencia suplementaria), los datos son una foto de un momento y no hay costos de intervención, así que el F1 no se traduce directamente en dinero.',
     'A single dataset was used, the test set had already been seen (it is supplementary evidence), the data are a snapshot in time and there are no intervention costs, so F1 does not translate directly into money.'),
    (['cómo se preparó la información', 'preprocesamiento', 'valores faltantes', 'preprocessing', 'missing values'],
     'TotalCharges tenía 11 vacíos, todos de clientes con antigüedad 0, que se reemplazaron por 0. Se unificaron las categorías "sin internet/sin teléfono" en "No", se creó el cargo promedio mensual, se codificó con One-Hot y se estandarizó. Todo dentro del pipeline, sin fuga de información.',
     'TotalCharges had 11 blanks, all for customers with tenure 0, replaced by 0. "No internet/no phone" categories were merged into "No", an average monthly charge was created, and data were one-hot encoded and standardized, all inside the pipeline with no leakage.'),
    (['qué sigue', 'trabajo futuro', 'future work', 'next steps'],
     'Los próximos pasos son validar en clientes futuros (validación temporal), incorporar costos explícitos para elegir el umbral y diseñar un modelo propio que supere a las referencias en desempeño y tiempo.',
     'Next steps are validating on future customers (temporal validation), adding explicit costs to choose the threshold, and designing a proposed model that beats the baselines in performance and time.'),
]

FAQ.append((['qué significa el color del mapa de riesgo', 'cómo leo el sunburst', 'qué muestra el mapa de riesgo', 'how do i read the risk map', 'what does the sunburst color mean'],
            'En *Datos → Segmentos de riesgo*, el mapa circular va de contrato (centro) a internet y método de pago (afuera). El tamaño de cada porción es el número de clientes y el color, la tasa de abandono: coral es más riesgo, turquesa menos. Haz clic en una porción para profundizar.',
            'In *Data → Risk segments*, the circular map goes from contract (center) to internet and payment method (outside). Slice size is the number of customers and color is the churn rate: coral means more risk, teal less. Click a slice to drill down.'))
FAQ.append((['qué variables tiene el dataset', 'qué significa tenure', 'qué es monthlycharges', 'what variables are there', 'what does tenure mean'],
            'Hay 19 variables candidatas: de **cuenta** (contrato, método de pago, factura electrónica, `tenure` = antigüedad en meses, `MonthlyCharges` = cargo mensual, `TotalCharges` = cargo acumulado), de **servicios** (teléfono, líneas múltiples, internet, seguridad y respaldo en línea, protección de dispositivos, soporte técnico, streaming) y **demográficas** (género, adulto mayor, pareja, dependientes). Puedes explorarlas en *El problema* y en *Datos*.',
            'There are 19 candidate variables: **account** (contract, payment method, paperless billing, `tenure` = months as a customer, `MonthlyCharges`, `TotalCharges`), **services** (phone, multiple lines, internet, online security and backup, device protection, tech support, streaming) and **demographics** (gender, senior citizen, partner, dependents). Explore them in *The problem* and *Data*.'))
# Guía del dash: qué muestra cada sección y de qué capítulo del Jupyter Book viene.
GUIDE = [
    (['qué hay en la portada', 'qué son los puntos rojos', 'qué muestra el celular de la portada', 'what is on the cover', 'what are the red dots', 'what does the phone show'],
     'La **portada** muestra una red de clientes conectados a un celular. Los puntos rojos son clientes a punto de irse: si pasas el cursor encima los retienes. El celular recorre clientes reales del test con la probabilidad de abandono que les dio el Random Forest finalista, las señales de su perfil y lo que pasó en realidad.',
     'The **cover** shows a network of customers connected to a phone. Red dots are customers about to leave: hover over them to retain them. The phone cycles through real test customers with the churn probability given by the finalist Random Forest, their profile signals and what actually happened.'),
    (['qué muestra la sección el problema', 'de dónde sale el problema', 'what is in the problem section'],
     '*El problema* resume la **Introducción** del Jupyter Book: el reto, el objetivo general, los 8 objetivos específicos y el **marco teórico** en tarjetas que se voltean. Al final está el **análisis bivariado** del EDA: eliges una variable y ves su tasa de abandono por categoría.',
     '*The problem* summarizes the Jupyter Book **Introduction**: the challenge, the main goal, the 8 specific objectives and the **theoretical framework** as flip cards. At the end is the EDA **bivariate analysis**: pick a variable and see its churn rate by category.'),
    (['de qué sección del libro es cada parte', 'cómo sé de qué capítulo viene cada gráfica', 'qué significan las etiquetas de colores', 'which book chapter is each part from', 'what do the colored tags mean'],
     'Cada parte del dash tiene una **etiqueta de procedencia** que dice el capítulo y la sección del Jupyter Book de donde sale, con enlace directo. Los colores identifican el capítulo: turquesa = Introducción, violeta = EDA, ámbar = Preprocesamiento, coral = Modelado avanzado y verde = Paper. Las etiquetas grises punteadas marcan herramientas creadas solo para el dash.',
     'Every part of the dash has a **source tag** with the Jupyter Book chapter and section it comes from, linking straight to it. Colors identify the chapter: teal = Introduction, violet = EDA, amber = Preprocessing, coral = Advanced modeling and green = Paper. Grey dashed tags mark tools built only for the dash.'),
    (['qué muestra la sección de datos', 'para qué sirven los filtros', 'what is in the data section', 'what do the filters do'],
     '*Datos* corresponde al capítulo de **Análisis exploratorio (EDA)** y tiene una pestaña por sección: 1–2 Calidad y ETL, 3 Análisis univariado, 4 Análisis bivariado, 5 Análisis multivariado y el Explorador de datos. Los filtros de arriba (universo, contrato, internet y antigüedad) recalculan todas las gráficas de esta sección.',
     '*Data* matches the **Exploratory data analysis (EDA)** chapter, with one tab per section: 1–2 Quality and ETL, 3 Univariate analysis, 4 Bivariate analysis, 5 Multivariate analysis and the Data explorer. The top filters (population, contract, internet and tenure) recompute every chart in this section.'),
    (['qué es el análisis univariado', 'qué muestra la pestaña de distribuciones', 'what is the univariate analysis'],
     'El **análisis univariado** (EDA, sección 3) describe cada variable por separado: histograma con caja para las numéricas y conteos para las categóricas, coloreados por si el cliente se fue o se quedó. Muestra media, mediana, valores extremos según el IQR y vacíos.',
     'The **univariate analysis** (EDA, section 3) describes each variable on its own: histogram with box plot for numeric ones and counts for categorical ones, colored by whether the customer left or stayed. It shows mean, median, IQR outliers and blanks.'),
    (['qué es el análisis bivariado', 'qué muestra la animación de antigüedad', 'what is the bivariate analysis', 'what does the tenure animation show'],
     'El **análisis bivariado** (EDA, sección 4) cruza cada variable con el abandono: el mapa de riesgo (contrato → internet → pago), la animación que muestra cómo cae la tasa de abandono con la antigüedad y las tasas por categoría con su número de clientes.',
     'The **bivariate analysis** (EDA, section 4) crosses each variable with churn: the risk map (contract → internet → payment), the animation showing how churn falls with tenure and the rates per category with their customer counts.'),
    (['qué es el análisis multivariado', 'qué muestra la matriz de correlación', 'what is the multivariate analysis', 'correlation matrix'],
     'El **análisis multivariado** (EDA, sección 5) muestra las correlaciones de Spearman entre antigüedad, cargo mensual, cargo acumulado y abandono, y un gráfico de dispersión de dos variables. Antigüedad y cargo acumulado están muy correlacionadas (ρ ≈ 0.89).',
     'The **multivariate analysis** (EDA, section 5) shows Spearman correlations among tenure, monthly charge, total charges and churn, plus a two-variable scatter plot. Tenure and total charges are highly correlated (ρ ≈ 0.89).'),
    (['qué muestra la sección de preparación', 'qué es el pipeline', 'what is in the preparation section', 'what is the pipeline'],
     '*Preparación* resume el capítulo de **Preprocesamiento**: los 7 pasos del pipeline (objetivo, limpieza, categorías, nueva variable, codificar y escalar, balancear y modelo), la validación cruzada en entrenamiento, el modelo base de regresión logística en test (AUC 0.842, Recall 0.786) y la ablación de variables.',
     '*Preparation* summarizes the **Preprocessing** chapter: the 7 pipeline steps (target, cleaning, categories, new feature, encode and scale, balance and model), cross-validation on training, the logistic regression baseline on test (AUC 0.842, recall 0.786) and the variable ablation.'),
    (['qué muestra la arena de modelos', 'qué es la galaxia 3d', 'qué es el podio', 'what is the model arena', 'what is the 3d galaxy'],
     'La *Arena de modelos* resume el capítulo de **Modelado avanzado**: el podio de finalistas, el ranking de las 112 configuraciones (sección 1), la galaxia 3D que se gira con el mouse (Recall, Precision y AUC de cada configuración), la calibración (1.8), la carrera de optimizadores (2) y la comparación estadística (3).',
     'The *Model arena* summarizes the **Advanced modeling** chapter: the finalists podium, the ranking of the 112 configurations (section 1), the 3D galaxy you can rotate (recall, precision and AUC of each configuration), calibration (1.8), the optimizer race (2) and the statistical comparison (3).'),
    (['qué es el mapa familia por balanceo', 'qué muestra el heatmap de modelos', 'what is the family by balancing heatmap'],
     'El mapa **familia × balanceo** promedia la métrica elegida sobre los cuatro optimizadores para cada combinación. Las columnas con SMOTE suelen ser las más altas, lo que muestra que el balanceo influyó más que el buscador de hiperparámetros.',
     'The **family × balancing** heatmap averages the chosen metric over the four optimizers for each combination. SMOTE columns are usually the highest, showing balancing mattered more than the hyperparameter search.'),
    (['qué muestra el simulador', 'el simulador sale del libro', 'what does the simulator show'],
     'El *Simulador* es una herramienta creada para el dash con datos del libro: **Perfil de riesgo** se basa en el análisis bivariado del EDA y **Clientes reales** y **Umbral de alerta** usan las predicciones guardadas de los tres modelos finalistas en el test histórico (Modelado avanzado, sección 3).',
     'The *Simulator* is a tool built for the dash with the book\'s data: **Risk profile** is based on the EDA bivariate analysis, and **Real customers** and **Alert threshold** use the saved predictions of the three finalist models on the historical test (Advanced modeling, section 3).'),
    (['qué muestran los 100 cuadros del umbral', 'qué es la cuadrícula de 100 clientes', 'what is the 100 customer grid'],
     'En *Umbral de alerta*, la cuadrícula representa 100 clientes del test: coral sólido son abandonos detectados, coral punteado los que se escapan, ámbar las falsas alertas y turquesa los clientes estables bien leídos. Al mover el umbral cambian las proporciones y el punto en la curva ROC.',
     'In *Alert threshold*, the grid represents 100 test customers: solid coral are churners caught, dashed coral the ones missed, amber the false alarms and teal the stable customers read correctly. Moving the threshold changes the proportions and the point on the ROC curve.'),
    (['qué muestra por qué se van', 'qué son los factores asociados', 'what is in why do they leave'],
     '*¿Por qué se van?* combina los **factores asociados** del EDA (sección 4.2: contrato mes a mes 42.7 %, fibra óptica 42.1 %, primer año como cliente) con la **interpretabilidad** del Modelado avanzado (sección 3.2): importancia global SHAP y dos clientes explicados con SHAP y LIME.',
     '*Why do they leave?* combines the EDA **associated factors** (section 4.2: month-to-month 42.7 %, fiber optic 42.1 %, first year as a customer) with the Advanced modeling **interpretability** (section 3.2): global SHAP importance and two customers explained with SHAP and LIME.'),
    (['qué muestra la sección de conclusiones', 'qué es la biblioteca', 'what is in the conclusions section', 'what is the library'],
     '*Conclusiones* reúne los hallazgos (Modelado avanzado, 3.3), las limitaciones y el trabajo futuro (Paper, 4 y 5), el enlace al paper y la **Biblioteca**: todo el texto de los notebooks del libro, buscable por palabra clave.',
     '*Conclusions* gathers the findings (Advanced modeling, 3.3), limitations and future work (Paper, 4 and 5), the link to the paper and the **Library**: all the text of the book\'s notebooks, searchable by keyword.'),
    (['dónde están los linkedin de las autoras', 'cómo contacto a las autoras', 'linkedin', 'how do i contact the authors'],
     'En la sección *Autoras* están las fotos de Laura y Natalý con sus enlaces de **LinkedIn**, GitHub y correo, además del acceso al Jupyter Book, los repositorios, el dataset y el paper.',
     'The *Authors* section has Laura\'s and Natalý\'s photos with their **LinkedIn**, GitHub and email links, plus access to the Jupyter Book, the repositories, the dataset and the paper.'),
]
FAQ.extend(GUIDE)

FAQ.append((['hola', 'buenas', 'hello', 'hi', 'gracias', 'thanks'],
            '¡Hola! Puedo contarte sobre los datos, quiénes abandonan más, cómo se compararon los modelos, qué significan las métricas o cómo usar cada sección del dash. ¿Qué te gustaría saber?',
            'Hi! I can tell you about the data, who churns the most, how the models were compared, what the metrics mean or how to use each section of the dash. What would you like to know?'))

SUGGESTIONS = [N('¿Qué clientes abandonan más?'), N('¿Cuál fue el mejor modelo?'), N('¿Qué es SMOTE y por qué se usó?'), N('¿Cómo uso el simulador?'), N('¿Cómo descargo los datos?')]

SYSTEM = """Eres el asistente del dashboard "Señal perdida", sobre el proyecto Telco Customer Churn de Laura S. Rivera Figueredo y Natalý A. Cárdenas Izaquita (Universidad del Norte).
Respondes preguntas de cualquier visitante sobre el proyecto, sus resultados y cómo usar el dashboard.

Hechos del proyecto:
- Datos: Telco Customer Churn de IBM, 7,043 clientes, 21 columnas, 26.5 % de abandono. Partición estratificada 80/20 (5,634 entrenamiento, 1,409 test).
- Tasas de abandono más altas: cheque electrónico 45.7 %, contrato mes a mes 42.7 %, fibra óptica 42.1 %, sin seguridad en línea 41.9 %, sin soporte técnico 41.8 %. Con contrato a dos años: 2.9 %.
- Diseño: 7 familias de modelos × 4 balanceos × 4 optimizadores = 112 configuraciones, validación cruzada anidada (5 folds externos, 3 internos), F1 de la clase de abandono como métrica principal.
- Resultados: Random Forest + SMOTE F1 0.638 ± 0.023 (AUC 0.844), XGBoost + SMOTE 0.638, regresión logística 0.634; sin diferencias estadísticamente detectables. SMOTE mejoró 6 de 7 familias. La calibración sigmoide mejoró Brier y ECE en 80 de 96 configuraciones.
- SHAP: contrato, servicio de internet y antigüedad son las variables más importantes. Las atribuciones no son causales.
- Dashboard y su relación con el Jupyter Book (cada parte lleva una etiqueta de procedencia con enlace; colores: turquesa Introducción, violeta EDA, ámbar Preprocesamiento, coral Modelado avanzado, verde Paper):
  - Portada: red de clientes que se retienen con el cursor y celular con clientes reales del test.
  - El problema: Introducción (reto, objetivos, marco teórico) y análisis bivariado del EDA (sección 4).
  - Datos: EDA con pestañas 1–2 Calidad y ETL, 3 Análisis univariado, 4 Análisis bivariado, 5 Análisis multivariado y Explorador (descarga de CSV con botón). Filtros de universo, contrato, internet y antigüedad.
  - Preparación: Preprocesamiento (4 pasos del pipeline, 4.6 validación cruzada, 7 evaluación en test, 4.7 ablación).
  - Arena de modelos: Modelado avanzado (1 ranking, 1.8 calibración, 2 carrera de optimizadores, 3 comparación estadística), podio y galaxia 3D.
  - Simulador (herramienta del dash): perfil de riesgo con clientes similares, clientes reales con las probabilidades de los 3 finalistas y umbral de alerta con cuadrícula de 100 clientes y curva ROC.
  - ¿Por qué se van?: factores del EDA 4.2 e interpretabilidad SHAP/LIME (Modelado 3.2).
  - Conclusiones: hallazgos (Modelado 3.3), limitaciones y trabajo futuro (Paper), biblioteca con el texto del libro.
  - Autoras: fotos, LinkedIn, GitHub y correo de Laura y Natalý.
  - Botones ES/EN y modo claro/oscuro arriba a la derecha; asistente sin límite de preguntas.
- Jupyter Book: """ + BOOK + """

Reglas:
- Responde en el idioma que te indiquen, en 2 a 5 frases, con markdown sencillo. Cita cifras solo si aparecen en estos hechos o en el contexto recibido.
- Si la pregunta no tiene que ver con el proyecto o el dashboard, dilo con amabilidad y sugiere qué se puede preguntar.
- Si el contexto no alcanza para responder, dilo y recomienda la sección del dashboard o del Jupyter Book donde buscar."""


def _chunks():
    """Fragmentos del texto del libro (en español) para recuperar contexto."""
    out = []
    for chapter, sections in a.CHAPTERS.items():
        for s in sections:
            text = re.sub(r'\s+', ' ', re.sub(r'[#*`|>$\\]', ' ', s['text'])).strip()
            for i in range(0, len(text), 700):
                out.append((f'{chapter} · {s["title"]}', text[i:i + 900]))
    return out


def _vectorizer(): return TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 4), strip_accents='unicode', lowercase=True, sublinear_tf=True)


@lru_cache(maxsize=1)
def index():
    # Preguntas de ejemplo: cada una se compara por separado y la entrada toma su mejor coincidencia.
    questions = [(i, q) for i, (qs, _, _) in enumerate(FAQ) for q in qs]
    qvec = _vectorizer(); qm = qvec.fit_transform([q for _, q in questions])
    chunks = _chunks()
    cvec = _vectorizer(); cm = cvec.fit_transform([t + ' ' + c for t, c in chunks])
    return qvec, qm, np.array([i for i, _ in questions]), cvec, cm, chunks


def retrieve(question):
    qvec, qm, owner, cvec, cm, chunks = index()
    qs = (qm @ qvec.transform([question]).T).toarray().ravel()
    fs = np.zeros(len(FAQ)); np.maximum.at(fs, owner, qs)
    cs = (cm @ cvec.transform([question]).T).toarray().ravel()
    return [(fs[i], i) for i in np.argsort(-fs)[:3]], [(cs[i], chunks[i]) for i in np.argsort(-cs)[:3]]


DATA_TERMS = [  # (patrón, columna, valor)
    (r'mes a mes|month.to.month|mensual', 'Contract', 'Month-to-month'), (r'un a[nñ]o|one.year|anual', 'Contract', 'One year'), (r'dos a[nñ]os|two.year|2 a[nñ]os', 'Contract', 'Two year'),
    (r'fibra|fiber', 'InternetService', 'Fiber optic'), (r'\bdsl\b', 'InternetService', 'DSL'), (r'sin internet|no internet', 'InternetService', 'No'),
    (r'cheque electr|electronic check', 'PaymentMethod', 'Electronic check'), (r'cheque por correo|mailed check', 'PaymentMethod', 'Mailed check'),
    (r'transferencia|bank transfer', 'PaymentMethod', 'Bank transfer (automatic)'), (r'tarjeta|credit card', 'PaymentMethod', 'Credit card (automatic)'),
    (r'adultos? mayor|senior', 'SeniorCitizen', 1), (r'\bmujer|\bfemale|\bwomen\b', 'gender', 'Female'), (r'\bhombre|\bmale\b|\bmen\b', 'gender', 'Male'),
    (r'sin pareja|without (a )?partner', 'Partner', 'No'), (r'con pareja|with (a )?partner', 'Partner', 'Yes'),
    (r'sin dependientes|without dependents', 'Dependents', 'No'), (r'con dependientes|with dependents', 'Dependents', 'Yes'),
    (r'sin soporte|no tech support|without tech support', 'TechSupport', 'No'), (r'con soporte|with tech support', 'TechSupport', 'Yes'),
    (r'sin seguridad|no online security|without online security', 'OnlineSecurity', 'No'), (r'con seguridad|with online security', 'OnlineSecurity', 'Yes'),
    (r'factura electr|paperless', 'PaperlessBilling', 'Yes'),
]


def _norm(text):
    import unicodedata
    return ''.join(c for c in unicodedata.normalize('NFD', text.lower()) if unicodedata.category(c) != 'Mn')


def data_answer(question, lang):
    """Calcula la tasa de abandono en entrenamiento para el perfil que menciona la pregunta; None si no aplica."""
    q = _norm(question)
    if not re.search(r'tasa|abandon|churn|cuant|porcentaje|rate|how many|se van|leave|riesgo|risk', q): return None
    d = a.RAW.loc[a.TRAIN_IDS]; desc = []
    used = set()
    for pattern, col, value in DATA_TERMS:
        if col not in used and re.search(_norm(pattern), q):
            d = d[d[col] == value]; used.add(col)
            name = a.LABELS.get(col, col)
            desc.append(f'{EN.get(name, name)}: {({1: "Yes"}).get(value, value)}' if lang == 'en' else f'{name}: {a.ES.get(value, {1: "Sí"}.get(value, value))}')
    m = re.search(r'(menos de|less than|under)\s+(\d+)\s*(mes|month)', q)
    if m: d = d[d.tenure < int(m.group(2))]; desc.append(('tenure < ' if lang == 'en' else 'antigüedad < ') + m.group(2))
    m = re.search(r'(mas de|more than|over)\s+(\d+)\s*(mes|month)', q)
    if m: d = d[d.tenure > int(m.group(2))]; desc.append(('tenure > ' if lang == 'en' else 'antigüedad > ') + m.group(2))
    if re.search(r'primer ano|first year', q) and 'antigüedad' not in ' '.join(desc): d = d[d.tenure <= 12]; desc.append('tenure ≤ 12' if lang == 'en' else 'antigüedad ≤ 12 meses')
    if not desc: return None
    if d.empty:
        return 'No training customers match that profile.' if lang == 'en' else 'Ningún cliente de entrenamiento tiene ese perfil.'
    n, k = len(d), int(d.target.sum())
    profile = ' · '.join(desc)
    if lang == 'en':
        return f'In the training set, customers with **{profile}** are {n:,}; {k:,} of them churned (**{k / n:.1%}**), versus 26.5 % overall. It is an observed association, not a cause. You can explore it in *Simulator → Risk profile* and *Data → 4 · Bivariate analysis*.'
    return f'En el conjunto de entrenamiento, los clientes con **{profile}** son {n:,}; de ellos abandonaron {k:,} (**{k / n:.1%}**), frente al 26.5 % general. Es una asociación observada, no una causa. Puedes explorarlo en *Simulador → Perfil de riesgo* y en *Datos → 4 · Análisis bivariado*.'


def local_answer(question, lang):
    faqs, chunks = retrieve(question)
    score, i = faqs[0]
    if score >= .5:
        return FAQ[i][2 if lang == 'en' else 1]
    computed = data_answer(question, lang)
    if computed: return computed
    best, (title, text) = chunks[0]
    if best >= .12:
        intro = 'This is what the book says (in Spanish), in' if lang == 'en' else 'Esto dice el libro en'
        return f'{intro} *{title}*:\n\n> {text[:380].strip()}…'
    return T('No encontré una respuesta a eso en el proyecto. Puedes preguntarme por los datos, los factores de abandono, los modelos, las métricas o cómo usar cada sección del dash.')


@lru_cache(maxsize=1)
def client():
    if not os.environ.get('ANTHROPIC_API_KEY'): return None
    try:
        import anthropic
        return anthropic.Anthropic()
    except Exception:
        return None


def llm_answer(question, history, lang):
    """Respuesta redactada por Claude con el contexto recuperado; None si no hay API o falla."""
    c = client()
    if c is None: return None
    import anthropic
    faqs, chunks = retrieve(question)
    context = '\n'.join(f'- {FAQ[i][1]}' for s, i in faqs if s > .08) + '\n' + '\n'.join(f'- [{t}] {x}' for s, (t, x) in chunks if s > .08)
    computed = data_answer(question, 'es')
    if computed: context += '\n- Cálculo sobre los datos de entrenamiento: ' + computed
    language = 'inglés' if lang == 'en' else 'español'
    messages = [{'role': m['role'], 'content': m['text']} for m in history[-6:]]
    messages.append({'role': 'user', 'content': f'<contexto>\n{context}\n</contexto>\n\nResponde en {language}.\n\nPregunta: {question}'})
    try:
        r = c.beta.messages.create(model=MODEL, max_tokens=1024, system=SYSTEM, messages=messages, cache_control={'type': 'ephemeral'},
                                   output_config={'effort': 'low'}, betas=['server-side-fallback-2026-07-01'], fallbacks='default')
    except (anthropic.APIStatusError, anthropic.APIConnectionError):
        return None
    if r.stop_reason == 'refusal': return None
    text = ''.join(b.text for b in r.content if b.type == 'text').strip()
    return text or None


def answer(question, history, lang):
    faqs, _ = retrieve(question)
    # Preguntas sobre perfiles concretos: la cifra calculada es más precisa que una respuesta general.
    computed = data_answer(question, lang) if faqs[0][0] < .8 else None
    return llm_answer(question, history, lang) or computed or local_answer(question, lang)


# ---------------------------------------------------------------- interfaz
def layout():
    return html.Div([
        dcc.Store(id='chat-log', storage_type='session', data=[]),
        html.Div([
            html.Div([html.Span(icon('signal'), className='chat-avatar'), html.Div([html.Strong(id='chat-title'), html.Small(id='chat-sub')]),
                      html.Button(icon('close'), id='chat-close', className='icon-btn', n_clicks=0)], className='chat-head'),
            html.Div([html.Span(id='chat-quota'), html.Button([icon('reset', 'sm'), html.Span(id='chat-new-t')], id='chat-new', className='chat-new', n_clicks=0)], className='chat-quota'),
            dcc.Loading(html.Div(id='chat-messages', className='chat-messages'), type='dot', color='#1BA7C4', parent_className='chat-loading'),
            html.Div([dcc.Input(id='chat-input', type='text', debounce=False, autoComplete='off', maxLength=400, n_submit=0),
                      html.Button(icon('send'), id='chat-send', className='chat-send', n_clicks=0)], className='chat-form')], id='chat-panel', className='chat-panel'),
        html.Button(icon('chat'), id='chat-fab', className='chat-fab', n_clicks=0)], id='chat', className='chat')


def bubble(role, text): return html.Div(dcc.Markdown(text, link_target='_blank'), className='msg ' + role)


def register(app):
    app.clientside_callback("function(a, b){ var o = document.getElementById('chat'); if(!o) return ''; var open = !o.classList.contains('open');"
                            " if (window.dash_clientside.callback_context.triggered[0].prop_id.indexOf('chat-close') === 0) open = false; return open ? 'chat open' : 'chat'; }",
                            Output('chat', 'className'), Input('chat-fab', 'n_clicks'), Input('chat-close', 'n_clicks'), prevent_initial_call=True)

    app.clientside_callback("function(c){ setTimeout(function(){ var m = document.getElementById('chat-messages'); if (m) m.scrollTop = m.scrollHeight; }, 60); return ''; }",
                            Output('chat-messages', 'title'), Input('chat-messages', 'children'))

    @app.callback(Output('chat-log', 'data'), Output('chat-input', 'value'),
                  Input('chat-send', 'n_clicks'), Input('chat-input', 'n_submit'), Input({'type': 'chat-sug', 'index': ALL}, 'n_clicks'),
                  State('chat-input', 'value'), State('chat-log', 'data'), State('lang', 'data'), prevent_initial_call=True)
    def ask(_send, _submit, sug_clicks, text, log, lang):
        set_lang(lang)
        trig = ctx.triggered_id
        if isinstance(trig, dict):
            if not any(sug_clicks): return no_update, no_update
            text = T(SUGGESTIONS[trig['index']])
        text = (text or '').strip()
        log = log or []
        if not text: return no_update, ''
        reply = answer(text, log, lang)
        return log + [{'role': 'user', 'text': text}, {'role': 'assistant', 'text': reply}], ''

    @app.callback(Output('chat-log', 'data', allow_duplicate=True), Input('chat-new', 'n_clicks'), prevent_initial_call=True)
    def new_chat(n): return [] if n else no_update

    @app.callback(Output('chat-messages', 'children'), Output('chat-quota', 'children'), Output('chat-title', 'children'), Output('chat-sub', 'children'),
                  Output('chat-input', 'placeholder'), Output('chat-new-t', 'children'), Input('chat-log', 'data'), Input('lang', 'data'))
    def render(log, lang):
        set_lang(lang)
        log = log or []
        asked = sum(m['role'] == 'user' for m in log)
        hello = T('¡Hola! Soy el asistente de Señal perdida. Pregúntame lo que quieras sobre el proyecto, sus datos o cualquier sección del dash: no hay límite de preguntas.')
        msgs = [bubble('assistant', hello)]
        if not log:
            msgs.append(html.Div([html.Button(T(s), id={'type': 'chat-sug', 'index': i}, className='chat-chip', n_clicks=0) for i, s in enumerate(SUGGESTIONS)], className='chat-chips'))
        msgs += [bubble(m['role'], m['text']) for m in log]
        quota = T('Preguntas ilimitadas · {n} en esta conversación').format(n=asked)
        mode = T('Respuestas con Claude') if client() else T('Respuestas del proyecto')
        return msgs, quota, T('Asistente Señal'), T('Datos: proyecto Telco Churn') + ' · ' + mode, T('Pregunta lo que quieras sobre el dash…'), T('Nueva conversación')
