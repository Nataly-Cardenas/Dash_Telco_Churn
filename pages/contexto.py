"""Sección 1 · Contexto: problema, pregunta, datos, metodología, limitaciones y autoras."""
import json
from functools import lru_cache

import dash
import numpy as np
import pandas as pd
from dash import Input, Output, State, callback, dcc, html

from src import data as a
from src.i18n import N, T, set_lang
from src.icons import icon
from src.ui import (ACTUALIZADO, BOOK, DASH_REPO, REPO, VERSION, card, cat, counter, eyebrow, grid, kpi, kpis, label,
                    note, page_head, subhead, table)

dash.register_page(__name__, path='/', name='Contexto', title='Contexto · Señal perdida', order=0)

FUENTE_CIFRA = 'https://hbr.org/2014/10/the-value-of-keeping-the-right-customers'
FLIP = [('target', N('Abandono y clasificación binaria'), N('El churn es la cancelación del servicio. Se estima para cada cliente la probabilidad P(abandono | x) y se separa a quienes se van (1) de quienes permanecen (0).')),
        ('chart', N('Análisis exploratorio'), N('Antes de modelar se mide la asociación de cada variable con el abandono: correlación punto-biserial, V de Cramér y Spearman, con corrección de Holm.')),
        ('block', N('Fuga de información'), N('Imputar, codificar, escalar y calibrar solo con datos de entrenamiento; si no, las métricas resultan optimistas.')),
        ('repeat', N('Validación cruzada anidada'), N('Un bucle interno elige hiperparámetros con TPE; uno externo mide el desempeño con datos que no intervinieron. Aquí, 5 repeticiones × 5 pliegues.')),
        ('sigma', N('Log-loss y calibración'), N('El log-loss castiga las probabilidades seguras y equivocadas. Platt ajusta una sigmoide para que un 30 % predicho se parezca a un 30 % observado.')),
        ('layers', N('Ensambles por región'), N('RegionalMAPEM pondera a 11 clasificadores con pesos distintos en cada región de clientes, estimados con MAP-EM y un prior de Dirichlet.')),
        ('scale', N('Comparación estadística'), N('t corregida de Nadeau-Bengio, permutaciones, Wilcoxon y Friedman-Nemenyi, con corrección de Holm y tamaño del efecto de Cliff.')),
        ('bulb', N('Interpretabilidad'), N('SHAP reparte cada predicción entre las variables. Describe cómo decide el modelo, no las causas del abandono.'))]
OBJECTIVES = [(N('Describir el problema y los datos'), N('Caracterizar el conjunto, su calidad y el desbalance de la variable objetivo.')),
              (N('Reservar un hold-out'), N('Separar el 20 % de los clientes con una partición estratificada antes de cualquier análisis.')),
              (N('Identificar los perfiles de abandono'), N('Analizar cada variable y su asociación con el abandono para encontrar los segmentos con más cancelaciones.')),
              (N('Comparar 11 benchmarks'), N('Optimizar cada modelo con el mismo presupuesto (30 evaluaciones de TPE) y calibrarlo con Platt dentro del bucle interno.')),
              (N('Proponer un modelo original'), N('Formular RegionalMAPEM, demostrar su convergencia e implementarlo desde cero con pruebas unitarias.')),
              (N('Validar con un protocolo pre-registrado'), N('Congelar hipótesis, métricas y pruebas antes de correr los experimentos definitivos.')),
              (N('Evaluar más allá de Telco'), N('Repetir la comparación en tres conjuntos de churn adicionales y estudiar robustez, calibración e incertidumbre.')),
              (N('Poner los modelos a disposición'), N('Publicar el Jupyter Book y este tablero con los artefactos finales para predecir en vivo.'))]
GROUPS = [(N('Identificador'), ['customerID']), (N('Demografía'), ['gender', 'SeniorCitizen', 'Partner', 'Dependents']),
          (N('Servicios'), ['PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']),
          (N('Cuenta'), ['tenure', 'Contract', 'PaperlessBilling', 'PaymentMethod', 'MonthlyCharges', 'TotalCharges']), (N('Objetivo'), ['Churn'])]
PIPELINE = [('table', N('ETL'), N('TotalCharges a número, categorías redundantes unificadas y cargo promedio mensual.')),
            ('chart', N('EDA'), N('Asociaciones con el abandono sobre los 5634 clientes de desarrollo.')),
            ('tree', N('11 benchmarks'), N('De KNN y Naive Bayes a LightGBM y CatBoost, todos calibrados con Platt.')),
            ('sliders', N('Optimización'), N('TPE con 30 evaluaciones por modelo y pliegue externo, minimizando el log-loss.')),
            ('repeat', N('Validación anidada'), N('5 repeticiones × 5 pliegues externos; 3 pliegues internos.')),
            ('scale', N('Comparación estadística'), N('223 contrastes con Holm, Friedman-Nemenyi y Cliff.')),
            ('layers', N('Modelo original'), N('RegionalMAPEM: pesos por región estimados con MAP-EM.')),
            ('flask', N('Evaluación ampliada'), N('3 conjuntos extra, robustez, conforme y hold-out abierto una vez.'))]


def people_row():
    leaving = {2, 7, 11, 16, 19}   # 5 de 20 ≈ 26.5 % de abandono
    return html.Div([html.Span(className='person' + (' leaving' if i in leaving else ''), style={'animationDelay': f'{(i % 5) * .9}s'}) for i in range(20)], className='people')


def flags(c):
    """Señales del perfil asociadas a más (▲) o menos (▼) abandono en el EDA."""
    out = []
    if c.Contract == 'Month-to-month': out.append('▲ ' + T('Contrato mes a mes'))
    if c.Contract == 'Two year': out.append('▼ ' + T('Contrato a dos años'))
    if c.InternetService == 'Fiber optic': out.append('▲ ' + T('Fibra óptica'))
    if c.PaymentMethod == 'Electronic check': out.append('▲ ' + T('Paga con cheque electrónico'))
    if c.tenure <= 12: out.append('▲ ' + T('Menos de un año como cliente'))
    if c.tenure > 48: out.append('▼ ' + T('Más de 4 años como cliente'))
    if c.TechSupport == 'No': out.append('▲ ' + T('Sin soporte técnico'))
    if c.OnlineSecurity == 'Yes': out.append('▼ ' + T('Con seguridad en línea'))
    return out[:4]


@lru_cache(maxsize=1)
def cover_rows():
    """30 clientes reales del hold-out, mezclando riesgo alto, medio y bajo según RegionalMAPEM."""
    p = a.predicciones_finales().query("evaluacion == 'holdout'").set_index('row_id').RegionalMAPEM
    rng = np.random.default_rng(7)
    picks = list(rng.choice(p.index[p >= .6], 12, replace=False)) + list(rng.choice(p.index[p.between(.3, .6)], 8, replace=False)) \
        + list(rng.choice(p.index[p < .3], 10, replace=False))
    rng.shuffle(picks)
    return [(int(r), float(p.loc[r])) for r in picks]


def cover_customers():
    out = []
    for r, prob in cover_rows():
        c = a.RAW.loc[r]
        out.append({'id': c.customerID, 'contract': cat(c.Contract), 'internet': cat(c.InternetService), 'payment': cat(c.PaymentMethod),
                    'tenure': int(c.tenure), 'monthly': float(c.MonthlyCharges), 'prob': round(prob, 3), 'churn': int(c.target), 'flags': flags(c)})
    return out


def phone():
    text = {'months': T('meses'), 'risk': {'alto': T('Riesgo alto'), 'medio': T('Riesgo medio'), 'bajo': T('Riesgo bajo')},
            'churned': T('En la realidad: abandonó'), 'stayed': T('En la realidad: permaneció'),
            'toast': {'alto': T('Alerta de retención enviada'), 'medio': T('Cliente en observación'), 'bajo': T('Cliente estable')}}
    return html.Div(html.Div([
        html.Div([html.Span('9:41'), html.Span([html.B(), html.B(), html.B(), html.B()], className='sig'), html.Span('5G', className='bat')], className='phone-status'),
        html.Div([
            html.Div([html.Span(T('TELCO · RADAR DE CLIENTES'), className='pc-app'), html.Span(T('EN VIVO'), className='pc-live')], className='between'),
            html.Div([html.Small(T('Cliente')), html.Strong(className='pc-id')], className='pc-row'),
            html.Div([html.Span(className='pc-contract pill'), html.Span(className='pc-internet pill'), html.Span(className='pc-payment pill')], className='pc-pills'),
            html.Div([html.Div([html.Small(T('Antigüedad')), html.Strong(className='pc-tenure')]), html.Div([html.Small(T('Cargo mensual')), html.Strong(className='pc-monthly')])], className='pc-grid'),
            html.Div([html.Small(T('Señales del perfil')), html.Ul(className='pc-flags')], className='pc-signals'),
            html.Div([html.Div(html.Div([html.Strong(className='ring-pct'), html.Small(T('prob. de abandono'))], className='ring-label'), className='ring'),
                      html.Div([html.Strong(className='pc-level'), html.Small(T('RegionalMAPEM · hold-out')), html.Small(className='pc-truth')], className='pc-verdict')], className='pc-risk'),
            html.Div([html.Span(className='toast-dot'), html.Span(className='toast-text')], className='toast')], className='phone-body')],
        id='phone', className='phone', **{'data-customers': json.dumps(cover_customers()), 'data-text': json.dumps(text)}), className='phone-stage')


def cover():
    r = a.telco('resumen_modelos.csv').set_index('modelo').loc['RegionalMAPEM']
    return html.Div([
        html.Canvas(id='net-canvas', className='net-canvas'),
        html.Div(className='cover-glow'),
        html.Div([
            html.Div([
                eyebrow('TELCO CUSTOMER CHURN · UNIVERSIDAD DEL NORTE'),
                html.H1([T('Señal'), html.Br(), html.Span(T('perdida'), className='glitch', **{'data-text': T('perdida')})], className='cover-title'),
                html.P([T('Uno de cada cuatro clientes '), html.B(T('cuelga para siempre')), T('. Descubre quiénes son, por qué se van y cómo anticiparlo con datos.')], className='cover-lead'),
                html.Div([
                    html.Div([counter(7043), html.Small(T('clientes analizados'))]),
                    html.Div([counter(26.5, 1, ' %'), html.Small(T('abandonan el servicio'))], className='hot'),
                    html.Div([counter(12), html.Small(T('modelos finales'))]),
                    html.Div([counter(round(float(r.log_loss_media), 3), 3), html.Small(T('log-loss del modelo propio'))])], className='cover-stats'),
                html.Div([html.A([T('Entrar al dashboard'), ' →'], href='#problema', className='btn primary'),
                          dcc.Link([T('Predecir un cliente'), icon('arrow', 'sm')], href='/modelos', className='btn ghost')], className='cover-cta'),
                html.Div([icon('cursor'), html.Span([T('Los puntos rojos son clientes a punto de irse: '), html.B(T('pasa el cursor (o el dedo) sobre ellos para retenerlos.'))]),
                          html.Span([T('Perdidos: '), html.B('0', id='cover-lost'), ' · ', T('Retenidos por ti: '), html.B('0', id='cover-saved')], className='score')], className='cover-hint'),
            ], className='cover-text'),
            phone()], className='cover-inner'),
        html.Div('Laura S. Rivera Figueredo · Natalý A. Cárdenas Izaquita', className='cover-foot')], className='cover')


def problema():
    return html.Div([
        html.Div([html.Div([eyebrow(T('CONTEXTO DEL PROBLEMA')), html.H2([T('1 de cada 4 clientes '), html.Span(T('se desconecta'), className='txt-hot')]),
                            html.P(T('Cada cancelación es un ingreso recurrente que se pierde, y conseguir un cliente nuevo cuesta entre 5 y 25 veces más que conservar uno '
                                     'que ya existe. Anticipar quién está por irse permite dirigir las acciones de retención a los clientes en riesgo en lugar de a toda la base.')),
                            html.P([T('Fuente de la cifra: '), html.A('Gallo, A. (2014). The value of keeping the right customers. Harvard Business Review.', href=FUENTE_CIFRA, target='_blank')],
                                   className='muted small'), people_row()], className='hero-text'),
                  kpis(kpi(T('CLIENTES'), 7043, T('una fila por cliente')), kpi(T('ABANDONOS'), 1869, T('cancelaron en el último mes')),
                       kpi(T('TASA DE ABANDONO'), 26.5, T('la clase minoritaria'), 'hot', 1, ' %'), kpi(T('VARIABLES'), 19, T('predictoras candidatas')), cls='k2')], className='hero', id='problema'),
        subhead(T('Pregunta de investigación')),
        card(html.P(T('¿Combinar 11 clasificadores calibrados con pesos que cambian según la región del espacio de clientes da probabilidades de abandono '
                      'mejores (menor log-loss) que el mejor de esos clasificadores por sí solo?'), className='big-text'),
             html.P(T('Respuesta corta: no en Telco. RegionalMAPEM empata con la logística con splines (0,41127 frente a 0,41130) y supera a 7 de los 11 benchmarks, '
                      'pero ninguna prueba del protocolo confirma la diferencia. El detalle está en la sección Modelos.'), className='insight'), cls='glow'),
        subhead(T('Objetivo general')),
        card(html.P(T('Identificar a los clientes de una empresa de telecomunicaciones con mayor riesgo de abandonar el servicio y los factores asociados a ese abandono, '
                      'construyendo y comparando de forma reproducible modelos de clasificación, incluido un modelo propio, que entreguen probabilidades de abandono bien calibradas.'), className='big-text')),
        subhead(T('Objetivos específicos')),
        html.Div([html.Div([html.Span(f'{i + 1:02d}', className='obj-n'), html.H4(T(t)), html.P(T(s))], className='obj') for i, (t, s) in enumerate(OBJECTIVES)], className='objectives')])


def datos():
    variables = pd.DataFrame([{'Grupo': T(g), 'Variable': v, 'Significado': label(v) if v in a.LABELS else T('Código del cliente') if v == 'customerID' else T('Abandono durante el último mes'),
                               'Tipo': T('Numérica') if v in a.NUMERIC + ['SeniorCitizen'] else T('Identificador') if v == 'customerID' else T('Categórica'),
                               'Papel': T('Objetivo') if v == 'Churn' else T('No predictor') if v == 'customerID' else T('Predictora')} for g, vs in GROUPS for v in vs])
    ficha = pd.DataFrame([
        (T('Fuente'), T('IBM, conjunto de muestra publicado en Kaggle (blastchar/telco-customer-churn)')),
        (T('Periodo'), T('Una sola fotografía de corte transversal; el archivo no trae fechas')),
        (T('Licencia'), T('La de la publicación en Kaggle; datos de demostración de IBM para uso educativo')),
        (T('Observaciones'), T('7043 clientes: 5634 de desarrollo y 1409 de hold-out')),
        (T('Variables'), T('21 columnas: 19 predictoras, el identificador y la respuesta')),
        (T('Variable objetivo'), T('Churn (Yes/No): canceló en el último mes; 26,5 % Yes')),
        (T('Tipo de tarea'), T('Clasificación binaria con salida probabilística')),
        (T('Faltantes'), T('11 vacíos de TotalCharges, todos con tenure = 0; se imputan con 0'))], columns=[T('Campo'), T('Valor')])
    return html.Div([subhead(T('Descripción de los datos'), T('Ficha resumida de DATA_STATEMENT.md y diccionario de variables.')),
                     grid(card(html.H3(T('Ficha del conjunto'), className='chart-title'), table(ficha, page_size=8, wrap=True)),
                          card(html.H3(T('Diccionario de variables'), className='chart-title'), table(variables, page_size=8, filter_action='native')))])


def metodologia():
    return html.Div([subhead(T('Metodología'), T('Del archivo original al veredicto del modelo propio.')),
                     card(html.Div([html.Div(className='flow-line'), *[html.Div([html.Span(icon(i), className='st-icon'), html.H4(T(t)), html.P(T(s))], className='station',
                                                                                  style={'animationDelay': f'{k * .12}s'}) for k, (i, t, s) in enumerate(PIPELINE)]], className='pipeline wide')),
                     subhead(T('Marco teórico'), T('Pasa el cursor (o toca) cada tarjeta para voltearla.')),
                     html.Div([html.Div(html.Div([html.Div([html.Span(icon(i), className='flip-icon'), html.H4(T(t))], className='flip-front'), html.Div([html.H4(T(t)), html.P(T(s))], className='flip-back')],
                                                 className='flip-inner'), className='flip', tabIndex=0) for i, t, s in FLIP], className='flips')])


def cierre():
    limits = [N('Un solo operador ficticio y una sola fotografía en el tiempo: los resultados no se trasladan sin validar a otra población.'),
              N('En Telco las regiones no aportan frente a pesos globales: el modelo propio se comporta como un ensamble ponderado.'),
              N('RegionalMAPEM necesita a los 11 expertos para predecir, lo que encarece la inferencia.'),
              N('Sin costos de retención explícitos, el umbral de alerta no se traduce en dinero.')]
    ethics = [N('gender y SeniorCitizen están entre las variables. Su peso SHAP es bajo, pero no se hizo un análisis formal de equidad entre grupos.'),
              N('El riesgo predicho no debe usarse para negar beneficios ni cambiar condiciones de un cliente sin revisión humana.'),
              N('Las atribuciones SHAP describen al modelo, no causas del abandono.')]
    people = [('laura.jpg', 'Laura S. Rivera Figueredo', 'sriveral@uninorte.edu.co', 'https://github.com/lauraformore', 'https://www.linkedin.com/in/laura-s-rivera-figueredo-862451315/'),
              ('nataly.jpg', 'Natalý A. Cárdenas Izaquita', 'nizaquita@uninorte.edu.co', 'https://github.com/Nataly-Cardenas', 'https://www.linkedin.com/in/n%C3%A1taly-c%C3%A1rdenas-iz%C3%A1quita')]
    res = [('code', T('Código y notebooks'), T('Repositorio Telco_Churn_Entregable'), REPO), ('sliders', T('Este dashboard'), T('Repositorio Dash_Telco_Churn'), DASH_REPO),
           ('table', T('Conjunto de datos'), 'Kaggle · Telco Customer Churn (IBM)', 'https://www.kaggle.com/datasets/blastchar/telco-customer-churn'), ('book', 'Jupyter Book', T('Informe técnico completo'), BOOK)]
    return html.Div([
        grid(card(eyebrow(T('LIMITACIONES')), html.Ul([html.Li(T(x)) for x in limits])),
             card(eyebrow(T('ÉTICA Y SESGO')), html.Ul([html.Li(T(x)) for x in ethics]))),
        subhead(T('Autoras')),
        html.Div([html.Div([html.Div(html.Img(src=dash.get_asset_url('img/' + photo), alt=T('Foto de {n}').format(n=n)), className='avatar'), html.H2(n),
                            html.P(T('Ciencia de Datos · Universidad del Norte'), className='muted'),
                            html.Div([html.A([icon('linkedin', 'sm'), 'LinkedIn'], href=li, target='_blank', className='chip-link linkedin'),
                                      html.A([icon('code', 'sm'), 'GitHub'], href=g, target='_blank', className='chip-link'),
                                      html.A([icon('mail', 'sm'), m], href='mailto:' + m, className='chip-link')], className='author-links')],
                           className='author') for photo, n, m, g, li in people], className='authors'),
        html.P(T('Departamento de Matemáticas, Física y Ciencia de Datos · Universidad del Norte · Barranquilla, Colombia'), className='center muted'),
        grid(*[html.A([html.H4([icon(i), t]), html.P(s)], href=u, target='_blank', className='res') for i, t, s, u in res], cols=4),
        html.P(T('Versión {v} · actualizado el {f} · referencias completas en el Jupyter Book.').format(v=VERSION, f=ACTUALIZADO), className='center muted version')])


def contenido():
    return html.Div([cover(),
                     html.Div([page_head(T('Contexto'), T('Por qué importa anticipar el abandono, qué datos usamos y cómo llegamos al modelo propio.'), 'intro'),
                               problema(), datos(), metodologia(), cierre()], className='page contexto')])


def layout(**_):
    return html.Div([dcc.Store(id='contexto-montaje', data=1), html.Div(id='contexto-contenido')])


@callback(Output('contexto-contenido', 'children'), Input('lang', 'data'), Input('contexto-montaje', 'data'))
def render(lang, _):
    set_lang(lang)
    return contenido()
