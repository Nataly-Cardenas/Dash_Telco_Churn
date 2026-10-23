"""Telco Atlas: recorrido del Jupyter Book de Laura Rivera y Natalý Cárdenas."""
import json
import os
import re
from functools import lru_cache
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, dcc, html, dash_table, Input, Output, State, no_update
from flask import send_from_directory
from sklearn.metrics import roc_curve, precision_recall_curve, roc_auc_score, average_precision_score
from scipy.stats import chi2_contingency
import analytics as a

BOOK = 'https://nataly-cardenas.github.io/Telco_Churn_Entregable/'
BLUE, RED, TEAL = '#5379e7','#e77470','#48b8a4'
COLORS = {'Permanece':BLUE,'Abandona':RED}
NAV = [('overview','01','Panorama'),('context','02','Proyecto y objetivos'),('quality','03','Datos y calidad'),('univariate','04','Análisis univariado'),('bivariate','05','Abandono por segmento'),('multivariate','06','Relaciones multivariadas'),('preprocessing','07','Preprocesamiento'),('models','08','Comparador de modelos'),('optimization','09','Optimización'),('calibration','10','Calibración y umbrales'),('statistics','11','Evidencia estadística'),('explain','12','SHAP y LIME'),('conclusions','13','Conclusiones'),('explorer','14','Explorador de datos'),('reader','15','Libro en detalle'),('authors','16','Autoras y fuentes')]
TITLES = {k:t for k,_,t in NAV}
app = Dash(__name__, suppress_callback_exceptions=True, title='Telco Atlas · Customer Churn', meta_tags=[{'name':'viewport','content':'width=device-width, initial-scale=1'}])
server = app.server
app.index_string = app.index_string.replace('<html>', '<html lang="es">')

@server.route('/evidence/<path:filename>')
def evidence(filename):
    return send_from_directory(a.DATA/'interpretabilidad', filename)

@server.route('/healthz')
def health(): return {'status':'ok','clients':len(a.RAW),'experiments':len(a.SUMMARY)}

def md(text): return dcc.Markdown(text, mathjax=True, className='prose')
def card(*children, cls=''): return html.Section(list(children), className='card '+cls)
def note(title,text): return html.Div([html.Strong(title),html.P(text)],className='note')
def kpi(label,value,sub='',accent=''): return html.Div([html.Span(label,className='k-label'),html.Strong(value,className='k-value '+accent),html.Small(sub)],className='kpi')
def grid(*kids): return html.Div(list(kids),className='grid2')
def options(values, labels=None): return [{'label':(labels or {}).get(v,v),'value':v} for v in values]
def dropdown(id,values,value,labels=None,multi=False): return dcc.Dropdown(options(values,labels),value,id=id,clearable=False,multi=multi)
def label(text,control): return html.Div([html.Label(text),control],className='field')
def table(d,id=None,**kw):
    args={'id':id} if id else {}
    return dash_table.DataTable(data=d.round(5).replace({np.nan:None}).to_dict('records'),columns=[{'name':c,'id':c} for c in d.columns],page_size=kw.pop('page_size',12),sort_action='native',filter_action='native',style_table={'overflowX':'auto'},style_header={'backgroundColor':'#202b43','color':'#dce5f7','border':'none','fontWeight':'600'},style_cell={'backgroundColor':'#151f32','color':'#c5cee0','border':'none','borderBottom':'1px solid #29334a','padding':'12px','textAlign':'left','fontFamily':'inherit','fontSize':12,'maxWidth':320,'overflow':'hidden','textOverflow':'ellipsis'},**args,**kw)
def fig(f,title=None,height=340):
    f.update_layout(template='plotly_dark',paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)',font={'family':'Inter, system-ui, sans-serif','color':'#aab8d1','size':12},colorway=[BLUE,RED,TEAL,'#c7a3e9'],margin={'l':25,'r':22,'t':25,'b':35},height=height,legend={'orientation':'h','y':1.12,'title_text':''})
    f.update_xaxes(gridcolor='#263149',zerolinecolor='#34415a');f.update_yaxes(gridcolor='#263149',zerolinecolor='#34415a')
    return dcc.Graph(figure=f,style={'height':height,'minHeight':height,'width':'100%'},config={'displaylogo':False,'responsive':True,'modeBarButtonsToRemove':['lasso2d','select2d']})
def chart(title,f,interpretation='',height=340): return card(html.Div([html.H3(title),html.Span('INTERACTIVO',className='chart-tag')],className='card-head'),fig(f,height=height),html.P(interpretation,className='interp'))
def source(chapter): return html.A('Consultar capítulo ↗',href=BOOK+chapter+'.html',target='_blank',className='text-link')
def intro(title,desc,chapter=None): return html.Div([html.Div([html.Span('TELCO CUSTOMER CHURN / '+TITLES.get(title,title).upper(),className='eyebrow'),source(chapter) if chapter else None],className='between'),html.H1(TITLES.get(title,title)),html.P(desc,className='lead')],className='page-head')
def chapters_details(chapter,pattern=None):
    chunks=a.CHAPTERS[chapter]
    if pattern: chunks=[c for c in chunks if re.search(pattern,c['title'],re.I)]
    return html.Div([html.Details([html.Summary(c['title']),md(c['text'])],className='chapter') for c in chunks])
def empty(): return note('Sin clientes en este segmento','Amplía los filtros para volver a calcular las visualizaciones.')
def filtered_info(d): return html.Div([html.Span('●',className='live-dot'),f' {len(d):,} clientes en la selección · análisis descriptivo'],className='scope')

app.layout=html.Div([
    dcc.Location(id='url'),dcc.Store(id='presentation',data=False),
    html.A('Saltar al contenido',href='#content',className='skip'),
    html.Aside([
        dcc.Link([html.Div('t',className='brand-icon'),html.Div([html.B('telco',className='wordmark'),html.Span('ATLAS',className='brand-sub')])],href='#overview',className='brand'),
        html.Div('RETENCIÓN · DATOS · DECISIONES',className='side-caption'),
        html.Div('RECORRIDO DEL PROYECTO',className='nav-label'),
        html.Nav([dcc.Link([html.Span(n,className='nav-number'),html.Span(t)],href='#'+k,id='nav-'+k,className='nav-link') for k,n,t in NAV]),
        html.Div([html.Div('Un proyecto de',className='muted'),html.B('Laura Rivera & Natalý Cárdenas'),html.Small('Universidad del Norte'),html.A('Abrir Jupyter Book ↗',href=BOOK,target='_blank',className='book-button')],className='side-footer')],className='sidebar'),
    html.Div([
        html.Header([html.Div([html.Span('LABORATORIO DE ANALÍTICA',className='eyebrow'),html.Span(' /  Customer Churn',className='muted')]),html.Div([html.Span('Datos originales',className='status'),html.Button('Modo presentación',id='present-btn',className='button secondary',n_clicks=0)],className='header-actions')],className='topbar'),
        html.Div([label('Universo',dropdown('split',['train','test','all'],'train',{'train':'Entrenamiento · 5.634','test':'Test histórico · 1.409','all':'Dataset completo · 7.043'})),label('Contrato',dropdown('contract',['Todos']+sorted(a.RAW.Contract.unique()),'Todos')),label('Internet',dropdown('internet',['Todos']+sorted(a.RAW.InternetService.unique()),'Todos')),label('Antigüedad · meses',dcc.RangeSlider(0,72,1,value=[0,72],marks={0:'0',24:'24',48:'48',72:'72'},id='tenure',tooltip={'placement':'bottom'})),html.Button('Restablecer',id='reset',className='button secondary')],className='filters',id='filters'),
        html.Main(id='content',tabIndex=-1),html.Footer([html.Span('Telco Atlas · Del análisis a la explicación'),html.Span('Fuente: Jupyter Book · Rivera & Cárdenas'),html.Div(id='pager')],className='footer')],className='workspace'),
    dcc.Download(id='download')],id='shell')


def overview(d):
    if d.empty:return empty()
    rate=d.target.mean();counts=d.Estado.value_counts()
    pie=go.Figure(go.Pie(labels=counts.index,values=counts.values,hole=.77,marker_colors=[COLORS[c] for c in counts.index],textinfo='none',sort=False))
    pie.add_annotation(text=f'<b>{rate:.1%}</b><br>abandono',x=.5,y=.5,showarrow=False,font_size=23)
    g=a.category_rates(d,'Contract').sort_values('tasa')
    bar=px.bar(g,x='tasa',y='Contract',orientation='h',text=g.tasa.map('{:.1%}'.format),color_discrete_sequence=[BLUE]);bar.update_xaxes(tickformat='.0%',title='Tasa de abandono');bar.update_yaxes(title='')
    bins=pd.cut(d.tenure,[-1,12,24,36,48,60,72],labels=['0–12','13–24','25–36','37–48','49–60','61–72'])
    g2=d.groupby(bins,observed=False).target.agg(tasa='mean',n='size').reset_index()
    line=px.line(g2,x='tenure',y='tasa',markers=True,custom_data=['n']);line.update_traces(line_color=TEAL,fill='tozeroy',fillcolor='rgba(72,184,164,.08)',hovertemplate='%{x} meses<br>Abandono: %{y:.1%}<br>n=%{customdata[0]}<extra></extra>');line.update_yaxes(tickformat='.0%',title='Tasa de abandono');line.update_xaxes(title='Antigüedad (meses)')
    high=g.loc[g.tasa.idxmax()]
    return html.Div([intro('overview','Entender quién abandona, comparar cómo lo modelamos y explicar qué podemos concluir. Un recorrido interactivo por todo el proyecto.','intro'),
        html.Div([html.Div([html.Span('LA PREGUNTA DEL PROYECTO',className='eyebrow'),html.H2(['¿Qué señales acompañan',html.Br(),'la pérdida de un cliente?']),html.P('Explora los perfiles de suscripción y la evidencia detrás de los modelos de abandono.'),dcc.Link('Comenzar el recorrido →',href='#context',className='button')]),html.Div([html.Span('7 × 4 × 4',className='hero-formula'),html.P('familias · balanceos · optimizadores'),html.Div([html.Span('112 configuraciones'),html.Span('1.680 unidades externas')],className='pills')],className='hero-right')],className='hero'),
        filtered_info(d),html.Div([kpi('CLIENTES',f'{len(d):,}','Universo seleccionado'),kpi('TASA DE ABANDONO',f'{rate:.2%}',f'{d.target.sum():,} clientes abandonan','coral'),kpi('ANTIGÜEDAD MEDIA',f'{d.tenure.mean():.1f} meses','Tiempo en la compañía'),kpi('CARGO MENSUAL MEDIO',f'{d.MonthlyCharges.mean():.2f}','Importe en unidades del dataset')],className='kpis'),
        grid(chart('Balance de clientes',pie,'La exactitud por sí sola no mide la detección de abandono: una regla de permanencia ignora todos los positivos.'),chart('El contrato marca diferencias',bar,f'En esta selección, {high.Contract} tiene la mayor tasa observada: {high.tasa:.1%} (n={int(high.n):,}). Asociación descriptiva.')),
        grid(chart('Abandono según antigüedad',line,'Cada punto resume un grupo de meses. Pasa el cursor para ver su tamaño; los grupos pequeños pueden ser inestables.'),card(html.Span('LECTURA DEL EXPERIMENTO',className='eyebrow'),html.H2('0,6384',className='big-stat'),html.H3('F1 medio de RF–SMOTE'),html.P('± 0,0227 entre cinco folds externos, realización principal. El F1 de XGBoost–SMOTE–Random es 0,6380.'),note('Una diferencia pequeña','El ranking es exploratorio. El test histórico ya había sido observado y aporta evidencia suplementaria.'),dcc.Link('Comparar las 112 configuraciones →',href='#models',className='text-link'))),
        note('Cómo usar los filtros','Los filtros superiores modifican el análisis descriptivo y el explorador. Los resultados publicados de los modelos conservan sus poblaciones y particiones originales.')])


def context(d):
    return html.Div([intro('context','La pregunta de negocio, el alcance de la clasificación y la ruta que conecta el libro.','intro'),
        grid(card(html.Span('PROBLEMA',className='eyebrow'),html.H2('Detectar abandono con información de suscripción'),md('Cada fila representa un cliente. **Churn = Yes** indica que abandonó durante el último mes observado. El proyecto relaciona servicios, datos de cuenta y características demográficas con ese resultado.\n\nUna mejor detección puede orientar futuras acciones de retención, pero el dataset no permite medir el efecto de una campaña.')),card(html.Span('OBJETIVO GENERAL',className='eyebrow'),html.H2('Evaluar modelos que generalicen'),html.P('Partir de una regresión logística, comparar alternativas bajo un protocolo reproducible y controlar la fuga de información.'),html.Ul([html.Li(t) for t in ['Entender estructura, calidad y asociaciones en train.','Ajustar transformaciones dentro de cada partición.','Comparar modelos con métricas del abandono.','Interpretar resultados, incertidumbre y límites.']]))),
        card(html.H3('Una ruta de cuatro etapas'),html.Div([html.Div([html.Span(n,className='step-num'),html.H3(t),html.P(s)]) for n,t,s in [('01','Explorar','Calidad, distribuciones y relaciones en entrenamiento.'),('02','Preparar','Reglas deterministas, codificación y escalado sin fuga.'),('03','Comparar','Validación anidada y presupuesto de búsqueda común.'),('04','Explicar','Calibración, estadística, SHAP y LIME.')]],className='steps')),
        note('Alcance temporal','Los importes y la antigüedad corresponden al corte observado. Distinguir churners en ese corte no equivale a demostrar predicción prospectiva del próximo mes.'),
        card(html.H3('Conceptos para explicar el libro'),md('**Clasificación binaria:** asignar permanencia (0) o abandono (1).\n\n**Desbalance:** los abandonos son aproximadamente el 27 %; acertar muchas permanencias puede ocultar una detección pobre.\n\n**Generalización:** desempeño en observaciones que no entrenaron ni seleccionaron el modelo.\n\n**Fuga de información:** aprender decisiones de preparación o parámetros con datos reservados.\n\n**Asociación y causalidad:** una diferencia de tasas o una atribución del modelo no demuestra que cambiar una variable evite el abandono.'))])


def quality(d):
    missing=d[a.NUMERIC+a.CATEGORICAL].isna().sum().rename_axis('Variable').reset_index(name='Vacíos')
    return html.Div([intro('quality','Del CSV a una población trazable: tipos, duplicados, faltantes y partición histórica.','EDA_'),filtered_info(d),
        html.Div([kpi('FILAS ORIGINALES','7.043','Una fila por cliente'),kpi('DESARROLLO','5.634','80 % · índices preservados'),kpi('TEST HISTÓRICO','1.409','20 % · evidencia suplementaria'),kpi('VACÍOS ORIGINALES','11','TotalCharges · tenure = 0')],className='kpis'),
        grid(card(html.H3('Auditoría de la selección'),table(pd.DataFrame({'Control':['Clientes','Identificadores repetidos','Filas repetidas (columnas originales)','TotalCharges vacío','Vacío con tenure distinto de 0'],'Resultado':[len(d),int(d.customerID.duplicated().sum()),int(d[a.RAW.columns[:21]].duplicated().sum()),int(d.TotalCharges.isna().sum()),int((d.TotalCharges.isna() & (d.tenure!=0)).sum())]})),html.P('TotalCharges llega como texto: los espacios se convierten a NaN. En el preprocesamiento se sustituyen por 0 tras verificar antigüedad cero.',className='interp')),card(html.H3('Ausencias por variable'),table(missing[missing['Vacíos']>0] if missing['Vacíos'].sum() else missing.head(3)),note('El faltante tiene contexto','Son clientes nuevos sin facturación acumulada; no se les asigna un cargo medio ficticio. Hay 8 casos en train y 3 en test.'))),
        card(html.H3('Diccionario de variables'),table(pd.DataFrame([{'Variable':c,'Significado':a.LABELS.get(c,'Identificador' if c=='customerID' else 'Abandono observado'),'Tipo':'Numérica' if c in a.NUMERIC else 'Categórica / binaria','Rol':'Objetivo' if c=='Churn' else 'Integridad, no predictor' if c=='customerID' else 'Candidato predictor'} for c in a.RAW.columns[:21]]))),
        note('Separación del EDA','La selección inicial usa entrenamiento. Cambiar al dataset completo o al test sirve para inspección descriptiva; no modifica ni vuelve a validar los modelos publicados.'),chapters_details('EDA_',r'ETL|División|faltantes|Revisión')])


def univariate(d):
    return html.Div([intro('univariate','Explora la forma de cada variable y compara frecuencias, dispersión y valores extremos.','EDA_'),filtered_info(d),card(label('Variable',dropdown('uni-var',a.NUMERIC+a.CATEGORICAL,'tenure',a.LABELS))),html.Div(id='uni-output'),note('Lectura de extremos','Los clientes con importes o antigüedad extremos no se eliminan automáticamente: un valor raro puede ser válido. Los gráficos usan todas las filas de la selección.'),chapters_details('EDA_',r'univariado|numéricas|categóricas|Outliers')])

def bivariate(d):
    return html.Div([intro('bivariate','¿Qué cambia entre quienes permanecen y quienes abandonan? Compara tasas y distribuciones.','EDA_'),filtered_info(d),card(label('Variable del segmento',dropdown('bi-var',a.CATEGORICAL+a.NUMERIC,'Contract',a.LABELS))),html.Div(id='bi-output'),note('Interpretación estadística','Una tasa necesita su denominador. Las pruebas del libro controlan comparaciones múltiples con Holm; la significancia no mide utilidad predictiva ni causalidad.'),chapters_details('EDA_',r'bivariado|objetivo vs')])

def multivariate(d):
    if d.empty:return empty()
    corr=d[a.NUMERIC+['target']].corr(method='spearman')
    f=px.imshow(corr,text_auto='.2f',zmin=-1,zmax=1,color_continuous_scale=[[0,BLUE],[.5,'#172237'],[1,RED]],labels={'color':'ρ Spearman'})
    return html.Div([intro('multivariate','La relación conjunta entre antigüedad, cargos y servicios ayuda a detectar redundancias.','EDA_'),filtered_info(d),grid(chart('Correlaciones de Spearman',f,'Spearman compara rangos. TotalCharges acumula antigüedad y tarifa; una correlación alta sugiere información compartida, no una regla causal.'),card(html.H3('Servicios que comparten información'),html.P('«No internet service» y «No phone service» repiten información de InternetService y PhoneService. El pipeline colapsa esas categorías conservando los indicadores originales.'),html.H3('Tres precauciones'),html.Ul([html.Li('Correlación marginal y coeficiente condicionado pueden tener signos diferentes.'),html.Li('No comparar magnitudes de coeficientes numéricos estandarizados con indicadores 0/1 sin contexto.'),html.Li('Evaluar el aporte con ablación en CV, en lugar de borrar por un test aislado.')]))),
        card(html.H3('Exploración de dos variables'),grid(label('Eje X',dropdown('scatter-x',a.NUMERIC,'tenure',a.LABELS)),label('Eje Y',dropdown('scatter-y',a.NUMERIC,'MonthlyCharges',a.LABELS))),html.Div(id='scatter-output')),chapters_details('EDA_',r'multivariado')])


def preprocessing(d):
    steps=[('1','Objetivo e identificador','Churn: Yes/No → 1/0. customerID se utiliza para integridad, no como predictor.'),('2','Reglas deterministas','TotalCharges numérico; blancos con tenure = 0 → 0. Colapsar categorías de ausencia de servicio.'),('3','Variable derivada','AvgChargePerMonth = TotalCharges / tenure. Si tenure = 0, usar MonthlyCharges para evitar 0/0.'),('4','Representación','One-Hot Encoding y StandardScaler ajustados solo en entrenamiento de cada fold.'),('5','Balanceo interno','Sin balanceo, pesos de clase, SMOTE o ADASYN. Nunca remuestrear validación o test.'),('6','Estimador','Ajustar el clasificador y evaluar sobre el fold reservado.')]
    abl=pd.DataFrame({'Retirar':['gender','PhoneService','Ambas','PhoneService + MultipleLines','AvgChargePerMonth'],'ΔAUC aproximado':[.0002,.0002,.0005,-.0006,.0006]})
    return html.Div([intro('preprocessing','Cada decisión se ajusta al contexto del dato y cada parámetro aprendido se estima dentro del entrenamiento.','PREPROCESAMIENTO'),card(html.H3('Pipeline de preparación'),html.Div([html.Div([html.Span(n,className='step-num'),html.H3(t),html.P(s)],className='pipeline-step') for n,t,s in steps],className='pipeline')),
        grid(chart('Ablación de la regresión logística',px.bar(abl,x='ΔAUC aproximado',y='Retirar',orientation='h',color_discrete_sequence=[TEAL]),'Diferencias pareadas por CV en train, redondeadas según el texto del capítulo. Cambios muy pequeños; no prueban irrelevancia causal.'),card(html.H3('Controles y línea base'),md('**Imputación:** AUC de CV ≈ 0,8459 con cero y 0,8460 con mediana más indicador. La diferencia es despreciable frente a la dispersión entre folds.\n\n**Etiquetas permutadas:** AUC ≈ 0,49. **DummyClassifier:** AUC ≈ 0,50.\n\n**Baseline en test:** AUC 0,8419, Recall 0,7861, Precision 0,5052 y Accuracy 0,7388. Son resultados de la etapa inicial; no sustituyen la evidencia anidada del modelado avanzado.\n\n**Redundancia:** AvgChargePerMonth tiene correlación ≈ 0,9963 con MonthlyCharges. Su utilidad incremental debe evaluarse.'))),
        note('Diferencia entre etapas','El capítulo inicial trata SeniorCitizen como categórica; el protocolo avanzado la escala entre las numéricas y usa una representación de 24 columnas. Se conserva esta distinción al explicar cada etapa.'),note('Limitación de SMOTE / ADASYN','Se aplicaron después del one-hot: pueden producir indicadores fraccionarios. Esta limitación forma parte del método publicado y no se cambia retrospectivamente.'),chapters_details('PREPROCESAMIENTO')])


def models(d):
    return html.Div([intro('models','Las 112 configuraciones bajo el mismo protocolo: siete familias × cuatro balanceos × cuatro optimizadores.','MODELADO_AVANZADO'),
        html.Div([kpi('FAMILIAS','7','Desde KNN hasta XGBoost'),kpi('FOLDS EXTERNOS / INTERNOS','5 / 3','Validación anidada estratificada'),kpi('REALIZACIONES','3','Sensibilidad algorítmica'),kpi('CRITERIO INTERNO','F1 positivo','24 evaluaciones por búsqueda')],className='kpis'),
        card(html.Div([label('Familia',dropdown('model-family',['Todas']+list(a.MODELS),'Todas',a.MODELS)),label('Balanceo',dropdown('model-balance',['Todos']+list(a.BALANCES),'Todos',a.BALANCES)),label('Métrica',dropdown('model-metric',['f1_positive','recall_positive','precision_positive','roc_auc','accuracy'],'f1_positive',{'f1_positive':'F1 positivo','recall_positive':'Recall','precision_positive':'Precision','roc_auc':'ROC-AUC','accuracy':'Accuracy'}))],className='model-controls')),
        html.Div(id='models-output'),note('Cómo leer esta comparación','Media ± desviación estándar de cinco folds externos en la realización 0. El máximo entre 112 configuraciones tiene sesgo de selección: sirve para describir candidatos exploratorios. Las semillas adicionales no son datasets independientes.'),chapters_details('MODELADO_AVANZADO',r'1\.1|1\.2|1\.3|1\.4|1\.7|1\.9')])


def optimization(d):
    return html.Div([intro('optimization','Una comparación de búsquedas con igual presupuesto de evaluaciones únicas.','MODELADO_AVANZADO'),grid(card(html.H3('Cuatro formas de buscar'),md('**Grid:** enumera 24 puntos prefijados.\n\n**Random:** muestrea categorías y escalas continuas declaradas.\n\n**TPE:** ocho propuestas iniciales aleatorias y densidades Parzen; no es un proceso gaussiano.\n\n**Genético:** población de ocho, torneo, cruce, mutación y una élite; incorpora un candidato inédito si una generación no aporta novedades.')),card(html.H3('Qué significa presupuesto común'),html.P('Cada método recibe 24 candidatos únicos. En KNN, árbol y Random Forest eso agota un soporte discreto de 24 puntos, explicando empates finales.'),note('Sin superioridad universal','Mejorar antes el F1 interno no asegura menor tiempo ni mejor desempeño externo. Una meseta con 24 evaluaciones no prueba convergencia global.'))),
        card(label('Familia para observar convergencia',dropdown('opt-family',list(a.MODELS),'random_forest',a.MODELS))),html.Div(id='opt-output'),chapters_details('MODELADO_AVANZADO',r'1\.5|1\.6|2\.')])


def calibration(d):
    return html.Div([intro('calibration','Distinguir clientes, asignar probabilidades y decidir a quién alertar son tres tareas diferentes.','MODELADO_AVANZADO'),
        grid(card(html.H3('Calibración sigmoid dentro de train'),md('El balanceo cambia el aprendizaje y puede desajustar probabilidades. La rama sigmoid se ajustó con tres folds dentro del entrenamiento externo, sin usar las etiquetas externas.\n\n**Brier:** error cuadrático medio de probabilidad; menor es mejor.\n\n**ECE:** diferencia media entre confianza y frecuencia en diez intervalos; depende de los bins.\n\nLa calibración mejoró conjuntamente Brier y ECE en **80 de 96** configuraciones con probabilidad nativa. No mejora todas, ni se usó para elegir el ranking principal.')),card(html.H3('Medias externas · realización 0'),label('Familia de calibración',dropdown('cal-family',list(a.MODELS),'random_forest',a.MODELS)),html.Div(id='cal-output'))),
        card(html.H3('Laboratorio de umbral · test histórico'),html.P('Simulación didáctica sobre predicciones nativas guardadas de modelos fijos. Cambiar el umbral aquí no selecciona una regla operativa validada.'),label('Modelo finalista',dropdown('threshold-model',a.FINALISTS,a.FINALISTS[0],{x:x.replace('__',' · ') for x in a.FINALISTS})),label('Umbral para clasificar abandono',dcc.Slider(.01,.99,.01,value=.5,marks={.1:'0,10',.3:'0,30',.5:'0,50',.7:'0,70',.9:'0,90'},id='threshold',tooltip={'always_visible':True})),html.Div(id='threshold-output')),
        note('Para una decisión de retención','El umbral requiere costes explícitos, capacidad de contacto y evaluación en datos futuros. Las probabilidades nativas usadas aquí no se presentan como riesgo prospectivo calibrado.')])


def statistics(d):
    stats=a.DATA/'estadistica';effects=pd.read_csv(stats/'delong_holm.csv'); f=json.loads((stats/'friedman.json').read_text(encoding="utf-8"))
    return html.Div([intro('statistics','El tamaño de una diferencia y la incertidumbre importan tanto como el orden del ranking.','MODELADO_AVANZADO'),
        html.Div([kpi('ΔF1 RF − XGBOOST','0,00043','Media externa · exploratoria'),kpi('BLOQUES FRIEDMAN','5','Folds con entrenamiento solapado'),kpi('p AJUSTADOS DELONG','1,0','Tres comparaciones dirigidas'),kpi('TEST HISTÓRICO','Suplementario','Expuesto en la primera entrega')],className='kpis'),
        grid(card(html.H3('Friedman / Nemenyi'),html.P(f'Estadístico Friedman: {f["statistic"]:.3f}; Kendall W: {f["kendall_W"]:.3f}. El valor p exploratorio es {f["p_exploratory"]:.2e}.'),note('No es confirmatorio','Los cinco folds no son bloques independientes y comparten entrenamientos. Las tres realizaciones no añaden datasets. Nemenyi y la diferencia crítica conservan alcance exploratorio.')),card(html.H3('DeLong y bootstrap'),html.P('Los tres p de AUC ajustados con Holm son 1,0. No detectar diferencia no demuestra equivalencia.'),html.P('RF − XGBoost: ΔAUC ≈ 0,00171; IC bootstrap ≈ [−0,00337; 0,00695]. El intervalo incluye cero.'),note('Intervalos condicionales','El remuestreo de clientes mantiene modelos fijos; no incorpora selección ni reentrenamiento. Tampoco corrige que el test ya había sido observado.'))),
        card(html.H3('Comparaciones dirigidas originales'),table(effects)),card(html.H3('Métricas suplementarias en test'),table(pd.read_csv(stats/'metricas_test_suplementarias.csv'))),chapters_details('MODELADO_AVANZADO',r'3\.1')])


def explain(d):
    return html.Div([intro('explain','Qué usa el modelo globalmente y cómo llegó a un acierto o a un error concreto.','MODELADO_AVANZADO'),
        card(label('Modelo explicado',dropdown('explain-model',a.FINALISTS[:2],a.FINALISTS[0],{a.FINALISTS[0]:'Random Forest · SMOTE · TPE',a.FINALISTS[1]:'XGBoost · SMOTE · Random'})),html.Div(id='explain-output')),
        grid(card(html.H3('SHAP: contribuciones al modelo'),html.P('TreeSHAP interventional explica la probabilidad nativa positiva. Usa 100 clientes reales de fondo y 5.634 de desarrollo para el resumen global. La referencia más las contribuciones reproduce la probabilidad dentro de tolerancia 10⁻⁵.'),html.P('Agrupar columnas one-hot resume atribuciones; no equivale a recalcular otro juego de Shapley.')),card(html.H3('LIME: sustituto local'),html.P('Usa 5.000 perturbaciones, diez variables y tres semillas. Conviene revisar R², error de aproximación y estabilidad: una selección estable de variables no garantiza fidelidad.'),html.P('En algunos casos de XGBoost el sustituto produce valores negativos. No son probabilidades del clasificador ni se recortan para esconder el error.'))),
        note('Las atribuciones no son causas','Explican la salida nativa, no la rama sigmoid. Las perturbaciones pueden romper dependencias entre servicios e importes. Los casos se eligieron por confianza máxima, no por representatividad.'),chapters_details('MODELADO_AVANZADO',r'3\.2')])


def conclusions(d):
    return html.Div([intro('conclusions','Qué aprendimos, qué permanece incierto y qué falta para una aplicación de retención.','MODELADO_AVANZADO'),
        grid(card(html.Span('EVIDENCIA DESCRIPTIVA',className='eyebrow'),html.H2('El perfil de cuenta contiene señal'),html.P('Contrato, antigüedad y servicio de internet destacan en el análisis y en las atribuciones de los modelos finalistas. El patrón es asociativo y no demuestra el efecto de intervenir.'),dcc.Link('Revisar segmentos →',href='#bivariate',className='text-link')),card(html.Span('EVIDENCIA DEL MODELADO',className='eyebrow'),html.H2('RF–SMOTE es un candidato exploratorio'),html.P('F1 0,6384 ± 0,0227, Recall 0,7351, Precision 0,5646 y AUC 0,8438. XGBoost está muy cerca; no hay ganador confirmado.'),dcc.Link('Revisar comparación →',href='#models',className='text-link'))),
        card(html.H3('Conclusiones que sostiene el libro'),html.Ol([html.Li(t) for t in ['El presupuesto común permite comparar búsquedas sin afirmar superioridad universal.','La calibración redujo Brier y ECE conjuntamente en 80 de 96 configuraciones; no lo hizo en todas.','SHAP ayuda a explicar falsos negativos de alta confianza. LIME requiere revisar fidelidad, además de estabilidad.','DeLong con Holm no detectó diferencias de AUC. La ausencia de rechazo no acredita equivalencia.','El test histórico y los intervalos condicionales no sustituyen una validación prospectiva independiente.']])),
        card(html.H3('Antes de trasladarlo a retención'),html.Div([html.Div([html.Span('01',className='step-num'),html.H3('Validación temporal'),html.P('Medir variables antes del evento y evaluar en clientes futuros.')]),html.Div([html.Span('02',className='step-num'),html.H3('Costes y capacidad'),html.P('Definir el coste de falsas alertas, omisiones y contacto.')]),html.Div([html.Span('03',className='step-num'),html.H3('Impacto de intervención'),html.P('Evaluar si una acción de retención cambia realmente el resultado.')])],className='steps')),
        note('Paper del proyecto','La página del libro marca el paper como documento en construcción. Se conserva ese estado y se enlaza su versión disponible.'),html.A('Consultar paper ↗',href=BOOK+'paper.html',target='_blank',className='button'),chapters_details('MODELADO_AVANZADO',r'3\.3')])


def explorer(d):
    columns=['customerID','Contract','InternetService','tenure','MonthlyCharges','TotalCharges','Churn']
    return html.Div([intro('explorer','Inspecciona registros reales. Los filtros superiores definen la selección que se descarga.','EDA_'),filtered_info(d),card(label('Columnas visibles',dropdown('explorer-cols',list(a.RAW.columns[:21]),columns,multi=True)),html.Div(id='explorer-table'),html.Button('Descargar selección CSV ↓',id='export',className='button'),html.P('El CSV conserva toda la selección de los filtros superiores y las columnas elegidas. Los filtros de la tabla solo afectan la vista.',className='interp'))])


def reader(d):
    return html.Div([intro('reader','Consulta las explicaciones originales por capítulo y sección sin perder el hilo del dashboard.'),card(grid(label('Capítulo',dropdown('reader-chapter',list(a.CHAPTERS),'EDA_',{'EDA_':'1 · EDA','PREPROCESAMIENTO':'2 · Preprocesamiento','MODELADO_AVANZADO':'3 · Modelado avanzado'})),label('Buscar en títulos y texto',dcc.Input(id='reader-search',placeholder='Ej. SMOTE, calibración, sesgo…',type='search',debounce=True))),html.P('Texto explicativo de los notebooks originales. Las visualizaciones y tablas de resultados se exploran en las secciones del dashboard; el código completo está en GitHub.',className='interp'),html.Div(id='reader-output'))])


def authors(d):
    return html.Div([intro('authors','Autoría, fuentes y trazabilidad de los datos usados para construir este dashboard.'),grid(card(html.Div('LR',className='avatar'),html.H2('Laura Rivera'),html.P('Autora del proyecto · Universidad del Norte'),html.A('GitHub ↗',href='https://github.com/lauraformore',target='_blank')),card(html.Div('NC',className='avatar'),html.H2('Natalý Cárdenas'),html.P('Autora del proyecto · Universidad del Norte'),html.A('GitHub ↗',href='https://github.com/Nataly-Cardenas',target='_blank'))),card(html.H3('Fuentes'),md(f'- [Jupyter Book del proyecto]({BOOK})\n- [Repositorio y resultados originales]({a.PROVENANCE["repository"]})\n- [Dataset Telco Customer Churn · Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)\n- [Dashboard de referencia · código](https://github.com/lauraformore/DASH_VIZ)\n- [Paper en construcción]({BOOK}paper.html)'),html.P('El diseño toma de la referencia el recorrido lateral, las preguntas guía, las interpretaciones y la exploración por variable. Los cálculos descriptivos de este dashboard se realizan con el CSV original.'),html.Code('Commit fuente: '+a.PROVENANCE['commit']),html.P('Archivos recuperados: '+a.PROVENANCE['retrieved'])),note('Reproducibilidad','Se preservan train_index.csv y test_index.csv. No se entrenan modelos al visitar la app ni se cargan archivos pickle: las comparaciones y el laboratorio utilizan artefactos numéricos publicados.')])

PAGES={k:globals()[k] for k,_,_ in NAV}

@app.callback(Output('content','children'),Output('pager','children'),*[Output('nav-'+k,'className') for k,_,_ in NAV],Output('filters','style'),Input('url','hash'),Input('split','value'),Input('contract','value'),Input('internet','value'),Input('tenure','value'))
def route(hash_,split,contract,internet,tenure):
    key=(hash_ or '#overview').lstrip('#');d=a.cohort(split,contract,internet,tenure)
    if key not in PAGES: key='overview'
    i=[k for k,_,_ in NAV].index(key)
    pager=[dcc.Link('← Anterior',href='#'+NAV[i-1][0],className='text-link') if i else html.Span(),html.Span(f'{i+1:02d} / {len(NAV)}'),dcc.Link('Siguiente →',href='#'+NAV[i+1][0],className='text-link') if i<len(NAV)-1 else html.Span()]
    return (PAGES[key](d),pager,*['nav-link active' if key==k else 'nav-link' for k,_,_ in NAV],{} if key in ['overview','quality','univariate','bivariate','multivariate','explorer'] else {'display':'none'})

@app.callback(Output('split','value'),Output('contract','value'),Output('internet','value'),Output('tenure','value'),Input('reset','n_clicks'),prevent_initial_call=True)
def reset(_): return 'train','Todos','Todos',[0,72]

@app.callback(Output('uni-output','children'),Input('uni-var','value'),Input('split','value'),Input('contract','value'),Input('internet','value'),Input('tenure','value'))
def uni(variable,split,contract,internet,tenure):
    d=a.cohort(split,contract,internet,tenure)
    if d.empty:return empty()
    if variable in a.NUMERIC:
        s=d[variable].dropna();q=s.quantile([.25,.5,.75]);iqr=q.loc[.75]-q.loc[.25];out=((s<q.loc[.25]-1.5*iqr)|(s>q.loc[.75]+1.5*iqr)).sum()
        return html.Div([html.Div([kpi('MEDIA',f'{s.mean():.2f}'),kpi('MEDIANA',f'{s.median():.2f}'),kpi('EXTREMOS IQR',str(out)),kpi('VACÍOS',str(d[variable].isna().sum()))],className='kpis'),grid(chart('Distribución · '+a.LABELS[variable],px.histogram(d,x=variable,nbins=35,color_discrete_sequence=[BLUE]),f'{len(s):,} observaciones válidas. Asimetría: {s.skew():.2f}. Los vacíos no entran en el histograma.'),chart('Dispersión y extremos',px.box(d,y=variable,color_discrete_sequence=[TEAL]),'La caja cubre el 50 % central; la línea marca la mediana. Los puntos extremos son señales para revisar, no errores automáticos.'))])
    g=d.groupby(variable).size().reset_index(name='Clientes');g[variable]=g[variable].astype(str)
    return chart('Frecuencias · '+a.LABELS[variable],px.bar(g,x=variable,y='Clientes',text_auto=True,color_discrete_sequence=[BLUE]),f'Variable con {len(g)} categorías en esta selección. Los conteos dependen del filtro, no de un remuestreo.',420)

@app.callback(Output('bi-output','children'),Input('bi-var','value'),Input('split','value'),Input('contract','value'),Input('internet','value'),Input('tenure','value'))
def bi(variable,split,contract,internet,tenure):
    d=a.cohort(split,contract,internet,tenure)
    if d.empty:return empty()
    if variable in a.NUMERIC:
        return grid(chart('Distribución por resultado',px.box(d,x='Estado',y=variable,color='Estado',color_discrete_map=COLORS),f'Medianas: '+', '.join(f'{k}: {v:.2f}' for k,v in d.groupby('Estado')[variable].median().items())),chart('Formas comparadas',px.histogram(d,x=variable,color='Estado',color_discrete_map=COLORS,barmode='overlay',opacity=.65,histnorm='probability density',nbins=30),'Densidades normalizadas por grupo: compara la forma, no el tamaño relativo de las clases.'))
    g=a.category_rates(d,variable);g[variable]=g[variable].astype(str)
    f=px.bar(g,x=variable,y='tasa',text=g.tasa.map('{:.1%}'.format),custom_data=['n','abandonos'],color_discrete_sequence=[RED]);f.update_yaxes(tickformat='.0%',title='Tasa de abandono');f.update_traces(hovertemplate='%{x}<br>Tasa: %{y:.2%}<br>n=%{customdata[0]}<br>Abandonos=%{customdata[1]}<extra></extra>')
    top=g.loc[g.tasa.idxmax()]
    return grid(chart('Tasa de abandono por categoría',f,f'Mayor tasa observada: {top[variable]} ({top.tasa:.1%}, n={int(top.n)}). Los grupos son descriptivos; no se infiere causalidad.',420),card(html.H3('Tasas y denominadores'),table(g.rename(columns={'n':'Clientes','abandonos':'Abandonos','tasa':'Tasa'})),html.P('Seleccionar una sola categoría con los filtros restringe la comparación. Amplía la selección para contrastar perfiles.',className='interp')))

@app.callback(Output('scatter-output','children'),Input('scatter-x','value'),Input('scatter-y','value'),Input('split','value'),Input('contract','value'),Input('internet','value'),Input('tenure','value'))
def scatter(x,y,split,contract,internet,tenure):
    d=a.cohort(split,contract,internet,tenure)
    if d.empty:return empty()
    sample=d.sample(min(2000,len(d)),random_state=42)
    return html.Div([fig(px.scatter(sample,x=x,y=y,color='Estado',symbol='Estado',color_discrete_map=COLORS,opacity=.45),height=430),html.P(f'Muestra reproducible de {len(sample):,} clientes, hasta 2.000 para legibilidad. Correlación de Spearman de toda la selección: {d[[x,y]].corr(method="spearman").iloc[0,1]:.3f}.' if x!=y else 'Seleccionaste la misma variable en ambos ejes.',className='interp')])

@app.callback(Output('models-output','children'),Input('model-family','value'),Input('model-balance','value'),Input('model-metric','value'))
def model_output(family,balance,metric):
    d=a.SUMMARY.copy()
    if family!='Todas':d=d[d.model==family]
    if balance!='Todos':d=d[d.balance==balance]
    d=d.sort_values(metric+'_mean',ascending=False);top=d.head(16).copy();top['Configuración']=top.experiment_id.str.replace('__',' · ')
    f=px.bar(top.iloc[::-1],x=metric+'_mean',y='Configuración',error_x=metric+'_std',orientation='h',color='model',color_discrete_sequence=[BLUE,TEAL,RED,'#ae91d9']);f.update_layout(showlegend=False);f.update_xaxes(title=metric+' · media externa',range=[0,1]);f.update_yaxes(title='')
    view=d[['model','balance','optimizer',metric+'_mean',metric+'_std','recall_positive_mean','precision_positive_mean','roc_auc_mean']].copy();view=view.loc[:,~view.columns.duplicated()];view['model']=view.model.map(a.MODELS);view['balance']=view.balance.map(a.BALANCES);view['optimizer']=view.optimizer.map(a.OPTIMIZERS)
    pivot=d.pivot_table(index='model',columns='balance',values=metric+'_mean',aggfunc='mean')
    heat=px.imshow(pivot.rename(index=a.MODELS,columns=a.BALANCES),text_auto='.3f',color_continuous_scale=['#1b2942',BLUE,TEAL],labels={'color':metric})
    return html.Div([grid(chart(f'Primeras {len(top)} de {len(d)} configuraciones',f,'Barras: medias de cinco folds. Líneas: desviación estándar entre folds, no intervalos de confianza. El orden se actualiza con la métrica.',600),chart('Familia × balanceo',heat,'Promedio descriptivo sobre los optimizadores de cada celda. No es el resultado del mejor optimizador ni un efecto aislado del balanceo.',450)),card(html.H3(f'Tabla completa · {len(d)} configuraciones'),table(view))])

@lru_cache(maxsize=1)
def anytime():return pd.read_csv(a.DATA/'anytime.csv')

@app.callback(Output('opt-output','children'),Input('opt-family','value'))
def opt_output(family):
    d=anytime();d=d[(d.model==family)&(d.replicate==0)]
    g=d.groupby(['optimizer','evaluation']).best_inner_f1.mean().reset_index();g['Método']=g.optimizer.map(a.OPTIMIZERS)
    u=a.UNITS[(a.UNITS.model==family)&(a.UNITS.replicate==0)].copy();u['Método']=u.optimizer.map(a.OPTIMIZERS)
    return grid(chart('Mejor F1 interno acumulado',px.line(g,x='evaluation',y='best_inner_f1',color='Método',markers=True),'Media sobre cinco folds externos y cuatro balanceos, realización 0. El objetivo interno guía la búsqueda; no estima por sí solo generalización.',400),chart('Tiempo de búsqueda por unidad',px.box(u,x='Método',y='search_seconds',color='Método'),'Segundos observados en el entorno del experimento; las unidades tienen distintos balanceos. Estos tiempos no prometen el rendimiento de otra máquina.',400))

@app.callback(Output('cal-output','children'),Input('cal-family','value'))
def cal_output(family):
    d=a.UNITS[(a.UNITS.model==family)&(a.UNITS.replicate==0)]
    g=d.groupby('balance')[['brier','cal_brier','ece','cal_ece']].mean().reset_index();g.balance=g.balance.map(a.BALANCES)
    if g.brier.isna().all():return note('Sin probabilidad nativa','SVM se configuró con probability=False. No se inventa Brier/ECE nativo; la rama calibrada sí tiene probabilidades.')
    f=px.bar(g.melt(id_vars='balance',value_vars=['brier','cal_brier'],var_name='Rama',value_name='Brier'),x='balance',y='Brier',color='Rama',barmode='group',color_discrete_sequence=[BLUE,TEAL])
    return html.Div([fig(f,height=300),table(g),html.P('Promedio sobre folds externos y optimizadores. cal_ identifica la rama sigmoid.',className='interp')])

@app.callback(Output('threshold-output','children'),Input('threshold-model','value'),Input('threshold','value'))
def threshold_output(experiment,threshold):
    p=a.predictions(experiment);cm,m=a.threshold_metrics(p,threshold)
    f=px.imshow(cm,x=['Predice permanencia','Predice abandono'],y=['Permanece real','Abandona real'],text_auto=True,color_continuous_scale=['#1c2a43',BLUE],labels={'color':'Clientes'})
    fpr,tpr,_=roc_curve(p.y_true,p.score);roc=go.Figure(go.Scatter(x=fpr,y=tpr,line_color=BLUE,name='ROC'));roc.add_trace(go.Scatter(x=[0,1],y=[0,1],line={'dash':'dash','color':'#64748b'},name='Azar'));roc.update_xaxes(title='Falsos positivos');roc.update_yaxes(title='Verdaderos positivos')
    return html.Div([html.Div([kpi(k,f'{v:.1%}' if k!='Alertas' else str(v),'Test histórico · simulación') for k,v in m.items()],className='kpis'),grid(chart('Errores al umbral seleccionado',f,f'TN={cm[0,0]}, FP={cm[0,1]}, FN={cm[1,0]}, TP={cm[1,1]}. Reducir el umbral suele aumentar detección y falsas alertas.'),chart('Discriminación · modelo fijo',roc,f'AUC test suplementario = {roc_auc_score(p.y_true,p.score):.4f}. La curva utiliza scores guardados; no cambia al mover el umbral.'))])

@app.callback(Output('explain-output','children'),Input('explain-model','value'))
def explain_output(experiment):
    path=a.DATA/'interpretabilidad'/experiment
    g=pd.read_csv(path/'importancia_global.csv').rename(columns={'Unnamed: 0':'Variable'}).head(12)
    ids=[4267,4386] if experiment==a.FINALISTS[0] else [2221,4513]
    local=[]
    for row in ids:
        s=pd.read_csv(path/f'shap_local_{row}.csv');s=s.reindex(s.shap.abs().sort_values(ascending=False).index).head(10).sort_values('shap');s['Dirección']=np.where(s.shap>0,'Aumenta salida','Reduce salida')
        pred=a.predictions(experiment).set_index('row_id').loc[row]
        local.append(card(html.H3(f'Cliente {row} · '+('acierto' if pred.y_true==pred.y_pred else 'error')),html.P(f'Observado: {"abandona" if pred.y_true else "permanece"} · probabilidad nativa: {pred.probability:.4f}'),fig(px.bar(s,x='shap',y='feature',orientation='h',color='Dirección',color_discrete_map={'Aumenta salida':RED,'Reduce salida':BLUE}),height=360),html.Details([html.Summary('Ver LIME del mismo caso'),html.Img(src=f'/evidence/{experiment}/lime_local_{row}.png',className='evidence-image',alt=f'Explicación LIME del cliente {row}')]),html.P('La atribución explica por qué el modelo produjo esa salida, incluso cuando se equivocó.',className='interp')))
    stability=pd.read_csv(a.DATA/'interpretabilidad/estabilidad_lime.csv');stability=stability[stability.model==experiment]
    return html.Div([chart('Importancia global · media |SHAP|',px.bar(g.iloc[::-1],x='mean_abs_shap',y='Variable',orientation='h',color_discrete_sequence=[TEAL]),'Atribuciones agrupadas por variable en los 5.634 clientes reales de desarrollo. Magnitud absoluta no expresa dirección causal.',420),grid(*local),html.H3('Estabilidad local de LIME'),table(stability)])

@app.callback(Output('explorer-table','children'),Input('explorer-cols','value'),Input('split','value'),Input('contract','value'),Input('internet','value'),Input('tenure','value'))
def explorer_table(cols,split,contract,internet,tenure):
    d=a.cohort(split,contract,internet,tenure)
    return table(d[cols or ['customerID']],id='data-table',page_size=15) if not d.empty else empty()

@app.callback(Output('download','data'),Input('export','n_clicks'),State('explorer-cols','value'),State('split','value'),State('contract','value'),State('internet','value'),State('tenure','value'),prevent_initial_call=True)
def export(_,cols,split,contract,internet,tenure):return dcc.send_data_frame(a.cohort(split,contract,internet,tenure)[cols or ['customerID']].to_csv,'telco_seleccion.csv',index=False)

@app.callback(Output('reader-output','children'),Input('reader-chapter','value'),Input('reader-search','value'))
def reader_output(chapter,search):
    chunks=a.CHAPTERS[chapter];search=(search or '').casefold();chunks=[c for c in chunks if search in c['text'].casefold()]
    return html.Div([html.P(f'{len(chunks)} secciones encontradas',className='muted'),*[html.Details([html.Summary(c['title']),md(c['text'])],className='chapter') for c in chunks]])

app.clientside_callback("function(n){return n%2?'shell presenting':'shell';}",Output('shell','className'),Input('present-btn','n_clicks'))
app.clientside_callback("function(h){window.scrollTo({top:0});var k=(h||'#overview').slice(1);document.title=("+json.dumps(TITLES)+")[k]+' · Telco Atlas';return '';}",Output('present-btn','title'),Input('url','hash'))

if __name__=='__main__':app.run(host='0.0.0.0',port=int(os.environ.get('PORT',8050)),debug=False)
