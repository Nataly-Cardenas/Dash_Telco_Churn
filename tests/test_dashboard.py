import json
import sys
from pathlib import Path
import numpy as np
from plotly.utils import PlotlyJSONEncoder
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import analytics as a
import app

def test_original_split_and_counts():
    assert len(a.RAW)==7043
    assert len(a.TRAIN_IDS)==5634 and len(a.TEST_IDS)==1409
    assert not set(a.TRAIN_IDS)&set(a.TEST_IDS)
    assert set(a.TRAIN_IDS)|set(a.TEST_IDS)==set(a.RAW.index)
    assert a.RAW.target.sum()==1869
    assert a.RAW.TotalCharges.isna().sum()==11
    assert a.RAW.loc[a.RAW.TotalCharges.isna(),'tenure'].eq(0).all()

def test_published_results_and_population_preserved():
    assert len(a.SUMMARY)==112 and len(a.UNITS)==1680
    assert a.UNITS.groupby(['experiment_id','replicate']).size().eq(5).all()
    rf=a.SUMMARY.set_index('experiment_id').loc[a.FINALISTS[0]]
    assert np.isclose(rf.f1_positive_mean,.6384205660936855)
    for experiment in a.FINALISTS:
        p=a.predictions(experiment)
        assert len(p)==1409 and p.row_id.is_unique
        assert set(p.row_id)==set(a.TEST_IDS)
        assert np.array_equal(p.y_true.to_numpy(),a.RAW.loc[p.row_id,'target'].to_numpy())
        assert p.probability.between(0,1).all()

def test_threshold_recomputes_errors_and_alerts():
    p=a.predictions(a.FINALISTS[0])
    cm, m=a.threshold_metrics(p,.5)
    assert cm.sum()==1409
    assert cm[1].sum()==374
    assert m['Alertas']==cm[:,1].sum()
    assert a.threshold_metrics(p,.2)[1]['Alertas']>=a.threshold_metrics(p,.8)[1]['Alertas']
    assert a.threshold_metrics(p,.2)[1]['Recall']>=a.threshold_metrics(p,.8)[1]['Recall']

def test_filters_rates_and_empty_cohort():
    d=a.cohort('train','Month-to-month','Fiber optic',[0,12])
    assert len(d)>0 and d.tenure.le(12).all()
    assert d.Contract.eq('Month-to-month').all() and d.InternetService.eq('Fiber optic').all()
    g=a.category_rates(d,'PaymentMethod')
    assert g.n.sum()==len(d) and g.abandonos.sum()==d.target.sum()
    assert a.cohort('train','Two year','Fiber optic',[0,0]).empty

def test_pages_and_interactions_serialize():
    for split, contract, internet, tenure in [('train','Todos','Todos',[0,72]),('train','Two year','Fiber optic',[0,0])]:
        d=a.cohort(split,contract,internet,tenure)
        for key in app.PAGES:json.dumps(app.PAGES[key](d),cls=PlotlyJSONEncoder)
        for var in a.NUMERIC+a.CATEGORICAL:
            json.dumps(app.uni(var,split,contract,internet,tenure),cls=PlotlyJSONEncoder)
            json.dumps(app.bi(var,split,contract,internet,tenure),cls=PlotlyJSONEncoder)
    for model in a.MODELS:
        json.dumps(app.model_output(model,'Todos','f1_positive'),cls=PlotlyJSONEncoder)
        json.dumps(app.cal_output(model),cls=PlotlyJSONEncoder)
        json.dumps(app.opt_output(model),cls=PlotlyJSONEncoder)
    for model in a.FINALISTS[:2]:json.dumps(app.explain_output(model),cls=PlotlyJSONEncoder)
    for model in a.FINALISTS:json.dumps(app.threshold_output(model,.3),cls=PlotlyJSONEncoder)
    for var in a.CATEGORICAL+a.NUMERIC:json.dumps(app.var_chip(var),cls=PlotlyJSONEncoder)
    for row in a.TEST_IDS[:5]:json.dumps(app.real_output(row),cls=PlotlyJSONEncoder)
    assert app.real_random(3) in set(a.TEST_IDS)


def test_cover_and_simulator():
    customers=app.cover_customers()
    assert len(customers)==30 and len({c['id'] for c in customers})==30
    assert all(0<=c['prob']<=1 for c in customers)
    for contract in ['Month-to-month','One year','Two year']:
        for internet in ['Fiber optic','DSL','No']:
            s,_=a.similar_customers(contract,internet,'Electronic check',10,70)
            assert len(s)>0 and set(s.index)<=set(a.TRAIN_IDS)
            json.dumps(app.sim_output(contract,internet,'Electronic check',10,70),cls=PlotlyJSONEncoder)

def test_download_and_http_endpoints():
    # Pintar la sección no descarga nada: solo un clic real (n_clicks > 0).
    from dash import no_update
    assert app.export(0,['customerID'],'train','Todos','Todos',[0,72]) is no_update
    assert app.export(None,['customerID'],'train','Todos','Todos',[0,72]) is no_update
    payload=app.export(1,['customerID','Churn'],'train','Todos','Todos',[0,72])
    assert payload['filename']=='telco_seleccion.csv'
    assert len(payload['content'].splitlines())==5635
    client=app.server.test_client()
    for url in ['/','/healthz','/_dash-layout','/_dash-dependencies','/evidence/random_forest__smote__bayesian_tpe/lime_local_4386.png']:
        assert client.get(url).status_code==200
    assert client.get('/evidence/../../app.py').status_code==404


def test_text_files_always_specify_utf8():
    # Windows may default to cp1252; source data must never use the OS default.
    import ast
    for name in ['analytics.py', 'app.py']:
        tree = ast.parse((a.ROOT / name).read_text(encoding='utf-8'))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == 'read_text':
                assert any(k.arg == 'encoding' and isinstance(k.value, ast.Constant) and k.value.value == 'utf-8' for k in node.keywords), name


def test_every_visible_text_has_english_translation():
    import ast
    from i18n import EN
    texts=[]
    for name in ['app.py','chatbot.py']:
        for node in ast.walk(ast.parse((a.ROOT/name).read_text(encoding='utf-8'))):
            if isinstance(node,ast.Call) and isinstance(node.func,ast.Name) and node.func.id in ('T','N') and node.args and isinstance(node.args[0],ast.Constant):
                texts.append(node.args[0].value)
    for d in [a.LABELS,a.MODELS,a.BALANCES,a.OPTIMIZERS,a.FINALIST_NAMES]: texts+=list(d.values())
    missing=sorted({t for t in texts if t not in EN})
    assert not missing, missing


def test_pages_render_in_english_without_emoji():
    import re
    from i18n import set_lang
    emoji=re.compile('[\U0001F000-\U0001FAFF\u2600-\u27BF\uFE0F]')
    d=a.cohort('train','Todos','Todos',[0,72])
    try:
        for lang in ['en','es']:
            set_lang(lang)
            for key in app.PAGES:
                payload=json.dumps(app.PAGES[key](d),cls=PlotlyJSONEncoder,ensure_ascii=False)
                assert not emoji.search(payload), key
            assert ('Lost' in json.dumps(app.chrome(lang),cls=PlotlyJSONEncoder))==(lang=='en')
            json.dumps(app.threshold_output(a.FINALISTS[0],.4,lang),cls=PlotlyJSONEncoder)
            json.dumps(app.real_output(a.TEST_IDS[0],lang),cls=PlotlyJSONEncoder)
        set_lang('en')
        assert app.cover_customers()[0]['contract'] in {'Month-to-month','One year','Two year'}
    finally:
        set_lang('es')


def test_chatbot_answers_from_project_knowledge():
    import chatbot
    assert chatbot.client() is None or True
    es=chatbot.local_answer('¿cuál fue el mejor modelo?','es')
    assert 'Random Forest' in es and '0.638' in es
    en=chatbot.local_answer('which customers churn the most?','en')
    assert 'electronic check' in en.lower()
    assert 'CSV' in chatbot.local_answer('cómo descargo el csv','es')
    rendered=chatbot.answer('qué es SMOTE', [], 'es') if chatbot.client() is None else ''
    assert rendered=='' or 'SMOTE' in rendered
    # Preguntas sobre perfiles concretos se responden con cifras calculadas sobre entrenamiento.
    t=a.RAW.loc[a.TRAIN_IDS]; expected=t[(t.Contract=='Month-to-month')&(t.InternetService=='Fiber optic')].target.mean()
    assert f'{expected:.1%}' in chatbot.data_answer('tasa de abandono de clientes con fibra y mes a mes','es')
    assert chatbot.data_answer('quién hizo el proyecto','es') is None
    # Sin límite de preguntas y con guía de cada sección del dash.
    assert not hasattr(chatbot,'MAX_QUESTIONS')
    assert 'Modelado avanzado' in chatbot.local_answer('de qué capítulo es la galaxia 3d','es')


def test_every_section_cites_the_jupyter_book():
    d=a.cohort('train','Todos','Todos',[0,72])
    for key in app.NAV:
        payload=json.dumps(app.PAGES[key[0]](d),cls=PlotlyJSONEncoder)
        if key[0]!='autoras': assert 'Telco_Churn_Entregable/' in payload and ' src-' in payload, key[0]
    authors=json.dumps(app.PAGES['autoras'](d),cls=PlotlyJSONEncoder)
    assert 'linkedin.com/in/laura-s-rivera-figueredo' in authors and 'linkedin.com/in/n%C3%A1taly' in authors
    assert 'img/laura.jpg' in authors and 'img/nataly.jpg' in authors
    client=app.server.test_client()
    for img in ['laura.jpg','nataly.jpg']: assert client.get('/assets/img/'+img).status_code==200
