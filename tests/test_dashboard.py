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
    for f in [('train','Todos','Todos',[0,72]),('train','Two year','Fiber optic',[0,0]),('test','Month-to-month','DSL',[0,12])]:
        d=a.cohort(*f)
        for key in app.PAGES:json.dumps(app.PAGES[key](d),cls=PlotlyJSONEncoder)
        for var in a.NUMERIC:
            json.dumps(app.uni_num(var,*f),cls=PlotlyJSONEncoder); json.dumps(app.bi_num(var,*f),cls=PlotlyJSONEncoder)
        for var in ['__all__']+a.CATEGORICAL: json.dumps(app.uni_cat(var,*f),cls=PlotlyJSONEncoder)
        for var in a.CATEGORICAL: json.dumps(app.bi_cat(var,*f),cls=PlotlyJSONEncoder)
        json.dumps(app.multi(*f),cls=PlotlyJSONEncoder); json.dumps(app.missing(*f),cls=PlotlyJSONEncoder)
    for model in a.MODELS:
        json.dumps(app.model_output(model,'Todos','f1_positive'),cls=PlotlyJSONEncoder)
        json.dumps(app.opt_output(model),cls=PlotlyJSONEncoder)
        json.dumps(app.res_output(model),cls=PlotlyJSONEncoder)
    for model in a.FINALISTS[:2]:json.dumps(app.explain_output(model),cls=PlotlyJSONEncoder)
    for model in app.BEST_BY_FAMILY:json.dumps(app.cal_map(model),cls=PlotlyJSONEncoder)
    for model in app.VARIANT_FAMILIES:json.dumps(app.eff_variants(model),cls=PlotlyJSONEncoder)
    for model in app.comp('escalamiento').model.unique():json.dumps(app.eff_scale(model),cls=PlotlyJSONEncoder)
    for model in a.FINALISTS:json.dumps(app.threshold_output(model,.3),cls=PlotlyJSONEncoder)
    for row in a.TEST_IDS[:5]:json.dumps(app.real_output(row),cls=PlotlyJSONEncoder)
    assert app.real_random(3) in set(a.TEST_IDS)


def test_eda_statistics_match_the_book():
    import eda
    d=a.RAW.loc[a.TRAIN_IDS]
    num,cat,fam=eda.holm_family(d)
    assert len(fam)==19
    assert round(num.set_index('Variable').loc['tenure','r punto-biserial'],4)==-0.3456
    v=cat.set_index('Variable')['V de Cramér']
    assert v.idxmax()=='Contract' and round(v.max(),4)==0.4111
    assert not fam.set_index('Variable').loc[['gender','PhoneService'],'Rechaza H0'].any()
    assert (eda.normality(d)['¿Normal?']=='No').all()
    assert round(d[a.NUMERIC].corr(method='spearman').loc['tenure','TotalCharges'],2)==0.89


def test_modeling_artifacts_match_the_book():
    from sklearn.metrics import confusion_matrix, roc_auc_score
    o=app.oof(); rf=o[o.experiment=='random_forest__smote__bayesian_tpe']
    assert rf.row_id.is_unique and set(rf.row_id)==set(a.TRAIN_IDS)
    assert confusion_matrix(rf.y_true,rf.y_pred).tolist()==[[3289,850],[396,1099]]
    b=app.baseline()
    assert confusion_matrix(b.y_true,(b.probability>=.5).astype(int)).tolist()==[[749,286],[81,293]]
    assert round(roc_auc_score(b.y_true,b.probability),4)==0.8419
    assert (b.y_true.to_numpy()==a.RAW.loc[b.row_id,'target'].to_numpy()).all()


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
    assert 'Nemenyi' in chatbot.local_answer('qué es el diagrama de nemenyi','es')
    assert not hasattr(chatbot,'_chunks')
    assert 'libro' not in chatbot.local_answer('qué es tenure','es').lower()


def test_sections_follow_the_book_with_one_link_each():
    d=a.cohort('train','Todos','Todos',[0,72])
    assert [k for k,_,_ in app.NAV]==['introduccion','eda','preprocesamiento','modelado','simulador','conclusiones','autoras']
    intro=json.dumps(app.PAGES['introduccion'](d),cls=PlotlyJSONEncoder)
    assert 'bivariado' not in intro.lower() and 'var-chip' not in intro
    for key,chapter in app.CHAPTER.items():
        page=json.dumps(app.PAGES[key](d),cls=PlotlyJSONEncoder)
        assert page.count('Telco_Churn_JBook/'+chapter+'.html')==1, key
    authors=json.dumps(app.PAGES['autoras'](d),cls=PlotlyJSONEncoder)
    assert 'linkedin.com/in/laura-s-rivera-figueredo' in authors and 'linkedin.com/in/n%C3%A1taly' in authors
    client=app.server.test_client()
    for img in ['laura.jpg','nataly.jpg']: assert client.get('/assets/img/'+img).status_code==200


def test_charts_use_one_color_plus_highlight():
    import re
    allowed={app.BASE.lower(),app.HOT.lower(),app.GREY.lower()}
    d=a.cohort('train','Todos','Todos',[0,72])
    payload=json.dumps([app.PAGES[k](d) for k in ['eda','preprocesamiento','modelado']]+[app.bi_cat('Contract','train','Todos','Todos',[0,72]),app.res_output('random_forest'),app.opt_output('xgboost')],cls=PlotlyJSONEncoder)
    def drop_templates(x):
        if isinstance(x,dict): return {k:drop_templates(v) for k,v in x.items() if k!='template'}
        if isinstance(x,list): return [drop_templates(v) for v in x]
        return x
    payload=json.dumps(drop_templates(json.loads(payload)))
    used={c.lower() for c in re.findall(r'"(#[0-9A-Fa-f]{6})"',payload)}
    assert used<=allowed|{'#0b1426','#eaf2ff','#8792a8','#ffffff'}, used-allowed


def test_new_jupyter_book_results_match_the_book():
    """Calibración isotónica, multifidelidad, eficiencia y tamaños del efecto del entregable 3."""
    u=a.UNITS[a.UNITS.replicate==0].groupby('experiment_id')[['brier','ece','cal_brier','cal_ece']].mean()
    m=u.join(app.comp('isotonica_unidades').groupby('experiment_id')[['iso_brier','iso_ece']].mean())
    assert (m.iso_brier<m.cal_brier).sum()==67 and (m.iso_ece<m.cal_ece).sum()==74
    n=m.dropna(subset=['brier'])
    assert ((n.iso_brier<n.brier)&(n.iso_ece<n.ece)).sum()==84 and ((n.cal_brier<n.brier)&(n.cal_ece<n.ece)).sum()==80
    r=app.comp('multifidelidad_resumen').set_index(['model','method']).segundos
    assert [round(r[x,'random_completo']/r[x,'successive_halving'],1) for x in ['random_forest','svm','xgboost']]==[3.7,7.1,1.6]
    v=app.comp('estandar_vs_optimizado').set_index('variant')
    assert round(v.loc['LinearSVC','aceleración_entrenamiento'])==36 and round(1/v.loc['exact','aceleración_entrenamiento'],1)==2.8
    assert round(app.comp('escalamiento_exponentes').set_index('model').loc['svm rbf','exponente_n'],1)==2.0
    assert app.comp('efecto_pares_nemenyi').pares.iloc[0]==31
    assert app.oof().probability_isotonic.notna().all()


def test_antena_greets_and_answers_who_she_is():
    import chatbot
    from i18n import set_lang
    set_lang('es')
    assert 'Antena' in chatbot.answer('¿cómo te llamas?',[],'es')
    assert 'Successive Halving' in chatbot.answer('qué es successive halving',[],'es')
    assert 'isotónica' in chatbot.answer('platt o isotónica',[],'es')
    js=(a.ROOT/'assets'/'antena.js').read_text(encoding='utf-8')
    assert 'Hola, soy Antena, estoy aquí para resolver tus dudas' in js and 'speechSynthesis' in js
