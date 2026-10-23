"""Datos originales y cálculos descriptivos. No entrena ni deserializa modelos."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'data'
RAW = pd.read_csv(DATA / 'WA_Fn-UseC_-Telco-Customer-Churn.csv')
RAW['TotalCharges'] = pd.to_numeric(RAW.TotalCharges, errors='coerce')
RAW['Estado'] = RAW.Churn.map({'Yes': 'Abandona', 'No': 'Permanece'})
RAW['target'] = (RAW.Churn == 'Yes').astype(int)
TRAIN_IDS = pd.read_csv(DATA / 'train_index.csv').iloc[:, 0].tolist()
TEST_IDS = pd.read_csv(DATA / 'test_index.csv').iloc[:, 0].tolist()
SUMMARY = pd.read_csv(DATA / 'resumen_112.csv')
UNITS = pd.read_csv(DATA / 'unidades.csv')
CHAPTERS = json.loads((DATA / 'chapters.json').read_text(encoding="utf-8"))
PROVENANCE = json.loads((DATA / 'provenance.json').read_text(encoding="utf-8"))
MODELS = {'knn':'KNN','naive_bayes':'Naive Bayes','logistic':'Regresión logística','decision_tree':'Árbol de decisión','random_forest':'Random Forest','xgboost':'XGBoost','svm':'SVM RBF'}
BALANCES = {'none':'Sin balanceo','balanced':'Pesos de clase','smote':'SMOTE','adasyn':'ADASYN'}
OPTIMIZERS = {'grid':'Grid','random':'Random','bayesian_tpe':'TPE','genetic':'Genético'}
NUMERIC = ['tenure', 'MonthlyCharges', 'TotalCharges']
CATEGORICAL = [c for c in RAW.columns if c not in NUMERIC + ['customerID','Churn','Estado','target']]
LABELS = {'tenure':'Antigüedad (meses)','MonthlyCharges':'Cargo mensual','TotalCharges':'Cargo acumulado','Contract':'Contrato','InternetService':'Internet','PaymentMethod':'Método de pago','TechSupport':'Soporte técnico','OnlineSecurity':'Seguridad en línea','gender':'Género','SeniorCitizen':'Adulto mayor','Partner':'Pareja','Dependents':'Dependientes','PhoneService':'Servicio telefónico','MultipleLines':'Líneas múltiples','OnlineBackup':'Respaldo en línea','DeviceProtection':'Protección de dispositivos','StreamingTV':'Streaming TV','StreamingMovies':'Streaming películas','PaperlessBilling':'Factura electrónica'}
FINALISTS = ['random_forest__smote__bayesian_tpe','xgboost__smote__random','logistic__smote__grid']

def cohort(split='train', contract='Todos', internet='Todos', tenure=(0,72)):
    d = RAW.loc[TRAIN_IDS if split == 'train' else TEST_IDS] if split in ['train','test'] else RAW
    if contract != 'Todos': d = d[d.Contract == contract]
    if internet != 'Todos': d = d[d.InternetService == internet]
    return d[d.tenure.between(*tenure)].copy()

def category_rates(d, variable):
    return d.groupby(variable, dropna=False).target.agg(n='size', abandonos='sum', tasa='mean').reset_index()

def predictions(experiment):
    if experiment not in FINALISTS: raise ValueError('Configuración no autorizada')
    return pd.read_csv(DATA/'predictions'/experiment/'test_predictions.csv')

def threshold_metrics(p, threshold):
    y, pred = p.y_true, (p.probability >= threshold).astype(int)
    cm = confusion_matrix(y,pred,labels=[0,1])
    return cm, {'Precision':precision_score(y,pred,zero_division=0),'Recall':recall_score(y,pred,zero_division=0),'F1':f1_score(y,pred,zero_division=0),'Alertas':int(pred.sum())}
