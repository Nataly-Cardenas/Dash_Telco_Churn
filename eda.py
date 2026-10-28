"""Cálculos del análisis exploratorio, con las mismas pruebas que el notebook EDA_ del Jupyter Book."""
import numpy as np
import pandas as pd
from scipy import stats
import analytics as a

ALPHA = 0.05
TARGET = 'Churn'


def holm(p):
    """Ajuste de Holm para una familia de valores p (NaN se conserva)."""
    p = pd.Series(p, dtype=float)
    ok = p.dropna().sort_values()
    m = len(ok)
    adj = np.minimum(1, np.maximum.accumulate([(m - i) * v for i, v in enumerate(ok.to_numpy())]))
    out = pd.Series(np.nan, index=p.index)
    out.loc[ok.index] = adj
    return out.to_numpy()


def contingency_test(ct):
    """Chi-cuadrado sin corrección, V de Cramér y Fisher exacta cuando las esperadas son insuficientes en 2×2."""
    ct = ct.loc[ct.sum(axis=1) > 0, ct.sum(axis=0) > 0]
    if min(ct.shape) < 2:
        return {'χ²': np.nan, 'gl': 0, 'p': np.nan, 'V de Cramér': np.nan, 'Prueba': '—'}
    chi, p, gl, exp = stats.chi2_contingency(ct, correction=False)
    v = np.sqrt(chi / (ct.to_numpy().sum() * min(ct.shape[0] - 1, ct.shape[1] - 1)))
    method = 'Chi-cuadrado'
    if not (exp >= 5).all():
        if ct.shape == (2, 2):
            p = stats.fisher_exact(ct).pvalue; method = 'Fisher exacta'
        else:
            p = np.nan; method = 'Esperadas < 5'
    return {'χ²': chi, 'gl': int(gl), 'p': p, 'V de Cramér': v, 'Prueba': method}


def target_summary(d):
    c = d[TARGET].value_counts().reindex(['No', 'Yes'], fill_value=0)
    return pd.DataFrame({'Churn': c.index, 'Clientes': c.values, 'Porcentaje (%)': 100 * c.values / max(len(d), 1)})


def numeric_summary(d):
    s = d[a.NUMERIC]
    r = s.describe().T.rename(columns={'count': 'N', 'mean': 'Media', 'std': 'DS', 'min': 'Mínimo', '25%': 'Q1', '50%': 'Mediana', '75%': 'Q3', 'max': 'Máximo'})
    r['IQR'] = r.Q3 - r.Q1; r['Nulos'] = s.isna().sum(); r['Asimetría'] = s.skew()
    return r.reset_index(names='Variable')


def outliers(d):
    rows = []
    for c in a.NUMERIC:
        s = d[c].dropna()
        if s.empty: continue
        q1, q3 = s.quantile([.25, .75]); iqr = q3 - q1; lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        n = int(((s < lo) | (s > hi)).sum())
        rows.append({'Variable': c, 'Q1': q1, 'Q3': q3, 'IQR': iqr, 'Límite inferior': lo, 'Límite superior': hi, 'Posibles outliers': n, 'Asimetría': s.skew()})
    return pd.DataFrame(rows)


def numeric_tests(d):
    rows = []
    for c in a.NUMERIC:
        no, yes = d.loc[d[TARGET] == 'No', c].dropna(), d.loc[d[TARGET] == 'Yes', c].dropna()
        if min(len(no), len(yes)) < 2:
            continue
        if c == 'MonthlyCharges':
            t = stats.ttest_ind(yes, no, equal_var=False); method = 't de Welch'
        else:
            t = stats.mannwhitneyu(yes, no, alternative='two-sided', method='asymptotic'); method = 'Mann–Whitney U'
        ok = d[c].notna()
        r = stats.pointbiserialr((d.loc[ok, TARGET] == 'Yes').astype(int), d.loc[ok, c]).statistic
        rows.append({'Variable': c, 'Prueba': method, 'Estadístico': t.statistic, 'p': t.pvalue, 'Diferencia media Yes−No': yes.mean() - no.mean(), 'r punto-biserial': r})
    return pd.DataFrame(rows)


def categorical_tests(d):
    rows = []
    for c in a.CATEGORICAL:
        ct = pd.crosstab(d[c], d[TARGET]).reindex(columns=['No', 'Yes'], fill_value=0)
        rows.append({'Variable': c, **contingency_test(ct)})
    return pd.DataFrame(rows)


def holm_family(d):
    """Familia de 19 contrastes (3 numéricos + 16 categóricos) corregida con Holm, como en el libro."""
    num, cat = numeric_tests(d), categorical_tests(d)
    fam = pd.concat([num[['Variable', 'p']], cat[['Variable', 'p', 'V de Cramér']]], ignore_index=True)
    fam['p Holm'] = holm(fam['p'])
    fam['Rechaza H0'] = fam['p Holm'] < ALPHA
    return num.merge(fam[['Variable', 'p Holm']], on='Variable'), cat.merge(fam[['Variable', 'p Holm']], on='Variable'), fam


def churn_rates(d):
    rows = []
    for c in a.CATEGORICAL:
        ct = pd.crosstab(d[c], d[TARGET]).reindex(columns=['No', 'Yes'], fill_value=0)
        tot = ct.sum(axis=1); rate = 100 * ct['Yes'] / tot
        for k in ct.index:
            rows.append({'Variable': c, 'Categoría': k, 'Yes': int(ct.loc[k, 'Yes']), 'Total': int(tot.loc[k]), 'Tasa (%)': rate.loc[k], 'Mayor de su variable': bool(rate.loc[k] == rate.max())})
    return pd.DataFrame(rows)


def normality(d):
    rows = []
    for c in a.NUMERIC:
        s = d[c].dropna()
        if len(s) < 8: continue
        k2, p = stats.normaltest(s)
        rows.append({'Variable': c, 'N': len(s), 'Asimetría': s.skew(), 'Curtosis': s.kurt(), "K² D'Agostino": k2, 'p': p})
    r = pd.DataFrame(rows)
    if not r.empty:
        r['p Holm'] = holm(r['p']); r['¿Normal?'] = np.where(r['p Holm'] >= ALPHA, 'Sí', 'No')
    return r


def qq(series):
    (theo, obs), (slope, inter, _) = stats.probplot(series.dropna(), dist='norm')
    return theo, obs, slope, inter


CAT_PAIRS = [('InternetService', 'OnlineSecurity'), ('InternetService', 'TechSupport'), ('PhoneService', 'MultipleLines'), ('Contract', 'PaperlessBilling'), ('Partner', 'Dependents')]


def categorical_pairs(d):
    rows = [{'Variable A': x, 'Variable B': y, **contingency_test(pd.crosstab(d[x], d[y]))} for x, y in CAT_PAIRS]
    r = pd.DataFrame(rows); r['p Holm'] = holm(r['p'])
    return r


def redundancy(d):
    rows = []
    for c in ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']:
        rows.append({'Regla': f'InternetService=No ↔ {c}=No internet service', 'Discrepancias': int((d.InternetService.eq('No') != d[c].eq('No internet service')).sum())})
    rows.append({'Regla': 'PhoneService=No ↔ MultipleLines=No phone service', 'Discrepancias': int((d.PhoneService.eq('No') != d.MultipleLines.eq('No phone service')).sum())})
    return pd.DataFrame(rows)


def kde(d, col, n=200):
    grid = np.linspace(d[col].min(), d[col].max(), n); out = {}
    for k in ['No', 'Yes']:
        v = d.loc[d[TARGET] == k, col].dropna()
        if len(v) > 2 and v.nunique() > 1: out[k] = stats.gaussian_kde(v, bw_method='scott')(grid)
    return grid, out


def missing_table(d):
    n = d[a.RAW.columns[:21]].isna().sum()
    return pd.DataFrame({'Variable': n.index, 'Nulos': n.values, 'Porcentaje (%)': 100 * n.values / max(len(d), 1)}).query('Nulos > 0')


def mcar_tests(d):
    miss = d.TotalCharges.isna()
    rows = [{'Comparación': name, **contingency_test(pd.crosstab(miss, comp))} for name, comp in [('Churn', d[TARGET]), ('tenure = 0', d.tenure.eq(0))]]
    r = pd.DataFrame(rows); r['p Holm'] = holm(r['p'])
    return r
