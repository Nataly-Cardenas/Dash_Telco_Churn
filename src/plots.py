"""Paleta, plantilla común y funciones que construyen las figuras de Plotly del tablero.

Un solo color para todas las series; el segundo color solo resalta el mejor resultado, la clase de abandono o el dato
que la gráfica quiere señalar, y el gris marca referencias. Son los mismos tonos del Jupyter Book.
"""
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve

from src.i18n import T

BASE, HOT, GREY = '#2E86AB', '#E4572E', '#9AA5B1'
LOW, MID, HIGH = BASE, GREY, HOT
SEQ = [[0, 'rgba(46,134,171,.06)'], [1, BASE]]
DIVERGING = [[0, BASE], [.5, 'rgba(154,165,177,.15)'], [1, HOT]]
GRID = 'rgba(139,147,167,.18)'


def highlight(values, best=None):
    """Colores de barras: todas en BASE y la mejor (o la indicada) en HOT."""
    values = list(values)
    if best is None: best = int(np.nanargmax(values)) if values else -1
    best = best if isinstance(best, (list, set, tuple)) else [best]
    return [HOT if i in best else BASE for i in range(len(values))]


def style(f, height=360, legend=True, margin=None):
    # Los colores de texto y rejilla se ajustan al tema desde CSS (assets/style.css, sección "gráficas").
    f.update_layout(template='plotly_white', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                    font={'family': 'Inter, sans-serif', 'color': '#8792a8', 'size': 13.5}, colorway=[BASE, GREY, HOT],
                    margin=margin or {'l': 20, 'r': 20, 't': 36, 'b': 30}, height=height, showlegend=legend,
                    legend={'orientation': 'h', 'y': 1.14, 'title_text': '', 'bgcolor': 'rgba(0,0,0,0)'},
                    transition={'duration': 600, 'easing': 'cubic-in-out'},
                    hoverlabel={'bgcolor': '#0b1426', 'bordercolor': BASE, 'font': {'family': 'Inter', 'color': '#eaf2ff', 'size': 13}})
    f.update_xaxes(gridcolor=GRID, zerolinecolor=GRID, linecolor=GRID); f.update_yaxes(gridcolor=GRID, zerolinecolor=GRID, linecolor=GRID)
    return f


def roc_animation(curves):
    """Curvas ROC que se trazan con un botón de play (mismo tipo de gráfica que el libro)."""
    steps = 24; f = go.Figure()
    for name, fpr, tpr, color in curves:
        f.add_trace(go.Scatter(x=fpr, y=tpr, name=name, mode='lines', line={'color': color, 'width': 3 if color == HOT else 2.4}))
    f.add_trace(go.Scatter(x=[0, 1], y=[0, 1], name=T('Azar'), mode='lines', line={'dash': 'dash', 'color': GREY}))
    frames = []
    for k in range(1, steps + 1):
        data = [go.Scatter(x=fpr[:max(2, int(len(fpr) * k / steps))], y=tpr[:max(2, int(len(tpr) * k / steps))]) for _, fpr, tpr, _ in curves]
        frames.append(go.Frame(data=data, traces=list(range(len(curves))), name=str(k)))
    f.frames = frames
    f.update_layout(updatemenus=[{'type': 'buttons', 'showactive': False, 'x': 1, 'y': -.18, 'xanchor': 'right', 'buttons': [
        {'label': '▶', 'method': 'animate', 'args': [None, {'frame': {'duration': 45, 'redraw': False}, 'transition': {'duration': 0}, 'fromcurrent': False}]}]}])
    f.update_xaxes(title=T('Tasa de falsos positivos'), range=[0, 1]); f.update_yaxes(title=T('Tasa de verdaderos positivos (Recall)'), range=[0, 1.02])
    return f


def gauge(value, title, color):
    return go.Figure(go.Indicator(mode='gauge+number', value=value * 100, number={'suffix': ' %', 'font': {'size': 34}}, title={'text': title, 'font': {'size': 13}},
                                  gauge={'axis': {'range': [0, 100], 'tickcolor': '#8792a8'}, 'bar': {'color': color, 'thickness': .28}, 'bgcolor': 'rgba(0,0,0,0)', 'borderwidth': 0,
                                         'steps': [{'range': [0, 35], 'color': 'rgba(46,134,171,.10)'}, {'range': [35, 60], 'color': 'rgba(154,165,177,.14)'}, {'range': [60, 100], 'color': 'rgba(228,87,46,.14)'}]}))


def confusion(y, p, umbral):
    cm = confusion_matrix(y, (p >= umbral).astype(int), labels=[0, 1])
    f = px.imshow(cm, x=[T('Predice permanencia'), T('Predice abandono')], y=[T('Permanece'), T('Abandona')], text_auto=True, color_continuous_scale=SEQ, aspect='auto')
    f.update_coloraxes(showscale=False); f.update_traces(textfont_size=17, hovertemplate='%{y} · %{x}: %{z}<extra></extra>')
    return f


def roc(y, p, nombre):
    fpr, tpr, _ = roc_curve(y, p)
    f = go.Figure([go.Scatter(x=fpr, y=tpr, mode='lines', line={'color': BASE, 'width': 2.6}, name=nombre),
                   go.Scatter(x=[0, 1], y=[0, 1], mode='lines', line={'dash': 'dash', 'color': GREY}, name=T('Azar'))])
    f.update_xaxes(title=T('Tasa de falsos positivos'), range=[0, 1]); f.update_yaxes(title=T('Tasa de verdaderos positivos (Recall)'), range=[0, 1.02])
    return f


def precision_recall(y, p, nombre):
    pr, rc, _ = precision_recall_curve(y, p)
    f = go.Figure([go.Scatter(x=rc, y=pr, mode='lines', line={'color': BASE, 'width': 2.6}, name=nombre),
                   go.Scatter(x=[0, 1], y=[y.mean()] * 2, mode='lines', line={'dash': 'dash', 'color': GREY}, name=T('Prevalencia'))])
    f.update_xaxes(title='Recall', range=[0, 1]); f.update_yaxes(title='Precision', range=[0, 1.02])
    return f


def reliability(y, p, bins=10):
    """Diagrama de confiabilidad con 10 intervalos uniformes; el tamaño del punto es el número de clientes."""
    b = np.minimum((p * bins).astype(int), bins - 1)
    g = pd.DataFrame({'b': b, 'p': p, 'y': y}).groupby('b').agg(p=('p', 'mean'), y=('y', 'mean'), n=('y', 'size')).reset_index()
    f = go.Figure([go.Scatter(x=[0, 1], y=[0, 1], mode='lines', line={'dash': 'dash', 'color': GREY}, name=T('Calibración perfecta')),
                   go.Scatter(x=g.p, y=g.y, mode='lines+markers', name=T('Modelo'), line={'color': BASE, 'width': 2.4},
                              marker={'size': 6 + 14 * np.sqrt(g.n / g.n.max()), 'color': BASE}, customdata=g.n,
                              hovertemplate=T('Probabilidad media') + ' %{x:.3f}<br>' + T('Frecuencia de abandono') + ' %{y:.3f}<br>n = %{customdata}<extra></extra>')])
    f.update_xaxes(title=T('Probabilidad predicha (media del intervalo)'), range=[0, 1]); f.update_yaxes(title=T('Frecuencia observada de abandono'), range=[0, 1])
    return f


def cd_diagram(rangos, cd, resaltar, nombres):
    """Diagrama de diferencia crítica (Demšar, 2006): rangos medios en un eje y barras que unen los grupos no distinguibles."""
    r = pd.Series(rangos).sort_values()
    k = len(r)
    f = go.Figure()
    izq = r.iloc[: (k + 1) // 2]; der = r.iloc[(k + 1) // 2:]
    for i, (m, v) in enumerate(izq.items()):
        y = 1 + i
        f.add_trace(go.Scatter(x=[v, v, r.min() - .6], y=[0, -y, -y], mode='lines', line={'color': HOT if m == resaltar else GREY, 'width': 1.4}, hoverinfo='skip', showlegend=False))
        f.add_annotation(x=r.min() - .7, y=-y, text=f'{nombres.get(m, m)} ({v:.2f})', showarrow=False, xanchor='right', font={'color': HOT if m == resaltar else None, 'size': 12})
    for i, (m, v) in enumerate(der[::-1].items()):
        y = 1 + i
        f.add_trace(go.Scatter(x=[v, v, r.max() + .6], y=[0, -y, -y], mode='lines', line={'color': HOT if m == resaltar else GREY, 'width': 1.4}, hoverinfo='skip', showlegend=False))
        f.add_annotation(x=r.max() + .7, y=-y, text=f'({v:.2f}) {nombres.get(m, m)}', showarrow=False, xanchor='left', font={'color': HOT if m == resaltar else None, 'size': 12})
    f.add_trace(go.Scatter(x=r.values, y=[0] * k, mode='markers', marker={'color': [HOT if m == resaltar else BASE for m in r.index], 'size': 8},
                           text=[nombres.get(m, m) for m in r.index], hovertemplate='%{text}<br>' + T('Rango medio') + ' %{x:.2f}<extra></extra>', showlegend=False))
    # grupos máximos de modelos cuyo rango difiere menos que la diferencia crítica
    grupos, vals = [], r.values
    for i in range(k):
        j = i
        while j + 1 < k and vals[j + 1] - vals[i] <= cd: j += 1
        if j > i and not any(a <= i and j <= b for a, b in grupos): grupos.append((i, j))
    for n, (i, j) in enumerate(grupos):
        y = .35 + .22 * n
        f.add_trace(go.Scatter(x=[vals[i] - .05, vals[j] + .05], y=[y, y], mode='lines', line={'color': BASE, 'width': 4}, hoverinfo='skip', showlegend=False))
    top = .35 + .22 * len(grupos) + .4
    f.add_trace(go.Scatter(x=[1, 1 + cd], y=[top, top], mode='lines+markers', line={'color': HOT, 'width': 3}, marker={'symbol': 'line-ns-open', 'size': 12, 'color': HOT},
                           hoverinfo='skip', showlegend=False))
    f.add_annotation(x=1 + cd / 2, y=top + .3, text=T('CD = {cd}').format(cd=f'{cd:.2f}'), showarrow=False)
    f.update_xaxes(range=[r.min() - 6, r.max() + 6], showgrid=False, zeroline=False, tickvals=list(range(1, k + 1)), side='top', title=T('Rango medio (menor es mejor)'))
    f.update_yaxes(visible=False, range=[-(max(len(izq), len(der)) + 1), top + .8])
    return f


def shap_waterfall(valores, base, etiquetas, top=10):
    """Cascada SHAP: de la probabilidad media del fondo a la del cliente."""
    v = valores.reindex(valores.abs().sort_values(ascending=False).index)
    cabeza, resto = v.head(top), v.iloc[top:].sum()
    y = ['E[f(X)]', *[etiquetas(c) for c in cabeza.index]] + ([T('{n} variables más').format(n=len(v) - top)] if len(v) > top else []) + ['f(x)']
    x = [base, *cabeza.values] + ([resto] if len(v) > top else []) + [0]
    f = go.Figure(go.Waterfall(orientation='h', y=y, x=x, measure=['absolute'] + ['relative'] * (len(x) - 2) + ['total'],
                               increasing={'marker': {'color': HOT}}, decreasing={'marker': {'color': BASE}}, totals={'marker': {'color': GREY}},
                               connector={'line': {'color': GREY, 'width': 1}}, hovertemplate='%{y}: %{x:+.3f}<extra></extra>'))
    f.update_yaxes(autorange='reversed'); f.update_xaxes(title=T('Probabilidad de abandono'))
    return f


def shap_beeswarm(s, etiquetas):
    """Resumen SHAP global: un punto por cliente y variable; color según el valor relativo de la variable."""
    orden = s.groupby('variable').shap.apply(lambda v: v.abs().mean()).sort_values().index.tolist()
    pos = {c: i for i, c in enumerate(orden)}
    jitter = np.random.default_rng(0).normal(0, .12, len(s))
    f = go.Figure(go.Scattergl(x=s.shap, y=s.variable.map(pos) + jitter, mode='markers',
                               marker={'size': 4, 'color': s.color, 'colorscale': [[0, BASE], [1, HOT]], 'opacity': .75,
                                       'colorbar': {'title': T('Valor'), 'tickvals': [0, 1], 'ticktext': [T('bajo'), T('alto')], 'thickness': 10}},
                               hovertemplate='SHAP %{x:+.3f}<extra></extra>'))
    f.update_yaxes(tickvals=list(range(len(orden))), ticktext=[etiquetas(c) for c in orden]); f.update_xaxes(title=T('Valor SHAP (cambio en la probabilidad de abandono)'))
    f.add_vline(x=0, line_color=GREY)
    return f


def bars_ci(nombres, media, bajo, alto, resaltar, titulo_x):
    """Barras horizontales con intervalo (puntos con bigotes asimétricos)."""
    f = go.Figure(go.Scatter(x=media, y=nombres, mode='markers', marker={'size': 10, 'color': [HOT if n == resaltar else BASE for n in nombres]},
                             error_x={'type': 'data', 'symmetric': False, 'array': np.asarray(alto) - np.asarray(media), 'arrayminus': np.asarray(media) - np.asarray(bajo), 'color': GREY},
                             hovertemplate='%{y}: %{x:.4f}<extra></extra>'))
    f.update_xaxes(title=titulo_x); f.update_yaxes(autorange='reversed')
    return f


__all__ = ['BASE', 'HOT', 'GREY', 'LOW', 'MID', 'HIGH', 'SEQ', 'DIVERGING', 'GRID', 'highlight', 'style', 'roc_animation', 'gauge',
           'confusion', 'roc', 'precision_recall', 'reliability', 'cd_diagram', 'shap_waterfall', 'shap_beeswarm', 'bars_ci', 'make_subplots', 'px', 'go']
