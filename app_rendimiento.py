# -*- coding: utf-8 -*-
"""
Dashboard interactivo - Sistema inteligente para analizar el rendimiento academico
Ejecutar:  python -m streamlit run app_rendimiento.py
"""

import re
import time
from pathlib import Path
from urllib.parse import quote

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score,
                             roc_curve, silhouette_score)
from sklearn.model_selection import (GridSearchCV, StratifiedKFold,
                                     cross_val_predict, train_test_split)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

SEED = 42
ID_COL, TARGET, NOTA_FINAL = "id_estudiante", "aprobo", "nota_final"
FEATURES = ["nota_parcial_1", "nota_parcial_2", "nota_practicas",
            "promedio_tareas", "asistencia_pct", "entregas_pct",
            "participaciones", "horas_estudio_sem", "cursos_repetidos"]
LABELS = {
    "nota_parcial_1": "Nota parcial 1", "nota_parcial_2": "Nota parcial 2",
    "nota_practicas": "Nota de prácticas", "promedio_tareas": "Promedio de tareas",
    "asistencia_pct": "% asistencia", "entregas_pct": "% entregas a tiempo",
    "participaciones": "Participaciones", "horas_estudio_sem": "Horas de estudio/sem",
    "cursos_repetidos": "Cursos repetidos",
}
C_OK, C_WARN, C_BAD, C_BLUE = "#2e9e5b", "#f0a030", "#d64545", "#2a7ab0"

st.set_page_config(page_title="Rendimiento Académico · IA",
                   page_icon=":material/school:", layout="wide")

# =========================================================================== #
# UI KIT: iconos SVG (estilo Lucide, trazo 2px) + CSS con microanimaciones
# =========================================================================== #
_ICONS = {
    "logo": '<path d="M22 10v6"/><path d="M2 10l10-5 10 5-10 5z"/><path d="M6 12v5c3 3 9 3 12 0v-5"/>',
    "chart": '<path d="M3 3v16a2 2 0 0 0 2 2h16"/><path d="M18 17V9"/><path d="M13 17V5"/><path d="M8 17v-3"/>',
    "target": '<circle cx="12" cy="12" r="10"/><circle cx="12" cy="12" r="6"/><circle cx="12" cy="12" r="2"/>',
    "clusters": '<circle cx="7" cy="7" r="3"/><circle cx="17" cy="9" r="3"/><circle cx="10" cy="17" r="3"/>',
    "sliders": '<line x1="21" x2="14" y1="4" y2="4"/><line x1="10" x2="3" y1="4" y2="4"/><line x1="21" x2="12" y1="12" y2="12"/><line x1="8" x2="3" y1="12" y2="12"/><line x1="21" x2="16" y1="20" y2="20"/><line x1="12" x2="3" y1="20" y2="20"/><line x1="14" x2="14" y1="2" y2="6"/><line x1="8" x2="8" y1="10" y2="14"/><line x1="16" x2="16" y1="18" y2="22"/>',
    "alert": '<path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3"/><path d="M12 9v4"/><path d="M12 17h.01"/>',
    "book": '<path d="M2 3h6a4 4 0 0 1 4 4v14a3 3 0 0 0-3-3H2z"/><path d="M22 3h-6a4 4 0 0 0-4 4v14a3 3 0 0 1 3-3h7z"/>',
    "users": '<path d="M16 21v-2a4 4 0 0 0-4-4H6a4 4 0 0 0-4 4v2"/><circle cx="9" cy="7" r="4"/><path d="M22 21v-2a4 4 0 0 0-3-3.87"/><path d="M16 3.13a4 4 0 0 1 0 7.75"/>',
    "check": '<path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/><path d="m9 11 3 3L22 4"/>',
    "calendar": '<rect width="18" height="18" x="3" y="4" rx="2"/><path d="M16 2v4"/><path d="M8 2v4"/><path d="M3 10h18"/>',
    "gauge": '<path d="m12 14 4-4"/><path d="M3.34 19a10 10 0 1 1 17.32 0"/>',
    "file": '<path d="M15 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V7z"/><path d="M14 2v4a2 2 0 0 0 2 2h4"/><path d="M10 9H8"/><path d="M16 13H8"/><path d="M16 17H8"/>',
    "clock": '<circle cx="12" cy="12" r="10"/><path d="M12 6v6l4 2"/>',
    "repeat": '<path d="m17 2 4 4-4 4"/><path d="M3 11v-1a4 4 0 0 1 4-4h14"/><path d="m7 22-4-4 4-4"/><path d="M21 13v1a4 4 0 0 1-4 4H3"/>',
    "database": '<ellipse cx="12" cy="5" rx="9" ry="3"/><path d="M3 5v14a9 3 0 0 0 18 0V5"/><path d="M3 12a9 3 0 0 0 18 0"/>',
    "traffic": '<rect x="8" y="2" width="8" height="20" rx="3"/><circle cx="12" cy="7" r="1"/><circle cx="12" cy="12" r="1"/><circle cx="12" cy="17" r="1"/>',
    "download": '<path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><path d="m7 10 5 5 5-5"/><path d="M12 15V3"/>',
    "save": '<path d="M15.2 3a2 2 0 0 1 1.4.6l3.8 3.8a2 2 0 0 1 .6 1.4V19a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2z"/><path d="M17 21v-7a1 1 0 0 0-1-1H8a1 1 0 0 0-1 1v7"/><path d="M7 3v4a1 1 0 0 0 1 1h7"/>',
}
_SVG_ATTR = ('xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" '
             'stroke-width="2" stroke-linecap="round" stroke-linejoin="round"')


def icon(name, size=20):
    return (f'<svg {_SVG_ATTR} width="{size}" height="{size}" stroke="currentColor">'
            f'{_ICONS[name]}</svg>')


def _mask(name):
    svg = f'<svg {_SVG_ATTR} stroke="black">{_ICONS[name]}</svg>'
    return 'url("data:image/svg+xml;utf8,' + quote(svg) + '")'


def _stg(prefix, suffix, n, step, base=0.0):
    """Escalona animation-delay segun el indice del hijo (efecto cascada)."""
    r = "".join(f"{prefix}:nth-child({i}){suffix}{{animation-delay:{base + (i - 1) * step:.2f}s}}"
                for i in range(1, n + 1))
    return r + f"{prefix}:nth-child(n+{n + 1}){suffix}{{animation-delay:{base + n * step:.2f}s}}"


_BAR, _HBAR, _LINE, _DOTS = (f'[class*="st-key-anim-{k}-"]' for k in ("bar", "hbar", "line", "dots"))
_PIE, _RAD, _GAU, _HEAT = (f'[class*="st-key-anim-{k}-"]' for k in ("pie", "radar", "gauge", "heat"))

_CHART_CSS = (
    # barras verticales: crecen desde la base, en cascada
    f'{_BAR} .barlayer .point path{{transform-box:fill-box;transform-origin:50% 100%;animation:growY .8s var(--ease) backwards}}'
    + _stg(f'{_BAR} .barlayer .point', ' path', 14, 0.045)
    + f'{_BAR} .barlayer .point text{{animation:fadeIn .6s .6s backwards}}'
    # barras horizontales: se revelan de izquierda a derecha
    + f'{_HBAR} .barlayer .point{{animation:reveal .9s var(--ease) backwards}}'
    + _stg(f'{_HBAR} .barlayer .point', '', 12, 0.07)
    + f'{_HBAR} .errorbars{{animation:fadeIn .6s .8s backwards}}'
    # lineas: se "dibujan" y luego aparecen los marcadores
    + f'{_LINE} .scatterlayer .js-line,{_LINE} .scatterlayer .js-fill{{animation:reveal 1.2s cubic-bezier(.4,0,.2,1) backwards}}'
    + f'{_LINE} .scatterlayer .points path{{transform-box:fill-box;transform-origin:center;animation:pop .5s cubic-bezier(.2,1.4,.4,1) backwards}}'
    + _stg(f'{_LINE} .scatterlayer .points path', '', 10, 0.08, 0.7)
    + f'{_LINE} .shapelayer path,{_LINE} .annotation{{animation:fadeIn .7s 1s backwards}}'
    # nubes de puntos: aparecen "saltando" con retardo pseudo-aleatorio
    + f'{_DOTS} .scatterlayer .points path{{transform-box:fill-box;transform-origin:center;animation:pop .6s cubic-bezier(.2,1.3,.4,1) backwards}}'
    + "".join(f'{_DOTS} .scatterlayer .points path:nth-child(7n+{k}){{animation-delay:{(k - 1) * 0.05:.2f}s}}'
              for k in range(1, 8))
    # anillo / torta
    + f'{_PIE} .pielayer .slice{{transform-box:fill-box;transform-origin:center;animation:pop .8s cubic-bezier(.2,1.2,.4,1) backwards}}'
    + _stg(f'{_PIE} .pielayer .slice', '', 8, 0.1)
    # radar
    + f'{_RAD} .polarlayer .scatterlayer .trace{{transform-box:fill-box;transform-origin:center;animation:radarIn .9s var(--ease) backwards}}'
    + _stg(f'{_RAD} .polarlayer .scatterlayer .trace', '', 8, 0.1)
    # velocimetro
    + f'{_GAU} .indicatorlayer{{transform-box:fill-box;transform-origin:50% 100%;animation:zoomIn .9s var(--ease) backwards}}'
    # mapas de calor / matrices
    + f'{_HEAT} .heatmaplayer,{_HEAT} .imagelayer{{animation:fadeIn .9s .1s backwards}}'
    + f'{_HEAT} .heatmaplayer text,{_HEAT} .annotation{{animation:fadeIn .6s .7s backwards}}'
)

_TAB_NAMES = ["chart", "target", "clusters", "sliders", "alert", "book"]
_ICON_CSS = (
    "".join(f'[data-baseweb="tab-list"] button[role="tab"]:nth-of-type({i + 1}){{--ic:{_mask(n)}}}'
            for i, n in enumerate(_TAB_NAMES))
    + f'[data-testid="stDownloadButton"] button{{--ic:{_mask("download")}}}'
    + f'[data-testid="stSidebar"] [data-testid="stButton"] button{{--ic:{_mask("save")}}}'
)

_BASE_CSS = """
:root{--ease:cubic-bezier(.2,.8,.2,1)}
@property --n{syntax:"<integer>";inherits:false;initial-value:0}
@keyframes rise{from{opacity:0;transform:translateY(14px)}}
@keyframes fadeIn{from{opacity:0}}
@keyframes pop{from{opacity:0;transform:scale(.3)}}
@keyframes growY{from{opacity:.2;transform:scaleY(0)}}
@keyframes reveal{from{clip-path:inset(0 100% 0 0)}to{clip-path:inset(0 0 0 0)}}
@keyframes zoomIn{from{opacity:0;transform:scale(.9)}}
@keyframes radarIn{from{opacity:0;transform:scale(.3)}}
@keyframes panelIn{from{opacity:0;transform:translateX(18px)}}
@keyframes cnt{from{--n:0}}
@keyframes drawLogo{from{stroke-dashoffset:1}to{stroke-dashoffset:0}}
@keyframes float{0%,100%{transform:translateY(0)}50%{transform:translateY(-3px)}}

/* Cambio de seccion (pestanas) */
[data-baseweb="tab-panel"]{animation:panelIn .45s var(--ease) backwards}
button[role="tab"]{display:flex;align-items:center;gap:.45rem;transition:color .25s,transform .25s var(--ease)}
button[role="tab"]:hover{transform:translateY(-1px)}
button[role="tab"]::before,[data-testid="stDownloadButton"] button::before,
[data-testid="stSidebar"] [data-testid="stButton"] button::before{content:"";flex:none;display:inline-block;
  width:17px;height:17px;background:currentColor;-webkit-mask:var(--ic) center/contain no-repeat;
  mask:var(--ic) center/contain no-repeat;transition:transform .35s var(--ease)}
[data-testid="stDownloadButton"] button::before,[data-testid="stSidebar"] [data-testid="stButton"] button::before{margin-right:8px;width:16px;height:16px}
button[role="tab"][aria-selected="true"]::before{transform:scale(1.18) rotate(-6deg)}
.stButton button,.stDownloadButton button{transition:transform .2s var(--ease),box-shadow .2s}
.stButton button:hover,.stDownloadButton button:hover{transform:translateY(-2px);box-shadow:0 6px 16px rgba(30,60,110,.15)}

/* Contenedor de cada grafico */
[class*="st-key-anim-"]{animation:rise .6s var(--ease) backwards;border-radius:14px;transition:box-shadow .3s}
[class*="st-key-anim-"]:hover{box-shadow:0 10px 28px rgba(30,60,110,.10)}
[data-testid="stMetric"]{animation:rise .5s var(--ease) backwards}

/* Cabecera */
.hero{display:flex;align-items:center;gap:16px;margin:0 0 .8rem;animation:rise .6s var(--ease) backwards}
.hero .logo{flex:none;width:58px;height:58px;border-radius:16px;display:grid;place-items:center;color:#fff;
  background:linear-gradient(135deg,#2a7ab0,#2e9e5b);box-shadow:0 8px 20px rgba(42,122,176,.35);animation:float 4.5s ease-in-out infinite}
.hero .logo svg path{stroke-dasharray:1;animation:drawLogo 1.3s var(--ease) .2s backwards}
.hero-title{font-size:1.9rem;font-weight:750;line-height:1.15}
.hero-sub{opacity:.7;font-size:.92rem;margin-top:3px}

/* KPIs con contador animado */
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:14px;margin:.4rem 0 1.1rem}
.kpi{display:flex;gap:14px;align-items:center;padding:14px 16px;border-radius:14px;border:1px solid rgba(127,140,160,.25);
  background:rgba(127,140,160,.07);animation:rise .6s var(--ease) var(--d) backwards;
  transition:transform .25s var(--ease),box-shadow .25s,border-color .25s}
.kpi:hover{transform:translateY(-3px);box-shadow:0 10px 24px rgba(30,60,110,.12);border-color:var(--c)}
.kpi .ico{flex:none;width:44px;height:44px;border-radius:12px;display:grid;place-items:center;color:var(--c);
  background:color-mix(in srgb,var(--c) 14%,transparent);transition:transform .35s var(--ease)}
.kpi:hover .ico{transform:rotate(-8deg) scale(1.12)}
.kpi .lbl{font-size:.78rem;opacity:.7;letter-spacing:.02em}
.kpi .val{font-size:1.65rem;font-weight:700;line-height:1.15}
.cnt{animation:cnt 1.4s var(--ease) var(--d) backwards;counter-reset:c var(--n)}
.cnt::after{content:counter(c)}
.pop{display:inline-block;animation:pop .7s cubic-bezier(.2,1.4,.4,1) .35s backwards}

/* Titulos de seccion, recomendaciones, sidebar */
.sec{display:flex;align-items:center;gap:10px;font-size:1.25rem;font-weight:650;margin:.8rem 0 .4rem;animation:rise .5s var(--ease) backwards}
.sec .si{width:34px;height:34px;border-radius:10px;display:grid;place-items:center;color:#2a7ab0;background:rgba(42,122,176,.12)}
.rec{display:flex;gap:12px;align-items:flex-start;padding:10px 14px;margin:8px 0;border-radius:12px;
  border:1px solid rgba(127,140,160,.25);background:rgba(127,140,160,.07);
  animation:rise .5s var(--ease) calc(var(--i)*.09s) backwards;transition:transform .25s var(--ease),border-color .25s}
.rec:hover{transform:translateX(4px);border-color:var(--c)}
.rec .ri{flex:none;width:30px;height:30px;border-radius:9px;display:grid;place-items:center;color:var(--c);
  background:color-mix(in srgb,var(--c) 14%,transparent)}
.brand{display:flex;align-items:center;gap:10px;font-weight:700;font-size:1.15rem;margin-bottom:.4rem}
.brand .b-ico{width:36px;height:36px;border-radius:10px;display:grid;place-items:center;color:#fff;
  background:linear-gradient(135deg,#2a7ab0,#2e9e5b)}
.slabel{display:flex;align-items:center;gap:8px;font-weight:600;margin:.9rem 0 .2rem;opacity:.85}

@media (prefers-reduced-motion:reduce){
  *,*::before,*::after{animation-duration:.01ms!important;animation-delay:0s!important;transition-duration:.01ms!important}
}
"""
st.markdown(f"<style>{_BASE_CSS}{_ICON_CSS}{_CHART_CSS}</style>", unsafe_allow_html=True)

_LOGO = (f'<svg {_SVG_ATTR} width="32" height="32" stroke="currentColor">'
         '<path pathLength="1" d="M22 10v6"/><path pathLength="1" d="M2 10l10-5 10 5-10 5z"/>'
         '<path pathLength="1" d="M6 12v5c3 3 9 3 12 0v-5"/></svg>')


def hero():
    st.markdown(
        f'<div class="hero"><div class="logo">{_LOGO}</div><div>'
        '<div class="hero-title">Sistema inteligente de rendimiento académico</div>'
        '<div class="hero-sub">Clasificación (aprobar / desaprobar) y segmentación de estudiantes · '
        'Regresión Logística vs Random Forest · K-Means</div></div></div>',
        unsafe_allow_html=True)


def section(icon_name, text):
    st.markdown(f'<div class="sec"><span class="si">{icon(icon_name, 20)}</span>'
                f'<span>{text}</span></div>', unsafe_allow_html=True)


def sb_label(icon_name, text):
    st.sidebar.markdown(f'<div class="slabel">{icon(icon_name, 17)}<span>{text}</span></div>',
                        unsafe_allow_html=True)


def _kpi(ic, label, value, color, delay, suffix="", static=None):
    if static is None:
        val = f'<span class="cnt" style="--n:{int(value)}"></span><span>{suffix}</span>'
    else:
        val = f'<span class="pop">{static}</span>'
    return (f'<div class="kpi" style="--d:{delay}s;--c:{color}"><div class="ico">{icon(ic, 22)}</div>'
            f'<div><div class="lbl">{label}</div><div class="val">{val}</div></div></div>')


# =========================================================================== #
# DATOS
# =========================================================================== #
@st.cache_data(show_spinner=False)
def generar_datos(n: int, seed: int = SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    arq = {"alto": (0.30, 15.0, 94, 92, 12, 0.05),
           "regular": (0.35, 12.0, 85, 78, 8, 0.25),
           "riesgo": (0.20, 8.5, 65, 50, 4, 0.90),
           "irregular": (0.15, 13.0, 60, 55, 5, 0.40)}
    nombres = list(arq)
    grupo = rng.choice(nombres, size=n, p=[arq[k][0] for k in nombres])
    filas = []
    for g in grupo:
        _, nota, asis, entr, horas, lam = arq[g]
        p1 = np.clip(rng.normal(nota, 2.0), 0, 20)
        p2 = np.clip(rng.normal(nota + 0.3, 2.0), 0, 20)
        pr = np.clip(rng.normal(nota + 0.5, 2.2), 0, 20)
        ta = np.clip(rng.normal(nota + 0.8, 2.0), 0, 20)
        a = np.clip(rng.normal(asis, 6), 20, 100)
        e = np.clip(rng.normal(entr, 9), 0, 100)
        part = rng.poisson(max(1.0, (a - 40) / 8))
        h = np.clip(rng.normal(horas, 2.0), 0, 30)
        rep = rng.poisson(lam)
        fin = (0.25 * p1 + 0.25 * p2 + 0.20 * pr + 0.15 * ta + 0.15 * (e / 5)
               + 0.02 * (a - 75) + rng.normal(0, 1.0))
        filas.append([p1, p2, pr, ta, a, e, part, h, rep, np.clip(fin, 0, 20)])
    df = pd.DataFrame(filas, columns=FEATURES + [NOTA_FINAL]).round(2)
    df.insert(0, ID_COL, [f"E{str(i).zfill(5)}" for i in range(1, n + 1)])
    return df


def plantilla_csv() -> bytes:
    return generar_datos(20).to_csv(index=False).encode("utf-8")


def preparar(df: pd.DataFrame, nota_min: float) -> pd.DataFrame:
    df = df.copy()
    if ID_COL not in df.columns:
        df.insert(0, ID_COL, [f"E{str(i).zfill(5)}" for i in range(1, len(df) + 1)])
    if TARGET not in df.columns:
        df[TARGET] = (df[NOTA_FINAL] >= nota_min).astype(int)
    df["resultado"] = np.where(df[TARGET] == 1, "Aprueba", "Desaprueba")
    return df


# =========================================================================== #
# MODULO SUPERVISADO
# =========================================================================== #
def _metricas(nombre, mod, X_te, y_te):
    pred = mod.predict(X_te)
    prob = mod.predict_proba(X_te)[:, 1]
    return {"Modelo": nombre,
            "Accuracy": accuracy_score(y_te, pred),
            "Precisión (aprueba)": precision_score(y_te, pred),
            "Recall (aprueba)": recall_score(y_te, pred),
            "Recall (desaprueba)": recall_score(y_te, pred, pos_label=0),
            "F1": f1_score(y_te, pred),
            "AUC": roc_auc_score(y_te, prob)}


@st.cache_data(show_spinner=False)
def entrenar_supervisado(df: pd.DataFrame):
    X, y = df[FEATURES], df[TARGET]
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=SEED)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    logreg = Pipeline([("imp", SimpleImputer(strategy="median")),
                       ("sc", StandardScaler()),
                       ("clf", LogisticRegression(max_iter=1000,
                                                  class_weight="balanced",
                                                  random_state=SEED))]).fit(X_tr, y_tr)
    rf_pipe = Pipeline([("imp", SimpleImputer(strategy="median")),
                        ("clf", RandomForestClassifier(class_weight="balanced",
                                                       random_state=SEED, n_jobs=-1))])
    gs = GridSearchCV(rf_pipe,
                      {"clf__n_estimators": [200, 400],
                       "clf__max_depth": [None, 6, 10],
                       "clf__min_samples_leaf": [1, 5]},
                      scoring="roc_auc", cv=cv, n_jobs=-1).fit(X_tr, y_tr)
    rf = gs.best_estimator_

    knn = Pipeline([("imp", SimpleImputer(strategy="median")),
                    ("sc", StandardScaler()),
                    ("clf", KNeighborsClassifier(n_neighbors=15))]).fit(X_tr, y_tr)
    t0 = time.perf_counter(); knn.predict(X_te); t_knn = time.perf_counter() - t0
    t0 = time.perf_counter(); rf.predict(X_te); t_rf = time.perf_counter() - t0

    tabla = pd.DataFrame([
        _metricas("Regresión Logística (baseline)", logreg, X_te, y_te),
        _metricas("Random Forest", rf, X_te, y_te),
        _metricas("KNN (solo referencia)", knn, X_te, y_te)]).set_index("Modelo")

    roc, cms = {}, {}
    for nom, mod in [("Regresión Logística", logreg), ("Random Forest", rf)]:
        fpr, tpr, _ = roc_curve(y_te, mod.predict_proba(X_te)[:, 1])
        roc[nom] = (fpr, tpr, roc_auc_score(y_te, mod.predict_proba(X_te)[:, 1]))
        cms[nom] = confusion_matrix(y_te, mod.predict(X_te))

    perm = permutation_importance(rf, X_te, y_te, scoring="roc_auc",
                                  n_repeats=20, random_state=SEED, n_jobs=-1)
    imp = pd.DataFrame({
        "Variable": [LABELS[f] for f in FEATURES],
        "Impureza (Gini)": rf.named_steps["clf"].feature_importances_,
        "Permutación (caída de AUC)": perm.importances_mean,
        "Desv": perm.importances_std})
    coef = pd.Series(logreg.named_steps["clf"].coef_[0],
                     index=[LABELS[f] for f in FEATURES])

    # Probabilidades fuera de muestra (honestas) para el panel de riesgo
    oof = cross_val_predict(rf, X, y, cv=cv, method="predict_proba", n_jobs=-1)[:, 1]
    return dict(tabla=tabla, roc=roc, cms=cms, imp=imp, coef=coef, rf=rf,
                logreg=logreg, params=gs.best_params_, oof=oof,
                t_knn=t_knn, t_rf=t_rf, n_train=len(X_tr), n_test=len(X_te))


# =========================================================================== #
# MODULO NO SUPERVISADO
# =========================================================================== #
def _codo(ks, inercias):
    x = (np.array(ks) - ks[0]) / (ks[-1] - ks[0])
    y = (np.array(inercias) - inercias[-1]) / (inercias[0] - inercias[-1])
    return int(ks[int(np.argmax(np.abs(x + y - 1)))])


def _nombrar(z: pd.DataFrame) -> dict:
    """Nombres automaticos a partir del perfil estandarizado de cada cluster."""
    notas = ["nota_parcial_1", "nota_parcial_2", "nota_practicas", "promedio_tareas"]
    comp = z[notas + ["asistencia_pct", "entregas_pct"]].mean(axis=1)
    orden = list(comp.sort_values(ascending=False).index)
    nombres, k = {}, len(orden)
    medios = orden[1:-1]
    for i, c in enumerate(orden):
        if i == 0:
            nombres[c] = "Alto rendimiento"
        elif i == k - 1:
            nombres[c] = "En riesgo"
        else:
            irregular = z.loc[c, notas].mean() > 0 and z.loc[c, "asistencia_pct"] < -0.3
            base = "Irregular (buen nivel, baja asistencia)" if irregular else "Regular"
            if len(medios) > 1 and not irregular:
                base += f" {chr(65 + medios.index(c))}"
            nombres[c] = base
    # evita nombres repetidos
    vistos = {}
    for c in orden:
        n = nombres[c]
        vistos[n] = vistos.get(n, 0) + 1
        if vistos[n] > 1:
            nombres[c] = f"{n} ({vistos[n]})"
    return nombres


@st.cache_data(show_spinner=False)
def entrenar_clusters(df: pd.DataFrame, k_manual: int | None):
    X = df[FEATURES]
    prep = Pipeline([("imp", SimpleImputer(strategy="median")),
                     ("sc", StandardScaler())])
    Xs = prep.fit_transform(X)

    ks = list(range(2, 11))
    inercias, sils = [], []
    for k in ks:
        km = KMeans(n_clusters=k, n_init=10, random_state=SEED).fit(Xs)
        inercias.append(km.inertia_)
        sils.append(silhouette_score(Xs, km.labels_))
    k_codo, k_sil = _codo(ks, inercias), ks[int(np.argmax(sils))]
    k_final = k_manual or k_sil

    modelo = Pipeline([("imp", SimpleImputer(strategy="median")),
                       ("sc", StandardScaler()),
                       ("km", KMeans(n_clusters=k_final, n_init=20,
                                     random_state=SEED))]).fit(X)
    labels = modelo.named_steps["km"].labels_
    z = pd.DataFrame(Xs, columns=FEATURES).groupby(labels).mean()
    nombres = _nombrar(z)

    pca = PCA(n_components=2, random_state=SEED)
    coords = pca.fit_transform(Xs)
    return dict(ks=ks, inercias=inercias, sils=sils, k_codo=k_codo, k_sil=k_sil,
                k_final=k_final, modelo=modelo, labels=labels, z=z,
                nombres=nombres, coords=coords,
                var_pca=float(pca.explained_variance_ratio_.sum()),
                sil_final=float(silhouette_score(Xs, labels)))


# =========================================================================== #
# UTILIDADES UI
# =========================================================================== #
KIND_BY_KEY = {"hist": "bar", "corr": "heat", "scatter": "dots", "metricas": "bar",
               "roc": "line", "cm_": "heat", "imp_": "hbar", "coef": "hbar",
               "codo": "line", "sil": "line", "pca": "dots", "apr_cluster": "bar",
               "heat_perfil": "heat", "radar": "radar", "gauge": "gauge",
               "pie_": "pie", "bar_riesgo": "bar"}


def show(fig, key, h=None):
    fig.update_layout(template="plotly_white", margin=dict(l=10, r=10, t=50, b=10))
    if h:
        fig.update_layout(height=h)
    kind = next((v for p, v in KIND_BY_KEY.items() if key.startswith(p)), "misc")
    slug = re.sub(r"[^a-zA-Z0-9_-]", "-", key)
    with st.container(key=f"anim-{kind}-{slug}"):   # clase CSS st-key-anim-<tipo>-...
        st.plotly_chart(fig, key=key)


def nivel(p, lo, hi):
    return "Alto" if p < lo else ("Medio" if p < hi else "Bajo")


COLOR_NIVEL = {"Alto": C_BAD, "Medio": C_WARN, "Bajo": C_OK}


# =========================================================================== #
# SIDEBAR
# =========================================================================== #
st.sidebar.markdown(f'<div class="brand"><span class="b-ico">{icon("logo", 20)}</span>'
                    '<span>Panel de control</span></div>', unsafe_allow_html=True)
sb_label("database", "Datos")
fuente = st.sidebar.radio("Fuente de datos",
                          ["Datos sintéticos de demostración", "Subir mi CSV"],
                          label_visibility="collapsed")
df_raw = None
if fuente.startswith("Datos"):
    n_syn = st.sidebar.slider("Número de estudiantes", 300, 3000, 800, 100)
    df_raw = generar_datos(n_syn)
else:
    up = st.sidebar.file_uploader("Archivo CSV", type="csv")
    st.sidebar.download_button("Descargar plantilla CSV", plantilla_csv(),
                               "plantilla_estudiantes.csv", "text/csv")
    if up is not None:
        df_raw = pd.read_csv(up)

sb_label("clusters", "Parámetros")
nota_min = st.sidebar.number_input("Nota mínima aprobatoria", 0.0, 20.0, 11.0, 0.5,
                                   help="Solo aplica si el CSV no trae la columna 'aprobo'.")
k_sel = st.sidebar.selectbox("Número de clusters (k)",
                             ["Automático (máx. silueta)", 2, 3, 4, 5, 6, 7, 8])
sb_label("traffic", "Semáforo de riesgo")
umbral_lo = st.sidebar.slider("Riesgo ALTO si prob. de aprobar <", 0.10, 0.60, 0.40, 0.05)
umbral_hi = st.sidebar.slider("Riesgo MEDIO si prob. de aprobar <", 0.50, 0.95, 0.70, 0.05)

if df_raw is None:
    hero()
    st.info("Sube un CSV en la barra lateral o elige los datos sintéticos de demostración.")
    st.stop()

faltan = [c for c in FEATURES if c not in df_raw.columns]
if faltan:
    st.error(f"Al CSV le faltan estas columnas: {faltan}")
    st.stop()
if TARGET not in df_raw.columns and NOTA_FINAL not in df_raw.columns:
    st.error(f"El CSV debe incluir '{TARGET}' (0/1) o '{NOTA_FINAL}' (0-20).")
    st.stop()

df = preparar(df_raw, nota_min)
if df[TARGET].nunique() < 2 or df[TARGET].value_counts().min() < 10:
    st.error("Se necesitan al menos 10 estudiantes en cada clase (aprueba / desaprueba).")
    st.stop()

with st.spinner("Entrenando modelos… (solo la primera vez)"):
    sup = entrenar_supervisado(df)
    clu = entrenar_clusters(df, None if isinstance(k_sel, str) else int(k_sel))

df["prob_aprobar"] = sup["oof"]
df["nivel_riesgo"] = [nivel(p, umbral_lo, umbral_hi) for p in df["prob_aprobar"]]
df["cluster"] = clu["labels"]
df["grupo"] = df["cluster"].map(clu["nombres"])

if st.sidebar.button("Guardar modelos (.joblib)"):
    carpeta = Path("modelos_app")
    carpeta.mkdir(exist_ok=True)
    joblib.dump(sup["rf"], carpeta / "clasificador_rf.joblib", compress=3)
    joblib.dump(sup["logreg"], carpeta / "clasificador_logreg.joblib", compress=3)
    joblib.dump(clu["modelo"], carpeta / "kmeans_pipeline.joblib", compress=3)
    st.sidebar.success(f"Modelos guardados en {carpeta.resolve()}")

# =========================================================================== #
# CABECERA + KPIs
# =========================================================================== #
hero()
_apr = round(df[TARGET].mean() * 100, 1)
_asi = round(df["asistencia_pct"].mean(), 1)
_frac = lambda v: f".{int(round((v - int(v)) * 10))}%"
_auc = sup["tabla"].loc["Random Forest", "AUC"]
st.markdown(
    '<div class="kpis">'
    + _kpi("users", "Estudiantes", len(df), C_BLUE, 0.0)
    + _kpi("check", "Tasa de aprobación", int(_apr), C_OK, 0.08, _frac(_apr))
    + _kpi("calendar", "Asistencia media", int(_asi), C_BLUE, 0.16, _frac(_asi))
    + _kpi("alert", "En riesgo alto", int((df["nivel_riesgo"] == "Alto").sum()), C_BAD, 0.24)
    + _kpi("gauge", "AUC Random Forest", 0, C_WARN, 0.32, static=f"{_auc:.3f}")
    + "</div>", unsafe_allow_html=True)

tabs = st.tabs(["Exploración", "Modelo supervisado", "Clusters",
                "Simulador", "Estudiantes en riesgo", "Guía"])

# --------------------------------------------------------------------------- #
# TAB 1 - EXPLORACION
# --------------------------------------------------------------------------- #
with tabs[0]:
    c1, c2 = st.columns(2)
    with c1:
        var = st.selectbox("Distribución de la variable", FEATURES,
                           format_func=LABELS.get, key="var_hist")
        fig = px.histogram(df, x=var, color="resultado", barmode="overlay",
                           nbins=30, opacity=0.65, marginal="box",
                           color_discrete_map={"Aprueba": C_OK, "Desaprueba": C_BAD},
                           labels={var: LABELS[var]})
        fig.update_layout(title=f"{LABELS[var]} según resultado")
        show(fig, "hist")
    with c2:
        corr = df[FEATURES + [TARGET]].corr().round(2)
        corr.index = corr.columns = [LABELS.get(c, "Aprobó") for c in corr.columns]
        fig = px.imshow(corr, text_auto=True, color_continuous_scale="RdBu_r",
                        zmin=-1, zmax=1, aspect="auto")
        fig.update_layout(title="Matriz de correlación")
        show(fig, "corr")

    sx, sy = st.columns(2)
    x_var = sx.selectbox("Eje X", FEATURES, index=4, format_func=LABELS.get)
    y_var = sy.selectbox("Eje Y", FEATURES, index=0, format_func=LABELS.get)
    fig = px.scatter(df, x=x_var, y=y_var, color="resultado", opacity=0.7, render_mode="svg",
                     color_discrete_map={"Aprueba": C_OK, "Desaprueba": C_BAD},
                     labels={x_var: LABELS[x_var], y_var: LABELS[y_var]},
                     hover_data=[ID_COL])
    fig.update_layout(title=f"{LABELS[y_var]} vs {LABELS[x_var]}")
    show(fig, "scatter", 430)

# --------------------------------------------------------------------------- #
# TAB 2 - SUPERVISADO
# --------------------------------------------------------------------------- #
with tabs[1]:
    section("target", "Baseline (Regresión Logística) vs modelo avanzado (Random Forest)")
    st.caption(f"Entrenamiento: {sup['n_train']} estudiantes · Prueba: {sup['n_test']} · "
               f"Hiperparámetros RF: {sup['params']}")

    t = sup["tabla"].drop(index="KNN (solo referencia)")
    met = t.reset_index().melt(id_vars="Modelo", var_name="Métrica", value_name="Valor")
    c1, c2 = st.columns([3, 2])
    with c1:
        fig = px.bar(met, x="Métrica", y="Valor", color="Modelo", barmode="group",
                     color_discrete_sequence=["#9aa5b1", C_BLUE], range_y=[0, 1.05],
                     text_auto=".3f")
        fig.update_layout(title="Métricas en el conjunto de prueba")
        show(fig, "metricas", 400)
    with c2:
        fig = go.Figure()
        for (nom, (fpr, tpr, auc)), col in zip(sup["roc"].items(), ["#9aa5b1", C_BLUE]):
            fig.add_scatter(x=fpr, y=tpr, mode="lines", name=f"{nom} (AUC={auc:.3f})",
                            line=dict(color=col, width=3))
        fig.add_scatter(x=[0, 1], y=[0, 1], mode="lines", name="Azar",
                        line=dict(color="gray", dash="dash"))
        fig.update_layout(title="Curvas ROC", xaxis_title="Falsos positivos",
                          yaxis_title="Verdaderos positivos",
                          legend=dict(orientation="h", y=-0.2))
        show(fig, "roc", 400)

    cm1, cm2 = st.columns(2)
    for col, (nom, cm) in zip([cm1, cm2], sup["cms"].items()):
        fig = px.imshow(cm, text_auto=True, color_continuous_scale="Blues",
                        x=["Pred. Desaprueba", "Pred. Aprueba"],
                        y=["Real Desaprueba", "Real Aprueba"])
        fig.update_layout(title=f"Matriz de confusión · {nom}", coloraxis_showscale=False)
        with col:
            show(fig, f"cm_{nom}", 340)

    st.divider()
    section("chart", "¿Qué factores explican el resultado? (Feature Importance)")
    imp = sup["imp"]
    i1, i2 = st.columns(2)
    d = imp.sort_values("Permutación (caída de AUC)")
    fig = px.bar(d, x="Permutación (caída de AUC)", y="Variable", orientation="h",
                 error_x="Desv", color_discrete_sequence=["#e07b39"])
    fig.update_layout(title="Importancia por permutación (más confiable)")
    with i1:
        show(fig, "imp_perm", 420)
    d = imp.sort_values("Impureza (Gini)")
    fig = px.bar(d, x="Impureza (Gini)", y="Variable", orientation="h",
                 color_discrete_sequence=[C_BLUE])
    fig.update_layout(title="Importancia por impureza (Gini)")
    with i2:
        show(fig, "imp_gini", 420)

    top = imp.sort_values("Permutación (caída de AUC)", ascending=False)["Variable"].head(3).tolist()
    st.success(f"**Lectura pedagógica:** los factores que más pesan en la predicción son "
               f"**{top[0]}**, **{top[1]}** y **{top[2]}**. Son las áreas donde conviene "
               f"concentrar las intervenciones.")

    cf = sup["coef"].sort_values()
    fig = px.bar(cf, orientation="h", color=cf.values > 0,
                 color_discrete_map={True: C_OK, False: C_BAD},
                 labels={"value": "Coeficiente (estandarizado)", "index": ""})
    fig.update_layout(title="Dirección del efecto (Regresión Logística): "
                            "verde = aumenta la prob. de aprobar", showlegend=False)
    show(fig, "coef", 380)

    with st.expander("¿Por qué se descarta KNN?"):
        k = sup["tabla"].loc["KNN (solo referencia)"]
        st.markdown(f"""
- **Costo de producción:** KNN guarda todo el dataset y calcula distancias en cada predicción
  (tiempo de predicción en prueba: KNN {sup['t_knn']*1000:.1f} ms vs RF {sup['t_rf']*1000:.1f} ms; el costo de KNN crece con los datos).
- **Sensible a escala y ruido:** variables poco relevantes distorsionan las distancias.
- **Poco explicable:** no entrega importancia de variables ni coeficientes, que es justo lo que
  necesita la institución.
- **Probabilidades toscas:** con k=15 solo produce múltiplos de 1/15, mal para rankear riesgo.
- En esta muestra su rendimiento fue parecido (AUC {k['AUC']:.3f}): se descarta por **explicabilidad y escalabilidad**, no por precisión.
""")

# --------------------------------------------------------------------------- #
# TAB 3 - CLUSTERS
# --------------------------------------------------------------------------- #
with tabs[2]:
    section("clusters", "Perfiles de estudiantes con K-Means")
    st.caption(f"k usado = {clu['k_final']} · k sugerido por silueta = {clu['k_sil']} · "
               f"por codo = {clu['k_codo']} · silueta del modelo final = {clu['sil_final']:.3f}")

    e1, e2 = st.columns(2)
    fig = go.Figure(go.Scatter(x=clu["ks"], y=clu["inercias"], mode="lines+markers",
                               line=dict(color=C_BLUE, width=3)))
    fig.add_vline(x=clu["k_codo"], line_dash="dash", line_color="gray",
                  annotation_text=f"Codo ≈ {clu['k_codo']}")
    fig.update_layout(title="Método del Codo", xaxis_title="k", yaxis_title="Inercia")
    with e1:
        show(fig, "codo", 340)
    fig = go.Figure(go.Scatter(x=clu["ks"], y=clu["sils"], mode="lines+markers",
                               line=dict(color="#e07b39", width=3)))
    fig.add_vline(x=clu["k_sil"], line_dash="dash", line_color="gray",
                  annotation_text=f"Máx. silueta k={clu['k_sil']}")
    fig.update_layout(title="Coeficiente de Silueta", xaxis_title="k",
                      yaxis_title="Silueta media")
    with e2:
        show(fig, "sil", 340)

    p1, p2 = st.columns([3, 2])
    dfp = pd.DataFrame(clu["coords"], columns=["PC1", "PC2"])
    dfp["Grupo"] = df["grupo"].values
    dfp["Estudiante"] = df[ID_COL].values
    fig = px.scatter(dfp, x="PC1", y="PC2", color="Grupo", hover_data=["Estudiante"],
                     opacity=0.8, render_mode="svg")
    fig.update_layout(title=f"Mapa de estudiantes (PCA, {clu['var_pca']:.0%} de la varianza)")
    with p1:
        show(fig, "pca", 440)

    perfil = df.groupby("grupo").agg(
        Estudiantes=(ID_COL, "count"), Aprobación=(TARGET, "mean"),
        **{LABELS[f]: (f, "mean") for f in FEATURES}).round(2)
    perfil["% del total"] = (perfil["Estudiantes"] / len(df) * 100).round(1)
    fig = px.bar(perfil.reset_index(), x="grupo", y="Aprobación", color="grupo",
                 text=perfil["Aprobación"].map("{:.0%}".format).values,
                 labels={"grupo": ""})
    fig.update_layout(title="Tasa de aprobación por grupo", showlegend=False,
                      yaxis_tickformat=".0%")
    with p2:
        show(fig, "apr_cluster", 440)

    z = clu["z"].rename(index=clu["nombres"]).rename(columns=LABELS)
    raw = (df.groupby("grupo")[FEATURES].mean().rename(columns=LABELS)
           .loc[z.index].round(1))
    fig = go.Figure(go.Heatmap(z=z.values, x=z.columns, y=z.index, text=raw.values,
                               texttemplate="%{text}", colorscale="RdYlGn", zmid=0,
                               colorbar=dict(title="z-score")))
    fig.update_layout(title="Perfil de cada grupo (color = posición relativa · número = promedio real)")
    show(fig, "heat_perfil", 120 + 90 * len(z))

    # Radar
    mn, mx = df[FEATURES].min(), df[FEATURES].max()
    fig = go.Figure()
    for g in sorted(df["grupo"].unique()):
        vals = ((df.loc[df["grupo"] == g, FEATURES].mean() - mn) / (mx - mn)).values
        fig.add_trace(go.Scatterpolar(r=list(vals) + [vals[0]],
                                      theta=[LABELS[f] for f in FEATURES] + [LABELS[FEATURES[0]]],
                                      name=g, fill="toself", opacity=0.45))
    fig.update_layout(title="Radar comparativo de grupos (escala 0-1 por variable)",
                      polar=dict(radialaxis=dict(range=[0, 1])))
    show(fig, "radar_grupos", 500)

    st.dataframe(perfil.style.format({"Aprobación": "{:.0%}"}))

# --------------------------------------------------------------------------- #
# TAB 4 - SIMULADOR
# --------------------------------------------------------------------------- #
with tabs[3]:
    section("sliders", "Simulador de un estudiante")
    st.caption("Ajusta los valores y observa la probabilidad de aprobar y el grupo asignado.")
    izq, der = st.columns([2, 3])
    med = df[FEATURES].median()
    vals = {}
    with izq:
        for f in FEATURES:
            lo, hi = float(df[f].min()), float(df[f].max())
            paso = 1.0 if f in ("participaciones", "cursos_repetidos") else 0.5
            vals[f] = st.slider(LABELS[f], lo, hi, float(np.clip(med[f], lo, hi)),
                                paso, key=f"sim_{f}")
    est = pd.DataFrame([vals])[FEATURES]
    prob = float(sup["rf"].predict_proba(est)[0, 1])
    prob_lr = float(sup["logreg"].predict_proba(est)[0, 1])
    cl = int(clu["modelo"].predict(est)[0])
    grupo = clu["nombres"][cl]
    nv = nivel(prob, umbral_lo, umbral_hi)

    with der:
        fig = go.Figure(go.Indicator(
            mode="gauge+number", value=prob * 100,
            number={"suffix": "%"}, title={"text": "Probabilidad de aprobar (Random Forest)"},
            gauge={"axis": {"range": [0, 100]}, "bar": {"color": "#333"},
                   "steps": [{"range": [0, umbral_lo * 100], "color": "#f4b6b6"},
                             {"range": [umbral_lo * 100, umbral_hi * 100], "color": "#fbe0b0"},
                             {"range": [umbral_hi * 100, 100], "color": "#b9e3c9"}]}))
        show(fig, "gauge", 300)
        m1, m2, m3 = st.columns(3)
        m1.metric("Nivel de riesgo", nv)
        m2.metric("Grupo asignado", grupo)
        m3.metric("Prob. (Reg. Logística)", f"{prob_lr:.0%}")

        mn, mx = df[FEATURES].min(), df[FEATURES].max()
        norm = lambda s: ((s - mn) / (mx - mn)).clip(0, 1).values
        base = df.loc[df["cluster"] == cl, FEATURES].mean()
        th = [LABELS[f] for f in FEATURES]
        fig = go.Figure()
        for nom, s, col in [("Estudiante simulado", est.iloc[0], C_BLUE),
                            (f"Promedio del grupo «{grupo}»", base, C_WARN)]:
            r = norm(s)
            fig.add_trace(go.Scatterpolar(r=list(r) + [r[0]], theta=th + [th[0]],
                                          name=nom, fill="toself", opacity=0.5,
                                          line=dict(color=col)))
        fig.update_layout(title="Comparación con su grupo", polar=dict(radialaxis=dict(range=[0, 1])))
        show(fig, "radar_sim", 400)

        recs = []
        if vals["asistencia_pct"] < 75:
            recs.append(("calendar", C_BAD, "Asistencia baja:",
                         "contactar al estudiante y revisar causas (trabajo, transporte, salud)."))
        if vals["entregas_pct"] < 70:
            recs.append(("file", C_WARN, "Pocas entregas a tiempo:",
                         "acordar un calendario de entregas y recordatorios."))
        if np.mean([vals["nota_parcial_1"], vals["nota_parcial_2"], vals["nota_practicas"]]) < nota_min:
            recs.append(("book", C_BAD, "Notas por debajo del mínimo:",
                         "ofrecer tutoría académica y reforzamiento."))
        if vals["horas_estudio_sem"] < 5:
            recs.append(("clock", C_WARN, "Pocas horas de estudio:",
                         "orientar en planificación y técnicas de estudio."))
        if vals["cursos_repetidos"] >= 2:
            recs.append(("repeat", C_BLUE, "Historial de repitencia:",
                         "derivar a acompañamiento académico."))
        if not recs:
            recs.append(("check", C_OK, "Sin alertas destacables:",
                         "mantener seguimiento regular."))
        st.markdown("**Recomendaciones sugeridas**")
        st.markdown("".join(
            f'<div class="rec" style="--i:{i};--c:{c}"><span class="ri">{icon(ic, 18)}</span>'
            f'<div><b>{t}</b> {d}</div></div>' for i, (ic, c, t, d) in enumerate(recs)),
            unsafe_allow_html=True)

# --------------------------------------------------------------------------- #
# TAB 5 - ESTUDIANTES EN RIESGO
# --------------------------------------------------------------------------- #
with tabs[4]:
    section("alert", "Panel de alerta temprana")
    st.caption("Las probabilidades provienen de validación cruzada (cada estudiante "
               "es evaluado por un modelo que no lo vio en el entrenamiento).")
    r1, r2 = st.columns([2, 3])
    cuenta = df["nivel_riesgo"].value_counts().reindex(["Alto", "Medio", "Bajo"]).fillna(0)
    fig = px.pie(values=cuenta.values, names=cuenta.index, hole=0.55, color=cuenta.index,
                 color_discrete_map=COLOR_NIVEL)
    fig.update_layout(title="Distribución por nivel de riesgo")
    with r1:
        show(fig, "pie_riesgo", 360)
    cross = (df.groupby(["grupo", "nivel_riesgo"]).size().reset_index(name="n"))
    fig = px.bar(cross, x="grupo", y="n", color="nivel_riesgo", color_discrete_map=COLOR_NIVEL,
                 category_orders={"nivel_riesgo": ["Alto", "Medio", "Bajo"]},
                 labels={"grupo": "", "n": "Estudiantes"})
    fig.update_layout(title="Riesgo por grupo de estudiantes")
    with r2:
        show(fig, "bar_riesgo", 360)

    filtro = st.multiselect("Filtrar por nivel de riesgo", ["Alto", "Medio", "Bajo"],
                            default=["Alto", "Medio"])
    tabla = (df[df["nivel_riesgo"].isin(filtro)]
             [[ID_COL, "prob_aprobar", "nivel_riesgo", "grupo", "asistencia_pct",
               "entregas_pct", "nota_parcial_1", "nota_parcial_2", "nota_practicas"]]
             .sort_values("prob_aprobar").reset_index(drop=True))
    st.dataframe(
        tabla,
        column_config={"prob_aprobar": st.column_config.ProgressColumn(
            "Prob. de aprobar", min_value=0.0, max_value=1.0, format="percent")})
    st.download_button("Descargar lista (CSV)", tabla.to_csv(index=False).encode("utf-8"),
                       "estudiantes_en_riesgo.csv", "text/csv")

# --------------------------------------------------------------------------- #
# TAB 6 - GUIA
# --------------------------------------------------------------------------- #
with tabs[5]:
    st.markdown(f"""
### Cómo leer este panel

**Modelo supervisado**
- La métrica clave es el **recall de "desaprueba"**: qué proporción de estudiantes realmente en riesgo detecta el modelo.
- La **probabilidad de aprobar** funciona como semáforo (umbrales ajustables en la barra lateral).
- La **importancia por permutación** indica qué variables, al desordenarse, más empeoran la predicción: ahí está el valor pedagógico.
- Importancia alta **no implica causalidad**.

**Clusters (no supervisado)**
- El **codo** y la **silueta** (cercana a 1 = grupos bien separados) orientan la elección de k; pueden diferir.
- Para intervenir, suele servir un k entre 3 y 5. Puedes forzarlo en la barra lateral.
- Los nombres de los grupos se asignan automáticamente según el perfil; revísalos con criterio pedagógico.

**Buenas prácticas**
- Usa solo variables disponibles **antes** de la decisión (evita fuga de datos).
- Valida con datos de semestres anteriores antes de usarlo en producción.
- Revisa sesgos y no uses el sistema para sanciones automáticas: es una herramienta de apoyo a la tutoría.

**Formato del CSV:** columnas `{', '.join(FEATURES)}` y `aprobo` (0/1) o `nota_final` (0-20). Opcional: `id_estudiante`.
""")
