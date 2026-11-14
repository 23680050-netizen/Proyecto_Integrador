import base64
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from scipy import stats
from sklearn.metrics import mean_squared_error

st.set_page_config(page_title="Agronomy", layout="wide")

VERDE, OSCURO, CLARO = "#558247", "#2E723A", "#E6EFE1"
PAL = {"Tratamiento_A": "#A9CC93", "Tratamiento_B": "#2E723A"}
PAGINAS = [("telemetria", "Telemetría SQL"), ("regresion", "Regresión y Correlación"),
           ("inferencia", "Inferencia de Tratamientos"), ("despliegue", "Despliegue de Resultados"),
           ("graficas", "Gráficas y Mapas")]
TITULO = "Optimización Inferencial del Rendimiento Agrícola y Eficiencia de Riego en Cultivos de Invernadero"
sns.set_theme(style="whitegrid")


# ---------------------------------------------------------------- datos (Hito 1)
@st.cache_data
def generar_datos():
    con = duckdb.connect(database=':memory:')
    con.execute("SELECT setseed(0.42);")
    con.execute("""
    CREATE TABLE telemetria AS
    WITH base AS (
        SELECT parcela_id,
               ROUND(100 + random() * 200, 2) AS volumen_riego,
               ROUND(50 + random() * 100, 2) AS fertilizante_kg,
               CASE WHEN random() < 0.5 THEN 'Tratamiento_A' ELSE 'Tratamiento_B' END AS tipo_tratamiento
        FROM generate_series(1, 100) AS t(parcela_id)
    )
    SELECT *, ROUND(5.0 + 0.035 * volumen_riego + 0.020 * fertilizante_kg
                    + CASE WHEN tipo_tratamiento = 'Tratamiento_B' THEN 1.8 ELSE 0.0 END
                    + (random() * 1.6 - 0.8), 2) AS rendimiento_ton
    FROM base;
    """)
    return con.execute("SELECT * FROM telemetria WHERE rendimiento_ton IS NOT NULL").df()


# ---------------------------------------------------------------- cálculos (Hitos 2 y 3)
def modelo(d):
    X = sm.add_constant(d[["volumen_riego", "fertilizante_kg"]])
    return sm.OLS(d["rendimiento_ton"], X).fit()


def rmse(d, m):
    return float(np.sqrt(mean_squared_error(d["rendimiento_ton"], m.fittedvalues)))


def prueba_t(d):
    a = d.loc[d.tipo_tratamiento == "Tratamiento_A", "rendimiento_ton"]
    b = d.loc[d.tipo_tratamiento == "Tratamiento_B", "rendimiento_ton"]
    if len(a) < 2 or len(b) < 2:
        return None
    t, p = stats.ttest_ind(b, a, alternative="greater")  # H1: mu_B - mu_A > 0
    gl = len(a) + len(b) - 2
    sp2 = ((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / gl
    ee = np.sqrt(sp2 * (1 / len(a) + 1 / len(b)))
    dif, tc = b.mean() - a.mean(), stats.t.ppf(0.975, gl)
    return dict(t=t, p=p, dif=dif, ic=(dif - tc * ee, dif + tc * ee), gl=gl, ma=a.mean(), mb=b.mean())


def fmt_p(p):
    return "< 0.0001" if p < 0.0001 else f"{p:.4f}"


# ---------------------------------------------------------------- gráficas (Hito 4)
def g_heatmap(d, ax):
    sns.heatmap(d[["volumen_riego", "fertilizante_kg", "rendimiento_ton"]].corr(), annot=True, fmt=".2f",
                cmap=sns.light_palette(VERDE, as_cmap=True), vmin=0, vmax=1, ax=ax)
    ax.set_title("Mapa de calor de correlación (Pearson)")


def g_dispersion(d, ax):
    sns.scatterplot(data=d, x="volumen_riego", y="rendimiento_ton", hue="tipo_tratamiento",
                    palette=PAL, edgecolor="white", ax=ax)
    sns.regplot(data=d, x="volumen_riego", y="rendimiento_ton", scatter=False, color=VERDE, ax=ax)
    ax.set_title("Riego vs rendimiento con recta OLS y banda de confianza")
    ax.set_xlabel("Volumen de riego (m³/ha)"); ax.set_ylabel("Rendimiento (ton/ha)")
    ax.legend(title="")


def g_cajas(d, ax):
    sns.boxplot(data=d, x="tipo_tratamiento", y="rendimiento_ton", hue="tipo_tratamiento",
                palette=PAL, legend=False, ax=ax)
    ax.set_title("Rendimiento por tratamiento")
    ax.set_xlabel("Tipo de tratamiento (A vs B)"); ax.set_ylabel("Rendimiento (ton/ha)")


def g_residuos(d, ax):
    r = modelo(d).resid
    sns.histplot(r, kde=True, stat="density", color="#9DC183", ax=ax)
    x = np.linspace(r.min(), r.max(), 200)
    ax.plot(x, stats.norm.pdf(x, 0, r.std()), "--", color=OSCURO, label="Normal teórica")
    ax.set_title("Histograma y densidad de residuos")
    ax.set_xlabel("Residuo = y real - y predicho"); ax.legend()


def mostrar(fn, d, size=(6, 4)):
    fig, ax = plt.subplots(figsize=size)
    fn(d, ax)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def tarjeta(titulo, valor):
    st.markdown(f"""<div style="background:{CLARO};border-left:6px solid {VERDE};padding:14px 16px;
        border-radius:6px;color:#2F3B2A"><div style="font-size:.9rem">{titulo}</div>
        <div style="font-size:1.6rem;font-weight:700;color:{OSCURO}">{valor}</div></div>""",
                unsafe_allow_html=True)


# ---------------------------------------------------------------- páginas
def p_inicio(d):
    st.markdown(f"<h2 style='text-align:center;margin-top:30px'>{TITULO}</h2>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center;font-size:1.05rem'>Proyecto Integrador U4 · Análisis y Exploración de Datos (INC-2505)</p>",
                unsafe_allow_html=True)
    st.markdown("<p style='text-align:center;max-width:760px;margin:10px auto 60px auto'>Monitoreo de invernaderos inteligentes: "
                "se simulan 100 parcelas con DuckDB, se modela el rendimiento con regresión lineal y se evalúa con una prueba t "
                "si el nuevo bio-estimulante (Tratamiento B) supera al método tradicional (Tratamiento A). "
                "Usa el menú superior para recorrer cada etapa.</p>", unsafe_allow_html=True)


def p_telemetria(d):
    st.subheader("Hito 1: Telemetría agroindustrial (DuckDB)")
    st.write("Datos simulados con semilla fija (`setseed(0.42)`): 100 parcelas con riego, fertilizante, tratamiento y rendimiento.")
    c = st.columns(4)
    for col, (t, v) in zip(c, [("Parcelas", len(d)), ("Tratamiento A", (d.tipo_tratamiento == "Tratamiento_A").sum()),
                               ("Tratamiento B", (d.tipo_tratamiento == "Tratamiento_B").sum()),
                               ("Rango de rendimiento", f"{d.rendimiento_ton.min():.1f} a {d.rendimiento_ton.max():.1f} ton/ha")]):
        with col:
            tarjeta(t, v)
    st.write("")
    st.write("**Primeros registros**")
    st.dataframe(d.head(10), use_container_width=True, hide_index=True)
    st.write("**Estadísticos descriptivos**")
    st.dataframe(d.drop(columns=["parcela_id"]).describe().round(2), use_container_width=True)


def p_regresion(d):
    m = modelo(d)
    st.subheader("Hito 2: Regresión lineal y correlación (Subtema 4.2)")
    corr = d[["volumen_riego", "fertilizante_kg", "rendimiento_ton"]].corr()
    st.write("**Matriz de correlación de Pearson (r)**")
    st.dataframe(corr.round(4).style.background_gradient(cmap="Greens", vmin=0, vmax=1), use_container_width=True)
    r = corr["rendimiento_ton"].drop("rendimiento_ton")
    st.info(f"El factor con mayor asociación positiva con el rendimiento es **{r.idxmax()}** (r = {r.max():.4f}).")
    c1, c2, c3 = st.columns(3)
    with c1:
        tarjeta("R² ajustado", f"{m.rsquared_adj * 100:.2f}%")
    with c2:
        tarjeta("RMSE", f"{rmse(d, m):.4f} ton/ha")
    with c3:
        tarjeta("Prueba F (p)", fmt_p(m.f_pvalue))
    st.write("")
    st.write("**Coeficientes y p-valores**")
    st.dataframe(pd.DataFrame({"coeficiente": m.params, "p-valor": m.pvalues}).round(4), use_container_width=True)
    st.code(f"Rendimiento = {m.params['const']:.3f} + ({m.params['volumen_riego']:.4f} * Riego) + "
            f"({m.params['fertilizante_kg']:.4f} * Fertilizante)")
    sig = all(m.pvalues < 0.05)
    st.write(f"Con un R² ajustado de {m.rsquared_adj * 100:.1f}%, el riego y el fertilizante explican ese porcentaje de la variabilidad "
             f"de la cosecha. Los coeficientes {'son' if sig else 'no todos son'} significativos (p < 0.05).")
    st.write("**Resumen OLS (statsmodels)**")
    st.text(str(m.summary()))


def p_inferencia(d):
    st.subheader("Hito 3: Inferencia estadística de tratamientos (Subtema 4.3)")
    st.write("**H₀:** μB − μA ≤ 0 (el nuevo tratamiento no incrementa el rendimiento promedio).  \n"
             "**H₁:** μB − μA > 0 (el Tratamiento B produce significativamente más rendimiento).  \n"
             "Prueba t de Student de dos muestras independientes, α = 0.05.")
    r = prueba_t(d)
    c = st.columns(4)
    for col, (t, v) in zip(c, [("Media A", f"{r['ma']:.2f} ton/ha"), ("Media B", f"{r['mb']:.2f} ton/ha"),
                               ("Estadístico t", f"{r['t']:.3f}"), ("Valor p", fmt_p(r["p"]))]):
        with col:
            tarjeta(t, v)
    st.write("")
    st.write(f"**Diferencia de medias (B − A):** {r['dif']:.3f} ton/ha  ·  **IC 95%:** [{r['ic'][0]:.3f}, {r['ic'][1]:.3f}]  ·  gl = {r['gl']}")
    if r["p"] < 0.05:
        st.success("Veredicto: se rechaza H₀ (p < 0.05). El Tratamiento B incrementa significativamente el rendimiento; "
                   "se recomienda autorizar la inversión para escalarlo a todo el invernadero.")
    else:
        st.warning("Veredicto: no se rechaza H₀ (p ≥ 0.05). No hay evidencia suficiente para autorizar la inversión en el Tratamiento B.")
    mostrar(g_cajas, d)


def p_despliegue(d):
    st.subheader("Hito 4: Despliegue de resultados")
    c1, c2 = st.columns(2)
    trat = c1.multiselect("Tratamiento", list(PAL), default=list(PAL))
    lo, hi = float(d.volumen_riego.min()), float(d.volumen_riego.max())
    rango = c2.slider("Volumen de riego (m³/ha)", lo, hi, (lo, hi))
    f = d[d.tipo_tratamiento.isin(trat) & d.volumen_riego.between(*rango)]
    if len(f) < 5:
        st.warning("Hay muy pocas parcelas con ese filtro. Amplía la selección.")
        return
    m, r = modelo(f), prueba_t(f)
    k = st.columns(3)
    with k[0]:
        tarjeta("Rendimiento promedio", f"{f.rendimiento_ton.mean():.2f} ton/ha")
    with k[1]:
        tarjeta("R² ajustado del modelo", f"{m.rsquared_adj * 100:.2f}%")
    with k[2]:
        tarjeta("P-valor de la prueba t", fmt_p(r["p"]) if r else "N/A (se necesitan ambos tratamientos)")
    st.write("")
    for fa, fb in [(g_heatmap, g_dispersion), (g_cajas, g_residuos)]:
        a, b = st.columns(2)
        with a:
            mostrar(fa, f)
        with b:
            mostrar(fb, f)


def p_graficas(d):
    st.subheader("Gráficas y mapas")
    for fn, txt in [(g_heatmap, "Muestra el grado de asociación entre riego, fertilizante y rendimiento; permite identificar el factor dominante."),
                    (g_dispersion, "Representa el modelo de regresión lineal: puntos observados, recta de mínimos cuadrados y banda de confianza."),
                    (g_cajas, "Soporte visual de la prueba de hipótesis: medianas, cuartiles, valores atípicos y separación de medias."),
                    (g_residuos, "Valida que los errores se distribuyan de forma normal alrededor de cero.")]:
        st.write(txt)
        mostrar(fn, d, (8, 4))
    st.write("**Mapa esquemático de parcelas.** Rendimiento de cada parcela en una cuadrícula de 10 x 10 según `parcela_id` "
             "(es una disposición ilustrativa, no representa la ubicación real).")
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(d.sort_values("parcela_id")["rendimiento_ton"].to_numpy().reshape(10, 10), annot=True, fmt=".0f",
                cmap="YlGn", cbar_kws={"label": "ton/ha"}, ax=ax)
    ax.set_title("Rendimiento por parcela")
    st.pyplot(fig)
    plt.close(fig)


# ---------------------------------------------------------------- encabezado, estilo y navegación
def b64(nombre):
    return base64.b64encode((Path(__file__).parent / nombre).read_bytes()).decode()


def ir(k):
    st.session_state.pag = k


pag = st.session_state.setdefault("pag", "inicio")

st.markdown(f"""<style>
@import url('https://fonts.googleapis.com/css2?family=Dancing+Script:wght@600&display=swap');
header[data-testid="stHeader"], #MainMenu, footer {{visibility:hidden;height:0}}
.block-container{{max-width:1150px;padding:1rem 1rem 0 1rem}}
h2,h3{{color:{OSCURO}}}
.st-key-cab [data-testid="stVerticalBlock"], .st-key-cab [data-testid="stHorizontalBlock"]{{gap:0 !important}}
.izq{{background:#F0F0F0;border-left:8px solid {VERDE};height:218px;display:flex;align-items:center;padding-left:40px;gap:12px}}
.izq img{{width:83px;height:83px}}
.marca{{font-family:'Dancing Script',cursive;font-size:3.2rem;color:{OSCURO}}}
.foto{{height:134px;background-size:cover;background-position:center}}
.st-key-nav{{background:{VERDE};height:84px}}
.st-key-nav button{{background:{VERDE} !important;border:0 !important;border-radius:0 !important;height:84px;width:100%;box-shadow:none !important}}
.st-key-nav button p{{color:#fff !important;font-size:.92rem;line-height:1.25;white-space:normal;text-align:center}}
.st-key-nav [data-testid="stColumn"]:not(:first-child) button{{border-left:2px solid #fff !important}}
.st-key-nav button:hover{{background:#6B9A5B !important}}
.st-key-nav button[kind="primary"], .st-key-nav button[data-testid="stBaseButton-primary"]{{background:#3F6B34 !important}}
.st-key-nav button[kind="primary"] p, .st-key-nav button[data-testid="stBaseButton-primary"] p{{font-weight:700}}
.pie{{background:{VERDE};color:#fff;height:55px;display:flex;justify-content:space-between;align-items:center;padding:0 30px;margin-top:40px}}
</style>""", unsafe_allow_html=True)

with st.container(key="cab"):
    izq, der = st.columns([5, 6])
    with izq:
        st.markdown(f'<div class="izq"><img src="data:image/png;base64,{b64("logo.png")}"><span class="marca">agronomy</span></div>',
                    unsafe_allow_html=True)
    with der:
        st.markdown(f'<div class="foto" style="background-image:url(data:image/jpeg;base64,{b64("campo.jpg")})"></div>',
                    unsafe_allow_html=True)
        with st.container(key="nav"):
            for col, (k, txt) in zip(st.columns(5), PAGINAS):
                with col:
                    st.button(txt, key=f"b_{k}", on_click=ir, args=(k,), use_container_width=True,
                              type="primary" if pag == k else "secondary")

if pag != "inicio":
    st.button("Inicio", key="b_inicio", on_click=ir, args=("inicio",))

PAG_FN = {"inicio": p_inicio, "telemetria": p_telemetria, "regresion": p_regresion, "inferencia": p_inferencia,
          "despliegue": p_despliegue, "graficas": p_graficas}
PAG_FN[pag](generar_datos())

st.markdown('<div class="pie"><span>Instituto Tecnológico de Cuautla</span><span>Gil Toledano Nicol Abigail</span></div>',
            unsafe_allow_html=True)
