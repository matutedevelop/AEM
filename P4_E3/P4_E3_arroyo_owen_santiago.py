# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "altair>=6.3.0",
#     "marimo>=0.25.1",
#     "matplotlib>=3.11.2",
#     "numpy>=2.5.4",
#     "pandas>=3.0.6",
#     "scikit-learn>=1.9.1",
#     "scipy>=1.18.1",
#     "seaborn>=0.13.2",
#     "statsmodels>=0.15.0",
# ]
# ///

import marimo

__generated_with = "0.25.1"
app = marimo.App(
    width="medium",
    layout_file="layouts/P4_E3_arroyo_owen_santiago.slides.json",
)

with app.setup:
    import marimo as mo
    import seaborn as sns
    import pandas as pd
    import altair as alt
    import numpy as np
    import statsmodels.api as sm
    from matplotlib import pyplot as plt
    from scipy import stats
    from sklearn.model_selection import train_test_split


@app.cell
def _():
    sns.set_theme(style="white", palette="RdGy", font="IBM Plex Mono")


    def ctheme():
        return {
            "config": {
                "theme": "powerbi",
                "background": "#ffffff",
                # "font": "Inter, system-ui, sans-serif",
                "font": " BlexMono Nerd Font Mono",
                "autosize": {"type": "fit", "contains": "padding"},
                "title": {
                    "color": "#111111",
                    "font": "Inter, system-ui, sans-serif",
                    "fontSize": 16,
                    "fontWeight": "bold",
                    "anchor": "middle",
                    "offset": 10,
                },
                "view": {
                    "height": 450,
                    "width": 800,
                    "stroke": "transparent",
                },
                "axis": {
                    "labelColor": "#333333",
                    "titleColor": "#111111",
                    "labelFontSize": 12,
                    "titleFontSize": 14,
                    "gridColor": "#e5e5e5",
                    "labelAngle": -45,
                },
            }
        }

        # alt.data_transformers.enable("vegafusion")


    alt.themes.register("ctheme", ctheme)
    alt.theme.enable("ctheme")
    return


@app.cell
def _():
    data = sns.load_dataset("titanic")
    df = data.copy(deep=True)
    df
    return data, df


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Estado de los datos
    """)
    return


@app.cell
def _(df):
    df.dtypes, df.shape
    return


@app.cell
def _(df):
    df.isna().sum()
    return


@app.cell
def _(df):
    df.nunique()
    return


@app.cell
def _(df):
    # checamos las filas con valores nulos  menos en `deck`

    df[df.drop(columns=["deck"]).isna().any(axis=1)]
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Tenemos un dataset relativamente pequeño, 15 variables y una de ellas es en casi en su totalidad  son valores nulos desde que es una variable categorica podemos imputarla como `unknown`

    tenemos 3 variables ademas de las guardadas como bool que tienen unicamente 2 categorias, `alive` , `sex` y `survived`. `sex` esta correcto sus labels pero de alive y surviveed deberiamos de homogenizar a un solo tipo de datos

    - dropear `deck`
    - arreglar el tipos de datos
    """)
    return


@app.function
def input_variables(df: pd.DataFrame) -> pd.DataFrame:
    df["deck"] = df["deck"].cat.add_categories(["unknown"]).fillna("unknown")
    return df


@app.function
def fix_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    return df.astype(
        {
            "alive": "category",
            "survived": bool,
            "who": "category",
            "embarked": "category",
            "pclass": "category",
            "sex": "category",
            "embark_town": "category",
        }
    )


@app.function
def drop_nulls(df: pd.DataFrame) -> pd.DataFrame:
    return df.dropna(ignore_index=True)


@app.cell
def _(df):
    # aplicamos los cambios
    df_clean = df.pipe(input_variables).pipe(fix_dtypes).pipe(drop_nulls)
    df_clean
    return (df_clean,)


@app.cell
def _(df_clean):
    # Preparar y separar los datos
    SEED = 76
    TARGET_CAT = "survived"
    TARGET_NUM = "age"
    NUM_FEATURES = list(
        set(df_clean.select_dtypes(exclude=["bool", "category"]).columns.tolist())
        - {TARGET_NUM}
    )
    CAT_FEATURES = list(
        set(df_clean.select_dtypes(include=["bool", "category"]).columns.tolist())
        - {TARGET_CAT}
    )
    FEATURES = NUM_FEATURES + CAT_FEATURES

    _x_train, _, _y_train, _ = train_test_split(
        df_clean[FEATURES], df_clean[[TARGET_CAT, TARGET_NUM]], random_state=SEED
    )
    df_train = pd.concat([_x_train, _y_train], axis=1)
    df_train
    return CAT_FEATURES, NUM_FEATURES, SEED, TARGET_CAT, TARGET_NUM, df_train


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Analisis de variables numericas
    """)
    return


@app.cell
def _(NUM_FEATURES, TARGET_NUM, df_train):
    df_train[NUM_FEATURES + [TARGET_NUM]].describe()
    return


@app.cell
def _(NUM_FEATURES, TARGET_CAT, df_train):
    sns.pairplot(
        df_train[NUM_FEATURES + [TARGET_CAT]], hue=TARGET_CAT, palette="RdGy"
    )
    return


@app.cell
def _(NUM_FEATURES, TARGET_NUM, df_train):
    alt.Chart(df_train[NUM_FEATURES + [TARGET_NUM]]).mark_circle(
        opacity=0.7
    ).encode(
        alt.X(alt.repeat("column"), type="quantitative"),
        alt.Y(alt.repeat("row"), type="quantitative"),
        color=alt.Color(
            "age:Q",
            scale=alt.Scale(scheme="viridis"),
            legend=alt.Legend(title="age"),
        ),
    ).properties(width=220, height=150).repeat(
        row=NUM_FEATURES, column=NUM_FEATURES[::-1]
    ).interactive()
    return


@app.cell
def _(NUM_FEATURES, TARGET_NUM, df_train):
    sns.heatmap(df_train[NUM_FEATURES + [TARGET_NUM]].corr(), cmap="coolwarm_r")
    return


@app.cell
def _(NUM_FEATURES, TARGET_CAT, df_train):
    _, _ax = plt.subplots(
        nrows=len(NUM_FEATURES), ncols=1, figsize=(10, len(NUM_FEATURES) * 15)
    )


    for i, num_col in enumerate(NUM_FEATURES):
        sns.boxplot(df_train, y=num_col, ax=_ax[i], hue=TARGET_CAT)

    plt.show()
    return


@app.cell
def _(NUM_FEATURES, TARGET_NUM, df_train):
    alt.Chart(df_train[NUM_FEATURES + [TARGET_NUM]]).mark_circle(
        opacity=0.7,
    ).encode(
        alt.X(alt.repeat("column"), type="quantitative"),
        alt.Y(f"{TARGET_NUM}:Q"),
    ).properties(width=250, height=180).repeat(column=NUM_FEATURES).interactive()
    return


@app.cell
def _(TARGET_NUM, df_train):
    alt.Chart(df_train).mark_bar().encode(
        alt.X(TARGET_NUM, bin=alt.Bin(maxbins=30)), y="count()"
    ).properties(title=f"histograma de age\n sesgo:{df_train[TARGET_NUM].skew()}")
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Descubrimientos del analisis de cuantitativas

    - No hay indicios de colinearidad entre las variables numericas
    - no hay relaciones lineales "Fuertes" entre las variabless numericas con el target numerico
    - el target numerico presenta sesgo positivo
    - los pasajeros que sobrevivieron presentan mayor varinza en sus respectivas features numericas
    - hay 2 diferentes escalas ~ (0,10) y (0,100)
    - las variables numericas aparentan ser buenos predictores para la target categorica (se ve cierta separabilidad en el pairplot)
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Analisis de Nominales
    """)
    return


@app.cell
def _(CAT_FEATURES, TARGET_CAT, TARGET_NUM, df_train):
    _df = df_train[CAT_FEATURES + [TARGET_CAT]].copy(deep=True)
    _df["normalized_num_target"] = (
        df_train[TARGET_NUM] - df_train[TARGET_NUM].min()
    ) / df_train[TARGET_NUM].max()

    _df[TARGET_CAT] = _df[TARGET_CAT].astype(str).str.lower()

    _charts = []

    for _cat_col in CAT_FEATURES:
        _hist_df = (
            _df[[_cat_col, TARGET_CAT]].value_counts(normalize=True).reset_index()
        )
        _line_df = (
            _df.groupby(_cat_col, observed=True)["normalized_num_target"]
            .mean()
            .reset_index()
        )
        _hist = (
            alt.Chart(_hist_df)
            .mark_bar()
            .encode(
                x=_cat_col,
                y="proportion",
                color=alt.Color(
                    f"{TARGET_CAT}:N",
                    legend=alt.Legend(
                        title=TARGET_CAT,
                        labelColor="black",
                        titleColor="black",
                    ),
                    scale=alt.Scale(
                        domain=["false", "true"], range=["#F45866", "#7EBDC2"]
                    ),
                ),
            )
            # .properties(width=600)
        )
        _line = (
            alt.Chart(_line_df)
            .mark_line(
                color="black",
                point=alt.OverlayMarkDef(
                    color="black", size=50, filled=False, fill="salmon"
                ),
            )
            .encode(x=_cat_col, y="normalized_num_target")
        )

        _chart = (_hist + _line).properties(
            title=f"proporciones de {_cat_col} y promedios normalizados de {TARGET_NUM}"
        )
        _charts.append(_chart)

    alt.vconcat(*_charts)
    return


@app.cell
def _(TARGET_CAT, df_train):
    _df = df_train[TARGET_CAT].value_counts(normalize=True).reset_index()
    alt.Chart(_df).mark_bar().encode(
        x="survived",
        y="proportion"
    )
    return


@app.cell
def _(df_train):
    # los graficos y los nombres de las variables como `embarked` y `embark_town` sugieren que  son colineares, vamo a checarlo

    (
        pd.crosstab(df_train["embark_town"], df_train["embarked"]),
        pd.crosstab(df_train["adult_male"], df_train["who"]),
        pd.crosstab(df_train["adult_male"], df_train["sex"]),
        pd.crosstab(df_train["alive"], df_train["survived"]),
        pd.crosstab(df_train["class"], df_train["pclass"]),
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Hallazgos de analisis de categoricas

    se habra advertido que no se analizo la relacion entre `age` y `survived` esto es una solucion deliberada dado que ambas son targets de dos modelos diferentes. No se ignora el hecho de que estas variables puedan estar considerablemente relacionadas. Es una decision del ejercicio

    - la mayoria de las clases no presentan desbalances graves con excepcion de `embarked` y `embark_town`, y `deck`
    - `embarked` y `embark_town` son colineares $\text{embarked} \iff \text{embark town}$ (dropear `embark_town`)
    - `adult_male` implica man en `sex` $x_i \in \text{adult male} \iff x_i \in \text{sex}_{\text{man}}$ (dropear `adult_male`)
    - `alive` y `survived` son colineares (dropear `alive`)
    - `class` y `pclass` son colineares (dropear `pclass`)
    - las siguientes variables tienen distribuciones sustancialmente diferentes de `survived`
        - `sex`
        - `alone`
        - `who`
        - `class`
        - `deck`
        - `embarked`
    - las esperanzas condionales de `age` e.g. $\mathbb{E}[\text{age}|\text{class}]$ son diferentes en las siguientes variables, lo que sugiere que podrian ser buenos predictores de `age`
        -  `deck`
        -  `class`
        -  `alone`
        -  `who`
        -  `sex` (mas debil)

        a continuacion completamos el preprocesamiento con lo anterior
    """)
    return


@app.function
def drop_columns(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop(columns=["alive", "adult_male", "embark_town", "pclass"])


@app.cell
def _(CAT_FEATURES, NUM_FEATURES, SEED, TARGET_CAT, TARGET_NUM, data):
    _df = (
        data.copy(deep=True)
        .pipe(input_variables)
        .pipe(fix_dtypes)
        .pipe(drop_nulls)
        .pipe(drop_columns)
    )
    _FEATURES = set(_df.columns.tolist()) - set([TARGET_CAT,TARGET_NUM])
    FEATURES_NUM = list(_FEATURES & set(NUM_FEATURES))
    FEATURES_CAT = list(_FEATURES & set(CAT_FEATURES))
    FEATURES_LIST = list(_FEATURES)
    x_train,x_test,y_train,y_test = train_test_split(_df[FEATURES_LIST],_df[[TARGET_CAT,TARGET_NUM]],random_state=SEED)

    y_train_cat,y_train_num = (y_train[[TARGET_CAT]],y_train[[TARGET_NUM]])
    y_test_cat,y_test_num = (y_test[[TARGET_CAT]],y_test[[TARGET_NUM]])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Parte 1: Regresión lineal

    **Objetivo:** predecir la edad (`age`) de los pasajeros a partir de la tarifa que pagaron (`fare`):

    $$\text{Age} = \beta_0 + \beta_1 \cdot \text{Fare} + \varepsilon$$

    Para esta parte se trabaja con el dataset completo cargado con `sns.load_dataset("titanic")`.

    ## 1. Análisis exploratorio de `age` y `fare`

    ### 1.1 Valores faltantes
    """)
    return


@app.cell
def _(data):
    _faltantes = data[["age", "fare"]].isna().sum()
    pd.DataFrame(
        {
            "faltantes": _faltantes,
            "porcentaje (%)": (_faltantes / len(data) * 100).round(2),
            "observados": data[["age", "fare"]].notna().sum(),
        }
    )
    return


@app.cell
def _(data):
    rl_df = data[["age", "fare"]].dropna().reset_index(drop=True)
    rl_n = len(rl_df)
    rl_df.shape
    return (rl_df,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### 1.2 Estadísticos descriptivos
    """)
    return


@app.cell
def _(data):
    _vars = data[["age", "fare"]]
    rl_desc = pd.DataFrame(
        {
            "n": _vars.count(),
            "media": _vars.mean(),
            "mediana": _vars.median(),
            "desv. est.": _vars.std(),
            "mín": _vars.min(),
            "Q1": _vars.quantile(0.25),
            "Q3": _vars.quantile(0.75),
            "máx": _vars.max(),
            "asimetría": _vars.skew(),
            "curtosis": _vars.kurt(),
        }
    ).round(3)
    rl_desc
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### 1.3 Gráficos exploratorios
    """)
    return


@app.cell
def _(data):
    _fig, _axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for _ax, _col, _color in zip(_axes, ["age", "fare"], ["#7EBDC2", "#F45866"]):
        sns.histplot(
            data[_col].dropna(), bins=30, stat="density", kde=True,
            color=_color, edgecolor="white", ax=_ax,
        )
        _ax.axvline(data[_col].mean(), color="black", ls="--", lw=1, label="media")
        _ax.axvline(data[_col].median(), color="black", ls=":", lw=1, label="mediana")
        _ax.set_title(f"Histograma y densidad de {_col}")
        _ax.legend()
    _fig.tight_layout()
    _fig
    return


@app.cell
def _(rl_df):
    _fig, _ax = plt.subplots(figsize=(8, 5))
    sns.scatterplot(rl_df, x="fare", y="age", alpha=0.5, color="#7EBDC2", edgecolor="black", ax=_ax)
    _ax.set_title("Dispersión: Fare vs Age")
    _ax.set_xlabel("Fare (libras esterlinas)")
    _ax.set_ylabel("Age (años)")
    _fig.tight_layout()
    _fig
    return


@app.cell
def _(data):
    _num = data[["age", "fare", "sibsp", "parch", "pclass", "survived"]]
    _fig, _ax = plt.subplots(figsize=(7, 5.5))
    sns.heatmap(
        _num.corr(), annot=True, fmt=".2f", cmap="coolwarm_r",
        vmin=-1, vmax=1, square=True, ax=_ax,
    )
    _ax.set_title("Matriz de correlación de Pearson")
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### 1.4 Coeficiente de correlación de Pearson entre `age` y `fare`
    """)
    return


@app.cell
def _(rl_df):
    rl_pearson = stats.pearsonr(rl_df["fare"], rl_df["age"])
    _ci = rl_pearson.confidence_interval(confidence_level=0.95)
    pd.DataFrame(
        {
            "r": [rl_pearson.statistic],
            "r²": [rl_pearson.statistic**2],
            "p-value": [rl_pearson.pvalue],
            "IC 95% inf": [_ci.low],
            "IC 95% sup": [_ci.high],
        }
    ).round(4)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Interpretación

    La variable `age` tiene 177 valores faltantes (19.9% de los 891 pasajeros) y `fare` no tiene ninguno,
    por lo que el modelo se ajusta con los 714 pasajeros que sí tienen edad registrada, con $n - 2 = 712$
    grados de libertad.

    La edad promedio es de 29.7 años con una mediana de 28, y su distribución es casi simétrica. En cambio,
    la tarifa promedio es de £32.20 pero la mediana es de £14.45: la mayoría pagó poco y unos cuantos
    pagaron mucho (hasta £512.33), por eso está muy sesgada a la derecha.

    En el diagrama de dispersión no se ve una tendencia clara entre tarifa y edad. La correlación de
    Pearson es $r = 0.096$ ($p = 0.010$), una relación positiva pero muy débil. Aunque es significativa
    al 5%, la tarifa casi no explica la edad ($r^2 = 0.009$), así que esperamos un modelo con poco poder
    predictivo.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 2. Ajuste del modelo por Máxima Verosimilitud (MLE)

    Suponiendo errores $\varepsilon_i \sim N(0, \sigma^2)$, la log-verosimilitud del modelo es:

    $$\ell(\beta_0, \beta_1, \sigma^2) = -\frac{n}{2}\ln(2\pi\sigma^2) - \frac{1}{2\sigma^2}\sum_{i=1}^{n}\left(y_i - \beta_0 - \beta_1 x_i\right)^2$$

    Maximizar $\ell$ respecto a $\beta_0$ y $\beta_1$ es lo mismo que minimizar la suma de cuadrados de los
    residuos, así que los coeficientes MLE coinciden con los de OLS. Lo único que cambia es la varianza:
    MLE usa $\hat\sigma^2 = SS_{res}/n$ y OLS usa $s^2 = SS_{res}/(n-2)$.
    """)
    return


@app.cell
def _(rl_df):
    from scipy import optimize as _optimize

    _y = rl_df["age"].to_numpy()
    _x = rl_df["fare"].to_numpy()


    def _neg_loglik(theta):
        # parámetros: beta0, beta1 y log(sigma) para que sigma sea positiva
        _b0, _b1, _log_sigma = theta
        return -np.sum(stats.norm.logpdf(_y - _b0 - _b1 * _x, loc=0, scale=np.exp(_log_sigma)))


    rl_mle = _optimize.minimize(
        _neg_loglik, x0=[_y.mean(), 0.0, np.log(_y.std())], method="BFGS", options={"gtol": 1e-8}
    )
    rl_mle.success, rl_mle.x
    return (rl_mle,)


@app.cell
def _(rl_df):
    rl_X = sm.add_constant(rl_df["fare"])
    rl_ols = sm.OLS(rl_df["age"], rl_X).fit()
    rl_b0, rl_b1 = rl_ols.params["const"], rl_ols.params["fare"]
    return rl_b0, rl_b1, rl_ols


@app.cell
def _(rl_b0, rl_b1, rl_mle, rl_ols):
    pd.DataFrame(
        {
            "MLE (scipy)": [rl_mle.x[0], rl_mle.x[1], -rl_mle.fun],
            "OLS (statsmodels)": [rl_b0, rl_b1, rl_ols.llf],
        },
        index=["β̂₀", "β̂₁", "log-verosimilitud"],
    ).round(4)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Modelo ajustado

    | Parámetro | Estimación |
    |---|---|
    | $\hat\beta_0$ | 28.7842 |
    | $\hat\beta_1$ | 0.0264 |
    | $\hat\sigma^2_{MLE} = SS_{res}/n$ | 208.7789 |
    | $s^2 = SS_{res}/(n-2)$ | 209.3653 |

    $$\widehat{\text{Age}} = 28.7842 + 0.0264 \cdot \text{Fare}$$

    **$\hat\beta_0 = 28.78$:** es la edad esperada de un pasajero que pagó £0. Queda cerca de la edad
    promedio porque la pendiente es casi cero.

    **$\hat\beta_1 = 0.0264$:** por cada libra adicional, la edad esperada sube 0.026 años (unos 10 días).
    Por cada £100 serían 2.6 años. Es un efecto muy pequeño y es solo una asociación: quienes pagan más
    suelen viajar en primera clase, donde los pasajeros son mayores.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 3. Inferencia estadística

    Para cada coeficiente se prueba:

    $$H_0: \beta_j = 0 \quad \text{vs} \quad H_1: \beta_j \neq 0$$

    con el estadístico de prueba y el intervalo de confianza al 95%:

    $$t = \frac{\hat\beta_j}{SE(\hat\beta_j)}, \qquad \hat\beta_j \pm t_{n-2,\,0.025} \cdot SE(\hat\beta_j)$$

    Los grados de libertad son $df = n - 2 = 714 - 2 = 712$ y el valor crítico es $t_{712,\,0.025} = 1.9633$.
    """)
    return


@app.cell
def _(rl_ols):
    rl_gl = int(rl_ols.nobs) - 2
    _t_crit = stats.t.ppf(0.975, rl_gl)

    _filas = []
    for _nombre, _param in [("β₀ (intercepto)", "const"), ("β₁ (fare)", "fare")]:
        _b = rl_ols.params[_param]
        _se = rl_ols.bse[_param]
        _t = _b / _se
        _p = 2 * stats.t.sf(abs(_t), rl_gl)
        _filas.append(
            {
                "coeficiente": _nombre,
                "estimación": round(_b, 4),
                "SE": round(_se, 4),
                "t": round(_t, 4),
                "p-value": f"{_p:.4g}",
                "IC 95% inf": round(_b - _t_crit * _se, 4),
                "IC 95% sup": round(_b + _t_crit * _se, 4),
            }
        )

    pd.DataFrame(_filas).set_index("coeficiente")
    return


@app.cell
def _(rl_ols):
    rl_ols.summary()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Interpretación

    | Coeficiente | Estimación | SE | t | p-value | IC 95% |
    |---|---|---|---|---|---|
    | $\beta_0$ | 28.7842 | 0.6476 | 44.44 | $1.5 \times 10^{-207}$ | [27.51, 30.06] |
    | $\beta_1$ | 0.0264 | 0.0102 | 2.58 | 0.0102 | [0.0063, 0.0465] |

    **$\beta_0$:** con $t = 44.44$ y un p-value prácticamente cero, se rechaza $H_0$ a cualquier nivel de
    significancia. Con 95% de confianza, la edad esperada para una tarifa de £0 está entre 27.5 y 30.1 años.

    **$\beta_1$:** con $t = 2.58$ y $p = 0.0102$, se rechaza $H_0$ al 5%, pero no al 1%. Sí existe una relación
    lineal significativa entre tarifa y edad, aunque es débil. El intervalo [0.0063, 0.0465] no incluye el
    cero, lo que confirma el resultado, y muestra que incluso en el mejor caso el efecto es pequeño: £100 más
    en la tarifa se asocian con, a lo mucho, 4.6 años más de edad.

    El p-value de $\beta_1$ es el mismo que el de la correlación de Pearson del paso 1, porque en una regresión
    simple las dos pruebas son equivalentes.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 4. Verificación de supuestos

    Todos los supuestos se revisan con los residuos del modelo, $e_i = y_i - \hat y_i$.

    ### 4.1 Linealidad

    Se grafican los residuos contra los valores predichos y se aplica la prueba RESET de Ramsey, con
    $H_0$: la relación entre `fare` y `age` es lineal.
    """)
    return


@app.cell
def _(rl_ols):
    rl_resid = rl_ols.resid
    rl_fitted = rl_ols.fittedvalues

    _fig, _ax = plt.subplots(figsize=(8, 5))
    sns.regplot(
        x=rl_fitted, y=rl_resid, lowess=True, ax=_ax,
        scatter_kws={"alpha": 0.4, "color": "#7EBDC2", "edgecolor": "black"},
        line_kws={"color": "#F45866"},
    )
    _ax.axhline(0, color="black", ls="--", lw=1)
    _ax.set_title("Residuos vs valores predichos")
    _ax.set_xlabel("Edad predicha")
    _ax.set_ylabel("Residuo")
    _fig.tight_layout()
    _fig
    return (rl_resid,)


@app.cell(hide_code=True)
def _(rl_ols):
    from statsmodels.stats.diagnostic import linear_reset as _linear_reset

    _reset = _linear_reset(rl_ols, power=3, use_f=True)
    pd.DataFrame(
        {"estadístico F": [float(_reset.fvalue)], "p-value": [float(_reset.pvalue)]},
        index=["RESET de Ramsey"],
    ).round(4)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    En la gráfica, casi todas las predicciones están entre 29 y 31 años porque la mayoría de las tarifas son
    bajas. La curva roja no es plana: para las tarifas más altas los residuos son negativos, es decir, el
    modelo sobreestima la edad de esos pasajeros. También se ve un límite en la parte de abajo porque la edad
    no puede ser negativa.

    La prueba RESET da $F = 4.79$ con $p = 0.0086 < 0.05$, así que se rechaza $H_0$: **no se cumple el
    supuesto de linealidad**. La relación entre tarifa y edad no es una recta, sobre todo por las tarifas
    extremas.

    ### 4.2 Normalidad de los errores

    $H_0$: los errores siguen una distribución normal. Se revisa con el Q-Q plot y las pruebas de
    Shapiro-Wilk y Jarque-Bera.
    """)
    return


@app.cell
def _(rl_resid):
    _fig, _ax = plt.subplots(figsize=(6, 5))
    sm.qqplot(rl_resid, line="s", ax=_ax, markerfacecolor="#7EBDC2", markeredgecolor="black", alpha=0.6)
    _ax.set_title("Q-Q plot de los residuos")
    _fig.tight_layout()
    _fig
    return


@app.cell
def _(rl_resid):
    _sw = stats.shapiro(rl_resid)
    _jb = stats.jarque_bera(rl_resid)
    pd.DataFrame(
        {
            "estadístico": [round(_sw.statistic, 4), round(_jb.statistic, 4)],
            "p-value": [f"{_sw.pvalue:.4g}", f"{_jb.pvalue:.4g}"],
        },
        index=["Shapiro-Wilk", "Jarque-Bera"],
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    En el Q-Q plot los residuos siguen la diagonal en el centro, pero se separan en las colas: la cola derecha
    queda por encima de la línea (residuos positivos más grandes de lo esperado) y la izquierda se aplana por
    el límite inferior de la edad.

    Shapiro-Wilk da $W = 0.9838$ ($p = 4.1 \times 10^{-7}$) y Jarque-Bera da $JB = 18.01$ ($p = 0.0001$). En
    las dos se rechaza $H_0$, así que **no se cumple la normalidad**. Aun así, la desviación es moderada
    (asimetría de 0.38) y con $n = 714$ las pruebas $t$ y los intervalos del paso 3 siguen siendo
    aproximadamente válidos por el teorema del límite central.

    ### 4.3 Homocedasticidad

    $H_0$: la varianza de los errores es constante. Se usa la gráfica de residuos vs predichos de 4.1 y la
    prueba de Breusch-Pagan.
    """)
    return


@app.cell
def _(rl_ols, rl_resid):
    from statsmodels.stats.diagnostic import het_breuschpagan as _het_bp

    _lm, _lm_p, _f, _f_p = _het_bp(rl_resid, rl_ols.model.exog)
    pd.DataFrame(
        {"estadístico LM": [_lm], "p-value": [_lm_p]},
        index=["Breusch-Pagan"],
    ).round(4)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    En la gráfica de 4.1 no se ve un embudo claro: la dispersión de los residuos es parecida en todo el
    rango, aunque a la derecha hay muy pocos puntos para juzgarlo bien.

    Breusch-Pagan da $LM = 1.50$ con $p = 0.2206 > 0.05$, así que no se rechaza $H_0$: **se cumple el
    supuesto de homocedasticidad**.

    ### 4.4 Independencia de los errores

    Se revisa con el estadístico de Durbin-Watson (un valor cercano a 2 indica que no hay autocorrelación) y
    con el gráfico ACF.
    """)
    return


@app.cell
def _(rl_resid):
    from statsmodels.stats.stattools import durbin_watson as _durbin_watson
    from statsmodels.graphics.tsaplots import plot_acf as _plot_acf

    _fig, _ax = plt.subplots(figsize=(8, 4))
    _plot_acf(rl_resid, lags=30, ax=_ax)
    _ax.set_title(f"ACF de los residuos (Durbin-Watson = {_durbin_watson(rl_resid):.3f})")
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    El estadístico de Durbin-Watson es 1.891, muy cercano a 2. En el ACF casi todos los rezagos quedan dentro
    de la banda de confianza; solo los rezagos 8 y 25 la rebasan apenas, lo cual es esperable por azar (con 30
    rezagos y 95% de confianza se esperan 1 o 2 fuera). **Se cumple la independencia**, lo cual tiene sentido
    porque cada fila es un pasajero distinto y los datos no tienen un orden en el tiempo.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 5. Métricas de desempeño del modelo

    $$R^2 = 1 - \frac{SS_{res}}{SS_{tot}}, \qquad R^2_{adj} = 1 - \frac{(1 - R^2)(n - 1)}{n - k - 1}$$

    $$RMSE = \sqrt{\frac{SS_{res}}{n - 2}}, \qquad MAE = \frac{1}{n}\sum_{i=1}^{n} |y_i - \hat y_i|$$

    con $n = 714$ y $k = 1$ predictor.
    """)
    return


@app.cell
def _(rl_ols, rl_resid):
    _y = rl_ols.model.endog
    _n = len(_y)
    _k = 1
    _ss_res = np.sum(rl_resid**2)
    _ss_tot = np.sum((_y - _y.mean()) ** 2)

    _r2 = 1 - _ss_res / _ss_tot
    _r2_adj = 1 - (1 - _r2) * (_n - 1) / (_n - _k - 1)
    _rmse = np.sqrt(_ss_res / (_n - 2))
    _mae = np.mean(np.abs(rl_resid))

    pd.DataFrame(
        {"valor": [_r2, _r2_adj, _rmse, _mae]},
        index=["R²", "R² ajustado", "RMSE (años)", "MAE (años)"],
    ).round(4)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    **$R^2 = 0.0092$:** el modelo explica apenas el 0.92% de la variabilidad de la edad.

    **$R^2_{adj} = 0.0078$:** es menor que $R^2$ porque penaliza por el número de predictores. Con un solo
    predictor y $n = 714$ la diferencia es muy pequeña.

    **RMSE = 14.47 años:** es el error típico de predicción. Es prácticamente igual a la desviación estándar
    de la edad (14.53), es decir, el modelo casi no mejora a simplemente predecir la edad promedio.

    **MAE = 11.25 años:** en promedio, la predicción se equivoca por 11.2 años. Si siempre se predijera la
    edad promedio, el MAE sería de 11.32, casi lo mismo.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 6. Conclusiones de la regresión lineal

    **¿Existe relación significativa entre tarifa y edad?** Sí, al 5%: $\hat\beta_1 = 0.0264$ con
    $p = 0.0102$. Pero es una relación muy débil, ya que la tarifa explica menos del 1% de la variación de la edad.

    **¿Se cumplen los supuestos?**

    | Supuesto | Prueba | Resultado |
    |---|---|---|
    | Linealidad | RESET ($p = 0.0086$) | No se cumple |
    | Normalidad | Shapiro-Wilk y Jarque-Bera ($p < 0.001$) | No se cumple |
    | Homocedasticidad | Breusch-Pagan ($p = 0.2206$) | Se cumple |
    | Independencia | Durbin-Watson = 1.891 | Se cumple |

    La falta de linealidad indica que una recta no describe bien la relación, sobre todo para las tarifas
    extremas. La falta de normalidad es moderada y, con $n = 714$, las pruebas $t$ siguen siendo
    aproximadamente válidas, aunque los p-values deben tomarse con cautela.

    **¿Qué tan confiable es para predecir?** Muy poco. Con un RMSE de 14.47 años y un MAE de 11.25 años,
    el modelo predice casi igual que usar la edad promedio para todos. No sirve para predecir la edad de un pasajero.

    **Mejoras sugeridas:**

    - Transformar `fare` con logaritmo para reducir el sesgo y el peso de las tarifas extremas.
    - Agregar predictores más relacionados con la edad, como `pclass`, `sibsp`, `parch` y `sex`.
    - Imputar `age` en lugar de eliminar los 177 registros faltantes.
    - Probar una regresión robusta, menos sensible a los valores extremos de `fare`.
    """)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
