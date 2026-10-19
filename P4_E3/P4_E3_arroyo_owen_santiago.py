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
app = marimo.App(width="medium")

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


if __name__ == "__main__":
    app.run()
