import marimo

__generated_with = "0.23.6"
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

    Para esta parte se trabaja con el dataset completo cargado con `sns.load_dataset("titanic")`,
    no con la partición de entrenamiento, porque el objetivo es inferencial (pruebas de hipótesis,
    intervalos de confianza y supuestos) y conviene usar todas las observaciones disponibles.

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
    # Solo se conservan los pasajeros con edad registrada (fare no tiene faltantes)
    rl_df = data[["age", "fare"]].dropna().reset_index(drop=True)
    rl_n = len(rl_df)
    rl_df.shape
    return rl_df, rl_n


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
    return (rl_desc,)


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
    return (rl_pearson,)


@app.cell(hide_code=True)
def _(data, rl_desc, rl_n, rl_pearson):
    _a, _f = rl_desc.loc["age"], rl_desc.loc["fare"]
    _r = rl_pearson.statistic
    mo.md(f"""
    ### Interpretación del análisis exploratorio

    **Faltantes.** `age` tiene {data["age"].isna().sum()} valores faltantes
    ({data["age"].isna().mean():.1%} de los {len(data)} pasajeros) y `fare` no tiene ninguno.
    Como la tarea no pide imputar en esta parte, el modelo se ajusta con los **{rl_n} pasajeros
    con edad registrada**, por lo que los grados de libertad serán $n-2 = {rl_n - 2}$.

    **Age.** Media de {_a["media"]:.2f} años y mediana de {_a["mediana"]:.1f}, con desviación estándar
    de {_a["desv. est."]:.2f}. Va de {_a["mín"]:.2f} (bebés) a {_a["máx"]:.0f} años y el 50% central
    está entre {_a["Q1"]:.1f} y {_a["Q3"]:.1f}. La asimetría es leve ({_a["asimetría"]:.2f}):
    la distribución es casi simétrica, con una pequeña cola derecha y un grupo de niños pequeños.

    **Fare.** Media de £{_f["media"]:.2f} contra una mediana de £{_f["mediana"]:.2f}: la media es más del
    doble de la mediana, señal de un sesgo positivo muy fuerte (asimetría de {_f["asimetría"]:.2f} y
    curtosis de {_f["curtosis"]:.2f}). La mayoría pagó menos de £{_f["Q3"]:.0f}, pero hay tarifas de
    hasta £{_f["máx"]:.2f} y {int((data["fare"] == 0).sum())} pasajeros con tarifa de cero.

    **Dispersión.** El gráfico no muestra una tendencia clara: los puntos se amontonan en tarifas bajas
    para todas las edades y los valores extremos de `fare` corresponden a pasajeros de edades muy distintas.

    **Correlación de Pearson.** $r = {_r:.4f}$ (p-value = {rl_pearson.pvalue:.4f}). La relación lineal
    entre tarifa y edad es **positiva pero muy débil**: es estadísticamente distinta de cero al 5%
    gracias al tamaño de muestra, pero $r^2 = {_r**2:.4f}$, es decir, `fare` comparte menos del 1%
    de su variabilidad con `age`. Desde aquí se anticipa que el modelo lineal tendrá un poder
    predictivo muy bajo.

    **Matriz de correlación.** `age` se relaciona más con `pclass` (−0.37), `sibsp` (−0.31) y `parch`
    (−0.19) que con `fare`. A su vez, `fare` está fuertemente asociada con `pclass` (−0.55). Esto sugiere
    que la clase del pasaje funciona como variable confusora: los pasajeros de primera clase son, en
    promedio, mayores y pagan más, y esa es la poca relación que alcanza a capturar `fare`.
    """)
    return


if __name__ == "__main__":
    app.run()