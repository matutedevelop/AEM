import marimo

__generated_with = "0.23.6"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo
    import seaborn as sns
    import pandas as pd
    import altair as alt
    import numpy as np
    import statsmodels as sm
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
    df = data
    df
    return (df,)


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
    Tenemos un dataset relativamente pequeño, 15 variables y una de ellas es en casi en su totalidad  son valores nulos por lo que en primera instancia es una variable que podriamos dropear

    tenemos 3 variables ademas de las guardadas como bool que tienen unicamente 2 categorias, `alive` , `sex` y `survived`. `sex` esta correcto sus labels pero de alive y surviveed deberiamos de homogenizar a un solo tipo de datos

    - dropear `deck`
    - arreglar el tipos de datos
    """)
    return


@app.function
def fix_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    return df.astype(
        {
            "alive": bool,
            "survived": bool,
            "who": "category",
            "embarked": "category",
            "pclass": "category",
            "sex": "category",
            "embark_town": "category",
        }
    )


@app.function
def drop_columns(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop(columns=["deck", "alive"])


@app.function
def drop_nulls(df: pd.DataFrame) -> pd.DataFrame:
    return df.dropna(ignore_index=True)


@app.cell
def _(df):
    # aplicamos los cambios
    df_clean = df.pipe(fix_dtypes).pipe(drop_columns).pipe(drop_nulls)
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

    x_train, x_test, y_train, y_test = train_test_split(
        df_clean[FEATURES], df_clean[[TARGET_CAT, TARGET_NUM]], random_state=SEED
    )
    df_train = pd.concat([x_train, y_train], axis=1)
    df_train
    return CAT_FEATURES, NUM_FEATURES, TARGET_CAT, TARGET_NUM, df_train


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
        opacity=0.7
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


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## faltan las conclusiones
    """)
    return


if __name__ == "__main__":
    app.run()
