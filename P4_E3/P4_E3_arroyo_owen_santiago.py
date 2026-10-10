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
    from scipy import stats


@app.cell
def _():
    df = sns.load_dataset("titanic")
    df
    return (df,)


@app.cell
def _(df):
    df
    return


if __name__ == "__main__":
    app.run()
