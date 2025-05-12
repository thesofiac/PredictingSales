import pandas as pd
import numpy as np
import streamlit as st
from datetime import timedelta
from sklearn.ensemble import GradientBoostingRegressor

# Configuração da página
st.set_page_config(page_title="Previsão de Vendas", layout="centered")
st.title("🔮 Previsão de Vendas por Loja e Categoria")

@st.cache_data
def load_data():
    df = pd.read_csv("train.csv")
    df["date"] = pd.to_datetime(df["date"])
    df["sales"] = df["sales"].astype("float32")
    return df

@st.cache_data
def prepare_data(df):
    df = df.sort_values(["store_nbr", "family", "date"]).copy()
    df["dayofweek"] = df["date"].dt.dayofweek
    df["day"] = df["date"].dt.day
    df["month"] = df["date"].dt.month
    df["year"] = df["date"].dt.year
    df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)
    df["weekofyear"] = df["date"].dt.isocalendar().week.astype("int")
    df["dayofyear"] = df["date"].dt.dayofyear
    df["quarter"] = df["date"].dt.quarter
    df["is_month_end"] = df["date"].dt.is_month_end.astype(int)
    df["is_month_start"] = df["date"].dt.is_month_start.astype(int)

    for lag in [1, 7, 14]:
        df[f"lag_{lag}"] = df.groupby(["store_nbr", "family"])["sales"].shift(lag)

    for window in [7, 14]:
        df[f"rolling_mean_{window}"] = df.groupby(["store_nbr", "family"])["sales"].shift(1).rolling(window).mean()
        df[f"rolling_std_{window}"] = df.groupby(["store_nbr", "family"])["sales"].shift(1).rolling(window).std()

    df = df.dropna()
    return df

# Carregar e preparar dados
df = load_data()
df = prepare_data(df)

# Seletor interativo
store_options = ["all IDs"] + sorted(df["store_nbr"].unique().tolist())
family_options = ["all families"] + sorted(df["family"].unique().tolist())

store_selected = st.selectbox("Selecione o ID da loja:", store_options)
family_selected = st.selectbox("Selecione a categoria do produto:", family_options)

# Determinar previsão
features = [col for col in df.columns if col not in ["date", "sales", "store_nbr", "family"]]
modelo = GradientBoostingRegressor()

if store_selected == "all IDs" and family_selected == "all families":
    results = []
    for key, group in df.groupby(["store_nbr", "family"]):
        train = group.copy()
        train = train.sort_values("date")
        X_train = train[features]
        y_train = train["sales"]
        modelo.fit(X_train, y_train)
        last_row = train.iloc[-1:].copy()
        next_date = last_row["date"].values[0] + np.timedelta64(1, "D")
        last_row["date"] = next_date
        for col in ["day", "month", "year", "dayofweek", "dayofyear", "weekofyear"]:
            last_row[col] = pd.to_datetime(next_date).__getattribute__(col) if col != "weekofyear" else pd.to_datetime(next_date).isocalendar().week
        pred = modelo.predict(last_row[features])[0]
        results.append(pred)
    total = sum(results)
    st.metric("💼 Faturamento total previsto para o próximo dia", f"${total:,.2f}")

else:
    if store_selected != "all IDs" and family_selected != "all families":
        data = df[(df["store_nbr"] == store_selected) & (df["family"] == family_selected)]
        label = f"Loja {store_selected} - {family_selected}"
    elif store_selected != "all IDs":
        data = df[df["store_nbr"] == store_selected]
        label = f"Loja {store_selected} - Todas as categorias"
    elif family_selected != "all families":
        data = df[df["family"] == family_selected]
        label = f"Todas as lojas - {family_selected}"
    else:
        st.warning("Erro inesperado.")
        st.stop()

    predicoes = []
    for key, group in data.groupby(["store_nbr", "family"]):
        train = group.copy()
        train = train.sort_values("date")
        X_train = train[features]
        y_train = train["sales"]
        modelo.fit(X_train, y_train)
        last_row = train.iloc[-1:].copy()
        next_date = last_row["date"].values[0] + np.timedelta64(1, "D")
        last_row["date"] = next_date
        for col in ["day", "month", "year", "dayofweek", "dayofyear", "weekofyear"]:
            last_row[col] = pd.to_datetime(next_date).__getattribute__(col) if col != "weekofyear" else pd.to_datetime(next_date).isocalendar().week
        pred = modelo.predict(last_row[features])[0]
        predicoes.append(pred)

    total = sum(predicoes)
    st.metric(f"📦 Previsão de vendas para {label}", f"${total:,.2f}")
