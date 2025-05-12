import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

# Carregar os dados
@st.cache_data
def load_data():
    train = pd.read_csv("train.csv")
    train["date"] = pd.to_datetime(train["date"])
    train["sales"] = train["sales"].astype("float32")
    return train.sort_values(["store_nbr", "family", "date"])

# Criar features EWM
def ewm_features(dataframe, alphas, lags):
    dataframe = dataframe.copy()
    for alpha in alphas:
        for lag in lags:
            dataframe[f'sales_ewm_alpha_{str(alpha).replace(".", "")}_lag_{lag}'] = (
                dataframe.groupby(["store_nbr", "family"])["sales"]
                .transform(lambda x: x.shift(lag).ewm(alpha=alpha).mean())
            )
    return dataframe

# Interface Streamlit
st.set_page_config(page_title="EWM Sales Visualization", layout="centered")
st.title("Visualização de Vendas com Média Móvel Exponencial")

# Carregar e processar dados
data = load_data()
alphas = [0.95]
lags = [16]
data = ewm_features(data, alphas, lags)

# Seleção de loja e família de produto
store_nbrs = sorted(data["store_nbr"].unique())
families = sorted(data["family"].unique())

store_nbr = st.selectbox("Selecione a loja (store_nbr):", store_nbrs)
family = st.selectbox("Selecione a categoria de produto (family):", families)

# Filtrar os dados
filtered = data[(data["store_nbr"] == store_nbr) & (data["family"] == family)].set_index("date")

# Plotar
st.subheader(f"Loja {store_nbr} - {family}")
fig, ax = plt.subplots(figsize=(10, 5))
filtered[["sales", "sales_ewm_alpha_095_lag_16"]].plot(ax=ax)
ax.set_title("Vendas Reais vs. EWM (α=0.95, lag=16)")
ax.set_ylabel("Vendas")
ax.grid(True)
st.pyplot(fig)

