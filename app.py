import streamlit as st
import pandas as pd

st.set_page_config(page_title="Consulta de Vendas", layout="centered")
st.title("📊 Consulta de Vendas por Loja e Categoria")

# Carregar arquivos
@st.cache_data
def load_data():
    df_cat = pd.read_csv("grouped_by_families.txt", sep=',', engine='python')
    df_loja = pd.read_csv("grouped_by_stores.txt", sep=',', engine='python')
    df_loja_cat = pd.read_csv("grouped_by_stores_families.txt", sep=',', engine='python')
    return df_cat, df_loja, df_loja_cat

df_cat, df_loja, df_loja_cat = load_data()

# Seletores
store_ids = sorted(df_loja_cat["ID_loja"].unique())
categories = sorted(df_loja_cat["categoria"].unique())

store_selected = st.selectbox("Selecione o ID da loja:", store_ids)
category_selected = st.selectbox("Selecione a categoria do produto:", categories)

# Consulta
resultado = df_loja_cat[
    (df_loja_cat["ID_loja"] == store_selected) &
    (df_loja_cat["categoria"] == category_selected)
]

# Exibição
if not resultado.empty:
    vendas = resultado["vendas"].values[0]
    st.metric(f"🛒 Vendas da Loja {store_selected} para a Categoria '{category_selected}'", f"R$ {vendas:,.2f}")
else:
    st.warning("Não há dados para essa combinação.")

# Mostrar tabelas completas como opção
with st.expander("📁 Ver tabelas completas"):
    st.subheader("Vendas por Categoria (todas as lojas)")
    st.dataframe(df_cat)

    st.subheader("Vendas por Loja (todas as categorias)")
    st.dataframe(df_loja)

    st.subheader("Vendas por Loja e Categoria")
    st.dataframe(df_loja_cat)

