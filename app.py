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
store_ids = ["Todas as lojas"] + sorted(df_loja_cat["ID_loja"].unique().tolist())
categories = ["Todas as categorias"] + sorted(df_loja_cat["categoria"].unique().tolist())

store_selected = st.selectbox("Selecione o ID da loja:", store_ids)
category_selected = st.selectbox("Selecione a categoria do produto:", categories)

# Consulta condicional
if store_selected != "Todas as lojas" and category_selected != "Todas as categorias":
    resultado = df_loja_cat[
        (df_loja_cat["ID_loja"] == store_selected) &
        (df_loja_cat["categoria"] == category_selected)
    ]
    if not resultado.empty:
        vendas = max(resultado["vendas"].values[0], 0)
        st.metric(f"🛒 Vendas da Loja {store_selected} para a Categoria '{category_selected}'", f"R$ {vendas:,.2f}")
    else:
        st.warning("Não há dados para essa combinação.")

elif store_selected != "Todas as lojas" and category_selected == "Todas as categorias":
    resultado = df_loja_cat[df_loja_cat["ID_loja"] == store_selected]
    total = resultado["vendas"].apply(lambda x: max(x, 0)).sum()
    st.metric(f"🏬 Vendas totais da Loja {store_selected}", f"R$ {total:,.2f}")
    st.dataframe(resultado.assign(vendas=resultado["vendas"].apply(lambda x: max(x, 0))))

elif store_selected == "Todas as lojas" and category_selected != "Todas as categorias":
    resultado = df_loja_cat[df_loja_cat["categoria"] == category_selected]
    total = resultado["vendas"].apply(lambda x: max(x, 0)).sum()
    st.metric(f"📦 Vendas totais da Categoria '{category_selected}'", f"R$ {total:,.2f}")
    st.dataframe(resultado.assign(vendas=resultado["vendas"].apply(lambda x: max(x, 0))))

else:
    total = df_loja_cat["vendas"].apply(lambda x: max(x, 0)).sum()
    st.metric("📈 Vendas totais (todas as lojas e categorias)", f"R$ {total:,.2f}")
    st.dataframe(df_loja_cat.assign(vendas=df_loja_cat["vendas"].apply(lambda x: max(x, 0))))

# Mostrar tabelas completas como opção
with st.expander("📁 Ver tabelas completas"):
    st.subheader("Vendas por Categoria (todas as lojas)")
    st.dataframe(df_cat)

    st.subheader("Vendas por Loja (todas as categorias)")
    st.dataframe(df_loja)

    st.subheader("Vendas por Loja e Categoria")
    st.dataframe(df_loja_cat)
