import streamlit as st
import pandas as pd
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit

st.sidebar.title("Menu")
opcao = st.sidebar.selectbox("Escolha uma opção:", ["Entenda os Dados", "Preveja as Vendas", "Avalie o modelo"])

if opcao == "Entenda os Dados":
    st.title("Entenda os Dados")
    st.markdown("""
    <h5><div style="text-align: justify;">
    Os dados utilizados referem-se às <span style='color:#E07A5F;'>vendas realizadas</span> por 54 lojas espalhadas pelo país, no período de 1º de janeiro de 2013 a 15 de agosto de 2017. As vendas foram <span style='color:#E07A5F;'>categorizadas</span> por departamento, como mercearia, beleza, casa, entre outros.<br><br>
    No tratamento dos dados, combinei as informações de vendas com dados de <span style='color:#E07A5F;'>feriados nacionais e regionais</span>, além da localização de cada loja, para identificar se a loja estaria ou não em funcionamento em determinada data.<br><br>
    Foram extraídas <span style='color:#E07A5F;'>variáveis temporais</span> como dia da semana, mês, início e fim de mês, considerando também datas típicas de <span style='color:#E07A5F;'>pagamento</span> nos locais.<br><br>
    Adicionei <span style='color:#E07A5F;'>lags</span> (valores de vendas passados) para capturar a <span style='color:#E07A5F;'>dependência temporal</span>, e calculei médias móveis e médias exponenciais para <span style='color:#E07A5F;'>identificar tendências e sazonalidades</span>.<br><br>
    Após testes com diferentes hiperparâmetros utilizando <span style='color:#E07A5F;'>GridSearch</span>, foi selecionado o modrlo <span style='color:#E07A5F;'>GradientBoostingRegressor</span> com as configurações <i>learning_rate=0.1, max_depth=3, n_estimators=100, subsample=1.0</i> para realizar as previsões.<br><br>
    Os dados exibidos na aba <i><span style='color:#E07A5F;'>Preveja as Vendas</span></i> representam as estimativas geradas pelo modelo para o dia <span style='color:#E07A5F;'>16 de agosto de 2017</span>.
    </div></h5>
    """, unsafe_allow_html=True)


elif opcao == "Preveja as Vendas":
    st.title("📊 Análise de Dados")

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


elif opcao == "Avalie o modelo":
    st.title("ℹ️ Avalie o modelo")
    
    # Carregamento dos dados
    @st.cache_data
    def load_data():
        df = pd.read_parquet("train.parquet")
        stores = pd.read_csv("stores.csv")
        holidays = pd.read_csv("holidays_events.csv")
        
        return df, stores, holidays
        
    df, stores, holidays = load_data()
        
    # Pré-processamento básico
    df["date"] = pd.to_datetime(df["date"])
    df["sales"] = df["sales"].astype("float32")
    df = df.sort_values(["store_nbr", "family", "date"])

    # Feriados nacionais
    holidays = holidays[holidays["locale"] == "National"].copy()
    holidays["date"] = pd.to_datetime(holidays["date"])
    holidays = holidays[["date"]].drop_duplicates()
    holidays["is_holiday"] = 1

    # Merge com cidade e feriados
    df = df.merge(stores[["store_nbr", "city"]], on="store_nbr", how="left")
    df = df.merge(holidays, on="date", how="left")
    df["is_holiday"] = df["is_holiday"].fillna(0).astype("int8")

    # Interface: escolha de loja e família
    store_options = sorted(df["store_nbr"].unique())
    family_options = sorted(df["family"].unique())

    store_selected = st.selectbox("Selecione a loja:", store_options, index=49)
    family_selected = st.selectbox("Selecione a categoria:", family_options, index=12)

    # Filtragem
    df = df[(df["store_nbr"] == store_selected) & (df["family"] == family_selected)].copy()

    # Features de tempo
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

    # EWM
    def ewm_features(dataframe, alphas, lags):
        dataframe = dataframe.copy()
        for alpha in alphas:
            for lag in lags:
                colname = f"ewm_a{str(alpha).replace('.', '')}_l{lag}"
                dataframe[colname] = (
                    dataframe["sales"].shift(lag).ewm(alpha=alpha).mean()
                )
        
        return dataframe

    alphas = [0.95, 0.9]
    lags = [7, 14]
    df = ewm_features(df, alphas, lags)

    # Lags e Rolling
    for lag in [1, 7, 14]:
        df[f"lag_{lag}"] = df["sales"].shift(lag)

    for window in [7, 14]:
        df[f"rolling_mean_{window}"] = df["sales"].shift(1).rolling(window).mean()
        df[f"rolling_std_{window}"] = df["sales"].shift(1).rolling(window).std()

    df = df.dropna()

    # Divisão treino/teste
    train_df = df.iloc[:-15]
    test_df = df.iloc[-15:]

    features = [col for col in df.columns if col not in ["date", "sales", "store_nbr", "family", "city"]]
    X_train = train_df[features]
    y_train = train_df["sales"]
    X_test = test_df[features]
    y_test = test_df["sales"]

    with st.spinner("Treinando modelo..."):
        best_model = GradientBoostingRegressor(learning_rate=0.1, max_depth=3, n_estimators=100, subsample=1.0)
        best_model.fit(X_train, y_train)

    # Previsões
    y_pred = best_model.predict(X_test)

    # Corrigindo valores negativos
    y_pred = np.where(y_pred < 0, 0, y_pred)

    # Métricas
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    st.subheader("📏 Métricas de desempenho")
    st.metric("MAE", f"{mae:.2f}")
    st.metric("RMSE", f"{rmse:.2f}")

    # Resultado
    test_df = test_df.copy()
    test_df["prediction"] = y_pred

    # Gráfico
    st.subheader("📊 Gráfico: Vendas Reais vs. Previsões")
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(test_df["date"], test_df["sales"], label="Vendas reais", marker="o")
    ax.plot(test_df["date"], test_df["prediction"], label="Previsão", marker="o")
    ax.set_xlabel("Data")
    ax.set_ylabel("Vendas")
    ax.set_title(f"Previsão - Loja {store_selected} | {family_selected}")
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)

    # Tabela
    with st.expander("📋 Ver dados de teste com previsão"):
        st.dataframe(test_df[["date", "sales", "prediction"]].reset_index(drop=True))
