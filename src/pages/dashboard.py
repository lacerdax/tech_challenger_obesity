import pandas as pd
import streamlit as st
import plotly.express as px

st.set_page_config(page_title="Dashboard Analítico - Obesidade", layout="wide")

# ==============================
# LOAD DATA
# ==============================
@st.cache_data
def load_data():
    df = pd.read_csv("data/obesity.csv")
    df.columns = [c.strip() for c in df.columns]
    return df

df = load_data()

# ==============================
# DICIONÁRIOS DE TRADUÇÃO
# ==============================
traducao_colunas = {
    "Gender": "Gênero",
    "Age": "Idade",
    "Height": "Altura (m)",
    "Weight": "Peso (kg)",
    "family_history": "Histórico Familiar de Excesso de Peso",
    "FAVC": "Consumo Frequente de Alimentos Calóricos",
    "FCVC": "Consumo de Vegetais",
    "NCP": "Número de Refeições Principais",
    "CAEC": "Lanches Entre Refeições",
    "SMOKE": "Fumante",
    "CH2O": "Consumo Diário de Água",
    "SCC": "Monitora Ingestão Calórica",
    "FAF": "Frequência de Atividade Física",
    "TUE": "Tempo de Uso de Dispositivos",
    "CALC": "Consumo de Álcool",
    "MTRANS": "Meio de Transporte",
    "Obesity": "Nível de Obesidade"
}

traducao_valores = {
    "Male": "Masculino",
    "Female": "Feminino",
    "yes": "Sim",
    "no": "Não",
    "Sometimes": "Às vezes",
    "Frequently": "Frequentemente",
    "Always": "Sempre",
    "Automobile": "Carro",
    "Motorbike": "Moto",
    "Bike": "Bicicleta",
    "Public_Transportation": "Transporte Público",
    "Walking": "A pé",
    "Insufficient_Weight": "Abaixo do Peso",
    "Normal_Weight": "Peso Normal",
    "Overweight_Level_I": "Sobrepeso I",
    "Overweight_Level_II": "Sobrepeso II",
    "Obesity_Type_I": "Obesidade I",
    "Obesity_Type_II": "Obesidade II",
    "Obesity_Type_III": "Obesidade III",
}

# Aplicar tradução
df = df.replace(traducao_valores)
df = df.rename(columns=traducao_colunas)

# ==============================
# AJUSTES NUMÉRICOS (ARREDONDAMENTOS)
# ==============================
colunas_arredondar = [
    "Consumo de Vegetais",
    "Número de Refeições Principais",
    "Consumo Diário de Água",
    "Frequência de Atividade Física",
    "Tempo de Uso de Dispositivos"
]

for col in colunas_arredondar:
    if col in df.columns:
        df[col] = df[col].round().astype(int)

# ==============================
# ORDEM DOS NÍVEIS DE OBESIDADE
# ==============================
ordem_obesidade = [
    "Abaixo do Peso",
    "Peso Normal",
    "Sobrepeso I",
    "Sobrepeso II",
    "Obesidade I",
    "Obesidade II",
    "Obesidade III"
]

df["Nível de Obesidade"] = pd.Categorical(
    df["Nível de Obesidade"],
    categories=ordem_obesidade,
    ordered=True
)

# ==============================
# TÍTULO
# ==============================
st.title("📊 Painel Analítico - Estudo sobre Obesidade")
st.markdown("Análise exploratória para apoio estratégico à equipe médica.")

# ==============================
# FILTROS
# ==============================
st.sidebar.header("Filtros")

gender_filter = st.sidebar.multiselect(
    "Gênero",
    options=df["Gênero"].unique(),
    default=df["Gênero"].unique()
)

df = df[df["Gênero"].isin(gender_filter)]

# ==============================
# MÉTRICAS
# ==============================
col1, col2, col3 = st.columns(3)

col1.metric("Total de Pacientes", len(df))
col2.metric("Média de Idade", round(df["Idade"].mean(), 1))
col3.metric("Média de Peso (kg)", round(df["Peso (kg)"].mean(), 1))

st.divider()

# ==============================
# DISTRIBUIÇÃO
# ==============================
st.subheader("Distribuição dos Níveis de Obesidade")

fig_dist = px.histogram(
    df,
    x="Nível de Obesidade",
    color="Nível de Obesidade"
)

st.plotly_chart(fig_dist, use_container_width=True)

# ==============================
# GÊNERO X OBESIDADE
# ==============================
st.subheader("Nível de Obesidade por Gênero")

fig_gender = px.histogram(
    df,
    x="Nível de Obesidade",
    color="Gênero",
    barmode="group"
)

st.plotly_chart(fig_gender, use_container_width=True)

# ==============================
# ATIVIDADE FÍSICA
# ==============================
st.subheader("Atividade Física x Nível de Obesidade")

fig_faf = px.box(
    df,
    x="Nível de Obesidade",
    y="Frequência de Atividade Física",
    color="Nível de Obesidade"
)

st.plotly_chart(fig_faf, use_container_width=True)

# ==============================
# TEMPO DE TELA
# ==============================
st.subheader("Tempo de Tela x Nível de Obesidade")

fig_tue = px.box(
    df,
    x="Nível de Obesidade",
    y="Tempo de Uso de Dispositivos",
    color="Nível de Obesidade"
)

st.plotly_chart(fig_tue, use_container_width=True)

# ==============================
# CONSUMO CALÓRICO
# ==============================
st.subheader("Consumo de Alimentos Altamente Calóricos")

favc_counts = (
    df.groupby(["Nível de Obesidade", "Consumo Frequente de Alimentos Calóricos"])
    .size()
    .reset_index(name="Quantidade")
)

fig_favc = px.bar(
    favc_counts,
    x="Nível de Obesidade",
    y="Quantidade",
    color="Consumo Frequente de Alimentos Calóricos",
    barmode="group"
)

st.plotly_chart(fig_favc, use_container_width=True)

# ==============================
# TRANSPORTE
# ==============================
st.subheader("Meio de Transporte x Nível de Obesidade")

transport_counts = (
    df.groupby(["Nível de Obesidade", "Meio de Transporte"])
    .size()
    .reset_index(name="Quantidade")
)

fig_transport = px.bar(
    transport_counts,
    x="Nível de Obesidade",
    y="Quantidade",
    color="Meio de Transporte",
    barmode="group"
)

st.plotly_chart(fig_transport, use_container_width=True)

# ==============================
# INSIGHTS AUTOMÁTICOS
# ==============================
st.divider()
st.subheader("🔎 Principais Insights Observados")

mean_faf = df.groupby("Nível de Obesidade")["Frequência de Atividade Física"].mean().sort_values()
mean_tue = df.groupby("Nível de Obesidade")["Tempo de Uso de Dispositivos"].mean().sort_values(ascending=False)

st.markdown("### 📌 Padrões Identificados:")

st.write(f"- O nível com menor média de atividade física é **{mean_faf.index[0]}**.")
st.write(f"- O nível com maior tempo médio de uso de dispositivos é **{mean_tue.index[0]}**.")
st.write("- Observa-se tendência de maior prevalência de obesidade em indivíduos com menor prática de atividade física.")
st.write("- O consumo frequente de alimentos altamente calóricos apresenta associação com níveis mais elevados de obesidade.")

st.divider()
st.caption("Painel desenvolvido para análise estratégica e apoio à tomada de decisão médica.")
