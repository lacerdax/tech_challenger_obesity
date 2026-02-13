import pandas as pd
import streamlit as st
import plotly.express as px

st.set_page_config(page_title="Dashboard Analítico - Obesidade", layout="wide")

# ==============================
# LOAD DATA
# ==============================
@st.cache_data
def load_data():
    df = pd.read_csv("data/Obesity.csv")
    df.columns = [c.strip() for c in df.columns]
    return df

df = load_data()

st.title("📊 Painel Analítico - Estudo sobre Obesidade")
st.markdown("Análise exploratória para apoio estratégico à equipe médica.")

# ==============================
# FILTROS LATERAIS
# ==============================
st.sidebar.header("Filtros")

gender_filter = st.sidebar.multiselect(
    "Gênero",
    options=df["Gender"].unique(),
    default=df["Gender"].unique()
)

df = df[df["Gender"].isin(gender_filter)]

# ==============================
# MÉTRICAS GERAIS
# ==============================
col1, col2, col3 = st.columns(3)

col1.metric("Total de Pacientes", len(df))
col2.metric("Média de Idade", round(df["Age"].mean(), 1))
col3.metric("Média de Peso (kg)", round(df["Weight"].mean(), 1))

st.divider()

# ==============================
# DISTRIBUIÇÃO DOS NÍVEIS
# ==============================
st.subheader("Distribuição dos Níveis de Obesidade")

fig_dist = px.histogram(
    df,
    x="Obesity",
    color="Obesity",
    title="Distribuição dos Níveis de Obesidade"
)

st.plotly_chart(fig_dist, use_container_width=True)

# ==============================
# GÊNERO X OBESIDADE
# ==============================
st.subheader("Nível de Obesidade por Gênero")

fig_gender = px.histogram(
    df,
    x="Obesity",
    color="Gender",
    barmode="group",
    title="Comparação por Gênero"
)

st.plotly_chart(fig_gender, use_container_width=True)

# ==============================
# HÁBITOS E COMPORTAMENTOS
# ==============================
st.subheader("Atividade Física x Nível de Obesidade")

fig_faf = px.box(
    df,
    x="Obesity",
    y="FAF",
    color="Obesity",
    title="Distribuição de Atividade Física por Nível"
)

st.plotly_chart(fig_faf, use_container_width=True)

st.subheader("Tempo de Tela (TUE) x Nível de Obesidade")

fig_tue = px.box(
    df,
    x="Obesity",
    y="TUE",
    color="Obesity",
    title="Tempo de Uso de Dispositivos por Nível"
)

st.plotly_chart(fig_tue, use_container_width=True)

# ==============================
# CONSUMO DE ALIMENTOS CALÓRICOS
# ==============================
st.subheader("Consumo de Alimentos Altamente Calóricos (FAVC)")

favc_counts = (
    df.groupby(["Obesity", "FAVC"])
    .size()
    .reset_index(name="count")
)

fig_favc = px.bar(
    favc_counts,
    x="Obesity",
    y="count",
    color="FAVC",
    barmode="group",
    title="FAVC por Nível de Obesidade"
)

st.plotly_chart(fig_favc, use_container_width=True)

# ==============================
# TRANSPORTE
# ==============================
st.subheader("Meio de Transporte x Nível de Obesidade")

transport_counts = (
    df.groupby(["Obesity", "MTRANS"])
    .size()
    .reset_index(name="count")
)

fig_transport = px.bar(
    transport_counts,
    x="Obesity",
    y="count",
    color="MTRANS",
    barmode="group",
    title="Meio de Transporte por Nível de Obesidade"
)

st.plotly_chart(fig_transport, use_container_width=True)

# ==============================
# INSIGHTS AUTOMÁTICOS
# ==============================
st.divider()
st.subheader("🔎 Principais Insights Observados")

mean_faf = df.groupby("Obesity")["FAF"].mean().sort_values()
mean_tue = df.groupby("Obesity")["TUE"].mean().sort_values(ascending=False)

st.markdown("### 📌 Padrões Identificados:")

st.write(
    f"- O nível com menor média de atividade física é **{mean_faf.index[0]}**."
)
st.write(
    f"- O nível com maior tempo médio de uso de dispositivos é **{mean_tue.index[0]}**."
)
st.write(
    "- Observa-se tendência de maior prevalência de obesidade em indivíduos com menor prática de atividade física."
)
st.write(
    "- O consumo frequente de alimentos altamente calóricos apresenta associação com níveis mais elevados de obesidade."
)

st.divider()
st.caption("Painel desenvolvido para análise estratégica e apoio à tomada de decisão médica.")
