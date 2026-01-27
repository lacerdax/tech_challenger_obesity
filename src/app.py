import os
import json
import joblib
import pandas as pd
import streamlit as st

# ===================== CONFIGURAÇÃO DO TEMA E LAYOUT =====================
st.set_page_config(
    page_title="Sistema de Predição de Obesidade - Uso Clínico",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Aplicar estilo hospitalar
st.markdown("""
    <style>
    /* Tema claro profissional */
    .main {
        background-color: #f8f9fa; !important
        padding: 20px;
    }
    
    /* Cabeçalhos médicos */
    h1, h2, h3 {
        color: #2c3e50;
        font-family: 'Segoe UI', sans-serif;
        font-weight: 600;
        border-bottom: 2px solid #3498db;
        padding-bottom: 10px;
    }
    
    /* Cards de entrada */
    .stNumberInput, .stSelectbox, .stSlider {
        background-color: white;
        border-radius: 8px;
        padding: 12px;
        border: 1px solid #e0e0e0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    
    /* Botão principal */
    .stButton > button {
        background-color: #3498db;
        color: white;
        font-weight: bold;
        padding: 12px 24px;
        border-radius: 6px;
        border: none;
        font-size: 16px;
        width: 100%;
        transition: background-color 0.3s;
    }
    
    .stButton > button:hover {
        background-color: #2980b9;
    }
    
    /* Container para resultados */
    .stSuccess {
        background-color: #d4edda;
        border-left: 4px solid #28a745;
        padding: 15px;
        border-radius: 4px;
    }
    
    /* Divisores */
    .stDivider {
        border-color: #3498db;
        margin: 25px 0;
    }
    
    /* Tooltip e informações */
    .stInfo {
        background-color: #e8f4fc;
        border-left: 4px solid #3498db;
    }
    
    /* Barra lateral (se for usar no futuro) */
    .sidebar .sidebar-content {
        background-color: #2c3e50;
    }
    
    /* Ajustes para mobile */
    @media (max-width: 768px) {
        .main {
            padding: 10px;
        }
    }
    </style>
""", unsafe_allow_html=True)

# ===================== CARREGAMENTO DO MODELO =====================
@st.cache_resource
def load_model():
    model_path = os.path.join("models", "obesity_model.joblib")
    return joblib.load(model_path)

def load_metrics():
    metrics_path = os.path.join("models", "metrics.json")
    if os.path.exists(metrics_path):
        with open(metrics_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None

model = load_model()
metrics = load_metrics()

# ===================== CABEÇALHO PROFISSIONAL =====================
col_logo, col_title = st.columns([1, 1])

with col_title:
    st.title("🏥 Sistema de Avaliação de Risco de Obesidade")
    st.caption("Ferramenta Clínica de Apoio à Decisão | Versão 1.0")

st.divider()

# ===================== CARDS DE INFORMAÇÕES DO MODELO =====================
# if metrics:
#     col_acc, col_model, col_version = st.columns(3)
#     with col_acc:
#         st.metric(
#             label="Acurácia do Modelo",
#             value=f"{metrics.get('accuracy', 0)*100:.1f}%",
#             help="Desempenho geral do modelo de predição"
#         )
#     with col_model:
#         st.metric(
#             label="Modelo Utilizado",
#             value=metrics.get('final_model', 'Desconhecido'),
#             help="Algoritmo de machine learning implementado"
#         )
#     with col_version:
#         st.metric(
#             label="Status",
#             value="Validado",
#             delta="Clínico",
#             help="Modelo validado para uso clínico"
#         )

# ===================== ÁREA DE ENTRADA DE DADOS =====================
st.header("📋 Dados do Paciente")
st.info("Preencha as informações abaixo para avaliação do risco de obesidade. Esta ferramenta é um apoio à decisão clínica e não substitui avaliação profissional completa.")

# ✅ Categorias reais do seu dataset
GENDER_OPTS = ["Mulher", "Homem"]
YESNO = ["Sim", "Não"]  # Traduzido para português
CAEC_OPTS = ["Não", "Ocasionalmente", "Frequentemente", "Sempre"]
CALC_OPTS = ["Não", "Ocasionalmente", "Frequentemente", "Sempre"]
MTRANS_OPTS = ["Transporte Público", "Caminhada", "Automóvel", "Motocicleta", "Bicicleta"]

# Organização em abas para melhor usabilidade
tab1, tab2 = st.tabs(["📊 Dados Demográficos e Antropométricos", "🎯 Hábitos e Estilo de Vida"])

with tab1:
    col_demo, col_antropo = st.columns(2)
    
    with col_demo:
        st.subheader("Dados Demográficos")
        gender = st.selectbox("Gênero Biológico", GENDER_OPTS)
        age = st.number_input("Idade (anos)", min_value=1, max_value=120, value=45, step=1,
                             help="Idade completa em anos")
        family_history = st.selectbox("Histórico Familiar de Obesidade", YESNO,
                                     help="Parentes de primeiro grau com diagnóstico de obesidade")
    
    with col_antropo:
        st.subheader("Medidas Antropométricas")
        height = st.number_input("Altura (metros)", min_value=0.80, max_value=2.50, value=1.70, step=0.01,
                                format="%.2f", help="Altura em metros")
        weight = st.number_input("Peso (kg)", min_value=10.0, max_value=300.0, value=70.0, step=0.1,
                                format="%.1f", help="Peso atual em quilogramas")
        
        # Cálculo automático do IMC
        if height > 0 and weight > 0:
            imc = weight / (height ** 2)
            st.metric("Índice de Massa Corporal (IMC)", f"{imc:.1f} kg/m²")

with tab2:
    col_habitos, col_atividade = st.columns(2)
    
    with col_habitos:
        st.subheader("Hábitos Alimentares")
        favc = st.selectbox("Consumo frequente de alimentos hipercalóricos", YESNO)
        fcvc = st.slider("Consumo de vegetais (porções/dia)", 1.0, 3.0, 2.0, 0.1,
                        help="1 = Baixo, 2 = Moderado, 3 = Alto")
        ncp = st.slider("Número de refeições principais", 1.0, 4.0, 3.0, 0.1)
        caec = st.selectbox("Come entre as refeições?", CAEC_OPTS)
        ch2o = st.slider("Consumo de água (litros/dia)", 1.0, 3.0, 2.0, 0.1)
        calc = st.selectbox("Consumo de bebidas alcoólicas", CALC_OPTS)
    
    with col_atividade:
        st.subheader("Atividade e Monitoramento")
        smoke = st.selectbox("Tabagismo", YESNO)
        scc = st.selectbox("Monitoramento de ingestão calórica", YESNO)
        faf = st.slider("Atividade física (horas/semana)", 0.0, 3.0, 1.0, 0.1)
        tue = st.slider("Tempo de uso de dispositivos eletrônicos (horas/dia)", 0.0, 2.0, 1.0, 0.1)
        mtrans = st.selectbox("Meio de transporte habitual", MTRANS_OPTS)

# ===================== BOTÃO DE PREDIÇÃO =====================
st.divider()

col_button, col_spacer = st.columns([1, 3])
with col_button:
    predict_btn = st.button("🔍 **EXECUTAR AVALIAÇÃO**", use_container_width=True)

# ===================== PROCESSAMENTO E RESULTADOS =====================
if predict_btn:
    # Converter Sim/Não para yes/no (se o modelo foi treinado em inglês)
    family_history_en = "yes" if family_history == "Sim" else "no"
    favc_en = "yes" if favc == "Sim" else "no"
    smoke_en = "yes" if smoke == "Sim" else "no"
    scc_en = "yes" if scc == "Sim" else "no"
    
    # Mapear categorias para inglês se necessário
    caec_map = {"Não": "no", "Ocasionalmente": "Sometimes", "Frequentemente": "Frequently", "Sempre": "Always"}
    calc_map = {"Não": "no", "Ocasionalmente": "Sometimes", "Frequentemente": "Frequently", "Sempre": "Always"}
    mtrans_map = {
        "Transporte Público": "Public_Transportation",
        "Caminhada": "Walking",
        "Automóvel": "Automobile",
        "Motocicleta": "Motorbike",
        "Bicicleta": "Bike"
    }
    
    row = pd.DataFrame([{
        "Gender": gender,
        "Age": float(age),
        "Height": float(height),
        "Weight": float(weight),
        "family_history": family_history_en,
        "FAVC": favc_en,
        "FCVC": float(fcvc),
        "NCP": float(ncp),
        "CAEC": caec_map.get(caec, caec),
        "SMOKE": smoke_en,
        "CH2O": float(ch2o),
        "SCC": scc_en,
        "FAF": float(faf),
        "TUE": float(tue),
        "CALC": calc_map.get(calc, calc),
        "MTRANS": mtrans_map.get(mtrans, mtrans),
    }])
    
    # Realizar predição
    with st.spinner("Processando avaliação..."):
        pred = model.predict(row)[0]
    
    # ===================== ÁREA DE RESULTADOS =====================
    st.header("📋 Resultado da Avaliação")
    
    # Container para resultado principal
    result_container = st.container()
    with result_container:
        col_result, col_imc = st.columns([2, 1])
        
        with col_result:
            st.success(f"**Classificação Prevista:** {pred}")
            
            # Interpretação baseada na classificação
            if "Obesity" in pred or "obesity" in pred.lower():
                st.warning("""
                **Recomendações:**
                - Encaminhamento para nutricionista
                - Avaliação endócrina
                - Programa de atividade física supervisionada
                - Acompanhamento multidisciplinar
                """)
            elif "Overweight" in pred or "overweight" in pred.lower():
                st.info("""
                **Orientações:**
                - Revisão dietética
                - Aumento progressivo de atividade física
                - Monitoramento trimestral
                - Educação em saúde
                """)
            else:
                st.info("""
                **Manutenção:**
                - Manter hábitos saudáveis
                - Check-up anual
                - Prevenção contínua
                """)
        
        with col_imc:
            if 'imc' in locals():
                st.metric("IMC Calculado", f"{imc:.1f}")
                if imc >= 30:
                    st.error("Obesidade")
                elif imc >= 25:
                    st.warning("Sobrepeso")
                else:
                    st.success("Normal")
    
    # ===================== DETALHES TÉCNICOS (expansível) =====================
    with st.expander("📊 Detalhes Técnicos da Predição"):
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(row)[0]
            classes = model.classes_
            
            # Criar gráfico de barras para probabilidades
            prob_df = pd.DataFrame({
                "Classificação": classes,
                "Probabilidade (%)": (proba * 100).round(1)
            }).sort_values("Probabilidade (%)", ascending=False)
            
            st.bar_chart(prob_df.set_index("Classificação")["Probabilidade (%)"])
            
            st.write("**Probabilidades por classe:**")
            st.dataframe(
                prob_df,
                column_config={
                    "Classificação": "Nível de Obesidade",
                    "Probabilidade (%)": st.column_config.NumberColumn(
                        format="%.1f%%"
                    )
                },
                hide_index=True,
                use_container_width=True
            )
    
    # ===================== SUGESTÕES DE AÇÃO =====================
    st.divider()
    st.subheader("📝 Plano de Ação Sugerido")
    
    action_col1, action_col2, action_col3 = st.columns(3)
    
    with action_col1:
        st.markdown("""
        **Avaliação Inicial**
        - Anamnese completa
        - Exames laboratoriais
        - Avaliação nutricional
        """)
    
    with action_col2:
        st.markdown("""
        **Intervenções**
        - Planejamento alimentar
        - Prescrição de exercícios
        - Acompanhamento psicológico
        """)
    
    with action_col3:
        st.markdown("""
        **Seguimento**
        - Consultas regulares
        - Reavaliação em 3 meses
        - Ajuste de conduta
        """)

# ===================== RODAPÉ =====================
st.divider()

footer_col1, footer_col2, footer_col3 = st.columns(3)
with footer_col1:
    st.caption("© 2024 Sistema de Apoio à Decisão Clínica")
with footer_col2:
    st.caption("Uso exclusivo para profissionais de saúde")
with footer_col3:
    st.caption("Versão 1.0 | Modelo validado")

# Notas importantes fixas
st.divider()
with st.container():
    st.warning("""
    **Avisos Importantes:**
    1. Esta ferramenta é um **apoio à decisão clínica** e não substitui o julgamento profissional
    2. Resultados devem ser interpretados **no contexto clínico completo** do paciente
    3. Modelos preditivos têm **limitações** e margem de erro
    4. Sempre considerar **comorbidades** e **fatores individuais**
    5. Manter **confidencialidade** dos dados conforme legislação vigente
    """)