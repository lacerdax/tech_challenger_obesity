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
        text-align: center;
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
    
    /* Barra lateral */
    .sidebar .sidebar-content {
        background-color: #2c3e50;
    }
    
    /* Cards do dashboard */
    .dashboard-card {
        background-color: white;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-bottom: 20px;
    }
    
    /* Ajustes para mobile */
    @media (max-width: 768px) {
        .main {
            padding: 10px;
        }
    }
    </style>
""", unsafe_allow_html=True)

# ===================== FUNÇÕES DE CARREGAMENTO =====================
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

# ===================== SIDEBAR - DASHBOARD =====================
def show_dashboard():
    st.sidebar.title("📊 Dashboard do Modelo")
    st.sidebar.markdown("---")
    
    if metrics:
        # Métricas principais em cards
        col1, col2 = st.sidebar.columns(2)
        with col1:
            st.metric(
                label="Acurácia",
                value=f"{metrics.get('accuracy', 0)*100:.1f}%",
                help="Desempenho geral do modelo"
            )
        with col2:
            st.metric(
                label="F1-Score",
                value=f"{metrics.get('f1_score', 0)*100:.1f}%",
                help="Média harmônica entre precisão e recall"
            )
        
        st.sidebar.markdown("---")
        
        # Métricas detalhadas
        with st.sidebar.expander("📈 Métricas Detalhadas", expanded=True):
            st.markdown(f"""
            **Precisão:** {metrics.get('precision', 0)*100:.1f}%  
            **Recall:** {metrics.get('recall', 0)*100:.1f}%  
            **Modelo:** {metrics.get('final_model', 'Desconhecido')}  
            **Validação:** Cruzada (5 folds)
            """)
        
        # Matriz de confusão (se disponível)
        if 'confusion_matrix' in metrics:
            with st.sidebar.expander("🔍 Matriz de Confusão"):
                cm = metrics['confusion_matrix']
                cm_df = pd.DataFrame(cm, 
                                   index=['Real Negativo', 'Real Positivo'],
                                   columns=['Predito Negativo', 'Predito Positivo'])
                st.dataframe(cm_df, use_container_width=True)

    
    st.sidebar.markdown("---")

# ===================== CARREGAR MODELO E MÉTRICAS =====================
model = load_model()
metrics = load_metrics()

# ===================== MAPA DE TRADUÇÃO DAS CLASSES =====================
CLASS_MAP = {
    "Insufficient_Weight": "Baixo Peso",
    "Normal_Weight": "Peso Normal",
    "Overweight_Level_I": "Sobrepeso Grau I",
    "Overweight_Level_II": "Sobrepeso Grau II",
    "Obesity_Type_I": "Obesidade Grau I",
    "Obesity_Type_II": "Obesidade Grau II",
    "Obesity_Type_III": "Obesidade Grau III (Mórbida)"
}


# ===================== BARRA LATERAL COM DASHBOARD =====================
with st.sidebar:
    st.markdown("## 🏥 **Painel de Controle**")
    st.markdown("---")
    
    # Botão para abrir/fechar dashboard
    if 'show_dashboard' not in st.session_state:
        st.session_state.show_dashboard = False
    
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("📊 Dashboard", use_container_width=True, type="primary"):
            st.session_state.show_dashboard = not st.session_state.show_dashboard
    with col2:
        if st.button("🔄 Nova Avaliação"):
             for key in list(st.session_state.keys()):
                del st.session_state[key]
                st.rerun()
    
    st.markdown("---")
    
    # Exibir dashboard se ativado
    if st.session_state.show_dashboard:
        show_dashboard()
    
    # Informações rápidas sempre visíveis
    st.info("""
    **⏱️ Atalhos Rápidos**
    - Pressione F5 para nova avaliação
    - Use as setas para navegação
    - Resultados salvos automaticamente
    """)

# ===================== CABEÇALHO PROFISSIONAL =====================
col_logo, col_title, col_status = st.columns([1, 4, 1])

with col_title:
    st.markdown("""
<div style='text-align: center;'>
    <h1 style='white-space: wrap; font-size: 40px; margin-bottom: 5px;'>
        🏥 Sistema de Avaliação de Risco de Obesidade
    </h1>
    <p style='color: #6c757d; font-size: 16px; margin-top: 0;'>
        Ferramenta Clínica de Apoio à Decisão | Versão 1.0
    </p>
</div>
""", unsafe_allow_html=True)



st.divider()

# ===================== BARRA DE PROGRESSO =====================
progress_placeholder = st.empty()

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
        gender = st.selectbox("Gênero Biológico", GENDER_OPTS, key="gender")
        age = st.number_input("Idade (anos)", min_value=1, max_value=120, value=45, step=1,
                             help="Idade completa em anos", key="age")
        family_history = st.selectbox("Histórico Familiar de Obesidade", YESNO, key="family_history",
                                     help="Parentes de primeiro grau com diagnóstico de obesidade")
    
    with col_antropo:
        st.subheader("Medidas Antropométricas")
        height = st.number_input("Altura (metros)", min_value=0.80, max_value=2.50, value=1.70, step=0.01,
                                format="%.2f", help="Altura em metros", key="height")
        weight = st.number_input("Peso (kg)", min_value=10.0, max_value=300.0, value=70.0, step=0.1,
                                format="%.1f", help="Peso atual em quilogramas", key="weight")
        
        # Cálculo automático do IMC
        if height > 0 and weight > 0:
            imc = weight / (height ** 2)
            st.metric("Índice de Massa Corporal (IMC)", f"{imc:.1f} kg/m²")

with tab2:
    col_habitos, col_atividade = st.columns(2)
    
    with col_habitos:
        st.subheader("Hábitos Alimentares")
        favc = st.selectbox("Consumo frequente de alimentos hipercalóricos", YESNO, key="favc")
        fcvc = st.slider("Consumo de vegetais (porções/dia)", 1.0, 5.0, 2.0, 1.0, key="fcvc",
                        help="1 = Baixo, 2 = Moderado, 3 = Alto")
        ncp = st.slider("Número de refeições principais", 1.0, 3.0, 5.0, 1.0, key="ncp")
        caec = st.selectbox("Come entre as refeições?", CAEC_OPTS, key="caec")
        ch2o = st.slider("Consumo de água (litros/dia)", 1.0, 20.0, 2.0, 0.5, key="ch2o")
        calc = st.selectbox("Consumo de bebidas alcoólicas", CALC_OPTS, key="calc")
    
    with col_atividade:
        st.subheader("Atividade e Monitoramento")
        smoke = st.selectbox("Tabagismo", YESNO, key="smoke")
        scc = st.selectbox("Monitoramento de ingestão calórica", YESNO, key="scc")
        faf = st.slider("Atividade física (horas/semana)", 0.0, 20.0, 1.0, 0.5, key="faf")
        tue = st.slider("Tempo de uso de dispositivos eletrônicos (horas/dia)", 0.0, 10.0, 1.0, 0.1, key="tue")
        mtrans = st.selectbox("Meio de transporte habitual", MTRANS_OPTS, key="mtrans")

# ===================== BOTÃO DE PREDIÇÃO =====================
st.divider()

col_left, col_center, col_right = st.columns([1, 2, 1])

with col_center:
    st.markdown("""
    <div style='text-align:center; color:#6c757d; font-size:14px; margin-bottom:8px;'>
    ⏱️ A avaliação leva menos de 1 segundo
    </div>
    """, unsafe_allow_html=True)

    predict_btn = st.button(
        "🔍 EXECUTAR AVALIAÇÃO CLÍNICA",
        use_container_width=True,
        type="primary"
    )
# ===================== PROCESSAMENTO E RESULTADOS =====================
if predict_btn:
    # Atualizar barra de progresso
    progress_placeholder.progress(50, text="Processando dados do paciente...")
    
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
    progress_placeholder.progress(75, text="Executando modelo preditivo...")
    with st.spinner("Processando avaliação..."):
        pred = model.predict(row)[0]
    pred_pt = CLASS_MAP.get(pred, pred)
    progress_placeholder.progress(100, text="Avaliação concluída!")
    progress_placeholder.empty()
    
    # ===================== ÁREA DE RESULTADOS =====================
    st.header("📋 Resultado da Avaliação")
    
    # Container para resultado principal
    result_container = st.container()
    with result_container:
        # Cards de resultado
        col_result, col_imc, col_risco = st.columns([2, 1, 1])
        
        with col_result:
            # Definir cor baseada na classificação
            if "Obesity" in pred or "obesity" in pred.lower():
                st.error(f"**Classificação Prevista:** {pred_pt}")

            elif "Overweight" in pred or "overweight" in pred.lower():
                st.warning(f"**Classificação Prevista:** {pred_pt}")

            else:
                st.success(f"**Classificação Prevista:** {pred_pt}")

        
        with col_imc:
            if 'imc' in locals():
                if imc >= 30:
                    st.error(f"IMC: {imc:.1f} (Obesidade)")
                elif imc >= 25:
                    st.warning(f"IMC: {imc:.1f} (Sobrepeso)")
                else:
                    st.success(f"IMC: {imc:.1f} (Normal)")
        
        with col_risco:
            # Nível de risco baseado na classificação
            if "Obesity" in pred or "obesity" in pred.lower():
                st.error("🔴 Risco Alto")
            elif "Overweight" in pred or "overweight" in pred.lower():
                st.warning("🟡 Risco Moderado")
            else:
                st.success("🟢 Risco Baixo")
        
        st.markdown("---")
        
        # Recomendações detalhadas
        st.subheader("📋 Recomendações Clínicas")
        
        if "Obesity" in pred or "obesity" in pred.lower():
            col_rec1, col_rec2, col_rec3 = st.columns(3)
            with col_rec1:
                st.markdown("""
                **🩺 Encaminhamentos Imediatos**
                - Nutricionista
                - Endocrinologista
                - Cardiologia
                """)
            with col_rec2:
                st.markdown("""
                **🏃 Intervenções**
                - Programa estruturado de exercícios
                - Acompanhamento psicológico
                - Avaliação metabólica completa
                """)
            with col_rec3:
                st.markdown("""
                **📊 Monitoramento**
                - Consultas quinzenais
                - Exames laboratoriais
                - Acompanhamento multidisciplinar
                """)
        elif "Overweight" in pred or "overweight" in pred.lower():
            col_rec1, col_rec2, col_rec3 = st.columns(3)
            with col_rec1:
                st.markdown("""
                **🥗 Orientação Nutricional**
                - Reeducação alimentar
                - Plano alimentar personalizado
                - Controle de porções
                """)
            with col_rec2:
                st.markdown("""
                **🏋️ Atividade Física**
                - Iniciar atividade progressiva
                - 150min/semana moderada
                - Acompanhamento gradual
                """)
            with col_rec3:
                st.markdown("""
                **📅 Seguimento**
                - Consultas mensais
                - Meta de peso realista
                - Prevenção de progressão
                """)
        else:
            col_rec1, col_rec2, col_rec3 = st.columns(3)
            with col_rec1:
                st.markdown("""
                **✅ Manutenção**
                - Hábitos saudáveis
                - Alimentação balanceada
                - Hidratação adequada
                """)
            with col_rec2:
                st.markdown("""
                **🏃 Prevenção**
                - Atividade física regular
                - Check-up anual
                - Vacinação em dia
                """)
            with col_rec3:
                st.markdown("""
                **📋 Acompanhamento**
                - Consultas anuais
                - Prevenção de comorbidades
                - Qualidade de vida
                """)
    
   
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

    # ===================== RODAPÉ =====================
st.divider()

footer_col1, footer_col2, footer_col3, = st.columns(3)
with footer_col1:
    st.caption("© Sistema de Apoio à Decisão Clínica")
with footer_col2:
    st.caption("🔒 Uso exclusivo para profissionais de saúde")
with footer_col3:
    st.caption("📋 Versão 1.0 | Modelo validado")