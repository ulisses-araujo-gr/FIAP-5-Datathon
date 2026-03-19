import streamlit as st
import pickle
import numpy as np

# =========================
# CONFIG
# =========================
st.set_page_config(
    page_title="Previsão de Risco - Passos Mágicos",
    page_icon="🎓",
    layout="centered"
)

# =========================
# LOAD MODELO
# =========================
@st.cache_resource
def load_model():
    model = pickle.load(open("modelo_risco.pkl", "rb"))
    scaler = pickle.load(open("scaler.pkl", "rb"))
    return model, scaler

model, scaler = load_model()

# =========================
# HEADER
# =========================
st.title("🎓 Previsão de Risco de Defasagem")
st.markdown("Simule o risco de defasagem com base nos indicadores do aluno.")

# =========================
# INPUTS
# =========================
st.subheader("📊 Indicadores do aluno")

col1, col2 = st.columns(2)

with col1:
    ida = st.slider("IDA (Desempenho Acadêmico)", 0.0, 10.0, 5.0)
    ieg = st.slider("IEG (Engajamento)", 0.0, 10.0, 5.0)
    ips = st.slider("IPS (Psicossocial)", 0.0, 10.0, 5.0)

with col2:
    ipp = st.slider("IPP (Psicopedagógico)", 0.0, 10.0, 5.0)
    iaa = st.slider("IAA (Autoavaliação)", 0.0, 10.0, 5.0)

# =========================
# PREDIÇÃO
# =========================
if st.button("🔍 Prever risco"):

    entrada = np.array([[ida, ieg, ips, ipp, iaa]])

    try:
        entrada_scaled = scaler.transform(entrada)
    except:
        st.warning("⚠️ Erro ao aplicar scaler. Usando dados sem normalização.")
        entrada_scaled = entrada

    proba = model.predict_proba(entrada_scaled)[0][1]

    # =========================
    # RESULTADO
    # =========================
    st.subheader("📈 Resultado")

    st.metric(
        label="Probabilidade de risco",
        value=f"{proba:.2%}"
    )

    # =========================
    # CLASSIFICAÇÃO MAIS INTELIGENTE
    # =========================
    if proba < 0.3:
        st.success("✅ Baixo risco")
    elif proba < 0.6:
        st.warning("⚠️ Risco moderado")
    else:
        st.error("🚨 Alto risco")

    # =========================
    # INSIGHTS AUTOMÁTICOS
    # =========================
    st.subheader("🧠 Insights")

    insights = []

    if ieg < 5:
        insights.append("Baixo engajamento pode impactar o desempenho.")
    if ida < 5:
        insights.append("Desempenho acadêmico abaixo do ideal.")
    if ips < 5:
        insights.append("Fatores psicossociais podem estar afetando o aluno.")
    if ipp < 5:
        insights.append("Necessidade de reforço psicopedagógico.")
    if iaa > 8 and ida < 5:
        insights.append("Possível desalinhamento entre percepção e desempenho.")

    if insights:
        for i in insights:
            st.write(f"- {i}")
    else:
        st.write("Indicadores equilibrados.")

    # =========================
    # INTERPRETAÇÃO GERENCIAL
    # =========================
    st.subheader("📌 Interpretação")

    if proba >= 0.6:
        st.write("Aluno deve ser priorizado para intervenção imediata.")
    elif proba >= 0.3:
        st.write("Aluno deve ser acompanhado de perto.")
    else:
        st.write("Aluno apresenta situação estável.")

# =========================
# RODAPÉ
# =========================
st.markdown("---")
st.caption("Modelo preditivo baseado nos dados educacionais da Passos Mágicos.")