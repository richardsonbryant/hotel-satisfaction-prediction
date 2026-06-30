import streamlit as st
import pandas as pd

#  Page Config 
st.set_page_config(
    page_title="Favehotel Nagoya Batam — Sistem Prediksi",
    page_icon="🏨",
    layout="wide",
    initial_sidebar_state="expanded"
)

#  Custom CSS 
st.markdown("""
<style>
    .main-header {
        font-size: 2rem;
        font-weight: 600;
        color: #1a1a2e;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1rem;
        color: #6c757d;
        margin-bottom: 2rem;
    }
    div[data-testid="metric-container"] {
        background: #f8f9fa;
        border-radius: 12px;
        padding: 1rem;
        border: 1px solid #e9ecef;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    return pd.read_csv('data/hotel_reviews_dataset.csv')


df = load_data()

#  Header 
st.markdown(
    '<p class="main-header">🏨 Favehotel Nagoya Batam — Sistem Prediksi Kepuasan Tamu</p>',
    unsafe_allow_html=True
)
st.markdown(
    '<p class="sub-header">Prediksi nilai overall_score berdasarkan rating per kategori dari berbagai platform OTA.</p>',
    unsafe_allow_html=True
)

st.divider()

#  Metric Cards 
col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric("Total Ulasan", f"{len(df):,}")
with col2:
    st.metric("Rata-rata Overall Score", f"{df['overall_score'].mean():.2f}")
with col3:
    st.metric("Platform OTA", f"{df['platform'].nunique()}")
with col4:
    min_score = df['overall_score'].min()
    max_score = df['overall_score'].max()
    st.metric("Rentang Score", f"{min_score:.1f}–{max_score:.1f}")
with col5:
    st.metric("Model Final", "XGBoost", "Weighted")

st.divider()

#  Hero Section 
st.markdown("## Selamat Datang")
st.markdown(
    "Sistem ini membantu manajemen Favehotel Nagoya Batam dalam memprediksi nilai "
    "overall_score berdasarkan rating dari lima platform OTA. Sistem juga menyediakan "
    "dashboard analitik dan penjelasan prediksi menggunakan SHAP serta feature importance."
)

st.divider()

#  Fitur Utama 
st.markdown("### Fitur Utama")

col1, col2, col3 = st.columns(3)

with col1:
    st.info(
        "###  Prediksi\n\n"
        "Masukkan nilai rating per kategori untuk memperoleh prediksi nilai overall_score."
    )
with col2:
    st.success(
        "### 📊 Dashboard\n\n"
        "Pantau distribusi rating dan overall_score dari berbagai platform OTA secara interaktif."
    )
with col3:
    st.warning(
        "### ℹ️ Tentang Sistem\n\n"
        "Lihat informasi dataset, metodologi, performa model, dan keterbatasan sistem."
    )

st.divider()

#  CTA 
st.markdown("### Mulai Gunakan Sistem")

col_cta1, col_cta2 = st.columns(2)

with col_cta1:
    st.page_link(
        "pages/1_Prediksi.py",
        label="Mulai Prediksi",
        use_container_width=True
    )

with col_cta2:
    st.page_link(
        "pages/2_Dashboard.py",
        label="Lihat Dashboard",
        use_container_width=True
    )

st.divider()
st.markdown(
    "<center><small style='color:#6c757d'>"
    "Favehotel Nagoya Batam — Sistem Prediksi Kepuasan Tamu | XGBoost Weighted"
    "</small></center>",
    unsafe_allow_html=True
)
