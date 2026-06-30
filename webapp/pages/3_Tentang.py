import streamlit as st
import pandas as pd

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"

# ── Page Config ──────────────────────────────────────────────
st.set_page_config(
    page_title="Tentang Sistem — Favehotel Nagoya Batam",
    page_icon="ℹ️",
    layout="wide"
)

#  Helper 
@st.cache_data
def load_main_dataset():
    return pd.read_csv(DATA_DIR / "hotel_reviews_dataset.csv")


@st.cache_data
def load_csv_if_exists(filename):
    path = DATA_DIR / filename

    if path.exists():
        return pd.read_csv(path)

    st.warning(f"File tidak ditemukan: {path}")
    return None


def find_column(df, candidates):
    if df is None:
        return None

    for candidate in candidates:
        if candidate in df.columns:
            return candidate

    return None


#  Load Data 
df = load_main_dataset()

baseline_weighted_results = load_csv_if_exists(
    "bab4_baseline_vs_weighted_results.csv"
)

per_platform_results = load_csv_if_exists(
    "bab4_xgboost_weighted_per_platform.csv"
)

feature_importance = load_csv_if_exists(
    "bab4_xgboost_feature_importance.csv"
)


def find_column(df, candidates):
    if df is None:
        return None
    for candidate in candidates:
        if candidate in df.columns:
            return candidate
    return None


#  Load Data 
def load_csv_if_exists(filename):
    path = DATA_DIR / filename

    if path.exists():
        return pd.read_csv(path)

    st.warning(f"File tidak ditemukan: {path}")
    return None


baseline_weighted_results = load_csv_if_exists(
    "bab4_baseline_vs_weighted_results.csv"
)

xgb_platform_results = load_csv_if_exists(
    "bab4_xgboost_weighted_per_platform.csv"
)

xgb_feature_importance = load_csv_if_exists(
    "bab4_xgboost_feature_importance.csv"
)

st.markdown("## ℹ️ Tentang Sistem")
st.divider()

# Tentang Sistem 
st.markdown("### Tentang Sistem Ini")

col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("""
    Sistem ini merupakan implementasi penelitian tugas akhir yang bertujuan untuk
    memprediksi nilai *overall_score* kepuasan tamu Favehotel Nagoya Batam
    berdasarkan data rating dari berbagai platform Online Travel Agency (OTA).

    **Tujuan Sistem:**
    - Memprediksi nilai *overall_score* berdasarkan rating per kategori.
    - Menyediakan penjelasan prediksi menggunakan SHAP.
    - Menampilkan kontribusi fitur secara global melalui *feature importance*.
    - Membantu manajemen hotel memahami data rating dari beberapa platform OTA.

    **Manfaat untuk Manajemen Hotel:**
    - Memantau distribusi nilai rating dan *overall_score* dari berbagai platform OTA.
    - Memahami fitur rating yang berkontribusi dalam model prediksi.
    - Mendapatkan informasi pendukung untuk pengambilan keputusan berbasis data.
    """)

st.divider()

# Dataset 
st.markdown("### Tentang Dataset")

col_d1, col_d2 = st.columns(2)

with col_d1:
    st.markdown("**Distribusi Data Per Platform:**")
    platform_counts = df['platform'].value_counts().reset_index()
    platform_counts.columns = ['Platform', 'Jumlah Ulasan']
    platform_counts['Persentase'] = (
        platform_counts['Jumlah Ulasan'] / len(df) * 100
    ).round(1).astype(str) + '%'
    st.dataframe(platform_counts, hide_index=True, use_container_width=True)

with col_d2:
    st.markdown("**Ketersediaan Kolom Per Platform:**")
    availability = {
        'Kolom': ['overall_score', 'cleanliness', 'location', 'service', 'facility', 'value_for_money'],
        'Traveloka': ['✅', '✅', '✅', '✅', '✅', '❌'],
        'Booking.com': ['✅', '✅', '✅', '✅', '✅', '✅'],
        'Trip.com': ['✅', '✅', '✅', '✅', '❌', '❌'],
        'Agoda': ['✅', '✅', '✅', '✅', '✅', '✅'],
        'Tiket.com': ['✅', '✅', '✅', '✅', '✅', '✅'],
    }
    st.dataframe(pd.DataFrame(availability), hide_index=True, use_container_width=True)

st.divider()

# Performa Model 
st.markdown("### Performa Model Regresi")
st.caption("Target prediksi: nilai overall_score")

if baseline_weighted_results is not None:
    display_cols = [col for col in [
    'Model',
    'RMSE Baseline',
    'RMSE Weighted',
    'MAE Baseline',
    'MAE Weighted',
    'R² Baseline',
    'R² Weighted',
    'R2 Baseline',
    'R2 Weighted',
    'Perubahan R²',
    'Perubahan R2'
] if col in baseline_weighted_results.columns]

    results_display = baseline_weighted_results[display_cols].copy()
    st.dataframe(results_display, hide_index=True, use_container_width=True)

    model_col = find_column(baseline_weighted_results, ['Model'])
    r2_col = find_column(baseline_weighted_results, ['R² Weighted', 'R2 Weighted'])
    rmse_col = find_column(baseline_weighted_results, ['RMSE Weighted'])
    mae_col = find_column(baseline_weighted_results, ['MAE Weighted'])

    if model_col and r2_col:
        xgb_row = baseline_weighted_results[
            baseline_weighted_results[model_col].astype(str).str.lower().str.contains('xgboost')
        ]
        if not xgb_row.empty:
            xgb_row = xgb_row.iloc[0]
            metric_lines = [
                "**Model yang digunakan di sistem: XGBoost Weighted**",
                "",
                f"- R² Test: {xgb_row[r2_col]:.4f}"
            ]
            if rmse_col:
                metric_lines.append(f"- RMSE Test: {xgb_row[rmse_col]:.4f}")
            if mae_col:
                metric_lines.append(f"- MAE Test: {xgb_row[mae_col]:.4f}")
            st.success("\n".join(metric_lines))
else:
    st.info(
        "File hasil evaluasi model belum ditemukan. Jalankan notebook evaluasi untuk menghasilkan "
        "data/bab4_baseline_vs_weighted_results.csv."
    )

if per_platform_results is not None:
    st.markdown("#### Evaluasi XGBoost Weighted Per Platform")
    st.dataframe(per_platform_results, hide_index=True, use_container_width=True)

if feature_importance is not None:
    st.markdown("#### Feature Importance XGBoost Weighted")
    st.dataframe(feature_importance, hide_index=True, use_container_width=True)

st.divider()

# Metodologi 
st.markdown("### Metodologi")

col_m1, col_m2 = st.columns(2)

with col_m1:
    st.markdown("**Pipeline Preprocessing:**")
    steps = [
        "1. Seleksi kolom berdasarkan missing values",
        "2. Penambahan variabel indikator",
        "3. Imputasi KNN dengan k=5",
        "4. One-hot encoding pada platform",
        "5. Normalisasi MinMaxScaler pada fitur rating",
        "6. Train-test split 80:20",
        "7. Evaluasi model regresi menggunakan RMSE, MAE, dan R²"
    ]
    for step in steps:
        st.markdown(f"- {step}")

with col_m2:
    st.markdown("**Algoritma yang Dibandingkan:**")
    algorithms = pd.DataFrame({
        'Algoritma': ['Random Forest', 'XGBoost', 'LightGBM', 'Ridge Regression', 'SVR'],
        'Skema': ['Baseline dan Weighted', 'Baseline dan Weighted', 'Baseline dan Weighted', 'Baseline dan Weighted', 'Baseline dan Weighted'],
        'Keterangan': ['Pembanding tree-based', 'Model final', 'Comparator boosting', 'Model linear regularisasi', 'Comparator non-linear']
    })
    st.dataframe(algorithms, hide_index=True, use_container_width=True)

st.divider()

# Keterbatasan 
st.markdown("### Keterbatasan Sistem")

st.warning("""
**Keterbatasan yang perlu diperhatikan:**

1. **Rentang Input** — Model dilatih terutama pada data rating dengan rentang nilai tinggi.
   Input yang jauh di luar pola data training perlu diinterpretasikan dengan hati-hati.

2. **Fitur Terbatas** — Sistem hanya menggunakan data rating numerik dan platform OTA.
   Sistem belum menggunakan teks ulasan, tanggal ulasan, atau profil tamu.

3. **Interpretasi Model** — SHAP dan feature importance menjelaskan kontribusi fitur dalam model,
   bukan membuktikan hubungan sebab-akibat secara langsung.

4. **Ketergantungan File Model** — Sistem membutuhkan file model, scaler, imputer,
   SHAP explainer, dan feature importance yang telah dihasilkan pada tahap pelatihan.
""")

st.divider()
st.markdown(
    "<center><small style='color:#6c757d'>"
    "Favehotel Nagoya Batam — Sistem Prediksi Kepuasan Tamu | XGBoost Weighted"
    "</small></center>",
    unsafe_allow_html=True
)
