import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go

# Page Config
st.set_page_config(
    page_title="Prediksi — Favehotel Nagoya Batam",
    page_icon="🔮",
    layout="wide"
)

# Konstanta 
PLATFORM_AVAILABILITY = {
    "Agoda": {"facility": True, "value_for_money": True},
    "Booking.com": {"facility": True, "value_for_money": True},
    "Tiket.com": {"facility": True, "value_for_money": True},
    "Traveloka": {"facility": True, "value_for_money": False},
    "Trip.com": {"facility": False, "value_for_money": False},
}

NUMERIC_COLS = [
    'cleanliness',
    'location',
    'service',
    'facility',
    'value_for_money'
]

# Load Models
@st.cache_resource
def load_models():
    return {
        'best_reg': joblib.load('models/best_regression_model.pkl'),
        'scaler': joblib.load('models/scaler.pkl'),
        'feature_cols': joblib.load('models/feature_cols.pkl'),
        'imputer': joblib.load('models/knn_imputer.pkl'),
        'shap_explainer': joblib.load('models/shap_explainer_xgb.pkl'),
        'feature_importance': joblib.load('models/xgb_feature_importance.pkl'),
    }


def get_shap_values(explainer, data):
    try:
        shap_values = explainer.shap_values(data)
        if isinstance(shap_values, list):
            shap_values = shap_values[0]
        return np.asarray(shap_values)[0]
    except Exception:
        explanation = explainer(data)
        return np.asarray(explanation.values)[0]


def format_feature_name(feature_name, platform_key=None, platform=None):
    rename_map = {
        'cleanliness': 'Cleanliness',
        'location': 'Location',
        'service': 'Service',
        'facility': 'Facility',
        'value_for_money': 'Value for Money',
        'is_facility_available': 'Facility Available',
        'is_value_for_money_available': 'Value for Money Available',
    }
    if platform_key is not None and feature_name == platform_key:
        return f'Platform ({platform})'
    if feature_name.startswith('platform_'):
        return feature_name.replace('platform_', 'Platform ')
    return rename_map.get(feature_name, feature_name)


def get_feature_importance_series(feature_importance):
    if isinstance(feature_importance, pd.Series):
        return feature_importance.copy()
    if isinstance(feature_importance, pd.DataFrame):
        if {'Fitur', 'Importance Score'}.issubset(feature_importance.columns):
            return feature_importance.set_index('Fitur')['Importance Score']
        if {'feature', 'importance'}.issubset(feature_importance.columns):
            return feature_importance.set_index('feature')['importance']
    return pd.Series(feature_importance)


models = load_models()

# ── Header ───────────────────────────────────────────────────
st.markdown("Prediksi Nilai Overall Score")
st.markdown(
    "Masukkan nilai rating per kategori untuk memperoleh prediksi nilai overall_score "
    "dan penjelasan kontribusi fitur terhadap prediksi."
)
st.divider()

# ── Input Form ───────────────────────────────────────────────
st.markdown("### Input Rating")

col1, col2 = st.columns([1, 1])

with col1:
    platform = st.selectbox(
        "Platform OTA",
        list(PLATFORM_AVAILABILITY.keys()),
        help="Pilih platform OTA sumber ulasan"
    )

    cleanliness = st.slider(
        "Cleanliness (Kebersihan)",
        min_value=1.0, max_value=10.0,
        value=9.0, step=0.1
    )
    location = st.slider(
        "Location (Lokasi)",
        min_value=1.0, max_value=10.0,
        value=9.0, step=0.1
    )
    service = st.slider(
        "Service (Layanan)",
        min_value=1.0, max_value=10.0,
        value=9.0, step=0.1
    )

with col2:
    st.markdown("**Ketersediaan Kategori Rating**")
    default_facility = PLATFORM_AVAILABILITY[platform]['facility']
    default_value = PLATFORM_AVAILABILITY[platform]['value_for_money']

    has_facility = st.checkbox(
        "Kategori Facility tersedia?",
        value=default_facility,
        help="Jika tidak tersedia, sistem akan mengisi nilai menggunakan KNN Imputer dan indikator ketersediaan bernilai 0."
    )
    if has_facility:
        facility = st.slider(
            "Facility (Fasilitas)",
            min_value=1.0, max_value=10.0,
            value=9.0, step=0.1
        )
    else:
        facility = np.nan
        st.info("Nilai Facility akan diimputasi oleh sistem.")

    has_value = st.checkbox(
        "Kategori Value for Money tersedia?",
        value=default_value,
        help="Jika tidak tersedia, sistem akan mengisi nilai menggunakan KNN Imputer dan indikator ketersediaan bernilai 0."
    )
    if has_value:
        value_for_money = st.slider(
            "Value for Money",
            min_value=1.0, max_value=10.0,
            value=9.0, step=0.1
        )
    else:
        value_for_money = np.nan
        st.info("Nilai Value for Money akan diimputasi oleh sistem.")

st.divider()

# Predict Button 
predict_btn = st.button(
    "Prediksi Sekarang",
    type="primary",
    use_container_width=True
)

if predict_btn:
    input_values = [cleanliness, location, service]
    if has_facility:
        input_values.append(facility)
    if has_value:
        input_values.append(value_for_money)

    if any(v < 7.0 for v in input_values):
        st.warning(
            "Satu atau lebih rating berada di bawah 7,0 yang merupakan batas bawah "
            "rentang data training. Prediksi tetap ditampilkan, namun hasilnya perlu "
            "diinterpretasikan dengan hati-hati karena berada di luar pola utama data latih."
        )

# Prepare Input
    input_data = {col: 0 for col in models['feature_cols']}

    input_data['cleanliness'] = cleanliness
    input_data['location'] = location
    input_data['service'] = service
    input_data['facility'] = facility
    input_data['value_for_money'] = value_for_money

    input_data['is_facility_available'] = 1 if has_facility else 0
    input_data['is_value_for_money_available'] = 1 if has_value else 0

    platform_key = f'platform_{platform}'
    if platform_key in input_data:
        input_data[platform_key] = 1

    input_df = pd.DataFrame([input_data])[models['feature_cols']]

    # Imputasi untuk kategori yang tidak tersedia
    input_imputed = input_df.copy()
    input_imputed[NUMERIC_COLS] = models['imputer'].transform(input_imputed[NUMERIC_COLS])

    # Scaling fitur rating
    input_scaled = input_imputed.copy()
    input_scaled[NUMERIC_COLS] = models['scaler'].transform(input_imputed[NUMERIC_COLS])

    # Prediksi Regresi 
    pred_score = models['best_reg'].predict(input_scaled)[0]

    # Tampilkan Hasil 
    st.markdown("### Hasil Prediksi")

    col_r1, col_r2 = st.columns([1, 2])

    with col_r1:
        st.metric(
            label="Overall Score Prediksi",
            value=f"{pred_score:.2f}",
            delta="skala 1–10"
        )

    with col_r2:
        st.markdown("**Ringkasan Input**")
        summary_data = {
            'Aspek': [
                'Platform',
                'Cleanliness',
                'Location',
                'Service',
                'Facility',
                'Value for Money',
                'Facility Available',
                'Value for Money Available'
            ],
            'Nilai': [
                platform,
                f"{cleanliness:.1f}",
                f"{location:.1f}",
                f"{service:.1f}",
                f"{input_imputed.loc[0, 'facility']:.2f}" + (" (imputasi)" if not has_facility else ""),
                f"{input_imputed.loc[0, 'value_for_money']:.2f}" + (" (imputasi)" if not has_value else ""),
                '1' if has_facility else '0',
                '1' if has_value else '0',
            ]
        }
        st.dataframe(
            pd.DataFrame(summary_data),
            hide_index=True,
            use_container_width=True
        )

    # SHAP Explanation
    st.divider()
    st.markdown("### Penjelasan Prediksi dengan SHAP")
    st.caption("SHAP menunjukkan fitur yang mendorong prediksi overall_score menjadi lebih tinggi atau lebih rendah.")

    try:
        shap_values = get_shap_values(models['shap_explainer'], input_scaled)

        shap_df = pd.DataFrame({
            'Fitur': models['feature_cols'],
            'SHAP Value': shap_values,
        })

        mask = (
            ~shap_df['Fitur'].str.startswith('platform_') |
            (shap_df['Fitur'] == platform_key)
        )
        shap_df = shap_df[mask]

        shap_df['Fitur'] = shap_df['Fitur'].map(
            lambda x: format_feature_name(x, platform_key, platform)
        )
        shap_df['Abs SHAP'] = shap_df['SHAP Value'].abs()
        shap_df = shap_df.sort_values('Abs SHAP', ascending=False).head(8)

        shap_df['Arah'] = shap_df['SHAP Value'].apply(
            lambda x: 'Menaikkan prediksi' if x > 0 else 'Menurunkan prediksi'
        )

        fig_shap = go.Figure(go.Bar(
            x=shap_df['SHAP Value'],
            y=shap_df['Fitur'],
            orientation='h',
            marker_color=[
                '#28a745' if v > 0 else '#dc3545'
                for v in shap_df['SHAP Value']
            ],
            text=[f"{v:+.4f}" for v in shap_df['SHAP Value']],
            textposition='outside'
        ))
        fig_shap.update_layout(
            title='Kontribusi Fitur terhadap Prediksi Overall Score',
            xaxis_title='SHAP Value (+ menaikkan prediksi, - menurunkan prediksi)',
            height=400,
            margin=dict(t=40, b=20, l=0, r=80)
        )
        st.plotly_chart(fig_shap, use_container_width=True)

        st.dataframe(
            shap_df[['Fitur', 'SHAP Value', 'Arah']].reset_index(drop=True),
            hide_index=True,
            use_container_width=True
        )

    except Exception as e:
        st.info("Penjelasan SHAP tidak tersedia untuk input ini.")
        st.caption(f"Detail: {e}")

    # Feature Importance Global 
    st.divider()
    st.markdown("### Feature Importance Global")
    st.caption(
        "Feature importance menunjukkan kontribusi relatif fitur dalam model XGBoost Weighted secara global."
    )

    try:
        fi = get_feature_importance_series(models['feature_importance'])
        fi = fi.reindex(models['feature_cols']).dropna()

        fi_filtered = fi[
            ~fi.index.str.startswith('is_') &
            (~fi.index.str.startswith('platform_') | (fi.index == platform_key))
        ].copy()

        fi_filtered.index = [format_feature_name(x, platform_key, platform) for x in fi_filtered.index]
        fi_filtered = fi_filtered.sort_values(ascending=True)

        fig_fi = go.Figure(go.Bar(
            x=fi_filtered.values,
            y=fi_filtered.index,
            orientation='h',
            marker_color='#2196F3',
            text=[f"{v:.4f}" for v in fi_filtered.values],
            textposition='outside'
        ))
        fig_fi.update_layout(
            title='Feature Importance Global — XGBoost Weighted',
            xaxis_title='Importance Score',
            height=340,
            margin=dict(t=40, b=20, l=0, r=80)
        )
        st.plotly_chart(fig_fi, use_container_width=True)

    except Exception as e:
        st.info("Feature importance tidak tersedia.")
        st.caption(f"Detail: {e}")
