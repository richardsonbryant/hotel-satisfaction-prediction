import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from pathlib import Path

# Page Config 
st.set_page_config(
    page_title="Dashboard — Favehotel Nagoya Batam",
    page_icon="📊",
    layout="wide"
)

#  Load Data 
@st.cache_data
def load_data():
    return pd.read_csv('data/hotel_reviews_dataset.csv')


df = load_data()

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"

#  Konstanta 
PLATFORMS = sorted(df['platform'].dropna().unique().tolist())
RATING_COLS = [
    col for col in ['cleanliness', 'location', 'service', 'facility', 'value_for_money']
    if col in df.columns
]
COLORS = ['#2196F3', '#4CAF50', '#FF5722', '#9C27B0', '#FF9800']

@st.cache_data
def load_feature_importance():
    fi_path = DATA_DIR / "bab4_xgboost_feature_importance.csv"

    if not fi_path.exists():
        return None

    fi = pd.read_csv(fi_path)

    feature_col = None
    importance_col = None

    for col in ["Fitur", "Feature", "feature"]:
        if col in fi.columns:
            feature_col = col
            break

    for col in ["Importance Score", "Importance score", "importance", "Importance"]:
        if col in fi.columns:
            importance_col = col
            break

    if feature_col is None or importance_col is None:
        return None

    fi = fi[[feature_col, importance_col]].copy()
    fi.columns = ["Fitur", "Importance Score"]

    return fi

#  Header 
st.markdown("## 📊 Dashboard Analitik Favehotel Nagoya Batam")
st.markdown("Gambaran umum data rating dan overall_score dari lima platform OTA.")
st.divider()

#  Filter Sidebar 
with st.sidebar:
    st.markdown("### Filter Data")
    selected_platforms = st.multiselect(
        "Pilih Platform",
        options=PLATFORMS,
        default=PLATFORMS
    )
    score_range = st.slider(
        "Rentang Overall Score",
        min_value=float(df['overall_score'].min()),
        max_value=float(df['overall_score'].max()),
        value=(float(df['overall_score'].min()), float(df['overall_score'].max())),
        step=0.1
    )
df_filtered = df[
    (df['platform'].isin(selected_platforms)) &
    (df['overall_score'] >= score_range[0]) &
    (df['overall_score'] <= score_range[1])
].copy()

if len(df_filtered) == 0:
    st.warning("Tidak ada data yang sesuai dengan filter yang dipilih.")
    st.stop()

# Ringkasan Statistik 
st.markdown("### Ringkasan Statistik")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Total Ulasan", f"{len(df_filtered):,}")
with col2:
    st.metric("Rata-rata Overall Score", f"{df_filtered['overall_score'].mean():.2f}")
with col3:
    st.metric("Platform Terpilih", f"{df_filtered['platform'].nunique()}")
with col4:
    min_score = df_filtered['overall_score'].min()
    max_score = df_filtered['overall_score'].max()
    st.metric("Rentang Score", f"{min_score:.1f}–{max_score:.1f}")

st.divider()

#  Per Platform 
st.markdown("### Analisis Per Platform")

col_l, col_r = st.columns(2)

with col_l:
    platform_counts = df_filtered['platform'].value_counts().reset_index()
    platform_counts.columns = ['Platform', 'Jumlah Ulasan']

    fig_count = px.bar(
        platform_counts,
        x='Platform',
        y='Jumlah Ulasan',
        title='Jumlah Ulasan Per Platform',
        color='Platform',
        text='Jumlah Ulasan'
    )
    fig_count.update_traces(textposition='outside')
    fig_count.update_layout(height=350, showlegend=False, margin=dict(t=40, b=20))
    st.plotly_chart(fig_count, use_container_width=True)

with col_r:
    avg_score = df_filtered.groupby('platform')['overall_score'].mean().reset_index()
    avg_score.columns = ['Platform', 'Rata-rata Overall Score']
    avg_score = avg_score.sort_values('Rata-rata Overall Score', ascending=True)

    fig_avg = px.bar(
        avg_score,
        x='Rata-rata Overall Score',
        y='Platform',
        orientation='h',
        title='Rata-rata Overall Score Per Platform',
        color='Rata-rata Overall Score',
        color_continuous_scale='Blues',
        range_x=[float(df['overall_score'].min()), float(df['overall_score'].max())],
        text=avg_score['Rata-rata Overall Score'].round(2)
    )
    fig_avg.update_traces(textposition='outside')
    fig_avg.update_layout(height=350, coloraxis_showscale=False, margin=dict(t=40, b=20))
    st.plotly_chart(fig_avg, use_container_width=True)

st.divider()

# Distribusi Fitur Rating Per Platform 
st.markdown("### 📊 Distribusi Fitur Rating Per Platform")

selected_rating_feature = st.selectbox(
    "Pilih fitur rating yang ingin dianalisis",
    options=RATING_COLS,
    format_func=lambda x: x.replace("_", " ").title()
)

fig_feature_box = px.box(
    df_filtered,
    x="platform",
    y=selected_rating_feature,
    color="platform",
    title=f"Distribusi {selected_rating_feature.replace('_', ' ').title()} Per Platform",
    labels={
        "platform": "Platform OTA",
        selected_rating_feature: selected_rating_feature.replace("_", " ").title()
    },
    points="outliers",
    color_discrete_sequence=COLORS
)

fig_feature_box.update_layout(
    height=400,
    showlegend=False,
    margin=dict(t=50, b=30)
)

st.plotly_chart(fig_feature_box, use_container_width=True)

# Ringkasan statistik fitur terpilih per platform
feature_summary = (
    df_filtered
    .groupby("platform")[selected_rating_feature]
    .agg(["count", "mean", "median", "min", "max", "std"])
    .round(2)
    .reset_index()
)

feature_summary.columns = [
    "Platform",
    "Jumlah Data",
    "Rata-rata",
    "Median",
    "Minimum",
    "Maksimum",
    "Standar Deviasi"
]

st.dataframe(
    feature_summary,
    hide_index=True,
    use_container_width=True
)
st.divider()

st.markdown("### Distribusi Antar Fitur Rating")

st.caption(
    "Visualisasi ini menampilkan persebaran nilai pada setiap aspek rating, "
    "seperti cleanliness, location, service, facility, dan value_for_money, "
    "tanpa memisahkan data berdasarkan platform."
)

rating_long = df_filtered[RATING_COLS].melt(
    var_name="Fitur Rating",
    value_name="Nilai Rating"
)

rating_long = rating_long.dropna(subset=["Nilai Rating"])

rating_long["Fitur Rating"] = rating_long["Fitur Rating"].str.replace("_", " ").str.title()

fig_rating_cols_box = px.box(
    rating_long,
    x="Fitur Rating",
    y="Nilai Rating",
    color="Fitur Rating",
    title="Distribusi Nilai Setiap Fitur Rating",
    labels={
        "Fitur Rating": "Fitur Rating",
        "Nilai Rating": "Nilai Rating"
    },
    points="outliers",
    color_discrete_sequence=COLORS
)

fig_rating_cols_box.update_layout(
    height=420,
    showlegend=False,
    margin=dict(t=50, b=30)
)

st.plotly_chart(fig_rating_cols_box, use_container_width=True)

# Ringkasan statistik per fitur rating
rating_cols_summary = (
    df_filtered[RATING_COLS]
    .agg(["count", "mean", "median", "min", "max", "std"])
    .T
    .round(2)
    .reset_index()
)

rating_cols_summary.columns = [
    "Fitur Rating",
    "Jumlah Data",
    "Rata-rata",
    "Median",
    "Minimum",
    "Maksimum",
    "Standar Deviasi"
]

rating_cols_summary["Fitur Rating"] = (
    rating_cols_summary["Fitur Rating"]
    .str.replace("_", " ")
    .str.title()
)

st.dataframe(
    rating_cols_summary,
    hide_index=True,
    use_container_width=True
)

st.divider()

# Distribusi Overall Score 
st.markdown("### Distribusi Overall Score")

col_l2, col_r2 = st.columns(2)

with col_l2:
    fig_hist = px.histogram(
        df_filtered,
        x='overall_score',
        nbins=30,
        title='Distribusi Overall Score',
        labels={'overall_score': 'Overall Score', 'count': 'Frekuensi'}
    )
    fig_hist.add_vline(
        x=df_filtered['overall_score'].mean(),
        line_dash='dash',
        line_color='red',
        annotation_text=f"Mean: {df_filtered['overall_score'].mean():.2f}"
    )
    fig_hist.update_layout(height=350, margin=dict(t=40, b=20))
    st.plotly_chart(fig_hist, use_container_width=True)

with col_r2:
    fig_box = px.box(
        df_filtered,
        x='platform',
        y='overall_score',
        color='platform',
        title='Boxplot Overall Score Per Platform',
        points='outliers'
    )
    fig_box.update_layout(height=350, showlegend=False, margin=dict(t=40, b=20))
    st.plotly_chart(fig_box, use_container_width=True)

st.divider()

# Heatmap Rata-rata Per Aspek 
st.markdown("### Rata-rata Rating Per Aspek Per Platform")

if RATING_COLS:
    avg_matrix = df_filtered.groupby('platform')[RATING_COLS].mean().round(2)

    fig_heat = px.imshow(
        avg_matrix,
        text_auto=True,
        color_continuous_scale='RdYlGn',
        title='Heatmap Rata-rata Rating Per Aspek Per Platform',
        labels=dict(x='Aspek', y='Platform', color='Score')
    )
    fig_heat.update_layout(height=400, margin=dict(t=40, b=20))
    st.plotly_chart(fig_heat, use_container_width=True)
else:
    st.info("Kolom rating tidak ditemukan pada dataset.")

st.divider()


# Prioritas Perbaikan Aspek 
st.markdown("### Prioritas Perbaikan Aspek")

st.caption(
    "Bagian ini membantu menentukan aspek rating yang perlu diprioritaskan "
    "berdasarkan kombinasi antara rata-rata rating dan kontribusi fitur dalam model."
)

target_rating = st.slider(
    "Target minimum rata-rata rating",
    min_value=7.0,
    max_value=10.0,
    value=9.0,
    step=0.1
)

aspect_avg = (
    df_filtered[RATING_COLS]
    .mean(numeric_only=True)
    .reset_index()
)

aspect_avg.columns = ["Fitur", "Rata-rata Rating"]

aspect_count = (
    df_filtered[RATING_COLS]
    .count()
    .reset_index()
)

aspect_count.columns = ["Fitur", "Jumlah Data"]

priority_df = aspect_avg.merge(
    aspect_count,
    on="Fitur",
    how="left"
)

fi = load_feature_importance()

if fi is not None:
    fi_rating = fi[fi["Fitur"].isin(RATING_COLS)].copy()

    priority_df = priority_df.merge(
        fi_rating,
        on="Fitur",
        how="left"
    )
else:
    priority_df["Importance Score"] = np.nan

priority_df["Gap terhadap Target"] = (
    target_rating - priority_df["Rata-rata Rating"]
).clip(lower=0)

max_gap = priority_df["Gap terhadap Target"].max()
if max_gap > 0:
    priority_df["Gap Normalized"] = priority_df["Gap terhadap Target"] / max_gap
else:
    priority_df["Gap Normalized"] = 0

if priority_df["Importance Score"].notna().sum() > 0:
    max_importance = priority_df["Importance Score"].max()
    if max_importance > 0:
        priority_df["Importance Normalized"] = priority_df["Importance Score"] / max_importance
    else:
        priority_df["Importance Normalized"] = 0
else:
    priority_df["Importance Normalized"] = 0

priority_df["Skor Prioritas"] = (
    (0.6 * priority_df["Importance Normalized"]) +
    (0.4 * priority_df["Gap Normalized"])
) * 100

def priority_label(score):
    if score >= 70:
        return "Prioritas Tinggi"
    elif score >= 40:
        return "Prioritas Sedang"
    else:
        return "Pantau / Pertahankan"

priority_df["Prioritas"] = priority_df["Skor Prioritas"].apply(priority_label)

priority_df["Fitur Tampil"] = (
    priority_df["Fitur"]
    .str.replace("_", " ")
    .str.title()
)

priority_display = priority_df[
    [
        "Fitur Tampil",
        "Rata-rata Rating",
        "Jumlah Data",
        "Importance Score",
        "Gap terhadap Target",
        "Skor Prioritas",
        "Prioritas"
    ]
].copy()

priority_display.columns = [
    "Aspek Rating",
    "Rata-rata Rating",
    "Jumlah Data",
    "Importance Score",
    "Gap terhadap Target",
    "Skor Prioritas",
    "Prioritas"
]

priority_display = priority_display.sort_values(
    "Skor Prioritas",
    ascending=False
)

priority_display[
    ["Rata-rata Rating", "Importance Score", "Gap terhadap Target", "Skor Prioritas"]
] = priority_display[
    ["Rata-rata Rating", "Importance Score", "Gap terhadap Target", "Skor Prioritas"]
].round(3)

st.dataframe(
    priority_display,
    hide_index=True,
    use_container_width=True
)

fig_priority = px.bar(
    priority_display,
    x="Skor Prioritas",
    y="Aspek Rating",
    orientation="h",
    color="Prioritas",
    title="Prioritas Perbaikan Aspek Rating",
    text="Skor Prioritas"
)

fig_priority.update_traces(texttemplate="%{text:.1f}", textposition="outside")
fig_priority.update_layout(
    height=420,
    yaxis={"categoryorder": "total ascending"},
    margin=dict(t=50, b=30)
)

st.plotly_chart(fig_priority, use_container_width=True)

st.info(
    "Skor prioritas dihitung dari kombinasi kontribusi fitur dalam model dan gap rata-rata rating "
    "terhadap target. Semakin tinggi skor, semakin layak aspek tersebut diprioritaskan untuk evaluasi."
)

st.divider()

# Aspek Terlemah Per Platform 
st.markdown("### Aspek Terlemah Per Platform")

st.caption(
    "Bagian ini menampilkan aspek rating dengan rata-rata terendah pada setiap platform OTA."
)

platform_aspect_mean = (
    df_filtered
    .groupby("platform")[RATING_COLS]
    .mean(numeric_only=True)
)

weakest_rows = []

for platform_name, row in platform_aspect_mean.iterrows():
    weakest_feature = row.idxmin()
    weakest_value = row.min()

    avg_overall = df_filtered[
        df_filtered["platform"] == platform_name
    ]["overall_score"].mean()

    if weakest_value < target_rating:
        note = "Perlu perhatian"
    else:
        note = "Relatif aman"

    weakest_rows.append({
        "Platform": platform_name,
        "Aspek Terlemah": weakest_feature.replace("_", " ").title(),
        "Rata-rata Aspek": weakest_value,
        "Rata-rata Overall Score": avg_overall,
        "Catatan": note
    })

weakest_platform_df = pd.DataFrame(weakest_rows)

weakest_platform_df = weakest_platform_df.sort_values(
    "Rata-rata Aspek",
    ascending=True
)

weakest_platform_display = weakest_platform_df.copy()
weakest_platform_display[
    ["Rata-rata Aspek", "Rata-rata Overall Score"]
] = weakest_platform_display[
    ["Rata-rata Aspek", "Rata-rata Overall Score"]
].round(2)

st.dataframe(
    weakest_platform_display,
    hide_index=True,
    use_container_width=True
)

fig_weakest = px.bar(
    weakest_platform_display,
    x="Platform",
    y="Rata-rata Aspek",
    color="Aspek Terlemah",
    title="Aspek Terlemah Pada Setiap Platform",
    text="Rata-rata Aspek",
    labels={
        "Rata-rata Aspek": "Rata-rata Rating",
        "Platform": "Platform OTA"
    }
)

fig_weakest.update_traces(textposition="outside")
fig_weakest.update_layout(
    height=420,
    margin=dict(t=50, b=30)
)

st.plotly_chart(fig_weakest, use_container_width=True)

st.info(
    "Tabel ini membantu manajemen melihat aspek yang paling lemah pada masing-masing platform, "
    "sehingga evaluasi layanan dapat dilakukan secara lebih spesifik berdasarkan sumber ulasan."
)

st.divider()

# Insight Otomatis 
st.markdown("### Insight Otomatis")

col_a, col_b, col_c = st.columns(3)

best_platform = avg_score.iloc[-1]['Platform']
best_score = avg_score.iloc[-1]['Rata-rata Overall Score']
lowest_platform = avg_score.iloc[0]['Platform']
lowest_score = avg_score.iloc[0]['Rata-rata Overall Score']

overall_avg = df_filtered[RATING_COLS].mean(numeric_only=True)
lowest_aspect = overall_avg.idxmin()
lowest_val = overall_avg.min()

with col_a:
    st.success(
        f"**Platform dengan Rata-rata Tertinggi**\n\n"
        f"{best_platform} memiliki rata-rata overall_score sebesar **{best_score:.2f}**."
    )
with col_b:
    st.warning(
        f"**Platform dengan Rata-rata Terendah**\n\n"
        f"{lowest_platform} memiliki rata-rata overall_score sebesar **{lowest_score:.2f}**."
    )
with col_c:
    st.info(
        f"**Aspek Rating Terendah**\n\n"
        f"{lowest_aspect.replace('_', ' ').title()} memiliki rata-rata sebesar **{lowest_val:.2f}**."
    )

st.divider()
st.markdown(
    "<center><small style='color:#6c757d'>"
    "Favehotel Nagoya Batam — Dashboard Analitik"
    "</small></center>",
    unsafe_allow_html=True
)
