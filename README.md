# 🏨 Hotel Guest Satisfaction Prediction — Data Mining untuk Prediksi Kepuasan Tamu Hotel

> Sistem prediksi kepuasan tamu hotel berbasis Machine Learning, dibangun dari data rating lintas 5 platform OTA (Online Travel Agency), lengkap dengan dashboard analitik dan aplikasi web interaktif.

**Status:** 🚧 Skripsi (Tugas Akhir) — CRISP-DM Methodology
**Program Studi:** S1 Ilmu Komputer — BINUS University

---

## ⚠️ Catatan Kerahasiaan Data

Repository ini **tidak menyertakan dataset asli maupun ringkasan statistik dari pihak hotel**, karena bersifat rahasia dan hanya digunakan untuk keperluan akademik dengan izin terbatas dari stakeholder. Nama hotel dalam dokumentasi ini disamarkan menjadi **"Hotel XYZ"**.

Folder `data/*.csv` dan `webapp/data/*.csv` sengaja di-*gitignore*. Untuk mereproduksi pipeline, silakan gunakan dataset Anda sendiri dengan struktur kolom yang sama (lihat bagian [Dataset](#-dataset)).

---

## 📌 Daftar Isi

- [Latar Belakang](#-latar-belakang)
- [Rumusan Masalah](#-rumusan-masalah)
- [Tujuan Penelitian](#-tujuan-penelitian)
- [Dataset](#-dataset)
- [Metodologi](#-metodologi-crisp-dm)
- [Struktur Repository](#-struktur-repository)
- [Exploratory Data Analysis](#-exploratory-data-analysis)
- [Data Preprocessing](#-data-preprocessing)
- [Pemodelan Machine Learning](#-pemodelan-machine-learning)
- [Eksperimen Weighted Training](#-eksperimen-weighted-training)
- [Hasil & Evaluasi](#-hasil--evaluasi)
- [Model Terpilih & Justifikasi](#-model-terpilih--justifikasi)
- [Aplikasi Web (Streamlit)](#-aplikasi-web-streamlit)
- [Cara Menjalankan](#-cara-menjalankan)
- [Keterbatasan Penelitian](#-keterbatasan-penelitian)
- [Rencana Pengembangan](#-rencana-pengembangan)
- [Tech Stack](#-tech-stack)

---

## 🎯 Latar Belakang

**Hotel XYZ** terdaftar di lima platform *Online Travel Agency* (OTA) sekaligus — **Traveloka, Booking.com, Trip.com, Agoda,** dan **Tiket.com**. Setiap platform menyajikan skema rating yang berbeda-beda (jumlah kategori, penamaan atribut, dan skala penilaian tidak seragam), sehingga manajemen hotel kesulitan mendapatkan **gambaran kepuasan tamu secara terintegrasi**.

Dua permasalahan teknis utama teridentifikasi dari data:

1. **Structural Missing Data** — Kekosongan nilai pada beberapa kolom (misalnya `facility`, `comfort`, `value_for_money`) bukan disebabkan kesalahan input, melainkan karena kolom tersebut memang tidak tersedia secara struktural di platform tertentu.
2. **Ceiling Effect & Class Imbalance** — Mayoritas rating berada pada rentang tinggi (7.0–10.0) dengan distribusi jumlah data yang tidak seimbang antar platform (satu platform mendominasi ±45% dari total data).

Penelitian ini menjawab kebutuhan tersebut dengan membangun **sistem prediksi kepuasan tamu berbasis Machine Learning** yang mengintegrasikan data dari lima platform sekaligus dan diimplementasikan dalam bentuk **aplikasi web** yang dapat digunakan langsung oleh manajemen hotel.

---

## ❓ Rumusan Masalah

1. Bagaimana pipeline *preprocessing* yang tepat untuk menangani *structural missing data* pada dataset multi-platform OTA?
2. Algoritma *machine learning* mana yang paling optimal untuk memprediksi *overall score* kepuasan tamu?
3. Bagaimana mengatasi bias yang timbul akibat ketidakseimbangan jumlah data antar platform (*platform bias*)?
4. Bagaimana merancang dan mengimplementasikan sistem web yang dapat digunakan manajemen hotel untuk memprediksi kepuasan tamu secara *real-time*?

---

## 🎯 Tujuan Penelitian

- Membangun sistem prediksi kepuasan tamu berbasis *machine learning* dari data multi-platform OTA.
- Merancang strategi *feature engineering* untuk menangani *structural missing data* (indicator variables + imputasi KNN).
- Membangun dan membandingkan performa 5 algoritma regresi: **Random Forest, XGBoost, LightGBM, Ridge Regression,** dan **SVR**.
- Mengidentifikasi dan mengatasi bias platform melalui eksperimen ***weighted training***.
- Mengimplementasikan model terbaik ke dalam **aplikasi web interaktif** lengkap dengan visualisasi *dashboard* dan interpretasi model berbasis **SHAP (SHapley Additive exPlanations)**.

---

## 📊 Dataset

> 🔒 Dataset asli **tidak disertakan** dalam repository ini (lihat [Catatan Kerahasiaan Data](#️-catatan-kerahasiaan-data)). Bagian ini menjelaskan struktur data secara umum untuk keperluan reproduksi.

Dataset dikonstruksi berdasarkan **ringkasan statistik (summary report)** yang diperoleh langsung dari pihak manajemen hotel (data primer via wawancara daring), kemudian direkonstruksi menjadi data individual per-baris menggunakan simulasi distribusi probabilistik yang mempertahankan karakteristik statistik asli (rata-rata, jumlah responden, dan struktur kolom per platform).

| Platform      | Proporsi Data (approx.) | Kolom Rating Tersedia |
|---------------|:------------------------:|------------------------|
| Agoda         | ~45%   | Overall, Cleanliness, Facility, Location, Service, Price & Value |
| Tiket.com     | ~19%   | Overall, Facility, Location, Cleanliness, Service, Value for Money |
| Traveloka     | ~17%   | Overall, Cleanliness, Room Comfort, Meal, Location, Service & Facility |
| Trip.com      | ~10%   | Overall, Cleanliness, Amenities, Service, Location |
| Booking.com   | ~8%    | Overall, Cleanliness, Comfort, Staff, Location, Facility, Value for Money, Free WiFi |
| **Total**     | **~4.900 baris** | |

**Fitur final digunakan (setelah seleksi):**
`cleanliness`, `location`, `service`, `facility`, `value_for_money`, `is_facility_available`, `is_value_for_money_available`, `platform_*` (one-hot encoded)

**Kolom yang di-*drop*** (korelasi sangat lemah terhadap `overall_score` & *missing rate* > 70%): `meal`, `free_wifi`, `amenities`, `comfort`.

---

## 🔬 Metodologi (CRISP-DM)

Penelitian mengikuti kerangka kerja **CRISP-DM** (*Cross-Industry Standard Process for Data Mining*) sesuai template BINUS:

```
1. Business Understanding   → Identifikasi masalah multi-platform OTA
2. Data Understanding       → EDA (01_EDA.ipynb)
3. Data Preparation         → Preprocessing (02_Preprocessing.ipynb)
4. Modeling                 → 03, 03b (5 algoritma × 2 skenario training)
5. Evaluation                → 05b (evaluasi, SHAP, error analysis)
6. Deployment                → Aplikasi web Streamlit
```

---

## 🗂 Struktur Repository

```
hotel-satisfaction-prediction/
├── data/                                 # 🔒 CSV di-gitignore (data rahasia)
├── models/                               # Model terlatih (.pkl)
│   ├── best_regression_model.pkl         # Model final (XGBoost Weighted)
│   ├── xgb_weighted_model.pkl
│   ├── shap_explainer_xgb.pkl
│   ├── scaler.pkl / feature_cols.pkl / knn_imputer.pkl
│   └── ...
│
├── plots/                                # Visualisasi hasil analisis
│
├── 01_EDA.ipynb                          # Exploratory Data Analysis
├── 02_Preprocessing.ipynb                # Data cleaning & feature engineering
├── 03_Modelling_Regression.ipynb         # Baseline modeling (5 algoritma)
├── 03b_Weighted_Experiment.ipynb         # Eksperimen weighted training
├── 04_Modelling_Classification.ipynb     # Eksperimen klasifikasi (dokumentasi, tidak digunakan di web)
├── 05_Evaluation.ipynb                   # Evaluasi awal (SVR/Ridge — arsip)
├── 05b_Evaluation.ipynb                  # Evaluasi model final (XGBoost Weighted)
│
└── webapp/                               # Aplikasi web Streamlit
    ├── app.py                            # Landing page
    ├── pages/
    │   ├── 1_Prediksi.py                 # Form prediksi + SHAP explanation
    │   ├── 2_Dashboard.py                # Dashboard analitik interaktif
    │   └── 3_Tentang.py                  # Metodologi & performa model
    ├── models/ · data/ (🔒 gitignored)
    └── requirements.txt
```

---

## 🔍 Exploratory Data Analysis

Analisis dilakukan pada 8 aspek utama (`01_EDA.ipynb`):

- **Missing values** — dikonfirmasi bersifat *structural* (100% NaN per kolom pada platform yang tidak memiliki atribut tersebut, bukan random).
- **Distribusi & ceiling effect** — skewness `overall_score` mendekati 0 (distribusi relatif normal setelah rekonstruksi data).
- **Korelasi terhadap `overall_score`** — `facility` dan `cleanliness` konsisten menjadi fitur dengan korelasi tertinggi di seluruh tahap analisis.
- **Ketidakseimbangan data antar platform** — satu platform mendominasi ±45% dari total data, menjadi akar masalah *platform bias* yang dibahas lebih lanjut pada eksperimen *weighted training*.

---

## 🧹 Data Preprocessing

Pipeline preprocessing (`02_Preprocessing.ipynb`) mengikuti tahapan berikut:

| Tahap | Deskripsi |
|---|---|
| 1. Drop kolom sparse | `meal`, `free_wifi`, `amenities`, `comfort` (missing > 70%, korelasi ≈ 0) |
| 2. Indicator variables | `is_facility_available`, `is_value_for_money_available` — menandai *structural missingness* |
| 3. Imputasi | KNN Imputer (k=5) untuk `facility` & `value_for_money` |
| 4. Encoding | One-Hot Encoding untuk kolom `platform` |
| 5. Train-Test Split | 80:20, stratifikasi berdasarkan platform dominan |
| 6. Scaling | MinMaxScaler — **di-*fit* hanya pada *training set*** untuk mencegah *data leakage* |

> ⚠️ **Catatan metodologis:** Pada iterasi awal ditemukan bahwa `MinMaxScaler` sempat di-*fit* pada keseluruhan dataset (train+test) sebelum split — hal ini telah dikoreksi dengan memindahkan proses *scaling* ke **setelah** train-test split.

---

## 🤖 Pemodelan Machine Learning

Lima algoritma regresi dibandingkan untuk memprediksi `overall_score` (skala kontinu 7.0–10.0), masing-masing di-*tuning* menggunakan **5-Fold Cross Validation**:

| Algoritma | Peran | Metode Tuning |
|---|---|---|
| Random Forest | Baseline | Default parameters |
| **XGBoost** | Model utama | RandomizedSearchCV (50 iterasi) |
| LightGBM | Comparator | RandomizedSearchCV (50 iterasi) |
| Ridge Regression | Baseline linear | GridSearchCV (exhaustive) |
| SVR | Non-linear alternatif | RandomizedSearchCV (50 iterasi) |

### Hasil Baseline (`03_Modelling_Regression.ipynb`)

| Model | RMSE (Test) | MAE (Test) | R² (Test) | Overfitting Gap |
|---|:---:|:---:|:---:|:---:|
| SVR (Tuned) | 0.1527 | 0.1229 | **0.6277** | -0.0048 ✅ |
| Ridge Regression (Tuned) | 0.1531 | 0.1229 | 0.6258 | -0.0129 ✅ |
| XGBoost (Tuned) | 0.1556 | 0.1248 | 0.6135 | 0.0434 ✅ |
| LightGBM (Tuned) | 0.1560 | 0.1257 | 0.6114 | 0.0624 ✅ |
| Random Forest (Baseline) | 0.1613 | 0.1290 | 0.5849 | 0.3472 ⚠️ Overfit |

---

## ⚖️ Eksperimen Weighted Training

### Masalah yang Ditemukan

*Feature importance* pada model baseline menunjukkan salah satu fitur platform sebagai fitur paling dominan — **melebihi** fitur rating substantif seperti `facility` dan `cleanliness`. Hal ini merupakan indikasi *platform bias* yang muncul akibat dominasi satu platform (±45% dari total data), bukan representasi hubungan bisnis yang valid antara kualitas layanan dan kepuasan tamu.

### Solusi: Sample Weighting

Diterapkan bobot sampel berdasarkan formula:

```
weight = total_samples / (n_platforms × n_samples_platform)
```

Platform dengan proporsi data terkecil mendapat bobot terbesar (≈2.6×), sedangkan platform dominan mendapat bobot terkecil (≈0.44×), sehingga kontribusi tiap platform terhadap proses pembelajaran model menjadi lebih seimbang.

### Dampak terhadap Feature Importance

*Weighted training* berhasil **menurunkan dominasi fitur platform** secara signifikan pada seluruh model *tree-based*, dengan `facility` konsisten menjadi fitur paling berpengaruh — hasil yang jauh lebih selaras dengan logika bisnis *hospitality* (kualitas layanan sebagai penentu utama kepuasan, bukan platform pemesanan).

### Trade-off pada Metrik Global

Eksperimen menunjukkan bahwa mengatasi *platform bias* melalui *weighted training* memiliki **konsekuensi penurunan R² pada skala global** (rata-rata −2.5% s.d. −3% di seluruh model) — sebuah *trade-off* yang secara sadar dipilih demi model yang lebih adil dan interpretable secara bisnis, bukan sekadar mengejar angka metrik tertinggi.

---

## 📈 Hasil & Evaluasi

### Perbandingan Lengkap: Baseline vs Weighted (5 Model)

| Model | R² Baseline | R² Weighted | Δ R² |
|---|:---:|:---:|:---:|
| SVR | 0.6564 | 0.6275 | −0.0289 |
| Ridge | 0.6528 | 0.6260 | −0.0268 |
| **XGBoost** | 0.6375 | **0.6109** | −0.0266 |
| LightGBM | 0.6351 | 0.6102 | −0.0249 |
| Random Forest | 0.6198 (⚠️ overfit) | 0.5882 (⚠️ overfit) | −0.0316 |

### Evaluasi Per Platform (Weighted, R²)

R² dievaluasi per platform untuk memastikan tidak ada platform yang secara sistematis diprediksi jauh lebih buruk dibanding lainnya (indikasi *fairness* model):

| Platform | Random Forest | XGBoost | LightGBM | Ridge | SVR |
|---|:---:|:---:|:---:|:---:|:---:|
| Platform A | 0.4907 | 0.5029 | 0.5075 | 0.5293 | 0.5242 |
| Platform B | 0.4548 | 0.4563 | 0.4561 | 0.5047 | 0.4581 |
| Platform C | 0.3809 | 0.4511 | 0.4431 | 0.4631 | 0.4603 |
| Platform D | 0.3976 | 0.4339 | 0.4278 | 0.4486 | 0.4520 |
| Platform E | 0.4726 | 0.5080 | 0.5079 | 0.5128 | **0.5724** |

### Benchmark: Model ML vs Rata-Rata Sederhana

Untuk menjustifikasi penggunaan *machine learning* dibandingkan pendekatan *rule-based* sederhana (rata-rata seluruh kolom rating), dilakukan uji banding langsung:

| Metrik | Rata-rata Sederhana | XGBoost (Weighted) | Selisih |
|---|:---:|:---:|:---:|
| RMSE | 0.1559 | 0.1561 | −0.0002 |
| MAE | 0.1249 | 0.1256 | −0.0007 |
| R² | 0.6120 | 0.6109 | −0.0011 |

> **Catatan jujur:** Pada eksperimen ini, pendekatan rata-rata sederhana ternyata menghasilkan performa yang **sebanding, bahkan sedikit lebih baik** secara metrik murni dibanding XGBoost. Namun, nilai tambah pendekatan ML tetap terletak pada: (1) kemampuan menangkap bobot kontribusi setiap aspek secara data-driven melalui *feature importance* & SHAP, (2) interpretasi *per-kasus* yang tidak dapat diberikan oleh rata-rata statis, dan (3) fleksibilitas menangani *structural missing data* melalui *indicator variables*. Temuan ini didokumentasikan secara transparan sebagai bagian dari evaluasi kritis penelitian.

---

## 🏆 Model Terpilih & Justifikasi

**Model final untuk implementasi web: `XGBoost (Weighted)`**

| Kriteria | SVR | Ridge | **XGBoost** | LightGBM | Random Forest |
|---|:---:|:---:|:---:|:---:|:---:|
| Bias Platform | Rendah | Rendah | Rendah | Rendah | Sedang |
| Risiko Overfitting | Tidak | Tidak | Tidak | Tidak | **Ya** |
| SHAP dapat disimpan (*deployable*) | ❌ Tidak | ✅ Ya | ✅ Ya | ✅ Ya | ✅ Ya |
| *Feature Importance* native | ❌ Tidak | ⚠️ Koefisien seragam | ✅ Ya | ✅ Ya | ✅ Ya |
| Kecepatan Prediksi | ⚠️ Lambat | ✅ Cepat | ✅ Cepat | ✅ Cepat | ✅ Cepat |
| **Siap untuk Deployment** | ❌ | ⚠️ Partial | **✅** | ✅ | ⚠️ Overfit |

**Alasan pemilihan XGBoost Weighted** (meskipun bukan R² tertinggi):

1. Selisih performa dengan SVR (model R² tertinggi) **tidak signifikan secara praktis**.
2. `SHAP KernelExplainer` milik SVR **tidak dapat di-*serialize*** (`joblib.dump` gagal — Cython memory view tidak mendukung *pickling*), sehingga tidak layak untuk *real-time inference* di web.
3. Koefisien Ridge Regression bernilai **seragam** akibat regularisasi L2 yang kuat, sehingga tidak informatif untuk interpretasi bisnis.
4. XGBoost mendukung `SHAP TreeExplainer` yang **cepat, akurat, dan dapat disimpan** — ideal untuk *deployment*.
5. *Platform bias* telah tertangani melalui *weighted training*, menghasilkan interpretasi model yang lebih selaras dengan realita bisnis *hospitality*.

---

## 💻 Aplikasi Web (Streamlit)

Sistem diimplementasikan sebagai aplikasi web multi-halaman menggunakan **Streamlit**:

| Halaman | Fungsi |
|---|---|
| 🏠 **Beranda** | Landing page, ringkasan statistik dataset |
| 🔮 **Prediksi** | Form input rating → prediksi *overall score* + status kepuasan (threshold manual **8.5**) + interpretasi **SHAP** per-prediksi + *feature importance* global |
| 📊 **Dashboard** | Visualisasi interaktif: distribusi per platform, heatmap rating, perbandingan *Satisfied* vs *Unsatisfied* |
| ℹ️ **Tentang Sistem** | Transparansi metodologi, performa model, dan keterbatasan penelitian |

**Fitur interpretabilitas:**
- **SHAP Waterfall** — menjelaskan kontribusi setiap aspek terhadap satu prediksi spesifik dibandingkan baseline rata-rata dataset.
- **Feature Importance Global** — insight strategis untuk manajemen hotel (prioritas alokasi perbaikan layanan).
- *Warning system* otomatis untuk input di luar rentang data training (< 7.0).

---

## ⚙️ Cara Menjalankan

### 1. Clone & Setup Environment

```bash
git clone https://github.com/richardsonbryant/hotel-satisfaction-prediction.git
cd hotel-satisfaction-prediction
python3 -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
```

### 2. Siapkan Dataset

> 🔒 Dataset tidak disertakan dalam repo ini. Siapkan file `data/hotel_reviews_dataset.csv` dengan struktur kolom sesuai bagian [Dataset](#-dataset) di atas sebelum menjalankan notebook.

### 3. Jalankan Notebook (opsional — reproduksi eksperimen)

```bash
pip install pandas numpy scikit-learn xgboost lightgbm shap \
            matplotlib seaborn plotly jupyter ipykernel imbalanced-learn scipy
jupyter notebook
```
Jalankan berurutan: `01_EDA` → `02_Preprocessing` → `03_Modelling_Regression` → `03b_Weighted_Experiment` → `05b_Evaluation`

### 4. Jalankan Aplikasi Web

```bash
cd webapp
pip install -r requirements.txt
streamlit run app.py
```
Aplikasi akan berjalan di `http://localhost:8501`

> **Catatan:** XGBoost & LightGBM di macOS memerlukan `libomp` — instal via `brew install libomp` jika muncul error `Library not loaded: libomp.dylib`.

---

## ⚠️ Keterbatasan Penelitian

- **Data sintetis berbasis summary** — dataset individual direkonstruksi dari ringkasan statistik pihak hotel (bukan data mentah per tamu), karena keterbatasan akses data individual dari stakeholder.
- **R² ≈ 0.61** mengindikasikan sekitar 39% variasi *overall score* belum dapat dijelaskan oleh fitur rating kategori yang tersedia — kemungkinan dipengaruhi faktor subjektif lain (ekspektasi personal, sentimen ulasan teks, dsb.) yang tidak tercakup dalam dataset numerik ini.
- Fitur terbatas pada **rating numerik**, tidak mencakup analisis sentimen dari teks ulasan.
- Model klasifikasi (Satisfied/Unsatisfied berbasis Ridge Classifier) dieksplorasi pada `04_Modelling_Classification.ipynb` namun **tidak digunakan pada implementasi akhir** — sistem web menggunakan *threshold* manual (`overall_score ≥ 8.5`) di atas hasil prediksi regresi untuk menjaga konsistensi output.

---

## 🚧 Rencana Pengembangan

- [ ] Analisis sentimen dari teks ulasan (jika data tersedia)
- [ ] Perbandingan performa model *multi-platform* vs *single-platform* (menunggu arahan dosen pembimbing)
- [ ] Deployment publik via Streamlit Community Cloud
- [ ] Evaluasi *usability* sistem (System Usability Scale) dengan pengguna manajemen hotel

---

## 🛠 Tech Stack

**Data Processing & ML:** Python · Pandas · NumPy · Scikit-learn · XGBoost · LightGBM · SHAP
**Visualisasi:** Matplotlib · Seaborn · Plotly
**Web Application:** Streamlit
**Environment:** Jupyter Notebook · venv

---

## 📄 Lisensi & Atribusi

Proyek ini merupakan bagian dari Tugas Akhir (Skripsi) Program Studi S1 Ilmu Komputer, BINUS University. Nama dan data stakeholder disamarkan untuk menjaga kerahasiaan sesuai kesepakatan dengan pihak hotel.

---

<p align="center"><i>Dibangun dengan pendekatan CRISP-DM untuk menjawab kebutuhan nyata industri perhotelan dalam mengintegrasikan data kepuasan tamu lintas platform OTA.</i></p>
