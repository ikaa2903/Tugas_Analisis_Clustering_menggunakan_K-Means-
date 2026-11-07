import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42
DEFAULT_FILE = "global_earthquakes_seismic.csv"
DEFAULT_FEATURES = ["depth", "mag", "nst", "dmin", "rms", "gap"]
EXTRA_COLS = ["sig", "tsunami", "confidence_score"]
MAX_POINTS = 5000  # batas titik pada scatter plot agar cepat

st.set_page_config(page_title="Clustering Gempa K-Means", layout="wide")
sns.set_theme(style="whitegrid")


# ------------------------------------------------------------- utilitas
def idn(x, d=2):
    """Format angka gaya Indonesia (koma desimal)."""
    return f"{x:,.{d}f}".replace(",", "X").replace(".", ",").replace("X", ".")


def tampil(fig):
    st.pyplot(fig)
    plt.close(fig)


# ------------------------------------------------- ruang analisis manual
# Isi analisis Anda di sini. Teks mendukung Markdown (**tebal**, - daftar, dll.)
# dan boleh multi-baris memakai tanda kutip tiga (""" ... """).
# Selama masih "" (kosong), aplikasi menampilkan kotak placeholder.
ANALISIS = {
    "ringkasan": "Berdasarkan hasil statistik deskriptif, dataset terdiri dari 10.000 data dengan 25 kolom, dan terdapat enam fitur yang digunakan untuk clustering, yaitu depth, mag, nst, dmin, rms, dan gap. Hasil analisis menunjukkan bahwa setiap variabel memiliki rentang dan variasi nilai yang berbeda. Variabel depth, nst, dan dmin memiliki penyebaran data yang cukup besar serta terindikasi memiliki nilai ekstrem. Oleh karena itu, diperlukan proses preprocessing, seperti pemeriksaan outlier dan standardisasi menggunakan StandardScaler, agar setiap fitur memiliki skala yang seimbang sebelum dilakukan clustering menggunakan algoritma K-Means.",
    # distribusi per fitur (kunci: "dist_<nama fitur>")
    "dist_depth": "Berdasarkan histogram dan boxplot, variabel depth memiliki distribusi yang tidak normal atau cenderung miring ke kanan (right-skewed). Sebagian besar data berada pada nilai depth yang rendah, sedangkan sebagian kecil memiliki nilai yang sangat tinggi. Pada boxplot juga terlihat banyak outlier, yang menunjukkan adanya nilai depth yang jauh berbeda dari mayoritas data. Oleh karena itu, perlu diperhatikan proses preprocessing, terutama normalisasi data, sebelum dilakukan clustering menggunakan algoritma K-Means.",
    "dist_mag": "Berdasarkan histogram dan boxplot, variabel mag memiliki distribusi yang cenderung miring ke kanan (right-skewed). Sebagian besar data berada pada rentang nilai 4,5–5,0, sedangkan semakin tinggi nilai magnitudo, jumlah datanya semakin sedikit. Pada boxplot juga terlihat beberapa outlier dengan nilai magnitudo yang cukup tinggi. Hal ini menunjukkan adanya beberapa kejadian gempa dengan kekuatan yang jauh lebih besar dibandingkan mayoritas data. Oleh karena itu, perlu dilakukan preprocessing dan normalisasi sebelum menggunakan algoritma K-Means.",
    "dist_nst": "Berdasarkan histogram dan boxplot, variabel nst memiliki distribusi yang cenderung miring ke kanan (right-skewed). Sebagian besar data berada pada rentang nilai yang rendah, sekitar 0–100, sedangkan jumlah data semakin sedikit pada nilai yang lebih tinggi. Pada boxplot terlihat banyak outlier, bahkan terdapat beberapa nilai yang mencapai lebih dari 500. Hal ini menunjukkan adanya perbedaan nilai yang cukup besar pada data. Oleh karena itu, perlu diperhatikan penanganan outlier dan normalisasi sebelum dilakukan clustering menggunakan algoritma K-Means.",
    "dist_dmin": "Berdasarkan histogram dan boxplot, variabel dmin memiliki distribusi yang cenderung miring ke kanan (right-skewed). Sebagian besar data berada pada rentang nilai yang rendah, sekitar 0–10, sedangkan jumlah data semakin sedikit pada nilai yang lebih tinggi. Pada boxplot terlihat banyak outlier dengan nilai yang cukup jauh dari mayoritas data, bahkan mencapai lebih dari 50. Hal ini menunjukkan adanya perbedaan nilai yang cukup besar. Oleh karena itu, perlu dilakukan preprocessing dan normalisasi sebelum menerapkan algoritma K-Means.",
    "dist_rms": "Berdasarkan histogram dan boxplot, variabel rms memiliki distribusi yang mendekati normal, meskipun sedikit miring ke kanan. Sebagian besar data berada pada rentang nilai 0,5–1,0. Pada boxplot terlihat beberapa outlier, baik pada nilai rendah maupun tinggi, bahkan terdapat satu nilai yang cukup jauh dari data lainnya. Oleh karena itu, perlu diperhatikan keberadaan outlier dan dilakukan normalisasi sebelum menerapkan algoritma K-Means.",
    "dist_gap": "Berdasarkan histogram dan boxplot, variabel gap memiliki distribusi yang sedikit miring ke kanan (right-skewed). Sebagian besar data berada pada rentang nilai sekitar 50–150, dengan frekuensi tertinggi di sekitar 120–130. Pada boxplot terlihat beberapa outlier pada nilai di atas 220. Hal ini menunjukkan terdapat beberapa data dengan nilai gap yang lebih tinggi dibandingkan mayoritas data. Oleh karena itu, perlu dilakukan normalisasi agar perbedaan skala data tidak memengaruhi hasil clustering menggunakan algoritma K-Means.",
    "korelasi": "Berdasarkan hasil correlation heatmap, terlihat bahwa beberapa variabel memiliki hubungan yang cukup kuat, seperti mag dengan sig sebesar 0,80 yang menunjukkan korelasi positif kuat. Selain itu, mag dengan nst memiliki korelasi positif sebesar 0,55. Sementara itu, terdapat korelasi negatif antara depth dengan confidence_score sebesar -0,49 serta mag dengan gap sebesar -0,42. Variabel lainnya memiliki nilai korelasi yang relatif rendah atau mendekati nol. Secara umum, hubungan antarvariabel cukup bervariasi. Heatmap ini dapat membantu mengetahui hubungan antarfitur sebelum dilakukan clustering menggunakan algoritma K-Means.",
    "elbow_silhouette": "Berdasarkan hasil pengujian menggunakan Elbow Method, Silhouette Score, dan Davies-Bouldin Index, diperoleh bahwa jumlah cluster terbaik adalah K = 4. Pada grafik Elbow, nilai inertia mengalami penurunan yang cukup signifikan hingga K = 4, kemudian penurunannya mulai melambat. Silhouette Score juga mencapai nilai tertinggi pada K = 4 sebesar 0,3066, yang menunjukkan hasil pemisahan cluster terbaik dibandingkan nilai K lainnya. Selain itu, Davies-Bouldin Index memperoleh nilai terendah pada K = 4 sebesar 1,1277. Dengan demikian, K = 4 dipilih sebagai jumlah cluster yang paling optimal karena memberikan hasil terbaik berdasarkan ketiga metode evaluasi yang digunakan.",
    "jumlah_cluster": "Berdasarkan grafik distribusi hasil clustering dengan K = 4, terlihat bahwa setiap cluster memiliki jumlah data yang berbeda. Cluster 0 memiliki jumlah data terbanyak, yaitu 6.742 data, diikuti Cluster 2 sebanyak 2.385 data. Sementara itu, Cluster 1 memiliki 513 data dan Cluster 3 sebanyak 360 data. Hal ini menunjukkan bahwa hasil clustering memiliki distribusi yang tidak merata, karena sebagian besar data terkumpul pada Cluster 0 dan Cluster 2, sedangkan Cluster 1 dan Cluster 3 memiliki jumlah anggota yang lebih sedikit.",
    "scatter": "Berdasarkan grafik hasil clustering dengan K = 4, terlihat bahwa setiap cluster memiliki pola persebaran yang berbeda berdasarkan kedalaman (depth) dan magnitudo (mag). Cluster 0 dan Cluster 2 lebih banyak tersebar pada kedalaman rendah hingga menengah, sedangkan Cluster 3 cenderung berada pada kedalaman yang lebih tinggi. Sementara itu, Cluster 1 memiliki persebaran data yang lebih sedikit dibandingkan cluster lainnya. Secara umum, hasil clustering menunjukkan bahwa setiap kelompok memiliki karakteristik yang berbeda, meskipun terdapat beberapa data yang saling berdekatan. Hal ini menunjukkan bahwa metode clustering dengan K = 4 mampu mengelompokkan data berdasarkan kemiripan nilai kedalaman dan magnitudo.",
    "pca": "Berdasarkan grafik hasil clustering dengan K = 4, terlihat bahwa setiap cluster memiliki pola persebaran yang berbeda pada komponen utama PC1 dan PC2. Cluster 0 memiliki persebaran data paling padat di bagian tengah dan kiri grafik, sedangkan Cluster 1 cenderung berada di bagian atas dengan nilai PC2 yang lebih tinggi. Cluster 2 lebih banyak tersebar di bagian kanan grafik, sementara Cluster 3 memiliki jumlah data yang relatif sedikit dan tersebar di sekitar bagian tengah hingga kanan bawah. Secara umum, hasil clustering menunjukkan bahwa setiap cluster memiliki karakteristik dan pola persebaran yang berbeda. Meskipun masih terdapat beberapa data yang saling tumpang tindih, pengelompokan ini sudah menunjukkan adanya pemisahan data berdasarkan kemiripan karakteristiknya melalui PCA.",
    "karakteristik": "Analisis Umum Hasil ClusteringbBerdasarkan tabel ringkasan karakteristik cluster, grafik perbandingan nilai Z-score, dan heatmap profil cluster, dapat diketahui bahwa data terbagi menjadi empat cluster dengan karakteristik yang berbeda. Cluster 0 memiliki jumlah data paling banyak, yaitu 6.742 data, dengan nilai rata-rata magnitudo sebesar 4,665, jumlah stasiun (nst) sebesar 48,987, dan gap sebesar 114,103. Cluster ini memiliki karakteristik yang relatif mendekati rata-rata keseluruhan data, meskipun nilai magnitudo dan nst cenderung lebih rendah. Sementara itu, cluster 1 terdiri dari 513 data dan memiliki nilai dmin sebesar 22,79, yang menunjukkan jarak minimum yang relatif lebih besar dibandingkan cluster lainnya. Cluster ini juga memiliki nilai RMS yang paling rendah, yaitu 0,674, sehingga menunjukkan karakteristik data dengan tingkat RMS yang relatif kecil. Selanjutnya, cluster 2 memiliki jumlah data sebanyak 2.385 dengan nilai rata-rata magnitudo sebesar 5,243 dan nst sebesar 121,041. Berdasarkan grafik dan heatmap, cluster ini memiliki nilai magnitudo dan nst yang relatif tinggi, sedangkan nilai gap lebih rendah dibandingkan cluster lainnya. Hal ini menunjukkan bahwa cluster 2 didominasi oleh data dengan magnitudo yang lebih besar dan jumlah stasiun pencatat yang lebih banyak. Adapun cluster 3 memiliki jumlah data paling sedikit, yaitu 360 data, tetapi mempunyai nilai rata-rata depth sebesar 514,257. Nilai tersebut jauh lebih tinggi dibandingkan cluster lainnya, sehingga menjadi karakteristik utama cluster 3. Berdasarkan hasil ini, clustering berhasil mengelompokkan data ke dalam beberapa kelompok yang memiliki ciri khas masing-masing berdasarkan kedalaman, magnitudo, jumlah stasiun, jarak minimum, RMS, dan gap. Berdasarkan variabel pembanding yang tidak digunakan dalam proses clustering, yaitu sig, tsunami, dan confidence_score, terlihat bahwa rata-rata nilai sig tertinggi terdapat pada cluster 2, yaitu 440,35. Sementara itu, proporsi kejadian berpotensi tsunami paling tinggi juga terdapat pada cluster 2, yaitu sebesar 0,029 atau sekitar 2,9%. Hal ini menunjukkan bahwa cluster 2 memiliki proporsi kejadian berpotensi tsunami yang relatif lebih tinggi dibandingkan cluster lainnya, meskipun nilainya masih tergolong rendah. Di sisi lain, nilai confidence_score pada seluruh cluster relatif tinggi, yaitu berkisar antara 0,949 hingga 0,979. Secara keseluruhan, hasil clustering menunjukkan bahwa setiap kelompok memiliki karakteristik yang berbeda, terutama pada variabel depth, magnitudo, nst, dan dmin. Oleh karena itu, hasil pengelompokan ini dapat digunakan sebagai dasar untuk memahami pola karakteristik data gempa, tetapi tidak dapat langsung dijadikan sebagai penentu tingkat bahaya gempa atau potensi tsunami tanpa analisis lanjutan.",
}


def analisis(kunci):
    """Tampilkan analisis manual dari ANALISIS[kunci], atau kotak kosong."""
    teks = ANALISIS.get(kunci, "").strip()
    with st.container(border=True):
        if teks:
            st.markdown(f"**Analisis:** {teks}")
        else:
            st.markdown("**Analisis:**")
            st.caption(f"(belum diisi — tulis di `ANALISIS[\"{kunci}\"]` pada app.py)")
            st.write("")
            st.write("")


# ------------------------------------------------------- data & model
@st.cache_data
def load_csv(source):
    return pd.read_csv(source)


@st.cache_data
def preprocess(df, features):
    """Hapus duplikat, imputasi median, standardisasi."""
    df_clean = df.drop_duplicates().reset_index(drop=True)
    X = df_clean[list(features)].replace([np.inf, -np.inf], np.nan)
    X_imp = pd.DataFrame(
        SimpleImputer(strategy="median").fit_transform(X),
        columns=list(features), index=df_clean.index,
    )
    X_sc = pd.DataFrame(
        StandardScaler().fit_transform(X_imp),
        columns=list(features), index=df_clean.index,
    )
    return df_clean, X_imp, X_sc


@st.cache_data
def evaluate_k(X_sc, k_max):
    rows = []
    for k in range(2, k_max + 1):
        model = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
        labels = model.fit_predict(X_sc)
        rows.append({
            "K": k,
            "Inertia": model.inertia_,
            "Silhouette": silhouette_score(X_sc, labels, sample_size=min(5000, len(X_sc)),
                                           random_state=RANDOM_STATE),
            "Davies-Bouldin": davies_bouldin_score(X_sc, labels),
        })
    out = pd.DataFrame(rows)
    out["Penurunan Inertia (%)"] = (-out["Inertia"].pct_change() * 100).round(2)
    return out


@st.cache_data
def run_kmeans(X_sc, k):
    model = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
    labels = model.fit_predict(X_sc)
    sil = silhouette_score(X_sc, labels, sample_size=min(5000, len(X_sc)),
                           random_state=RANDOM_STATE)
    return labels, model.inertia_, sil, davies_bouldin_score(X_sc, labels)


# ---------------------------------------------------- judul & deskripsi
st.title("Analisis dan Visualisasi Clustering Gempa Bumi dengan K-Means")
st.markdown(
    "Dashboard ini mengelompokkan kejadian gempa bumi dunia berdasarkan karakteristik "
    "seismiknya menggunakan algoritma **K-Means**. Data berasal dari "
    "[Global Earthquakes Seismic Telemetry (Kaggle)]"
    "(https://www.kaggle.com/datasets/luthfimuthathohirin/global-earthquakes-seismic-telemetry). "
    "Alur analisis: ringkasan data → eksplorasi (distribusi dan korelasi) → pra-pemrosesan "
    "(hapus duplikat, imputasi median, standardisasi) → pemilihan K (Elbow, Silhouette, "
    "Davies-Bouldin) → hasil dan karakteristik cluster. Setiap grafik disertai ruang "
    "analisis."
)

# ---------------------------------------------------------- sidebar
st.sidebar.header("Pengaturan")
uploaded = st.sidebar.file_uploader("Unggah CSV (opsional)", type="csv")

if uploaded is not None:
    df = load_csv(uploaded)
elif os.path.exists(DEFAULT_FILE):
    df = load_csv(DEFAULT_FILE)
else:
    st.warning(f"File '{DEFAULT_FILE}' tidak ditemukan di folder ini. "
               "Unduh CSV dari Kaggle lalu letakkan di folder yang sama dengan app.py, "
               "atau unggah lewat sidebar.")
    st.stop()

numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
default_feats = [c for c in DEFAULT_FEATURES if c in numeric_cols] or numeric_cols[:4]

features = st.sidebar.multiselect(
    "Fitur untuk clustering (minimal 2)", numeric_cols, default=default_feats,
    help="Disarankan tidak memakai time, latitude/longitude mentah, sig, tsunami, "
         "dan confidence_score sebagai fitur clustering.",
)
if len(features) < 2:
    st.info("Pilih minimal dua fitur clustering di sidebar.")
    st.stop()

df_clean, X_imp, X_sc = preprocess(df, tuple(features))
metrics_k = evaluate_k(X_sc, 10)
best_k = int(metrics_k.loc[metrics_k["Silhouette"].idxmax(), "K"])
best_k_db = int(metrics_k.loc[metrics_k["Davies-Bouldin"].idxmin(), "K"])

k_final = st.sidebar.slider(
    "Jumlah cluster (K)", 2, 10, best_k, key="k_" + "_".join(features),
    help=f"Nilai awal = K dengan Silhouette tertinggi ({best_k}).",
)

st.sidebar.markdown("---")
st.sidebar.subheader("Fitur yang divisualisasikan")
viz_options = list(dict.fromkeys(
    features + [c for c in ["latitude", "longitude", "sig", "tsunami", "confidence_score"]
                if c in df.columns]
))
dist_feats = st.sidebar.multiselect("Distribusi fitur", features, default=features[:3])
corr_default = [c for c in features + EXTRA_COLS if c in df.columns]
corr_feats = st.sidebar.multiselect("Korelasi antar fitur", viz_options,
                                    default=list(dict.fromkeys(corr_default)))
x_axis = st.sidebar.selectbox("Scatter plot: sumbu X", viz_options, index=0)
y_axis = st.sidebar.selectbox("Scatter plot: sumbu Y", viz_options,
                              index=min(1, len(viz_options) - 1))

labels, inertia, sil, dbi = run_kmeans(X_sc, k_final)
hasil = df_clean.copy()
hasil["cluster"] = labels
counts = hasil["cluster"].value_counts().sort_index()
z_means = X_sc.assign(cluster=labels).groupby("cluster")[features].mean()

tab_ringkas, tab_eda, tab_k, tab_hasil, tab_profil = st.tabs([
    "1. Ringkasan Dataset", "2. Distribusi & Korelasi", "3. Elbow & Silhouette",
    "4. Hasil Clustering", "5. Karakteristik Cluster",
])

# ===================================================== 1. RINGKASAN
with tab_ringkas:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Jumlah data", idn(len(df), 0))
    c2.metric("Jumlah kolom", df.shape[1])
    c3.metric("Fitur clustering", len(features))
    c4.metric("Data setelah hapus duplikat", idn(len(df_clean), 0))

    st.subheader("Cuplikan data")
    st.dataframe(df.head(20))

    st.subheader("Statistik deskriptif fitur clustering")
    st.dataframe(df[features].describe().T.round(3))

    analisis("ringkasan")

    with st.expander("Mengapa fitur-fitur ini yang dipakai?"):
        st.markdown(
            "- `time` hanya penanda waktu, bukan karakteristik gempa.\n"
            "- `latitude`/`longitude` mentah kurang cocok untuk jarak Euclidean (longitude "
            "melingkar di ±180°); dipakai untuk visualisasi saja.\n"
            "- `confidence_score` bukan besaran seismik (tampaknya anotasi tambahan).\n"
            "- `sig` diturunkan dari `mag`, sehingga informasinya tumpang tindih.\n"
            "- `tsunami` bersifat biner (0/1), kurang cocok untuk K-Means.\n"
            "- Outlier **dipertahankan** karena magnitudo dan kedalaman ekstrem adalah "
            "kejadian gempa nyata yang informatif."
        )

# ================================================ 2. DISTRIBUSI & KORELASI
with tab_eda:
    st.subheader("Distribusi fitur")
    if not dist_feats:
        st.info("Pilih fitur pada sidebar (Distribusi fitur).")
    for col in dist_feats:
        st.markdown(f"##### {col}")
        c1, c2 = st.columns(2)
        fig, ax = plt.subplots(figsize=(6, 3.3))
        sns.histplot(df[col], kde=True, ax=ax)
        ax.set_title(f"Histogram {col}")
        c1.pyplot(fig)
        plt.close(fig)
        fig, ax = plt.subplots(figsize=(6, 3.3))
        sns.boxplot(x=df[col], ax=ax)
        ax.set_title(f"Boxplot {col}")
        c2.pyplot(fig)
        plt.close(fig)

        analisis(f"dist_{col}")

    st.subheader("Korelasi antar fitur")
    if len(corr_feats) < 2:
        st.info("Pilih minimal dua kolom pada sidebar (Korelasi antar fitur).")
    else:
        corr = df[corr_feats].corr()
        fig, ax = plt.subplots(figsize=(8, 5.5))
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
        ax.set_title("Correlation Heatmap")
        tampil(fig)

        analisis("korelasi")

# ====================================================== 3. ELBOW & SILHOUETTE
with tab_k:
    c1, c2, c3 = st.columns(3)
    fig, ax = plt.subplots(figsize=(5, 3.6))
    ax.plot(metrics_k["K"], metrics_k["Inertia"], marker="o")
    ax.axvline(k_final, color="red", linestyle="--", alpha=0.5)
    ax.set_title("Elbow Method (Inertia)")
    ax.set_xlabel("K")
    ax.set_ylabel("Inertia")
    c1.pyplot(fig)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 3.6))
    ax.plot(metrics_k["K"], metrics_k["Silhouette"], marker="o", color="tab:orange")
    ax.axvline(k_final, color="red", linestyle="--", alpha=0.5)
    ax.set_title("Silhouette Score (makin tinggi makin baik)")
    ax.set_xlabel("K")
    ax.set_ylabel("Silhouette")
    c2.pyplot(fig)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 3.6))
    ax.plot(metrics_k["K"], metrics_k["Davies-Bouldin"], marker="o", color="tab:green")
    ax.axvline(k_final, color="red", linestyle="--", alpha=0.5)
    ax.set_title("Davies-Bouldin (makin rendah makin baik)")
    ax.set_xlabel("K")
    ax.set_ylabel("Davies-Bouldin")
    c3.pyplot(fig)
    plt.close(fig)
    st.caption("Garis putus-putus merah menandai K yang dipilih di sidebar.")

    st.dataframe(metrics_k.round(4))

    analisis("elbow_silhouette")

# ======================================================== 4. HASIL CLUSTERING
with tab_hasil:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("K", k_final)
    c2.metric("Inertia", idn(inertia, 1))
    c3.metric("Silhouette", idn(sil, 4))
    c4.metric("Davies-Bouldin", idn(dbi, 4))

    st.subheader("Jumlah data per cluster")
    fig, ax = plt.subplots(figsize=(7, 3.3))
    sns.barplot(x=counts.index.astype(str), y=counts.values, ax=ax)
    ax.set_xlabel("Cluster")
    ax.set_ylabel("Jumlah data")
    for i, v in enumerate(counts.values):
        ax.text(i, v, f"{v:,}", ha="center", va="bottom", fontsize=9)
    tampil(fig)
    analisis("jumlah_cluster")

    st.subheader("Scatter plot cluster")
    plot_df = hasil.sample(n=min(MAX_POINTS, len(hasil)), random_state=RANDOM_STATE)
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.scatterplot(data=plot_df, x=x_axis, y=y_axis, hue="cluster", palette="tab10",
                    alpha=0.7, s=18, ax=ax)
    ax.set_title(f"{x_axis} vs {y_axis} (sampel maks. {MAX_POINTS:,} titik)")
    tampil(fig)

    analisis("scatter")

    st.subheader("Proyeksi PCA 2D")
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    comp = pca.fit_transform(X_sc)
    pca_df = pd.DataFrame(comp, columns=["PC1", "PC2"])
    pca_df["cluster"] = labels
    pca_s = pca_df.sample(n=min(MAX_POINTS, len(pca_df)), random_state=RANDOM_STATE)
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.scatterplot(data=pca_s, x="PC1", y="PC2", hue="cluster", palette="tab10",
                    alpha=0.7, s=18, ax=ax)
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}%)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}%)")
    tampil(fig)
    analisis("pca")

    st.subheader("Label cluster pada data")
    st.dataframe(hasil[["cluster"] + [c for c in features] +
                       [c for c in ["latitude", "longitude", "place"] if c in hasil.columns]]
                 .head(50))
    st.download_button("Unduh hasil clustering (CSV)",
                       hasil.to_csv(index=False).encode("utf-8"),
                       file_name="Hasil_Clustering_KMeans.csv", mime="text/csv")

# ====================================================== 5. KARAKTERISTIK
with tab_profil:
    stat = st.radio("Statistik", ["Mean", "Median"], horizontal=True)
    grouped = X_imp.assign(cluster=labels).groupby("cluster")[features]
    profil = grouped.mean() if stat == "Mean" else grouped.median()

    ciri = {}
    for c in z_means.index:
        top = z_means.loc[c].reindex(z_means.loc[c].abs().sort_values(ascending=False).index)[:3]
        ciri[c] = ", ".join(f"{f} ({v:+.2f})".replace(".", ",") for f, v in top.items())

    ringkas = profil.round(3).copy()
    ringkas.insert(0, "jumlah_data", counts)
    ringkas["ciri_utama (z-score)"] = pd.Series(ciri)
    st.subheader(f"Tabel ringkasan karakteristik cluster ({stat.lower()}, satuan asli)")
    st.dataframe(ringkas)

    extras = [c for c in EXTRA_COLS if c in hasil.columns and c not in features]
    ex_tab = None
    if extras:
        st.subheader("Variabel pembanding (tidak dipakai clustering)")
        ex_tab = hasil.groupby("cluster")[extras].mean().round(3)
        st.dataframe(ex_tab)
        st.caption("Untuk tsunami, nilai rata-rata = proporsi kejadian berpotensi tsunami.")

    st.subheader("Rata-rata fitur terstandar per cluster")
    fig, ax = plt.subplots(figsize=(10, 4.5))
    z_means.T.plot(kind="bar", ax=ax, width=0.8)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("z-score (0 = rata-rata seluruh data)")
    plt.xticks(rotation=0)
    tampil(fig)

    fig, ax = plt.subplots(figsize=(8, 2.5 + 0.45 * k_final))
    sns.heatmap(z_means, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
    ax.set_title("Profil cluster (rata-rata z-score)")
    tampil(fig)

    analisis("karakteristik")