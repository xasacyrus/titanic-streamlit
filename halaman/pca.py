import pandas as pd
import streamlit as st
from utils import load_hasil
import charts

hasil = load_hasil()
p = hasil["pca"]
st.title("Reduksi Dimensi dengan PCA")
st.caption("Pipeline bebas leakage: imputasi + scaling + one-hot -> PCA -> model. Semua fit hanya pada data latih.")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Fitur sebelum PCA", p["n_features"])
c2.metric("Komponen untuk 90% varians", p["k90"], f"-{(1 - p['k90'] / p['n_features']) * 100:.0f}% dimensi", delta_color="off")
c3.metric("Komponen untuk 95% varians", p["k95"], f"-{(1 - p['k95'] / p['n_features']) * 100:.0f}% dimensi", delta_color="off")
c4.metric("Varians PC1 + PC2", f"{sum(p['explained'][:2]) * 100:.1f}%")

t1, t2, t3, t4 = st.tabs(["Scree Plot", "Interpretasi Komponen", "Benchmark Model", "Kesimpulan PCA"])

with t1:
    k = st.slider("Pilih jumlah komponen", 1, p["n_features"], p["k95"])
    cum = p["cumulative"][k - 1]
    a, b = st.columns(2)
    a.metric("Varians yang dipertahankan", f"{cum * 100:.1f}%")
    b.metric("Pengurangan dimensi", f"{(1 - k / p['n_features']) * 100:.0f}%")
    charts.tampil(charts.scree(p, k))

with t2:
    st.write("Fitur dengan *loading* terbesar menentukan makna tiap komponen. Tanda (+/-) menunjukkan arah pengaruhnya.")
    kol = st.columns(3)
    for i, (pc, d) in enumerate(p["loadings"].items()):
        with kol[i]:
            st.markdown(f"**{pc}** ({p['explained'][i] * 100:.1f}%)")
            st.dataframe(pd.Series(d, name="Loading"))
    charts.tampil(charts.scatter2d(p))
    st.caption("Proyeksi 2D menunjukkan kelas selamat dan tidak selamat masih tumpang tindih, "
               "sehingga 2 komponen saja tidak cukup untuk klasifikasi.")

with t3:
    b = pd.DataFrame(p["benchmark"])
    charts.tampil(charts.pca_bench(b))
    model = st.selectbox("Lihat tabel untuk model", b["Model"].unique())
    st.dataframe(b[b["Model"] == model].drop(columns="Model"), hide_index=True)
    charts.tampil(charts.k_curve(p))

with t4:
    b = pd.DataFrame(p["benchmark"])
    lr = b[b["Model"] == "Logistic Regression"].set_index("Praproses")
    sv = b[b["Model"] == "SVM (RBF)"].set_index("Praproses")
    st.markdown(
        f"- **Reduksi dimensi:** {p['n_features']} kolom (setelah one-hot) menjadi {p['k90']} komponen (90% varians) "
        f"atau {p['k95']} komponen (95%).\n"
        f"- **Kinerja:** pada Logistic Regression, AUC CV {lr.loc['Tanpa PCA', 'AUC CV5']:.3f} tanpa PCA, "
        f"{lr.loc['PCA 95%', 'AUC CV5']:.3f} dengan PCA 95%, dan {lr.loc['PCA 90%', 'AUC CV5']:.3f} dengan PCA 90%.\n"
        f"- **Kecepatan:** SVM latih {sv.loc['Tanpa PCA', 'Train (s)']:.3f} s tanpa PCA dan {sv.loc['PCA 95%', 'Train (s)']:.3f} s "
        "dengan PCA 95%. Waktu bergantung pada mesin.\n"
        "- **PCA 2 komponen** hanya layak untuk visualisasi.")
    st.info("Trade-off: komponen utama sulit dijelaskan ke pengguna awam. Karena itu model untuk halaman Prediksi memakai "
            "**fitur asli** agar input pengguna tetap bermakna dan hasilnya mudah dipertanggungjawabkan.")
