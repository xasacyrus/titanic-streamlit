import pandas as pd
import streamlit as st
from utils import BUANG, KAMUS, load_data, load_raw
import charts

df, raw = load_data(), load_raw()
st.title("Tentang Dataset")
st.caption("Sumber: Kaggle - Titanic: Machine Learning from Disaster (format CSV)")

tab1, tab2, tab3, tab4 = st.tabs(["Ringkasan", "Kamus Kolom", "Lihat Data", "Kualitas Data"])

with tab1:
    n_selamat = int(df["Survived"].sum())
    n_tidak = int((df["Survived"] == 0).sum())
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Baris", f"{df.shape[0]:,}")
    c2.metric("Kolom", df.shape[1])
    c3.metric("Penumpang selamat", f"{n_selamat:,}")
    c4.metric("Rasio kelas", f"{n_tidak / n_selamat:.2f} : 1")
    st.markdown(
        "- **Jenis masalah:** klasifikasi biner (supervised learning)\n"
        "- **Target:** `Survived` (1 = selamat, 0 = tidak selamat)\n"
        "- **Tujuan:** memprediksi apakah seorang penumpang selamat berdasarkan data dirinya dan tiketnya\n"
        "- **Catatan:** kelas cukup tidak seimbang, jadi akurasi saja tidak cukup")
    charts.tampil(charts.distribusi_kelas(df))

with tab2:
    kamus = pd.DataFrame([(k, v[0], v[1], str(raw[k].dtype)) for k, v in KAMUS.items()],
                         columns=["Kolom", "Kelompok", "Keterangan", "Tipe data"])
    grup = sorted(kamus["Kelompok"].unique())
    pilih = st.multiselect("Filter kelompok", grup, default=grup)
    st.dataframe(kamus[kamus["Kelompok"].isin(pilih)], hide_index=True)

with tab3:
    kiri, kanan = st.columns(2)
    kelas = kiri.multiselect("Kelas tiket", sorted(df["Pclass"].unique().tolist()),
                             default=sorted(df["Pclass"].unique().tolist()))
    status = kanan.radio("Status", ["Semua", "Selamat", "Tidak selamat"], horizontal=True)
    sub = df[df["Pclass"].isin(kelas)]
    if status == "Selamat":
        sub = sub[sub["Survived"] == 1]
    elif status == "Tidak selamat":
        sub = sub[sub["Survived"] == 0]
    st.write(f"Menampilkan {len(sub):,} baris")
    st.dataframe(sub.head(500))
    st.download_button("Download data terfilter (CSV)", sub.to_csv(index=False).encode("utf-8"),
                       file_name="titanic_terfilter.csv", mime="text/csv")

with tab4:
    hilang = raw.isna().sum()
    hilang = hilang[hilang > 0]
    st.subheader("Data hilang per kolom")
    st.dataframe(pd.DataFrame({"Jumlah kosong": hilang, "Persentase (%)": (hilang / len(raw) * 100).round(1)}))
    st.subheader("Kolom yang tidak dipakai sebagai fitur")
    buang = pd.DataFrame([(k, KAMUS[k][0], KAMUS[k][1]) for k in BUANG],
                         columns=["Kolom", "Kelompok", "Keterangan"])
    st.dataframe(buang, hide_index=True)
    st.warning("Kolom **Cabin** kosong sekitar 77%, sehingga tidak dapat diandalkan. "
               "**PassengerId, Name, Ticket** adalah pengenal atau teks tanpa pola prediktif yang jelas.")
