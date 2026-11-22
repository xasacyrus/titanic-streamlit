import streamlit as st
from utils import load_data
import charts

df = load_data()
st.title("Eksplorasi Data (EDA)")
st.caption("Pola apa yang membedakan penumpang yang selamat dan yang tidak selamat?")

# filter global di sidebar
with st.sidebar:
    st.header("Filter EDA")
    kelas = st.multiselect("Kelas tiket", sorted(df["Pclass"].unique().tolist()),
                           default=sorted(df["Pclass"].unique().tolist()))
    gender = st.multiselect("Jenis kelamin", ["male", "female"], default=["male", "female"])
    umur = st.slider("Rentang umur", 0, 80, (0, 80))
mask = df["Pclass"].isin(kelas) & df["Sex"].isin(gender) & (df["Age"].between(umur[0], umur[1]) | df["Age"].isna())
dff = df[mask]
if dff.empty:
    st.warning("Tidak ada data untuk filter ini. Ubah filter di sidebar.")
    st.stop()

c1, c2, c3 = st.columns(3)
base = df["Survived"].mean() * 100
rate = dff["Survived"].mean() * 100
c1.metric("Penumpang terfilter", f"{len(dff):,}")
c2.metric("Tingkat selamat terfilter", f"{rate:.1f}%", f"{rate - base:+.1f} poin vs keseluruhan")
c3.metric("Rata-rata harga tiket", f"{dff['Fare'].mean():.2f}")

t1, t2, t3, t4, t5 = st.tabs(["Distribusi Keselamatan", "Fitur Kategorikal", "Fitur Numerik", "Korelasi", "Kombinasi Jenis Kelamin & Kelas"])
with t1:
    charts.tampil(charts.distribusi_kelas(dff))
    st.info("Kelas tidak seimbang: model yang menebak semua penumpang 'tidak selamat' sudah akurat sekitar "
            f"{(1 - df['Survived'].mean()) * 100:.1f}%, tetapi tidak pernah menemukan penumpang yang selamat. "
            "Karena itu yang dipantau adalah Recall, F1, dan ROC-AUC.")
with t2:
    kat = ["Sex", "Pclass", "Embarked", "SibSp", "Parch"]
    pilih = st.selectbox("Pilih fitur", kat)
    charts.tampil(charts.rate_kategori(dff, pilih))
    with st.expander("Lihat tabel angka"):
        tb = dff.groupby(pilih)["Survived"].agg(Jumlah="count", Selamat="sum", Tingkat_selamat="mean")
        tb["Tingkat_selamat"] = (tb["Tingkat_selamat"] * 100).round(1)
        st.dataframe(tb.sort_values("Tingkat_selamat", ascending=False))
with t3:
    pilih = st.radio("Pilih fitur numerik", ["Age", "Fare", "SibSp", "Parch"], horizontal=True)
    charts.tampil(charts.hist_numerik(dff, pilih))
    st.caption("Perhatikan perbedaan sebaran Fare antara kedua kelompok: penumpang yang selamat umumnya membayar tiket lebih mahal.")
with t4:
    kiri, kanan = st.columns([1, 1])
    with kiri:
        charts.tampil(charts.korelasi(dff))
    with kanan:
        st.markdown("**Cara membaca:** nilai mendekati 1 atau -1 berarti hubungan kuat, dan nilai mendekati 0 berarti lemah.\n\n"
                    "- `Fare` biasanya berkorelasi positif dengan `Survived`, karena penumpang kelas atas membayar lebih mahal.\n"
                    "- `SibSp` dan `Parch` saling berkorelasi karena keduanya mengukur ukuran keluarga.")
with t5:
    pv = dff.pivot_table(index="Sex", columns="Pclass", values="Survived", aggfunc="mean") * 100
    st.write("Tingkat selamat (%) untuk setiap kombinasi jenis kelamin dan kelas tiket:")
    st.dataframe(pv.style.format("{:.1f}%").background_gradient(cmap="Blues"))
    st.caption("Kombinasi perempuan di kelas 1 dan 2 umumnya memiliki tingkat selamat tertinggi.")

with st.expander("Ringkasan temuan utama"):
    g = lambda col, val: df.loc[df[col] == val, "Survived"].mean() * 100
    st.markdown(
        f"- Jenis kelamin **perempuan**: tingkat selamat {g('Sex', 'female'):.1f}% vs laki-laki {g('Sex', 'male'):.1f}%.\n"
        f"- Kelas **1**: {g('Pclass', 1):.1f}% vs kelas **3**: {g('Pclass', 3):.1f}%.\n"
        f"- Pelabuhan **Cherbourg (C)**: {g('Embarked', 'C'):.1f}% vs **Southampton (S)**: {g('Embarked', 'S'):.1f}%.")
