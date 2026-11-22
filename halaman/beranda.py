import plotly.graph_objects as go
import streamlit as st
from utils import DOSEN, KAMPUS, MODEL_NAMES, PEMBUAT, load_data, load_hasil

df, hasil = load_data(), load_hasil()
best = hasil["best_model"]
mb = hasil["models"][best]["metrics"]

st.title("Dashboard Keselamatan Titanic")
st.caption("Ringkasan data dan performa model prediksi")

# ---------- Baris 1: kartu KPI ----------
k1, k2, k3, k4 = st.columns(4)
with k1.container(border=True):
    st.metric("Penumpang", f"{len(df):,}")
    st.caption("catatan dalam dataset")
with k2.container(border=True):
    st.metric("Tingkat selamat", f"{df['Survived'].mean() * 100:.1f}%")
    st.caption(f"{int(df['Survived'].sum())} orang selamat")
with k3.container(border=True):
    st.metric("Model terbaik", best)
    st.caption(f"F1 {mb['f1']:.3f}")
with k4.container(border=True):
    st.metric("ROC-AUC", f"{mb['roc_auc']:.3f}")
    st.caption("model terbaik")

# ---------- Baris 2: grafik ----------
g1, g2 = st.columns([3, 2])

with g1.container(border=True):
    st.markdown('<div class="judul-kartu">Tingkat selamat per jenis kelamin dan kelas</div>', unsafe_allow_html=True)
    pv = df.pivot_table(index="Sex", columns="Pclass", values="Survived", aggfunc="mean") * 100
    fig = go.Figure()
    warna = {1: "#2563EB", 2: "#38BDF8", 3: "#CBD5E1"}
    for kelas in pv.columns:
        fig.add_trace(go.Bar(
            name=f"Kelas {kelas}", x=["Perempuan" if s == "female" else "Laki-laki" for s in pv.index],
            y=pv[kelas].round(1), marker_color=warna.get(kelas, "#2563EB"),
            text=pv[kelas].round(1).astype(str) + "%", textposition="outside"))
    fig.update_layout(barmode="group", height=330, margin=dict(l=10, r=10, t=10, b=10),
                      plot_bgcolor="white", paper_bgcolor="white", yaxis=dict(title="Tingkat selamat (%)", range=[0, 110]),
                      legend=dict(orientation="h", y=1.12, x=0))
    st.plotly_chart(fig, use_container_width=True)

with g2.container(border=True):
    st.markdown('<div class="judul-kartu">Perbandingan F1-Score model</div>', unsafe_allow_html=True)
    skor = sorted([(n, hasil["models"][n]["metrics"]["f1"]) for n in MODEL_NAMES], key=lambda x: x[1])
    fig2 = go.Figure(go.Bar(
        x=[s for _, s in skor], y=[n for n, _ in skor], orientation="h",
        marker_color=["#2563EB" if n == best else "#CBD5E1" for n, _ in skor],
        text=[f"{s:.3f}" for _, s in skor], textposition="outside"))
    fig2.update_layout(height=330, margin=dict(l=10, r=40, t=10, b=10), plot_bgcolor="white",
                       paper_bgcolor="white", xaxis=dict(range=[0, 1], title="F1-Score"))
    st.plotly_chart(fig2, use_container_width=True)

# ---------- Baris 3: langkah proyek ----------
st.write("")
st.subheader("Alur proyek")
langkah = [
    ("01", "Dataset", f"{df.shape[0]} baris dan {df.shape[1]} kolom, klasifikasi biner."),
    ("02", "Preprocessing", "Mengisi data kosong, membuang kolom tidak informatif, dan encoding."),
    ("03", "Reduksi PCA", f"{hasil['pca']['n_features']} fitur dipadatkan menjadi {hasil['pca']['k90']} sampai {hasil['pca']['k95']} komponen."),
    ("04", "Modelling", "Lima algoritma dibandingkan, di-tuning, lalu dipakai untuk prediksi."),
]
kolom = st.columns(4)
for col, (no, judul, isi) in zip(kolom, langkah):
    with col.container(border=True):
        st.caption(f"LANGKAH {no}")
        st.markdown(f'<div class="judul-kartu">{judul}</div><div class="sub-kartu">{isi}</div>', unsafe_allow_html=True)

# ---------- Baris 4: tim ----------
st.write("")
st.subheader("Tim kelompok")
kolom = st.columns(3)
for col, (nama, nim) in zip(kolom, PEMBUAT):
    with col.container(border=True):
        st.markdown(f'<div class="judul-kartu">{nama}</div><div class="sub-kartu">{nim}</div>', unsafe_allow_html=True)
st.caption(f"Dosen pengampu: {DOSEN} · {KAMPUS}")