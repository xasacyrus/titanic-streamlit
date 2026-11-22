import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from utils import MODEL_NAMES, PELABUHAN, daftar_fitur, load_hasil, muat_model_aman, penjelasan

hasil = load_hasil()
FITUR = daftar_fitur(hasil)

st.title("Prediksi Satu Penumpang")
st.caption("Isi data penumpang di kiri, lalu lihat hasilnya di kanan.")

CONTOH = {
    "Isi manual": {},
    "Contoh: perempuan kelas 1": {"Pclass": 1, "Sex": "female", "Age": 29, "SibSp": 0, "Parch": 0, "Fare": 120.0, "Embarked": "S"},
    "Contoh: laki-laki dewasa kelas 3": {"Pclass": 3, "Sex": "male", "Age": 32, "SibSp": 0, "Parch": 0, "Fare": 8.0, "Embarked": "S"},
}

kiri, kanan = st.columns([2, 3], gap="large")

with kiri:
    with st.container(border=True):
        st.markdown('<div class="judul-kartu">Pengaturan</div>', unsafe_allow_html=True)
        nama_model = st.selectbox("Model", MODEL_NAMES, index=MODEL_NAMES.index(hasil["best_model"]))
        contoh = st.selectbox("Profil awal", list(CONTOH))
        ambang = st.slider("Ambang keputusan", 0.1, 0.9, 0.5, 0.05,
                           help="Turunkan ambang agar lebih banyak penumpang yang selamat terdeteksi.")

    d = CONTOH[contoh]
    k = contoh.replace(" ", "_").replace(":", "").replace(",", "")

    with st.form("form_prediksi"):
        with st.container(border=True):
            st.markdown('<div class="judul-kartu">Data penumpang</div>', unsafe_allow_html=True)
            c1, c2 = st.columns(2)
            pclass = c1.selectbox("Kelas tiket", [1, 2, 3], index=[1, 2, 3].index(d.get("Pclass", 3)),
                                  key=f"{k}_pclass", format_func=lambda x: f"Kelas {x}")
            sex = c2.radio("Jenis kelamin", ["male", "female"], horizontal=True,
                           index=["male", "female"].index(d.get("Sex", "male")), key=f"{k}_sex",
                           format_func=lambda x: "Laki-laki" if x == "male" else "Perempuan")
            age = st.slider("Umur", 0, 80, int(d.get("Age", 30)), key=f"{k}_age")
        with st.container(border=True):
            st.markdown('<div class="judul-kartu">Tiket dan keluarga</div>', unsafe_allow_html=True)
            c3, c4 = st.columns(2)
            sibsp = c3.slider("Saudara / pasangan", 0, 8, int(d.get("SibSp", 0)), key=f"{k}_sibsp")
            parch = c4.slider("Orang tua / anak", 0, 6, int(d.get("Parch", 0)), key=f"{k}_parch")
            fare = c3.number_input("Harga tiket (Fare)", 0.0, 520.0, float(d.get("Fare", 32.0)), 1.0, key=f"{k}_fare")
            embarked = c4.selectbox("Pelabuhan naik", list(PELABUHAN), index=list(PELABUHAN).index(d.get("Embarked", "S")),
                                    key=f"{k}_embarked", format_func=lambda x: f"{PELABUHAN[x]} ({x})")
        kirim = st.form_submit_button("Lihat hasil", type="primary", use_container_width=True)

with kanan:
    if not kirim:
        with st.container(border=True):
            st.info("Isi data penumpang lalu tekan **Lihat hasil**.")
    else:
        profil = {"Pclass": pclass, "Sex": sex, "Age": age, "SibSp": sibsp, "Parch": parch, "Fare": fare, "Embarked": embarked}
        baris = pd.DataFrame([profil])[FITUR]
        model = muat_model_aman(nama_model, hasil)
        proba = float(model.predict_proba(baris)[0, 1])
        selamat = proba >= ambang

        if selamat:
            st.balloons()
            st.toast("Penumpang diprediksi selamat", icon="🚢")
        else:
            st.toast("Penumpang diprediksi tidak selamat", icon="⚓")

        with st.container(border=True):
            h1, h2 = st.columns([3, 2])
            with h1:
                st.markdown('<div class="judul-kartu">Keputusan model</div>', unsafe_allow_html=True)
                if selamat:
                    st.markdown('<span class="badge badge-ok">SELAMAT</span>', unsafe_allow_html=True)
                else:
                    st.markdown('<span class="badge badge-no">TIDAK SELAMAT</span>', unsafe_allow_html=True)
                st.caption(f"{nama_model} · ambang {ambang:.2f}")
            with h2:
                st.metric("Peluang selamat", f"{proba * 100:.1f}%")

            gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=proba * 100,
                number={"suffix": "%", "font": {"color": "#0F172A"}},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": "#2563EB"},
                    "steps": [{"range": [0, ambang * 100], "color": "#FEE2E2"},
                              {"range": [ambang * 100, 100], "color": "#DCFCE7"}],
                    "threshold": {"line": {"color": "#0F172A", "width": 3}, "thickness": 0.8, "value": ambang * 100},
                },
            ))
            gauge.update_layout(height=240, margin=dict(l=20, r=20, t=10, b=0), paper_bgcolor="white")
            st.plotly_chart(gauge, use_container_width=True)
            st.caption("Garis hitam adalah ambang keputusan. Di sebelah kanan garis dianggap selamat.")

        with st.container(border=True):
            st.markdown('<div class="judul-kartu">Faktor yang memengaruhi</div>', unsafe_allow_html=True)
            for s in penjelasan(profil, proba):
                st.write("• " + s)

        with st.expander("Bandingkan dengan semua model"):
            rows = []
            for n in MODEL_NAMES:
                pr = float(muat_model_aman(n, hasil).predict_proba(baris)[0, 1])
                rows.append({"Model": n, "Peluang (%)": round(pr * 100, 1),
                             "Prediksi": "Selamat" if pr >= ambang else "Tidak selamat"})
            st.dataframe(pd.DataFrame(rows).set_index("Model"), use_container_width=True)
        with st.expander("Data yang dikirim ke model"):
            st.dataframe(baris.T.rename(columns={0: "Nilai"}).astype(str), use_container_width=True)

        st.caption("Hasil ini adalah estimasi statistik dari data historis, bukan kepastian nasib seorang penumpang.")