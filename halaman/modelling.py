import pandas as pd
import streamlit as st
from utils import MODEL_NAMES, load_hasil
import charts

hasil = load_hasil()
M = hasil["models"]
best = hasil["best_model"]
st.title("Modelling & Evaluasi")
st.caption("Lima algoritma klasifikasi dibandingkan pada data uji (20%) dengan metrik yang relevan untuk klasifikasi biner.")

alasan = pd.DataFrame({
    "Algoritma": MODEL_NAMES,
    "Alasan dipilih": ["Baseline klasifikasi biner, mudah diinterpretasi", "Model berbasis jarak (non-parametrik)",
                       "Aturan keputusan yang mudah dijelaskan", "Ensemble bagging, stabil dan tahan overfitting (di-tuning)",
                       "Ensemble boosting, umumnya akurat untuk data tabular (di-tuning)"]})
with st.expander("Lima algoritma yang dipakai"):
    st.dataframe(alasan, hide_index=True)

tabel = pd.DataFrame({n: M[n]["metrics"] for n in MODEL_NAMES}).T
tabel.columns = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC", "PR-AUC"]
tabel = tabel.sort_values("F1-Score", ascending=False)

st.success(f"Model terbaik berdasarkan F1: **{best}** (F1 {M[best]['metrics']['f1']:.3f}, Recall {M[best]['metrics']['recall']:.3f}, "
           f"ROC-AUC {M[best]['metrics']['roc_auc']:.3f}).")

t1, t2, t3, t4, t5, t6 = st.tabs(["Perbandingan", "Confusion Matrix", "Kurva ROC & PR", "Cross-validation",
                                  "Tuning & Threshold", "Fitur Penting"])
with t1:
    st.dataframe(tabel.style.format("{:.4f}").highlight_max(axis=0, color="#CFE0FB"))
    charts.tampil(charts.metrik_bar(tabel))
    st.caption("Kelas tidak seimbang (sekitar 38% selamat dan 62% tidak selamat), jadi Accuracy dibaca bersama F1 dan ROC-AUC.")

with t2:
    pilih = st.selectbox("Pilih model", MODEL_NAMES, index=MODEL_NAMES.index(best), key="cm")
    a, b = st.columns([1, 1])
    with a:
        charts.tampil(charts.confusion(M[pilih]["confusion"], pilih))
    with b:
        cm = M[pilih]["confusion"]
        st.metric("TP - selamat terdeteksi benar", cm["TP"])
        st.metric("FN - penumpang selamat yang terlewat", cm["FN"], help="Kesalahan yang paling merugikan")
        st.metric("FP - tidak selamat tetapi diprediksi selamat", cm["FP"])
        st.metric("TN - tidak selamat terdeteksi benar", cm["TN"])

with t3:
    dipilih = st.multiselect("Model yang ditampilkan", MODEL_NAMES, default=MODEL_NAMES)
    if dipilih:
        charts.tampil(charts.roc_pr({n: M[n] for n in dipilih}))
    else:
        st.info("Pilih minimal satu model.")

with t4:
    st.write("Stratified 5-Fold pada data latih, menampilkan rata-rata dan simpangan baku.")
    rows = []
    for n in MODEL_NAMES:
        rows.append({"Model": n, **{k.capitalize().replace("_auc", "-AUC"): f"{v[0]:.4f} ± {v[1]:.4f}" for k, v in M[n]["cv"].items()}})
    st.dataframe(pd.DataFrame(rows).set_index("Model"))

with t5:
    st.subheader("Hyperparameter tuning (GridSearchCV, skor F1)")
    rows = []
    for n in ["Random Forest", "Gradient Boosting"]:
        bt = hasil["baseline_tuning"][n]
        rows.append({"Model": n, "F1 sebelum tuning": bt["f1"], "F1 sesudah tuning": M[n]["metrics"]["f1"],
                     "AUC sebelum": bt["roc_auc"], "AUC sesudah": M[n]["metrics"]["roc_auc"],
                     "Parameter terbaik": str(M[n]["best_params"])})
    st.dataframe(pd.DataFrame(rows).set_index("Model"))
    st.subheader("Threshold tuning")
    pilih = st.selectbox("Model", MODEL_NAMES, index=MODEL_NAMES.index(best), key="th")
    a, b = st.columns(2)
    with a:
        charts.tampil(charts.threshold_plot(M[pilih]["threshold"]))
    with b:
        st.dataframe(pd.DataFrame(M[pilih]["threshold"]).set_index("threshold"))
        st.caption("Menurunkan threshold menaikkan Recall tetapi menurunkan Precision. Pilihannya bergantung pada "
                   "seberapa besar kerugian melewatkan penumpang yang selamat dibanding salah menganggapnya selamat.")

with t6:
    pilih = st.selectbox("Model", [n for n in MODEL_NAMES if "importance" in M[n]], index=0, key="fi")
    charts.tampil(charts.importance(M[pilih]["importance"], f"15 fitur terpenting - {pilih} ({M[pilih]['importance_type']})"))
