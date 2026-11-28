"""
Dashboard Prediksi Keselamatan Penumpang Titanic (Streamlit).
Jalankan:  streamlit run app.py      lalu buka  http://localhost:8501
"""
import streamlit as st

st.set_page_config(page_title="Titanic Dashboard", page_icon=":ship:", layout="wide")

from utils import pasang_tema  

pasang_tema()

with st.sidebar:
    st.markdown("### 🚢 Titanic Dashboard")
    st.caption("Tugas Machine Learning - Klasifikasi Keselamatan Penumpang")
    st.divider()

halaman = {
    "Ringkasan": [
        st.Page("halaman/beranda.py", title="Beranda", icon=":material/dashboard:", default=True),
        st.Page("halaman/dataset.py", title="Dataset", icon=":material/database:"),
        st.Page("halaman/eda.py", title="Eksplorasi Data", icon=":material/bar_chart:"),
    ],
    "Model": [
        st.Page("halaman/preprocessing.py", title="Preprocessing", icon=":material/tune:"),
        st.Page("halaman/pca.py", title="Reduksi PCA", icon=":material/scatter_plot:"),
        st.Page("halaman/modelling.py", title="Modelling", icon=":material/model_training:"),
    ],
    "Prediksi": [
        st.Page("halaman/prediksi.py", title="Prediksi Satu Penumpang", icon=":material/person_search:"),
        st.Page("halaman/massal.py", title="Prediksi Massal", icon=":material/upload_file:"),
        st.Page("halaman/kesimpulan.py", title="Kesimpulan", icon=":material/task_alt:"),
    ],
}
nav = st.navigation(halaman)
nav.run()