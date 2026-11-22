import pandas as pd
import streamlit as st
from utils import MODEL_NAMES, load_data, load_hasil

df, hasil = load_data(), load_hasil()
M, P = hasil["models"], hasil["pca"]
best = hasil["best_model"]
mb = M[best]["metrics"]
top = list(M["Random Forest"]["importance"])[:5]
g = lambda c, v: df.loc[df[c] == v, "Survived"].mean() * 100

st.title("Kesimpulan & Saran")

st.subheader("Kesimpulan")
st.markdown(
    f"1. **Model terbaik** berdasarkan F1 adalah **{best}** dengan F1 {mb['f1']:.3f}, Recall {mb['recall']:.3f}, "
    f"Precision {mb['precision']:.3f}, dan ROC-AUC {mb['roc_auc']:.3f}.\n"
    f"2. Akurasi ({mb['accuracy']:.3f}) perlu dibaca bersama Recall dan F1, karena kelas tidak seimbang. "
    f"Recall menunjukkan model menemukan {mb['recall'] * 100:.0f}% penumpang yang benar-benar selamat.\n"
    f"3. Setelah dituning, Random Forest {'naik' if M['Random Forest']['metrics']['f1'] >= hasil['baseline_tuning']['Random Forest']['f1'] else 'berubah'} "
    f"F1-nya dari {hasil['baseline_tuning']['Random Forest']['f1']:.3f} menjadi {M['Random Forest']['metrics']['f1']:.3f}.\n"
    f"4. **PCA** memadatkan {P['n_features']} fitur menjadi {P['k90']}-{P['k95']} komponen (90-95% varians). "
    "Ini mengurangi dimensi, tetapi model prediksi tetap memakai fitur asli agar mudah dijelaskan.\n"
    f"5. Fitur paling berpengaruh (Random Forest): {', '.join(f'`{t}`' for t in top)}.\n"
    "6. Menurunkan *threshold* meningkatkan Recall tetapi menurunkan Precision, sehingga pilihannya bergantung pada "
    "seberapa besar kerugian melewatkan penumpang yang selamat.")

st.subheader("Perbandingan ringkas lima model")
tb = pd.DataFrame({n: M[n]["metrics"] for n in MODEL_NAMES}).T[["recall", "f1", "roc_auc"]]
tb.columns = ["Recall", "F1-Score", "ROC-AUC"]
st.dataframe(tb.sort_values("F1-Score", ascending=False).style.format("{:.3f}"))

st.subheader("Temuan dari data")
st.markdown(
    f"- **Jenis kelamin:** perempuan selamat {g('Sex', 'female'):.0f}% dibanding laki-laki {g('Sex', 'male'):.0f}%.\n"
    f"- **Kelas tiket:** kelas 1 selamat {g('Pclass', 1):.0f}%, kelas 2 {g('Pclass', 2):.0f}%, kelas 3 {g('Pclass', 3):.0f}%.\n"
    f"- **Pelabuhan Cherbourg (C)** memiliki tingkat selamat {g('Embarked', 'C'):.0f}%, sedangkan Southampton (S) {g('Embarked', 'S'):.0f}%.\n"
    "- Faktor sosial seperti jenis kelamin dan kelas tiket tampak lebih kuat dibanding faktor lain dalam data ini.")

st.subheader("Saran pengembangan")
st.markdown(
    "- Buat **fitur baru** dari Name, misalnya gelar (Mr, Mrs, Miss, Master), karena gelar sering mencerminkan jenis kelamin dan status.\n"
    "- Buat fitur **ukuran keluarga** (SibSp + Parch + 1) dan **sendirian** untuk menangkap pengaruh keluarga.\n"
    "- Gunakan **Cabin** setelah diambil huruf dek-nya (A-G) dan kosong dianggap kategori tersendiri.\n"
    "- Uji beberapa **random state** dan validasi silang yang lebih banyak agar hasil lebih stabil.")

st.subheader("Keterbatasan")
st.markdown(
    "- Data hanya mencakup satu kejadian (Titanic, 1912), sehingga belum tentu berlaku untuk situasi lain.\n"
    "- Model memprediksi **kemungkinan** berdasarkan pola data, bukan sebab-akibat langsung.\n"
    "- Jumlah data relatif kecil (891 baris), sehingga hasil dapat berubah cukup banyak jika data bertambah.")
