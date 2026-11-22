import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from utils import BUANG, KAMUS, load_data, load_hasil, load_raw

df, raw, hasil = load_data(), load_raw(), load_hasil()
NUM, CAT = hasil["features"]["num"], hasil["features"]["cat"]

st.title("Preprocessing")
st.caption("Menyiapkan data agar siap dilatih dan bebas dari kebocoran informasi (data leakage).")


@st.cache_data
def ukuran_setelah_prep():
    Xtr, Xts, ytr, _ = train_test_split(df[NUM + CAT], df["Survived"], test_size=0.2, stratify=df["Survived"], random_state=42)
    p = ColumnTransformer([
        ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), NUM),
        ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                          ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), CAT)]).fit(Xtr)
    return Xtr.shape, p.transform(Xtr).shape, len(Xts)


st.progress(1.0, text="Alur: Cleaning -> Pembuangan kolom -> Split -> Imputasi -> Scaling -> One-Hot Encoding")

with st.expander("1. Data cleaning: data yang hilang", expanded=True):
    hilang = raw[["Age", "Cabin", "Embarked"]].isna().sum()
    c1, c2, c3 = st.columns(3)
    c1.metric("Age kosong", int(hilang["Age"]), help="Diisi median data latih")
    c2.metric("Embarked kosong", int(hilang["Embarked"]), help="Diisi nilai paling sering muncul")
    c3.metric("Cabin kosong", int(hilang["Cabin"]), help="Kolom dibuang")
    st.write("`Age` diisi dengan **median** data latih karena sebarannya miring. `Embarked` yang hanya kosong "
             "2 baris diisi dengan **modus** (nilai paling sering). Imputasi dilakukan di dalam pipeline agar nilai pengisi "
             "hanya dihitung dari data latih.")
    st.dataframe(raw.loc[raw["Embarked"].isna(), ["PassengerId", "Pclass", "Sex", "Age", "Fare", "Embarked"]])

with st.expander("2. Pembuangan kolom yang tidak informatif", expanded=True):
    st.write("Kolom yang dibuang:", ", ".join(f"`{k}`" for k in BUANG))
    st.markdown(
        "- **PassengerId** hanya nomor urut dan tidak memiliki makna prediktif.\n"
        "- **Name** berisi teks unik untuk setiap penumpang. Dibutuhkan proses ekstraksi gelar (Mr, Mrs, Miss) "
        "yang tidak dibahas di sini.\n"
        "- **Ticket** memiliki ratusan format berbeda dan sulit diolah tanpa rekayasa fitur khusus.\n"
        "- **Cabin** kosong sekitar 77%, sehingga bila diisi akan banyak menambahkan nilai buatan.")

with st.expander("3. Train-test split"):
    shape_tr, shape_p, n_test = ukuran_setelah_prep()
    c1, c2, c3 = st.columns(3)
    c1.metric("Data latih", f"{shape_tr[0]:,}")
    c2.metric("Data uji", f"{n_test:,}")
    c3.metric("Proporsi selamat", f"{hasil['split']['rate_train'] * 100:.1f}% / {hasil['split']['rate_test'] * 100:.1f}%")
    st.write("Pembagian 80:20 dengan `stratify` agar proporsi penumpang selamat sama di data latih dan uji. "
             "Split dilakukan **sebelum** transformasi agar tidak ada kebocoran dari scaling atau imputasi.")

with st.expander("4. Pipeline: imputasi, scaling, dan encoding"):
    st.markdown(f"- Fitur numerik ({len(NUM)}): {', '.join(NUM)} -> imputasi median -> `StandardScaler`\n"
                f"- Fitur kategorikal ({len(CAT)}): {', '.join(CAT)} -> imputasi modus -> `OneHotEncoder`\n"
                f"- Hasil: **{shape_tr[1]} kolom** awal menjadi **{shape_p[1]} kolom** numerik")
    st.code('''num_pipe = Pipeline([("imputer", SimpleImputer(strategy="median")),
                     ("scaler", StandardScaler())])
cat_pipe = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                     ("ohe", OneHotEncoder(handle_unknown="ignore"))])
preprocess = ColumnTransformer([("num", num_pipe, num_cols),
                                ("cat", cat_pipe, cat_cols)])''', language="python")

with st.expander("5. Penanganan ketidakseimbangan kelas"):
    rasio = (df["Survived"] == 0).sum() / (df["Survived"] == 1).sum()
    st.write(f"Rasio kelas tidak selamat : selamat = {rasio:.2f} : 1. Penanganan: `stratify` saat split, "
             "`class_weight='balanced'` pada Logistic Regression, Decision Tree, dan Random Forest, serta evaluasi "
             "dengan Recall, F1, ROC-AUC, dan PR-AUC.")
