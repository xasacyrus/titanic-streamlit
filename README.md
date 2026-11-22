# Prediksi Keselamatan Penumpang Titanic (Streamlit)

Website machine learning untuk memprediksi apakah seorang penumpang Titanic **selamat (1)** atau **tidak selamat (0)**.
Dibuat dengan struktur dan tampilan yang sama dengan proyek *Prediksi Churn Telco*, tetapi memakai dataset Titanic.

## Struktur Folder

```
titanic-app/
├── app.py                 # Entry point: navigasi multi-halaman
├── utils.py               # Fungsi bersama (load data/model, tema, kamus kolom)
├── charts.py              # Semua grafik (matplotlib)
├── train_all.py           # Melatih 5 model + PCA, menyimpan hasil
├── requirements.txt
├── data/titanic.csv       # Dataset Titanic (891 penumpang)
├── artifacts/
│   ├── hasil.json         # Metrik, confusion matrix, PCA, dll (dibuat train_all.py)
│   └── models/*.joblib    # Model yang sudah dilatih
├── halaman/               # Satu file per halaman website
└── .streamlit/config.toml # Tema warna
```

## Cara Menjalankan

```bash
pip install -r requirements.txt
python train_all.py          # hanya perlu jika artifacts belum ada atau data berubah
streamlit run app.py
```

Buka `http://localhost:8501` di browser.

## Model

Lima algoritma dibandingkan: Logistic Regression, KNN, Decision Tree, Random Forest, dan Gradient Boosting.
Random Forest dan Gradient Boosting di-tuning dengan GridSearchCV (skor F1).
Model terbaik ditentukan dari skor F1 pada data uji.

## Catatan

- Ganti nama dan NIM anggota serta nama dosen di `utils.py` (variabel `PEMBUAT` dan `DOSEN`).
- Jika muncul error "Model gagal dimuat", jalankan ulang `python train_all.py` dengan versi library yang sama seperti `requirements.txt`.
