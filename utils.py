"""Fungsi bersama untuk semua halaman: pemuatan data/model, tema, dan data referensi."""
import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
ART = ROOT / "artifacts"

PEMBUAT = [
    ("Fachri Siddiq Faizal", "241011402609"),
    ("Mohamad Adi Putra", "241011400570"),
    ("Muhammad Kevin Akbar", "241011400566"),
]
DOSEN = "Farizi Ilham, M.Kom."
KAMPUS = "Program Studi Informatika, Fakultas Ilmu Komputer, Universitas Pamulang"

TARGET = "Survived"
MODEL_NAMES = ["Logistic Regression", "KNN", "Decision Tree", "Random Forest", "Gradient Boosting"]
PELABUHAN = {"S": "Southampton", "C": "Cherbourg", "Q": "Queenstown"}

KAMUS = {
    "PassengerId": ("Identitas", "ID unik penumpang (tidak dipakai)"),
    "Survived": ("Target", "TARGET: 1 = selamat, 0 = tidak selamat"),
    "Pclass": ("Tiket", "Kelas tiket: 1 (atas), 2 (menengah), 3 (ekonomi)"),
    "Name": ("Identitas", "Nama lengkap beserta gelar (tidak dipakai)"),
    "Sex": ("Demografi", "Jenis kelamin (male / female)"),
    "Age": ("Demografi", "Umur dalam tahun (sebagian data kosong)"),
    "SibSp": ("Keluarga", "Jumlah saudara kandung / pasangan di kapal"),
    "Parch": ("Keluarga", "Jumlah orang tua / anak di kapal"),
    "Ticket": ("Identitas", "Nomor tiket dalam teks (tidak dipakai)"),
    "Fare": ("Tiket", "Harga tiket penumpang"),
    "Cabin": ("Lokasi", "Nomor kabin (hampir seluruhnya kosong, tidak dipakai)"),
    "Embarked": ("Perjalanan", "Pelabuhan naik: S = Southampton, C = Cherbourg, Q = Queenstown"),
}
BUANG = ["PassengerId", "Name", "Ticket", "Cabin"]

BIRU, BIRU_MUDA, BIRU_TUA, KORAL, ABU = "#1F4E79", "#A8CCE4", "#163A5C", "#B5473A", "#7A8A99"

CSS = """
<style>
.block-container {padding-top: 1.6rem; padding-bottom: 2rem; max-width: 1280px;}
h1 {color: #0F172A; font-weight: 800; letter-spacing: -0.5px;}
h2, h3 {color: #1E293B; font-weight: 700;}

/* Kartu dashboard */
div[data-testid="stVerticalBlockBorderWrapper"] {
    background: #FFFFFF; border-radius: 14px !important;
    box-shadow: 0 1px 3px rgba(15,23,42,.06); border-color: #E2E8F0 !important;
}

/* Metric di dalam kartu */
div[data-testid="stMetric"] {background: transparent; border: none; padding: 0; box-shadow: none;}
div[data-testid="stMetricLabel"] p {font-size: .8rem; color: #64748B; font-weight: 600;
                                    text-transform: uppercase; letter-spacing: .6px;}
div[data-testid="stMetricValue"] {color: #0F172A; font-weight: 800;}

/* Sidebar */
section[data-testid="stSidebar"] {background: #FFFFFF; border-right: 1px solid #E2E8F0;}
section[data-testid="stSidebar"] h3 {color: #2563EB;}

/* Tab */
.stTabs [data-baseweb="tab-list"] {gap: 4px; border-bottom: 1px solid #E2E8F0;}
.stTabs [data-baseweb="tab"] {padding: 8px 14px; color: #64748B; font-weight: 600;}
.stTabs [aria-selected="true"] {color: #2563EB !important;}
.stTabs [data-baseweb="tab-highlight"] {background: #2563EB !important;}

/* Tombol */
button[kind="primary"] {border-radius: 8px; font-weight: 700;}

/* Badge hasil */
.badge {display: inline-block; padding: 6px 14px; border-radius: 999px; font-weight: 700; font-size: .9rem;}
.badge-ok {background: #DCFCE7; color: #166534;}
.badge-no {background: #FEE2E2; color: #991B1B;}
.judul-kartu {font-weight: 700; color: #0F172A; font-size: 1.02rem; margin-bottom: 2px;}
.sub-kartu {color: #64748B; font-size: .85rem;}
</style>
"""


def pasang_tema():
    st.markdown(CSS, unsafe_allow_html=True)


@st.cache_data
def load_data() -> pd.DataFrame:
    return pd.read_csv(ROOT / "data" / "titanic.csv")


@st.cache_data
def load_raw() -> pd.DataFrame:
    return pd.read_csv(ROOT / "data" / "titanic.csv")


@st.cache_data
def load_hasil() -> dict:
    f = ART / "hasil.json"
    if not f.exists():
        st.error("File artifacts/hasil.json belum ada. Jalankan dulu:  python train_all.py")
        st.stop()
    return json.loads(f.read_text(encoding="utf-8"))


@st.cache_resource
def load_model(kode: str):
    return joblib.load(ART / "models" / f"{kode}.joblib")


def muat_model_aman(nama: str, hasil: dict):
    """Memuat model; jika gagal (versi scikit-learn berbeda), tampilkan petunjuk dan hentikan halaman."""
    try:
        return load_model(hasil["models"][nama]["kode"])
    except Exception as e:
        st.error("Model gagal dimuat (kemungkinan versi scikit-learn berbeda). Jalankan:  python train_all.py")
        st.caption(f"Detail: {e}")
        st.stop()


def daftar_fitur(hasil: dict) -> list:
    return hasil["features"]["num"] + hasil["features"]["cat"]


def penjelasan(profil: dict, proba: float) -> list:
    """Penjelasan singkat faktor yang memengaruhi peluang selamat, berdasarkan pengetahuan umum kejadian Titanic."""
    catatan = []
    if profil["Sex"] == "female":
        catatan.append("Penumpang perempuan mendapat prioritas naik sekoci, sehingga peluang selamatnya lebih tinggi.")
    else:
        catatan.append("Penumpang laki-laki cenderung memiliki peluang selamat lebih rendah pada kejadian ini.")
    if profil["Pclass"] == 1:
        catatan.append("Kelas 1 berada di dek yang lebih dekat ke sekoci, dan tingkat selamatnya paling tinggi.")
    elif profil["Pclass"] == 3:
        catatan.append("Kelas 3 berada di dek bawah yang lebih jauh dari sekoci, dan tingkat selamatnya paling rendah.")
    if profil["Age"] < 12:
        catatan.append("Anak-anak mendapat prioritas penyelamatan, sehingga peluangnya lebih besar.")
    if profil["SibSp"] + profil["Parch"] > 3:
        catatan.append("Keluarga besar lebih sulit diselamatkan bersama, sehingga peluangnya cenderung turun.")
    if not catatan[1:]:
        catatan.append("Faktor lain seperti harga tiket dan pelabuhan naik ikut memengaruhi hasil model.")
    return catatan
