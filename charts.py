"""Semua grafik (matplotlib) bertema biru soft. Setiap fungsi mengembalikan Figure."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from utils import ABU, BIRU, BIRU_MUDA, BIRU_TUA, KORAL

BG = "#F4F8FE"
plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": "#FFFFFF", "axes.edgecolor": "#C9D8EE",
    "axes.grid": True, "grid.color": "#E4EDF9", "grid.linewidth": 0.8, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False, "axes.titleweight": "bold",
    "axes.titlecolor": "#2F5DA8", "axes.labelcolor": "#33475E", "xtick.color": "#33475E",
    "ytick.color": "#33475E", "font.size": 10, "legend.frameon": False,
})
SERIES = [BIRU, KORAL, "#6FC3B2", "#F2B880", "#9C8FE0"]
LABEL_KELAS = ["Tidak selamat", "Selamat"]


def tampil(fig):
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def _fig(w=6.5, h=3.6, n=1):
    fig, ax = plt.subplots(1, n, figsize=(w, h))
    return fig, ax


# ----------------------------------------------------------------------------- EDA
def distribusi_kelas(df):
    fig, (a, b) = _fig(8, 3.4, 2)
    c = df["Survived"].value_counts().reindex([0, 1]).fillna(0)
    a.bar(LABEL_KELAS, c.values, color=[BIRU, KORAL])
    for i, v in enumerate(c.values):
        a.text(i, v + 6, f"{int(v):,}", ha="center")
    a.set_title("Jumlah penumpang"); a.set_ylim(0, max(c.max() * 1.15, 1))
    b.pie(c.values, labels=LABEL_KELAS, colors=[BIRU, KORAL], autopct="%1.1f%%",
          startangle=90, wedgeprops={"width": 0.45, "edgecolor": BG}, textprops={"color": "#1E2B3C"})
    b.set_title("Proporsi"); b.grid(False)
    return fig


def rate_kategori(df, col):
    g = df.groupby(col)["Survived"].agg(["mean", "count"]).sort_values("mean", ascending=False)
    fig, ax = _fig(7, 3.6)
    bars = ax.bar(g.index.astype(str), g["mean"] * 100, color=BIRU)
    bars[0].set_color(KORAL)
    base = df["Survived"].mean() * 100
    ax.axhline(base, color=ABU, ls="--", lw=1, label=f"Rata-rata {base:.1f}%")
    for rect, (m, n) in zip(bars, g.values):
        ax.text(rect.get_x() + rect.get_width() / 2, m * 100 + 0.8, f"{m * 100:.1f}%\n(n={int(n):,})", ha="center", fontsize=8)
    ax.set_ylabel("Tingkat selamat (%)"); ax.set_title(f"Tingkat selamat per {col}")
    ax.set_ylim(0, max(g["mean"].max() * 100 * 1.25, base * 1.3, 10)); ax.legend()
    plt.setp(ax.get_xticklabels(), rotation=15 if len(g) > 3 else 0, ha="right" if len(g) > 3 else "center")
    return fig


def hist_numerik(df, col):
    fig, (a, b) = _fig(9, 3.6, 2)
    for lab, c, nm in [(0, BIRU, "Tidak selamat"), (1, KORAL, "Selamat")]:
        a.hist(df.loc[df["Survived"] == lab, col].dropna(), bins=30, alpha=.65, color=c, label=nm)
    a.set_title(f"Distribusi {col}"); a.set_xlabel(col); a.set_ylabel("Jumlah"); a.legend()
    data = [df.loc[df["Survived"] == k, col].dropna() for k in (0, 1)]
    bp = b.boxplot(data, patch_artist=True, medianprops={"color": "#1E2B3C"})
    b.set_xticks([1, 2]); b.set_xticklabels(LABEL_KELAS)
    for patch, c in zip(bp["boxes"], [BIRU_MUDA, "#F2C0C0"]):
        patch.set_facecolor(c)
    b.set_title(f"{col} menurut status keselamatan")
    return fig


def korelasi(df):
    cols = ["Age", "Fare", "SibSp", "Parch", "Survived"]
    c = df[cols].corr()
    n = len(cols)
    fig, ax = _fig(5.4, 4.2)
    im = ax.imshow(c, cmap="Blues", vmin=-1, vmax=1)
    ax.set_xticks(range(n)); ax.set_xticklabels(cols, rotation=25, ha="right"); ax.set_yticks(range(n)); ax.set_yticklabels(cols)
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{c.iloc[i, j]:.2f}", ha="center", va="center", color="white" if abs(c.iloc[i, j]) > .55 else "#1E2B3C")
    ax.grid(False); ax.set_title("Korelasi fitur numerik"); fig.colorbar(im, ax=ax, shrink=.8)
    return fig


# ----------------------------------------------------------------------------- model
def metrik_bar(tabel: pd.DataFrame):
    fig, ax = _fig(10, 4)
    n = len(tabel.columns); w = 0.8 / n; x = np.arange(len(tabel))
    cmap = [BIRU_TUA, BIRU, BIRU_MUDA, "#6FC3B2", "#F2B880", KORAL]
    for i, c in enumerate(tabel.columns):
        ax.bar(x + i * w - 0.4 + w / 2, tabel[c], w, label=c, color=cmap[i % len(cmap)])
    ax.set_xticks(x); ax.set_xticklabels(tabel.index, rotation=12); ax.set_ylim(0, 1.05)
    ax.set_title("Perbandingan metrik pada data uji"); ax.legend(ncol=6, loc="upper center", bbox_to_anchor=(.5, -.18))
    return fig


def confusion(cm: dict, judul: str):
    m = np.array([[cm["TN"], cm["FP"]], [cm["FN"], cm["TP"]]])
    fig, ax = _fig(4.6, 3.8)
    ax.imshow(m, cmap="Blues")
    lab = [["TN", "FP"], ["FN", "TP"]]
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{lab[i][j]}\n{m[i, j]:,}", ha="center", va="center", fontsize=12,
                    color="white" if m[i, j] > m.max() * .55 else "#1E2B3C")
    ax.set_xticks([0, 1]); ax.set_xticklabels(LABEL_KELAS); ax.set_yticks([0, 1]); ax.set_yticklabels(LABEL_KELAS)
    ax.set_xlabel("Prediksi"); ax.set_ylabel("Aktual"); ax.set_title(judul); ax.grid(False)
    return fig


def roc_pr(models: dict):
    fig, (a, b) = _fig(11, 4.4, 2)
    for i, (n, e) in enumerate(models.items()):
        a.plot(e["roc"]["fpr"], e["roc"]["tpr"], color=SERIES[i % 5], lw=2, label=f"{n} (AUC {e['metrics']['roc_auc']:.3f})")
        b.plot(e["pr"]["recall"], e["pr"]["precision"], color=SERIES[i % 5], lw=2, label=f"{n} (AP {e['metrics']['pr_auc']:.3f})")
    a.plot([0, 1], [0, 1], "--", color=ABU, lw=1)
    a.set(title="Kurva ROC", xlabel="False Positive Rate", ylabel="True Positive Rate"); a.legend(fontsize=8, loc="lower right")
    b.set(title="Kurva Precision-Recall", xlabel="Recall", ylabel="Precision"); b.legend(fontsize=8, loc="upper right")
    return fig


def importance(imp: dict, judul: str):
    s = pd.Series(imp).sort_values()
    fig, ax = _fig(7, 4.8)
    ax.barh(s.index, s.values, color=BIRU)
    ax.set_title(judul); ax.set_xlabel("Tingkat kepentingan")
    return fig


def threshold_plot(rows: list):
    d = pd.DataFrame(rows)
    fig, ax = _fig(6.5, 3.6)
    for c, col in [("precision", BIRU), ("recall", KORAL), ("f1", "#6FC3B2")]:
        ax.plot(d["threshold"], d[c], marker="o", color=col, label=c.capitalize())
    ax.set(title="Dampak threshold", xlabel="Threshold", ylabel="Skor"); ax.legend()
    return fig


def proba_hist(proba, ambang):
    fig, ax = _fig(6.5, 3.2)
    ax.hist(proba, bins=25, color=BIRU, alpha=.85)
    ax.axvline(ambang, color=KORAL, ls="--", label=f"Threshold {ambang:.2f}")
    ax.set(title="Sebaran probabilitas selamat", xlabel="Probabilitas selamat", ylabel="Jumlah penumpang"); ax.legend()
    return fig


# ----------------------------------------------------------------------------- PCA
def scree(pca: dict, k_tandai=None):
    ind, cum = np.array(pca["explained"]), np.array(pca["cumulative"])
    idx = np.arange(1, len(ind) + 1)
    fig, ax = _fig(7.5, 3.8)
    ax.bar(idx, ind, color=BIRU_MUDA, label="Varians individual")
    ax.step(idx, cum, where="mid", color=BIRU_TUA, lw=2, label="Kumulatif")
    ax.axhline(.90, color=ABU, ls="--", lw=1); ax.axhline(.95, color=ABU, ls=":", lw=1)
    ax.axvline(pca["k90"], color=KORAL, ls="--", lw=1, label=f"90% = {pca['k90']} komponen")
    ax.axvline(pca["k95"], color=KORAL, ls=":", lw=1.5, label=f"95% = {pca['k95']} komponen")
    if k_tandai:
        ax.scatter([k_tandai], [cum[k_tandai - 1]], color=KORAL, zorder=5, s=60)
    ax.set(title="Scree plot", xlabel="Komponen utama", ylabel="Explained variance ratio"); ax.legend(loc="center right")
    return fig


def scatter2d(pca: dict):
    p = pca["proj2d"]
    x, y, l = np.array(p["x"]), np.array(p["y"]), np.array(p["label"])
    fig, ax = _fig(6.2, 4.2)
    ax.scatter(x[l == 0], y[l == 0], s=10, alpha=.45, color=BIRU, label="Tidak selamat")
    ax.scatter(x[l == 1], y[l == 1], s=10, alpha=.55, color=KORAL, label="Selamat")
    v = sum(pca["explained"][:2])
    ax.set(title=f"Proyeksi 2D (varians {v:.1%})", xlabel="PC1", ylabel="PC2"); ax.legend()
    return fig


def pca_bench(b: pd.DataFrame):
    fig, ax = _fig(14, 3.8, 3)
    for a, (col, jd) in zip(ax, [("AUC CV5", "ROC-AUC (CV 5-fold)"), ("Train (s)", "Waktu latih (detik)"), ("Infer (s)", "Waktu prediksi (detik)")]):
        pv = b.pivot(index="Praproses", columns="Model", values=col).reindex(["Tanpa PCA", "PCA 90%", "PCA 95%", "PCA 2 komponen"])
        x = np.arange(len(pv)); w = 0.8 / pv.shape[1]
        for i, m in enumerate(pv.columns):
            a.bar(x + i * w - .4 + w / 2, pv[m], w, label=m, color=SERIES[i])
        a.set_xticks(x); a.set_xticklabels(pv.index, rotation=15, fontsize=8); a.set_title(jd)
    ax[0].set_ylim(.5, 1.0)
    ax[1].legend(fontsize=8, ncol=3, loc="upper center", bbox_to_anchor=(.5, -.2))
    return fig


def k_curve(pca: dict):
    k = pca["k_curve"]
    fig, ax = _fig(6.8, 3.6)
    ax.plot(k["k"], k["auc"], marker="o", color=BIRU, label="Logistic Regression + PCA(k)")
    ax.axhline(k["base_auc"], color=KORAL, ls="--", label="Tanpa PCA")
    ax.axvline(pca["k95"], color=ABU, ls=":", label=f"k untuk 95% varians = {pca['k95']}")
    ax.set(title="Kinerja vs jumlah komponen", xlabel="Jumlah komponen k", ylabel="ROC-AUC (CV)"); ax.legend(fontsize=8)
    return fig
