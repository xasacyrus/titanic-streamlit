"""
Melatih 5 model + menghitung hasil PCA untuk website Prediksi Keselamatan Penumpang Titanic.
Dataset : data/titanic.csv (Kaggle - Titanic: Machine Learning from Disaster)
Target  : Survived (1 = selamat, 0 = tidak selamat)

Jalankan:  python train_all.py
Output  :  artifacts/models/*.joblib  dan  artifacts/hasil.json
"""
import json, time, warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix, f1_score,
                             precision_recall_curve, precision_score, recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score, cross_validate, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")
RS = 42
ROOT = Path(__file__).resolve().parent
OUT = ROOT / "artifacts"
(OUT / "models").mkdir(parents=True, exist_ok=True)

# ----------------------------------------------------------------------------- data
df = pd.read_csv(ROOT / "data" / "titanic.csv")
y = df["Survived"]
# PassengerId, Name, Ticket: pengenal/teks tanpa pola prediktif. Cabin: ~77% kosong.
DROP = ["Survived", "PassengerId", "Name", "Ticket", "Cabin"]
X = df.drop(columns=DROP)
NUM = ["Age", "Fare", "SibSp", "Parch"]
CAT = ["Pclass", "Sex", "Embarked"]
X = X[NUM + CAT]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=RS)


def make_prep():
    return ColumnTransformer([
        ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), NUM),
        ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                          ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), CAT)])


def metrik(yt, pred, proba):
    return {"accuracy": accuracy_score(yt, pred), "precision": precision_score(yt, pred),
            "recall": recall_score(yt, pred), "f1": f1_score(yt, pred),
            "roc_auc": roc_auc_score(yt, proba), "pr_auc": average_precision_score(yt, proba)}


def r(d, n=4):
    return {k: round(float(v), n) for k, v in d.items()}


skf = StratifiedKFold(5, shuffle=True, random_state=RS)
hasil = {"split": {"n_train": int(len(X_train)), "n_test": int(len(X_test)),
                   "rate_train": round(float(y_train.mean()), 4), "rate_test": round(float(y_test.mean()), 4)},
         "features": {"num": NUM, "cat": CAT}, "models": {}, "baseline_tuning": {}}


# ----------------------------------------------------------------------------- 5 model
def pipe_of(clf):
    return Pipeline([("prep", make_prep()), ("model", clf)])


spec = {
    "Logistic Regression": ("lr", pipe_of(LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RS)), None),
    "KNN": ("knn", pipe_of(KNeighborsClassifier(n_neighbors=15)), None),
    "Decision Tree": ("dt", pipe_of(DecisionTreeClassifier(max_depth=5, class_weight="balanced", random_state=RS)), None),
    "Random Forest": ("rf", pipe_of(RandomForestClassifier(class_weight="balanced", random_state=RS, n_jobs=-1)),
                      {"model__n_estimators": [200, 400], "model__max_depth": [4, 6, None], "model__min_samples_leaf": [1, 5, 10]}),
    "Gradient Boosting": ("gb", pipe_of(GradientBoostingClassifier(random_state=RS)),
                          {"model__n_estimators": [100, 200], "model__learning_rate": [0.05, 0.1], "model__max_depth": [2, 3]}),
}
scoring = {"accuracy": "accuracy", "precision": "precision", "recall": "recall", "f1": "f1", "roc_auc": "roc_auc"}

for nama, (kode, pipe, grid) in spec.items():
    t0 = time.time()
    entry = {"kode": kode, "tuned": grid is not None}
    if grid is not None:
        # baseline tanpa tuning untuk perbandingan
        base = (RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=RS, n_jobs=-1)
                if kode == "rf" else GradientBoostingClassifier(random_state=RS))
        b = pipe_of(base).fit(X_train, y_train)
        hasil["baseline_tuning"][nama] = r(metrik(y_test, b.predict(X_test), b.predict_proba(X_test)[:, 1]))
        gs = GridSearchCV(pipe, grid, cv=skf, scoring="f1", n_jobs=1).fit(X_train, y_train)
        best = gs.best_estimator_
        entry["best_params"] = {k.replace("model__", ""): v for k, v in gs.best_params_.items()}
        entry["best_cv_f1"] = round(float(gs.best_score_), 4)
    else:
        best = pipe.fit(X_train, y_train)
    pred, proba = best.predict(X_test), best.predict_proba(X_test)[:, 1]
    entry["metrics"] = r(metrik(y_test, pred, proba))
    tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()
    entry["confusion"] = {"TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp)}
    fpr, tpr, _ = roc_curve(y_test, proba)
    pr, rc, _ = precision_recall_curve(y_test, proba)
    ix = np.linspace(0, len(fpr) - 1, min(150, len(fpr))).astype(int)
    ip = np.linspace(0, len(pr) - 1, min(150, len(pr))).astype(int)
    entry["roc"] = {"fpr": np.round(fpr[ix], 4).tolist(), "tpr": np.round(tpr[ix], 4).tolist()}
    entry["pr"] = {"precision": np.round(pr[ip], 4).tolist(), "recall": np.round(rc[ip], 4).tolist()}
    entry["threshold"] = []
    for th in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]:
        p = (proba >= th).astype(int)
        entry["threshold"].append({"threshold": th,
                                   "precision": round(float(precision_score(y_test, p, zero_division=0)), 4),
                                   "recall": round(float(recall_score(y_test, p)), 4),
                                   "f1": round(float(f1_score(y_test, p)), 4)})
    cvs = cross_validate(best, X_train, y_train, cv=skf, scoring=scoring, n_jobs=1)
    entry["cv"] = {k: [round(float(cvs["test_" + k].mean()), 4), round(float(cvs["test_" + k].std()), 4)] for k in scoring}
    # pentingnya fitur
    names = [n.split("__")[-1] for n in best.named_steps["prep"].get_feature_names_out()]
    mdl = best.named_steps["model"]
    imp = None
    if hasattr(mdl, "feature_importances_"):
        imp = pd.Series(mdl.feature_importances_, index=names)
    elif hasattr(mdl, "coef_"):
        imp = pd.Series(np.abs(mdl.coef_[0]), index=names)
    if imp is not None:
        entry["importance"] = {k: round(float(v), 4) for k, v in imp.sort_values(ascending=False).head(15).items()}
        entry["importance_type"] = "feature_importances_" if hasattr(mdl, "feature_importances_") else "|koefisien|"
    joblib.dump(best, OUT / "models" / f"{kode}.joblib", compress=3)
    hasil["models"][nama] = entry
    print(f"{nama}: F1={entry['metrics']['f1']} AUC={entry['metrics']['roc_auc']} ({time.time() - t0:.0f}s)", flush=True)

hasil["best_model"] = max(hasil["models"], key=lambda n: hasil["models"][n]["metrics"]["f1"])

# ----------------------------------------------------------------------------- PCA
prep = make_prep()
Xtr_p = prep.fit_transform(X_train)
feat = prep.get_feature_names_out()
pca_full = PCA(random_state=RS).fit(Xtr_p)
ind = pca_full.explained_variance_ratio_
cum = np.cumsum(ind)
k90, k95 = int(np.argmax(cum >= .90) + 1), int(np.argmax(cum >= .95) + 1)
load = pd.DataFrame(pca_full.components_[:3].T, index=[f.split("__")[-1] for f in feat], columns=["PC1", "PC2", "PC3"])
p2 = PCA(n_components=2, random_state=RS).fit_transform(Xtr_p)
rs = np.random.RandomState(RS).choice(len(p2), min(1500, len(p2)), replace=False)

skema = {"Tanpa PCA": None, "PCA 90%": 0.90, "PCA 95%": 0.95, "PCA 2 komponen": 2}
pm = {"Logistic Regression": LogisticRegression(max_iter=2000, class_weight="balanced"),
      "KNN (k=15)": KNeighborsClassifier(n_neighbors=15, n_jobs=-1),
      "SVM (RBF)": SVC(kernel="rbf", class_weight="balanced", random_state=RS)}
bench = []
for mn, mdl in pm.items():
    for sn, k in skema.items():
        steps = [("prep", make_prep())] + ([("pca", PCA(n_components=k, random_state=RS))] if k is not None else []) + [("model", mdl)]
        pipe = Pipeline(steps)
        t0 = time.perf_counter(); pipe.fit(X_train, y_train); ttr = time.perf_counter() - t0
        t0 = time.perf_counter(); pred = pipe.predict(X_test); tin = time.perf_counter() - t0
        score = pipe.decision_function(X_test) if hasattr(pipe, "decision_function") else pipe.predict_proba(X_test)[:, 1]
        bench.append({"Model": mn, "Praproses": sn, "Kolom": int(pipe[:-1].transform(X_test.head(5)).shape[1]),
                      "Train (s)": round(ttr, 3), "Infer (s)": round(tin, 3),
                      "Akurasi": round(accuracy_score(y_test, pred), 4), "F1 makro": round(f1_score(y_test, pred, average="macro"), 4),
                      "Recall selamat": round(recall_score(y_test, pred), 4), "AUC uji": round(roc_auc_score(y_test, score), 4),
                      "AUC CV5": round(float(cross_val_score(pipe, X_train, y_train, cv=skf, scoring="roc_auc", n_jobs=1).mean()), 4)})
    print("PCA benchmark:", mn, flush=True)
ks = list(range(1, len(feat) + 1)); aucs = []
for k in ks:
    p = Pipeline([("prep", make_prep()), ("pca", PCA(n_components=k, random_state=RS)),
                  ("model", LogisticRegression(max_iter=2000, class_weight="balanced"))])
    aucs.append(round(float(cross_val_score(p, X_train, y_train, cv=skf, scoring="roc_auc", n_jobs=1).mean()), 4))
base_auc = [b["AUC CV5"] for b in bench if b["Model"] == "Logistic Regression" and b["Praproses"] == "Tanpa PCA"][0]
hasil["pca"] = {"n_features": int(Xtr_p.shape[1]), "k90": k90, "k95": k95,
                "explained": np.round(ind, 5).tolist(), "cumulative": np.round(cum, 5).tolist(),
                "loadings": {pc: {k: round(float(v), 3) for k, v in load[pc].reindex(load[pc].abs().sort_values(ascending=False).index).head(6).items()} for pc in load.columns},
                "proj2d": {"x": np.round(p2[rs, 0], 3).tolist(), "y": np.round(p2[rs, 1], 3).tolist(), "label": y_train.values[rs].tolist()},
                "benchmark": bench, "k_curve": {"k": ks, "auc": aucs, "base_auc": base_auc}}

(OUT / "hasil.json").write_text(json.dumps(hasil, indent=1, ensure_ascii=False), encoding="utf-8")
print("SELESAI. Model terbaik (F1):", hasil["best_model"], "| PCA k90/k95:", k90, k95)
