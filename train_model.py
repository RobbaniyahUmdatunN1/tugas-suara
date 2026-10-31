"""
Training + TUNING parameter Random Forest (berisik vs non-berisik).

Output:
    fitur_suara.csv    : tabel fitur semua file
    hasil_tuning.csv   : semua kombinasi parameter + skor CV (untuk laporan)
    model_rf.joblib    : model terbaik + info (dipakai app.py)
"""

import os
import sys
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import (
    train_test_split, StratifiedKFold, GridSearchCV, cross_val_score,
)
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from fitur import ekstrak_fitur, NAMA_FITUR

BASE = os.path.dirname(os.path.abspath(__file__))
KELAS = ["berisik", "non-berisik"]
EKSTENSI = (".wav", ".mp3", ".flac", ".ogg", ".m4a")
PATH_CSV = os.path.join(BASE, "fitur_suara.csv")

# Ruang parameter yang dicoba. Ubah sesuai kebutuhan.
PARAM_GRID = {
    "n_estimators": [50, 100, 200],        # jumlah pohon
    "max_depth": [None, 5, 10],            # kedalaman maksimum pohon
    "min_samples_split": [2, 5],           # minimal sampel untuk memecah node
    "min_samples_leaf": [1, 2],            # minimal sampel di daun
    "max_features": ["sqrt", "log2"],      # jumlah fitur yang dicoba tiap split
}


def ekstrak_dari_folder():
    baris = []
    for kelas in KELAS:
        folder = os.path.join(BASE, kelas)
        if not os.path.isdir(folder):
            raise SystemExit(f"Folder '{kelas}' tidak ditemukan di {BASE}")
        for root, _, files in os.walk(folder):
            for f in sorted(files):
                if not f.lower().endswith(EKSTENSI):
                    continue
                path = os.path.join(root, f)
                try:
                    fitur = ekstrak_fitur(path)
                except Exception as e:
                    print(f"[GAGAL] {path}: {e}")
                    continue
                baris.append([f, *fitur, kelas])
                print(f"[OK] {kelas}/{f}")
    df = pd.DataFrame(baris, columns=["file", *NAMA_FITUR, "label"])
    df.to_csv(PATH_CSV, index=False)
    return df


def main():
    if "--ekstrak-ulang" in sys.argv or not os.path.exists(PATH_CSV):
        df = ekstrak_dari_folder()
    else:
        df = pd.read_csv(PATH_CSV)
        print(f"Memakai {PATH_CSV} (tambahkan --ekstrak-ulang untuk ekstrak ulang)")

    # Ubah ke array NumPy biasa (pandas versi baru memakai tipe string Arrow
    # yang tidak cocok dengan scikit-learn)
    X = df[list(NAMA_FITUR)].to_numpy(dtype=float)
    y = np.array(df["label"].astype(str).tolist())
    print(f"\nTotal data: {len(y)} file")
    for k in KELAS:
        print(f"  {k}: {(y == k).sum()} file")
    if min((y == k).sum() for k in KELAS) < 4:
        raise SystemExit("Butuh minimal 4 file per kelas agar tuning bermakna.")

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    n_fold = min(5, min((y_tr == k).sum() for k in KELAS))
    cv = StratifiedKFold(n_fold, shuffle=True, random_state=42)

    # 1) Baseline: parameter default
    base = RandomForestClassifier(random_state=42)
    cv_base = cross_val_score(base, X_tr, y_tr, cv=cv).mean()
    acc_base = accuracy_score(y_te, base.fit(X_tr, y_tr).predict(X_te))
    print(f"\n[BASELINE default] CV={cv_base:.3f} | uji={acc_base:.3f}")

    # 2) Tuning dengan GridSearchCV (hanya memakai data latih)
    print(f"\nTuning: {np.prod([len(v) for v in PARAM_GRID.values()])} kombinasi x {n_fold} fold ...")
    gs = GridSearchCV(
        RandomForestClassifier(random_state=42),
        PARAM_GRID, cv=cv, scoring="accuracy", n_jobs=-1, refit=True,
    )
    gs.fit(X_tr, y_tr)

    hasil = pd.DataFrame(gs.cv_results_)
    kolom = ["rank_test_score", "mean_test_score", "std_test_score"] + [
        f"param_{p}" for p in PARAM_GRID
    ]
    hasil = hasil[kolom].sort_values("rank_test_score")
    hasil.to_csv(os.path.join(BASE, "hasil_tuning.csv"), index=False)
    print("\n5 kombinasi terbaik:")
    print(hasil.head(5).to_string(index=False))

    best = gs.best_estimator_
    pred = best.predict(X_te)
    acc_tuned = accuracy_score(y_te, pred)
    print(f"\nParameter terbaik: {gs.best_params_}")
    print(f"[TUNED] CV={gs.best_score_:.3f} | uji={acc_tuned:.3f}")
    print(f"[BASELINE] CV={cv_base:.3f} | uji={acc_base:.3f}\n")

    print(classification_report(y_te, pred, zero_division=0))
    print(f"Confusion matrix (urutan kelas: {list(best.classes_)}):")
    print(confusion_matrix(y_te, pred, labels=best.classes_))

    imp = pd.Series(best.feature_importances_, index=NAMA_FITUR).sort_values(ascending=False)
    print("\n10 fitur terpenting:")
    print(imp.head(10).round(4))

    # 3) Model akhir: parameter terbaik, dilatih ulang dengan SEMUA data
    final = RandomForestClassifier(random_state=42, **gs.best_params_).fit(X, y)
    joblib.dump({
        "model": final,
        "nama_fitur": NAMA_FITUR,
        "akurasi_uji": float(acc_tuned),
        "akurasi_cv": float(gs.best_score_),
        "akurasi_baseline": float(acc_base),
        "param_terbaik": gs.best_params_,
        "jumlah_data": int(len(y)),
    }, os.path.join(BASE, "model_rf.joblib"))
    print("\nModel disimpan -> model_rf.joblib")


if __name__ == "__main__":
    main()
