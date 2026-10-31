"""
Ekstraksi fitur suara: RMS, Spectral Centroid, ZCR, MFCC (13 koefisien)

Struktur folder yang diasumsikan:
dataset/
├── berisik/
└── non_berisik/

Install dulu:  pip install librosa pandas numpy soundfile
Jalankan    :  python ekstraksi_fitur.py
"""

import os
import numpy as np
import pandas as pd
import librosa

FOLDER_DATASET = "berisik/"          # ganti sesuai lokasi datamu
SR = 22050                          # sample rate (semua file di-resample ke sini)
N_MFCC = 13
EKSTENSI = (".wav", ".mp3", ".flac", ".ogg", ".m4a")


def ekstrak_fitur(path_file):
    """Ekstrak fitur dari satu file audio, return dict (1 baris data)."""
    y, sr = librosa.load(path_file, sr=SR, mono=True)

    # 1. RMS (energi suara) -> per frame
    rms = librosa.feature.rms(y=y)[0]

    # 2. Spectral centroid (pusat spektrum) -> per frame, satuan Hz
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]

    # 3. ZCR (zero crossing rate) -> per frame
    zcr = librosa.feature.zero_crossing_rate(y)[0]

    # 4. MFCC 13 koefisien -> shape (13, jumlah_frame)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)

    # Setiap fitur punya banyak nilai (satu per frame), jadi diringkas
    # jadi mean dan std supaya 1 file = 1 baris.
    fitur = {
        "rms_mean": np.mean(rms),
        "rms_std": np.std(rms),
        "centroid_mean": np.mean(centroid),
        "centroid_std": np.std(centroid),
        "zcr_mean": np.mean(zcr),
        "zcr_std": np.std(zcr),
    }
    for i in range(N_MFCC):
        fitur[f"mfcc{i+1}_mean"] = np.mean(mfcc[i])
        fitur[f"mfcc{i+1}_std"] = np.std(mfcc[i])

    return fitur


def main():
    baris = []
    for label in sorted(os.listdir(FOLDER_DATASET)):
        folder_label = os.path.join(FOLDER_DATASET, label)
        if not os.path.isdir(folder_label):
            continue

        for nama_file in sorted(os.listdir(folder_label)):
            if not nama_file.lower().endswith(EKSTENSI):
                continue
            path_file = os.path.join(folder_label, nama_file)
            try:
                fitur = ekstrak_fitur(path_file)
            except Exception as e:
                print(f"[GAGAL] {path_file}: {e}")
                continue

            fitur["file"] = nama_file
            fitur["label"] = label
            baris.append(fitur)
            print(f"[OK] {label}/{nama_file}")

    df = pd.DataFrame(baris)
    # taruh kolom file & label di depan
    kolom = ["file", "label"] + [c for c in df.columns if c not in ("file", "label")]
    df = df[kolom]
    df.to_csv("fitur_suara.csv", index=False)
    print(f"\nSelesai. {len(df)} file diproses -> fitur_suara.csv")


if __name__ == "__main__":
    main()
