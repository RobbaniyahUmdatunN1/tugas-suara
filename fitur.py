"""
Modul ekstraksi fitur. Dipakai oleh train_model.py DAN app.py
supaya fitur saat training dan saat prediksi PERSIS sama.
"""

import numpy as np
import librosa

SR = 22050
N_MFCC = 13
FRAME_LENGTH = 2048
HOP_LENGTH = 512

NAMA_FITUR = (
    ["rms_mean", "rms_std", "centroid_mean", "centroid_std", "zcr_mean", "zcr_std"]
    + [f"mfcc{i}_mean" for i in range(1, N_MFCC + 1)]
    + [f"mfcc{i}_std" for i in range(1, N_MFCC + 1)]
)  # total 32 fitur


def ekstrak_fitur(path_file):
    """Return vektor 32 fitur (np.array) dari satu file audio."""
    y, sr = librosa.load(path_file, sr=SR, mono=True)
    return ekstrak_fitur_dari_sinyal(y, sr)


def ekstrak_fitur_dari_sinyal(y, sr):
    """Return vektor 32 fitur dari sinyal (array) yang sudah dimuat/direkam."""
    rms = librosa.feature.rms(y=y, frame_length=FRAME_LENGTH, hop_length=HOP_LENGTH)[0]
    centroid = librosa.feature.spectral_centroid(
        y=y, sr=sr, n_fft=FRAME_LENGTH, hop_length=HOP_LENGTH
    )[0]
    zcr = librosa.feature.zero_crossing_rate(
        y, frame_length=FRAME_LENGTH, hop_length=HOP_LENGTH
    )[0]
    mfcc = librosa.feature.mfcc(
        y=y, sr=sr, n_mfcc=N_MFCC, n_fft=FRAME_LENGTH, hop_length=HOP_LENGTH
    )

    return np.hstack([
        np.mean(rms), np.std(rms),
        np.mean(centroid), np.std(centroid),
        np.mean(zcr), np.std(zcr),
        np.mean(mfcc, axis=1), np.std(mfcc, axis=1),
    ])
