"""
Ekstraksi RMS per frame (versi perbaikan).

Perbedaan dengan versi sebelumnya:
- Mencari file audio di SEMUA subfolder (rekursif), jadi struktur folder lebih bebas.
- Path dataset dihitung dari lokasi skrip, bukan dari folder terminal.
- Label = nama folder tempat file berada.
- Kalau tidak ada file ditemukan, muncul pesan yang jelas (bukan error aneh).
"""

import os
import numpy as np
import pandas as pd
import librosa

# Folder berisi audio. Boleh relatif terhadap lokasi skrip, atau path lengkap.
# Contoh path lengkap: r"D:\Semester 5\PSD\Tugas 1\Tugas Suara"
NAMA_FOLDER = "."

SR = 22050
FRAME_LENGTH = 2048
HOP_LENGTH = 512
OVERLAP_LENGTH = FRAME_LENGTH - HOP_LENGTH
EKSTENSI = (".wav", ".mp3", ".flac", ".ogg", ".m4a")

BASE = os.path.dirname(os.path.abspath(__file__))
FOLDER_DATASET = os.path.abspath(os.path.join(BASE, NAMA_FOLDER))


def cari_audio(folder):
    """Kumpulkan semua file audio (rekursif) -> list (label, path)."""
    hasil = []
    for root, _, files in os.walk(folder):
        for f in sorted(files):
            if f.lower().endswith(EKSTENSI):
                hasil.append((os.path.basename(root), os.path.join(root, f)))
    return hasil


def main():
    print(f"Mencari audio di: {FOLDER_DATASET}")
    daftar = cari_audio(FOLDER_DATASET)
    print(f"Ditemukan {len(daftar)} file audio\n")

    if not daftar:
        print("TIDAK ADA file audio ditemukan.")
        print("Cek: (1) NAMA_FOLDER sudah benar? (2) ekstensi file termasuk", EKSTENSI)
        return

    print(f"Frame length  : {FRAME_LENGTH} sampel = {FRAME_LENGTH / SR * 1000:.1f} ms")
    print(f"Hop length    : {HOP_LENGTH} sampel = {HOP_LENGTH / SR * 1000:.1f} ms")
    print(f"Overlap length: {OVERLAP_LENGTH} sampel ({OVERLAP_LENGTH / FRAME_LENGTH:.0%})\n")

    semua_frame, ringkasan = [], []

    for label, path_file in daftar:
        nama_file = os.path.basename(path_file)
        try:
            y, sr = librosa.load(path_file, sr=SR, mono=True)
        except Exception as e:
            print(f"[GAGAL] {path_file}: {e}")
            continue

        rms = librosa.feature.rms(
            y=y, frame_length=FRAME_LENGTH, hop_length=HOP_LENGTH
        )[0]
        waktu = librosa.times_like(rms, sr=sr, hop_length=HOP_LENGTH)

        semua_frame.append(pd.DataFrame({
            "file": nama_file,
            "label": label,
            "frame": np.arange(len(rms)),
            "waktu_detik": waktu,
            "rms": rms,
        }))
        ringkasan.append({
            "file": nama_file,
            "label": label,
            "jumlah_frame": len(rms),
            "rms_mean": np.mean(rms),
            "rms_std": np.std(rms),
            "rms_min": np.min(rms),
            "rms_max": np.max(rms),
            "rms_median": np.median(rms),
        })
        print(f"[OK] {label}/{nama_file} -> {len(rms)} frame")

    if not semua_frame:
        print("Semua file gagal dibaca. Cek pesan [GAGAL] di atas (kemungkinan ffmpeg belum terpasang).")
        return

    out1 = os.path.join(BASE, "rms_per_frame.csv")
    out2 = os.path.join(BASE, "rms_ringkasan.csv")
    pd.concat(semua_frame, ignore_index=True).to_csv(out1, index=False)
    pd.DataFrame(ringkasan).to_csv(out2, index=False)
    print(f"\nSelesai -> {out1}\n           {out2}")


if __name__ == "__main__":
    main()
