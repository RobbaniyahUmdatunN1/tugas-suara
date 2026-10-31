"""
Monitor suara sekitar (real-time) lewat mikrofon.
Kalau terdeteksi terlalu ramai -> tampil notifikasi "Terlalu ramai. Jangan berisik!"

Jalankan:  py -V:3.13 monitor.py
"""

import os
import time
import queue
import threading
from collections import deque

import joblib
import numpy as np
import sounddevice as sd
import librosa
import tkinter as tk

from fitur import ekstrak_fitur_dari_sinyal, SR

try:
    import winsound          # hanya ada di Windows
except ImportError:
    winsound = None

# ---------------- Pengaturan ----------------
DURASI = 3            # detik per jendela rekaman
RIWAYAT = 3           # lihat N jendela terakhir
MIN_BERISIK = 2       # alarm bila >= 2 dari 3 jendela terakhir berisik
COOLDOWN_BEEP = 10    # detik jeda antar bunyi alarm
KELAS_BERISIK = "berisik"
# --------------------------------------------

BASE = os.path.dirname(os.path.abspath(__file__))
info = joblib.load(os.path.join(BASE, "model_rf.joblib"))
model = info["model"]
idx_berisik = list(model.classes_).index(KELAS_BERISIK)

antrian = queue.Queue()
berhenti = threading.Event()


def loop_rekam():
    """Thread: rekam terus-menerus, ekstrak fitur, prediksi, kirim hasil ke antrian."""
    try:
        sr_dev = int(sd.query_devices(kind="input")["default_samplerate"])
        while not berhenti.is_set():
            data = sd.rec(int(DURASI * sr_dev), samplerate=sr_dev,
                          channels=1, dtype="float32")
            sd.wait()
            y = data[:, 0]
            if sr_dev != SR:
                y = librosa.resample(y, orig_sr=sr_dev, target_sr=SR)
            fitur = ekstrak_fitur_dari_sinyal(y, SR)
            p = model.predict_proba(fitur.reshape(1, -1))[0][idx_berisik]
            antrian.put(("hasil", float(p), float(fitur[0])))
    except Exception as e:
        antrian.put(("error", str(e), 0.0))


class App:
    def __init__(self, root):
        self.root = root
        root.title("Monitor Suara Sekitar")
        root.geometry("460x300")

        self.status = tk.Label(root, text="Mendengarkan...", font=("Segoe UI", 20, "bold"),
                               wraplength=420, height=3, bg="#2e7d32", fg="white")
        self.status.pack(fill="x", padx=12, pady=12)

        self.detail = tk.Label(root, text="Menunggu rekaman pertama...", font=("Segoe UI", 11))
        self.detail.pack()

        tk.Label(root, text="Ambang kepekaan (peluang 'berisik' minimal):").pack(pady=(14, 0))
        self.ambang = tk.Scale(root, from_=0.3, to=0.9, resolution=0.05,
                               orient="horizontal", length=300)
        self.ambang.set(0.5)
        self.ambang.pack()

        self.riwayat = deque(maxlen=RIWAYAT)
        self.beep_terakhir = 0.0
        root.protocol("WM_DELETE_WINDOW", self.tutup)
        self.cek_antrian()

    def cek_antrian(self):
        while not antrian.empty():
            jenis, a, b = antrian.get()
            if jenis == "error":
                self.status.config(text=f"Error mikrofon: {a}", bg="#616161", font=("Segoe UI", 12))
                continue
            self.proses(a, b)
        self.root.after(200, self.cek_antrian)

    def proses(self, p_berisik, rms):
        self.riwayat.append(p_berisik >= self.ambang.get())
        ramai = sum(self.riwayat) >= MIN_BERISIK
        self.detail.config(text=f"Peluang berisik: {p_berisik:.0%}   |   RMS: {rms:.4f}")

        if ramai:
            self.status.config(text="TERLALU RAMAI!\nJangan berisik", bg="#c62828",
                               font=("Segoe UI", 20, "bold"))
            if time.time() - self.beep_terakhir > COOLDOWN_BEEP:
                self.beep_terakhir = time.time()
                if winsound:
                    winsound.Beep(1000, 500)
                else:
                    self.root.bell()
        else:
            self.status.config(text="Suasana tenang", bg="#2e7d32",
                               font=("Segoe UI", 20, "bold"))

    def tutup(self):
        berhenti.set()
        self.root.destroy()


if __name__ == "__main__":
    threading.Thread(target=loop_rekam, daemon=True).start()
    root = tk.Tk()
    App(root)
    root.mainloop()
