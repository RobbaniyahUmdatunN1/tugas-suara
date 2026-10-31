"""
Aplikasi klasifikasi suara berisik vs non-berisik (Streamlit).
Dua cara input: unggah file audio, atau rekam dari mikrofon lewat browser.

Jalankan lokal :  py -V:3.13 -m streamlit run app.py
Jalankan Docker:  docker compose up --build
"""

import os
import tempfile
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import librosa

from fitur import ekstrak_fitur, SR

BASE = os.path.dirname(os.path.abspath(__file__))
PATH_MODEL = os.path.join(BASE, "model_rf.joblib")
KELAS_BERISIK = "berisik"

st.set_page_config(page_title="Klasifikasi Suara", page_icon="🔊")
st.title("🔊 Klasifikasi Suara: Berisik vs Non-Berisik")
st.caption("Fitur librosa (RMS, Spectral Centroid, ZCR, MFCC 13) + Random Forest.")


@st.cache_resource
def muat_model():
    return joblib.load(PATH_MODEL)


if not os.path.exists(PATH_MODEL):
    st.error("File model_rf.joblib tidak ditemukan. Jalankan dulu: `python train_model.py`")
    st.stop()

info = muat_model()
model = info["model"]
idx_berisik = list(model.classes_).index(KELAS_BERISIK)

with st.sidebar:
    st.header("Info Model")
    st.write("Algoritma: Random Forest")
    st.write(f"Data latih: {info['jumlah_data']} file")
    st.write(f"Akurasi uji: {info['akurasi_uji']:.1%}")
    st.write(f"Akurasi CV: {info['akurasi_cv']:.1%}")
    if "param_terbaik" in info:
        st.write("Parameter terbaik:")
        st.json(info["param_terbaik"])
    st.header("Pengaturan")
    ambang = st.slider("Ambang peluang 'berisik'", 0.3, 0.9, 0.5, 0.05)


def analisis(data_bytes, suffix):
    """Ekstrak fitur dari audio, prediksi, lalu tampilkan hasilnya."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(data_bytes)
        path_tmp = tmp.name
    try:
        with st.spinner("Mengekstrak fitur..."):
            fitur = ekstrak_fitur(path_tmp)
            y, _ = librosa.load(path_tmp, sr=SR, mono=True)
    except Exception as e:
        st.error(f"Gagal membaca audio: {e}")
        return
    finally:
        os.remove(path_tmp)

    proba = model.predict_proba(fitur.reshape(1, -1))[0]
    p_berisik = proba[idx_berisik]

    if p_berisik >= ambang:
        st.error("🚨 Terlalu ramai. Jangan berisik!")
    else:
        st.success("✅ Suasana tenang.")

    c1, c2 = st.columns(2)
    c1.metric("Peluang berisik", f"{p_berisik:.1%}")
    c2.metric("Ambang", f"{ambang:.0%}")
    st.bar_chart(pd.DataFrame({"probabilitas": proba}, index=model.classes_))

    d1, d2, d3 = st.columns(3)
    d1.metric("RMS (energi)", f"{fitur[0]:.4f}")
    d2.metric("Centroid (Hz)", f"{fitur[2]:.0f}")
    d3.metric("ZCR", f"{fitur[4]:.4f}")

    st.subheader("Waveform")
    st.line_chart(y[:: max(1, len(y) // 3000)])
    with st.expander("Lihat semua 32 fitur"):
        st.dataframe(pd.DataFrame({"fitur": info["nama_fitur"], "nilai": fitur}))


tab_unggah, tab_rekam = st.tabs(["📁 Unggah file", "🎙️ Rekam mikrofon"])

with tab_unggah:
    file = st.file_uploader("Unggah file audio", type=["mp3", "wav", "ogg", "flac", "m4a"])
    if file is not None:
        st.audio(file)
        analisis(file.getbuffer(), os.path.splitext(file.name)[1] or ".wav")

with tab_rekam:
    st.write("Rekam suara sekitar beberapa detik, lalu hentikan untuk dianalisis. "
             "Browser akan meminta izin mikrofon (hanya jalan di localhost atau HTTPS).")
    if hasattr(st, "audio_input"):
        rekaman = st.audio_input("Rekam suara sekitar")
        if rekaman is not None:
            analisis(rekaman.getbuffer(), ".wav")
    else:
        st.warning("Versi Streamlit terlalu lama untuk fitur rekam. Jalankan: pip install -U streamlit")
