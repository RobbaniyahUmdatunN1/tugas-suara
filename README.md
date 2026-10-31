# Klasifikasi Suara Berisik vs Non-Berisik

Ekstraksi fitur audio (RMS, Spectral Centroid, ZCR, MFCC 13) dengan librosa,
klasifikasi dengan Random Forest (hyperparameter di-tuning dengan GridSearchCV),
disajikan sebagai aplikasi web Streamlit. Satu port: 8501 (di dalam container).

## Menjalankan dengan Docker
```
docker compose up --build
```
Buka http://localhost:8001 (angka 8001 = nomor kelompok, atur di docker-compose.yml).

## Melatih ulang model (opsional, butuh folder berisik/ dan non-berisik/)
```
pip install -r requirements.txt
python train_model.py --ekstrak-ulang
```
Hasil: model_rf.joblib, fitur_suara.csv, hasil_tuning.csv. Setelah itu build ulang image.

## Pemantau mikrofon desktop (di luar Docker)
```
pip install -r requirements-monitor.txt
python monitor.py
```
