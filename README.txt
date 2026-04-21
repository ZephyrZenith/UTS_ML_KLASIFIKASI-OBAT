===========================================================
PROYEK: CLINICAL-AI ULTIMATE - REKOMENDASI OBAT KLINIS
===========================================================
Nama         : Putri Adelia
NIM          : F1G123033
Program Studi: Ilmu Komputer
Mata Kuliah  : Machine Learning 
-----------------------------------------------------------

DESKRIPSI PROYEK:
Sistem ini dirancang untuk memberikan rekomendasi farmakoterapi 
(jenis obat) yang paling tepat berdasarkan profil klinis pasien. 
Menggunakan perbandingan tiga algoritma Gradient Boosting 
Decision Tree (GBDT) mutakhir: CatBoost, XGBoost, dan LightGBM.

STRUKTUR FOLDER:
Klasifikasi_Obat/
├── dataset/
│   └── patient_clinical_profiles_real_drugs.csv   # Data mentah
├── src/
│   ├── preprocessing.py   # Logika pembersihan & encoding data
│   ├── train.py           # Script untuk melatih model AI
│   └── app.py             # Aplikasi antarmuka web (Gradio)
├── catboost_model.pkl      # File model (Otak AI) - Dihasilkan setelah training
├── scaler.pkl              # File standarisasi data
├── label_encoders.pkl      # File penerjemah teks ke angka
└── README.txt              # Dokumentasi ini

PERSYARATAN SISTEM (REQUISITES):
- Python 3.10 - 3.13
- Pustaka pendukung (Instalasi melalui terminal):
  pip install pandas numpy<2.3 scikit-learn xgboost lightgbm catboost joblib gradio

CATATAN PENTING (FIX BUGS):
1. Masalah Numpy: Gunakan numpy versi < 2.3 untuk menghindari konflik dengan Scipy.
2. Python 3.13: Pastikan label (y) diubah ke format np.int32 di train.py 
   sebelum dimasukkan ke model CatBoost.

CARA MENJALANKAN:
1. Tahap Latih (Training):
   Buka terminal di folder utama, lalu ketik:
   python src/train.py
   (Tunggu hingga file .pkl tercipta di folder utama)

2. Tahap Aplikasi (Deployment):
   Setelah file model muncul, jalankan perintah:
   python src/app.py
   (Klik link http://127.0.0.1:7860 yang muncul di terminal)

METRIK EVALUASI TERBAIK:
- Accuracy : > 99%
- F1-Score : > 0.99
(Hasil pengujian pada unseen data 20%)

===========================================================
Copyright (c) 2026 - Putri Adelia. All Rights Reserved.
===========================================================