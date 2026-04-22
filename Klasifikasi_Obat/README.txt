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
│   └── patient_clinical_profiles_real_drugs.csv   # Data mentah pasien
├── src/
│   ├── preprocessing.py   # Logika pembersihan & encoding data
│   ├── train.py           # Script untuk melatih & membangun model AI
│   └── app.py             # Script utama aplikasi antarmuka web (Gradio)
├── F1G123033_PUTRI_ADELIA_KLASIFIKASI_JENIS_OBAT.ipynb # Buku Jurnal Riset (EDA, Tuning, Visualisasi Data)
├── catboost_model.pkl      # File model (Otak AI) - Dihasilkan setelah training
├── scaler.pkl              # File standarisasi rentang data
├── label_encoders.pkl      # File penerjemah teks ke angka metrik
└── README.txt              # Dokumentasi teknis ini

PERSYARATAN SISTEM (REQUISITES):
- Python 3.10 - 3.13
- Pustaka pendukung (Instalasi melalui terminal):
  pip install pandas "numpy<2.3" scikit-learn xgboost lightgbm catboost joblib gradio

CATATAN PENTING (FIX BUGS & COMPATIBILITY):
1. Masalah Numpy: Wajib menggunakan numpy versi < 2.3 untuk menghindari konflik dependensi dengan Scipy.
2. Python 3.13: Pastikan label target (y) dikonversi ke format np.int32 di train.py 
   sebelum dimasukkan ke model CatBoost untuk mencegah TypeError.
3. Dimensi Scaler: Pada app.py, pastikan scaler.transform memproses seluruh 10 kolom 
   secara bersamaan (menggunakan df[:]) untuk menghindari ValueError (Feature names mismatch).

CARA MENJALANKAN (WORKFLOW):

1. Tahap Pembuktian & Visualisasi Riset (Opsional):
   - Buka file .ipynb di Visual Studio Code.
   - Pastikan ekstensi "Jupyter" telah terinstal.
   - File ini berisi seluruh bukti Exploratory Data Analysis (EDA), Hyperparameter Tuning, 
     serta grafik komparasi akurasi lintas rasio partisi data.

2. Tahap Latih Model (Training):
   - Buka terminal di VS Code (pastikan berada di folder Klasifikasi_Obat).
   - Jalankan perintah: python src/train.py
   - Tunggu proses selesai hingga 3 file .pkl tercipta di dalam folder.

3. Tahap Operasional Aplikasi (Deployment):
   - Setelah file model (.pkl) tersedia, jalankan perintah: python src/app.py
   - Sistem akan memberikan Local URL (contoh: http://127.0.0.1:7860).
   - Klik link tersebut (Ctrl + Click) untuk membuka antarmuka Clinical-AI di browser.

METRIK EVALUASI TERBAIK:
- Accuracy : > 99%
- F1-Score : > 0.99
(Hasil pengujian divalidasi menggunakan 20% Unseen Testing Data)

===========================================================
Copyright (c) 2026 - Putri Adelia. All Rights Reserved.
===========================================================