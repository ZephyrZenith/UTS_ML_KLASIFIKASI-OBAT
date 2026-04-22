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

1. TAHAP EVALUASI RISET
   File ini berisi seluruh rekam jejak riset mulai dari Data Understanding, 
   Visualisasi (EDA), Hyperparameter Tuning, hingga Evaluasi Metrik.
   
   => File: F1G123033_PUTRI_ADELIA_KLASIFIKASI_JENIS_OBAT.ipynb
   
   Cara menjalankan file ini di Visual Studio Code:
   a. Klik file .ipynb tersebut di panel sebelah kiri untuk membukanya.
   b. Pilih "Kernel" (Mesin Python): Lihat ke pojok kanan atas layar VS Code, 
      klik tombol "Select Kernel".
   c. Pilih "Python Environments", lalu klik "Python 3.13.1" (atau versi Python 
      terbaru yang tersedia di sistem).
   d. Jika muncul notifikasi kecil di pojok kanan bawah yang meminta Anda 
      menginstal "ipykernel", klik "Install" dan tunggu hingga selesai.
   e. Di deretan menu atas notebook tersebut, klik "Clear All Outputs" 
      terlebih dahulu untuk membersihkan sisa riwayat Google Colab.
   f. Terakhir, klik tombol "Run All" (ikon panah ganda) dan tunggu hingga 
      seluruh grafik visualisasi dan hasil evaluasinya muncul.

2. TAHAP IMPLEMENTASI & ANTARMUKA WEB (Deployment Clinical-AI)
   Gunakan tahap ini untuk menjalankan prototipe aplikasi web 
   interaktif yang bisa dicoba langsung oleh pengguna.

   Langkah-langkah Eksekusi (Tutorial Terminal):
   
   a. Buka Terminal & Cek Posisi Folder (SANGAT PENTING):
      - Di VS Code, klik menu "Terminal" di atas layar -> "New Terminal".
      - Perhatikan teks di sebelah kiri tempat Anda mengetik (Current Directory).
      - Pastikan posisi terminal Anda sudah berada TEPAT di dalam folder utama 
        proyek. Teksnya harus berakhiran: ...\Klasifikasi_Obat>
      - Jika ada tambahan 'src' di ujungnya (...\Klasifikasi_Obat\src>), 
        ketik perintah: cd .. lalu tekan Enter.

   b. Pastikan File Model (.pkl) Tersedia:
      - Cek di daftar file Anda apakah file 'catboost_model.pkl', 'scaler.pkl', 
        dan 'label_encoders.pkl' sudah ada di dalam folder Klasifikasi_Obat.
      - Jika file tersebut belum ada, latih ulang modelnya dengan mengetik: 
        => python src/train.py (lalu tekan Enter dan tunggu hingga selesai).

   c. Menjalankan Aplikasi Web:
      - Jika file model sudah dipastikan ada, ketik perintah ini di terminal:
        => python src/app.py atau python app.py
      - Lalu tekan Enter.

   d. Mengakses Aplikasi di Browser:
      - Tunggu beberapa detik hingga terminal memunculkan tulisan berwarna biru: 
        * Running on local URL:  http://127.0.0.1:7860
      - Arahkan kursor mouse Anda ke link tersebut.
      - Tahan tombol 'Ctrl' (Windows) atau 'Cmd' (Mac) di keyboard, lalu 
        Klik kiri pada link tersebut. (Bisa juga dengan memblok link tersebut 
        dan copy-paste ke Google Chrome/browser Anda).

   e. Uji Coba Prototipe:
      - Halaman antarmuka Clinical-AI akan terbuka di browser.
      - Masukkan data profil pasien (Usia, Tekanan Darah, Fungsi Ginjal, dll) 
        pada panel sebelah kiri.
      - Klik tombol "PROSES ANALISIS KLINIS".
      - Hasil rekomendasi obat akan muncul di panel sebelah kanan secara instan.

METRIK EVALUASI TERBAIK:
- Accuracy : > 99%
- F1-Score : > 0.99
(Hasil pengujian divalidasi menggunakan 20% Unseen Testing Data)

===========================================================
Copyright (c) 2026 - Putri Adelia. All Rights Reserved.
===========================================================