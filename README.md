# 🧒 Stunting Risk Prediction

Machine learning model & web app (Streamlit) untuk memprediksi risiko stunting pada anak berdasarkan data antropometri, praktik pemberian makan, faktor ibu, serta kondisi lingkungan/sosial-ekonomi — dibangun mengikuti standar **WHO Child Growth Standards (Z-score TB/U & BB/U)** dan ambang batas **Permenkes RI No. 2 Tahun 2020**.

🔗 **Live demo:** [Link](https://stunting-risk-prediction-cgrkjyshn6dlgkm4dwfykg.streamlit.app/)

---

## 📌 Ringkasan Proyek

Proyek ini dibangun sebagai bagian dari kerja analisis deteksi dini stunting untuk Yayasan Indonesia Emas Merdeka, mencakup keseluruhan alur data science end-to-end:

1. **Exploratory Data Analysis (EDA)** — profil data, distribusi numerik & kategorikal, korelasi antar variabel
2. **Data Cleaning** — penanganan nilai ekstrem (0/999), konversi tipe data, missing value
3. **Perhitungan Z-score resmi WHO** — via library `pygrowup` (metode LMS WHO Child Growth Standards 2006), diklasifikasikan sesuai Permenkes RI No. 2/2020
4. **Analisis Faktor Risiko** — perbandingan tingkat stunting antar kelompok risiko (berat lahir rendah, ASI eksklusif, KEK ibu, pendidikan ibu, akses air bersih)
5. **Preprocessing** — feature selection, train-test split, imputation, SMOTENC (penanganan class imbalance), encoding & scaling
6. **Modeling** — perbandingan 4 algoritma klasifikasi (Logistic Regression, Decision Tree, Random Forest, Gradient Boosting), evaluasi dengan Precision/Recall/F1/ROC-AUC/PR-AUC, threshold tuning
7. **Deployment** — web app interaktif (Streamlit) untuk prediksi risiko stunting per anak

> **Catatan:** dataset yang digunakan untuk training adalah data dummy/simulasi (10.000 baris), dibuat untuk validasi pipeline. Seluruh alur (cleaning → Z-score → modeling → deployment) sudah divalidasi berjalan dengan benar dan siap diterapkan ulang ke data asli Yayasan.

---

## 🗂️ Struktur Repository

```
stunting-risk-prediction/
├── app.py                          # Aplikasi web Streamlit
├── requirements.txt                # Dependencies
├── model_artifacts/                # Model & preprocessing objects (hasil training)
│   ├── stunting_model.pkl
│   ├── numeric_imputer.pkl
│   ├── categorical_imputer.pkl
│   ├── onehot_encoder.pkl
│   ├── scaler.pkl
│   └── metadata.pkl
├── notebooks/
│   └── Stunting_Analysis_Modeling_Deployment.ipynb   # Notebook lengkap (EDA s.d. modeling)
└── README.md
```

---

## 🚀 Cara Menjalankan

### Jalankan secara lokal

```bash
git clone https://github.com/<username>/stunting-risk-prediction.git
cd stunting-risk-prediction
pip install -r requirements.txt
streamlit run app.py
```

Aplikasi akan terbuka otomatis di `http://localhost:8501`.

### Coba versi live

Aplikasi ini di-deploy via **Streamlit Community Cloud** — buka link demo di bagian atas README ini.

---

## 🧠 Metodologi Singkat

- **Target:** `Stunted_Flag` (klasifikasi biner, berdasarkan Z-score TB/U < -2 SD)
- **Class imbalance:** 91,99% tidak stunting vs 8,01% stunting → ditangani dengan **SMOTENC** (hanya pada data training)
- **Metrik evaluasi utama:** Recall & F1-Score (bukan hanya Accuracy), karena False Negative — anak stunting yang tidak terdeteksi — jauh lebih berisiko secara medis dibanding False Positive
- **Model terbaik** dipilih otomatis berdasarkan F1-Score tertinggi di data testing, dengan threshold keputusan yang juga dioptimalkan (bukan default 0.5)

---

## 🛠️ Tech Stack

`Python` · `pandas` · `scikit-learn` · `imbalanced-learn` · `pygrowup` · `plotly` · `Streamlit`

---

## 👤 Author

**Rizqy Arya Pratama**
Data Analyst | Informatics Engineering
BNSP Associate Data Scientist Certified

- Portfolio: [rizqy-arya-portfolio.vercel.app](https://rizqy-arya-portfolio.vercel.app)
- LinkedIn: [link](https://www.linkedin.com/in/rizqyap/)
- Email: rizqyarya704@gmail.com

---

## ⚠️ Disclaimer

Hasil prediksi aplikasi ini bersifat pendukung keputusan (decision support), **bukan diagnosis medis**. Keputusan klinis dan tindak lanjut tetap harus melalui pemeriksaan tenaga kesehatan profesional.
