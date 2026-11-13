# 🛍️ Aplikasi Prediksi Pembatalan Pesanan E-Commerce

Aplikasi web (Streamlit) untuk memakai model machine learning **XGBoost** hasil
Tugas Besar CRISP-DM. Aplikasi memprediksi **probabilitas sebuah pesanan
dibatalkan**, menampilkan faktor penyebabnya, dan memberi rekomendasi tindakan.

Model asli: `model_ecommerce_ganang.pth` (XGBoost Tuned, hasil hyperparameter
tuning + SMOTE). File tersebut sudah dikonversi ke `models/model_ecommerce_ganang_app.pth`
agar bisa dibuka oleh XGBoost versi baru.

---

## ✨ Fitur aplikasi

| Tab | Isi |
|-----|-----|
| **🎯 Prediksi** | Form 13 fitur pesanan, contoh data sekali klik, hasil probabilitas + gauge, tingkat risiko, faktor SHAP, rekomendasi tindakan |
| **📁 Prediksi Batch** | Unggah CSV (boleh ratusan/ribuan baris), tabel hasil, unduh CSV, statistik ringkas, evaluasi otomatis bila kolom label tersedia |
| **📊 Performa Model** | Akurasi, presisi, recall, F1, ROC-AUC, feature importance, hyperparameter, info dataset |
| **ℹ️ Panduan** | Alur ML, cara menjalankan, struktur proyek, catatan penggunaan |

Pengaturan penting di sidebar: **ambang keputusan** (default 50%). Turunkan ambang
bila ingin menangkap lebih banyak pembatalan (recall naik, presisi turun).

Tingkat risiko dihitung relatif terhadap ambang:

- 🟢 **Rendah** — probabilitas < ½ ambang
- 🟠 **Sedang** — ½ ambang ≤ probabilitas < ambang
- 🔴 **Tinggi** — probabilitas ≥ ambang (diprediksi batal)

---

## 🚀 Cara menjalankan

### Opsi 1 — Windows, sekali klik

Klik dua kali **`run_app.bat`**. Skrip akan membuat virtual environment,
memasang dependensi, lalu membuka aplikasi di browser.

### Opsi 2 — manual

```bash
cd "Part 3/aplikasi_prediksi_pembatalan"

python -m venv .venv
.venv\Scripts\activate            # Windows
# source .venv/bin/activate       # macOS / Linux

pip install -r requirements.txt
streamlit run app.py
```

Buka alamat yang muncul (biasanya <http://localhost:8501>).

> Butuh Python 3.10+ dan koneksi internet saat instalasi pertama.

---

## 📈 Performa model

| Metrik | Nilai |
|--------|------:|
| Akurasi latih | 91,98% |
| Akurasi uji | 89,14% |
| Gap train–test | 2,84% (Goodfit) |
| Presisi | 78,07% |
| Recall | 28,35% |
| F1-Score | 41,60% |
| ROC-AUC | 75,19% |

Model sengaja dioptimalkan pada **F1-Score** karena kelas pembatalan tidak
seimbang (13,6% pesanan batal). Recall yang relatif rendah berarti model
konservatif: prediksi "batal" yang dihasilkan cukup akurat, tetapi tidak semua
pembatalan berhasil ditangkap. Gunakan aplikasi sebagai alat bantu prioritas.

---

## 🧠 Pipeline machine learning

```
Data pesanan mentah
   │  jumlah produk, berat, diskon, kategori, ongkir, kurir,
   │  metode bayar, kota, provinsi, bulan/hari/jam
   ▼
Imputasi (median / modus) → LabelEncoder → StandardScaler
   ▼
XGBoost (200 pohon, max_depth=6, learning_rate=0,1, subsample=0,8)
   ▼
Probabilitas batal → ambang keputusan → tingkat risiko
```

13 fitur yang dipakai: `total_qty`, `total_weight_gr`, `Total Diskon`,
`product_categories`, `num_product_categories`, `Opsi Pengiriman`,
`Metode Pembayaran`, `Kota/Kabupaten`, `Provinsi`, `Perkiraan Ongkos Kirim`,
`order_month`, `order_dayofweek`, `order_hour`.

Nilai kategori yang belum pernah dilihat model otomatis diarahkan ke kategori
paling umum agar aplikasi tidak error.

---

## 📂 Struktur proyek

```
aplikasi_prediksi_pembatalan/
├── app.py                       # antarmuka Streamlit
├── models/
│   ├── model_ecommerce_ganang_app.pth   # bundle siap pakai (1,7 MB)
│   └── model_ecommerce_ganang_app.model.ubj  # buffer model (backup)
├── src/
│   ├── config.py                # path, label fitur, aturan risiko
│   ├── bundle.py                # pemuat bundle model
│   ├── preprocess.py            # imputasi, encoding, scaling
│   ├── predict.py               # inferensi + SHAP
│   ├── explain.py               # tabel faktor & rekomendasi
│   └── ui.py                    # CSS dan komponen tampilan
├── tools/
│   ├── convert_model.py         # konversi .pth asli → bundle aplikasi
│   └── legacy_xgb.py            # pembaca/penulis UBJSON XGBoost
├── data/contoh_pesanan.csv      # contoh 3 pesanan untuk uji batch
├── .streamlit/config.toml       # tema aplikasi
├── requirements.txt
├── run_app.bat
└── README.md
```

---

## 🔧 Tentang konversi model

File `.pth` menyimpan objek `XGBClassifier` melalui `torch.save`. Di dalamnya,
XGBoost menyimpan model sebagai **snapshot biner internal**. Snapshot itu gagal
dibuka langsung oleh XGBoost ≥ 2.0
(`XGBoostError: input stream corrupted`).

`tools/convert_model.py` melakukan:

1. Membuka arsip `.pth` dan unpickle isinya tanpa memanggil loader XGBoost.
2. Mengambil byte snapshot booster.
3. Menerjemahkan struktur UBJSON snapshot menjadi format **model resmi** XGBoost.
4. Menyimpan ulang semua objek preprocessing + buffer model ke bundle aplikasi.

Konversi hanya perlu dijalankan sekali (sudah dilakukan). Untuk mengulang:

```bash
python tools/convert_model.py
```

Model hasil konversi sudah diverifikasi menghasilkan prediksi yang identik
dengan model asli (akurasi 89,14%, F1 41,60%, ROC-AUC 75,19% pada data uji).
Script juga memerlukan dataset bersumber (`crispdm-homedecor/data/all_months_clean.csv`)
hanya untuk membuat contoh data; bila tidak ada, aplikasi tetap berjalan.

---

## 🧪 Uji cepat

1. Buka tab **Prediksi** → klik **🔴 Risiko Tinggi** → hasil keluar otomatis.
2. Buka tab **Prediksi Batch** → unggah `data/contoh_pesanan.csv` →
   **Jalankan prediksi batch**.
3. Ubah **ambang keputusan** di sidebar → label risiko dan prediksi ikut berubah.

---

## ❓ Troubleshooting

| Masalah | Solusi |
|---------|--------|
| `Model tidak ditemukan` | Pastikan `models/model_ecommerce_ganang_app.pth` ada. Bila hilang, jalankan `python tools/convert_model.py` |
| `ModuleNotFoundError: streamlit` | Aktifkan venv lalu `pip install -r requirements.txt` |
| `XGBoostError` saat memuat model | Pastikan memakai bundle `*_app.pth`, bukan file `.pth` asli |
| Port 8501 dipakai | `streamlit run app.py --server.port 8502` |
| Kolom CSV tidak dikenali | Unduh template CSV di tab Prediksi Batch dan samakan nama kolom |

---

## 👤 Penyusun

**Ganang Ardi Pratama** — NPM 202343501804 — Kelas X7L  
Tugas Besar Mata Kuliah Machine Learning (CRISP-DM)
