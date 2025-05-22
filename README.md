# 📊 Analisis dan Prediksi Attrition Karyawan - Proyek Data Science

## Project Overview

Proyek ini bertujuan mengatasi dua permasalahan utama:

- **Pernyataan Masalah 1**  
  Tingginya tingkat *attrition* (keluar/mundurnya karyawan) yang melebihi 10% pada perusahaan Jaya Jaya Maju, menyebabkan ketidakstabilan dan peningkatan biaya rekrutmen.

- **Pernyataan Masalah 2**  
  Kurangnya sistem monitoring berbasis data yang dapat membantu manajer HR dalam mengidentifikasi faktor-faktor utama penyebab *attrition* dan mengambil keputusan berbasis data.

### Alur dan Metode Proyek

1. **Business Understanding**
2. **Data Understanding**
3. **Data Preparation**
4. **Modeling (Linear Regression, Ridge, Lasso)**
5. **Evaluation**
6. **Deploy / Interpretasi Bisnis**

### Tujuan Proyek

- Mengidentifikasi faktor-faktor utama yang memengaruhi tingkat *attrition* karyawan.
- Membuat model prediksi *attrition* berdasarkan fitur-fitur karyawan.
- Menyediakan insight dan dashboard monitoring berbasis data untuk HR.

## Business Understanding

Jaya Jaya Maju merupakan salah satu perusahaan multinasional yang telah berdiri sejak tahun 2000. Ia memiliki lebih dari 1000 karyawan yang tersebar di seluruh penjuru negeri.

Walaupun telah menjadi menjadi perusahaan yang cukup besar, Jaya Jaya Maju masih cukup kesulitan dalam mengelola karyawan. Hal ini berimbas tingginya attrition rate (rasio jumlah karyawan yang keluar dengan total karyawan keseluruhan) hingga lebih dari 10%.

Untuk mencegah hal ini semakin parah, manajer departemen HR ingin meminta bantuan Anda mengidentifikasi berbagai faktor yang mempengaruhi tingginya attrition rate tersebut. Selain itu, ia juga meminta Anda untuk membuat business dashboard untuk membantunya memonitori berbagai faktor tersebut. gunakan preddiction analis dengan regression linear dan ada tahap data understanding, data preparation/preprocessingg, modeling dan evaluation

## 🎯 Goals

- Membangun model regresi untuk memprediksi nilai *Attrition*.
- Memberikan insight untuk tindakan preventif manajemen SDM.
- Visualisasi korelasi faktor-faktor seperti usia, pendapatan, dan jarak rumah.

## Data Understanding

### Informasi Dataset:

- **Jumlah Data**: Sekitar 1470 baris data karyawan
- **Kondisi Data**:
  - **Missing Value**: Ditemukan pada kolom `Attrition`
  - **Jumlah Duplikasi Ditemukan**: Akan ditentukan melalui `.duplicated().sum()`
  - **Jumlah Data Setelah Menghapus Duplikasi**: Sekitar 1470 (jika tidak ada duplikasi)
  - **Outlier**: Diperiksa pada kolom numerik seperti `MonthlyIncome`.
  - ![image](https://github.com/user-attachments/assets/f6d41724-bfcf-44c8-bf8a-2a549099af90)

- **Tautan Sumber Data**: Dataset tersedia melalui [Link ke Dataset](https://github.com/dicodingacademy/dicoding_dataset/tree/main/employee)

### Uraian Fitur Dataset:

| Field                    | Description                                                                 |
|--------------------------|-----------------------------------------------------------------------------|
| EmployeeId               | Employee Identifier                                                         |
| Attrition                | Apakah karyawan keluar? (0=no, 1=yes)                                       |
| Age                      | Usia karyawan                                                               |
| BusinessTravel           | Tingkat perjalanan dinas                                                    |
| DailyRate                | Gaji harian                                                                 |
| Department               | Departemen tempat bekerja                                                   |
| DistanceFromHome         | Jarak rumah ke kantor (km)                                                  |
| Education                | 1-Below College, ..., 5-Doctor                                              |
| EducationField           | Bidang pendidikan                                                           |
| EnvironmentSatisfaction  | 1-Low, ..., 4-Very High                                                     |
| Gender                   | Jenis kelamin                                                               |
| HourlyRate               | Gaji per jam                                                                |
| JobInvolvement           | Keterlibatan kerja (1–4)                                                    |
| JobLevel                 | Level jabatan (1–5)                                                         |
| JobRole                  | Peran pekerjaan                                                             |
| JobSatisfaction          | Kepuasan kerja (1–4)                                                        |
| MaritalStatus            | Status pernikahan                                                           |
| MonthlyIncome            | Gaji bulanan                                                                |
| MonthlyRate              | Tarif bulanan                                                               |
| NumCompaniesWorked       | Jumlah perusahaan yang pernah dikerjakan                                   |
| Over18                   | Apakah lebih dari 18 tahun                                                  |
| OverTime                 | Apakah bekerja lembur                                                       |
| PercentSalaryHike        | Persentase kenaikan gaji tahun lalu                                         |
| PerformanceRating        | 1-Low, ..., 4-Outstanding                                                   |
| RelationshipSatisfaction | Kepuasan relasi sosial (1–4)                                                |
| StandardHours            | Jam kerja standar                                                           |
| StockOptionLevel         | Level opsi saham                                                            |
| TotalWorkingYears        | Total tahun pengalaman kerja                                                |
| TrainingTimesLastYear    | Jumlah pelatihan dalam setahun terakhir                                     |
| WorkLifeBalance          | Keseimbangan kerja dan hidup (1–4)                                          |
| YearsAtCompany           | Lama bekerja di perusahaan                                                  |
| YearsInCurrentRole       | Lama dalam peran saat ini                                                   |
| YearsSinceLastPromotion  | Lama sejak promosi terakhir                                                 |
| YearsWithCurrManager     | Lama bekerja dengan manajer saat ini                                        |
![download (17)](https://github.com/user-attachments/assets/825dcc39-9d68-4440-9543-84c645ad4e2a)
![download (16)](https://github.com/user-attachments/assets/5c82ab84-5cee-4446-9546-919bb5d3c449)

Deskripsi penjelasan : 
1. Data Overview

- Jumlah data: 1.470 baris, 35 kolom.
- Tipe data: Campuran numerik (int, float) dan kategorikal (object).
- Kolom target: Attrition (1 = keluar, 0 = tetap).
- Masalah: Terdapat 412 nilai hilang di kolom Attrition, sehingga hanya 1058 baris valid untuk analisis prediktif.

2. Missing Values
Kolom dengan missing value: Hanya Attrition (412 missing).

- Solusi ke depan: Perlu menangani missing value ini, misalnya dengan menghapus atau imputasi, tergantung tujuan.

3. Statistik Deskriptif
- Variabel seperti Age, MonthlyIncome, TotalWorkingYears, dll menunjukkan distribusi normal dengan rentang nilai bervariasi.

- Beberapa kolom seperti StandardHours dan EmployeeCount tidak bervariasi (semua nilainya sama), sehingga tidak informatif untuk analisis/prediksi.

4. Distribusi Variabel Penting
- Age: Mayoritas usia berada di rentang 30–45 tahun.

- Attrition: Terdistribusi tidak seimbang (lebih banyak karyawan tetap).

- MonthlyIncome: Distribusi tidak merata, terdapat outlier.

- Department: Mayoritas berasal dari Research & Development.

5. Korelasi terhadap Attrition
- Korelasi Attrition dengan variabel numerik umumnya lemah (semua korelasi < 0.2).

- Korelasi positif lemah:

- DistanceFromHome (0.078)

- NumCompaniesWorked (0.037)

- Korelasi negatif tertinggi:

- TotalWorkingYears: -0.177

- Age: -0.172

- JobLevel: -0.169

- MonthlyIncome: -0.164

- YearsInCurrentRole: -0.159

Artinya: Semakin tua, berpengalaman, dan tinggi jabatan/penghasilan, makin kecil kemungkinan keluar.

## 🧹 Data Preparation

1. Mengisi Nilai Kosong pada 'Attrition':

- Nilai kosong pada kolom Attrition diisi dengan modus (nilai terbanyak).

- Namun muncul peringatan (FutureWarning) karena metode inplace=True digunakan dengan chained assignment, yang di masa depan tidak akan didukung oleh pandas.

2. One-Hot Encoding:

- Semua kolom bertipe object (kategorikal) dikonversi menjadi kolom numerik menggunakan one-hot encoding.

- Parameter drop_first=True digunakan untuk menghindari dummy variable trap (mengurangi multikolinearitas).

3. Menghapus Duplikat:

- Jumlah baris duplikat yang ditemukan: 0

- Tidak ada baris yang dihapus karena tidak ada duplikasi.

📊 Hasil Data Setelah Preprocessing:

-- Jumlah Baris dan Kolom:
-  1470 baris, 48 kolom

--Kolom dengan Missing Values:
-  Tidak ada kolom yang memiliki nilai kosong (semua nol)

--Tipe Data:

- int64: 26 kolom

- float64: 1 kolom (Attrition)

- bool: 21 kolom (hasil one-hot encoding)

--Contoh Kolom One-Hot Encoding:

- BusinessTravel_Travel_Frequently, JobRole_Manager, OverTime_Yes, dll.

#Feature Engineering
1. Menambahkan Fitur Baru:

- experience_level:
- ➤ Kombinasi TotalWorkingYears * JobLevel → menggambarkan tingkat pengalaman.

- Income_JobLevel:
- ➤ MonthlyIncome * JobLevel → menilai pengaruh pendapatan dan jenjang kerja.

- YearsAtCompany_JobSatisfaction:
- ➤ Interaksi masa kerja dan kepuasan kerja.

- Age_WorkLifeBalance:
- ➤ Interaksi usia dan keseimbangan kerja-hidup.

2. Standarisasi (Scaling):

- Fitur numerik distandarisasi menggunakan StandardScaler agar memiliki mean = 0 dan std = 1.

- Kolom yang distandarisasi meliputi: Age, DailyRate, MonthlyIncome, YearsAtCompany, experience_level, dll.

3. Analisis Korelasi dengan 'Attrition':

- Mengukur kekuatan hubungan absolut antar fitur dan target (Attrition).

-- Hasil korelasi tertinggi (paling berpengaruh terhadap kemungkinan karyawan keluar):

- OverTime_Yes → 0.218

- MaritalStatus_Single → 0.164

- StockOptionLevel → 0.147

- TotalWorkingYears, Age, JobLevel, experience_level, dan fitur interaksi juga menunjukkan kontribusi positif dan signifikan.

4. Fitur Tidak Relevan (Korelasi Rendah):

-- Fitur dengan korelasi hampir nol atau tidak relevan:

- PerformanceRating, Gender_Male, PercentSalaryHike, dll.

- Bahkan ada yang mendekati 0.000: EducationField_Other, JobRole_Sales Executive.

- Fitur seperti EmployeeId, EmployeeCount, dan StandardHours memiliki nilai NaN (tidak valid untuk korelasi).

#Data Splitting
Memisahkan Fitur dan Target
Fitur (X): semua kolom kecuali Attrition

Target (y): kolom Attrition (status keluar atau tidak)

Membagi Data Menjadi Train dan Temp
70% data dimasukkan ke dalam set pelatihan (training set)

30% sisanya menjadi data sementara (temporary set)

Menggunakan teknik stratified sampling agar distribusi target tetap seimbang

Membagi Data Temp Menjadi Validation dan Test
Dari 30% data sementara, dibagi dua secara seimbang:

15% untuk validation set (digunakan saat tuning model)

15% untuk test set (digunakan untuk evaluasi akhir)

Hasil Pembagian Data
Data train (X_train): 1029 baris, 51 kolom

Data validasi (X_val): 220 baris, 51 kolom

Data test (X_test): 221 baris, 51 kolom

Label train (y_train): 1029 baris

Label validasi (y_val): 220 baris

Label test (y_test): 221 baris

## Modelling
⚙️ Proses Model Training (Linear Regression)
1. Inisialisasi Model

- Menggunakan model Linear Regression dari scikit-learn.

2. Pelatihan Model

- Model dilatih menggunakan data pelatihan (X_train, y_train).

3. Prediksi

- Setelah dilatih, model digunakan untuk memprediksi nilai y pada data validasi (X_val).

📊 Evaluasi Model
Mean Squared Error (MSE): 0.0918

1. Rata-rata kuadrat dari selisih antara nilai aktual dan prediksi.

- Semakin kecil, semakin baik.

2. Root Mean Squared Error (RMSE): 0.3030

- Akar kuadrat dari MSE, satuannya sama seperti target.

- Menunjukkan rata-rata jarak prediksi dari nilai sebenarnya.

3. R-squared (R²): 0.1472

- Mengukur seberapa baik variabel input menjelaskan variasi target.

- Nilai 0.147 artinya hanya sekitar 14.7% variabilitas yang bisa dijelaskan oleh model. Ini termasuk rendah.

4. Mean Absolute Error (MAE): 0.2081

- Rata-rata selisih absolut antara nilai aktual dan prediksi.

- Semakin kecil, semakin akurat prediksi.

#Model Optimization
1. Ridge Regression
Ridge adalah regresi linear dengan penalti L2 (mengurangi kompleksitas model).

- GridSearch digunakan untuk mencari alpha terbaik dari [0.001, 0.01, 0.1, 1, 10, 100].

-- Evaluasi menggunakan:

- MSE (Mean Squared Error)

- RMSE (Root Mean Squared Error)

- R² (R-squared)

- MAE (Mean Absolute Error)

📌 Hasil:

- Alpha terbaik dicetak.

- Kinerja model divalidasi menggunakan data X_val.

2. Lasso Regression
Lasso adalah regresi linear dengan penalti L1 (bisa mengecilkan beberapa koefisien jadi nol).

- Prosedur GridSearch dan evaluasi sama seperti Ridge.

📌 Hasil:

- Menampilkan alpha terbaik dan performa evaluasi.

3. Perbandingan Model
Jika R² Ridge > R² Lasso, maka Ridge dipilih sebagai model terbaik.

Jika sebaliknya, maka Lasso yang dipilih.

Dicetak juga nilai hyperparameter (alpha) dari model terbaik.

## 📈 Evaluation

Hasil Evaluasi
- MSE (Mean Squared Error): 0.0815
- → Rata-rata kesalahan kuadrat antara prediksi dan nilai aktual.

- RMSE (Root Mean Squared Error): 0.2855
- → Akar MSE, memudahkan interpretasi karena memiliki satuan yang sama dengan target.

- R² (R-squared): 0.2397
- → Model menjelaskan sekitar 24% variasi pada data uji. Nilai ini masih rendah → model belum ideal.

- MAE (Mean Absolute Error): 0.1901
- → Rata-rata kesalahan prediksi sekitar 0.19 unit (absolut).

Model evaluation : 
Evaluasi pada Data Uji:
MSE: 0.0815, RMSE: 0.2855, R2: 0.2397, MAE: 0.1901

Ringkasan Evaluasi Model:
Model Lasso Regression dengan alpha = 0.001 menghasilkan performa sebagai berikut pada data uji:
- R-squared (R2): 0.2397. Artinya, model mampu menjelaskan sebesar 23.97% variasi pada data uji.
- MSE: 0.0815. Rata-rata kuadrat selisih antara prediksi dan nilai sebenarnya.
- RMSE: 0.2855. Rata-rata selisih antara prediksi dan nilai aktual (dalam satuan yang sama).
- MAE: 0.1901. Rata-rata absolut dari kesalahan prediksi.

Interpretasi dan Saran Perbaikan:
Nilai R-squared menunjukkan bahwa kemampuan prediksi model masih terbatas. Performa model bisa ditingkatkan melalui rekayasa fitur, eksplorasi hubungan non-linear, atau menggunakan algoritma lain seperti regresi logistik atau pohon keputusan.
Dalam konteks bisnis seperti prediksi attrition (keluar-masuk karyawan), model yang lebih akurat sangat penting untuk mengidentifikasi dan mengurangi risiko turnover.
Nilai RMSE dan MAE memberikan gambaran seberapa besar kesalahan dalam prediksi. Model saat ini dapat dijadikan titik awal, namun diperlukan peningkatan agar hasilnya lebih bermanfaat secara praktis.

### 🎯 Evaluasi terhadap Problem Statement dan Goals
![image](https://github.com/user-attachments/assets/0fa43683-0f39-4d92-9dee-538c3205cf77)
![image](https://github.com/user-attachments/assets/d77fb7de-7942-44b8-8866-0f77c4b15937)
![image](https://github.com/user-attachments/assets/912189f6-a57d-4038-9464-8bd79a7a8e82)
![image](https://github.com/user-attachments/assets/e1c882d9-6216-4f19-88dc-665c53523c6d)
![image](https://github.com/user-attachments/assets/683c7e9a-f280-4f05-96c3-569a385e1148)
![image](https://github.com/user-attachments/assets/6c4d1091-ab99-4ec2-8390-436a488b0a07)
![image](https://github.com/user-attachments/assets/ac501087-8c4d-4290-8aa5-1a7a411a2b7f)

#### 📉 1. Distribusi Attrition
- **Deskripsi**: Grafik batang menunjukkan mayoritas karyawan tetap bekerja, namun persentase yang keluar (>10%) tergolong tinggi.
- **Kesimpulan**: Masalah *attrition* memang nyata dan perlu ditangani secara sistematis.

#### 🔑 2. Top 10 Faktor Penting dari Model Lasso
- **Deskripsi**: Grafik menunjukkan fitur paling berpengaruh terhadap attrition.
- **Hasil Penting**:
  - `OverTime`, `MonthlyIncome`, `Age`, `DistanceFromHome`, dan `JobLevel` menjadi fitur dominan.
- **Kesimpulan**: Karyawan muda, bergaji rendah, dan yang sering lembur lebih berisiko keluar.

#### 📦 3. Boxplot Faktor vs. Attrition
- **Insight**:
  - Karyawan lembur lebih sering keluar.
  - Gaji rendah dan jarak rumah jauh berkontribusi pada keputusan resign.
- **Kesimpulan**: Terdapat perbedaan distribusi signifikan pada fitur-fitur penting antar kelompok attrition.

#### 📊 4. Evaluasi Kinerja Model
- **Hasil Evaluasi**:
  - R² ≈ 0.24 → Model menjelaskan ~24% variansi data.
  - MAE dan RMSE cukup tinggi → Perlu perbaikan.
- **Kesimpulan**: Model regresi saat ini belum cukup akurat untuk prediksi individual, perlu eksplorasi model non-linear seperti Random Forest atau XGBoost.

#### 🏢 5. Attrition Berdasarkan Departemen
- **Insight**:
  - Departemen Sales dan HR menunjukkan attrition rate lebih tinggi.
- **Kesimpulan**: Evaluasi khusus diperlukan pada divisi-divisi dengan beban kerja tinggi atau kepuasan rendah.

#### 👥 6. Attrition Berdasarkan Usia
- **Insight**:
  - Karyawan muda lebih sering keluar.
- **Kesimpulan**: Stabilitas kerja lebih rendah pada kelompok usia muda, penting untuk pendekatan retensi yang disesuaikan.

#### 📅 7. Tren Bulanan (Simulasi)
- **Insight**:
  - Terdapat fluktuasi attrition bulanan (misalnya pasca-bonus tahunan).
- **Kesimpulan**: HR dapat menyusun strategi intervensi berbasis waktu.

---

### 💼 Dampak terhadap Business Understanding

- 🔍 **Identifikasi Masalah**: Proyek berhasil mengonfirmasi bahwa attrition >10% adalah masalah serius dan mengidentifikasi faktor-faktor utamanya.
- 📊 **Insight Berbasis Data**: Memberikan visualisasi dan analisis yang mudah dipahami oleh pemangku kepentingan non-teknis.
- 🧭 **Arah Kebijakan SDM**:
  - Penyusunan strategi retensi khusus untuk karyawan muda.
  - Peninjauan sistem lembur dan kompensasi.
  - Fokus pada kesejahteraan departemen dengan attrition tinggi.
- 🚀 **Manfaat Jangka Panjang**: 
  - Mengurangi biaya rekrutmen dan pelatihan.
  - Meningkatkan stabilitas tim dan produktivitas.
 
Gambar dashboard :
![dashboard](https://github.com/user-attachments/assets/e51063fe-8541-4325-b4e3-c68eaebb77bf)

Link dashboard :
[akses dashboard](https://lookerstudio.google.com/reporting/0edb5137-4d81-4b4e-97af-0856ec7bd25e)

