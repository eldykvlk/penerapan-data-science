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

Manajemen ingin memahami penyebab utama dari tingginya attrition rate. Dengan pemodelan prediktif dan dashboard analitik, proyek ini diharapkan mampu mengurangi attrition dan meningkatkan kepuasan kerja.

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
  - **Outlier**: Diperiksa pada kolom numerik seperti `MonthlyIncome`, `Age`, `DistanceFromHome`, dll.
- **Tautan Sumber Data**: Dataset tersedia melalui file `employee_data.csv`

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

## 🧹 Data Preparation

- Menghapus duplikasi dan nilai hilang
- Normalisasi dan scaling fitur numerik
- Encoding variabel kategorikal

## 📈 Evaluation

Model dievaluasi menggunakan:
- **Mean Squared Error (MSE)**
- **Mean Absolute Error (MAE)**
- **R² Score**

### 🎯 Evaluasi terhadap Problem Statement dan Goals

- Model berhasil mengidentifikasi fitur-fitur penting yang berpengaruh terhadap *attrition*
- Dapat digunakan sebagai alat bantu prediksi dan monitoring HR ke depan

### 💼 Dampak terhadap Business Understanding

- Manajemen dapat mengambil keputusan berbasis data
- Efisiensi dalam mempertahankan karyawan potensial
- Penurunan biaya rekrutmen dan pelatihan akibat attrition tinggi
"""
