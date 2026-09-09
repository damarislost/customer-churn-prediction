# 🏦 Prediksi Churn Nasabah di Industri Perbankan

Machine Learning project untuk memprediksi **customer churn pada industri perbankan** dengan membandingkan beberapa algoritma klasifikasi, melakukan feature engineering, feature selection menggunakan **ANOVA dan Mutual Information**, serta hyperparameter tuning menggunakan **GridSearchCV**.

Model terbaik yang diperoleh adalah **Gradient Boosting** dengan **accuracy 80.8%, precision 85%, dan recall 81%**.

---

## 📌 Project Overview

Customer churn merupakan kondisi ketika nasabah berhenti menggunakan layanan atau produk dari suatu bank.

Dalam industri perbankan yang kompetitif, mempertahankan nasabah merupakan salah satu faktor penting bagi keberlanjutan bisnis. Tingkat churn yang tinggi dapat menjadi indikasi adanya masalah pada layanan, produk, maupun hubungan antara bank dengan nasabah.

Nasabah dapat berpindah ke bank lain karena berbagai alasan, seperti ketidakpuasan terhadap layanan atau adanya penawaran yang dianggap lebih menarik dari kompetitor.

Oleh karena itu, kemampuan untuk memprediksi kemungkinan seorang nasabah melakukan churn dapat membantu bank mengambil tindakan lebih awal dalam mempertahankan nasabah.

Project ini berfokus pada pembangunan dan evaluasi beberapa model Machine Learning untuk melakukan klasifikasi nasabah berdasarkan kemungkinan churn.

---

## 🎯 Objectives

Tujuan project ini adalah:

- Mengidentifikasi karakteristik nasabah yang berpotensi melakukan churn.
- Melakukan Exploratory Data Analysis (EDA).
- Melakukan data preprocessing.
- Melakukan encoding terhadap variabel kategorikal.
- Membuat fitur baru melalui feature engineering.
- Menganalisis kontribusi fitur menggunakan ANOVA.
- Menganalisis kontribusi fitur menggunakan Mutual Information.
- Melakukan feature selection.
- Melakukan hyperparameter tuning menggunakan GridSearchCV.
- Membandingkan performa beberapa algoritma Machine Learning.
- Menentukan model terbaik untuk memprediksi churn nasabah.

---

## 📊 Dataset

Dataset yang digunakan berasal dari Kaggle:

### Predicting Churn for Bank Customers

🔗 https://www.kaggle.com/datasets/adammaus/predicting-churn-for-bank-customers

Dataset terdiri dari **10.000 data nasabah**.

Beberapa variabel yang terdapat pada dataset antara lain:

| Feature | Description |
|---|---|
| `RowNumber` | Nomor baris data |
| `CustomerId` | ID unik nasabah |
| `Surname` | Nama belakang nasabah |
| `CreditScore` | Skor kredit nasabah |
| `Geography` | Negara asal nasabah |
| `Gender` | Jenis kelamin nasabah |
| `Age` | Usia nasabah |
| `Tenure` | Lama menjadi nasabah |
| `Balance` | Saldo rekening |
| `NumOfProducts` | Jumlah produk bank yang digunakan |
| `HasCrCard` | Kepemilikan kartu kredit |
| `IsActiveMember` | Status keaktifan nasabah |
| `EstimatedSalary` | Estimasi pendapatan nasabah |
| `Exited` | Status churn nasabah |

---

## 🎯 Target Variable

Target yang digunakan dalam proses klasifikasi adalah:

```text
Exited
```

Dengan interpretasi:

```text
0 = Nasabah tetap / Loyal
1 = Nasabah Churn
```

---

# 🔍 Exploratory Data Analysis

## 1. Proporsi Nasabah Churn dan Loyal

Hasil eksplorasi data menunjukkan bahwa:

- **79.6% nasabah tetap loyal**
- **20.4% nasabah melakukan churn**

Dengan demikian, mayoritas nasabah pada dataset tetap menggunakan layanan bank.

```text
Loyal  : 79.6%
Churn  : 20.4%
```

---

## 2. Geography

Nasabah dalam dataset berasal dari tiga negara:

```text
France
Germany
Spain
```

Hasil eksplorasi menunjukkan adanya perbedaan jumlah churn berdasarkan wilayah.

Nasabah dari **France memiliki jumlah churn lebih sedikit dibandingkan Spain dan Germany**.

Hal ini menunjukkan bahwa faktor geografis memiliki hubungan dengan pola churn nasabah.

---

## 3. Gender

Berdasarkan visualisasi data, baik nasabah laki-laki maupun perempuan sebagian besar tetap menggunakan layanan bank.

Namun terdapat perbedaan tingkat churn berdasarkan gender.

Variabel `Gender` kemudian dianalisis lebih lanjut pada proses feature selection.

---

## 4. Credit Card Ownership

Sebagian besar nasabah, baik yang memiliki kartu kredit maupun yang tidak memiliki kartu kredit, tetap loyal.

Namun terdapat perbedaan jumlah churn berdasarkan kepemilikan kartu kredit.

Variabel ini kemudian diuji kembali menggunakan metode feature selection untuk mengetahui seberapa besar kontribusinya terhadap target churn.

---

## 5. Active Member

Status keaktifan nasabah menunjukkan pola yang cukup jelas.

Nasabah yang **tidak aktif memiliki tingkat churn lebih tinggi dibandingkan dengan nasabah yang aktif**.

Hal ini menunjukkan bahwa tingkat engagement atau keaktifan nasabah berpotensi menjadi salah satu faktor penting dalam customer retention.

---

## 6. Credit Score

Nasabah yang churn memiliki skor kredit yang sedikit lebih rendah dibandingkan dengan nasabah loyal.

Namun distribusi `CreditScore` antara kedua kelompok relatif serupa.

Hal ini menunjukkan bahwa:

```text
CreditScore bukan merupakan faktor dominan
dalam membedakan nasabah churn dan loyal.
```

---

## 7. Age

Variabel `Age` menunjukkan perbedaan yang cukup jelas.

Nasabah yang berusia lebih tua memiliki kecenderungan churn yang lebih tinggi dibandingkan nasabah yang lebih muda.

Dengan demikian, usia menjadi salah satu variabel yang berpotensi memiliki hubungan kuat terhadap churn.

---

## 8. Tenure

Distribusi `Tenure` antara nasabah loyal dan churn relatif mirip.

Tidak terlihat perbedaan yang signifikan antara kedua kelompok.

Hal ini mengindikasikan bahwa lama menjadi nasabah bukan merupakan faktor utama dalam menentukan churn.

---

## 9. Balance

Nasabah dengan saldo yang lebih tinggi menunjukkan kecenderungan churn yang lebih besar dibandingkan sebagian nasabah dengan saldo rendah.

Variabilitas balance pada kelompok nasabah churn juga terlihat lebih besar dibandingkan nasabah loyal.

---

## 10. NumOfProducts

Distribusi jumlah produk yang digunakan oleh nasabah tidak menunjukkan perbedaan yang sangat jelas antara nasabah churn dan loyal berdasarkan visualisasi awal.

Namun pada analisis **Mutual Information**, `NumOfProducts` menjadi fitur dengan MI Score tertinggi.

---

## 11. Estimated Salary

Distribusi `EstimatedSalary` antara nasabah churn dan loyal relatif serupa.

Hasil feature selection kemudian menunjukkan bahwa fitur ini memiliki kontribusi yang relatif rendah terhadap target.

---

# 🧹 Data Preparation

Beberapa kolom dihapus karena hanya berfungsi sebagai identifier dan tidak memberikan informasi yang relevan untuk proses prediksi.

Kolom yang dihapus:

```text
RowNumber
CustomerId
Surname
```

Setelah kolom tersebut dihapus, data yang digunakan berisi fitur-fitur yang berkaitan dengan karakteristik dan aktivitas nasabah.

---

# 🔢 Encoding Categorical Features

Karena algoritma Machine Learning membutuhkan data dalam bentuk numerik, variabel kategorikal perlu dikonversi.

---

## Label Encoding

Variabel `Gender` dikonversi menjadi data numerik.

```text
Female → 0
Male   → 1
```

Contoh:

```text
Gender
------
0
0
0
0
0
```

---

## Dummy Variable

Variabel `Geography` memiliki tiga kategori:

```text
France
Germany
Spain
```

Karena ketiga kategori tersebut tidak memiliki tingkatan atau urutan, digunakan **Dummy Variable Encoding**.

Awalnya Geography diubah menjadi:

```text
France
Germany
Spain
```

Kemudian salah satu kategori dihapus untuk menghindari redundant information.

Dalam project ini, `France` digunakan sebagai baseline.

Sehingga hanya digunakan:

```text
Germany
Spain
```

Interpretasinya:

```text
Germany = 0, Spain = 0 → France
Germany = 1, Spain = 0 → Germany
Germany = 0, Spain = 1 → Spain
```

---

# 🛠 Feature Engineering

Selain menggunakan variabel yang sudah tersedia pada dataset, dilakukan penambahan fitur baru.

---

## Credit Age Ratio

Fitur `CreditAgeRatio` merupakan rasio antara Credit Score dan usia nasabah.

Secara sederhana:

```text
CreditAgeRatio = CreditScore / Age
```

Contoh nilai:

```text
CreditScore = 619
Age         = 42

CreditAgeRatio ≈ 14.74
```

Fitur ini digunakan untuk menangkap hubungan antara usia dengan skor kredit nasabah.

---

## Balance Salary Ratio

Selain `CreditAgeRatio`, terdapat fitur:

```text
BalanceSalaryRatio
```

Fitur ini digunakan dalam proses feature selection untuk mengevaluasi hubungan antara saldo nasabah dengan estimasi pendapatannya.

---

# 🧪 Feature Selection

Feature selection dilakukan untuk mengetahui fitur-fitur yang memiliki kontribusi lebih besar terhadap target `Exited`.

Dua pendekatan digunakan:

1. **ANOVA (Analysis of Variance)**
2. **Mutual Information**

---

# 1️⃣ ANOVA

ANOVA digunakan untuk mengevaluasi hubungan antara masing-masing fitur dengan target churn.

Implementasi:

```python
features = data.drop(['Exited'], axis=1)
target = data['Exited']

f_values, p_values = f_classif(features, target)

scores_df = pd.DataFrame({
    'Feature': features.columns,
    'F-Value': f_values,
    'P-Value': p_values
})
```

Threshold yang digunakan:

```python
p_value_threshold = 0.05
```

Fitur dengan:

```text
P-Value < 0.05
```

dipertahankan.

---

## ANOVA Result

| Feature | F-Value | P-Value |
|---|---:|---:|
| CreditScore | 7.344522 | 6.738214e-03 |
| Gender | 114.727989 | 1.258505e-26 |
| Age | 886.063275 | 1.239931e-186 |
| Tenure | 1.960164 | 1.615268e-01 |
| Balance | 142.473832 | 1.275563e-32 |
| NumOfProducts | 22.915223 | 1.717333e-06 |
| HasCrCard | 0.509401 | 4.754149e-01 |
| IsActiveMember | 249.800794 | 1.348269e-55 |
| EstimatedSalary | 1.463262 | 2.264404e-01 |
| CreditAgeRatio | 686.250857 | 2.352796e-146 |
| BalanceSalaryRatio | 6.535096 | 1.059130e-02 |
| Germany | 310.258384 | 2.059537e-68 |
| Spain | 27.809468 | 1.366655e-07 |

---

## Features Selected by ANOVA

```text
CreditScore
Gender
Age
Balance
NumOfProducts
IsActiveMember
CreditAgeRatio
BalanceSalaryRatio
Germany
Spain
```

---

## Features Removed by ANOVA

```text
Tenure
HasCrCard
EstimatedSalary
```

---

# 2️⃣ Mutual Information

Mutual Information digunakan untuk mengukur seberapa banyak informasi suatu fitur dapat memberikan terhadap target.

Implementasi:

```python
features = data.drop(['Exited'], axis=1)
target = data['Exited']

mi_scores = mutual_info_classif(
    features,
    target,
    discrete_features='auto',
    random_state=0
)

mi_scores_df = pd.DataFrame({
    'Feature': features.columns,
    'MI Score': mi_scores
}).sort_values(
    by='MI Score',
    ascending=False
)
```

---

## Mutual Information Result

| Feature | MI Score |
|---|---:|
| NumOfProducts | 0.072248 |
| Age | 0.067552 |
| CreditAgeRatio | 0.046311 |
| IsActiveMember | 0.018648 |
| Germany | 0.015234 |
| BalanceSalaryRatio | 0.009443 |
| Balance | 0.008394 |
| Gender | 0.007739 |
| Spain | 0.003949 |
| EstimatedSalary | 0.002704 |
| HasCrCard | 0.000525 |
| CreditScore | 0.000000 |
| Tenure | 0.000000 |

Threshold yang digunakan:

```python
mi_threshold = 0.003
```

---

## Features Selected by Mutual Information

```text
NumOfProducts
Age
CreditAgeRatio
IsActiveMember
Germany
BalanceSalaryRatio
Balance
Gender
Spain
```

---

## Features Removed by Mutual Information

```text
EstimatedSalary
HasCrCard
CreditScore
Tenure
```

---

# ⚙️ Hyperparameter Tuning

Untuk mendapatkan parameter terbaik dari setiap algoritma digunakan:

## GridSearchCV

GridSearchCV merupakan teknik pencarian hyperparameter yang melakukan pengujian terhadap kombinasi parameter yang telah ditentukan.

Tujuannya adalah menemukan konfigurasi parameter yang menghasilkan performa terbaik untuk masing-masing model.

Model yang diuji:

```text
Gradient Boosting
Random Forest
Support Vector Machine
Decision Tree
```

---

# 🤖 Machine Learning Models

Empat algoritma klasifikasi dibandingkan dalam project ini.

---

## 1. Gradient Boosting

Best Parameters:

```python
{
    "learning_rate": 0.1,
    "n_estimators": 100,
    "max_depth": 3
}
```

Hasil:

| Metric | Score |
|---|---:|
| Accuracy | **0.8080** |
| Precision | **0.85** |
| Recall | **0.81** |

---

## 2. Random Forest

Best Parameters:

```python
{
    "n_estimators": 100,
    "max_features": "sqrt",
    "max_depth": None,
    "min_samples_split": 2,
    "min_samples_leaf": 1,
    "bootstrap": True
}
```

Hasil:

| Metric | Score |
|---|---:|
| Accuracy | **0.7995** |
| Precision | **0.83** |
| Recall | **0.80** |

---

## 3. Decision Tree

Best Parameters:

```python
{
    "criterion": "gini",
    "splitter": "best",
    "max_depth": None,
    "min_samples_split": 2,
    "min_samples_leaf": 1,
    "max_features": None,
    "max_leaf_nodes": None
}
```

Hasil:

| Metric | Score |
|---|---:|
| Accuracy | **0.7505** |
| Precision | **0.81** |
| Recall | **0.75** |

---

## 4. Support Vector Machine

Best Parameters:

```python
{
    "C": 1.0,
    "kernel": "rbf",
    "degree": 3,
    "gamma": "scale",
    "coef0": 0.0
}
```

Hasil:

| Metric | Score |
|---|---:|
| Accuracy | **0.4675** |
| Precision | **0.75** |
| Recall | **0.47** |

---

# 📊 Model Comparison

| Model | Accuracy | Precision | Recall |
|---|---:|---:|---:|
| 🥇 **Gradient Boosting** | **80.80%** | **85%** | **81%** |
| 🥈 Random Forest | 79.95% | 83% | 80% |
| 🥉 Decision Tree | 75.05% | 81% | 75% |
| SVM | 46.75% | 75% | 47% |

---

# 🏆 Best Model

Model dengan performa terbaik adalah:

## Gradient Boosting

Dengan hasil:

```text
Accuracy  : 80.80%
Precision : 85%
Recall    : 81%
```

Gradient Boosting memberikan performa terbaik secara keseluruhan pada skenario pengujian yang dilakukan.

Random Forest berada di posisi kedua dengan performa yang hampir sebanding.

Decision Tree masih memberikan hasil yang cukup baik, tetapi berada di bawah Gradient Boosting dan Random Forest.

Sementara itu, SVM menghasilkan performa yang paling rendah pada skenario pengujian ini.

---

# 💡 Key Insights

Berdasarkan Exploratory Data Analysis dan feature selection, diperoleh beberapa insight penting.

### 1. Mayoritas nasabah tetap loyal

Sekitar:

```text
79.6% nasabah loyal
20.4% nasabah churn
```

Mayoritas nasabah masih menggunakan layanan bank.

---

### 2. Age memiliki hubungan kuat dengan churn

Nasabah yang lebih tua menunjukkan kecenderungan churn yang lebih tinggi.

Hal ini juga diperkuat dari hasil ANOVA, di mana `Age` memiliki F-Value yang tinggi.

```text
Age F-Value = 886.063275
```

---

### 3. NumOfProducts memiliki Mutual Information tertinggi

Dalam Mutual Information, fitur dengan MI Score terbesar adalah:

```text
NumOfProducts = 0.072248
```

Hal ini menunjukkan bahwa jumlah produk yang digunakan nasabah mengandung informasi yang cukup penting dalam membedakan churn dan loyal.

---

### 4. Keaktifan nasabah merupakan faktor penting

Nasabah yang tidak aktif memiliki kecenderungan churn lebih tinggi.

Fitur:

```text
IsActiveMember
```

juga memperoleh nilai signifikan pada ANOVA dan Mutual Information.

---

### 5. Geography berhubungan dengan churn

Nasabah dari France, Germany, dan Spain memiliki pola churn yang berbeda.

Variabel:

```text
Germany
Spain
```

tetap terpilih dalam ANOVA dan Mutual Information.

---

### 6. Balance memiliki hubungan dengan churn

Nasabah dengan balance yang lebih tinggi menunjukkan kecenderungan churn yang lebih besar pada eksplorasi data.

Fitur `Balance` juga dipertahankan dalam kedua metode feature selection.

---

### 7. CreditAgeRatio memberikan informasi tambahan

Feature engineering menghasilkan:

```text
CreditAgeRatio
```

yang mendapatkan hasil cukup tinggi baik pada ANOVA maupun Mutual Information.

ANOVA:

```text
F-Value = 686.250857
P-Value = 2.352796e-146
```

Mutual Information:

```text
MI Score = 0.046311
```

---

### 8. Beberapa fitur memiliki kontribusi rendah

Berdasarkan feature selection, beberapa fitur menunjukkan kontribusi yang relatif rendah.

Di antaranya:

```text
EstimatedSalary
Tenure
HasCrCard
```

Fitur tersebut dapat dipertimbangkan untuk dihapus agar model menjadi lebih sederhana dan efisien.

---

# 📌 Conclusion

Berdasarkan hasil penelitian, **Gradient Boosting dan Random Forest merupakan dua model dengan performa terbaik** untuk melakukan klasifikasi churn nasabah.

Gradient Boosting menghasilkan performa tertinggi dengan:

```text
Accuracy  : 80.8%
Precision : 85%
Recall    : 81%
```

Sedangkan Random Forest menghasilkan:

```text
Accuracy  : 79.95%
Precision : 83%
Recall    : 80%
```

Hasil feature selection menggunakan **ANOVA dan Mutual Information** juga menunjukkan bahwa tidak seluruh variabel memiliki kontribusi yang sama terhadap prediksi churn.

Beberapa fitur yang menunjukkan kontribusi penting antara lain:

```text
Age
NumOfProducts
IsActiveMember
Balance
Germany
Spain
CreditAgeRatio
BalanceSalaryRatio
Gender
```

Sementara beberapa fitur dengan kontribusi relatif rendah antara lain:

```text
EstimatedSalary
Tenure
HasCrCard
```

Penghapusan fitur dengan kontribusi rendah dapat membantu meningkatkan efisiensi model tanpa harus menggunakan seluruh fitur yang tersedia.

---

# 💼 Business Implication

Model prediksi churn dapat membantu industri perbankan untuk melakukan pendekatan yang lebih proaktif terhadap nasabah.

Beberapa implementasi yang dapat dilakukan antara lain:

- Mengidentifikasi nasabah yang memiliki risiko churn.
- Menentukan prioritas customer retention.
- Meningkatkan engagement terhadap nasabah yang kurang aktif.
- Memberikan penawaran atau program yang lebih sesuai dengan karakteristik nasabah.
- Mengoptimalkan strategi retensi nasabah.
- Mendukung pengambilan keputusan berbasis data.

Dengan mengetahui nasabah yang memiliki risiko churn lebih awal, bank dapat melakukan tindakan retensi sebelum nasabah benar-benar berhenti menggunakan layanan.

---

# 📈 Evaluation Metrics

Model dievaluasi menggunakan:

### Accuracy

Mengukur proporsi keseluruhan prediksi yang berhasil diklasifikasikan dengan benar.

### Precision

Mengukur proporsi prediksi positif yang benar-benar merupakan kelas positif.

### Recall

Mengukur kemampuan model dalam menemukan data yang benar-benar termasuk ke dalam kelas positif.

Dalam konteks project ini, evaluasi tersebut digunakan untuk membandingkan kemampuan masing-masing model dalam melakukan klasifikasi churn.

---

# 👥 Contributors

Project ini dikerjakan oleh:

### Damarjati Maulana Hilmi
### Ika Christine Purba
### Nurul Fadilah


---
