# Rangkuman

Berikut adalah metode deteksi anomali yang dirangkum dari teks:

## **Summary:**

1. **Metode Statistik** cocok untuk dataset kecil hingga menengah dengan asumsi distribusi data tertentu seperti normal atau Gaussian.
2. **Metode Probabilistik** ideal untuk dataset dengan pola distribusi kompleks atau hubungan antar variabel.
3. **Metode Kedekatan** efektif untuk mendeteksi anomali berdasarkan jarak atau kepadatan lokal, cocok untuk dataset dengan cluster atau pola kepadatan berbeda.
4. **Metode Machine Learning** lebih fleksibel, mendukung dataset besar dan berdimensi tinggi tanpa memerlukan asumsi distribusi tertentu.
5. **Metode Deep Learning** unggul dalam mendeteksi anomali pada data yang kompleks, seperti gambar, teks, atau data berdimensi tinggi.
6. Pilih metode berdasarkan ukuran dataset, jumlah fitur, distribusi data, dan kecepatan yang diinginkan untuk hasil optimal.

## Metode Statistik
- **Z-Score**: Digunakan pada data berdistribusi normal untuk dataset kecil-menengah.
- **IQR**: Cocok untuk data tidak berdistribusi normal, mendeteksi outlier pada dataset kecil-menengah.
- **Grubbs' Test**: Dikhususkan untuk data univariat kecil dengan uji hipotesis.
- **Elliptic Envelope**: Untuk data multivariat dengan asumsi Gaussian.

## Metode Probabilistik
- **Gaussian Mixture Model (GMM)**: Untuk dataset menengah-besar dengan distribusi Gaussian campuran.
- **Bayesian Networks**: Memodelkan hubungan antar variabel pada dataset kecil-menengah.
- **Hidden Markov Model (HMM)**: Cocok untuk data sekuensial atau berbasis waktu.
- **Kernel Density Estimation (KDE)**: Digunakan saat distribusi data tidak diketahui.

## Metode Kedekatan (Proximity)
- **KNN**: Anomali diidentifikasi berdasarkan jarak ke tetangga.
- **DBSCAN**: Mengelompokkan data berdasarkan kepadatan dan mengidentifikasi outlier.
- **LOF**: Deteksi anomali berdasarkan perbedaan kepadatan lokal.
- **OPTICS**: Untuk dataset besar dengan cluster kepadatan berbeda.

## Metode Machine Learning
- **Isolation Forest**: Menggunakan pohon untuk memisahkan anomali pada dataset besar.
- **One-Class SVM**: Untuk dataset kecil-menengah dengan distribusi konsisten.
- **Random Cut Forest (RCF)**: Membagi data acak untuk deteksi outlier pada dataset besar.

## Metode Deep Learning
- **Variational Autoencoder (VAE)**: Deteksi anomali berdasarkan rekonstruksi error, cocok untuk data kompleks seperti gambar atau teks.

## Cocok untuk Tipe Data
- **Univariat**: Z-Score, IQR, Grubbs' Test.
- **Multivariat**: GMM, Elliptic Envelope, Bayesian Networks.
- **Data Sequential atau Time Series**: HMM, LSTM (untuk pengembangan lebih lanjut).
- **Data Tidak Berdistribusi Normal**: IQR, KDE, LOF.
- **Dataset dengan Pola Kompleks**: Isolation Forest, RCF, VAE.
