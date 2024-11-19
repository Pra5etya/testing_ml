1. StandardScaler: 
    * Model: Regresi Linear, Support Vector Machines (SVM), K-Nearest Neighbors (KNN), Neural Networks.
    * Alasan: Model ini sensitif terhadap skala fitur. Penggunaan StandardScaler membantu untuk memastikan bahwa setiap fitur memiliki distribusi normal dengan mean 0 dan variance 1, yang mempercepat konvergensi dalam optimasi model.

2. MinMaxScaler
    * Model: Jaringan Saraf, K-Nearest Neighbors (KNN), Decision Trees, dan Random Forests.
    * Alasan: MinMaxScaler mengubah skala fitur ke rentang [0, 1], yang berguna untuk algoritma yang mengasumsikan bahwa fitur memiliki rentang serupa.

3. LabelEncoder
    * Model: Decision Trees, Random Forests, Gradient Boosting, dan Support Vector Machines (SVM).
    Alasan: Digunakan untuk mengonversi label kategori menjadi angka. Ini membantu model dalam memproses data kategorikal dengan efisien.

4. OneHotEncoder
    * Model: Regresi Logistik, K-Nearest Neighbors (KNN), Decision Trees, Random Forests, dan Neural Networks.
    * Alasan: Membantu mengonversi variabel kategorikal menjadi representasi yang dapat dimengerti oleh model, sehingga tidak ada hubungan ordinal yang salah dalam pengkodean.

5. SimpleImputer
    * Model: Hampir semua model machine learning, termasuk Regresi, Decision Trees, dan Neural Networks.
    * Alasan: Menangani nilai yang hilang dalam dataset. Mengimputasi nilai yang hilang dapat meningkatkan performa model dan mencegah kesalahan.

6. Polynomial Features
    * Model: Regresi Polinomial, Jaringan Saraf.
    * Alasan: Menambahkan fitur polinomial dapat membantu dalam menangkap hubungan non-linear dalam data, yang bermanfaat untuk model yang lebih kompleks.

7. Binarizer
    * Model: Logistic Regression, Support Vector Machines (SVM), Decision Trees.
    * Alasan: Membuat fitur biner bisa membantu model untuk melakukan pemisahan yang lebih baik dalam klasifikasi.

8. Function Transformer
    * Model: Regresi, Klasifikasi.
    * Alasan: Menyediakan cara untuk menerapkan transformasi kustom yang dapat meningkatkan kemampuan model dalam memahami data.

9. Normalizer
    * Model: K-Nearest Neighbors (KNN), Neural Networks, Support Vector Machines (SVM).
    * Alasan: Memastikan bahwa setiap sampel memiliki norma 1 (atau norma lainnya), membantu model untuk berfokus pada arah dan bukan magnitudo.