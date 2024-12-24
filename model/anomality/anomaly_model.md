## **Metode Statistik:**

1. **Z-Score**  
   - Mengukur standar deviasi data dari mean.

   **Kapan Digunakan:**
   - Dataset kecil hingga menengah.  
   - Data harus mengikuti distribusi normal.  
   - Cocok untuk deteksi anomali sederhana dalam data univariat.

   **Hyperparameter:**
   - `threshold`: Batas deviasi standar untuk mendeteksi anomali (default: 3).

   ```python
   import numpy as np
   from scipy.stats import zscore

   z_scores = zscore(data)
   threshold = 3  # Contoh threshold
   anomaly_indices = np.where(np.abs(z_scores) > threshold)

   # Filter data
   normal_data = data[np.abs(z_scores) <= threshold]
   anomaly_data = data[np.abs(z_scores) > threshold]

   print("Jumlah data normal:", normal_data.shape[0])
   print("Jumlah data anomali:", anomaly_data.shape[0])
   ```

2. **IQR (Interquartile Range)**  
   - Mengidentifikasi outlier berdasarkan Q1 dan Q3.

   **Kapan Digunakan:**
   - Dataset kecil hingga menengah.  
   - Data tidak perlu distribusi normal.  
   - Cocok jika Anda memiliki outlier yang jauh dari rentang kuartil.

   **Hyperparameter:**
   - `multiplier`: Faktor pengali untuk IQR (default: 1.5).

   ```python
   Q1 = data.quantile(0.25)
   Q3 = data.quantile(0.75)
   IQR = Q3 - Q1

   lower_bound = Q1 - 1.5 * IQR
   upper_bound = Q3 + 1.5 * IQR

   anomalies = data[(data < lower_bound) | (data > upper_bound)]
   normal_data = data[(data >= lower_bound) & (data <= upper_bound)]

   print("Jumlah data normal:", normal_data.shape[0])
   print("Jumlah data anomali:", anomalies.shape[0])
   ```

3. **Gaussian Mixture Model (GMM)**  
   - Memodelkan distribusi data menggunakan gabungan distribusi Gaussian.

   **Kapan Digunakan:**
   - Dataset menengah hingga besar.  
   - Ketika data mengikuti distribusi Gaussian campuran.  
   - Cocok untuk deteksi anomali di data multivariat dengan distribusi kompleks.

   **Hyperparameter:**
   - `n_components`: Jumlah distribusi Gaussian yang digunakan (default: 1).
   - `covariance_type`: Jenis kovarians (default: 'full').

   ```python
   from sklearn.mixture import GaussianMixture

   gmm = GaussianMixture(n_components=2, random_state=42)
   gmm.fit(X)

   scores = gmm.score_samples(X)
   threshold = np.percentile(scores, 5)  # Contoh threshold pada persentil ke-5
   anomalies = X[scores < threshold]
   normal_data = X[scores >= threshold]

   print("Jumlah data normal:", normal_data.shape[0])
   print("Jumlah data anomali:", anomalies.shape[0])
   ```

4. **Elliptic Envelope**  
   - Mendeteksi outlier menggunakan asumsi distribusi Gaussian.

   **Kapan Digunakan:**
   - Dataset kecil hingga menengah.  
   - Jika data mendekati distribusi Gaussian dan Anda ingin mendeteksi outlier berbasis multivariat.

   **Hyperparameter:**
   - `contamination`: Proporsi data yang diharapkan sebagai anomaly (default: 0.1).
   - `random_state`: Penentu hasil acak (default: None).

   ```python
   from sklearn.covariance import EllipticEnvelope

   model = EllipticEnvelope(contamination=0.1, random_state=42)
   model.fit(X)
   preds = model.predict(X)  # 1 = Normal, -1 = Anomaly

   # Filter data
   normal_data = X[preds == 1]
   anomaly_data = X[preds == -1]

   print("Jumlah data normal:", normal_data.shape[0])
   print("Jumlah data anomali:", anomaly_data.shape[0])
   ```

5. **Grubbs' Test**  
   - Mengidentifikasi outlier dalam data univariat.

   **Kapan Digunakan:**
   - Dataset kecil.  
   - Data univariat.  
   - Digunakan untuk uji hipotesis apakah nilai tertentu merupakan outlier.

   **Hyperparameter:**
   - `alpha`: Tingkat signifikansi (default: 0.05).

   ```python
   from outliers import smirnov_grubbs as grubbs

   anomalies = grubbs.test(X, alpha=0.05)
   normal_data = [x for x in X if x not in anomalies]

   print("Jumlah data normal:", len(normal_data))
   print("Jumlah data anomali:", len(anomalies))
   ```

## **Metode Probabilistik:**

1. **Gaussian Mixture Model (GMM)**  
   - Memodelkan distribusi data menggunakan gabungan distribusi Gaussian.

   **Kapan Digunakan:**
   - Dataset menengah hingga besar.  
   - Ketika data mengikuti distribusi Gaussian campuran.  
   - Cocok untuk deteksi anomali di data multivariat dengan distribusi kompleks.

   **Hyperparameter:**
   - `n_components`: Jumlah distribusi Gaussian yang digunakan (default: 1).
   - `covariance_type`: Jenis kovarians (default: 'full').

   ```python
   from sklearn.mixture import GaussianMixture

   gmm = GaussianMixture(n_components=2, random_state=42)
   gmm.fit(X)

   scores = gmm.score_samples(X)
   threshold = np.percentile(scores, 5)  # Contoh threshold pada persentil ke-5
   anomalies = X[scores < threshold]
   normal_data = X[scores >= threshold]

   print("Jumlah data normal:", normal_data.shape[0])
   print("Jumlah data anomali:", anomalies.shape[0])
   ```

2. **Bayesian Networks**  
   - Menggunakan struktur probabilistik untuk memodelkan ketergantungan antar variabel.

   **Kapan Digunakan:**
   - Dataset kecil hingga menengah.  
   - Ketika terdapat ketergantungan yang signifikan antar variabel.  
   - Cocok untuk data multivariat dengan hubungan yang kompleks.

   **Hyperparameter:**
   - `scoring_method`: Metode untuk mengevaluasi struktur jaringan (default: 'BIC').
   - `max_iter`: Iterasi maksimum untuk pembelajaran parameter (default: 100).

   ```python
   from pgmpy.models import BayesianNetwork
   from pgmpy.estimators import HillClimbSearch, BicScore

   model = BayesianNetwork([('A', 'B'), ('B', 'C')])
   model.fit(data)

   anomalies = []
   for idx, row in data.iterrows():
       likelihood = model.score(row)
       if likelihood < threshold:  # Contoh threshold likelihood rendah
           anomalies.append(idx)

   normal_data = data.drop(index=anomalies)
   anomaly_data = data.loc[anomalies]

   print("Jumlah data normal:", normal_data.shape[0])
   print("Jumlah data anomali:", anomaly_data.shape[0])
   ```

3. **Hidden Markov Model (HMM)**  
   - Menganalisis data sekuensial dengan asumsi probabilistik terhadap status tersembunyi.

   **Kapan Digunakan:**
   - Data bersifat sekuensial atau berbasis waktu.  
   - Cocok untuk mendeteksi pola yang tidak terlihat secara langsung.

   **Hyperparameter:**
   - `n_components`: Jumlah status tersembunyi (default: 1).
   - `covariance_type`: Jenis kovarians (default: 'diag').

   ```python
   from hmmlearn.hmm import GaussianHMM

   model = GaussianHMM(n_components=2, covariance_type='diag', random_state=42)
   model.fit(X)

   scores = model.score_samples(X)
   threshold = np.percentile(scores, 5)  # Contoh threshold pada persentil ke-5
   anomalies = X[scores < threshold]
   normal_data = X[scores >= threshold]

   print("Jumlah data normal:", normal_data.shape[0])
   print("Jumlah data anomali:", anomalies.shape[0])
   ```

4. **Kernel Density Estimation (KDE)**  
   - Memodelkan distribusi data tanpa asumsi bentuk tertentu.

   **Kapan Digunakan:**
   - Dataset kecil hingga menengah.  
   - Ketika distribusi data tidak diketahui.  
   - Cocok untuk deteksi anomali berbasis probabilistik tanpa asumsi distribusi tertentu.

   **Hyperparameter:**
   - `bandwidth`: Lebar kernel yang digunakan (default: otomatis).

   ```python
   from sklearn.neighbors import KernelDensity

   kde = KernelDensity(kernel='gaussian', bandwidth=1.0).fit(X)
   scores = kde.score_samples(X)
   threshold = np.percentile(scores, 5)  # Contoh threshold pada persentil ke-5
   anomalies = X[scores < threshold]
   normal_data = X[scores >= threshold]

   print("Jumlah data normal:", normal_data.shape[0])
   print("Jumlah data anomali:", anomalies.shape[0])
   ```

## **Metode Kedekatan / Proximity:**

1. **K-Nearest Neighbors (KNN) untuk Anomali**  
   - Mengidentifikasi anomali berdasarkan jarak ke tetangga terdekat.

   **Kapan Digunakan:**
   - Dataset kecil hingga menengah.  
   - Ketika distribusi data tidak diketahui dan jarak antar data signifikan.  
   - Cocok untuk data multivariat dengan pola lokal yang spesifik.

   **Hyperparameter:**
   - `n_neighbors`: Jumlah tetangga yang digunakan (default: 5).  
   - `algorithm`: Algoritma pencarian tetangga (default: 'auto').
   - `metric`: Metode perhitungan jarak (default: 'minkowski').

   ```python
   from sklearn.neighbors import NearestNeighbors
   import numpy as np

   knn = NearestNeighbors(n_neighbors=5, algorithm='auto')
   knn.fit(X)
   distances, _ = knn.kneighbors(X)

   # Menghitung threshold
   threshold = np.percentile(distances[:, -1], 95)  # Threshold persentil ke-95
   anomalies = X[distances[:, -1] > threshold]
   normal_data = X[distances[:, -1] <= threshold]

   print("Jumlah data normal:", normal_data.shape[0])
   print("Jumlah data anomali:", anomalies.shape[0])
   ```

2. **DBSCAN (Density-Based Spatial Clustering of Applications with Noise)**  
   - Mengidentifikasi anomali sebagai data yang tidak termasuk dalam cluster.

   **Kapan Digunakan:**
   - Dataset kecil hingga besar.  
   - Ketika data memiliki cluster dengan kepadatan berbeda.  
   - Cocok untuk data non-linear dan berdimensi tinggi.

   **Hyperparameter:**
   - `eps`: Jarak maksimum antara dua sampel untuk dianggap satu cluster (default: 0.5).  
   - `min_samples`: Jumlah sampel minimum untuk membentuk cluster (default: 5).

   ```python
   from sklearn.cluster import DBSCAN

   dbscan = DBSCAN(eps=0.5, min_samples=5)
   labels = dbscan.fit_predict(X)

   anomalies = X[labels == -1]  # Data yang tidak termasuk dalam cluster
   normal_data = X[labels != -1]

   print("Jumlah data normal:", normal_data.shape[0])
   print("Jumlah data anomali:", anomalies.shape[0])
   ```

3. **LOF (Local Outlier Factor)**  
   - Mengukur kepadatan lokal suatu data dibandingkan dengan tetangganya.

   **Kapan Digunakan:**
   - Dataset kecil hingga menengah.  
   - Ketika kepadatan lokal berbeda signifikan.  
   - Cocok untuk mendeteksi anomali di area dengan kepadatan berbeda.

   **Hyperparameter:**
   - `n_neighbors`: Jumlah tetangga untuk menghitung faktor kepadatan lokal (default: 20).

   ```python
   from sklearn.neighbors import LocalOutlierFactor

   lof = LocalOutlierFactor(n_neighbors=20, contamination=0.1)
   preds = lof.fit_predict(X)  # 1 = Normal, -1 = Anomaly

   normal_data = X[preds == 1]
   anomalies = X[preds == -1]

   print("Jumlah data normal:", normal_data.shape[0])
   print("Jumlah data anomali:", anomalies.shape[0])
   ```

4. **OPTICS (Ordering Points To Identify the Clustering Structure)**  
   - Memperluas DBSCAN untuk mendeteksi cluster dengan kepadatan bervariasi.

   **Kapan Digunakan:**
   - Dataset besar.  
   - Ketika cluster memiliki kepadatan berbeda-beda.  
   - Cocok untuk data non-linear.

   **Hyperparameter:**
   - `min_samples`: Jumlah sampel minimum dalam satu cluster (default: 5).  
   - `max_eps`: Jarak maksimum antara dua sampel untuk satu cluster (default: np.inf).

   ```python
   from sklearn.cluster import OPTICS

   optics = OPTICS(min_samples=5, max_eps=np.inf)
   labels = optics.fit_predict(X)

   anomalies = X[labels == -1]  # Data yang tidak termasuk dalam cluster
   normal_data = X[labels != -1]

   print("Jumlah data normal:", normal_data.shape[0])
   print("Jumlah data anomali:", anomalies.shape[0])
   ```

## **Metode Machine Learning:**

1. **Isolation Forest**  
   - Menggunakan pohon keputusan untuk memisahkan data normal dan anomali.

   **Kapan Digunakan:**
   - Dataset besar dengan banyak fitur.
   - Ketika data memiliki distribusi yang tidak teratur.
   - Cocok untuk deteksi anomali berbasis distribusi secara umum.

   **Hyperparameter:**
   - `n_estimators`: Jumlah pohon di hutan (default: 100).  
   - `contamination`: Proporsi data yang diharapkan sebagai anomali (default: auto).  
   - `max_samples`: Jumlah maksimum sampel untuk setiap pohon (default: auto).

   ```python
   from sklearn.ensemble import IsolationForest

   model = IsolationForest(n_estimators=100, contamination=0.1, random_state=42)
   model.fit(X)
   preds = model.predict(X)  # 1 = Normal, -1 = Anomaly

   # Filter data
   normal_data = X[preds == 1]
   anomaly_data = X[preds == -1]

   print("Jumlah data normal:", normal_data.shape[0])
   print("Jumlah data anomali:", anomaly_data.shape[0])
   ```

2. **One-Class SVM (Support Vector Machine)**  
   - Memisahkan data normal dari anomali dengan menemukan hyperplane di ruang fitur.

   **Kapan Digunakan:**
   - Dataset kecil hingga menengah.
   - Ketika distribusi data normal cukup konsisten.
   - Cocok untuk data berdimensi tinggi.

   **Hyperparameter:**
   - `kernel`: Kernel yang digunakan dalam algoritma (default: 'rbf').  
   - `nu`: Upper bound untuk rasio anomali (default: 0.5).  
   - `gamma`: Parameter kernel (default: 'scale').

   ```python
   from sklearn.svm import OneClassSVM

   model = OneClassSVM(kernel='rbf', gamma='scale', nu=0.1)
   model.fit(X)
   preds = model.predict(X)  # 1 = Normal, -1 = Anomaly

   # Filter data
   normal_data = X[preds == 1]
   anomaly_data = X[preds == -1]

   print("Jumlah data normal:", normal_data.shape[0])
   print("Jumlah data anomali:", anomaly_data.shape[0])
   ```

3. **Random Cut Forest (RCF)**  
   - Membagi data menjadi "cuts" acak untuk mendeteksi outlier berdasarkan distribusi.

   **Kapan Digunakan:**
   - Dataset besar dengan banyak fitur.
   - Ketika distribusi data sulit didefinisikan.

   **Hyperparameter:**
   - `n_estimators`: Jumlah pohon dalam hutan (default: 100).  
   - `max_samples`: Jumlah sampel maksimum untuk membangun pohon (default: 256).

   ```python
   from sklearn.ensemble import IsolationForest  # Proxy untuk Random Cut Forest

   model = IsolationForest(n_estimators=100, max_samples=256, random_state=42)
   model.fit(X)
   preds = model.predict(X)  # 1 = Normal, -1 = Anomaly

   # Filter data
   normal_data = X[preds == 1]
   anomaly_data = X[preds == -1]

   print("Jumlah data normal:", normal_data.shape[0])
   print("Jumlah data anomali:", anomaly_data.shape[0])
   ```

4. **Extreme Learning Machine (ELM)**  
   - Jaringan saraf dengan satu lapisan tersembunyi yang cepat untuk deteksi anomali.

   **Kapan Digunakan:**
   - Dataset kecil hingga menengah.  
   - Ketika kecepatan eksekusi menjadi prioritas.

   **Hyperparameter:**
   - `n_hidden`: Jumlah neuron di lapisan tersembunyi.  
   - `activation_function`: Fungsi aktivasi yang digunakan.

   ```python
   from hpelm import ELM
   import numpy as np

   elm = ELM(X.shape[1], 1)
   elm.add_neurons(10, "sigm")
   elm.train(X, y, "c")

   preds = elm.predict(X)
   anomalies = X[np.abs(preds - y) > threshold]
   normal_data = X[np.abs(preds - y) <= threshold]

   print("Jumlah data normal:", normal_data.shape[0])
   print("Jumlah data anomali:", anomalies.shape[0])
   ```

## **Metode Deep Learning:**

1. **Variational Autoencoder (VAE)**  
   - Model generatif yang merekonstruksi input untuk mendeteksi anomali berdasarkan rekonstruksi error.

   **Kapan Digunakan:**
   - Dataset dengan distribusi kompleks dan dimensi tinggi.
   - Cocok untuk data gambar atau teks.

   **Hyperparameter:**
   - `latent_dim`: Dimensi ruang laten (default: 2).  
   - `epochs`: Jumlah iterasi pelatihan (default: 50).  
   - `batch_size`: Jumlah sampel per batch (default: 32).

   ```python
   import tensorflow as tf
   from tensorflow.keras.layers import Input, Dense, Lambda
   from tensorflow.keras.models import Model
   from tensorflow.keras import backend as K

   # Definisikan encoder
   input_dim = X.shape[1]
   latent_dim = 2

   inputs = Input(shape=(input_dim,))
   h = Dense(64, activation='relu')(inputs)
   z_mean = Dense(latent_dim)(h)
   z_log_var = Dense(latent_dim)(h)

   def sampling(args):
       z_mean, z_log_var = args
       epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim))
       return z_mean + K.exp(0.5 * z_log_var) * epsilon

   z = Lambda(sampling)([z_mean, z_log_var])

   # Decoder
   decoder_h = Dense(64, activation='relu')
   decoder_mean = Dense(input_dim, activation='sigmoid')
   h_decoded = decoder_h(z)
   x_decoded_mean = decoder_mean(h_decoded)

   # VAE model
   vae = Model(inputs, x_decoded_mean)
   vae.compile(optimizer='adam', loss='mse')

   vae.fit(X, X, epochs=50, batch_size=32)

   # Rekonstruksi error
   reconstruction = vae.predict(X)
   reconstruction_error = tf.reduce_mean(tf.square(X - reconstruction), axis=1)

   threshold = tf.reduce_mean(reconstruction_error) + 2 * tf.math.reduce_std(reconstruction_error)
   preds = (reconstruction_error > threshold).numpy()

   # Filter data
   normal_data = X[preds == 0]
   anomaly_data = X[preds == 1]

   print("Jumlah data normal:", normal_data.shape[0])
   print("Jumlah data anomali:", anomaly_data.shape[0])
   ```

2. **Generative Adversarial Networks (GAN)**  
   - Menggunakan generator dan discriminator untuk mendeteksi anomali berdasarkan kemampuan generator menghasilkan data.

   **Kapan Digunakan:**
   - Dataset besar dengan pola distribusi kompleks.
   - Cocok untuk data gambar, suara, atau teks.

   **Hyperparameter:**
   - `latent_dim`: Dimensi ruang laten (default: 100).  
   - `epochs`: Jumlah iterasi pelatihan (default: 50).  
   - `batch_size`: Jumlah sampel per batch (default: 32).

   ```python
   from tensorflow.keras.models import Sequential
   from tensorflow.keras.layers import Dense, LeakyReLU
   import numpy as np

   # Generator
   generator = Sequential([
       Dense(64, input_dim=100),
       LeakyReLU(0.2),
       Dense(X.shape[1], activation='sigmoid')
   ])

   # Discriminator
   discriminator = Sequential([
       Dense(64, input_dim=X.shape[1]),
       LeakyReLU(0.2),
       Dense(1, activation='sigmoid')
   ])
   discriminator.compile(optimizer='adam', loss='binary_crossentropy')

   # GAN
   discriminator.trainable = False
   gan = Sequential([generator, discriminator])
   gan.compile(optimizer='adam', loss='binary_crossentropy')

   # Training loop
   for epoch in range(50):
       noise = np.random.normal(0, 1, (32, 100))
       generated_data = generator.predict(noise)
       real_data = X[np.random.randint(0, X.shape[0], 32)]
       labels_real = np.ones((32, 1))
       labels_fake = np.zeros((32, 1))

       discriminator.train_on_batch(real_data, labels_real)
       discriminator.train_on_batch(generated_data, labels_fake)

       noise = np.random.normal(0, 1, (32, 100))
       gan.train_on_batch(noise, np.ones((32, 1)))

   # Evaluate anomalies
   noise = np.random.normal(0, 1, (X.shape[0], 100))
   generated_data = generator.predict(noise)
   reconstruction_error = np.mean(np.abs(X - generated_data), axis=1)

   threshold = np.mean(reconstruction_error) + 2 * np.std(reconstruction_error)
   preds = (reconstruction_error > threshold)

   # Filter data
   normal_data = X[~preds]
   anomaly_data = X[preds]

   print("Jumlah data normal:", normal_data.shape[0])
   print("Jumlah data anomali:", anomaly_data.shape[0])
   ```

3. **Recurrent Neural Networks (RNN)**  
   - Menggunakan urutan data (time series) untuk mendeteksi anomali berdasarkan pola temporal.

   **Kapan Digunakan:**
   - Data time series atau sekuensial.
   - Cocok untuk data keuangan, IoT, atau log aktivitas.

   **Hyperparameter:**
   - `units`: Jumlah neuron di lapisan tersembunyi (default: 50).  
   - `epochs`: Jumlah iterasi pelatihan (default: 50).  
   - `batch_size`: Jumlah sampel per batch (default: 32).

   ```python
   from tensorflow.keras.models import Sequential
   from tensorflow.keras.layers import LSTM, Dense

   model = Sequential([
       LSTM(50, input_shape=(X.shape[1], X.shape[2]), return_sequences=True),
       LSTM(50),
       Dense(X.shape[2])
   ])

   model.compile(optimizer='adam', loss='mse')
   model.fit(X, X, epochs=50, batch_size=32)

   # Rekonstruksi error
   reconstruction = model.predict(X)
   reconstruction_error = np.mean(np.abs(X - reconstruction), axis=1)

   threshold = np.mean(reconstruction_error) + 2 * np.std(reconstruction_error)
   preds = (reconstruction_error > threshold)

   # Filter data
   normal_data = X[~preds]
   anomaly_data = X[preds]

   print("Jumlah data normal:", normal_data.shape[0])
   print("Jumlah data anomali:", anomaly_data.shape[0])
   ```

4. **Deep Support Vector Data Description (Deep SVDD)**  
   - Memproyeksikan data ke ruang laten untuk mendeteksi anomali berdasarkan jarak dari pusat data normal.

   **Kapan Digunakan:**
   - Dataset berdimensi tinggi dengan pola distribusi non-linear.

   **Hyperparameter:**
   - `hidden_layers`: Arsitektur lapisan tersembunyi.  
   - `epochs`: Jumlah iterasi pelatihan (default: 50).  
   - `batch_size`: Jumlah sampel per batch (default: 32).

   ```python
   from keras.models import Sequential
   from keras.layers import Dense

   model = Sequential([
       Dense(64, activation='relu', input_dim=X.shape[1]),
       Dense(32, activation='relu'),
       Dense(1, activation='linear')
   ])

   model.compile(optimizer='adam', loss='mse')
   model.fit(X, X, epochs=50, batch_size=32)

   # Prediksi dan deteksi anomali
   predictions = model.predict(X)
   distances = np.linalg.norm(X - predictions, axis=1)

   threshold = np.mean(distances) + 2 * np.std(distances)
   preds = distances > threshold

   # Filter data
   normal_data = X[~preds]
   anomaly_data = X[preds]

   print("Jumlah data normal:", normal_data.shape[0])
   print("Jumlah data anomali:", anomaly_data.shape[0])
   ```
