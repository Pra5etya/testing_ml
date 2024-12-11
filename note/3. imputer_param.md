# Parameter pada SimpleImputer

## `missing_values` (default: `np.nan`)
- Menentukan nilai yang akan dianggap sebagai *missing value*.
- **Tipe**: `int`, `float`, `str`, atau `np.nan`.
- **Contoh**:
  - Untuk menangani nilai kosong (`NaN`): `missing_values=np.nan`.
  - Untuk menangani nilai tertentu, misalnya `-1`: `missing_values=-1`.

---

## `strategy` (default: `'mean'`)
- Menentukan metode imputasi yang digunakan.
- **Opsi**:
  - `'mean'`: Mengganti nilai hilang dengan rata-rata (*hanya untuk data numerik*).
  - `'median'`: Mengganti nilai hilang dengan median (*hanya untuk data numerik*).
  - `'most_frequent'`: Mengganti nilai hilang dengan nilai yang paling sering muncul (*bisa untuk data numerik atau kategorikal*).
  - `'constant'`: Mengganti nilai hilang dengan nilai tetap (*gunakan dengan parameter `fill_value`*).

---

## `fill_value` (default: `None`)
- Digunakan jika `strategy='constant'`.
- Menentukan nilai tetap yang akan digunakan untuk mengganti nilai hilang.
- **Default**:
  - Untuk data numerik: `0`.
  - Untuk data string/obyek: `'missing'`.

---

## `verbose` (default: `0`)
- Menentukan apakah jumlah nilai yang hilang untuk setiap kolom akan dilaporkan.
- **Tipe**: `int`.
  - `0`: Tidak menampilkan informasi.
  - `1`: Menampilkan jumlah nilai yang hilang yang sedang diimput.

---

## `copy` (default: `True`)
- Menentukan apakah data input akan disalin sebelum diimputasi.
- **Tipe**: `bool`.
  - `True`: Membuat salinan data.
  - `False`: Mengubah data asli secara langsung.
- **Catatan**: 
  - Set `copy=False` untuk menghemat memori, tetapi hati-hati jika mengubah data asli tidak diinginkan.

---

## `add_indicator` (default: `False`)
- Menentukan apakah akan menambahkan indikator (*fitur tambahan*) untuk menunjukkan lokasi nilai yang hilang.
- **Tipe**: `bool`.
  - `True`: Menambahkan kolom indikator untuk setiap fitur dengan nilai hilang.
  - `False`: Tidak menambahkan indikator.
