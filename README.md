# Link 1: https://www.megabagus.id/deep-learning-convolutional-neural-networks/
### **Deep Learning: Convolutional Neural Networks (CNN)**

Convolutional Neural Networks (CNN) adalah jenis jaringan saraf dalam deep learning yang telah terbukti sangat efektif dalam menangani data berbentuk gambar, video, dan data spasial lainnya. CNN telah menjadi bagian penting dalam berbagai aplikasi seperti pengenalan wajah, klasifikasi gambar, deteksi objek, dan pemrosesan video. Keunggulan utama CNN terletak pada kemampuannya untuk secara otomatis belajar mengekstrak fitur penting dari data, tanpa memerlukan ekstraksi fitur manual seperti yang dibutuhkan oleh model lain seperti Support Vector Machines (SVM).

#### **1. Apa Itu Convolutional Neural Network?**

Convolutional Neural Networks (CNN) adalah salah satu arsitektur jaringan saraf tiruan yang terinspirasi oleh cara manusia melihat dan memahami gambar. Dalam konteks pengolahan gambar, CNN menggunakan konsep konvolusi untuk mengekstraksi fitur penting dari gambar, kemudian melanjutkan untuk melakukan klasifikasi atau prediksi berdasarkan fitur-fitur tersebut. CNN adalah jenis jaringan saraf yang terdiri dari berbagai lapisan (layers) yang masing-masing memiliki fungsi tertentu untuk memproses dan menyaring informasi yang ada dalam gambar.

CNN memiliki beberapa lapisan utama yang bekerja secara bertahap, yang meliputi **Convolutional Layer**, **Pooling Layer**, **Flattening**, dan **Fully Connected Layer**. Setiap lapisan memiliki peran tertentu dalam memproses data untuk mengidentifikasi pola yang relevan dalam gambar atau data spasial lainnya.

#### **2. Arsitektur CNN: Lapisan-lapisan Utama**

##### **a. Convolutional Layer**

Lapisan konvolusi (Convolutional Layer) adalah lapisan pertama dalam CNN yang bertugas untuk mengekstraksi fitur dari data input, biasanya berupa gambar. Fungsi utama dari lapisan konvolusi adalah untuk mendeteksi pola-pola dasar dalam gambar, seperti tepi, sudut, tekstur, dan bentuk lainnya. Lapisan ini menggunakan **filter** atau **kernel**, yang merupakan matriks kecil yang bergerak melintasi gambar, menghitung hasil perkalian antara nilai pixel gambar dengan elemen-elemen dari filter, dan menghasilkan output yang disebut **feature map**.

**Proses Konvolusi:**
Misalnya, kita memiliki gambar dengan ukuran 5x5 dan filter berukuran 3x3. Filter ini akan bergerak melintasi gambar dengan langkah tertentu yang disebut **stride**, dan setiap kali filter diterapkan pada bagian gambar, hasilnya adalah perkalian elemen-elemen filter dengan nilai-nilai pixel gambar yang dilaluinya. Proses ini menghasilkan nilai output yang lebih kecil yang menunjukkan seberapa kuat keberadaan fitur tertentu dalam gambar.

**Fungsi Filter/Kernels:**
- **Edge Detection:** Filter dapat digunakan untuk mendeteksi tepi dalam gambar, yang merupakan pola dasar yang sering digunakan untuk mengenali objek.
- **Feature Detection:** Filter juga dapat mendeteksi fitur lain seperti sudut atau tekstur yang lebih kompleks, yang akan digunakan untuk identifikasi objek yang lebih maju.

CNN menggunakan **filter yang dapat dipelajari**. Ini berarti bahwa filter-filter tersebut tidak ditentukan secara manual, tetapi jaringan akan mengubah dan menyesuaikan filter selama proses pelatihan untuk mendeteksi fitur-fitur yang paling penting dari data input.

##### **b. Activation Function (ReLU)**

Setelah konvolusi dilakukan, hasilnya biasanya akan melalui fungsi aktivasi seperti **Rectified Linear Unit (ReLU)**. ReLU adalah fungsi non-linear yang paling sering digunakan dalam CNN karena kemampuannya untuk mengatasi masalah gradien yang hilang dan mempercepat proses pelatihan. ReLU mengubah semua nilai negatif dalam hasil konvolusi menjadi nol, sementara nilai positif tetap tidak berubah. Hal ini memungkinkan jaringan untuk belajar lebih cepat dan menghasilkan representasi yang lebih baik dari data.

##### **c. Pooling Layer (Max Pooling)**

Setelah lapisan konvolusi, CNN biasanya akan menggunakan lapisan **Pooling** untuk mengurangi dimensi dari data yang diproses dan mengurangi beban komputasi yang diperlukan. Tujuan utama dari pooling adalah untuk **mengurangi ukuran feature map** dengan cara merangkum informasi yang ada, tanpa kehilangan fitur penting yang diperlukan untuk pengenalan objek.

**Max Pooling:**
Salah satu teknik pooling yang paling populer adalah **Max Pooling**, di mana kita mengambil nilai maksimum dari setiap blok kecil dalam feature map. Misalnya, jika kita memiliki sebuah feature map berukuran 4x4 dan menggunakan window 2x2, maka max pooling akan memilih nilai terbesar dari setiap blok 2x2 dalam matriks tersebut. Proses ini mengurangi ukuran feature map tetapi tetap mempertahankan informasi yang relevan.

**Manfaat Max Pooling:**
- **Reduksi Dimensi:** Max pooling mengurangi ukuran data, yang membantu mengurangi beban komputasi dan waktu pelatihan model.
- **Pengurangan Overfitting:** Dengan mengurangi dimensi data, model menjadi kurang terikat pada noise atau detail yang tidak relevan dalam data.
- **Kestabilan terhadap Perubahan Posisi:** Max pooling juga membuat model lebih stabil terhadap perubahan posisi objek dalam gambar.

##### **d. Flattening**

Setelah lapisan konvolusi dan pooling, kita memiliki sebuah output yang masih berupa matriks dua dimensi. Namun, untuk bisa diproses oleh lapisan berikutnya yang berupa jaringan saraf penuh (Fully Connected Layer), data perlu diubah menjadi format satu dimensi. **Flattening** adalah proses untuk meratakan hasil feature map yang sudah melalui konvolusi dan pooling menjadi sebuah vektor satu dimensi.

**Contoh Flattening:**
Misalkan hasil pooling kita adalah sebuah matriks berukuran 3x3:
```
1 2 3
4 5 6
7 8 9
```
Proses flattening akan mengubah matriks tersebut menjadi vektor satu dimensi:
```
[1, 2, 3, 4, 5, 6, 7, 8, 9]
```

Vektor ini kemudian akan menjadi input untuk lapisan selanjutnya dalam jaringan saraf, yaitu lapisan **Fully Connected**.

##### **e. Fully Connected Layer**

Setelah proses konvolusi, pooling, dan flattening, data akan diproses oleh lapisan **Fully Connected (FC)**. Pada lapisan ini, setiap neuron pada lapisan sebelumnya akan terhubung dengan setiap neuron di lapisan berikutnya. FC Layer bertanggung jawab untuk **menggabungkan fitur yang telah dipelajari** dan menghasilkan output prediksi berdasarkan informasi tersebut.

Di dalam FC Layer, setiap neuron akan melakukan perhitungan berbasis bobot yang dioptimalkan selama pelatihan. Proses ini berfungsi untuk **mengklasifikasikan gambar** atau memberikan prediksi berdasarkan fitur-fitur yang telah dipelajari oleh jaringan. Di akhir FC Layer, biasanya digunakan fungsi aktivasi **Softmax** untuk menghasilkan probabilitas untuk setiap kelas yang ada, dan model akan memilih kelas dengan probabilitas tertinggi sebagai hasil prediksi.

**Contoh Output FC Layer:**
Misalnya, jika kita mengklasifikasikan gambar menjadi dua kelas: "kucing" dan "anjing", output dari FC Layer mungkin akan seperti ini:
- Probabilitas untuk kucing: 0.85
- Probabilitas untuk anjing: 0.15

Dengan probabilitas tersebut, model akan memilih kelas dengan probabilitas tertinggi, dalam hal ini "kucing".

#### **3. Gabungan Proses: Max Pooling, Flattening, dan Fully Connected**

- **Max Pooling:** Fungsi utama pooling adalah mengurangi dimensi data tanpa mengurangi informasi penting, menjaga fitur utama dan mengurangi noise. Max Pooling membantu menjaga fitur yang relevan dengan memilih nilai maksimum dari blok kecil dalam data, yang memperkecil ukuran feature map dan membantu mengurangi overfitting.
  
- **Flattening:** Proses ini meratakan data dari dua dimensi menjadi satu dimensi, yang memungkinkan data tersebut untuk diproses oleh jaringan saraf penuh (Fully Connected Layer). Tanpa flattening, jaringan tidak dapat menghubungkan informasi dari layer sebelumnya dengan lapisan berikutnya yang berupa FC.
  
- **Fully Connected Layer:** Pada tahap ini, seluruh informasi yang telah dipelajari oleh model dikombinasikan dan digunakan untuk membuat prediksi akhir. FC Layer menghasilkan output berupa prediksi kelas atau nilai yang sesuai dengan tujuan model, baik itu klasifikasi gambar, deteksi objek, atau regresi.

#### **4. Keunggulan CNN dalam Pengenalan Pola**

CNN memiliki berbagai keunggulan yang menjadikannya pilihan utama dalam berbagai tugas pengenalan pola, terutama dalam pengolahan gambar. Beberapa keuntungan utamanya adalah:

- **Efektivitas dalam Ekstraksi Fitur:** CNN secara otomatis belajar untuk mengekstraksi fitur-fitur penting dari data, tanpa memerlukan pemrograman fitur manual.
- **Reduksi Parameter:** Melalui penggunaan filter dan weight sharing, CNN mengurangi jumlah parameter yang perlu dilatih dibandingkan dengan jaringan saraf tradisional.
- **Kemampuan Generalisasi yang Baik:** Karena proses konvolusi berfokus pada pola lokal dalam data, CNN mampu bekerja dengan baik pada data dengan variasi posisi dan orientasi objek.
- **Kinerja Tinggi:** CNN sangat efisien dalam memproses gambar beresolusi tinggi dan dapat menangani data dengan volume besar.

#### **Kesimpulan:**

Convolutional Neural Networks (CNN) adalah arsitektur jaringan saraf dalam deep learning yang sangat efisien dan efektif dalam memproses data spasial seperti gambar. Dengan memanfaatkan lapisan-lapisan penting seperti **Convolutional Layer**, **Pooling Layer**, **Flattening**, dan **Fully Connected Layer**, CNN mampu mengekstraksi fitur yang relevan dan membuat prediksi yang akurat. Keunggulan CNN terletak pada kemampuannya untuk secara otomatis belajar fitur-fitur dari data, yang membuatnya sangat berguna dalam aplikasi seperti pengenalan objek, klasifikasi gambar, dan deteksi wajah.

Dengan kemampuan untuk menangani data dalam jumlah besar dan menghasilkan hasil yang sangat baik, CNN telah menjadi dasar bagi banyak teknologi canggih dalam pengolahan gambar dan visi komputer, serta menjadi pilar utama dalam perkembangan kecerdasan buatan modern.

# Link 2: https://www.megabagus.id/deep-learning-convolutional-neural-networks-aplikasi/
### **Deep Learning: Convolutional Neural Networks (aplikasi)**

Artikel ini membahas aplikasi Convolutional Neural Networks (CNN) dalam melakukan klasifikasi gambar dengan Python, khususnya untuk membedakan antara gambar kucing (*cats*) dan anjing (*dogs*). Sebelum praktik, pembaca disarankan memahami dasar-dasar Python, konsep dasar deep learning, dan teori CNN melalui artikel terkait. Untuk menjalankan proyek, pembaca perlu mengunduh dataset besar (200-300 MB) yang disediakan melalui Google Drive.

### 1. **Dataset dan Struktur Data**  
Dataset terdiri dari 10.000 gambar:  
- **Training set**: 8.000 gambar (4.000 kucing, 4.000 anjing).  
- **Test set**: 2.000 gambar (1.000 kucing, 1.000 anjing).  

Dataset dibagi ke dalam dua folder utama (`training_set` dan `test_set`), masing-masing berisi subfolder `cats` dan `dogs`. File gambar diberi nama urut (*e.g.*, `cat1.jpg`, `dog1.jpg`) agar algoritma dapat mengenali kategori gambar berdasarkan nama file. Variabel independen adalah piksel gambar, sedangkan variabel dependen adalah kategori (kucing/anjing).

### 2. **Langkah Implementasi CNN dengan Python**  
Berikut adalah penjelasan untuk masing-masing bagian kode yang diberikan:

 

## **Kode 1: Melatih Model CNN untuk Klasifikasi Gambar**

### **1. Mengimpor Library**
```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
```
- **`Sequential`**: Membuat model CNN secara bertahap.
- **`Conv2D`, `MaxPooling2D`, `Flatten`, `Dense`**: Layer-layer untuk membangun arsitektur CNN.
- **`ImageDataGenerator`**: Digunakan untuk augmentasi data gambar dan normalisasi.

 

### **2. Membangun Arsitektur CNN**
```python
MesinKlasifikasi = Sequential()
MesinKlasifikasi.add(Conv2D(filters=32, kernel_size=(3, 3), input_shape=(128, 128, 3), activation='relu'))
MesinKlasifikasi.add(MaxPooling2D(pool_size=(2, 2)))
```
- **`Conv2D`**: Layer konvolusi dengan 32 filter dan ukuran kernel \(3 \times 3\). Fungsi aktivasi ReLU digunakan untuk menambahkan non-linearitas.
- **`input_shape=(128, 128, 3)`**: Model mengharapkan input gambar berukuran \(128 \times 128\) dengan 3 channel (RGB).
- **`MaxPooling2D`**: Layer pooling dengan ukuran \(2 \times 2\), digunakan untuk mengurangi ukuran gambar tanpa kehilangan fitur penting.

```python
MesinKlasifikasi.add(Conv2D(32, (3, 3), activation='relu'))
MesinKlasifikasi.add(MaxPooling2D(pool_size=(2, 2)))
```
- Menambahkan layer konvolusi dan pooling tambahan untuk meningkatkan kemampuan ekstraksi fitur.

```python
MesinKlasifikasi.add(Flatten())
```
- **`Flatten`**: Mengubah output matriks 2D dari layer sebelumnya menjadi vektor 1D.

```python
MesinKlasifikasi.add(Dense(units=128, activation='relu'))
MesinKlasifikasi.add(Dense(units=1, activation='sigmoid'))
```
- **Dense Layer**: 
  - Layer pertama memiliki 128 neuron dengan fungsi aktivasi ReLU.
  - Layer kedua memiliki 1 neuron dengan fungsi aktivasi **sigmoid**, karena tugas klasifikasi adalah **biner** (anjing/kucing).

 

### **3. Kompilasi Model**
```python
MesinKlasifikasi.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
```
- **`adam`**: Optimizer untuk mempercepat pelatihan.
- **`binary_crossentropy`**: Fungsi loss yang cocok untuk klasifikasi biner.
- **`accuracy`**: Metrik evaluasi model.

 

### **4. Augmentasi Data dan Normalisasi**
```python
train_datagen = ImageDataGenerator(rescale=1./255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)
```
- **Rescaling**: Mengubah nilai piksel gambar menjadi antara 0 dan 1.
- **Augmentasi Data**: Data pelatihan diperbesar secara virtual dengan transformasi seperti rotasi, zoom, dan flipping.

```python
training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                 target_size=(128, 128),
                                                 batch_size=32,
                                                 class_mode='binary')

test_set = test_datagen.flow_from_directory('dataset/test_set',
                                            target_size=(128, 128),
                                            batch_size=32,
                                            class_mode='binary')
```
- **`flow_from_directory`**: Memuat gambar dari folder.
- **`target_size=(128, 128)`**: Mengubah ukuran semua gambar menjadi \(128 \times 128\).
- **`class_mode='binary'`**: Tugas klasifikasi biner (anjing/kucing).

 

### **5. Melatih Model**
```python
MesinKlasifikasi.fit(
    training_set,
    steps_per_epoch=8000 // 32,
    epochs=50,
    validation_data=test_set,
    validation_steps=2000 // 32
)
```
- **`steps_per_epoch`**: Jumlah batch yang diproses dalam satu epoch. Dataset pelatihan memiliki 8.000 gambar.
- **`epochs=50`**: Model dilatih selama 50 epoch.
- **`validation_data`**: Data pengujian digunakan untuk mengevaluasi model di setiap epoch.

 

## **Kode 2: Menggunakan Model untuk Prediksi**

### **1. Memuat Gambar Individu**
```python
import numpy as np
from keras.preprocessing import image
```
- **`image`**: Digunakan untuk memuat dan memproses gambar untuk inferensi.

 

### **2. Membuat Prediksi**
```python
for i in range(4001, 5001): 
    test_image = image.load_img('dataset/test_set/dogs/dog.' + str(i) + '.jpg', target_size = (128, 128))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis = 0)
    result = MesinKlasifikasi.predict(test_image)
```
- **Memuat gambar**:
  - Gambar dari folder `dogs` dengan nama file berformat `dog.4001.jpg`, dst.
  - **`target_size=(128, 128)`**: Mengubah ukuran gambar agar sesuai dengan input model.
- **Konversi ke array**: Gambar diubah menjadi array menggunakan `img_to_array`.
- **Menambahkan dimensi batch**: Gambar dimasukkan dalam batch menggunakan `expand_dims`.
- **Prediksi**: Model memberikan probabilitas untuk kelas **dog** (1) atau **cat** (0).

 

### **3. Menentukan Kelas**
```python
    if result[0][0] == 0:
        prediction = 'cat'
        count_cat = count_cat + 1
    else:
        prediction = 'dog'
        count_dog = count_dog + 1
```
- Jika **probabilitas** hasil prediksi adalah 0, kelasnya adalah **cat**; jika 1, kelasnya adalah **dog**.
- Variabel `count_cat` dan `count_dog` digunakan untuk menghitung jumlah prediksi untuk setiap kelas.

 

### **4. Menampilkan Hasil**
```python
print("count_dog:" + str(count_dog))    
print("count_cat:" + str(count_cat))
```
- Menampilkan jumlah prediksi untuk masing-masing kelas (dog/cat).

 

## **Kesimpulan**
- **Kode 1** melatih model CNN untuk klasifikasi gambar biner (anjing/kucing) menggunakan augmentasi data.
- **Kode 2** menggunakan model terlatih untuk memprediksi gambar individu dari folder pengujian. Hasilnya dihitung dan dirangkum dalam bentuk jumlah prediksi untuk masing-masing kelas.

# Link 3: https://modul-praktikum-ai.vercel.app/Materi/4-convolutional-neural-network
Berikut adalah penjelasan masing-masing bagian dari kode pada link tersebut:

### **1. Memuat dan Menyiapkan Data CIFAR-10**
```python
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
 
# Memuat data CIFAR-10
(train_images, train_labels), (test_images, test_labels) = cifar10.load_data()
```
- **`cifar10.load_data()`**: Memuat dataset CIFAR-10, yang terdiri dari 60.000 gambar berukuran 32x32 dalam 10 kelas. Dataset ini dibagi menjadi **50.000 data pelatihan** dan **10.000 data pengujian**.

```python
# Normalisasi data gambar
train_images = train_images.astype('float32') / 255.0
test_images = test_images.astype('float32') / 255.0
```
- **Normalisasi**: Data gambar diubah menjadi nilai antara 0 hingga 1 dengan membagi setiap piksel dengan 255.0. Ini mempercepat konvergensi model selama pelatihan.

```python
# Mengonversi label ke bentuk kategorikal
train_labels = to_categorical(train_labels, 10)
test_labels = to_categorical(test_labels, 10)
```
- **Label Kategorikal**: Label angka (0–9) diubah menjadi bentuk **one-hot encoding** untuk digunakan dalam klasifikasi multi-kelas.

 

### **2. Membangun Model CNN**
```python
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    tf.keras.layers.MaxPooling2D((2, 2)),
```
- **`Conv2D`**: Layer konvolusi dengan 32 filter, ukuran kernel \(3 \times 3\), dan fungsi aktivasi ReLU. Layer ini bertugas mengekstraksi fitur dari gambar.
- **`MaxPooling2D`**: Layer pooling dengan ukuran kernel \(2 \times 2\) untuk mengurangi ukuran gambar dan meminimalkan kompleksitas.

```python
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
```
- Menambahkan layer konvolusi kedua dengan 64 filter dan pooling untuk memperdalam jaringan.

```python
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
```
- Menambahkan layer konvolusi ketiga dengan 128 filter untuk menangkap lebih banyak fitur kompleks.

```python
    tf.keras.layers.Flatten(),
```
- **`Flatten`**: Mengubah output dari matriks 2D menjadi vektor 1D untuk input ke layer fully connected.

```python
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.5),
```
- **`Dense`**: Layer fully connected dengan 128 neuron.
- **`Dropout`**: Mengurangi overfitting dengan menonaktifkan 50% neuron secara acak selama pelatihan.

```python
    tf.keras.layers.Dense(10, activation='softmax')
])
```
- **Layer Output**: Menghasilkan probabilitas untuk 10 kelas menggunakan fungsi aktivasi **softmax**.

```python
model.summary()
```
- Menampilkan ringkasan struktur model, termasuk jumlah parameter yang dapat dilatih.

 

### **3. Kompilasi dan Pelatihan Model**
```python
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
```
- **`adam`**: Optimizer adaptif yang populer untuk mempercepat pelatihan.
- **`categorical_crossentropy`**: Fungsi loss untuk klasifikasi multi-kelas.
- **`metrics=['accuracy']`**: Menggunakan akurasi sebagai metrik evaluasi.

```python
history = model.fit(train_images, train_labels, epochs=10, batch_size=64, validation_data=(test_images, test_labels))
```
- **Pelatihan Model**: Model dilatih dengan **10 epoch** menggunakan batch size 64, dengan data validasi dari data pengujian.

 

### **4. Evaluasi Model**
```python
test_loss, test_acc = model.evaluate(test_images, test_labels)
print('Test accuracy:', test_acc)
```
- **Evaluasi**: Mengukur akurasi model pada data pengujian.

 

### **5. Demo Inferensi**
```python
from google.colab import files
from keras.models import load_model
from PIL import Image
import numpy as np
```
- **`files.upload()`**: Fungsi untuk mengunggah file gambar.
- **`load_model`**: Memuat model CNN yang telah disimpan.
- **`PIL.Image`**: Digunakan untuk memuat dan memproses gambar.

```python
def load_and_prepare_image(file_path):
    img = Image.open(file_path)
    img = img.resize((32, 32))  # Sesuaikan dengan dimensi yang model Anda harapkan
    img = np.array(img) / 255.0  # Normalisasi
    img = np.expand_dims(img, axis=0)  # Tambahkan batch dimension
    return img
```
- **Fungsi `load_and_prepare_image`**: Memuat gambar, mengubah ukuran ke \(32 \times 32\), melakukan normalisasi, dan menambahkan dimensi batch.

```python
for filename in uploaded.keys():
    img = load_and_prepare_image(filename)
    prediction = model.predict(img)
    predicted_class_index = np.argmax(prediction, axis=1)[0]
    predicted_class_name = class_names[predicted_class_index]
    print(f'File: {filename}, Predicted Class Index: {predicted_class_index}, Predicted Class Name: {predicted_class_name}')
```
- **Inferensi**: Model membuat prediksi pada gambar yang diunggah, memberikan nama kelas dengan probabilitas tertinggi berdasarkan output softmax.

 ### **Kesimpulan**  
- Dataset CIFAR-10 diproses dengan normalisasi dan konversi label menjadi one-hot encoding untuk pelatihan model.  
- Model CNN dibangun dengan beberapa **Convolutional Layer**, **Pooling Layer**, dan **Dense Layer** untuk klasifikasi 10 kelas.  
- Model dilatih menggunakan TensorFlow dengan **optimizer Adam** dan **loss categorical_crossentropy** untuk mencapai akurasi optimal.  
- Hasil model dievaluasi pada data pengujian, dan dapat digunakan untuk memprediksi gambar baru melalui proses inferensi.
