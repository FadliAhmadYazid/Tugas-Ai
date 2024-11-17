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

Convolutional Neural Networks (CNN) adalah arsitektur jaringan saraf dalam deep learning yang sangat efisien dan efektif dalam memproses data spasial seperti gambar. Dengan memanfaatkan lapisan-lapisan penting seperti **Convolutional Layer**, **Pooling Layer**, **Flattening**, dan **Fully Connected Layer**, CNN mampu mengekstraksi fitur yang relevan dan membuat prediksi yang akurat. Keunggulan CNN terlet

ak pada kemampuannya untuk secara otomatis belajar fitur-fitur dari data, yang membuatnya sangat berguna dalam aplikasi seperti pengenalan objek, klasifikasi gambar, dan deteksi wajah.

Dengan kemampuan untuk menangani data dalam jumlah besar dan menghasilkan hasil yang sangat baik, CNN telah menjadi dasar bagi banyak teknologi canggih dalam pengolahan gambar dan visi komputer, serta menjadi pilar utama dalam perkembangan kecerdasan buatan modern.
