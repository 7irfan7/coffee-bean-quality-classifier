# coffee-bean-quality-classifier
web based coffee bean classification using Python, OpenCV, and SVM
Project Overview

Proyek ini merupakan aplikasi web pengolahan citra digital (*Computer Vision*) yang dirancang untuk mengotomatisasi proses klasifikasi mutu biji kopi Robusta. Sistem ini dikembangkan untuk mengatasi kendala sortir manual yang cenderung subjektif dan memakan waktu, sehingga proses penentuan kualitas biji kopi dapat dilakukan secara lebih cepat, terstandarisasi, dan akurat.

Dalam prosesnya, pengguna mengunggah citra biji kopi melalui antarmuka web berbasis **Flask**. Sistem kemudian melakukan *preprocessing* citra menggunakan **OpenCV** (pembersihan *noise* dan penyesuaian ukuran citra), dilanjutkan dengan tahap ekstraksi fitur yang mengombinasikan **Fitur Warna (RGB)** dan **Fitur Tekstur (GLCM)**. Hasil ekstraksi fitur tersebut selanjutnya diklasifikasikan menggunakan algoritma **Support Vector Machine (SVM)** dengan **Kernel RBF**.

Melalui pengujian data, model klasifikasi SVM ini berhasil mencapai tingkat **Akurasi sebesar 85%** (dengan nilai *Precision* 86%, *Recall* 83%, dan *F1-Score* 85%). Selain pengolahan data, platform ini juga dilengkapi dengan arsitektur web yang mendukung sistem otentikasi pengguna (*login/register*) serta manajemen profil untuk menyimpan histori hasil klasifikasi.
