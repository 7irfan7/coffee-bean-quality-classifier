import cv2
import numpy as np
import joblib
import os
from skimage.feature import graycomatrix, graycoprops

# LOAD MODEL
model = joblib.load('model_svm.pkl')
scaler = joblib.load('scaler.pkl')

# LOAD GAMBAR BARU
path = 'test'

for file in os.listdir(path):
    if not file.endswith(('.jpg', '.jpeg', '.png')):
        continue
    img_path = os.path.join(path, file)
    
    print("\n===============")
    print("Gambar:", file)

    img = cv2.imread(img_path)
    if img is None:
        print("Gagal membaca gambar")
        continue

    #PREPROCESSING
    img_resize = cv2.resize(img, (224, 224))
    gray = cv2.cvtColor(img_resize, cv2.COLOR_BGR2GRAY)

    # EKSTRAKSI FITUR
    mean_r = np.mean(img_resize[:,:,2])
    mean_g = np.mean(img_resize[:,:,1])
    mean_b = np.mean(img_resize[:,:,0])

    # FITUR GLCM

    glcm = graycomatrix(gray, [1], [0], 256, symmetric=True, normed=True)

    contrast = graycoprops(glcm, 'contrast')[0,0]
    energy = graycoprops(glcm, 'energy')[0,0]
    homogeneity = graycoprops(glcm, 'homogeneity')[0,0]
    dissimilarity = graycoprops(glcm, 'dissimilarity')[0,0]
    correlation = graycoprops(glcm, 'correlation')[0,0]
    asm = graycoprops(glcm, 'ASM')[0,0]

    # GABUNG FITUR
        
    fitur = [[
        mean_r, mean_g, mean_b,
        contrast, energy, homogeneity,
        dissimilarity, correlation, asm
        ]]
    
    # SCALER
    fitur = scaler.transform(fitur)

    # PREDIKSI
    hasil = model.predict(fitur)

    # Output
    if hasil[0] == 1:
        print("Hasil: BAIK")
    else:
        print("Hasil: BURUK")
