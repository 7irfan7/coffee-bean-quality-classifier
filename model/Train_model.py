import cv2
import os
import numpy as np
from skimage.feature import graycomatrix, graycoprops

# TEMPAT SIMPAN DATA
X = []
Y = []

# LOOP FOLDER

for label in ['baik', 'buruk']:
    path = 'dataset/' + label
    files = os.listdir(path)

    for file in files:
        img_path = os.path.join(path, file)
        img = cv2.imread(img_path)

        if img is None:
            continue

        # PREPROCESSING
        
        img_resize = cv2.resize(img, (224, 224))
        gray = cv2.cvtColor(img_resize, cv2.COLOR_BGR2GRAY)

        # FITUR RGB

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
        
        fitur = [
            mean_r, mean_g, mean_b,
            contrast, energy, homogeneity,
            dissimilarity, correlation, asm
            ]

        X.append(fitur)
        if label == 'baik':
            Y.append(1)
        else:
            Y.append(0)

# CEK HASIL

print("Jumlah Data:", len(X))
for i in range(1):
    fitur = X[i]

    fitur_dict = {
        "R": fitur[0],
        "G": fitur[1],
        "B": fitur[2],
        "Contrast": fitur[3],
        "Energy": fitur[4],
        "Homogeneity": fitur[5],
        "Dissimilarity": fitur[6],
        "Correlation": fitur[7],
        "ASM": fitur[8]
    }

    print(f"Data ke-{i+1}:")
    print(fitur_dict)
    print("Label:", Y[i])
    print("---------------------")

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# SPLIT DATA
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y,
    test_size=0.2,
    random_state=42,
    stratify=Y
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# TRAINING SVM
model = SVC(kernel='rbf', C=5, gamma=0.1)
model.fit(X_train, Y_train)

# PREDIKSI
Y_pred = model.predict(X_test)

# EVALUASI
accuracy = accuracy_score(Y_test, Y_pred)

print("Akurasi:", accuracy)

print("Confusion Matrix:")
print(confusion_matrix(Y_test, Y_pred))

print("Classification Report:")
print(classification_report(Y_test, Y_pred))

import joblib
joblib.dump(model, 'model_svm.pkl')
joblib.dump(scaler, 'scaler.pkl')
print("Model Berhasil disimpan sebagai model_svm.pkl")