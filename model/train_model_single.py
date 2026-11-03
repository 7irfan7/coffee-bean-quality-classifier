import cv2
import os
import matplotlib.pyplot as plt
import numpy as np
from skimage.feature import graycomatrix, graycoprops


path = 'dataset/baik'
files = os. listdir(path)

img_path = os.path.join(path, files[0])
img = cv2.imread(img_path)

#==============================================
# Preprocessing
#==============================================

# 1. resize
img_resize = cv2.resize(img, (224, 224))

# 2. grayscale
gray = cv2.cvtColor(img_resize, cv2.COLOR_BGR2GRAY)

#==============================================
# Ekstraksi fitur
#==============================================

# RGB
mean_r = np.mean(img_resize[:,:,2])
mean_g = np.mean(img_resize[:,:,1])
mean_b = np.mean(img_resize[:,:,0])

# GLCM
glcm = graycomatrix(gray, [1], [0], 256, symmetric=True, normed=True)

contrast = graycoprops(glcm, 'contrast')[0,0]
energy = graycoprops(glcm, 'energy')[0,0]
homogeneity = graycoprops(glcm, 'homogeneity')[0,0]

# Gabung Fitur
fitur = [mean_r, mean_g, mean_b, contrast, energy, homogeneity]

print("Nama File :", files[0])
print("Fitur:", fitur)

#==============================================
# Visualisasi
#==============================================

# gambar warma(hasil resize)
img_rgb = cv2.cvtColor(img_resize, cv2.COLOR_BGR2RGB)

plt.subplot(1,2,1)
plt.imshow(img_rgb)
plt.title("Resize")
plt.axis('off')

plt.subplot(1,2,2)
plt.imshow(gray,cmap='gray')
plt.title("Grayscale")
plt.axis('off')

plt.show()