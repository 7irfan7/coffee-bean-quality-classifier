import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

# ==========================
# PATH DATASET
# ==========================
dataset_path = "dataset"

samples = []

# Ambil 5 citra baik dan 5 citra buruk
for label in ["baik", "buruk"]:

    folder = os.path.join(dataset_path, label)

    files = sorted(os.listdir(folder))[:5]

    for file in files:
        samples.append((label, file, os.path.join(folder, file)))

# ==========================
# VISUALISASI RGB
# ==========================
for no, (label, filename, img_path) in enumerate(samples, start=1):

    img = cv2.imread(img_path)

    if img is None:
        continue

    # ==========================
    # SAMA PERSIS DENGAN TRAINING
    # ==========================
    img_resize = cv2.resize(img, (224, 224))

    # Hitung nilai RGB (SAMA PERSIS)
    mean_r = np.mean(img_resize[:, :, 2])
    mean_g = np.mean(img_resize[:, :, 1])
    mean_b = np.mean(img_resize[:, :, 0])

    # ==========================
    # OUTPUT TERMINAL
    # ==========================
    print("=" * 60)
    print(f"No        : {no}")
    print(f"Nama File : {filename}")
    print(f"Kelas     : {label.capitalize()}")
    print(f"Red (R)   : {mean_r:.2f}")
    print(f"Green (G) : {mean_g:.2f}")
    print(f"Blue (B)  : {mean_b:.2f}")

    # ==========================
    # VISUALISASI CHANNEL
    # ==========================

    # Untuk ditampilkan
    img_rgb = cv2.cvtColor(img_resize, cv2.COLOR_BGR2RGB)

    # Channel sesuai coding training
    red = img_rgb.copy()
    red[:, :, 1] = 0
    red[:, :, 2] = 0

    green = img_rgb.copy()
    green[:, :, 0] = 0
    green[:, :, 2] = 0

    blue = img_rgb.copy()
    blue[:, :, 0] = 0
    blue[:, :, 1] = 0

    plt.figure(figsize=(14,4))

    plt.subplot(1,4,1)
    plt.imshow(img_rgb)
    plt.title("Resize")
    plt.axis("off")

    plt.subplot(1,4,2)
    plt.imshow(red)
    plt.title("Red")
    plt.axis("off")

    plt.subplot(1,4,3)
    plt.imshow(green)
    plt.title("Green")
    plt.axis("off")

    plt.subplot(1,4,4)
    plt.imshow(blue)
    plt.title("Blue")
    plt.axis("off")

    plt.suptitle(
        f"{label.capitalize()} | {filename}",
        fontsize=13,
        fontweight="bold"
    )

    plt.tight_layout()

    plt.show()