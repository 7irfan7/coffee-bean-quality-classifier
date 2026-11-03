from flask import Flask, render_template, request, redirect, session
from werkzeug.utils import secure_filename
import os
import sqlite3
import cv2
import numpy as np
import joblib

from skimage.feature import graycomatrix, graycoprops

model = joblib.load('model_svm.pkl')
scaler = joblib.load('scaler.pkl')

app = Flask(__name__)
app.secret_key = "coffevision"

@app.route('/')
def home():
    user_login = False
    if 'id_user' in session:
        user_login = True

    return render_template(
        'home.html',
        user_login=user_login
    )

@app.route('/registrasi', methods=['GET', 'POST'])
def registrasi():
    if request.method == 'POST':

        nama = request.form['nama']
        email = request.form['email']
        umur = request.form['umur']
        pekerjaan = request.form['pekerjaan']
        password = request.form['password']

        conn = sqlite3.connect('database_klasifikasi.db')
        cursor = conn.cursor()

        cursor.execute(
            "SELECT * FROM users WHERE email=?",
            (email,)
        )

        cek_email = cursor.fetchone()

        if cek_email:
            conn.close()
            return render_template(
                'registrasi.html',
                pesan="Email Sudah Terdaftar!"
            )

        cursor.execute('''
        INSERT INTO users (nama, email, umur, pekerjaan, password, foto_profil)
        VALUES (?, ?, ?, ?, ?, ?)
        ''', (nama, email, umur, pekerjaan, password, 'default.png'))

        conn.commit()
        conn.close()

        return redirect('/login')
    return render_template('registrasi.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']

        conn = sqlite3.connect('database_klasifikasi.db')
        cursor = conn.cursor()

        cursor.execute(
            "SELECT * FROM users WHERE email=? AND password=?",
            (email, password)
        )
        user = cursor.fetchone()
        conn.close()
        if user:
            session['id_user'] = user[0]
            session['nama'] = user[1]
            session['email'] = user[2]
            session['foto_profil'] = user[6]
            return redirect('/')
        return render_template(
            'login.html',
            pesan='Email atau Password Salah'
        )
    return render_template('login.html')

@app.route('/klasifikasi', methods=['GET','POST'])
def klasifikasi():

    if 'id_user' not in session:
        return redirect('/login')

    hasil = None
    gambar = None

    rgb = None
    glcm_data= None

    if request.method == 'POST':

        file = request.files['gambar']

        if file:

            nama_file = secure_filename(file.filename)

            gambar = f"static/uploads/{nama_file}"

            lokasi = os.path.join(
                app.root_path,
                "static",
                "uploads",
            nama_file
            )

            file.save(lokasi)

            # =====================
            # PREPROCESSING
            # =====================

            img = cv2.imread(lokasi)

            img_resize = cv2.resize(img, (224,224))

            gray = cv2.cvtColor(
                img_resize,
                cv2.COLOR_BGR2GRAY
            )

            # =====================
            # RGB
            # =====================

            mean_r = np.mean(img_resize[:,:,2])
            mean_g = np.mean(img_resize[:,:,1])
            mean_b = np.mean(img_resize[:,:,0])

            rgb = {
            "r": round(mean_r, 2),
            "g": round(mean_g, 2),
            "b": round(mean_b, 2)
            }

            # =====================
            # GLCM
            # =====================

            glcm = graycomatrix(
                gray,
                [1],
                [0],
                256,
                symmetric=True,
                normed=True
            )

            contrast = graycoprops(glcm,'contrast')[0,0]
            energy = graycoprops(glcm,'energy')[0,0]
            homogeneity = graycoprops(glcm,'homogeneity')[0,0]
            dissimilarity = graycoprops(glcm,'dissimilarity')[0,0]
            correlation = graycoprops(glcm,'correlation')[0,0]
            asm = graycoprops(glcm,'ASM')[0,0]

            glcm_data = {
                "contrast": round(contrast, 4),
                "energy": round(energy, 4),
                "homogeneity": round(homogeneity, 4),
                "dissimilarity": round(dissimilarity, 4),
                "correlation": round(correlation, 4),
                "asm": round(asm, 4)
            }

            fitur = [[
                mean_r,
                mean_g,
                mean_b,
                contrast,
                energy,
                homogeneity,
                dissimilarity,
                correlation,
                asm
            ]]

            fitur = scaler.transform(fitur)

            prediksi = model.predict(fitur)

            if prediksi[0] == 1:
                hasil = "BAIK"
            else:
                hasil = "BURUK"

            conn = sqlite3.connect('database_klasifikasi.db')
            cursor = conn.cursor()

            cursor.execute("""
            INSERT INTO history(
            id_user,
            gambar,
            hasil,
            r,
            g,
            b,
            contrast,
            energy,
            homogeneity,
            dissimilarity,
            correlation,
            asm
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                session['id_user'],
                gambar,
                hasil,
                mean_r,
                mean_g,
                mean_b,
                contrast,
                energy,
                homogeneity,
                dissimilarity,
                correlation,
                asm
            ))
            conn.commit()
            conn.close()

    return render_template(
        'klasifikasi.html',
        hasil=hasil,
        gambar=gambar,
        rgb=rgb,
        glcm_data=glcm_data
    )

@app.route('/profil')
def profile():
    if 'email' not in session:
        return redirect('/login')

    conn = sqlite3.connect('database_klasifikasi.db')
    cursor = conn.cursor()

    cursor.execute(
        "SELECT nama, email, umur, pekerjaan, foto_profil FROM users WHERE id_user=?",
        (session['id_user'],)
    )
    user = cursor.fetchone()

    cursor.execute("""
    SELECT id_history, 
    gambar,
    hasil,
    r,
    g,
    b,
    contrast,
    energy,
    homogeneity,
    dissimilarity,
    correlation,
    asm,
    tanggal
    FROM history 
    WHERE id_user=?
    ORDER BY id_history DESC
    """, (session['id_user'],))

    history = cursor.fetchall()

    print(history)

    conn.close()
        
    return render_template(
        'profil.html',
        user=user,
        history=history
    )

@app.route('/edit_profil', methods=['POST'])
def edit_profil():

    if 'id_user' not in session:
        return redirect('/login')

    nama = request.form['nama']
    umur = request.form['umur']
    pekerjaan = request.form['pekerjaan']

    conn = sqlite3.connect('database_klasifikasi.db')
    cursor = conn.cursor()

    cursor.execute("""
    UPDATE users SET
        nama=?,
        umur=?,
        pekerjaan=?
    WHERE id_user=?
    """, (nama, umur, pekerjaan, session['id_user']
    ))

    conn.commit()
    conn.close()

    session['nama'] = nama

    return redirect('/profil')

@app.route('/upload_foto', methods=['POST'])
def upload_foto():

    if 'id_user' not in session:
        return redirect('/login')

    file = request.files['foto']

    if file.filename == '':
        return redirect('/profil')

    nama_file = secure_filename(file.filename)

    lokasi = os.path.join(
        'static',
        'profil',
        nama_file
    )

    file.save(lokasi)

    conn = sqlite3.connect('database_klasifikasi.db')
    cursor = conn.cursor()

    cursor.execute("""
    UPDATE users
    SET foto_profil=?
    WHERE id_user=?
    """, (
        nama_file,
        session['id_user']
    ))

    conn.commit()
    conn.close()

    session['foto_profil'] = nama_file

    return redirect('/profil')

@app.route('/upload', methods=['POST'])
def upload():
    file = request.files['gambar']
    if file:
        lokasi = os.path.join('static/uploads', file.filename)
        file.save(lokasi)
        return "Gambar berhasil diupload"
    return "Upload Gagal"

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/logout')
def logout():
    session.clear()
    return redirect('/')

if __name__ == '__main__':
    app.run(debug=True)