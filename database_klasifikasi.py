import sqlite3

conn = sqlite3.connect('database_klasifikasi.db')

cursor = conn.cursor()

cursor.execute('''
CREATE TABLE IF NOT EXISTS users(
    id_user INTEGER PRIMARY KEY AUTOINCREMENT,
    nama TEXT NOT NULL,
    email TEXT UNIQUE NOT NULL,
    umur INTEGER,
    pekerjaan TEXT,
    password TEXT NOT NULL,
    foto_profil TEXT DEFAULT 'default.png'
)
''')
cursor.execute('''
CREATE TABLE IF NOT EXISTS history(
    id_history INTEGER PRIMARY KEY AUTOINCREMENT,
    id_user INTEGER NOT NULL,

    gambar TEXT NOT NULL,
    hasil TEXT NOT NULL,

    r REAL,
    g REAL,
    b REAL,

    contrast REAL,
    energy REAL,
    homogeneity REAL,
    dissimilarity REAL,
    correlation REAL,
    asm REAL,
    
    tanggal TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(id_user) REFERENCES users(id_user)
)
''')

conn.commit()
conn.close()

print("Database Berhasil Dibuat")