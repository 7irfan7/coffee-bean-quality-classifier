import sqlite3

conn = sqlite3.connect('database_klasifikasi.db')
cursor = conn.cursor()

cursor.execute("""DELETE FROM history""")

conn.commit()
conn.close()

print("berhasil dihapus")