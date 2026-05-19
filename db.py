import sqlite3

# ---------------- CREATE TABLES ----------------
def create_tables():
    conn = sqlite3.connect("database.db")
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE,
            password TEXT
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT,
            filename TEXT,
            result TEXT,
            confidence REAL
        )
    """)

    conn.commit()
    conn.close()


# ---------------- ADD USER ----------------
def add_user(username, password):
    conn = sqlite3.connect("database.db")
    cursor = conn.cursor()

    try:
        cursor.execute(
            "INSERT INTO users (username, password) VALUES (?, ?)",
            (username, password)
        )
        conn.commit()
        success = True
    except:
        success = False

    conn.close()
    return success


# ---------------- VERIFY USER ----------------
def verify_user(username, password):
    conn = sqlite3.connect("database.db")
    cursor = conn.cursor()

    cursor.execute(
        "SELECT * FROM users WHERE username=? AND password=?",
        (username, password)
    )

    user = cursor.fetchone()
    conn.close()

    return user is not None


# ---------------- ADD HISTORY ----------------
def add_history(username, filename, result, confidence):
    conn = sqlite3.connect("database.db")
    cursor = conn.cursor()

    cursor.execute(
        "INSERT INTO history (username, filename, result, confidence) VALUES (?, ?, ?, ?)",
        (username, filename, result, confidence)
    )

    conn.commit()
    conn.close()


# ---------------- GET HISTORY ----------------
def get_history(username):
    conn = sqlite3.connect("database.db")
    cursor = conn.cursor()

    cursor.execute(
        "SELECT username, filename, result, confidence FROM history WHERE username=?",
        (username,)
    )

    data = cursor.fetchall()
    conn.close()

    return data
