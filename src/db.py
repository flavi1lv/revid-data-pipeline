import sqlite3
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "../data/revid.db")

def get_db():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row # Permet d'accéder aux colonnes par leur nom
    return conn

def init_db():
    conn = get_db()
    cursor = conn.cursor()
    
    # Table principale pour les vidéos
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS videos (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            vehicule TEXT,
            url TEXT UNIQUE,
            title TEXT,
            description TEXT,
            duration INTEGER,
            clean_name TEXT,
            status TEXT DEFAULT 'scraped' 
        )
    ''') # Status possibles : 'scraped', 'cleaned', 'invalid', 'downloaded'
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS pipeline_state (
            vehicule TEXT PRIMARY KEY,
            etape_actuelle INTEGER,
            minutes_reelles INTEGER
        )
    ''')
    
    conn.commit()
    conn.close()

init_db()