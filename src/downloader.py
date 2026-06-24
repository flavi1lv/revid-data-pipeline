import os
import yt_dlp
from db import get_db

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "../data/03_audio")


def download_audio(vehicule_name, log_fn=print):
    log_fn("🎧 Démarrage du téléchargement (Connecté à SQLite)...\n")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    conn = get_db()
    c    = conn.cursor()

    c.execute("SELECT id, clean_name, url FROM videos WHERE vehicule = ? AND status = 'cleaned'", (vehicule_name,))
    a_telecharger = [dict(row) for row in c.fetchall()]

    if not a_telecharger:
        log_fn("   ✅ Toutes les vidéos valides sont déjà téléchargées.")
        conn.close()
        return

    log_fn(f"📦 {len(a_telecharger)} nouvelles vidéos à télécharger.")

    for index, video in enumerate(a_telecharger, 1):
        nom_propre = video["clean_name"].replace(" ", "_")
        url        = video["url"]
        vid_id     = video["id"]

        log_fn(f"⏳ [{index}/{len(a_telecharger)}] {video['clean_name']}...")

        ydl_opts = {
            'format': 'bestaudio/best',
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'wav',
                'preferredquality': '192',
            }],
            'outtmpl': os.path.join(OUTPUT_DIR, f"{nom_propre}_%(id)s.%(ext)s"),
            'quiet': True,
            'no_warnings': True,
            'concurrent_fragment_downloads': 4,
        }

        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([url])
            c.execute("UPDATE videos SET status = 'downloaded' WHERE id = ?", (vid_id,))
            conn.commit()
            log_fn(f"   ✅ Succès !")
        except Exception as e:
            log_fn(f"   ❌ Erreur de téléchargement : {e}")

    conn.close()
    log_fn(f"\n🎉 Téléchargements terminés pour ce lot.")