import os
import yt_dlp
from db import get_db

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "../data/03_audio")

def download_audio(vehicule_name):
    print("🎧 Démarrage du téléchargement (Connecté à SQLite)...\n")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    conn = get_db()
    c = conn.cursor()

    # On ne télécharge que les vidéos nettoyées qui n'ont pas encore été téléchargées
    c.execute("SELECT id, clean_name, url FROM videos WHERE vehicule = ? AND status = 'cleaned'", (vehicule_name,))
    a_telecharger = [dict(row) for row in c.fetchall()]

    if not a_telecharger:
        print("   ✅ Toutes les vidéos valides sont déjà téléchargées.")
        conn.close()
        return

    print(f"📦 {len(a_telecharger)} nouvelles vidéos à télécharger.")

    for index, video in enumerate(a_telecharger, 1):
        nom_propre = video["clean_name"].replace(" ", "_")
        url = video["url"]
        vid_id = video["id"]

        print(f"⏳ [{index}/{len(a_telecharger)}] {video['clean_name']}...")

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
            # Mise à jour de la DB après un téléchargement réussi
            c.execute("UPDATE videos SET status = 'downloaded' WHERE id = ?", (vid_id,))
            conn.commit()
            print(f"   ✅ Succès !")
        except Exception as e:
            print(f"   ❌ Erreur de téléchargement : {e}")

    conn.close()
    print(f"\n🎉 Téléchargements terminés pour ce lot.")