import yt_dlp
from db import get_db

DUREE_MIN_SECONDES = 30
DUREE_MAX_SECONDES = 1200

def scrape_youtube_urls(vehicule_name, target_minutes):
    print(f"🔍 Recherche YouTube pour : '{vehicule_name}'")
    target_seconds = target_minutes * 60

    conn = get_db()
    c = conn.cursor()

    # On vérifie si on a déjà assez de vidéos en base pour ce véhicule
    c.execute("SELECT SUM(duration) FROM videos WHERE vehicule = ?", (vehicule_name,))
    total_seconds_cumules = c.fetchone()[0] or 0

    if total_seconds_cumules >= target_seconds:
        print(f"✅ Quota de {target_minutes} min déjà atteint en base de données.")
        conn.close()
        return total_seconds_cumules // 60

    ydl_opts = {
        'quiet': True,
        'extract_flat': True, # Extraction rapide sans télécharger les vidéos
        'force_generic_extractor': False
    }

    # ytsearch50 va chercher les 50 premiers résultats YouTube
    search_query = f"ytsearch50:{vehicule_name} exhaust sound pure"

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(search_query, download=False)

        if 'entries' in info:
            for entry in info['entries']:
                url = entry.get('url')
                title = entry.get('title')
                duration = entry.get('duration') or 0
                description = entry.get('description') or ""

                if not url or duration < DUREE_MIN_SECONDES or duration > DUREE_MAX_SECONDES:
                    continue

                if not url.startswith('http'):
                    url = f"https://www.youtube.com/watch?v={url}"

                try:
                    # Insertion sécurisée (ignore si l'URL existe déjà)
                    c.execute("""
                        INSERT INTO videos (vehicule, url, title, description, duration, status)
                        VALUES (?, ?, ?, ?, ?, 'scraped')
                    """, (vehicule_name, url, title, description, duration))
                    
                    conn.commit()
                    total_seconds_cumules += duration
                    print(f"   ➕ [{duration//60:02d}:{duration%60:02d}] (Total : {total_seconds_cumules // 60} min / {target_minutes} min)")

                    if total_seconds_cumules >= target_seconds:
                        break
                except Exception:
                    # Si l'URL est déjà dans la base de données, on passe
                    continue

    conn.close()
    minutes_finales = total_seconds_cumules // 60
    print(f"\n✅ Scraping terminé. {minutes_finales} minutes enregistrées en base SQLite.")
    return minutes_finales