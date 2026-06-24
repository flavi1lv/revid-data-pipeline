import yt_dlp
from db import get_db

DUREE_MIN_SECONDES = 30
DUREE_MAX_SECONDES = 1200


def scrape_youtube_urls(vehicule_name, target_minutes, log_fn=print):
    log_fn(f"🔍 Recherche YouTube pour : '{vehicule_name}'")
    target_seconds = target_minutes * 60

    conn = get_db()
    c    = conn.cursor()

    c.execute("SELECT SUM(duration) FROM videos WHERE vehicule = ?", (vehicule_name,))
    total_seconds_cumules = c.fetchone()[0] or 0

    if total_seconds_cumules >= target_seconds:
        log_fn(f"✅ Quota de {target_minutes} min déjà atteint en base de données.")
        conn.close()
        return total_seconds_cumules // 60

    ydl_opts = {
        'quiet': True,
        'extract_flat': True,
        'force_generic_extractor': False
    }

    search_query = f"ytsearch50:{vehicule_name} exhaust sound pure"

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(search_query, download=False)

        if 'entries' in info:
            for entry in info['entries']:
                url         = entry.get('url')
                title       = entry.get('title')
                duration    = entry.get('duration') or 0
                description = entry.get('description') or ""

                if not url or duration < DUREE_MIN_SECONDES or duration > DUREE_MAX_SECONDES:
                    continue

                if not url.startswith('http'):
                    url = f"https://www.youtube.com/watch?v={url}"

                # INSERT OR IGNORE : gère nativement les doublons sans masquer les vraies erreurs DB
                c.execute("""
                    INSERT OR IGNORE INTO videos (vehicule, url, title, description, duration, status)
                    VALUES (?, ?, ?, ?, ?, 'scraped')
                """, (vehicule_name, url, title, description, duration))
                conn.commit()

                # rowcount == 0 → URL déjà en base, durée déjà comptée
                if c.rowcount > 0:
                    total_seconds_cumules += duration
                    log_fn(f"   ➕ [{duration//60:02d}:{duration%60:02d}] (Total : {total_seconds_cumules // 60} min / {target_minutes} min)")

                if total_seconds_cumules >= target_seconds:
                    break

    conn.close()
    minutes_finales = total_seconds_cumules // 60
    log_fn(f"\n✅ Scraping terminé. {minutes_finales} minutes enregistrées en base SQLite.")
    return minutes_finales