import os
import shutil
from db import get_db

from scraper import scrape_youtube_urls
from cleaner_groq import clean_titles_with_ai
from downloader import download_audio
from audio_cutter import cut_audio_into_chunks
from yamnet_filter import run_intelligent_filter
from spectrogram_generator import generate_spectrograms

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "../data")

ETAPES = ["scrape", "clean", "download", "cut", "filter", "spectrogram"]


def load_checkpoint(vehicule):
    conn = get_db()
    c = conn.cursor()
    c.execute("SELECT etape_actuelle, minutes_reelles FROM pipeline_state WHERE vehicule = ?", (vehicule,))
    row = c.fetchone()
    conn.close()
    if row:
        return {"etape_actuelle": row['etape_actuelle'], "minutes_reelles": row['minutes_reelles']}
    return {"etape_actuelle": 0, "minutes_reelles": 0}


def save_checkpoint(vehicule, etape_index, minutes_reelles=0):
    conn = get_db()
    c = conn.cursor()
    c.execute("""
        REPLACE INTO pipeline_state (vehicule, etape_actuelle, minutes_reelles)
        VALUES (?, ?, ?)
    """, (vehicule, etape_index, minutes_reelles))
    conn.commit()
    conn.close()


def delete_checkpoint(vehicule):
    conn = get_db()
    c = conn.cursor()
    c.execute("DELETE FROM pipeline_state WHERE vehicule = ?", (vehicule,))
    conn.commit()
    conn.close()


def clean_temporary_folders():
    dossiers = ["03_audio", "04_audio_chunks", "05_audio_valide"]
    for dossier in dossiers:
        chemin = os.path.join(DATA_DIR, dossier)
        if os.path.exists(chemin):
            try:
                shutil.rmtree(chemin)
            except Exception as e:
                pass  # Non bloquant — les fichiers seront écrasés au prochain run


def verify_and_consolidate(target_name, log_fn=print):
    log_fn("\n🕵️ Vérification finale et consolidation des dossiers...")
    dataset_dir   = os.path.join(DATA_DIR, "06_spectrograms")
    target_folder = target_name.upper().replace(" ", "_")
    target_path   = os.path.join(dataset_dir, target_folder)

    if not os.path.exists(dataset_dir):
        return
    os.makedirs(target_path, exist_ok=True)

    for dossier in os.listdir(dataset_dir):
        chemin_dossier = os.path.join(dataset_dir, dossier)
        if not os.path.isdir(chemin_dossier) or dossier == target_folder:
            continue

        if dossier.startswith(target_folder) or target_folder.startswith(dossier):
            for fichier in os.listdir(chemin_dossier):
                shutil.move(os.path.join(chemin_dossier, fichier), os.path.join(target_path, fichier))
            log_fn(f"   🔄 Fusion automatique : '{dossier}' a été absorbé par '{target_folder}'.")

    for dossier in os.listdir(dataset_dir):
        chemin_dossier = os.path.join(dataset_dir, dossier)
        if os.path.isdir(chemin_dossier) and not os.listdir(chemin_dossier):
            os.rmdir(chemin_dossier)


def run_full_pipeline(vehicule_name, target_minutes=60, log_fn=print):
    log_fn(f"\n{'='*50}")
    log_fn(f"🚀 PIPELINE POUR : {vehicule_name}")
    log_fn(f"{'='*50}\n")

    checkpoint     = load_checkpoint(vehicule_name)
    etape_depart   = checkpoint["etape_actuelle"]
    minutes_reelles = checkpoint["minutes_reelles"]

    if etape_depart > 0:
        log_fn(f"♻️  Reprise depuis l'étape {etape_depart + 1} ({ETAPES[etape_depart]})...\n")

    etapes_config = [
        ("scrape",      lambda: scrape_youtube_urls(vehicule_name, target_minutes, log_fn=log_fn)),
        ("clean",       lambda: clean_titles_with_ai(vehicule_name, log_fn=log_fn)),
        ("download",    lambda: download_audio(vehicule_name, log_fn=log_fn)),
        ("cut",         lambda: cut_audio_into_chunks(log_fn=log_fn)),
        ("filter",      lambda: run_intelligent_filter(log_fn=log_fn)),
        ("spectrogram", lambda: generate_spectrograms(log_fn=log_fn)),
    ]

    try:
        for i, (nom_etape, fn) in enumerate(etapes_config):
            if i < etape_depart:
                log_fn(f"⏭️  Étape {i+1} ({nom_etape}) — déjà complétée.")
                continue

            log_fn(f"\n{'─'*50}")
            log_fn(f"📍 Étape {i+1}/{len(etapes_config)} : {nom_etape.upper()}")
            log_fn(f"{'─'*50}")

            result = fn()

            if nom_etape == "scrape":
                minutes_reelles = result or target_minutes

            save_checkpoint(vehicule_name, i + 1, minutes_reelles)

        log_fn("\n🧹 Nettoyage des fichiers temporaires...")
        clean_temporary_folders()
        delete_checkpoint(vehicule_name)
        verify_and_consolidate(vehicule_name, log_fn=log_fn)

        log_fn(f"\n✨ Pipeline terminé ! Dataset dans '06_spectrograms'.")
        return {
            "status": "success",
            "message": f"Dataset généré ({minutes_reelles} min d'audio réel)."
        }

    except Exception as e:
        log_fn(f"\n❌ Erreur critique à l'étape en cours : {e}")
        log_fn("⏸️  L'état est sauvegardé en base. Relancez le pipeline pour reprendre.")
        return {"status": "error", "message": str(e)}