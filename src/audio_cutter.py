import os
from pydub import AudioSegment
from concurrent.futures import ProcessPoolExecutor, as_completed

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR  = os.path.join(BASE_DIR, "../data/03_audio")
OUTPUT_DIR = os.path.join(BASE_DIR, "../data/04_audio_chunks")

CHUNK_LENGTH_MS        = 3000
SILENCE_THRESHOLD_DBFS = -40.0


def process_file(fichier):
    """Worker multiprocessus — découpe un fichier audio en chunks."""
    chemin_complet = os.path.join(INPUT_DIR, fichier)
    nom_base       = fichier.replace('.wav', '')

    try:
        audio = AudioSegment.from_wav(chemin_complet)
        audio = audio.normalize()

        duree_totale     = len(audio)
        chunks_sauvegardes = 0

        for i in range(0, duree_totale, CHUNK_LENGTH_MS):
            chunk = audio[i:i + CHUNK_LENGTH_MS]
            if len(chunk) < CHUNK_LENGTH_MS:
                continue
            if chunk.dBFS <= SILENCE_THRESHOLD_DBFS:
                continue

            nom_chunk = f"{nom_base}_chunk_{chunks_sauvegardes:04d}.wav"
            chunk.export(os.path.join(OUTPUT_DIR, nom_chunk), format="wav")
            chunks_sauvegardes += 1

        return fichier, chunks_sauvegardes, None

    except Exception as e:
        return fichier, 0, str(e)


def cut_audio_into_chunks(log_fn=print):
    log_fn("✂️ Démarrage de l'usine de découpe (multi-processus)...\n")

    if not os.path.exists(INPUT_DIR):
        raise Exception(f"Le dossier source {INPUT_DIR} est introuvable.")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    fichiers_audio = [f for f in os.listdir(INPUT_DIR) if f.endswith('.wav')]

    if not fichiers_audio:
        log_fn(f"❌ Aucun fichier .wav trouvé dans {INPUT_DIR}.")
        return

    total_chunks_generes = 0

    with ProcessPoolExecutor() as executor:
        futures = {executor.submit(process_file, f): f for f in fichiers_audio}
        for future in as_completed(futures):
            fichier, nb_chunks, erreur = future.result()
            if erreur:
                log_fn(f"   ❌ Erreur sur {fichier} : {erreur}")
            else:
                log_fn(f"   ✅ {fichier} → {nb_chunks} chunks valides.")
                total_chunks_generes += nb_chunks

    log_fn(f"\n🎉 Usine terminée ! {total_chunks_generes} chunks valides générés.")


if __name__ == "__main__":
    cut_audio_into_chunks()