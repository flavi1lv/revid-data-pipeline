import os
import re
import librosa
import librosa.display
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed

BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR    = os.path.join(BASE_DIR, "../data/05_audio_valide")
OUTPUT_ROOT  = os.path.join(BASE_DIR, "../data/06_spectrograms")

IMG_SIZE_PX  = 224
DPI          = 100
N_MELS       = 128
FMAX         = 8000
N_FFT        = 2048
HOP_LENGTH   = 512
SR           = 16000


def extraire_nom_classe(nom_fichier):
    """
    Extraction robuste du nom de classe depuis le nom de fichier.
    Format attendu : {CLEAN_NAME}_{YOUTUBE_ID_11_CHARS}_chunk_{NNNN}.wav
    On retire le YouTube ID (toujours 11 caractères alphanumériques) via regex.
    """
    sans_extension   = nom_fichier.replace('.wav', '').replace('.png', '')
    partie_avant_chunk = sans_extension.split('_chunk_')[0]
    # YouTube IDs : exactement 11 caractères [A-Za-z0-9_-]
    nom_classe = re.sub(r'_[A-Za-z0-9_\-]{11}$', '', partie_avant_chunk)
    return nom_classe


def generate_spectrogram_from_audio(audio_path, output_image_path):
    """
    Convertit un fichier audio en image spectrogramme 224×224.
    Paramètres identiques à generate_single — garantit la cohérence train/predict.
    Utilisé par app.py pour la prédiction en temps réel.
    """
    y, sr = librosa.load(audio_path, sr=SR, duration=10.0)

    S    = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_MELS, fmax=FMAX, n_fft=N_FFT, hop_length=HOP_LENGTH)
    S_dB = librosa.power_to_db(S, ref=np.max)
    S_dB = (S_dB - S_dB.min()) / (S_dB.max() - S_dB.min() + 1e-8)

    taille_pouces = IMG_SIZE_PX / DPI
    fig, ax = plt.subplots(figsize=(taille_pouces, taille_pouces), dpi=DPI)
    ax.axis('off')
    librosa.display.specshow(S_dB, sr=sr, fmax=FMAX, ax=ax, cmap='magma')
    plt.savefig(output_image_path, bbox_inches='tight', pad_inches=0, dpi=DPI)
    plt.close(fig)


def generate_single(args):
    """Worker multiprocessus — génère un spectrogramme pour un chunk audio."""
    fichier, output_root = args
    chemin_audio = os.path.join(INPUT_DIR, fichier)

    nom_classe    = extraire_nom_classe(fichier)
    dossier_cible = os.path.join(output_root, nom_classe)
    os.makedirs(dossier_cible, exist_ok=True)

    chemin_image = os.path.join(dossier_cible, fichier.replace('.wav', '.png'))

    if os.path.exists(chemin_image):
        return True

    try:
        y, sr = librosa.load(chemin_audio, sr=SR)

        S    = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_MELS, fmax=FMAX, n_fft=N_FFT, hop_length=HOP_LENGTH)
        S_dB = librosa.power_to_db(S, ref=np.max)
        S_dB = (S_dB - S_dB.min()) / (S_dB.max() - S_dB.min() + 1e-8)

        taille_pouces = IMG_SIZE_PX / DPI
        fig, ax = plt.subplots(figsize=(taille_pouces, taille_pouces), dpi=DPI)
        ax.axis('off')
        librosa.display.specshow(S_dB, sr=sr, fmax=FMAX, ax=ax, cmap='magma')
        plt.savefig(chemin_image, bbox_inches='tight', pad_inches=0, dpi=DPI)
        plt.close(fig)

        return True

    except Exception as e:
        print(f"❌ Erreur sur {fichier} : {e}")
        return False


def generate_spectrograms(log_fn=print):
    log_fn("🎨 Démarrage de l'Atelier Visuel (multi-processus, 224×224px)...\n")

    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    fichiers = [f for f in os.listdir(INPUT_DIR) if f.endswith('.wav')]

    if not fichiers:
        log_fn("❌ Aucun audio trouvé dans le dossier validé.")
        return

    log_fn(f"📦 {len(fichiers)} audios à transformer...\n")

    succes    = 0
    echecs    = 0
    args_list = [(f, OUTPUT_ROOT) for f in fichiers]

    with ProcessPoolExecutor() as executor:
        futures = {executor.submit(generate_single, args): args[0] for args in args_list}
        for i, future in enumerate(as_completed(futures), 1):
            if future.result():
                succes += 1
            else:
                echecs += 1
            if i % 100 == 0:
                log_fn(f"   ⏳ {i}/{len(fichiers)} images générées...")

    log_fn(f"\n🎉 Terminé ! {succes} spectrogrammes générés, {echecs} échecs.")
    log_fn(f"📁 Organisés par classe dans : {OUTPUT_ROOT}")


if __name__ == "__main__":
    generate_spectrograms()