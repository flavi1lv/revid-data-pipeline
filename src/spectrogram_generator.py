import os
import librosa
import librosa.display
import matplotlib
matplotlib.use('Agg')  # Backend non-interactif — obligatoire pour le multiprocessing
import matplotlib.pyplot as plt
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR = os.path.join(BASE_DIR, "../data/05_audio_valide")
OUTPUT_ROOT = os.path.join(BASE_DIR, "../data/06_spectrograms")

# 224x224 pour compatibilité directe avec EfficientNetB0 (et la plupart des CNNs modernes)
IMG_SIZE_PX = 224
DPI = 100

def extraire_nom_classe(nom_fichier):
    """
    Extraction robuste du nom de classe depuis le nom de fichier.
    """
    sans_extension = nom_fichier.replace('.wav', '').replace('.png', '')
    # Coupe au niveau de "_chunk_" qui est notre délimiteur fiable
    partie_avant_chunk = sans_extension.split('_chunk_')[0]
    nom_classe = partie_avant_chunk[:-12]
    return nom_classe

def generate_single(args):
    """
    Génère un spectrogramme mel pour un fichier audio.
    Conçu pour être appelé en parallèle — chaque worker crée sa propre figure.
    """
    fichier, output_root = args
    chemin_audio = os.path.join(INPUT_DIR, fichier)

    nom_classe = extraire_nom_classe(fichier)
    dossier_cible = os.path.join(output_root, nom_classe)
    os.makedirs(dossier_cible, exist_ok=True)

    chemin_image = os.path.join(dossier_cible, fichier.replace('.wav', '.png'))

    # Skip si déjà généré (reprise possible)
    if os.path.exists(chemin_image):
        return True

    try:
        y, sr = librosa.load(chemin_audio, sr=16000)

        # Spectrogramme Mel : représentation fréquentielle adaptée à la perception humaine
        S = librosa.feature.melspectrogram(
            y=y, sr=sr, n_mels=128, fmax=8000, n_fft=2048, hop_length=512
        )
        S_dB = librosa.power_to_db(S, ref=np.max)

        # Normalisation [0, 1] par image : deux voitures enregistrées
        # à des volumes différents auront des spectrogrammes comparables
        S_dB = (S_dB - S_dB.min()) / (S_dB.max() - S_dB.min() + 1e-8)

        # Taille fixe en pixels : DPI × taille_pouces = pixels exacts
        taille_pouces = IMG_SIZE_PX / DPI
        fig, ax = plt.subplots(figsize=(taille_pouces, taille_pouces), dpi=DPI)
        ax.axis('off')
        librosa.display.specshow(S_dB, sr=sr, fmax=8000, ax=ax, cmap='magma')

        plt.savefig(chemin_image, bbox_inches='tight', pad_inches=0, dpi=DPI)
        plt.close(fig)  # Libération mémoire critique en multiprocessing

        return True

    except Exception as e:
        print(f"❌ Erreur sur {fichier} : {e}")
        return False

def generate_spectrograms():
    print("🎨 Démarrage de l'Atelier Visuel (multi-processus, 224×224px)...\n")

    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    fichiers = [f for f in os.listdir(INPUT_DIR) if f.endswith('.wav')]

    if not fichiers:
        print("❌ Aucun audio trouvé dans le dossier validé.")
        return

    print(f"📦 {len(fichiers)} audios à transformer...\n")

    succes = 0
    echecs = 0
    args_list = [(f, OUTPUT_ROOT) for f in fichiers]

    with ProcessPoolExecutor() as executor:
        futures = {executor.submit(generate_single, args): args[0] for args in args_list}
        for i, future in enumerate(as_completed(futures), 1):
            if future.result():
                succes += 1
            else:
                echecs += 1
            if i % 100 == 0:
                print(f"   ⏳ {i}/{len(fichiers)} images générées...")

    print(f"\n🎉 Terminé ! {succes} spectrogrammes générés, {echecs} échecs.")
    print(f"📁 Organisés par classe dans : {OUTPUT_ROOT}")

if __name__ == "__main__":
    generate_spectrograms()