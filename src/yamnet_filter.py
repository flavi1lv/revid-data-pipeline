import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import shutil
import csv
import numpy as np
import librosa
import tensorflow as tf
import tensorflow_hub as hub

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR = os.path.join(BASE_DIR, "../data/04_audio_chunks")
VALID_DIR = os.path.join(BASE_DIR, "../data/05_audio_valide")
REJECT_DIR = os.path.join(BASE_DIR, "../data/05_audio_rejete")

# Sons que l'on veut GARDER
MOTS_CLES_MOTEUR = ["Engine", "Vehicle", "Car", "Motor vehicle", "Exhaust", "Motorcycle", "Race car"]

# Sons qui contaminent le dataset → rejet même si un moteur est présent en fond
# Un chunk avec de la musique plus forte que le moteur fausse l'entraînement
MOTS_CLES_PARASITES = ["Music", "Musical instrument", "Speech", "Singing", "Voice",
                        "Song", "Narration", "Conversation", "Male speech", "Female speech"]

# Seuil de confiance minimum pour accepter un son comme "moteur"
# En dessous de 12%, le signal est trop ambigu pour être utile au dataset
SEUIL_MOTEUR = 0.12

def run_intelligent_filter():
    print("🧠 Démarrage du Filtre IA YAMNet (mode précision + anti-musique)...\n")

    os.makedirs(VALID_DIR, exist_ok=True)
    os.makedirs(REJECT_DIR, exist_ok=True)

    fichiers = [f for f in os.listdir(INPUT_DIR) if f.endswith('.wav')]
    if not fichiers:
        print("❌ Aucun chunk trouvé. Lance d'abord la découpe.")
        return

    print("⏳ Chargement du modèle YAMNet (30s la première fois)...")
    model = hub.load('https://tfhub.dev/google/yamnet/1')

    class_map_path = model.class_map_path().numpy()
    class_names = []
    with tf.io.gfile.GFile(class_map_path) as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            class_names.append(row['display_name'])

    # Pré-calcul des indices une seule fois avant la boucle principale
    # → évite de reparser les 521 noms à chaque chunk (gain de temps réel)
    indices_moteur = [
        i for i, name in enumerate(class_names)
        if any(mot in name for mot in MOTS_CLES_MOTEUR)
    ]
    indices_parasites = [
        i for i, name in enumerate(class_names)
        if any(mot in name for mot in MOTS_CLES_PARASITES)
    ]

    print(f"✅ Modèle chargé !")
    print(f"   🏎️  {len(indices_moteur)} classes moteur surveillées")
    print(f"   🎵  {len(indices_parasites)} classes parasites bloquées")
    print(f"   🎯  Seuil de confiance moteur : {SEUIL_MOTEUR*100:.0f}%")
    print(f"\n🔍 Tri de {len(fichiers)} fichiers en cours...\n")

    valides = 0
    rejetes_parasite = 0
    rejetes_trop_faible = 0

    for fichier in fichiers:
        chemin_source = os.path.join(INPUT_DIR, fichier)

        try:
            # YAMNet attend du 16000 Hz mono — on force le resampling ici
            waveform, _ = librosa.load(chemin_source, sr=16000, mono=True)

            # Un seul forward pass → 521 scores calculés simultanément
            scores, _, _ = model(waveform)
            moyenne_scores = np.mean(scores.numpy(), axis=0)

            # Score max parmi TOUTES les classes moteur (jamais top-1 seul)
            # Exemple : si "Car" est à 0.08 et "Engine" à 0.20, on prend 0.20
            score_moteur = float(np.max(moyenne_scores[indices_moteur])) if indices_moteur else 0.0

            # Score max parmi TOUTES les classes parasites
            score_parasite = float(np.max(moyenne_scores[indices_parasites])) if indices_parasites else 0.0

            # DOUBLE CONDITION pour un dataset propre :
            # 1. Le moteur est suffisamment présent (> seuil absolu)
            # 2. Le moteur domine sur la musique/parole
            est_moteur_pur = (score_moteur >= SEUIL_MOTEUR) and (score_moteur >= score_parasite)

            if est_moteur_pur:
                shutil.copy(chemin_source, os.path.join(VALID_DIR, fichier))
                valides += 1
            else:
                if score_parasite > score_moteur:
                    raison = f"parasite dominant ({score_parasite*100:.1f}% > moteur {score_moteur*100:.1f}%)"
                    rejetes_parasite += 1
                else:
                    raison = f"moteur trop faible ({score_moteur*100:.1f}% < {SEUIL_MOTEUR*100:.0f}%)"
                    rejetes_trop_faible += 1
                print(f"🔴 REJETÉ : {fichier} → {raison}")
                shutil.copy(chemin_source, os.path.join(REJECT_DIR, fichier))

        except Exception as e:
            print(f"❌ Erreur sur {fichier} : {e}")

    total = max(valides + rejetes_parasite + rejetes_trop_faible, 1)
    taux_proprete = valides / total * 100

    print("\n==========================================")
    print("📊 BILAN DU TRI INTELLIGENT")
    print(f"✅ Sons moteur purs (Gardés)          : {valides}")
    print(f"🎵 Parasites musique/parole (Rejetés) : {rejetes_parasite}")
    print(f"❓ Signal moteur trop faible (Rejetés): {rejetes_trop_faible}")
    print(f"📈 Taux de pureté du dataset          : {taux_proprete:.1f}%")
    print("==========================================")

if __name__ == "__main__":
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    run_intelligent_filter()