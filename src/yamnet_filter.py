import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import shutil
import csv
import numpy as np
import librosa
import tensorflow as tf
import tensorflow_hub as hub

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR = os.path.join(BASE_DIR, "../data/04_audio_chunks")
VALID_DIR = os.path.join(BASE_DIR, "../data/05_audio_valide")

MOTS_CLES_MOTEUR   = ["Engine", "Vehicle", "Car", "Motor vehicle", "Exhaust", "Motorcycle", "Race car"]
MOTS_CLES_PARASITES = ["Music", "Musical instrument", "Speech", "Singing", "Voice",
                        "Song", "Narration", "Conversation", "Male speech", "Female speech"]

SEUIL_MOTEUR = 0.12


def run_intelligent_filter(log_fn=print):
    log_fn("🧠 Démarrage du Filtre IA YAMNet (mode précision + anti-musique)...\n")

    os.makedirs(VALID_DIR, exist_ok=True)

    fichiers = [f for f in os.listdir(INPUT_DIR) if f.endswith('.wav')]
    if not fichiers:
        log_fn("❌ Aucun chunk trouvé. Lance d'abord la découpe.")
        return

    log_fn("⏳ Chargement du modèle YAMNet (30s la première fois)...")
    model = hub.load('https://tfhub.dev/google/yamnet/1')

    class_map_path = model.class_map_path().numpy()
    class_names = []
    with tf.io.gfile.GFile(class_map_path) as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            class_names.append(row['display_name'])

    indices_moteur    = [i for i, name in enumerate(class_names) if any(mot in name for mot in MOTS_CLES_MOTEUR)]
    indices_parasites = [i for i, name in enumerate(class_names) if any(mot in name for mot in MOTS_CLES_PARASITES)]

    log_fn(f"✅ Modèle chargé !")
    log_fn(f"   🏎️  {len(indices_moteur)} classes moteur surveillées")
    log_fn(f"   🎵  {len(indices_parasites)} classes parasites bloquées")
    log_fn(f"   🎯  Seuil de confiance moteur : {SEUIL_MOTEUR*100:.0f}%")
    log_fn(f"\n🔍 Tri de {len(fichiers)} fichiers en cours...\n")

    valides            = 0
    rejetes_parasite   = 0
    rejetes_trop_faible = 0

    for fichier in fichiers:
        chemin_source = os.path.join(INPUT_DIR, fichier)

        try:
            waveform, _ = librosa.load(chemin_source, sr=16000, mono=True)
            scores, _, _ = model(waveform)
            moyenne_scores = np.mean(scores.numpy(), axis=0)

            score_moteur   = float(np.max(moyenne_scores[indices_moteur]))   if indices_moteur   else 0.0
            score_parasite = float(np.max(moyenne_scores[indices_parasites])) if indices_parasites else 0.0

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
                log_fn(f"🔴 REJETÉ : {fichier} → {raison}")

        except Exception as e:
            log_fn(f"❌ Erreur sur {fichier} : {e}")

    total        = max(valides + rejetes_parasite + rejetes_trop_faible, 1)
    taux_proprete = valides / total * 100

    log_fn("\n==========================================")
    log_fn("📊 BILAN DU TRI INTELLIGENT")
    log_fn(f"✅ Sons moteur purs (Gardés)          : {valides}")
    log_fn(f"🎵 Parasites musique/parole (Rejetés) : {rejetes_parasite}")
    log_fn(f"❓ Signal moteur trop faible (Rejetés): {rejetes_trop_faible}")
    log_fn(f"📈 Taux de pureté du dataset          : {taux_proprete:.1f}%")
    log_fn("==========================================")


if __name__ == "__main__":
    run_intelligent_filter()