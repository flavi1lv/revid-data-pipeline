import os
import json
import numpy as np
import tensorflow as tf

# On désactive les warnings inutiles de TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "../models/revid_model.keras")
CLASSES_PATH = os.path.join(BASE_DIR, "../models/classes.json")

def load_revid_model():
    """Charge le modèle et le dictionnaire de classes en mémoire."""
    if not os.path.exists(MODEL_PATH) or not os.path.exists(CLASSES_PATH):
        print("❌ Erreur : Modèle ou fichier classes.json introuvable dans le dossier 'models'.")
        return None, None
    
    print("🧠 Chargement du modèle en mémoire...")
    model = tf.keras.models.load_model(MODEL_PATH)
    
    with open(CLASSES_PATH, "r", encoding="utf-8") as f:
        class_names = json.load(f)
        
    return model, class_names

def predict_spectrogram(image_path, model, class_names):
    """Fait analyser une image de spectrogramme par l'IA."""
    # 1. Charger et préparer l'image (exactement comme à l'entraînement)
    img = tf.keras.utils.load_img(image_path, target_size=(224, 224))
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0) # Créer un "batch" virtuel d'une seule image

    # 2. Demander la prédiction à l'IA
    predictions = model.predict(img_array, verbose=0)
    score = tf.nn.softmax(predictions[0]) # Transforme les résultats en pourcentages

    # 3. Trouver la meilleure réponse
    predicted_class_index = np.argmax(score)
    predicted_class_name = class_names[predicted_class_index]
    confidence = 100 * np.max(score)

    return predicted_class_name, confidence

if __name__ == "__main__":
    # Test rapide
    model, class_names = load_revid_model()
    
    if model:
        print("\n✅ Modèle chargé avec succès ! Il connaît ces voitures :", class_names)
        print("\nPour tester l'IA, tu peux utiliser une fonction qui prendra un fichier audio,")
        print("créera son spectrogramme temporaire, et l'enverra à la fonction predict_spectrogram().")