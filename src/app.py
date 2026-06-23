import os
import sys
import json
import shutil
import threading
import time
import logging
import tempfile
import numpy as np
import librosa
import librosa.display
import matplotlib
from collections import deque
from werkzeug.utils import secure_filename
from flask import Flask, request, jsonify, render_template, Response
from flask_cors import CORS
from orchestrator import run_full_pipeline

# OBLIGATOIRE POUR FLASK : Empêche Matplotlib de chercher un écran d'affichage
matplotlib.use('Agg')
import matplotlib.pyplot as plt

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# ── IMPORTS DES MODULES IA ──
try:
    from train_model import run_training
    TRAINING_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Erreur d'import (Entraînement) : {e}")
    TRAINING_AVAILABLE = False

try:
    from predict import load_revid_model, predict_spectrogram
    model_ai, class_names_ai = load_revid_model()
    IA_READY = model_ai is not None
except Exception as e:
    print(f"⚠️ IA de prédiction non chargée : {e}")
    IA_READY = False

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "../data/06_spectrograms")
TEMPLATE_DIR = os.path.join(BASE_DIR, "templates")

app = Flask(__name__, template_folder=TEMPLATE_DIR)
CORS(app)

log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)


# ── SÉCURISATION CONCURRENCE (THREAD-SAFE) ──
status_lock = threading.Lock()
pipeline_status = {
    "running": False,
    "type": None,       # "generate" ou "train"
    "vehicule": None,
    "message": "En attente...",
    "result": None
}

log_buffer = deque(maxlen=500)

def stream_log(text):
    """Fonction d'injection de log thread-safe et directe (sans patcher stdout)."""
    stripped = text.strip()
    if stripped:
        log_buffer.append(stripped)

def get_dataset_stats():
    if not os.path.exists(DATASET_DIR):
        return []
    stats = []
    for dossier in os.listdir(DATASET_DIR):
        chemin_dossier = os.path.join(DATASET_DIR, dossier)
        if os.path.isdir(chemin_dossier):
            nb_images = len([f for f in os.listdir(chemin_dossier) if f.endswith('.png')])
            stats.append({
                "nom_dossier": dossier,
                "nom_propre": dossier.replace("_", " "),
                "nb_images": nb_images
            })
    return sorted(stats, key=lambda x: x["nom_propre"])

def audio_to_spectrogram(audio_path, output_image_path):
    """Convertit un audio de 10s max en image 224x224 pour l'IA"""
    # On charge l'audio (limité à 10 secondes pour aller vite)
    y, sr = librosa.load(audio_path, duration=10.0)
    
    # Création du Mel-spectrogramme (les mêmes réglages que pour ton entraînement)
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    S_dB = librosa.power_to_db(S, ref=np.max)
    
    # 224x224 pixels (Idéal pour EfficientNet)
    fig = plt.figure(figsize=(2.24, 2.24), dpi=100)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    
    librosa.display.specshow(S_dB, sr=sr, ax=ax)
    plt.savefig(output_image_path, format='png')
    plt.close(fig)

@app.route('/')
def home():
    # Rendu propre du fichier séparé
    return render_template('dashboard.html')

@app.route('/api/dataset', methods=['GET'])
def api_dataset():
    return jsonify(get_dataset_stats())

@app.route('/api/status', methods=['GET'])
def api_status():
    with status_lock:
        return jsonify(pipeline_status)

@app.route('/api/logs')
def api_logs():
    def generate():
        sent_count = 0
        while True:
            with status_lock:
                is_running = pipeline_status["running"]
            
            buffer_size = len(log_buffer)
            while sent_count < buffer_size:
                log_line = list(log_buffer)[sent_count]
                yield f"data: {json.dumps({'log': log_line})}\n\n"
                sent_count += 1
                
            if not is_running and sent_count >= len(log_buffer):
                break
            time.sleep(0.2)
        yield f"data: {json.dumps({'done': True})}\n\n"

    return Response(generate(), mimetype='text/event-stream',
                    headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'})

@app.route('/api/generate', methods=['POST'])
def generate():
    with status_lock:
        if pipeline_status["running"]:
            return jsonify({"status": "error", "message": "Un pipeline tourne déjà."}), 409

    data = request.json
    vehicule = data.get("vehicle")
    target_minutes = data.get("target_minutes", 60)

    if not vehicule:
        return jsonify({"status": "error", "message": "Nom de véhicule manquant."}), 400

    dossier = os.path.join(DATASET_DIR, vehicule.upper().replace(" ", "_"))
    if os.path.exists(dossier):
        shutil.rmtree(dossier)

    def run_async():
        with status_lock:
            pipeline_status.update({"running": True, "type": "generate", "vehicule": vehicule, "result": None})
        log_buffer.clear()

        try:
            # Note : Idéalement, modifier run_full_pipeline pour accepter log_fn=stream_log
            # En attendant, on utilise la capture sécurisée localisée à ce thread si nécessaire.
            result = run_full_pipeline(vehicule, target_minutes=target_minutes)
            with status_lock:
                pipeline_status["result"] = result["status"]
                pipeline_status["message"] = result["message"]
        except Exception as e:
            with status_lock:
                pipeline_status["result"] = "error"
                pipeline_status["message"] = str(e)
        finally:
            with status_lock:
                pipeline_status["running"] = False

    threading.Thread(target=run_async, daemon=True).start()
    return jsonify({"status": "started"})

@app.route('/api/delete', methods=['POST'])
def delete_vehicle():
    dossier = request.json.get("folder")
    if not dossier:
        return jsonify({"status": "error", "message": "Dossier manquant."}), 400
    chemin = os.path.join(DATASET_DIR, dossier)
    if os.path.exists(chemin):
        shutil.rmtree(chemin)
    return jsonify({"status": "success"})

@app.route('/api/train', methods=['POST'])
def train_api():
    if not TRAINING_AVAILABLE:
        return jsonify({"status": "error", "message": "train_model.py introuvable."}), 500
    with status_lock:
        if pipeline_status["running"]:
            return jsonify({"status": "error", "message": "Un pipeline tourne déjà."}), 409

    data = request.json or {}
    epochs_p1 = int(data.get("epochs_p1", 15))
    epochs_p2 = int(data.get("epochs_p2", 10))

    def run_train_async():
        with status_lock:
            pipeline_status.update({"running": True, "type": "train", "result": None})
        log_buffer.clear()

        try:
            run_training(log_fn=stream_log, epochs_p1=epochs_p1, epochs_p2=epochs_p2)
            with status_lock:
                pipeline_status["result"] = "success"
                pipeline_status["message"] = "Modèle entraîné avec succès."
        except Exception as e:
            with status_lock:
                pipeline_status["result"] = "error"
                pipeline_status["message"] = str(e)
        finally:
            with status_lock:
                pipeline_status["running"] = False

    threading.Thread(target=run_train_async, daemon=True).start()
    return jsonify({"status": "started"})

@app.route('/api/predict', methods=['POST'])
def api_predict():
    if not IA_READY:
        return jsonify({"error": "L'IA n'est pas prête ou le modèle est introuvable."}), 500

    if 'file' not in request.files:
        return jsonify({"error": "Aucun fichier audio envoyé."}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "Fichier vide."}), 400

    temp_dir = tempfile.gettempdir()
    audio_filepath = os.path.join(temp_dir, secure_filename(file.filename))
    spectro_filepath = os.path.join(temp_dir, "temp_spectro.png")
    
    file.save(audio_filepath)

    try:
        # Transforme l'audio en Spectrogramme
        audio_to_spectrogram(audio_filepath, spectro_filepath)
        
        # L'IA analyse l'image générée
        vehicule, confiance = predict_spectrogram(spectro_filepath, model_ai, class_names_ai)
        
        # On nettoie
        os.remove(audio_filepath)
        os.remove(spectro_filepath)

        return jsonify({
            "vehicule": vehicule.replace("_", " "), 
            "confiance": round(float(confiance), 2)
        })
    except Exception as e:
        if os.path.exists(audio_filepath):
            os.remove(audio_filepath)
        if os.path.exists(spectro_filepath):
            os.remove(spectro_filepath)
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    print("🌍 Lancement du Dashboard MLOps REVID propre sur http://localhost:5000")
    app.run(debug=True, port=5000, use_reloader=False, threaded=True)