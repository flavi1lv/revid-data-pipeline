import os
import sys
import json
import shutil
import threading
import time
import logging
import tempfile
import numpy as np
from collections import deque
from werkzeug.utils import secure_filename
from flask import Flask, request, jsonify, render_template, Response
from flask_cors import CORS
from orchestrator import run_full_pipeline
from spectrogram_generator import generate_spectrogram_from_audio

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL']  = '2'

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

BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR  = os.path.join(BASE_DIR, "../data/06_spectrograms")
TEMPLATE_DIR = os.path.join(BASE_DIR, "templates")

app = Flask(__name__, template_folder=TEMPLATE_DIR)
CORS(app)

log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

status_lock     = threading.Lock()
pipeline_status = {
    "running": False,
    "type":    None,
    "vehicule": None,
    "message": "En attente...",
    "result":  None
}

log_buffer = deque(maxlen=500)


def stream_log(text):
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
                "nom_propre":  dossier.replace("_", " "),
                "nb_images":   nb_images
            })
    return sorted(stats, key=lambda x: x["nom_propre"])


@app.route('/')
def home():
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

            buffer_snapshot = list(log_buffer)
            while sent_count < len(buffer_snapshot):
                log_line = buffer_snapshot[sent_count]
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

    data          = request.json
    vehicule      = data.get("vehicle")
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
            result = run_full_pipeline(vehicule, target_minutes=target_minutes, log_fn=stream_log)
            if not isinstance(result, dict) or "status" not in result:
                result = {"status": "error", "message": "Réponse inattendue du pipeline."}
            with status_lock:
                pipeline_status["result"]  = result["status"]
                pipeline_status["message"] = result.get("message", "")
        except Exception as e:
            with status_lock:
                pipeline_status["result"]  = "error"
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

    data      = request.json or {}
    epochs_p1  = int(data.get("epochs_p1",  15))
    epochs_p2  = int(data.get("epochs_p2",  10))
    batch_size = int(data.get("batch_size", 64))

    def run_train_async():
        with status_lock:
            pipeline_status.update({"running": True, "type": "train", "result": None})
        log_buffer.clear()

        try:
            run_training(log_fn=stream_log, epochs_p1=epochs_p1, epochs_p2=epochs_p2, batch_size=batch_size)
            with status_lock:
                pipeline_status["result"]  = "success"
                pipeline_status["message"] = "Modèle entraîné avec succès."
        except Exception as e:
            with status_lock:
                pipeline_status["result"]  = "error"
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

    temp_dir        = tempfile.gettempdir()
    audio_filepath  = os.path.join(temp_dir, secure_filename(file.filename))
    spectro_filepath = os.path.join(temp_dir, "temp_spectro.png")

    file.save(audio_filepath)

    try:
        # Mêmes paramètres qu'à l'entraînement — cohérence garantie via generate_spectrogram_from_audio
        generate_spectrogram_from_audio(audio_filepath, spectro_filepath)
        vehicule, confiance = predict_spectrogram(spectro_filepath, model_ai, class_names_ai)

        os.remove(audio_filepath)
        os.remove(spectro_filepath)

        return jsonify({
            "vehicule":  vehicule.replace("_", " "),
            "confiance": round(float(confiance), 2)
        })
    except Exception as e:
        if os.path.exists(audio_filepath):
            os.remove(audio_filepath)
        if os.path.exists(spectro_filepath):
            os.remove(spectro_filepath)
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    print("🌍 Lancement du Dashboard MLOps REVID sur http://localhost:5000")
    app.run(debug=True, port=5000, use_reloader=False, threaded=True)