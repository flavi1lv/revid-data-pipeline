import os
import json
import time
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras import layers, models

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "../data/06_spectrograms")
MODEL_DIR   = os.path.join(BASE_DIR, "../models")

CACHE_DIR_TRAIN = "/content/cache_train"
CACHE_DIR_VAL   = "/content/cache_val"

IMG_SIZE         = (224, 224)
VALIDATION_SPLIT = 0.20
SEED             = 123


def run_training(log_fn=print, epochs_p1=15, epochs_p2=10, batch_size=64, force_train=False):
    t_global_start = time.time()
    _log_section(log_fn, "INITIALISATION")
    _setup_gpu(log_fn, batch_size)

    if not os.path.exists(DATASET_DIR):
        log_fn(f"❌ Dossier introuvable : {DATASET_DIR}")
        return False

    os.makedirs(MODEL_DIR,       exist_ok=True)
    os.makedirs(CACHE_DIR_TRAIN, exist_ok=True)
    os.makedirs(CACHE_DIR_VAL,   exist_ok=True)

    _log_section(log_fn, "CHARGEMENT DU DATASET")
    t0 = time.time()
    log_fn("📁 Lecture des répertoires depuis Google Drive...")

    common_kwargs = dict(
        directory        = DATASET_DIR,
        validation_split = VALIDATION_SPLIT,
        seed             = SEED,
        image_size       = IMG_SIZE,
        batch_size       = batch_size,
    )
    train_dataset = tf.keras.utils.image_dataset_from_directory(subset="training",   **common_kwargs)
    val_dataset   = tf.keras.utils.image_dataset_from_directory(subset="validation", **common_kwargs)

    class_names = train_dataset.class_names
    num_classes = len(class_names)
    log_fn(f"✅ Dataset scanné en {time.time() - t0:.1f}s")
    log_fn(f"🚗 {num_classes} classes détectées : {class_names}")
    log_fn(f"   Batch size : {batch_size}  |  Val split : {VALIDATION_SPLIT*100:.0f}%")

    if num_classes < 2:
        log_fn("❌ Il faut au moins 2 classes pour entraîner l'IA.")
        return False

    with open(os.path.join(MODEL_DIR, "classes.json"), "w", encoding="utf-8") as f:
        json.dump(class_names, f, ensure_ascii=False, indent=2)
    log_fn(f"💾 classes.json sauvegardé ({num_classes} entrées)")

    _log_section(log_fn, "CONSTRUCTION DU PIPELINE tf.data")
    log_fn("🔧 Application de l'augmentation et mise en cache sur SSD local...")
    log_fn(f"   → Cache train : {CACHE_DIR_TRAIN}")
    log_fn(f"   → Cache val   : {CACHE_DIR_VAL}")
    log_fn("   (Premier passage = lecture Drive + écriture cache. Peut prendre 5-15 min.)")

    AUTOTUNE = tf.data.AUTOTUNE

    augment = tf.keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomBrightness(factor=0.10),
        layers.RandomContrast(factor=0.10),
    ], name="augmentation")

    train_dataset = (
        train_dataset
        .cache(os.path.join(CACHE_DIR_TRAIN, "train"))
        .shuffle(buffer_size=max(2000, batch_size * 16), seed=SEED)
        .map(lambda x, y: (augment(x, training=True), y), num_parallel_calls=AUTOTUNE)
        .prefetch(AUTOTUNE)
    )
    val_dataset = (
        val_dataset
        .cache(os.path.join(CACHE_DIR_VAL, "val"))
        .prefetch(AUTOTUNE)
    )

    _log_section(log_fn, "WARM-UP DU CACHE")
    log_fn("🔥 Initialisation du cache (lecture Drive → SSD local)...")
    t_cache = time.time()
    _warmup_cache(train_dataset, val_dataset, log_fn)
    elapsed_cache = time.time() - t_cache
    log_fn(f"✅ Cache prêt en {elapsed_cache/60:.1f} min ({elapsed_cache:.0f}s)")

    _log_section(log_fn, "MODÈLE — ANTI-CRASH RÉSUMÉ")

    checkpoint_p1      = os.path.join(MODEL_DIR, "best_model_phase1.keras")
    checkpoint_p2      = os.path.join(MODEL_DIR, "best_model_phase2.keras")
    training_done_flag = os.path.join(MODEL_DIR, "training_complete.json")

    if force_train:
        log_fn("⚠️ FORÇAGE ACTIVÉ : Suppression des anciennes sauvegardes...")
        fichiers_a_supprimer = [training_done_flag, checkpoint_p1, checkpoint_p2]
        for f in fichiers_a_supprimer:
            if os.path.exists(f):
                os.remove(f)
                log_fn(f"   🗑️ Supprimé : {os.path.basename(f)}")

    skip_phase1     = False
    best_p1         = 0.0
    best_p1_display = "0.0%"

    if os.path.exists(training_done_flag):
        log_fn("🏁 Entraînement déjà complet (training_complete.json trouvé).")
        log_fn("   Cochez 'Forcer l'entraînement' sur le Dashboard pour relancer depuis zéro (pour ajouter des models par ex).")
        return True

    skip_phase1     = False
    best_p1         = 0.0
    best_p1_display = "0.0%"

    if os.path.exists(training_done_flag):
        log_fn("🏁 Entraînement déjà complet (training_complete.json trouvé).")
        log_fn("   Supprimez ce fichier pour relancer depuis zéro.")
        return True

    target_checkpoint = None
    if os.path.exists(checkpoint_p2):
        target_checkpoint = checkpoint_p2
        skip_phase1       = True
        # P2 existe mais pas forcément terminée — on la reprend depuis le checkpoint
        log_fn(f"📂 Checkpoint Phase 2 détecté : {os.path.basename(checkpoint_p2)}")
    elif os.path.exists(checkpoint_p1):
        target_checkpoint = checkpoint_p1
        skip_phase1       = True
        log_fn(f"📂 Checkpoint Phase 1 détecté : {os.path.basename(checkpoint_p1)}")
    else:
        log_fn("📂 Aucun checkpoint existant — entraînement from scratch.")

    model = None
    if target_checkpoint:
        try:
            log_fn(f"🔍 Vérification de compatibilité : {os.path.basename(target_checkpoint)}...")
            temp_model      = tf.keras.models.load_model(target_checkpoint, compile=False)
            old_num_classes = temp_model.layers[-1].output_shape[-1]

            if old_num_classes == num_classes:
                log_fn(f"✅ Compatible ! ({old_num_classes} classes). Reprise du modèle.")
                model = tf.keras.models.load_model(target_checkpoint)
            else:
                log_fn(f"⚠️  Incompatible : checkpoint={old_num_classes} classes, dataset={num_classes} classes.")
                log_fn("🔄 Re-création du modèle. Entraînement repart de zéro.")
                skip_phase1       = False
                target_checkpoint = None
        except Exception as e:
            log_fn(f"⚠️  Checkpoint illisible : {e}")
            log_fn("🔄 Re-création du modèle.")
            skip_phase1       = False
            target_checkpoint = None

    if model is None:
        log_fn("🤖 Construction d'un nouveau modèle EfficientNetB0...")
        base_model = EfficientNetB0(
            input_shape = (*IMG_SIZE, 3),
            include_top = False,
            weights     = "imagenet",
        )
        base_model.trainable = False

        model = models.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dense(256, activation="relu"),
            layers.Dropout(0.40),
            layers.Dense(num_classes, activation="softmax", dtype="float32"),
        ], name="REVID_EfficientNetB0")

        log_fn(f"   Paramètres totaux  : {model.count_params():,}")
        log_fn(f"   Couches backbone   : {len(base_model.layers)}")

    # ── PHASE 1 : Tête seule ──
    if not skip_phase1:
        _log_section(log_fn, f"PHASE 1 — TÊTE SEULE ({epochs_p1} epochs max)")
        log_fn(f"   Backbone : gelé  |  LR : 1e-3  |  Batch : {batch_size}")

        model.compile(
            optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3),
            loss      = "sparse_categorical_crossentropy",
            metrics   = ["accuracy"],
        )

        t_p1 = time.time()
        callbacks_p1 = _get_callbacks(MODEL_DIR, phase=1, log_fn=log_fn)
        history1 = model.fit(
            train_dataset,
            validation_data = val_dataset,
            epochs          = epochs_p1,
            callbacks       = callbacks_p1,
            verbose         = 1,
        )

        best_p1         = max(history1.history.get("val_accuracy", [0.0]))
        best_p1_display = f"{best_p1*100:.1f}%"
        elapsed_p1      = time.time() - t_p1
        log_fn(f"\n✅ Phase 1 terminée en {elapsed_p1/60:.1f} min")
        log_fn(f"   Meilleure val_accuracy : {best_p1:.4f} ({best_p1_display})")
        log_fn(f"   Epochs effectués       : {len(history1.history['loss'])}/{epochs_p1}")
    else:
        log_fn("\n⏩ PHASE 1 IGNORÉE — checkpoint chargé directement en mémoire")
        best_p1_display = "N/A (reprise checkpoint)"

    # ── PHASE 2 : Fine-tuning ──
    _log_section(log_fn, f"PHASE 2 — FINE-TUNING ({epochs_p2} epochs max)")
    log_fn("🔓 Dégel des 30 dernières couches du backbone EfficientNetB0...")

    base_model_ref = model.layers[0]
    base_model_ref.trainable = True

    frozen_count   = 0
    unfrozen_count = 0
    for layer in base_model_ref.layers[:-30]:
        layer.trainable = False
        frozen_count += 1
    for layer in base_model_ref.layers[-30:]:
        unfrozen_count += 1

    log_fn(f"   Couches gelées    : {frozen_count}")
    log_fn(f"   Couches dégelées  : {unfrozen_count}")
    log_fn(f"   LR : 1e-5  (×100 plus faible qu'en P1 pour préserver ImageNet)")

    model.compile(
        optimizer = tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss      = "sparse_categorical_crossentropy",
        metrics   = ["accuracy"],
    )
    log_fn("   Recompilation terminée — démarrage de Phase 2...")

    t_p2 = time.time()
    callbacks_p2 = _get_callbacks(MODEL_DIR, phase=2, log_fn=log_fn)
    history2 = model.fit(
        train_dataset,
        validation_data = val_dataset,
        epochs          = epochs_p2,
        callbacks       = callbacks_p2,
        verbose         = 1,
    )

    best_p2    = max(history2.history.get("val_accuracy", [0.0]))
    elapsed_p2 = time.time() - t_p2
    log_fn(f"\n✅ Phase 2 terminée en {elapsed_p2/60:.1f} min")
    log_fn(f"   Meilleure val_accuracy : {best_p2:.4f} ({best_p2*100:.1f}%)")
    log_fn(f"   Epochs effectués       : {len(history2.history['loss'])}/{epochs_p2}")

    # ── Sauvegarde finale ──
    _log_section(log_fn, "SAUVEGARDE FINALE")

    model_path = os.path.join(MODEL_DIR, "revid_model.keras")
    model.save(model_path)
    log_fn(f"💾 Modèle final sauvegardé → {model_path}")

    best_overall = max(best_p1, best_p2)

    done_info = {
        "status"            : "complete",
        "best_val_accuracy" : round(float(best_overall), 4),
        "best_p1"           : round(float(best_p1), 4),
        "best_p2"           : round(float(best_p2), 4),
        "class_names"       : class_names,
        "total_time_min"    : round((time.time() - t_global_start) / 60, 1),
    }
    with open(training_done_flag, "w", encoding="utf-8") as f:
        json.dump(done_info, f, ensure_ascii=False, indent=2)

    elapsed_total = time.time() - t_global_start
    _log_section(log_fn, "RÉSUMÉ FINAL")
    log_fn(f"🏆 Meilleure val_accuracy globale : {best_overall*100:.1f}%")
    log_fn(f"   Phase 1 : {best_p1_display}  |  Phase 2 : {best_p2*100:.1f}%")
    log_fn(f"⏱️  Temps total : {elapsed_total/60:.1f} min")
    log_fn("✅ Entraînement terminé avec succès !\n")

    return True


def _setup_gpu(log_fn, batch_size):
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        log_fn(f"🎮 GPU détecté : {len(gpus)} device(s)")
        for g in gpus:
            log_fn(f"   → {g.name}")
        tf.keras.mixed_precision.set_global_policy("mixed_float16")
        log_fn("⚡ Mixed Precision activée (float16) → speedup ~1.5x, VRAM divisée par 2")
        log_fn(f"   Batch size : {batch_size}")
    else:
        log_fn("⚠️  Aucun GPU détecté — entraînement sur CPU (très lent !)")
        log_fn("   → Runtime > Modifier le type d'exécution > GPU")


def _warmup_cache(train_ds, val_ds, log_fn):
    log_fn("   [Train] Début de l'écriture du cache...")
    t = time.time()
    batch_count = 0
    for i, _ in enumerate(train_ds):
        batch_count = i + 1
        if batch_count % 10 == 0:
            log_fn(f"   [Train] {batch_count} batches mis en cache... ({time.time()-t:.0f}s)")
    log_fn(f"   [Train] ✅ Terminé ({batch_count} batches en {time.time()-t:.1f}s)")

    log_fn("   [Val]   Début de l'écriture du cache...")
    t = time.time()
    batch_count = 0
    for i, _ in enumerate(val_ds):
        batch_count = i + 1
        if batch_count % 10 == 0:
            log_fn(f"   [Val]   {batch_count} batches mis en cache... ({time.time()-t:.0f}s)")
    log_fn(f"   [Val]   ✅ Terminé ({batch_count} batches en {time.time()-t:.1f}s)")


def _get_callbacks(model_dir: str, phase: int, log_fn=print):
    checkpoint_path = os.path.join(model_dir, f"best_model_phase{phase}.keras")
    log_fn(f"   Checkpoint → {checkpoint_path}")
    return [
        tf.keras.callbacks.ModelCheckpoint(
            filepath       = checkpoint_path,
            monitor        = "val_accuracy",
            save_best_only = True,
            verbose        = 1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor              = "val_accuracy",
            patience             = 5,
            restore_best_weights = True,
            verbose              = 1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor  = "val_loss",
            factor   = 0.50,
            patience = 3,
            min_lr   = 1e-7,
            verbose  = 1,
        ),
    ]


def _log_section(log_fn, title: str):
    line = "═" * 52
    log_fn(f"\n{line}")
    log_fn(f"  {title}")
    log_fn(f"{line}")


if __name__ == "__main__":
    run_training()