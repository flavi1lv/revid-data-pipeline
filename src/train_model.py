import os
import json
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras import layers, models

# ─── CHEMINS ──────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "../data/06_spectrograms")
MODEL_DIR   = os.path.join(BASE_DIR, "../models")

# ─── HYPERPARAMÈTRES ──────────────────────────────────────────────────────────
IMG_SIZE         = (224, 224)
BATCH_SIZE       = 64   # Boost de RAM (nb de spectrogrammes traités en parallèle)
EPOCHS_PHASE1    = 15   # Backbone gelé — on entraîne seulement la tête
EPOCHS_PHASE2    = 10   # Fine-tuning — on dégèle les 30 dernières couches
VALIDATION_SPLIT = 0.20
SEED             = 123

def run_training(log_fn=print):
    log_fn("\n==================================================")
    log_fn("🧠 DÉMARRAGE DE L'ENTRAÎNEMENT (EfficientNetB0)")
    log_fn("==================================================\n")

    # 1. Vérifications préliminaires
    if not os.path.exists(DATASET_DIR):
        log_fn(f"❌ Dossier introuvable : {DATASET_DIR}")
        return False

    os.makedirs(MODEL_DIR, exist_ok=True)

    # 2. Chargement du dataset
    log_fn("📁 Chargement des données...")
    common_kwargs = dict(
        directory        = DATASET_DIR,
        validation_split = VALIDATION_SPLIT,
        seed             = SEED,
        image_size       = IMG_SIZE,
        batch_size       = BATCH_SIZE,
    )
    train_dataset = tf.keras.utils.image_dataset_from_directory(subset="training",  **common_kwargs)
    val_dataset   = tf.keras.utils.image_dataset_from_directory(subset="validation", **common_kwargs)

    class_names = train_dataset.class_names
    num_classes = len(class_names)
    log_fn(f"\n🚗 {num_classes} classes détectées : {class_names}\n")

    if num_classes < 2:
        log_fn("❌ Il faut au moins 2 classes pour entraîner l'IA.")
        return False

    # Sauvegarde du mapping classes → index (utile pour l'inférence Flask)
    with open(os.path.join(MODEL_DIR, "classes.json"), "w", encoding="utf-8") as f:
        json.dump(class_names, f, ensure_ascii=False, indent=2)

    # 3. Augmentation + optimisation tf.data
    AUTOTUNE = tf.data.AUTOTUNE

    augment = tf.keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomBrightness(factor=0.10),
        layers.RandomContrast(factor=0.10),
    ], name="augmentation")

    train_dataset = (
        train_dataset
        .map(lambda x, y: (augment(x, training=True), y), num_parallel_calls=AUTOTUNE)
        .cache()
        .shuffle(2000, seed=SEED)
        .prefetch(AUTOTUNE)
    )
    val_dataset = val_dataset.cache().prefetch(AUTOTUNE)

# 4. Construction ou Chargement du modèle (Logique Anti-Crash 2.0)
    checkpoint_p1 = os.path.join(MODEL_DIR, "best_model_phase1.keras")
    checkpoint_p2 = os.path.join(MODEL_DIR, "best_model_phase2.keras")
    
    skip_phase1 = False
    skip_phase2 = False
    best_p1 = 0.0 # Valeur par défaut si on saute la phase

    # On vérifie l'état d'avancement du pipeline
    target_checkpoint = None
    if os.path.exists(checkpoint_p2):
        target_checkpoint = checkpoint_p2
        skip_phase1 = True # Si P2 existe, P1 est forcément finie
    elif os.path.exists(checkpoint_p1):
        target_checkpoint = checkpoint_p1
        skip_phase1 = True

    # Chargement intelligent
    if target_checkpoint:
        try:
            temp_model = tf.keras.models.load_model(target_checkpoint, compile=False)
            old_num_classes = temp_model.layers[-1].output_shape[-1]
            
            if old_num_classes == num_classes:
                log_fn(f"✅ Sauvegarde compatible trouvée ! Reprise depuis {os.path.basename(target_checkpoint)}")
                model = tf.keras.models.load_model(target_checkpoint)
            else:
                log_fn(f"⚠️ Le nombre de voitures a changé ({old_num_classes} -> {num_classes}).")
                log_fn("🔄 Re-génération de l'architecture. L'entraînement repart de zéro.")
                target_checkpoint = None
                skip_phase1 = False
        except Exception as e:
            log_fn("⚠️ Sauvegarde illisible ou corrompue.")
            target_checkpoint = None
            skip_phase1 = False

    # Construction à neuf si aucune sauvegarde valide
    if not target_checkpoint:
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
            layers.Dense(num_classes, activation="softmax"),
        ], name="REVID_EfficientNetB0")

    # ─── PHASE 1 : Tête seule ─────────────────────────────────────────────────
    if not skip_phase1:
        log_fn(f"\n📌 PHASE 1 — Entraînement de la tête ({EPOCHS_PHASE1} epochs max)")
        log_fn("   Backbone : gelé  |  LR : 1e-3")

        model.compile(
            optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3),
            loss      = "sparse_categorical_crossentropy",
            metrics   = ["accuracy"],
        )

        callbacks_p1 = _get_callbacks(MODEL_DIR, phase=1)
        history1 = model.fit(
            train_dataset,
            validation_data = val_dataset,
            epochs          = EPOCHS_PHASE1,
            callbacks       = callbacks_p1,
        )
        best_p1 = max(history1.history.get("val_accuracy", [0]))
        log_fn(f"\n   ✅ Phase 1 — meilleure val_accuracy : {best_p1:.4f} ({best_p1*100:.1f}%)")
    else:
        log_fn("\n⏩ PHASE 1 IGNORÉE — (Déjà complétée, poids chargés en mémoire)")

    # ─── PHASE 1 : Tête seule ─────────────────────────────────────────────────
    log_fn(f"\n📌 PHASE 1 — Entraînement de la tête ({EPOCHS_PHASE1} epochs max)")
    log_fn("   Backbone : gelé  |  LR : 1e-3")

    model.compile(
        optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss      = "sparse_categorical_crossentropy",
        metrics   = ["accuracy"],
    )

    callbacks_p1 = _get_callbacks(MODEL_DIR, phase=1)
    history1 = model.fit(
        train_dataset,
        validation_data = val_dataset,
        epochs          = EPOCHS_PHASE1,
        callbacks       = callbacks_p1,
    )
    
    # On gère le cas où l'EarlyStopping a tout coupé avant la fin
    best_p1 = max(history1.history.get("val_accuracy", [0]))
    log_fn(f"\n   ✅ Phase 1 — meilleure val_accuracy : {best_p1:.4f} ({best_p1*100:.1f}%)")

    # ─── PHASE 2 : Fine-tuning ────────────────────────────────────────────────
    log_fn(f"\n🔓 PHASE 2 — Fine-tuning ({EPOCHS_PHASE2} epochs max)")
    log_fn("   Backbone : 30 dernières couches dégelées  |  LR : 1e-5")

    # Récupération du backbone (indispensable si on a chargé une sauvegarde)
    base_model_ref = model.layers[0] 
    base_model_ref.trainable = True
    for layer in base_model_ref.layers[:-30]:
        layer.trainable = False

    # LR très faible pour ne pas écraser les poids ImageNet
    model.compile(
        optimizer = tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss      = "sparse_categorical_crossentropy",
        metrics   = ["accuracy"],
    )

    callbacks_p2 = _get_callbacks(MODEL_DIR, phase=2)
    history2 = model.fit(
        train_dataset,
        validation_data = val_dataset,
        epochs          = EPOCHS_PHASE2,
        callbacks       = callbacks_p2,
    )
    
    best_p2 = max(history2.history.get("val_accuracy", [0]))
    log_fn(f"\n   ✅ Phase 2 — meilleure val_accuracy : {best_p2:.4f} ({best_p2*100:.1f}%)")

    # ─── SAUVEGARDE FINALE ────────────────────────────────────────────────────
    model_path = os.path.join(MODEL_DIR, "revid_model.keras")
    model.save(model_path)

    best_overall = max(best_p1, best_p2)
    log_fn(f"\n🏆 Meilleure val_accuracy globale : {best_overall*100:.1f}%")
    log_fn(f"💾 Modèle sauvegardé → {model_path}")
    log_fn("✅ Entraînement terminé !\n")

    return True


def _get_callbacks(model_dir: str, phase: int):
    """Callbacks communs aux deux phases."""
    checkpoint_path = os.path.join(model_dir, f"best_model_phase{phase}.keras")
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

if __name__ == "__main__":
    run_training()