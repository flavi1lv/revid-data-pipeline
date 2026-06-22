import os
import time
import json
import re
from groq import Groq
from dotenv import load_dotenv
from db import get_db

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

BATCH_SIZE = 10

# Tous les termes qui signifient "pas identifiable" — LLM peut répondre en anglais ou en français
TERMES_INVALIDES = {"INVALIDE", "INVALID", "N/A", "NA", "UNKNOWN", "NON IDENTIFIE", "INCONNU", "NONE", ""}

def nettoyer_nom(texte):
    """
    Nettoyage chirurgical — retire UNIQUEMENT les caractères structurels parasites.
    On retire seulement ce qu'un LLM peut produire accidentellement :
    accolades, crochets, guillemets, deux-points, etc.
    """
    nom = re.sub(r'[{}\[\]()"\',:;#@!?\\/*]', '', str(texte))
    nom = ' '.join(nom.split())
    return nom.strip().upper()


def extraire_valeur(resultat):
    """
    Bouclier anti-hallucination : normalise la réponse du LLM quelle que soit sa forme.
    """
    if isinstance(resultat, dict):
        valeur = " ".join(str(v) for v in resultat.values() if isinstance(v, str)).strip()
    elif isinstance(resultat, list):
        # [["PORSCHE 911 GT3"]] → "PORSCHE 911 GT3"
        valeur = str(resultat[0]).strip() if resultat else "INVALIDE"
    else:
        valeur = resultat

    return nettoyer_nom(valeur)


def build_batch_prompt(batch):
    lignes = []
    for i, video in enumerate(batch):
        desc = (video["description"] or "")[:200]
        lignes.append(f'{i+1}. Titre: "{video["title"]}" | Description: "{desc}"')

    return f"""Tu es un expert automobile. Extrais la Marque et le Modèle exact de chaque vidéo.

RÈGLES ABSOLUES :
- Format de réponse : "MARQUE MODELE" en majuscules (ex : PORSCHE 911 GT3, FERRARI 488 GTB)
- Ignore : années, marques d'échappement (Akrapovic, Milltek, Borla), tags (ASMR, POV, RAW, 1 hour)
- Si aucun véhicule clairement identifiable : écris exactement INVALIDE
- Réponds UNIQUEMENT avec un JSON array de strings. AUCUN dictionnaire. AUCUN objet.
- Exemple de réponse valide : ["PORSCHE 911 GT3", "INVALIDE", "FERRARI 488 GTB"]

Vidéos à analyser :
{chr(10).join(lignes)}

JSON array de {len(batch)} strings :"""


def clean_single(video):
    """
    Fallback 1-par-1 si un batch échoue totalement.
    Prompt minimaliste pour maximiser les chances d'une réponse propre.
    """
    try:
        r = client.chat.completions.create(
            messages=[{"role": "user", "content":
                f'Marque et modèle exact en majuscules UNIQUEMENT, ou "INVALIDE". '
                f'Aucun autre mot, aucune ponctuation. Titre: "{video["title"]}"'}],
            model="llama-3.1-8b-instant",
            temperature=0.1,
        )
        nom = nettoyer_nom(r.choices[0].message.content)
        return nom if nom and nom not in TERMES_INVALIDES else "INVALIDE"
    except Exception:
        return "INVALIDE"


def clean_titles_with_ai(vehicule_name):
    print("🤖 Analyse IA démarrée...\n")

    conn = get_db()
    c = conn.cursor()

    c.execute(
        "SELECT id, title, description, url FROM videos WHERE vehicule = ? AND status = 'scraped'",
        (vehicule_name,)
    )
    videos = [dict(row) for row in c.fetchall()]

    if not videos:
        print("   ✅ Aucune nouvelle vidéo à nettoyer.")
        conn.close()
        return

    total_batches = (len(videos) + BATCH_SIZE - 1) // BATCH_SIZE
    total_valides  = 0
    total_invalides = 0

    for batch_num in range(total_batches):
        batch = videos[batch_num * BATCH_SIZE : (batch_num + 1) * BATCH_SIZE]
        print(f"⏳ Batch {batch_num + 1}/{total_batches} ({len(batch)} vidéos)...")

        try:
            response = client.chat.completions.create(
                messages=[{"role": "user", "content": build_batch_prompt(batch)}],
                model="llama-3.1-8b-instant",
                temperature=0.1,
            )
            raw = response.choices[0].message.content.strip()

            # Extraction robuste : on isole le array JSON même si le modèle ajoute du texte autour
            start = raw.find('[')
            end   = raw.rfind(']') + 1
            if start == -1 or end == 0:
                raise ValueError("Aucun array JSON détecté dans la réponse.")

            resultats = json.loads(raw[start:end])

            batch_valides = 0
            for i, resultat in enumerate(resultats):
                if i >= len(batch): break

                nom    = extraire_valeur(resultat)
                vid_id = batch[i]['id']

                if nom and nom not in TERMES_INVALIDES:
                    c.execute(
                        "UPDATE videos SET clean_name = ?, status = 'cleaned' WHERE id = ?",
                        (nom, vid_id)
                    )
                    total_valides += 1
                    batch_valides += 1
                else:
                    c.execute("UPDATE videos SET status = 'invalid' WHERE id = ?", (vid_id,))
                    total_invalides += 1

            conn.commit()
            print(f"   ✅ {batch_valides}/{len(batch)} valides.")

        except (json.JSONDecodeError, ValueError) as e:
            # Le batch a retourné du JSON illisible → fallback 1 par 1
            print(f"   ⚠️  JSON illisible (batch {batch_num + 1}) : {e}")
            print(f"   🔄  Fallback individuel sur {len(batch)} vidéos...")

            for video in batch:
                nom = clean_single(video)
                if nom != "INVALIDE":
                    c.execute(
                        "UPDATE videos SET clean_name = ?, status = 'cleaned' WHERE id = ?",
                        (nom, video['id'])
                    )
                    total_valides += 1
                else:
                    c.execute("UPDATE videos SET status = 'invalid' WHERE id = ?", (video['id'],))
                    total_invalides += 1
                time.sleep(0.3)

            conn.commit()

        except Exception as e:
            print(f"   ❌ Erreur inattendue (batch {batch_num + 1}) : {e}")

        # Pause entre les batches pour respecter le rate limit Groq
        if batch_num + 1 < total_batches:
            time.sleep(1)

    conn.close()
    taux = total_valides / max(total_valides + total_invalides, 1) * 100
    print(f"\n✅ Analyse IA terminée.")
    print(f"   Valides   : {total_valides}")
    print(f"   Invalides : {total_invalides}")
    print(f"   Taux      : {taux:.1f}%")