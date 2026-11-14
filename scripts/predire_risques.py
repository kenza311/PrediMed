# Etape 20 - Utiliser le modele "pour de vrai"
# Situation au 1er janvier 2026 -> liste des appareils a verifier en priorite
# pour le mois de janvier, du plus risque au moins risque, avec la raison.

import os
import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# --- Dossiers ---
DOSSIER = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DONNEES = os.path.join(DOSSIER, "donnees")
MODELES = os.path.join(DOSSIER, "modeles")
RESULTATS = os.path.join(DOSSIER, "resultats")

DATE_PREVISION = pd.Timestamp("2026-01-01")
SEUIL = 0.15   # meme seuil qu'a l'etape 18

# --- 1. Lire les donnees ---
equipements = pd.read_csv(os.path.join(DONNEES, "equipements.csv"), sep=";")
pannes = pd.read_csv(os.path.join(DONNEES, "historique_pannes.csv"), sep=";")
pannes["date_panne"] = pd.to_datetime(pannes["date_panne"])
mps = pd.read_csv(os.path.join(DONNEES, "maintenances_preventives.csv"), sep=";")
mps["date_mp"] = pd.to_datetime(mps["date_mp"])
alarmes = pd.read_csv(os.path.join(DONNEES, "alarmes.csv"), sep=";")
alarmes["date_alarme"] = pd.to_datetime(alarmes["date_alarme"])

DEBUT = pd.Timestamp("2023-01-01")

# --- 2. Prendre la "photo" de chaque appareil au 1er janvier 2026 ---
# (exactement les memes calculs qu'a l'etape 15)
date = DATE_PREVISION
lignes = []
for _, eq in equipements.iterrows():
    p = pannes[(pannes["id_equipement"] == eq["id_equipement"]) & (pannes["date_panne"] < date)]
    m = mps[(mps["id_equipement"] == eq["id_equipement"]) & (mps["date_mp"] < date)]
    a = alarmes[(alarmes["id_equipement"] == eq["id_equipement"]) & (alarmes["date_alarme"] < date)]

    jours_depuis_mp = (date - m["date_mp"].max()).days
    lignes.append({
        "id_equipement": eq["id_equipement"],
        "type": eq["type"],
        "service": eq["service"],
        "age_ans": date.year - eq["annee_installation"],
        "usage_h_jour": eq["heures_utilisation_jour"],
        "criticite": eq["criticite"],
        "sous_contrat": int(eq["sous_contrat"]),
        "pannes_12m": (p["date_panne"] >= date - pd.Timedelta(days=365)).sum(),
        "pannes_3m": (p["date_panne"] >= date - pd.Timedelta(days=90)).sum(),
        "jours_depuis_panne": (date - p["date_panne"].max()).days if len(p) else (date - DEBUT).days,
        "jours_depuis_mp": jours_depuis_mp,
        "retard_mp_jours": jours_depuis_mp - eq["intervalle_mp_mois"] * 30,
        "alarmes_7j": (a["date_alarme"] >= date - pd.Timedelta(days=7)).sum(),
        "alarmes_30j": (a["date_alarme"] >= date - pd.Timedelta(days=30)).sum(),
    })
photo = pd.DataFrame(lignes)

# --- 3. Re-entrainer le modele simple sur TOUTES les donnees (2023-2025) ---
# A l'etape 18 on l'a teste sur 2025 et il a fait ses preuves.
# Pour l'utiliser en vrai, on lui apprend aussi 2025 : plus d'exemples = mieux.
dataset = pd.read_csv(os.path.join(DONNEES, "dataset_ia.csv"), sep=";")
indices = ["age_ans", "usage_h_jour", "criticite", "sous_contrat",
           "pannes_12m", "pannes_3m", "jours_depuis_panne",
           "jours_depuis_mp", "retard_mp_jours", "alarmes_7j", "alarmes_30j"]
colonnes = joblib.load(os.path.join(MODELES, "colonnes.joblib"))

X_tout = pd.get_dummies(dataset[indices + ["type"]], columns=["type"], dtype=int)[colonnes]
modele = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
modele.fit(X_tout, dataset["panne_30j"])
joblib.dump(modele, os.path.join(MODELES, "modele_final.joblib"))

# --- 4. Calculer le risque de chaque appareil pour janvier 2026 ---
X_photo = pd.get_dummies(photo[indices + ["type"]], columns=["type"], dtype=int)
X_photo = X_photo.reindex(columns=colonnes, fill_value=0)
photo["risque_pct"] = (100 * modele.predict_proba(X_photo)[:, 1]).round(1)


# --- 5. Expliquer chaque alerte avec des mots simples ---
def raisons(r):
    liste = []
    if r["alarmes_30j"] >= 3:
        liste.append(f"{r['alarmes_30j']} alarmes ce mois (dont {r['alarmes_7j']} cette semaine)")
    elif r["alarmes_7j"] >= 1:
        liste.append(f"{r['alarmes_7j']} alarme(s) cette semaine")
    if r["retard_mp_jours"] > 0:
        liste.append(f"MP en retard de {r['retard_mp_jours']} j")
    elif r["retard_mp_jours"] > -30:
        liste.append(f"MP due dans {-r['retard_mp_jours']} j")
    if r["type"] in ["Generateur d'hemodialyse", "Echographe"]:
        liste.append("type d'appareil fragile")
    if r["age_ans"] >= 10:
        liste.append(f"{r['age_ans']} ans")
    if r["pannes_12m"] >= 2:
        liste.append(f"{r['pannes_12m']} pannes en 12 mois")
    return ", ".join(liste) if liste else "-"


photo["raisons"] = photo.apply(raisons, axis=1)
photo["alerte"] = photo["risque_pct"] >= 100 * SEUIL
photo = photo.sort_values("risque_pct", ascending=False)

# --- 6. Afficher la liste a verifier ---
a_verifier = photo[photo["alerte"]]
print(f"=== APPAREILS A VERIFIER EN JANVIER 2026 (risque >= {int(SEUIL * 100)} %) ===")
print(f"{len(a_verifier)} appareils sur {len(photo)}")
print()
print(a_verifier[["id_equipement", "type", "service", "risque_pct", "raisons"]]
      .to_string(index=False))
print()
print("Repartition des alertes par service :")
print(a_verifier["service"].value_counts().to_string())

# --- 7. Enregistrer la liste complete (ouvrable dans Excel) ---
chemin = os.path.join(RESULTATS, "risques_janvier_2026.csv")
photo[["id_equipement", "type", "service", "risque_pct", "alerte", "raisons",
       "alarmes_7j", "alarmes_30j", "retard_mp_jours", "pannes_12m", "age_ans"]] \
    .to_csv(chemin, sep=";", index=False, encoding="utf-8-sig")
print()
print("Liste complete enregistree :", chemin)