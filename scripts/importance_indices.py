# Etape 19 - Quels indices comptent le plus pour le modele ?
# Methode : on "brouille" un indice (on melange ses valeurs entre les photos)
# et on regarde de combien la note de tri (AUC) baisse sur l'examen 2025.
#   -> si la note chute beaucoup : le modele avait besoin de cet indice
#   -> si la note ne bouge pas  : l'indice ne sert a rien au modele
# On regarde aussi le SENS : l'indice fait-il monter ou baisser le risque ?

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

# --- Dossiers ---
DOSSIER = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DONNEES = os.path.join(DOSSIER, "donnees")
MODELES = os.path.join(DOSSIER, "modeles")
RESULTATS = os.path.join(DOSSIER, "resultats")

# --- 1. Reprendre l'examen 2025 et le modele simple (le meilleur) ---
dataset = pd.read_csv(os.path.join(DONNEES, "dataset_ia.csv"), sep=";",
                      parse_dates=["date"])
colonnes = joblib.load(os.path.join(MODELES, "colonnes.joblib"))
modele = joblib.load(os.path.join(MODELES, "modele_simple.joblib"))

indices = ["age_ans", "usage_h_jour", "criticite", "sous_contrat",
           "pannes_12m", "pannes_3m", "jours_depuis_panne",
           "jours_depuis_mp", "retard_mp_jours", "alarmes_7j", "alarmes_30j"]
X = pd.get_dummies(dataset[indices + ["type"]], columns=["type"], dtype=int)[colonnes]

examen = dataset["date"] >= "2025-01-01"
X_examen = X[examen].reset_index(drop=True)
y_examen = dataset.loc[examen, "panne_30j"].reset_index(drop=True)

note_normale = roc_auc_score(y_examen, modele.predict_proba(X_examen)[:, 1])
print(f"Note de tri normale (AUC) : {note_normale:.3f}")
print()

# --- 2. Brouiller chaque indice, 30 fois, et mesurer la baisse ---
# Le "type d'appareil" est reparti sur 11 colonnes : on les brouille ensemble
groupes = {nom: [nom] for nom in indices}
groupes["type_appareil"] = [c for c in colonnes if c.startswith("type_")]

rng = np.random.default_rng(42)
resultats = []
for nom, cols in groupes.items():
    baisses = []
    for _ in range(30):
        X_brouille = X_examen.copy()
        ordre = rng.permutation(len(X_brouille))
        X_brouille[cols] = X_examen[cols].values[ordre]
        note = roc_auc_score(y_examen, modele.predict_proba(X_brouille)[:, 1])
        baisses.append(note_normale - note)
    resultats.append({"indice": nom, "baisse_note": np.mean(baisses)})

importance = pd.DataFrame(resultats).sort_values("baisse_note", ascending=False)

# --- 3. Le sens de chaque indice (coefficient du modele simple) ---
# coefficient > 0 : plus l'indice est grand, plus le risque monte
coef = pd.Series(modele[-1].coef_[0], index=colonnes)
importance["sens"] = importance["indice"].map(
    lambda n: "" if n == "type_appareil" else
    ("fait MONTER le risque" if coef[n] > 0 else "fait BAISSER le risque"))

# Part de chaque indice dans l'importance totale (on ignore les baisses negatives)
positif = importance["baisse_note"].clip(lower=0)
importance["part_pct"] = (100 * positif / positif.sum()).round(1)
importance["baisse_note"] = importance["baisse_note"].round(4)

print("=== Importance des indices (du plus utile au moins utile) ===")
print(importance.to_string(index=False))

# --- 4. Graphique ---
noms_clairs = {
    "alarmes_30j": "Alarmes (30 derniers jours)",
    "alarmes_7j": "Alarmes (7 derniers jours)",
    "jours_depuis_mp": "Jours depuis la derniere MP",
    "retard_mp_jours": "Retard de maintenance preventive",
    "pannes_12m": "Pannes (12 derniers mois)",
    "pannes_3m": "Pannes (3 derniers mois)",
    "jours_depuis_panne": "Jours depuis la derniere panne",
    "type_appareil": "Type d'appareil",
    "age_ans": "Age de l'appareil",
    "usage_h_jour": "Heures d'utilisation par jour",
    "criticite": "Criticite",
    "sous_contrat": "Sous contrat",
}
graphe = importance.sort_values("part_pct")
fig, ax = plt.subplots(figsize=(10, 6))
ax.barh([noms_clairs[n] for n in graphe["indice"]], graphe["part_pct"], color="#2E86C1")
ax.set_xlabel("Part de l'importance (%)")
ax.set_title("Sur quoi le modele se base-t-il pour predire une panne ?")
for i, v in enumerate(graphe["part_pct"]):
    ax.text(v + 0.5, i, f"{v:.0f} %", va="center")
plt.tight_layout()

chemin = os.path.join(RESULTATS, "importance_indices.png")
plt.savefig(chemin, dpi=150)
print()
print("Graphique enregistre :", chemin)
plt.show()