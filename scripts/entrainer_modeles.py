# Etapes 16 et 17 - Separer cours / examen, puis entrainer deux modeles
#   Cours  (entrainement) : photos de 2023 et 2024
#   Examen (test)         : photos de 2025, jamais vues par le modele

import os
import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# --- Dossiers ---
DOSSIER = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DONNEES = os.path.join(DOSSIER, "donnees")
MODELES = os.path.join(DOSSIER, "modeles")
os.makedirs(MODELES, exist_ok=True)

# --- 1. Lire le tableau d'exemples (etape 15) ---
dataset = pd.read_csv(os.path.join(DONNEES, "dataset_ia.csv"), sep=";",
                      parse_dates=["date"])

# --- 2. Choisir les indices (X) et la reponse (y) ---
indices = ["age_ans", "usage_h_jour", "criticite", "sous_contrat",
           "pannes_12m", "pannes_3m", "jours_depuis_panne",
           "jours_depuis_mp", "retard_mp_jours", "alarmes_7j", "alarmes_30j"]

# Le type d'appareil est du texte : on le transforme en colonnes 0/1
# (une colonne par type : "type_Echographe" = 1 si c'est un echographe)
X = pd.get_dummies(dataset[indices + ["type"]], columns=["type"], dtype=int)
y = dataset["panne_30j"]

# --- 3. ETAPE 16 : separer le cours et l'examen selon la DATE ---
cours = dataset["date"] < "2025-01-01"
examen = ~cours

X_cours, y_cours = X[cours], y[cours]
X_examen, y_examen = X[examen], y[examen]

print("=== ETAPE 16 : separation ===")
print(f"Cours  (2023-2024) : {len(X_cours)} photos, dont {y_cours.sum()} pannes")
print(f"Examen (2025)      : {len(X_examen)} photos, dont {y_examen.sum()} pannes")
print(f"Nombre d'indices utilises : {X.shape[1]}")
print()

# --- 4. ETAPE 17 : entrainer les deux modeles ---
# Modele 1 : regression logistique (simple)
# StandardScaler met tous les indices a la meme echelle (ans, heures, jours...)
modele_simple = make_pipeline(
    StandardScaler(),
    LogisticRegression(max_iter=1000),
)
modele_simple.fit(X_cours, y_cours)

# Modele 2 : foret aleatoire (300 petits arbres qui votent)
modele_foret = RandomForestClassifier(
    n_estimators=300,
    min_samples_leaf=30,      # chaque feuille doit contenir au moins 30 photos
    random_state=42,
)
modele_foret.fit(X_cours, y_cours)

print("=== ETAPE 17 : entrainement termine ===")
print("Modele 1 : regression logistique")
print("Modele 2 : foret aleatoire (300 arbres)")
print()

# --- 5. Premier coup d'oeil : le modele donne un RISQUE entre 0 et 100 % ---
apercu = dataset.loc[examen, ["id_equipement", "date", "type", "panne_30j"]].copy()
apercu["risque_simple_%"] = (100 * modele_simple.predict_proba(X_examen)[:, 1]).round(1)
apercu["risque_foret_%"] = (100 * modele_foret.predict_proba(X_examen)[:, 1]).round(1)

print("=== Apercu : 8 photos de l'examen (janvier 2025) ===")
print(apercu.head(8).to_string(index=False))
print()

# Risque moyen donne quand il y a eu une panne / quand il n'y en a pas eu
print("Risque moyen donne par chaque modele :")
for col in ["risque_simple_%", "risque_foret_%"]:
    moy = apercu.groupby("panne_30j")[col].mean().round(1)
    print(f"  {col:16s} -> SANS panne ensuite : {moy[0]} %   AVEC panne ensuite : {moy[1]} %")

# --- 6. Enregistrer les modeles pour l'etape 18 ---
joblib.dump(modele_simple, os.path.join(MODELES, "modele_simple.joblib"))
joblib.dump(modele_foret, os.path.join(MODELES, "modele_foret.joblib"))
joblib.dump(list(X.columns), os.path.join(MODELES, "colonnes.joblib"))
print()
print("Modeles enregistres dans :", MODELES)