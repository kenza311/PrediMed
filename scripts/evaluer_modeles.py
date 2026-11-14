# Etape 18 - Noter les modeles sur l'examen (2025)
#   - note de tri (AUC) : le modele range-t-il les appareils a risque en haut ?
#   - pour plusieurs seuils : pannes vues, ratees, fausses alertes, rappel, precision
#   - matrice de confusion au seuil choisi
#   - graphique pour choisir le seuil

import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, confusion_matrix

# --- Dossiers ---
DOSSIER = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DONNEES = os.path.join(DOSSIER, "donnees")
MODELES = os.path.join(DOSSIER, "modeles")
RESULTATS = os.path.join(DOSSIER, "resultats")
os.makedirs(RESULTATS, exist_ok=True)

SEUIL_CHOISI = 0.15   # au-dessus de 15 % de risque -> alerte

# --- 1. Reprendre les donnees de l'examen (2025) ---
dataset = pd.read_csv(os.path.join(DONNEES, "dataset_ia.csv"), sep=";",
                      parse_dates=["date"])
colonnes = joblib.load(os.path.join(MODELES, "colonnes.joblib"))
indices = ["age_ans", "usage_h_jour", "criticite", "sous_contrat",
           "pannes_12m", "pannes_3m", "jours_depuis_panne",
           "jours_depuis_mp", "retard_mp_jours", "alarmes_7j", "alarmes_30j"]
X = pd.get_dummies(dataset[indices + ["type"]], columns=["type"], dtype=int)
X = X[colonnes]   # memes colonnes, dans le meme ordre qu'a l'entrainement

examen = dataset["date"] >= "2025-01-01"
X_examen = X[examen]
y_examen = dataset.loc[examen, "panne_30j"]
print(f"Examen 2025 : {len(y_examen)} photos, {y_examen.sum()} vraies pannes")
print()

# --- 2. Charger les deux modeles ---
modeles = {
    "Modele simple": joblib.load(os.path.join(MODELES, "modele_simple.joblib")),
    "Foret aleatoire": joblib.load(os.path.join(MODELES, "modele_foret.joblib")),
}
risques = {nom: m.predict_proba(X_examen)[:, 1] for nom, m in modeles.items()}


# --- 3. Fonction : compter pour un seuil donne ---
def compter(risque, seuil):
    alerte = risque >= seuil
    tn, fp, fn, tp = confusion_matrix(y_examen, alerte, labels=[0, 1]).ravel()
    rappel = tp / (tp + fn) if (tp + fn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    return dict(seuil_pct=int(seuil * 100), alertes=tp + fp, pannes_vues=tp,
                pannes_ratees=fn, fausses_alertes=fp,
                rappel_pct=round(100 * rappel), precision_pct=round(100 * precision))


# --- 4. Note de tri (AUC) et tableau des seuils ---
for nom, risque in risques.items():
    auc = roc_auc_score(y_examen, risque)
    print(f"=== {nom} ===")
    print(f"Note de tri (AUC) : {auc:.2f}   (0.50 = hasard, 1.00 = parfait)")
    tableau = pd.DataFrame([compter(risque, s) for s in [0.10, 0.15, 0.20, 0.30, 0.50]])
    print(tableau.to_string(index=False))
    print()

# --- 5. Matrice de confusion au seuil choisi ---
print(f"=== Matrices de confusion au seuil de {int(SEUIL_CHOISI * 100)} % ===")
fig, axes = plt.subplots(1, 3, figsize=(17, 5))

for ax, (nom, risque) in zip(axes[:2], risques.items()):
    c = compter(risque, SEUIL_CHOISI)
    matrice = [[c["pannes_vues"], c["fausses_alertes"]],
               [c["pannes_ratees"], len(y_examen) - c["alertes"] - c["pannes_ratees"]]]
    print(f"{nom} : {c['pannes_vues']} pannes vues, {c['pannes_ratees']} ratees, "
          f"{c['fausses_alertes']} fausses alertes "
          f"-> rappel {c['rappel_pct']} %, precision {c['precision_pct']} %")

    ax.imshow([[1, 0.4], [0.4, 1]], cmap="Blues", vmin=0, vmax=1.5)
    textes = [["Pannes vues", "Fausses alertes"], ["Pannes ratees", "Rien a signaler"]]
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{textes[i][j]}\n{matrice[i][j]}", ha="center",
                    va="center", fontsize=12)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Panne reelle", "Pas de panne"])
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Alerte", "Pas d'alerte"])
    ax.set_title(f"{nom} (seuil {int(SEUIL_CHOISI * 100)} %)")

# --- 6. Graphique : rappel et precision selon le seuil ---
seuils = [s / 100 for s in range(5, 51)]
ax = axes[2]
for nom, risque, style in [("Modele simple", risques["Modele simple"], "-"),
                           ("Foret aleatoire", risques["Foret aleatoire"], "--")]:
    lignes = [compter(risque, s) for s in seuils]
    ax.plot([100 * s for s in seuils], [l["rappel_pct"] for l in lignes],
            color="#C0392B", linestyle=style, label=f"Rappel - {nom}")
    ax.plot([100 * s for s in seuils], [l["precision_pct"] for l in lignes],
            color="#2E86C1", linestyle=style, label=f"Precision - {nom}")
ax.axvline(100 * SEUIL_CHOISI, color="gray", linestyle=":")
ax.set_xlabel("Seuil d'alerte (%)")
ax.set_ylabel("%")
ax.set_title("Choisir le seuil : rappel vs precision")
ax.legend(fontsize=8)

plt.tight_layout()
chemin = os.path.join(RESULTATS, "evaluation_modeles.png")
plt.savefig(chemin, dpi=150)
print()
print("Graphique enregistre :", chemin)
plt.show()