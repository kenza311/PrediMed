# Etape 14 - Classification ABC des equipements
# On classe chaque appareil selon le cout de ses pannes :
#   A = les appareils qui font les 80 premiers % du cout
#   B = ceux qui font les 15 % suivants (de 80 a 95 %)
#   C = ceux qui font les 5 derniers %

import os
import pandas as pd
import matplotlib.pyplot as plt

# --- Dossiers ---
DOSSIER = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DONNEES = os.path.join(DOSSIER, "donnees")
RESULTATS = os.path.join(DOSSIER, "resultats")
os.makedirs(RESULTATS, exist_ok=True)

# --- 1. Lire les donnees ---
equipements = pd.read_csv(os.path.join(DONNEES, "equipements.csv"), sep=";")
pannes = pd.read_csv(os.path.join(DONNEES, "historique_pannes.csv"), sep=";")

# --- 2. Cout total et nombre de pannes par appareil ---
bilan = pannes.groupby("id_equipement").agg(
    nb_pannes=("id_panne", "count"),
    cout_total=("cout_mad", "sum"),
)

# On ajoute aussi les appareils qui n'ont eu aucune panne (cout = 0)
bilan = equipements.set_index("id_equipement")[["type", "service"]].join(bilan)
bilan = bilan.fillna({"nb_pannes": 0, "cout_total": 0})
bilan["nb_pannes"] = bilan["nb_pannes"].astype(int)

# --- 3. Trier du plus cher au moins cher et calculer le cumul ---
bilan = bilan.sort_values("cout_total", ascending=False)
bilan["pourcentage"] = 100 * bilan["cout_total"] / bilan["cout_total"].sum()
bilan["cumul_pct"] = bilan["pourcentage"].cumsum()


# --- 4. Donner la classe A, B ou C ---
# On regarde le cumul AVANT l'appareil : s'il commence sous 80 %, il est en A
def classe(cumul_avant):
    if cumul_avant < 80:
        return "A"
    elif cumul_avant < 95:
        return "B"
    else:
        return "C"


cumul_avant = bilan["cumul_pct"] - bilan["pourcentage"]
bilan["classe"] = cumul_avant.apply(classe)

# --- 5. Resume par classe ---
resume = bilan.groupby("classe").agg(
    nb_appareils=("type", "count"),
    cout_total=("cout_total", "sum"),
)
resume["pct_appareils"] = 100 * resume["nb_appareils"] / resume["nb_appareils"].sum()
resume["pct_cout"] = 100 * resume["cout_total"] / resume["cout_total"].sum()

print("=== RESUME PAR CLASSE ===")
print(resume.round(1).to_string())
print()

# Combien d'appareils de chaque type dans chaque classe
print("=== TYPES D'APPAREILS PAR CLASSE ===")
print(pd.crosstab(bilan["type"], bilan["classe"]).to_string())
print()

print("=== LES 10 APPAREILS LES PLUS COUTEUX ===")
print(bilan.head(10)[["type", "service", "nb_pannes", "cout_total", "classe"]].to_string())

# --- 6. Enregistrer le tableau ---
chemin_csv = os.path.join(RESULTATS, "classification_abc.csv")
bilan.round(2).to_csv(chemin_csv, sep=";", encoding="utf-8-sig")
print()
print("Tableau enregistre :", chemin_csv)

# --- 7. Graphique : courbe ABC ---
couleurs = {"A": "#C0392B", "B": "#F39C12", "C": "#27AE60"}
x = range(1, len(bilan) + 1)

fig, ax = plt.subplots(figsize=(12, 6))
ax.bar(x, bilan["cout_total"], color=bilan["classe"].map(couleurs))
ax.set_xlabel("Appareils (du plus couteux au moins couteux)")
ax.set_ylabel("Cout total des pannes (MAD)")
ax.set_title("Classification ABC des equipements")

ax2 = ax.twinx()
ax2.plot(list(x), bilan["cumul_pct"], color="black")
ax2.axhline(80, color="gray", linestyle="--", linewidth=1)
ax2.axhline(95, color="gray", linestyle="--", linewidth=1)
ax2.set_ylim(0, 105)
ax2.set_ylabel("Cumul du cout (%)")

# Legende
for c in ["A", "B", "C"]:
    n = resume.loc[c, "nb_appareils"]
    p = resume.loc[c, "pct_cout"]
    ax.bar(0, 0, color=couleurs[c], label=f"Classe {c} : {n} appareils, {p:.0f} % du cout")
ax.legend(loc="center right")

plt.tight_layout()
chemin_png = os.path.join(RESULTATS, "classification_abc.png")
plt.savefig(chemin_png, dpi=150)
print("Graphique enregistre :", chemin_png)
plt.show()