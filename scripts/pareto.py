# Etape 13 - Diagramme de Pareto
# Question : quels types d'appareils coutent le plus cher en pannes ?
#            quelles causes de panne reviennent le plus ?

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
pannes = pannes.merge(equipements[["id_equipement", "type"]], on="id_equipement")


# --- 2. Fonction qui prepare un tableau de Pareto ---
def tableau_pareto(serie):
    t = serie.sort_values(ascending=False).to_frame("valeur")
    t["pourcentage"] = 100 * t["valeur"] / t["valeur"].sum()
    t["cumul_pct"] = t["pourcentage"].cumsum()
    return t


# Pareto 1 : cout total des pannes par type d'appareil
pareto_cout = tableau_pareto(pannes.groupby("type")["cout_mad"].sum())

# Pareto 2 : nombre de pannes par cause
pareto_cause = tableau_pareto(pannes["type_panne"].value_counts())


# --- 3. Afficher les resultats dans le terminal ---
print("=== PARETO 1 : cout des pannes par type d'appareil (MAD) ===")
print(pareto_cout.round(1).to_string())
print()
print("=== PARETO 2 : nombre de pannes par cause ===")
print(pareto_cause.round(1).to_string())
print()

# Types qui font ensemble 80 % du cout
nb_80 = (pareto_cout["cumul_pct"] < 80).sum() + 1
top = list(pareto_cout.index[:nb_80])
print(f"--> {nb_80} types sur {len(pareto_cout)} font 80 % du cout total :")
for t in top:
    print("   -", t)


# --- 4. Dessiner les deux diagrammes ---
def dessiner(ax, t, titre, label_y):
    x = range(len(t))
    ax.bar(x, t["valeur"], color="#2E86C1")
    ax.set_xticks(list(x))
    ax.set_xticklabels(t.index, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel(label_y)
    ax.set_title(titre)

    ax2 = ax.twinx()
    ax2.plot(list(x), t["cumul_pct"], color="#E67E22", marker="o")
    ax2.axhline(80, color="red", linestyle="--", linewidth=1)
    ax2.set_ylim(0, 105)
    ax2.set_ylabel("Cumul (%)")


fig, (a1, a2) = plt.subplots(1, 2, figsize=(15, 6))
dessiner(a1, pareto_cout, "Cout des pannes par type d'appareil", "Cout total (MAD)")
dessiner(a2, pareto_cause, "Nombre de pannes par cause", "Nombre de pannes")
plt.tight_layout()

chemin = os.path.join(RESULTATS, "pareto.png")
plt.savefig(chemin, dpi=150)
print()
print("Graphique enregistre :", chemin)
plt.show()