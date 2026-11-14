import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# PrediMed - Etapes 10 a 12 : MTBF, MTTR et disponibilite

racine = Path(__file__).resolve().parent.parent
eq = pd.read_csv(racine / "donnees" / "equipements.csv", sep=";")
pannes = pd.read_csv(racine / "donnees" / "historique_pannes.csv", sep=";", parse_dates=["date_panne"])

DEBUT = pd.Timestamp("2023-01-01")
FIN = pd.Timestamp("2025-12-31")
HEURES_PERIODE = ((FIN - DEBUT).days + 1) * 24   # 3 ans = 26 280 heures

# --- Calculs par equipement ---
par_eq = pannes.groupby("id_equipement").agg(
    nb_pannes=("id_panne", "count"),
    arret_total_h=("duree_arret_h", "sum"),
)
ind = eq.merge(par_eq, on="id_equipement", how="left").fillna({"nb_pannes": 0, "arret_total_h": 0})
ind["nb_pannes"] = ind["nb_pannes"].astype(int)

# Temps de fonctionnement = periode totale - temps passe en panne
ind["fonctionnement_h"] = HEURES_PERIODE - ind["arret_total_h"]

# MTBF = temps de fonctionnement / nombre de pannes
ind["MTBF_h"] = ind["fonctionnement_h"] / ind["nb_pannes"].where(ind["nb_pannes"] > 0)
ind["MTBF_jours"] = ind["MTBF_h"] / 24

# MTTR = temps d'arret total / nombre de pannes
ind["MTTR_h"] = ind["arret_total_h"] / ind["nb_pannes"].where(ind["nb_pannes"] > 0)

# Disponibilite = temps de fonctionnement / periode totale
ind["disponibilite_%"] = 100 * ind["fonctionnement_h"] / HEURES_PERIODE

ind = ind.round({"MTBF_h": 0, "MTBF_jours": 0, "MTTR_h": 1, "disponibilite_%": 2, "arret_total_h": 1})

# --- Resume par type d'equipement (methode correcte : totaux) ---
par_type = ind.groupby("type").agg(
    appareils=("id_equipement", "count"),
    pannes=("nb_pannes", "sum"),
    fonctionnement_h=("fonctionnement_h", "sum"),
    arret_h=("arret_total_h", "sum"),
)
par_type["MTBF_jours"] = par_type["fonctionnement_h"] / par_type["pannes"] / 24
par_type["MTTR_h"] = par_type["arret_h"] / par_type["pannes"]
par_type["disponibilite_pct"] = 100 * par_type["fonctionnement_h"] / (par_type["appareils"] * HEURES_PERIODE)
par_type = par_type[["appareils", "pannes", "MTBF_jours", "MTTR_h", "disponibilite_pct"]].round(1).sort_values("MTBF_jours")

# --- Affichage ---
print("=== PrediMed : indicateurs de maintenance (2023-2025) ===\n")
total_pannes = ind["nb_pannes"].sum()
print(f"MTBF du parc             : {ind['fonctionnement_h'].sum() / total_pannes / 24:.0f} jours")
print(f"MTTR du parc             : {ind['arret_total_h'].sum() / total_pannes:.1f} heures")
print(f"Disponibilite moyenne    : {ind['disponibilite_%'].mean():.2f} %\n")

print("--- Par type d'equipement (du plus fragile au plus fiable) ---")
print(par_type.to_string())

print("\n--- Les 5 equipements les moins disponibles ---")
cols = ["id_equipement", "type", "service", "nb_pannes", "MTBF_jours", "MTTR_h", "disponibilite_%"]
print(ind.sort_values("disponibilite_%")[cols].head(5).to_string(index=False))

# --- Enregistrement ---
(racine / "resultats").mkdir(exist_ok=True)
ind.to_csv(racine / "resultats" / "indicateurs_equipements.csv", index=False, sep=";", encoding="utf-8-sig")

# --- Graphique ---
fig, axes = plt.subplots(1, 2, figsize=(13, 6), sharey=True)
axes[0].barh(par_type.index, par_type["MTBF_jours"], color="#2a7ab0")
axes[0].set_title("MTBF moyen (jours)\nplus c'est grand, plus c'est fiable")
axes[1].barh(par_type.index, par_type["MTTR_h"], color="#d9822b")
axes[1].set_title("MTTR moyen (heures)\nplus c'est petit, plus c'est vite repare")
for ax in axes:
    ax.grid(axis="x", alpha=0.3)
fig.suptitle("PrediMed - Indicateurs de maintenance par type d'equipement")
plt.tight_layout()
plt.savefig(racine / "resultats" / "mtbf_mttr_par_type.png", dpi=150)
print("\nFichiers enregistres dans le dossier 'resultats'.")
plt.show()