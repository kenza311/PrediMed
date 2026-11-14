import pandas as pd
from pathlib import Path

# PrediMed - Etapes 8 et 9 : verifier et decouvrir les donnees

dossier = Path(__file__).resolve().parent.parent / "donnees"
eq = pd.read_csv(dossier / "equipements.csv", sep=";")
pannes = pd.read_csv(dossier / "historique_pannes.csv", sep=";", parse_dates=["date_panne"])

# --- Etape 8 : controle qualite des donnees ---
print("=== 1. Controle des donnees ===")
print("Cases vides (equipements) :", int(eq.isna().sum().sum()))
print("Cases vides (pannes)      :", int(pannes.isna().sum().sum()))
print("Doublons de pannes        :", int(pannes.duplicated("id_panne").sum()))
inconnus = set(pannes["id_equipement"]) - set(eq["id_equipement"])
print("Pannes sur un equipement inconnu :", len(inconnus))

# On relie chaque panne a son equipement (comme une RECHERCHEV dans Excel)
data = pannes.merge(eq, on="id_equipement", how="left")

# --- Etape 9 : premier apercu ---
print("\n=== 2. Vue d'ensemble ===")
print(f"Periode           : {pannes['date_panne'].min().date()} -> {pannes['date_panne'].max().date()}")
print(f"Equipements       : {len(eq)}")
print(f"Pannes            : {len(pannes)}")
print(f"Arret total       : {pannes['duree_arret_h'].sum():,.0f} heures")
print(f"Cout total        : {pannes['cout_mad'].sum():,.0f} MAD")
sans_panne = len(set(eq["id_equipement"]) - set(pannes["id_equipement"]))
print(f"Jamais en panne   : {sans_panne} equipements")

print("\n=== 3. Pannes par service ===")
print(data.groupby("service").agg(pannes=("id_panne", "count"),
                                  cout_mad=("cout_mad", "sum")).sort_values("pannes", ascending=False).to_string())

print("\n=== 4. Pannes par type d'equipement ===")
print(data["type"].value_counts().to_string())

print("\n=== 5. Pannes par type de panne ===")
print(data["type_panne"].value_counts().to_string())

print("\n=== 6. Pannes par annee ===")
print(data["date_panne"].dt.year.value_counts().sort_index().to_string())