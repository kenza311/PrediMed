# Etape 15 - Fabriquer le tableau d'exemples pour l'IA
# Une "photo" de chaque appareil au 1er de chaque mois.
# Pour chaque photo :
#   - les indices = ce qu'on savait CE JOUR-LA (jamais le futur)
#   - la reponse  = est-ce que l'appareil tombe en panne dans les 30 jours suivants ?

import os
import pandas as pd

# --- Dossiers ---
DOSSIER = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DONNEES = os.path.join(DOSSIER, "donnees")

# --- 1. Lire les donnees ---
equipements = pd.read_csv(os.path.join(DONNEES, "equipements.csv"), sep=";")
pannes = pd.read_csv(os.path.join(DONNEES, "historique_pannes.csv"), sep=";")
pannes["date_panne"] = pd.to_datetime(pannes["date_panne"])
mps = pd.read_csv(os.path.join(DONNEES, "maintenances_preventives.csv"), sep=";")
mps["date_mp"] = pd.to_datetime(mps["date_mp"])
alarmes = pd.read_csv(os.path.join(DONNEES, "alarmes.csv"), sep=";")
alarmes["date_alarme"] = pd.to_datetime(alarmes["date_alarme"])

DEBUT = pd.Timestamp("2023-01-01")   # debut de l'historique
HORIZON = 30                          # on predit les 30 jours suivants

# Les dates des photos : le 1er de chaque mois, de janvier 2023 a decembre 2025
dates_photos = pd.date_range("2023-01-01", "2025-12-01", freq="MS")

# --- 2. Prendre les photos ---
lignes = []
for date in dates_photos:
    fin = date + pd.Timedelta(days=HORIZON)

    for _, eq in equipements.iterrows():
        p = pannes[pannes["id_equipement"] == eq["id_equipement"]]
        m = mps[mps["id_equipement"] == eq["id_equipement"]]
        a = alarmes[alarmes["id_equipement"] == eq["id_equipement"]]

        # Ce qu'on connait a la date de la photo : seulement les pannes AVANT
        passe = p[p["date_panne"] < date]
        pannes_12m = (passe["date_panne"] >= date - pd.Timedelta(days=365)).sum()
        pannes_3m = (passe["date_panne"] >= date - pd.Timedelta(days=90)).sum()

        if len(passe) > 0:
            jours_depuis = (date - passe["date_panne"].max()).days
        else:
            # aucune panne connue : on compte depuis le debut de l'historique
            jours_depuis = (date - DEBUT).days

        # Maintenance preventive : combien de jours depuis la derniere ?
        # retard = jours depuis la MP - intervalle prevu (positif = MP en retard)
        jours_depuis_mp = (date - m.loc[m["date_mp"] < date, "date_mp"].max()).days
        retard_mp = jours_depuis_mp - eq["intervalle_mp_mois"] * 30

        # Alarmes affichees pendant les 7 et 30 derniers jours (AVANT la photo)
        a_passe = a[a["date_alarme"] < date]
        alarmes_7j = (a_passe["date_alarme"] >= date - pd.Timedelta(days=7)).sum()
        alarmes_30j = (a_passe["date_alarme"] >= date - pd.Timedelta(days=30)).sum()

        # La reponse : y a-t-il une panne entre la photo et 30 jours apres ?
        futur = p[(p["date_panne"] >= date) & (p["date_panne"] < fin)]
        panne_30j = 1 if len(futur) > 0 else 0

        lignes.append({
            "id_equipement": eq["id_equipement"],
            "date": date,
            "type": eq["type"],
            "service": eq["service"],
            "age_ans": date.year - eq["annee_installation"],
            "usage_h_jour": eq["heures_utilisation_jour"],
            "criticite": eq["criticite"],
            "sous_contrat": int(eq["sous_contrat"]),
            "pannes_12m": pannes_12m,
            "pannes_3m": pannes_3m,
            "jours_depuis_panne": jours_depuis,
            "jours_depuis_mp": jours_depuis_mp,
            "retard_mp_jours": retard_mp,
            "alarmes_7j": alarmes_7j,
            "alarmes_30j": alarmes_30j,
            "panne_30j": panne_30j,
        })

dataset = pd.DataFrame(lignes)

# --- 3. Verifications ---
print("Nombre de photos (lignes) :", len(dataset))
print("Nombre d'appareils        :", dataset["id_equipement"].nunique())
print("Nombre de mois            :", dataset["date"].nunique())
print()

nb_oui = dataset["panne_30j"].sum()
print(f"Photos suivies d'une panne (reponse = 1) : {nb_oui} "
      f"soit {100 * nb_oui / len(dataset):.1f} %")
print(f"Photos sans panne         (reponse = 0) : {len(dataset) - nb_oui}")
print()


# Petit controle de bon sens : les appareils avec beaucoup de pannes passees
# tombent-ils plus souvent en panne ensuite ?
print("=== Risque de panne selon les pannes des 12 derniers mois ===")
controle = dataset.groupby("pannes_12m")["panne_30j"].agg(["count", "mean"])
controle.columns = ["nb_photos", "taux_panne_30j"]
controle["taux_panne_30j"] = (100 * controle["taux_panne_30j"]).round(1)
print(controle.to_string())

print()
print("=== Risque de panne selon les alarmes des 30 derniers jours ===")
groupes = pd.cut(dataset["alarmes_30j"], [-1, 0, 1, 2, 4, 100],
                 labels=["0", "1", "2", "3-4", "5 et +"])
controle = dataset.groupby(groupes, observed=True)["panne_30j"].agg(["count", "mean"])
controle.columns = ["nb_photos", "taux_panne_30j"]
controle["taux_panne_30j"] = (100 * controle["taux_panne_30j"]).round(1)
print(controle.to_string())

print()
print("=== Risque de panne selon le retard de maintenance preventive ===")
groupes = pd.cut(dataset["retard_mp_jours"], [-1000, 0, 60, 120, 1000],
                 labels=["a l'heure", "retard 0-2 mois", "retard 2-4 mois", "retard > 4 mois"])
controle = dataset.groupby(groupes, observed=True)["panne_30j"].agg(["count", "mean"])
controle.columns = ["nb_photos", "taux_panne_30j"]
controle["taux_panne_30j"] = (100 * controle["taux_panne_30j"]).round(1)
print(controle.to_string())

# --- 4. Enregistrer ---
chemin = os.path.join(DONNEES, "dataset_ia.csv")
dataset.to_csv(chemin, sep=";", index=False, encoding="utf-8-sig")
print()
print("Tableau enregistre :", chemin)