import numpy as np
import pandas as pd
from pathlib import Path

# =====================================================================
# PrediMed - Etapes 5 a 7 : creer des donnees de pannes SIMULEES (v3)
# Parc fictif d'equipements biomedicaux suivi sur 3 ans (2023-2025).
# Chaque type d'appareil a ses propres regles (hypotheses de travail,
# basees sur la pratique biomedicale) :
#   nb       : nombre d'appareils dans l'hopital (min, max)
#   usage    : heures d'utilisation par jour (min, max)
#   taux     : pannes par an pour un appareil neuf, utilise normalement
#   mttr     : duree d'arret "typique" (mediane) en heures
#   contrat  : probabilite d'etre sous contrat de maintenance
#   causes   : repartition des causes de panne
#   cout_acc : prix d'un accessoire remplace (MAD)
#   cout_pce : prix d'une piece interne remplacee (MAD)
#
# Nouveau en v3 (pour que l'IA ait des signaux a apprendre) :
#   - maintenance preventive (MP) : chaque appareil a une MP periodique.
#     Plus la MP est en retard, plus l'usure augmente et plus le risque de panne monte.
#   - alarmes : l'appareil affiche des alarmes/erreurs de temps en temps,
#     et beaucoup plus souvent dans les semaines qui precedent une panne.
# =====================================================================

rng = np.random.default_rng(42)  # 42 = toujours les memes donnees

DEBUT = pd.Timestamp("2023-01-01")
FIN = pd.Timestamp("2025-12-31")
NB_JOURS = (FIN - DEBUT).days

CAUSES = ["Electrique", "Mecanique", "Logicielle", "Accessoire / consommable", "Erreur d'utilisation"]

TYPES = {
    # Pannes surtout sur la pedale, le cable et l'electrode neutre -> reparation rapide
    "Bistouri electrique": dict(
        service="Bloc operatoire", critique=3, marques=["Medtronic (Valleylab)", "Erbe"],
        nb=(4, 6), usage=(4, 10), taux=1.0, mttr=8, contrat=0.5,
        causes=[0.25, 0.05, 0.10, 0.45, 0.15], cout_acc=(1500, 6000), cout_pce=(8000, 30000)),
    # Verins hydrauliques, telecommande, batterie
    "Table d'operation": dict(
        service="Bloc operatoire", critique=2, marques=["Maquet", "Trumpf"],
        nb=(5, 7), usage=(6, 10), taux=0.6, mttr=24, contrat=0.6,
        causes=[0.25, 0.45, 0.05, 0.15, 0.10], cout_acc=(1500, 5000), cout_pce=(5000, 25000)),
    # Eclairage LED fiable ; pannes d'alimentation et de bras articule
    "Scialytique": dict(
        service="Bloc operatoire", critique=2, marques=["Maquet", "Steris"],
        nb=(5, 7), usage=(6, 10), taux=0.4, mttr=12, contrat=0.5,
        causes=[0.45, 0.35, 0.05, 0.10, 0.05], cout_acc=(800, 3000), cout_pce=(4000, 20000)),
    # Fuites, valves, cellule O2, circuit patient
    "Respirateur d'anesthesie": dict(
        service="Bloc operatoire", critique=3, marques=["Draeger", "GE"],
        nb=(6, 8), usage=(6, 10), taux=1.0, mttr=24, contrat=0.8,
        causes=[0.20, 0.30, 0.10, 0.25, 0.15], cout_acc=(1500, 6000), cout_pce=(8000, 40000)),
    # Fonctionne 24h/24 ; capteur de debit, valve expiratoire, turbine
    "Ventilateur de reanimation": dict(
        service="Reanimation", critique=3, marques=["Hamilton", "Draeger"],
        nb=(6, 9), usage=(16, 24), taux=1.2, mttr=36, contrat=0.8,
        causes=[0.20, 0.25, 0.10, 0.30, 0.15], cout_acc=(2000, 8000), cout_pce=(10000, 60000)),
    # Surtout les accessoires (cables ECG, capteurs SpO2, brassards) -> echange rapide
    "Moniteur multiparametrique": dict(
        service="Reanimation", critique=3, marques=["Philips", "Mindray"],
        nb=(10, 14), usage=(18, 24), taux=0.8, mttr=4, contrat=0.5,
        causes=[0.15, 0.05, 0.10, 0.55, 0.15], cout_acc=(500, 4000), cout_pce=(5000, 20000)),
    # Petit appareil : chutes, mecanisme d'entrainement, batterie. Remplace tout de suite.
    "Pousse-seringue": dict(
        service="Reanimation", critique=2, marques=["B. Braun", "Fresenius Kabi"],
        nb=(15, 20), usage=(8, 24), taux=0.7, mttr=3, contrat=0.3,
        causes=[0.25, 0.25, 0.10, 0.10, 0.30], cout_acc=(300, 1500), cout_pce=(1000, 5000)),
    # Appareil en attente : surtout batterie et electrodes/palettes
    "Defibrillateur": dict(
        service="Urgences", critique=3, marques=["Zoll", "Philips"],
        nb=(5, 7), usage=(0.5, 2), taux=0.4, mttr=8, contrat=0.6,
        causes=[0.40, 0.05, 0.05, 0.40, 0.10], cout_acc=(1500, 5000), cout_pce=(5000, 15000)),
    # Cable patient, imprimante thermique
    "Electrocardiographe": dict(
        service="Urgences", critique=1, marques=["Schiller", "GE"],
        nb=(6, 8), usage=(2, 6), taux=0.6, mttr=4, contrat=0.3,
        causes=[0.15, 0.10, 0.05, 0.55, 0.15], cout_acc=(500, 3000), cout_pce=(2000, 8000)),
    # Les sondes sont la principale cause ; envoyees en reparation chez le fournisseur
    "Echographe": dict(
        service="Imagerie", critique=2, marques=["GE", "Siemens", "Philips"],
        nb=(4, 6), usage=(6, 12), taux=0.8, mttr=72, contrat=0.7,
        causes=[0.20, 0.05, 0.15, 0.50, 0.10], cout_acc=(10000, 60000), cout_pce=(10000, 60000)),
    # 2 a 3 seances par jour ; circuit hydraulique (pompes, fuites), capteurs de conductivite
    "Generateur d'hemodialyse": dict(
        service="Hemodialyse", critique=3, marques=["Fresenius", "Nipro", "B. Braun"],
        nb=(8, 12), usage=(10, 16), taux=2.0, mttr=24, contrat=0.7,
        causes=[0.25, 0.35, 0.10, 0.20, 0.10], cout_acc=(1500, 6000), cout_pce=(5000, 25000)),
}

# Effet de la cause sur la duree d'arret (x mediane du type)
EFFET_DUREE = {"Electrique": 1.3, "Mecanique": 1.3, "Logicielle": 0.8,
               "Accessoire / consommable": 0.7, "Erreur d'utilisation": 0.3}
# Probabilite qu'une piece soit remplacee selon la cause
PROBA_PIECE = {"Electrique": 0.6, "Mecanique": 0.6, "Logicielle": 0.05,
               "Accessoire / consommable": 0.8, "Erreur d'utilisation": 0.2}

# Intervalle de maintenance preventive par type (en mois), selon les recommandations
# habituelles des fabricants
INTERVALLE_MP = {
    "Bistouri electrique": 12, "Table d'operation": 12, "Scialytique": 12,
    "Respirateur d'anesthesie": 6, "Ventilateur de reanimation": 6,
    "Moniteur multiparametrique": 12, "Pousse-seringue": 12, "Defibrillateur": 6,
    "Electrocardiographe": 12, "Echographe": 12, "Generateur d'hemodialyse": 6,
}

# Les pannes "Erreur d'utilisation" arrivent sans prevenir (pas d'alarmes avant)
CAUSES_SANS_ALERTE = ["Erreur d'utilisation"]

# --- 1. Creer le parc d'equipements --------------------------------
equipements = []
numero = 1
for nom_type, t in TYPES.items():
    for _ in range(int(rng.integers(t["nb"][0], t["nb"][1] + 1))):
        equipements.append({
            "id_equipement": f"EQ{numero:03d}",
            "type": nom_type,
            "service": t["service"],
            "marque": str(rng.choice(t["marques"])),
            "annee_installation": int(rng.integers(2010, 2024)),
            "heures_utilisation_jour": round(float(rng.uniform(*t["usage"])), 1),
            "criticite": t["critique"],
            "sous_contrat": bool(rng.random() < t["contrat"]),
            "intervalle_mp_mois": INTERVALLE_MP[nom_type],
        })
        numero += 1

df_eq = pd.DataFrame(equipements)

# --- 2. Simuler, jour par jour, la MP, les pannes et les alarmes ----
mps, pannes, alarmes = [], [], []
numero_panne = 1

for _, eq in df_eq.iterrows():
    t = TYPES[eq["type"]]
    usage_normal = sum(t["usage"]) / 2
    intervalle = eq["intervalle_mp_mois"] * 30  # en jours

    # Derniere MP avant 2023 : a un moment au hasard dans l'intervalle precedent
    derniere_mp = -int(rng.integers(0, intervalle))
    # On l'enregistre aussi : la GMAO de l'hopital connait la derniere MP
    mps.append({"id_equipement": eq["id_equipement"],
                "date_mp": (DEBUT + pd.Timedelta(days=derniere_mp)).date()})

    def retard():
        # Sous contrat : le fournisseur passe presque a l'heure.
        # Sans contrat : la MP est souvent repoussee (manque de temps, de pieces...)
        if eq["sous_contrat"]:
            return int(rng.integers(0, 15))
        return int(min(rng.exponential(90), 300))

    prochaine_mp = derniere_mp + intervalle + retard()

    for jour in range(NB_JOURS + 1):
        date = DEBUT + pd.Timedelta(days=jour)

        # a) Maintenance preventive du jour ?
        if jour >= prochaine_mp:
            mps.append({"id_equipement": eq["id_equipement"], "date_mp": date.date()})
            derniere_mp = jour
            prochaine_mp = jour + intervalle + retard()

        # b) Usure : depend du temps depuis la derniere MP
        u = (jour - derniere_mp) / intervalle      # 0 = MP faite, 1 = MP due, >1 = en retard
        usure = 0.45 + 0.85 * u ** 2

        age = max(date.year - eq["annee_installation"], 0)
        taux = t["taux"] * (1 + 0.08 * age) * (eq["heures_utilisation_jour"] / usage_normal) ** 0.5
        proba_jour = taux * usure / 365

        # c) Alarmes "de fond" : environ 1 par mois, sans lien avec une panne
        if rng.random() < 1 / 30:
            alarmes.append({"id_equipement": eq["id_equipement"], "date_alarme": date.date()})

        # d) Panne du jour ?
        if rng.random() >= proba_jour:
            continue

        cause = str(rng.choice(CAUSES, p=t["causes"]))

        # Duree d'arret : mediane du type x effet de la cause, avec de la variabilite
        duree_h = t["mttr"] * EFFET_DUREE[cause] * rng.lognormal(0, 0.6)
        if eq["sous_contrat"]:
            duree_h *= 0.7  # intervention plus rapide, pieces disponibles
        duree_h = round(max(duree_h, 0.5), 1)

        # Cout : main-d'oeuvre + piece eventuelle
        piece = bool(rng.random() < PROBA_PIECE[cause])
        cout = rng.uniform(200, 600)
        if piece:
            plage = t["cout_acc"] if cause == "Accessoire / consommable" else t["cout_pce"]
            cout += rng.uniform(*plage)
        if eq["sous_contrat"]:
            cout *= 0.3  # le contrat couvre la plus grande partie

        pannes.append({
            "id_panne": f"P{numero_panne:04d}",
            "id_equipement": eq["id_equipement"],
            "date_panne": date.date(),
            "type_panne": cause,
            "duree_arret_h": duree_h,
            "piece_changee": piece,
            "cout_mad": round(float(cout), 0),
        })
        numero_panne += 1

        # e) Alarmes "precurseurs" : dans les semaines AVANT la panne,
        #    l'appareil affiche de plus en plus d'alarmes (sauf erreur d'utilisation)
        if cause not in CAUSES_SANS_ALERTE:
            duree_signes = int(rng.integers(15, 46))       # 15 a 45 jours avant
            for _ in range(int(rng.poisson(5))):
                # plus on se rapproche de la panne, plus il y a d'alarmes
                avant = int(duree_signes * rng.random() ** 2)
                j = jour - avant
                if j >= 0:
                    alarmes.append({"id_equipement": eq["id_equipement"],
                                    "date_alarme": (DEBUT + pd.Timedelta(days=j)).date()})

df_pannes = pd.DataFrame(pannes).sort_values("date_panne").reset_index(drop=True)
df_mp = pd.DataFrame(mps).sort_values("date_mp").reset_index(drop=True)
df_alarmes = pd.DataFrame(alarmes).sort_values("date_alarme").reset_index(drop=True)

# --- 3. Enregistrer ------------------------------------------------
dossier = Path(__file__).resolve().parent.parent / "donnees"
dossier.mkdir(exist_ok=True)
df_eq.to_csv(dossier / "equipements.csv", index=False, sep=";", encoding="utf-8-sig")
df_pannes.to_csv(dossier / "historique_pannes.csv", index=False, sep=";", encoding="utf-8-sig")
df_mp.to_csv(dossier / "maintenances_preventives.csv", index=False, sep=";", encoding="utf-8-sig")
df_alarmes.to_csv(dossier / "alarmes.csv", index=False, sep=";", encoding="utf-8-sig")

print("=== PrediMed : donnees simulees creees (v3) ===")
print(f"Equipements               : {len(df_eq)}")
print(f"Pannes                    : {len(df_pannes)}  (du {df_pannes['date_panne'].min()} au {df_pannes['date_panne'].max()})")
print(f"Maintenances preventives  : {len(df_mp)}")
print(f"Alarmes                   : {len(df_alarmes)}")
print()
print("Pannes par annee :")
print(pd.to_datetime(df_pannes["date_panne"]).dt.year.value_counts().sort_index().to_string())
print()
print("Apercu des pannes :")
print(df_pannes.head().to_string(index=False))
print(f"\nFichiers enregistres dans : {dossier}")