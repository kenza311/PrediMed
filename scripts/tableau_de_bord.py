# Etapes 21 a 23 - Tableau de bord PrediMed
# Rassemble tous les resultats du projet dans UNE page web (fichier HTML)
# qu'on ouvre dans le navigateur. Les graphiques sont interactifs
# (on passe la souris dessus pour voir les valeurs).
#
# A lancer APRES : indicateurs.py, classification_abc.py, predire_risques.py

import os
import webbrowser
from pathlib import Path
import pandas as pd
import plotly.graph_objects as go

# --- Dossiers ---
DOSSIER = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DONNEES = os.path.join(DOSSIER, "donnees")
RESULTATS = os.path.join(DOSSIER, "resultats")

BLEU, ROUGE, ORANGE, VERT, GRIS = "#2E86C1", "#C0392B", "#E67E22", "#27AE60", "#7F8C8D"

# =====================================================================
# ETAPE 21 - Les chiffres cles (KPI)
# =====================================================================
equipements = pd.read_csv(os.path.join(DONNEES, "equipements.csv"), sep=";")
pannes = pd.read_csv(os.path.join(DONNEES, "historique_pannes.csv"), sep=";")
pannes["date_panne"] = pd.to_datetime(pannes["date_panne"])
pannes = pannes.merge(equipements[["id_equipement", "type", "service"]], on="id_equipement")

indic = pd.read_csv(os.path.join(RESULTATS, "indicateurs_equipements.csv"), sep=";")
abc = pd.read_csv(os.path.join(RESULTATS, "classification_abc.csv"), sep=";")
risques = pd.read_csv(os.path.join(RESULTATS, "risques_janvier_2026.csv"), sep=";")

# MTBF et MTTR du parc : totaux / nombre total de pannes (jamais une moyenne de moyennes)
nb_pannes = len(pannes)
mtbf_jours = indic["fonctionnement_h"].sum() / nb_pannes / 24
mttr_h = indic["arret_total_h"].sum() / nb_pannes
dispo = 100 * indic["fonctionnement_h"].sum() / (indic["fonctionnement_h"].sum() + indic["arret_total_h"].sum())
cout_total = pannes["cout_mad"].sum()
nb_alertes = int(risques["alerte"].sum())

kpis = [
    ("Equipements suivis", f"{len(equipements)}", "11 types, 5 services"),
    ("Pannes (2023-2025)", f"{nb_pannes}", "sur 3 ans"),
    ("MTBF du parc", f"{mtbf_jours:.0f} j", "temps moyen entre 2 pannes"),
    ("MTTR du parc", f"{mttr_h:.1f} h", "duree moyenne de reparation"),
    ("Disponibilite", f"{dispo:.2f} %", "temps de fonctionnement"),
    ("Cout des pannes", f"{cout_total / 1000:.0f} k MAD", "sur 3 ans"),
    ("Alertes janvier 2026", f"{nb_alertes}", "appareils a verifier"),
]

# =====================================================================
# ETAPE 22 - Les graphiques
# =====================================================================
graphes = {}


def style(fig, titre, hauteur=380):
    fig.update_layout(title=titre, height=hauteur, template="plotly_white", autosize=True,
                      margin=dict(l=40, r=20, t=50, b=40), font=dict(size=12))
    return fig


# 1) Pannes par mois
par_mois = pannes.set_index("date_panne").resample("MS").size()
fig = go.Figure(go.Bar(x=par_mois.index, y=par_mois.values, marker_color=BLEU))
graphes["mois"] = style(fig, "Nombre de pannes par mois")

# 2) Pareto du cout par type
cout_type = pannes.groupby("type")["cout_mad"].sum().sort_values(ascending=False)
cumul = 100 * cout_type.cumsum() / cout_type.sum()
fig = go.Figure()
fig.add_bar(x=cout_type.index, y=cout_type.values, name="Cout (MAD)", marker_color=BLEU)
fig.add_scatter(x=cout_type.index, y=cumul.values, name="Cumul (%)", yaxis="y2",
                mode="lines+markers", line=dict(color=ORANGE))
fig.update_layout(yaxis2=dict(overlaying="y", side="right", range=[0, 105], dtick=20, title="Cumul (%)"),
                  shapes=[dict(type="line", xref="paper", x0=0, x1=1, yref="y2", y0=80, y1=80,
                               line=dict(color=ROUGE, dash="dash"))],
                  legend=dict(orientation="h", y=-0.35))
graphes["pareto"] = style(fig, "Pareto : cout des pannes par type d'appareil", 420)

# 3) MTBF par type (du plus fragile au plus fiable)
par_type = indic.groupby("type").agg(fonct=("fonctionnement_h", "sum"),
                                     nb=("nb_pannes", "sum"))
par_type["MTBF_jours"] = par_type["fonct"] / par_type["nb"] / 24
# Plotly dessine la 1re barre en BAS : on trie du plus fiable au plus fragile
par_type = par_type.sort_values("MTBF_jours", ascending=False)
fig = go.Figure(go.Bar(x=par_type["MTBF_jours"], y=par_type.index, orientation="h",
                       marker_color=BLEU))
fig.update_xaxes(title="MTBF (jours)")
graphes["mtbf"] = style(fig, "MTBF par type (en haut = le plus fragile)", 420)

# 4) Causes de panne
causes = pannes["type_panne"].value_counts()
fig = go.Figure(go.Pie(labels=causes.index, values=causes.values, hole=0.45))
graphes["causes"] = style(fig, "Causes de panne")

# 5) Classification ABC
resume_abc = abc.groupby("classe").agg(nb=("id_equipement", "count"), cout=("cout_total", "sum"))
resume_abc["pct_cout"] = 100 * resume_abc["cout"] / resume_abc["cout"].sum()
fig = go.Figure(go.Bar(
    x=[f"Classe {c}" for c in resume_abc.index], y=resume_abc["nb"],
    marker_color=[ROUGE, ORANGE, VERT],
    text=[f"{n} appareils<br>{p:.0f} % du cout" for n, p in zip(resume_abc["nb"], resume_abc["pct_cout"])],
    textposition="auto"))
fig.update_yaxes(title="Nombre d'appareils")
graphes["abc"] = style(fig, "Classification ABC des equipements")

# 6) Alertes de janvier par service
alertes = risques[risques["alerte"]]
par_service = alertes["service"].value_counts()
fig = go.Figure(go.Bar(x=par_service.index, y=par_service.values, marker_color=ROUGE))
fig.update_yaxes(title="Appareils a verifier")
graphes["service"] = style(fig, "Alertes de janvier 2026 par service")

# =====================================================================
# ETAPE 23 - Le tableau des appareils a verifier
# =====================================================================
lignes_html = ""
for _, r in alertes.sort_values("risque_pct", ascending=False).iterrows():
    couleur = ROUGE if r["risque_pct"] >= 25 else ORANGE
    lignes_html += f"""
    <tr>
      <td><b>{r['id_equipement']}</b></td><td>{r['type']}</td><td>{r['service']}</td>
      <td><span class="risque" style="background:{couleur}">{r['risque_pct']:.0f} %</span></td>
      <td>{r['raisons']}</td>
    </tr>"""

# =====================================================================
# Assembler la page HTML
# =====================================================================
cartes = "".join(f"""
  <div class="carte"><div class="titre">{t}</div><div class="valeur">{v}</div>
  <div class="detail">{d}</div></div>""" for t, v, d in kpis)

# Le premier graphique embarque la bibliotheque Plotly (la page marche sans internet)
html_graphes = {}
premier = True
for nom, fig in graphes.items():
    html_graphes[nom] = fig.to_html(full_html=False, include_plotlyjs=premier,
                                    config={"displayModeBar": False, "responsive": True},
                                    default_width="100%")
    premier = False

page = f"""<!DOCTYPE html>
<html lang="fr">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>PrediMed - Tableau de bord</title>
<style>
  body {{ font-family: Segoe UI, Arial, sans-serif; background: #F4F6F8; margin: 0; color: #2C3E50; }}
  header {{ background: #1B4F72; color: white; padding: 20px 32px; }}
  header h1 {{ margin: 0; font-size: 26px; }}
  header p {{ margin: 6px 0 0; opacity: .85; }}
  main {{ padding: 24px 32px; max-width: 1400px; margin: auto; }}
  h2 {{ font-size: 19px; margin: 28px 0 12px; color: #1B4F72; }}
  .cartes {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); gap: 14px; }}
  .carte {{ background: white; border-radius: 10px; padding: 16px; box-shadow: 0 1px 4px rgba(0,0,0,.08); }}
  .carte .titre {{ font-size: 13px; color: {GRIS}; }}
  .carte .valeur {{ font-size: 28px; font-weight: bold; margin: 6px 0; color: #1B4F72; }}
  .carte .detail {{ font-size: 12px; color: {GRIS}; }}
  .grille {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(min(480px, 100%), 1fr)); gap: 16px; }}
  .bloc {{ min-width: 0; overflow: hidden; background: white; border-radius: 10px; padding: 8px; box-shadow: 0 1px 4px rgba(0,0,0,.08); }}
  table {{ width: 100%; border-collapse: collapse; background: white; border-radius: 10px; overflow: hidden;
          box-shadow: 0 1px 4px rgba(0,0,0,.08); font-size: 14px; }}
  th {{ background: #1B4F72; color: white; text-align: left; padding: 10px; }}
  td {{ padding: 9px 10px; border-bottom: 1px solid #ECF0F1; }}
  .risque {{ color: white; padding: 3px 9px; border-radius: 12px; font-weight: bold; }}
  footer {{ text-align: center; font-size: 12px; color: {GRIS}; padding: 24px; }}
</style>
</head>
<body>
<header>
  <h1>PrediMed - Maintenance predictive des equipements biomedicaux</h1>
  <p>Parc simule de {len(equipements)} equipements - historique 2023-2025 - prevision janvier 2026</p>
</header>
<main>
  <h2>Chiffres cles</h2>
  <div class="cartes">{cartes}</div>

  <h2>Appareils a verifier en priorite - janvier 2026 (risque de panne a 30 jours &ge; 15 %)</h2>
  <table>
    <tr><th>Appareil</th><th>Type</th><th>Service</th><th>Risque</th><th>Pourquoi</th></tr>
    {lignes_html}
  </table>

  <h2>Fiabilite du parc</h2>
  <div class="grille">
    <div class="bloc">{html_graphes['mois']}</div>
    <div class="bloc">{html_graphes['mtbf']}</div>
  </div>

  <h2>Couts et priorites</h2>
  <div class="grille">
    <div class="bloc">{html_graphes['pareto']}</div>
    <div class="bloc">{html_graphes['abc']}</div>
  </div>

  <h2>Causes et alertes</h2>
  <div class="grille">
    <div class="bloc">{html_graphes['causes']}</div>
    <div class="bloc">{html_graphes['service']}</div>
  </div>
</main>
<script>
  // Ajuste chaque graphique a la largeur de son cadre une fois la page chargee
  window.addEventListener("load", function () {{
    document.querySelectorAll(".plotly-graph-div").forEach(function (g) {{ Plotly.Plots.resize(g); }});
  }});
</script>
<footer>Projet PrediMed - Kenza Akouz - donnees simulees, modele de regression logistique
(note de tri AUC 0,75 sur 2025)</footer>
</body>
</html>"""

chemin = os.path.join(RESULTATS, "tableau_de_bord.html")
with open(chemin, "w", encoding="utf-8") as f:
    f.write(page)

print("Tableau de bord cree :", chemin)
webbrowser.open(Path(chemin).as_uri())   # l'ouvre dans le navigateur