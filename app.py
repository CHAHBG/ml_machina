import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
import warnings, os, io
warnings.filterwarnings('ignore')

st.set_page_config(page_title="EDA - Maintenance Predictive", layout="wide", initial_sidebar_state="expanded")

C_OK = '#2196F3'
C_PANNE = '#F44336'
PALETTE = ['#2196F3','#F44336','#4CAF50','#FF9800','#9C27B0','#00BCD4','#795548','#607D8B','#E91E63','#3F51B5']

@st.cache_data
def load_data():
    p = 'dataset_maintenance_predictive.csv'
    if not os.path.exists(p):
        st.error(f"Fichier {p} introuvable."); st.stop()
    return pd.read_csv(p)

df = load_data()

# ── Sidebar navigation ──
st.sidebar.title("Navigation")
st.sidebar.markdown("Utilisez les boutons ci-dessous pour naviguer rapidement vers chaque section de l'analyse.")
sections = [
    "Introduction","Apercu des donnees","Qualite des donnees","Distributions numeriques",
    "Variables categorielles","Variable cible","Correlations","Relations entre variables",
    "Profil Panne vs Sain","Taux de panne par categorie","Importance des variables",
    "ACP (Reduction de dimension)","SVM (Modelisation)","Prediction sur nouvelles donnees"
]
for i, s in enumerate(sections, 1):
    st.sidebar.markdown(f"[{i}. {s}](#{s.lower().replace(' ','-').replace('(','').replace(')','').replace(',','')})")

# ══════════════════════════════════════════════════════════════
# INTRODUCTION
# ══════════════════════════════════════════════════════════════
st.title("Analyse Exploratoire et Modelisation - Maintenance Predictive")
st.markdown("""
Ce tableau de bord presente l'analyse complete d'un jeu de donnees de **10 000 equipements industriels**.
L'objectif est double :
1. **Comprendre les donnees** a travers une analyse exploratoire (EDA) approfondie.
2. **Construire un modele predictif** (SVM) capable de detecter les pannes avant qu'elles ne surviennent, en utilisant l'ACP pour reduire la dimensionnalite.

Chaque section est accompagnee d'explications detaillees pour permettre une comprehension complete de la demarche.
""")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Observations", f"{len(df):,}")
c2.metric("Variables", f"{df.shape[1]}")
c3.metric("Taux de panne", f"{df['panne'].mean()*100:.1f}%")
c4.metric("Valeurs manquantes", f"{df.isnull().sum().sum():,}")

st.markdown("---")

# ══════════════════════════════════════════════════════════════
# APERCU DES DONNEES
# ══════════════════════════════════════════════════════════════
st.header("1. Apercu des donnees")
st.markdown("""
Le jeu de donnees contient 28 variables decrivant chaque equipement : son age, son taux d'utilisation,
son historique de maintenance et de pannes, ainsi que des caracteristiques techniques (modele, fiabilite, mode de fonctionnement).
La variable cible est `panne` (0 = fonctionnement normal, 1 = panne detectee).

**Variables numeriques** : age, indice d'usure, heures par jour, taux d'utilisation, cadence, nombre de maintenances, nombre de pannes, duree moyenne de panne, frequence de pannes, intervalles, etc.

**Variables categorielles** : modele de machine, fiabilite, phase de vie, mode de fonctionnement, charge, type de maintenance, frequence de maintenance, type de panne recurrente, cause identifiee.
""")
st.dataframe(df.head(20), use_container_width=True)

@st.cache_data
def to_excel(dataframe):
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine='xlsxwriter') as w:
        dataframe.to_excel(w, index=False, sheet_name='Donnees')
    return buf.getvalue()

st.download_button("Telecharger les donnees (Excel)", to_excel(df),
    "dataset_maintenance_predictive.xlsx",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

st.markdown("---")

# ══════════════════════════════════════════════════════════════
# QUALITE DES DONNEES
# ══════════════════════════════════════════════════════════════
st.header("2. Qualite des donnees")

# 2a. Valeurs manquantes
st.subheader("2.1 Valeurs manquantes")
st.markdown("""
Avant toute analyse, il faut verifier la completude des donnees. Les valeurs manquantes (NaN) peuvent
fausser les statistiques et empecher les algorithmes de Machine Learning de fonctionner correctement.

- **Seuil 2%** : en dessous, l'impact est negligeable (imputation simple par la mediane ou le mode).
- **Seuil 5%** : au-dela, il faut investiguer la cause des donnees manquantes (capteur defaillant ? saisie incomplete ?).
""")

missing = df.isnull().sum().sort_values(ascending=False)
missing = missing[missing > 0]
if not missing.empty:
    miss_pct = (missing / len(df) * 100).round(2)
    fig_miss = make_subplots(rows=1, cols=2, subplot_titles=("Taux de valeurs manquantes par colonne","Carte des valeurs manquantes (500 lignes)"),
        column_widths=[0.5, 0.5])
    colors = ['#EF5350' if v>5 else '#FFA726' if v>2 else '#66BB6A' for v in miss_pct.values]
    fig_miss.add_trace(go.Bar(y=miss_pct.index, x=miss_pct.values, orientation='h', marker_color=colors,
        text=[f"{v:.1f}%" for v in miss_pct.values], textposition='outside',
        hovertemplate='%{y}<br>Manquants: %{x:.2f}%<extra></extra>'), row=1, col=1)
    fig_miss.add_vline(x=5, line_dash="dash", line_color="red", annotation_text="Seuil 5%", row=1, col=1)
    fig_miss.add_vline(x=2, line_dash="dash", line_color="orange", annotation_text="Seuil 2%", row=1, col=1)

    sample = df.sample(min(500, len(df)), random_state=42)[missing.index]
    null_map = sample.isnull().astype(int)
    fig_miss.add_trace(go.Heatmap(z=null_map.T.values, y=list(missing.index), colorscale=[[0,'#f8f9fa'],[1,'#EF5350']],
        showscale=False, hovertemplate='Variable: %{y}<br>Observation: %{x}<extra></extra>'), row=1, col=2)

    fig_miss.update_layout(height=450, template='plotly_white', showlegend=False)
    st.plotly_chart(fig_miss, use_container_width=True)
    st.info(f"Total NaN : {df.isnull().sum().sum():,} | Taux global : {df.isnull().sum().sum()/(df.shape[0]*df.shape[1])*100:.2f}%")
else:
    st.success("Aucune valeur manquante detectee.")

# 2b. Valeurs aberrantes
st.subheader("2.2 Valeurs aberrantes (regles metier)")
st.markdown("""
Les valeurs aberrantes sont des observations qui violent les regles logiques du domaine industriel :
- Une machine ne peut pas fonctionner plus de **24 heures par jour**.
- Le taux d'utilisation ne peut pas depasser **100%**.
- L'indice d'usure est compris entre **0 et 100**.
- Une machine ne peut pas avoir ete fabriquee avant **1990** ni apres **2023**.

Ces anomalies doivent etre corrigees ou supprimees avant la modelisation, sous peine de fausser les predictions.
""")
aberrants = {
    'annee_fabrication': ((df['annee_fabrication']<1990)|(df['annee_fabrication']>2023)).sum(),
    'heures_par_jour': ((df['heures_par_jour']<1)|(df['heures_par_jour']>24)).sum(),
    'taux_utilisation_pct': (df['taux_utilisation_pct']>100).sum(),
    'indice_usure': ((df['indice_usure']<0)|(df['indice_usure']>100)).sum(),
    'nombre_maintenances': (df['nombre_maintenances']<0).sum(),
    'jours_depuis_maintenance': (df['jours_depuis_maintenance']>1000).sum(),
    'duree_moyenne_panne_h': ((df['duree_moyenne_panne_h']<0)|(df['duree_moyenne_panne_h']>200)).sum(),
}
ab_labels = ['Annee fab.','Heures/jour','Taux util.','Indice usure','Nb maint.','Jours dep. maint.','Duree panne']
fig_ab = go.Figure(go.Bar(x=ab_labels, y=list(aberrants.values()),
    marker_color='#EF5350', text=list(aberrants.values()), textposition='outside',
    hovertemplate='%{x}<br>Aberrants: %{y}<extra></extra>'))
fig_ab.update_layout(title="Nombre de valeurs aberrantes par variable", height=400, template='plotly_white',
    xaxis_tickangle=-25, yaxis_title="Nombre de valeurs aberrantes",
    xaxis=dict(automargin=True, tickfont=dict(size=11)),
    margin=dict(b=80))
st.plotly_chart(fig_ab, use_container_width=True)

# 2c. Boxplots
st.subheader("2.3 Detection visuelle par boxplots")
st.markdown("""
Les boxplots (boites a moustaches) permettent de visualiser la distribution d'une variable :
- La **boite** contient 50% des donnees (du 1er au 3eme quartile).
- La **ligne au milieu** est la mediane.
- Les **moustaches** s'etendent jusqu'aux valeurs extremes normales.
- Les **points isoles** au-dela des moustaches sont les outliers statistiques.

Les lignes horizontales en pointille representent les **limites metier** (valeurs physiquement possibles).
""")
box_data = [('heures_par_jour','Heures/jour',1,24), ('taux_utilisation_pct','Taux utilisation (%)',0,100),
    ('indice_usure','Indice usure',0,100), ('duree_moyenne_panne_h','Duree panne (h)',0,72),
    ('annee_fabrication','Annee fabrication',1990,2023), ('jours_depuis_maintenance','Jours depuis maint.',0,730)]
fig_box = make_subplots(rows=2, cols=3, subplot_titles=[b[1] for b in box_data])
for i, (col, lbl, lo, hi) in enumerate(box_data):
    r, c = i//3+1, i%3+1
    fig_box.add_trace(go.Box(y=df[col].dropna(), name=lbl, marker_color='#90CAF9',
        line_color='#1565C0', boxmean=True, hovertemplate='%{y}<extra></extra>'), row=r, col=c)
    fig_box.add_hline(y=lo, line_dash="dash", line_color="green", row=r, col=c)
    fig_box.add_hline(y=hi, line_dash="dash", line_color="red", row=r, col=c)
fig_box.update_layout(height=550, template='plotly_white', showlegend=False,
    title_text="Boxplots avec limites metier (vert = min, rouge = max)",
    margin=dict(t=60, b=40))
for ann in fig_box['layout']['annotations']:
    ann['font'] = dict(size=11)
st.plotly_chart(fig_box, use_container_width=True)

st.markdown("---")

# ══════════════════════════════════════════════════════════════
# DISTRIBUTIONS NUMERIQUES
# ══════════════════════════════════════════════════════════════
st.header("3. Distributions numeriques")
st.markdown("""
Les histogrammes ci-dessous montrent la **forme de la distribution** de chaque variable numerique.
Cela permet de detecter :
- Les **distributions normales** (en forme de cloche) : adaptees a la plupart des algorithmes.
- Les **distributions asymetriques** : la moyenne est tiree par les valeurs extremes (la mediane est plus representative).
- Les **distributions bimodales** : deux "pics" distincts, pouvant indiquer deux sous-populations.

La **ligne rouge** represente la moyenne, la **ligne orange** la mediane. Si elles sont proches, la distribution est symetrique.
""")
num_cols = ['age_machine_ans','indice_usure','heures_par_jour','taux_utilisation_pct',
    'cadence_moyenne_unit_h','nombre_maintenances','nombre_total_pannes','duree_moyenne_panne_h',
    'frequence_pannes_par_an','intervalle_entre_maintenances_j','jours_depuis_maintenance']
labels_num = ['Age machine (ans)','Indice usure','Heures/jour','Taux utilisation (%)',
    'Cadence (u/h)','Nb maintenances','Nb pannes total','Duree moy. panne (h)',
    'Freq. pannes/an','Intervalle maint. (j)','Jours depuis maint.']

fig_dist = make_subplots(rows=3, cols=4, subplot_titles=labels_num)
for i, (col, lbl) in enumerate(zip(num_cols, labels_num)):
    r, c = i//4+1, i%4+1
    d = df[col].dropna()
    fig_dist.add_trace(go.Histogram(x=d, nbinsx=40, marker_color=C_OK, opacity=0.85, name=lbl,
        hovertemplate='Valeur: %{x}<br>Effectif: %{y}<extra></extra>'), row=r, col=c)
    fig_dist.add_vline(x=d.mean(), line_color='#F44336', line_width=2, row=r, col=c)
    fig_dist.add_vline(x=d.median(), line_color='#FF9800', line_width=2, line_dash='dash', row=r, col=c)
fig_dist.update_layout(height=750, template='plotly_white', showlegend=False,
    title_text="Distributions des variables numeriques (rouge=moyenne, orange=mediane)",
    margin=dict(t=60, b=40))
for ann in fig_dist['layout']['annotations']:
    ann['font'] = dict(size=10)
st.plotly_chart(fig_dist, use_container_width=True)

st.markdown("---")

# ══════════════════════════════════════════════════════════════
# VARIABLES CATEGORIELLES
# ══════════════════════════════════════════════════════════════
st.header("4. Variables categorielles")
st.markdown("""
Les variables categorielles representent des attributs qualitatifs de chaque machine (son modele, sa fiabilite, etc.).
Les graphiques ci-dessous montrent la **repartition** de chaque categorie. Cela permet de verifier :
- Si les classes sont **equilibrees** (proportions similaires) ou **desequilibrees** (une categorie domine).
- Si certaines categories sont **tres rares**, ce qui pourrait poser probleme pour la modelisation.
""")
cat_cols = ['modele_machine','fiabilite_modele','phase_vie','mode_fonctionnement',
    'charge_machine','type_maintenance','frequence_maintenance','type_panne_recurrente','cause_identifiee']
cat_labels = ['Modele Machine','Fiabilite Modele','Phase Vie','Mode Fonctionnement',
    'Charge Machine','Type Maintenance','Frequence Maintenance','Type Panne Recurrente','Cause Identifiee']

fig_cat = make_subplots(rows=3, cols=3, subplot_titles=cat_labels, specs=[[{"type":"pie"}]*3]*3)
for i, col in enumerate(cat_cols):
    r, c = i//3+1, i%3+1
    vc = df[col].value_counts()
    fig_cat.add_trace(go.Pie(labels=vc.index, values=vc.values, hole=0.35,
        marker_colors=PALETTE[:len(vc)], textinfo='percent', textfont_size=9,
        textposition='inside', insidetextorientation='radial',
        hovertemplate='%{label}<br>Effectif: %{value}<br>%{percent}<extra></extra>'), row=r, col=c)
fig_cat.update_layout(height=900, template='plotly_white', showlegend=True,
    title_text="Repartition des variables categorielles",
    legend=dict(font=dict(size=9), orientation='h', y=-0.05),
    margin=dict(t=60, b=80))
for ann in fig_cat['layout']['annotations']:
    ann['font'] = dict(size=11)
st.plotly_chart(fig_cat, use_container_width=True)

st.markdown("---")

# ══════════════════════════════════════════════════════════════
# VARIABLE CIBLE
# ══════════════════════════════════════════════════════════════
st.header("5. Variable cible : panne")
st.markdown("""
La variable cible `panne` est ce que notre modele doit apprendre a predire.
C'est une variable **binaire** : 0 (pas de panne) ou 1 (panne).

**Probleme du desequilibre** : si une classe est largement majoritaire, un modele "naif" pourrait simplement
predire toujours "pas de panne" et obtenir une bonne accuracy sans jamais detecter de panne.
C'est pourquoi nous utiliserons des techniques de compensation (poids equilibres) lors de la modelisation.
""")
vc = df['panne'].value_counts()
fig_target = make_subplots(rows=1, cols=3, subplot_titles=(
    "Distribution de la cible","Taux de panne par phase de vie","Taux de panne par fiabilite"))

fig_target.add_trace(go.Bar(x=['Pas de panne (0)','Panne (1)'], y=vc.values,
    marker_color=[C_OK, C_PANNE], text=[f"{v:,} ({v/len(df)*100:.1f}%)" for v in vc.values],
    textposition='outside', hovertemplate='%{x}<br>%{y}<extra></extra>'), row=1, col=1)

cross_phase = df.groupby('phase_vie')['panne'].mean().sort_values(ascending=False)*100
colors_p = ['#EF5350' if v>20 else '#FFA726' if v>10 else '#66BB6A' for v in cross_phase.values]
fig_target.add_trace(go.Bar(x=cross_phase.index, y=cross_phase.values, marker_color=colors_p,
    text=[f"{v:.1f}%" for v in cross_phase.values], textposition='outside',
    hovertemplate='%{x}<br>Taux: %{y:.1f}%<extra></extra>'), row=1, col=2)

cross_fid = df.groupby('fiabilite_modele')['panne'].mean().sort_values(ascending=False)*100
colors_f = ['#EF5350' if v>20 else '#FFA726' if v>10 else '#66BB6A' for v in cross_fid.values]
fig_target.add_trace(go.Bar(x=cross_fid.index, y=cross_fid.values, marker_color=colors_f,
    text=[f"{v:.1f}%" for v in cross_fid.values], textposition='outside',
    hovertemplate='%{x}<br>Taux: %{y:.1f}%<extra></extra>'), row=1, col=3)

fig_target.update_layout(height=420, template='plotly_white', showlegend=False,
    margin=dict(b=80))
fig_target.update_xaxes(tickangle=-20, tickfont=dict(size=10), automargin=True, row=1, col=2)
fig_target.update_xaxes(tickangle=-20, tickfont=dict(size=10), automargin=True, row=1, col=3)
fig_target.update_yaxes(title_text="Effectif", row=1, col=1)
fig_target.update_yaxes(title_text="Taux de panne (%)", row=1, col=2)
fig_target.update_yaxes(title_text="Taux de panne (%)", row=1, col=3)
for ann in fig_target['layout']['annotations']:
    ann['font'] = dict(size=11)
st.plotly_chart(fig_target, use_container_width=True)
st.info(f"Ratio de desequilibre : {vc.iloc[0]/vc.iloc[1]:.1f}:1 -- Technique recommandee : class_weight='balanced'")

st.markdown("---")

# ══════════════════════════════════════════════════════════════
# CORRELATIONS
# ══════════════════════════════════════════════════════════════
st.header("6. Correlations")
st.markdown("""
La **matrice de correlation de Pearson** mesure la force de la relation lineaire entre deux variables numeriques.
- **+1** : correlation positive parfaite (quand l'une augmente, l'autre aussi).
- **-1** : correlation negative parfaite (quand l'une augmente, l'autre diminue).
- **0** : aucune relation lineaire.

**Comment lire ce graphique** : concentrez-vous sur la **derniere ligne/colonne** (`panne`) pour identifier
les variables les plus correlees avec la cible. Les variables fortement correlees entre elles (en rouge ou bleu fonce)
portent une information redondante -- l'ACP permettra de traiter ce probleme.
""")
num_corr = ['age_machine_ans','indice_usure','heures_par_jour','taux_utilisation_pct',
    'cadence_moyenne_unit_h','nombre_maintenances','nombre_total_pannes','duree_moyenne_panne_h',
    'frequence_pannes_par_an','jours_depuis_maintenance','nombre_pannes_repetees','panne']
corr_short = {'age_machine_ans':'Age','indice_usure':'Usure','heures_par_jour':'Heures/j',
    'taux_utilisation_pct':'Taux util.','cadence_moyenne_unit_h':'Cadence','nombre_maintenances':'Nb maint.',
    'nombre_total_pannes':'Nb pannes','duree_moyenne_panne_h':'Duree panne','frequence_pannes_par_an':'Freq. panne',
    'jours_depuis_maintenance':'Jrs dep maint.','nombre_pannes_repetees':'Pannes rep.','panne':'Panne'}
corr = df[num_corr].corr()
mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
corr_masked = corr.where(~mask)
short_labels = [corr_short[c] for c in corr.columns]

fig_corr = go.Figure(go.Heatmap(z=corr_masked.values, x=short_labels, y=short_labels,
    colorscale='RdBu_r', zmid=0, zmin=-1, zmax=1, text=corr_masked.round(2).values,
    texttemplate="%{text}", textfont=dict(size=9),
    hovertemplate='%{x} vs %{y}<br>Correlation: %{z:.3f}<extra></extra>',
    colorbar=dict(title="Coeff.")))
fig_corr.update_layout(height=600, template='plotly_white', title="Matrice de correlation (Pearson)",
    xaxis=dict(tickangle=-35, tickfont=dict(size=10), automargin=True),
    yaxis=dict(tickfont=dict(size=10), automargin=True),
    margin=dict(l=100, b=100))
st.plotly_chart(fig_corr, use_container_width=True)

# Top correlations
st.subheader("Variables les plus correlees avec la panne")
top_c = corr['panne'].drop('panne').abs().sort_values(ascending=False)
top_df = pd.DataFrame({'Variable': top_c.index, 'Correlation absolue': top_c.values,
    'Signe': ['+' if corr['panne'][c]>0 else '-' for c in top_c.index]})
st.dataframe(top_df.style.background_gradient(subset=['Correlation absolue'], cmap='Blues').format({'Correlation absolue':'{:.4f}'}),
    use_container_width=True)

st.markdown("---")

# ══════════════════════════════════════════════════════════════
# RELATIONS ENTRE VARIABLES
# ══════════════════════════════════════════════════════════════
st.header("7. Relations entre variables")
st.markdown("""
Les nuages de points permettent de visualiser la relation entre deux variables simultanement, en colorant
chaque point par son etat (panne ou non). Si les points rouges (panne) se concentrent dans une zone
specifique du graphique, cela signifie que la combinaison de ces deux variables est un indicateur de risque.
""")
pairs = [('age_machine_ans','indice_usure','Age vs Indice usure'),
    ('jours_depuis_maintenance','nombre_total_pannes','Jours depuis maint. vs Nb pannes'),
    ('taux_utilisation_pct','duree_moyenne_panne_h','Taux utilisation vs Duree panne'),
    ('age_machine_ans','nombre_total_pannes','Age vs Nb pannes total'),
    ('nombre_maintenances','nombre_total_pannes','Nb maintenances vs Nb pannes'),
    ('frequence_pannes_par_an','indice_usure','Freq. pannes vs Indice usure')]

fig_scatter = make_subplots(rows=2, cols=3, subplot_titles=[p[2] for p in pairs])
for i, (x, y, t) in enumerate(pairs):
    r, c = i//3+1, i%3+1
    samp = df[[x,y,'panne']].dropna().sample(min(2000,len(df)), random_state=42)
    for pv, col, nm in [(0,C_OK,'Sain'),(1,C_PANNE,'Panne')]:
        sub = samp[samp['panne']==pv]
        fig_scatter.add_trace(go.Scatter(x=sub[x], y=sub[y], mode='markers', name=nm,
            marker=dict(size=3, color=col, opacity=0.5), showlegend=(i==0),
            hovertemplate=f'{x}: %{{x}}<br>{y}: %{{y}}<extra>{nm}</extra>'), row=r, col=c)
fig_scatter.update_layout(height=650, template='plotly_white', title_text="Relations entre variables (colore par etat)",
    margin=dict(t=60, b=40))
for ann in fig_scatter['layout']['annotations']:
    ann['font'] = dict(size=10)
st.plotly_chart(fig_scatter, use_container_width=True)

st.markdown("---")

# ══════════════════════════════════════════════════════════════
# PROFIL COMPARE
# ══════════════════════════════════════════════════════════════
st.header("8. Profil compare : Panne vs Sain")
st.markdown("""
Ces histogrammes superposent la distribution de chaque variable pour les machines **saines** (bleu) et **en panne** (rouge).
Si les deux courbes sont decalees, cela signifie que la variable a un pouvoir discriminant fort pour predire la panne.
Par exemple, si les machines en panne ont un age moyen plus eleve, cela confirme que l'age est un facteur de risque.
""")
prof_cols = ['age_machine_ans','indice_usure','heures_par_jour','taux_utilisation_pct',
    'nombre_total_pannes','duree_moyenne_panne_h','jours_depuis_maintenance',
    'nombre_maintenances','frequence_pannes_par_an']
prof_labels = ['Age (ans)','Indice usure','Heures/jour','Taux util. (%)','Nb pannes',
    'Duree panne (h)','Jours depuis maint.','Nb maintenances','Freq. pannes/an']

fig_prof = make_subplots(rows=3, cols=3, subplot_titles=prof_labels)
for i, (col, lbl) in enumerate(zip(prof_cols, prof_labels)):
    r, c = i//3+1, i%3+1
    d0, d1 = df[df['panne']==0][col].dropna(), df[df['panne']==1][col].dropna()
    fig_prof.add_trace(go.Histogram(x=d0, nbinsx=30, marker_color=C_OK, opacity=0.6,
        name='Sain', histnorm='probability density', showlegend=(i==0)), row=r, col=c)
    fig_prof.add_trace(go.Histogram(x=d1, nbinsx=30, marker_color=C_PANNE, opacity=0.6,
        name='Panne', histnorm='probability density', showlegend=(i==0)), row=r, col=c)
    fig_prof.update_xaxes(title_text=lbl, row=r, col=c)
fig_prof.update_layout(height=750, template='plotly_white', barmode='overlay',
    title_text="Distributions comparees : Sain vs Panne",
    margin=dict(t=60, b=40))
for ann in fig_prof['layout']['annotations']:
    ann['font'] = dict(size=10)
st.plotly_chart(fig_prof, use_container_width=True)

st.markdown("---")

# ══════════════════════════════════════════════════════════════
# TAUX DE PANNE PAR CATEGORIE
# ══════════════════════════════════════════════════════════════
st.header("9. Taux de panne par variable categorielle")
st.markdown("""
Pour chaque variable categorielle, on calcule le **taux de panne** dans chaque categorie.
Cela permet d'identifier les facteurs de risque categoriques :
- Un taux eleve (rouge) indique une categorie a haut risque de panne.
- Un taux bas (vert) indique une categorie fiable.
""")
cat_vs = ['phase_vie','fiabilite_modele','type_maintenance','charge_machine','mode_fonctionnement','type_panne_recurrente']
cat_lbl = ['Phase de vie','Fiabilite modele','Type maintenance','Charge machine','Mode fonctionnement','Type panne recurrente']
fig_cat_t = make_subplots(rows=2, cols=3, subplot_titles=[f"Par {l}" for l in cat_lbl],
    vertical_spacing=0.18, horizontal_spacing=0.08)
for i, (col, lbl) in enumerate(zip(cat_vs, cat_lbl)):
    r, c = i//3+1, i%3+1
    cross = df.groupby(col)['panne'].mean().sort_values(ascending=False)*100
    cols_ = ['#EF5350' if v>20 else '#FFA726' if v>12 else '#66BB6A' for v in cross.values]
    fig_cat_t.add_trace(go.Bar(x=cross.index, y=cross.values, marker_color=cols_,
        text=[f"{v:.1f}%" for v in cross.values], textposition='outside',
        hovertemplate='%{x}<br>Taux: %{y:.1f}%<extra></extra>'), row=r, col=c)
    fig_cat_t.update_xaxes(tickangle=-25, tickfont=dict(size=9), automargin=True, row=r, col=c)
    fig_cat_t.update_yaxes(title_text="Taux (%)", row=r, col=c)
fig_cat_t.update_layout(height=600, template='plotly_white', showlegend=False,
    title_text="Taux de panne par categorie", margin=dict(b=80, t=60))
for ann in fig_cat_t['layout']['annotations']:
    ann['font'] = dict(size=10)
st.plotly_chart(fig_cat_t, use_container_width=True)

st.markdown("---")

# ══════════════════════════════════════════════════════════════
# IMPORTANCE DES VARIABLES (RF)
# ══════════════════════════════════════════════════════════════
st.header("10. Importance des variables (Random Forest)")
st.markdown("""
Pour identifier objectivement les variables les plus utiles pour predire les pannes, on entraine un modele
**Random Forest** exploratoire. Ce modele calcule l'**importance de Gini** de chaque variable : plus la valeur
est elevee, plus la variable est utile pour separer les cas de panne des cas normaux.

Ce classement guide le choix des variables a conserver et confirme (ou infirme) les intuitions de l'EDA.
""")

@st.cache_resource
def compute_rf_importance(dataframe):
    d = dataframe.copy()
    for c in d.select_dtypes('object').columns:
        d[c] = LabelEncoder().fit_transform(d[c].astype(str))
    X = d.drop(columns=['id','panne','date_derniere_maintenance'])
    y = d['panne']
    X_imp = SimpleImputer(strategy='median').fit_transform(X)
    rf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced', n_jobs=-1)
    rf.fit(X_imp, y)
    return pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=True)

feat_imp = compute_rf_importance(df)
cols_rf = ['#EF5350' if v>feat_imp.quantile(0.75) else '#FFA726' if v>feat_imp.median() else '#90CAF9' for v in feat_imp.values]
fig_rf = go.Figure(go.Bar(y=feat_imp.index, x=feat_imp.values, orientation='h', marker_color=cols_rf,
    hovertemplate='%{y}<br>Importance: %{x:.4f}<extra></extra>'))
fig_rf.add_vline(x=feat_imp.median(), line_dash="dash", line_color="gray", annotation_text="Mediane")
fig_rf.update_layout(height=600, template='plotly_white', title="Importance des variables (Random Forest - 100 arbres)",
    xaxis_title="Importance (Gini)")
st.plotly_chart(fig_rf, use_container_width=True)

st.markdown("---")

# ══════════════════════════════════════════════════════════════
# ACP (Analyse en Composantes Principales)
# ══════════════════════════════════════════════════════════════
st.header("11. Analyse en Composantes Principales (ACP)")
st.markdown("""
### Objectif et fondements theoriques

Conformement a la theorie, l'**A.C.P.** permet d'analyser des donnees multidimensionnelles 
composées de plusieurs **variables correlées**, et de les reduire en un nombre reduit de 
**nouvelles variables decorrelees** appelées "composantes principales", tout en 
conservant le maximum d'information (variance).

**Le processus theorique applique ici :**
1. **Donnees centrees-reduites** : Pour annuler l'effet des unites, toutes les variables sont standardisees (moyenne = 0, ecart-type = 1).
2. **Inertie totale** : Elle mesure la dispersion globale du nuage de points et correspond a la trace de la matrice de variances-covariances.
3. **Axes principaux** : Ce sont les vecteurs propres de la matrice de variances-covariances, ordonnes par valeur propre (variance) decroissante.
4. **Composantes principales** : Ce sont de nouvelles variables (combinaisons lineaires des variables initiales). Elles fournissent les coordonnees des individus sur les axes principaux. Elles sont non-correlees entre elles.
5. **Critere global** : On retient les premieres composantes dont la somme cumulée des variances represente au moins **95% de l'inertie totale**.

L'ACP est d'autant plus efficace que les variables de depart sont correlees (ce qui est notre cas, cf EDA).
""")

# Variables selectionnees sur la base de l'EDA et du Random Forest
SELECTED_NUM = ['age_machine_ans', 'indice_usure', 'nombre_total_pannes',
    'nombre_pannes_repetees', 'nombre_maintenances', 'jours_depuis_maintenance']
SELECTED_CAT = ['phase_vie']
SELECTED_FEATURES = SELECTED_NUM + SELECTED_CAT

@st.cache_resource
def run_pca_v2(dataframe):
    d = dataframe.copy()
    y = d['panne']
    X = d[SELECTED_FEATURES].copy()
    nc = [c for c in SELECTED_NUM if c in X.columns]
    cc = [c for c in SELECTED_CAT if c in X.columns]
    num_t = Pipeline([('imp', SimpleImputer(strategy='median')),('sc', StandardScaler())])
    cat_t = Pipeline([('imp', SimpleImputer(strategy='most_frequent')),('oh', OneHotEncoder(handle_unknown='ignore', sparse_output=False))])
    prep = ColumnTransformer([('n', num_t, nc),('c', cat_t, cc)])
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    Xtr_s = prep.fit_transform(X_tr); Xte_s = prep.transform(X_te)
    nf = nc + prep.named_transformers_['c']['oh'].get_feature_names_out(cc).tolist()
    pf = PCA(random_state=42); pf.fit(Xtr_s)
    cv = np.cumsum(pf.explained_variance_ratio_)*100
    n95 = int(np.argmax(cv>=95)+1)
    return pf, cv, n95, Xtr_s, Xte_s, y_tr, y_te, nf

pca_full, cum_var, n95, Xtr_s, Xte_s, y_train, y_test, feat_names = run_pca_v2(df)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Variables selectionnees", len(SELECTED_FEATURES))
c2.metric("Colonnes apres encodage", Xtr_s.shape[1])
c3.metric("Composantes ACP (95%)", n95)
c4.metric("Reduction totale", f"{len(df.columns)-3} -> {n95}")

# Scree Plot
n_show = min(30, len(cum_var))
fig_sc = make_subplots(rows=1, cols=2, subplot_titles=("Variance individuelle","Variance cumulee (Scree Plot)"))
fig_sc.add_trace(go.Bar(x=list(range(1,n_show+1)), y=(pca_full.explained_variance_ratio_[:n_show]*100).tolist(),
    marker_color='#2196F3', hovertemplate='PC%{x}<br>%{y:.2f}%<extra></extra>'), row=1, col=1)
fig_sc.add_trace(go.Scatter(x=list(range(1,n_show+1)), y=cum_var[:n_show].tolist(),
    mode='lines+markers', marker=dict(size=5, color='#2196F3'),
    hovertemplate='PC1-PC%{x}<br>Cumul: %{y:.2f}%<extra></extra>'), row=1, col=2)
fig_sc.add_hline(y=95, line_dash="dash", line_color="#F44336", annotation_text="95%", row=1, col=2)
fig_sc.add_vline(x=n95, line_dash="dash", line_color="#4CAF50", annotation_text=f"{n95} comp.", row=1, col=2)
fig_sc.update_layout(height=400, template='plotly_white', showlegend=False)
fig_sc.update_xaxes(title_text="Composante", row=1, col=1)
fig_sc.update_xaxes(title_text="Nombre de composantes", row=1, col=2)
fig_sc.update_yaxes(title_text="Variance (%)", row=1, col=1)
fig_sc.update_yaxes(title_text="Variance cumulee (%)", row=1, col=2)
st.plotly_chart(fig_sc, use_container_width=True)

# Slider
n_slider = st.slider("Nombre de composantes principales :", 2, min(Xtr_s.shape[1],40), n95,
    help="Deplacez pour voir l'impact sur la variance capturee et les performances du SVM.")
st.info(f"Avec {n_slider} composantes : variance capturee = {cum_var[n_slider-1]:.2f}% (reduction de {Xtr_s.shape[1]} a {n_slider} colonnes)")

# Projection 2D
st.subheader("Projection 2D (PC1 vs PC2)")
st.markdown("""
Ce graphique projette les 8000 machines d'entrainement dans l'espace defini par les deux premieres composantes.
Si les points rouges (panne) et bleus (sain) se separent, cela confirme que l'ACP capture des tendances discriminantes.
""")
p2d = PCA(n_components=2, random_state=42); X2d = p2d.fit_transform(Xtr_s)
df_2d = pd.DataFrame({'PC1':X2d[:,0],'PC2':X2d[:,1],'Etat':y_train.map({0:'Sain',1:'Panne'}).values})
fig_2d = px.scatter(df_2d, x='PC1', y='PC2', color='Etat', color_discrete_map={'Sain':C_OK,'Panne':C_PANNE},
    opacity=0.5, labels={'PC1':f'PC1 ({p2d.explained_variance_ratio_[0]*100:.1f}%)',
    'PC2':f'PC2 ({p2d.explained_variance_ratio_[1]*100:.1f}%)'})
fig_2d.update_traces(marker_size=3)
fig_2d.update_layout(height=500, template='plotly_white', legend_title="Etat")
st.plotly_chart(fig_2d, use_container_width=True)

# Contributions
st.subheader("Contributions des variables aux composantes principales")
st.markdown("Ce tableau montre le **poids** de chaque variable originale dans les premieres composantes. Un poids eleve signifie que la variable contribue fortement.")
nl = min(5, n95)
ld = pd.DataFrame(pca_full.components_[:nl].T, index=feat_names, columns=[f'PC{i+1}' for i in range(nl)])
top = ld.apply(lambda x: x.abs()).mean(axis=1).sort_values(ascending=False).head(15)
st.dataframe(ld.loc[top.index].style.background_gradient(cmap='RdBu_r', axis=None).format("{:.3f}"), use_container_width=True)

st.markdown("---")

# ══════════════════════════════════════════════════════════════
# SVM
# ══════════════════════════════════════════════════════════════
st.header("12. Modelisation SVM sur les composantes ACP")
st.markdown(f"""
### Qu'est-ce que le SVM ?

Le **Support Vector Machine** (SVM) est un algorithme de classification supervisee qui cherche l'**hyperplan optimal**
separant les deux classes (panne / pas de panne) dans un espace a haute dimension.

Avec un noyau **RBF** (Radial Basis Function), le SVM peut tracer des frontieres de decision **non lineaires**,
ce qui est essentiel car la relation entre les variables et la panne n'est pas necessairement lineaire.

### Pipeline complet applique :
1. **Preprocessing** : imputation des manquants, encodage One-Hot, standardisation (centrage-reduction).
2. **ACP** : reduction a **{n_slider} composantes** (variance conservee : {cum_var[n_slider-1]:.1f}%).
3. **SVM** : noyau RBF, `class_weight='balanced'` pour compenser le desequilibre des classes.
4. **Evaluation** : sur l'ensemble de test (20% des donnees, jamais vu pendant l'entrainement).
""")

@st.cache_resource
def train_svm(_Xtr, _Xte, _ytr, _yte, nc):
    p = PCA(n_components=nc, random_state=42)
    Xtr_p = p.fit_transform(_Xtr); Xte_p = p.transform(_Xte)
    s = SVC(kernel='rbf', class_weight='balanced', random_state=42)
    s.fit(Xtr_p, _ytr); yp = s.predict(Xte_p)
    return yp, p, s, Xtr_p, Xte_p

with st.spinner(f"Entrainement du SVM sur {n_slider} composantes..."):
    y_pred, pca_m, svm_m, Xtr_p, Xte_p = train_svm(Xtr_s, Xte_s, y_train, y_test, n_slider)

acc = accuracy_score(y_test, y_pred)
rep = classification_report(y_test, y_pred, output_dict=True)
c1, c2, c3, c4 = st.columns(4)
c1.metric("Accuracy", f"{acc:.2%}")
c2.metric("Precision (Panne)", f"{rep['1']['precision']:.2%}")
c3.metric("Recall (Panne)", f"{rep['1']['recall']:.2%}")
c4.metric("F1-Score (Panne)", f"{rep['1']['f1-score']:.2%}")

st.markdown("""
### Interpretation des metriques

| Metrique | Definition | Importance en maintenance predictive |
|---|---|---|
| **Accuracy** | Proportion totale de predictions correctes | Trompeuse si les classes sont desequilibrees |
| **Precision** | Parmi les alertes "Panne", combien sont reelles | Limite les fausses alertes (interventions inutiles) |
| **Recall** | Parmi les vraies pannes, combien ont ete detectees | **Metrique prioritaire** : rater une panne = danger |
| **F1-Score** | Moyenne harmonique precision/recall | Compromis entre fausses alertes et pannes ratees |
""")

# Confusion matrix
st.subheader("Matrice de Confusion")
st.markdown("""
La matrice de confusion detaille les predictions du modele :
- **Vrai Negatif** (haut-gauche) : machine saine correctement identifiee. C'est le cas ideal pour les machines saines.
- **Faux Positif** (haut-droite) : fausse alerte. Le modele predit une panne sur une machine saine.
- **Faux Negatif** (bas-gauche) : **le cas le plus dangereux**. Une panne reelle non detectee.
- **Vrai Positif** (bas-droite) : panne correctement detectee. C'est le cas ideal pour la maintenance predictive.
""")
cm = confusion_matrix(y_test, y_pred); tn, fp, fn, tp = cm.ravel()
fig_cm = go.Figure(go.Heatmap(
    z=[[tn,fp],[fn,tp]], x=['Predit: Sain','Predit: Panne'], y=['Reel: Sain','Reel: Panne'],
    text=[[f"Vrai Negatif<br>{tn}",f"Faux Positif<br>{fp}"],[f"Faux Negatif<br>{fn}",f"Vrai Positif<br>{tp}"]],
    texttemplate="%{text}", textfont_size=16, colorscale='Blues', showscale=False,
    hovertemplate='%{y} / %{x}<br>Nombre: %{z}<extra></extra>'))
fig_cm.update_layout(height=400, template='plotly_white', title="Matrice de Confusion : SVM + ACP",
    yaxis=dict(autorange='reversed'), xaxis_title="Prediction", yaxis_title="Realite")
st.plotly_chart(fig_cm, use_container_width=True)

# Decision boundary
st.subheader("Frontiere de decision du SVM (projection 2D)")
st.markdown("""
Ce graphique montre la **frontiere de decision** apprise par le SVM, projetee dans l'espace des deux premieres composantes.
Les zones colorees representent les regions ou le modele predit "Sain" (bleu clair) ou "Panne" (rouge clair).
Les points sont les observations reelles du jeu de test. Un point dans la mauvaise zone est une erreur du modele.
""")
p2v = PCA(n_components=2, random_state=42)
Xtr2 = p2v.fit_transform(Xtr_s); Xte2 = p2v.transform(Xte_s)
s2 = SVC(kernel='rbf', class_weight='balanced', random_state=42); s2.fit(Xtr2, y_train)
xm, xM = Xte2[:,0].min()-1, Xte2[:,0].max()+1
ym, yM = Xte2[:,1].min()-1, Xte2[:,1].max()+1
xx, yy = np.meshgrid(np.linspace(xm,xM,120), np.linspace(ym,yM,120))
Z = s2.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

fig_db = go.Figure()
fig_db.add_trace(go.Contour(x=np.linspace(xm,xM,120), y=np.linspace(ym,yM,120), z=Z,
    colorscale=[[0,'rgba(33,150,243,0.15)'],[1,'rgba(244,67,54,0.15)']], showscale=False,
    line=dict(width=1, color='gray'), contours=dict(showlabels=False), hoverinfo='skip'))
for pv, col, nm in [(0,C_OK,'Sain (test)'),(1,C_PANNE,'Panne (test)')]:
    mk = y_test.values==pv
    fig_db.add_trace(go.Scatter(x=Xte2[mk,0], y=Xte2[mk,1], mode='markers', name=nm,
        marker=dict(size=5, color=col, opacity=0.7, line=dict(width=0.5, color='white')),
        hovertemplate=f'{nm}<br>PC1: %{{x:.2f}}<br>PC2: %{{y:.2f}}<extra></extra>'))
fig_db.update_layout(height=500, template='plotly_white', title='Frontiere de decision SVM (PC1 vs PC2)',
    xaxis_title=f'PC1 ({p2v.explained_variance_ratio_[0]*100:.1f}%)',
    yaxis_title=f'PC2 ({p2v.explained_variance_ratio_[1]*100:.1f}%)')
st.plotly_chart(fig_db, use_container_width=True)

# Rapport complet
st.subheader("Rapport de classification complet")
rdf = pd.DataFrame(rep).T.rename(index={'0':'Sain (0)','1':'Panne (1)','accuracy':'Accuracy globale',
    'macro avg':'Moyenne Macro','weighted avg':'Moyenne Ponderee'})
st.dataframe(rdf.style.format({'precision':'{:.4f}','recall':'{:.4f}','f1-score':'{:.4f}','support':'{:.0f}'}).background_gradient(
    cmap='Blues', subset=['precision','recall','f1-score']), use_container_width=True)

st.markdown("""
### Conclusion de la modelisation

L'analyse exploratoire a revele que les variables **age de la machine**, **indice d'usure**,
**nombre total de pannes** et **frequence des pannes** sont les principaux facteurs de risque.

L'**ACP** a permis de reduire significativement la dimensionnalite tout en conservant l'essentiel de l'information.
Le **SVM** entraine sur ces composantes obtient un **Recall eleve** sur la classe Panne, ce qui est
la metrique la plus critique en maintenance predictive : chaque panne non detectee represente un risque
materiel et humain.
""")

st.markdown("---")

# ======================================================================
# PREDICTION SUR NOUVELLES DONNEES
# ======================================================================
st.header("13. Prediction sur de nouvelles donnees")
st.markdown("""
### Utilisation du modele entraine

Le modele SVM entraine ci-dessus peut maintenant etre utilise pour **predire si une nouvelle machine
risque de tomber en panne**. Remplissez le formulaire ci-dessous avec les caracteristiques d'un
nouvel equipement, et le modele retournera sa prediction ainsi que le niveau de risque.

Le pipeline complet est applique automatiquement :
1. Les valeurs saisies sont transformees exactement comme les donnees d'entrainement (imputation, encodage, standardisation).
2. L'ACP projette les donnees dans l'espace reduit.
3. Le SVM classe l'equipement comme "Sain" ou "Panne".
""")

# -- Guide des variables discriminantes --
st.subheader("Guide : variables les plus discriminantes")
st.markdown("""
Le tableau ci-dessous identifie les **7 variables qui influencent le plus la prediction**.
Les champs correspondants sont marques en **rouge** dans le formulaire. Ce sont ceux qu'il faut
faire varier en priorite pour tester le modele.
""")

guide_data = pd.DataFrame({
    'Variable': ['Phase de vie', 'Indice d usure', 'Age machine (ans)',
        'Nombre total de pannes', 'Nombre de pannes repetees',
        'Nombre de maintenances', 'Jours depuis maintenance'],
    'Impact': ['Tres fort', 'Tres fort', 'Fort', 'Fort', 'Fort', 'Moyen', 'Moyen'],
    'Valeur typique SAIN': ['Normale / Nouvelle', '43', '12', '9', '7', '18', '61'],
    'Valeur typique PANNE': ['Ancienne', '92', '26', '21', '19', '39', '90'],
})
def color_impact(val):
    if val == 'Tres fort':
        return 'background-color: #FFCDD2; color: #B71C1C; font-weight: bold'
    elif val == 'Fort':
        return 'background-color: #FFE0B2; color: #E65100; font-weight: bold'
    else:
        return 'background-color: #FFF9C4; color: #F57F17'
st.dataframe(guide_data.style.applymap(color_impact, subset=['Impact']), use_container_width=True, hide_index=True)

st.subheader("Formulaire de saisie")

# Variables that are most discriminant -- will be highlighted
KEY_NUM_VARS = {'age_machine_ans', 'indice_usure', 'nombre_total_pannes',
    'nombre_pannes_repetees', 'nombre_maintenances', 'jours_depuis_maintenance'}
KEY_CAT_VARS = {'phase_vie'}

# Pipeline de prediction : applique PCA sur toutes les variables
@st.cache_resource
def get_full_pipeline(dataframe):
    d = dataframe.copy()
    y = d['panne']
    X = d.drop(columns=['id','panne','date_derniere_maintenance'])
    nc = X.select_dtypes(include=['int64','float64']).columns
    cc = X.select_dtypes(include=['object']).columns
    num_t = Pipeline([('imp', SimpleImputer(strategy='median')),('sc', StandardScaler())])
    cat_t = Pipeline([('imp', SimpleImputer(strategy='most_frequent')),('oh', OneHotEncoder(handle_unknown='ignore', sparse_output=False))])
    prep = ColumnTransformer([('n', num_t, nc),('c', cat_t, cc)])
    X_tr, _, y_tr, _ = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    Xtr_s = prep.fit_transform(X_tr)
    pca_pred = PCA(n_components=n_slider, random_state=42)
    Xtr_p = pca_pred.fit_transform(Xtr_s)
    svm_pred = SVC(kernel='rbf', class_weight='balanced', random_state=42, probability=True)
    svm_pred.fit(Xtr_p, y_tr)
# Pipeline de prediction : utilise les memes variables selectionnees que l'ACP+SVM
@st.cache_resource
def get_full_pipeline_v2(dataframe):
    d = dataframe.copy()
    y = d['panne']
    X = d[SELECTED_FEATURES].copy()
    nc = SELECTED_NUM
    cc = SELECTED_CAT
    num_t = Pipeline([('imp', SimpleImputer(strategy='median')),('sc', StandardScaler())])
    cat_t = Pipeline([('imp', SimpleImputer(strategy='most_frequent')),('oh', OneHotEncoder(handle_unknown='ignore', sparse_output=False))])
    prep = ColumnTransformer([('n', num_t, nc),('c', cat_t, cc)])
    X_tr, _, y_tr, _ = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    Xtr_s = prep.fit_transform(X_tr)
    pca_pred = PCA(n_components=n_slider, random_state=42)
    Xtr_p = pca_pred.fit_transform(Xtr_s)
    svm_pred = SVC(kernel='rbf', class_weight='balanced', random_state=42, probability=True)
    svm_pred.fit(Xtr_p, y_tr)
    return prep, pca_pred, svm_pred, nc, cc

prep_pred, pca_pred, svm_pred, num_cols_pred, cat_cols_pred = get_full_pipeline_v2(df)

# Build the form with sensible defaults from dataset medians/modes
with st.form("prediction_form"):
    st.markdown('**Variables numeriques** (6 variables cles) -- Ces variables ont ete selectionnees '
        'car elles sont les plus discriminantes pour predire les pannes :')
    
    num_defaults = {c: float(df[c].median()) for c in num_cols_pred}
    num_ranges = {c: (float(df[c].min()), float(df[c].max())) for c in num_cols_pred}
    
    num_labels_map = {
        'age_machine_ans': 'Age de la machine (annees)',
        'indice_usure': 'Indice d usure (0-100)',
        'nombre_maintenances': 'Nombre de maintenances',
        'nombre_total_pannes': 'Nombre total de pannes',
        'jours_depuis_maintenance': 'Jours depuis la derniere maintenance',
        'nombre_pannes_repetees': 'Nombre de pannes repetees',
    }
    
    num_inputs = {}
    cols_form = st.columns(3)
    for idx, c in enumerate(num_cols_pred):
        label = num_labels_map.get(c, c.replace('_', ' ').title())
        lo, hi = num_ranges[c]
        with cols_form[idx % 3]:
            num_inputs[c] = st.number_input(label, min_value=lo, max_value=hi,
                value=num_defaults[c], step=(hi-lo)/100, format="%.2f")
    
    st.markdown("---")
    st.markdown('**Variable categorielle** -- La variable la plus discriminante du dataset :')
    
    cat_labels_map = {'phase_vie': 'Phase de vie'}
    
    cat_inputs = {}
    for c in cat_cols_pred:
        label = cat_labels_map.get(c, c.replace('_', ' ').title())
        options = sorted(df[c].dropna().unique().tolist())
        cat_inputs[c] = st.selectbox(label, options)
    
    submitted = st.form_submit_button("Lancer la prediction", type="primary")

if submitted:
    # Build a single-row DataFrame matching the training format
    new_data = {}
    for c in num_cols_pred:
        new_data[c] = [num_inputs[c]]
    for c in cat_cols_pred:
        new_data[c] = [cat_inputs[c]]
    new_df = pd.DataFrame(new_data)
    
    # Apply the full pipeline
    new_scaled = prep_pred.transform(new_df)
    new_pca = pca_pred.transform(new_scaled)
    prediction = svm_pred.predict(new_pca)[0]
    proba = svm_pred.predict_proba(new_pca)[0]
    
    st.subheader("Resultat de la prediction")
    
    if prediction == 1:
        st.error(f"PREDICTION : RISQUE DE PANNE DETECTE")
        st.markdown(f"Le modele estime avec une probabilite de **{proba[1]*100:.1f}%** que cet equipement "
            f"presente un risque de panne. Il est recommande de planifier une intervention de maintenance preventive.")
    else:
        st.success(f"PREDICTION : EQUIPEMENT SAIN")
        st.markdown(f"Le modele estime avec une probabilite de **{proba[0]*100:.1f}%** que cet equipement "
            f"fonctionne normalement. Aucune intervention immediate n'est necessaire.")
    
    # Risk gauge
    risk_pct = proba[1] * 100
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=risk_pct,
        number={'suffix': '%', 'font': {'size': 40}},
        title={'text': 'Niveau de risque de panne', 'font': {'size': 16}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 2},
            'bar': {'color': '#F44336' if risk_pct > 50 else '#FF9800' if risk_pct > 25 else '#4CAF50'},
            'bgcolor': 'white',
            'steps': [
                {'range': [0, 25], 'color': '#E8F5E9'},
                {'range': [25, 50], 'color': '#FFF3E0'},
                {'range': [50, 75], 'color': '#FFEBEE'},
                {'range': [75, 100], 'color': '#FFCDD2'}
            ],
            'threshold': {'line': {'color': '#212121', 'width': 3}, 'thickness': 0.8, 'value': risk_pct}
        }
    ))
    fig_gauge.update_layout(height=300, template='plotly_white', margin=dict(t=60, b=20))
    st.plotly_chart(fig_gauge, use_container_width=True)
    
    # Summary table
    st.subheader("Recapitulatif des valeurs saisies")
    recap = pd.DataFrame({
        'Variable': [num_labels_map.get(c, c) for c in num_cols_pred] + [cat_labels_map.get(c, c) for c in cat_cols_pred],
        'Valeur': [num_inputs[c] for c in num_cols_pred] + [cat_inputs[c] for c in cat_cols_pred],
        'Mediane/Mode dataset': [f"{df[c].median():.2f}" for c in num_cols_pred] + [df[c].mode()[0] for c in cat_cols_pred]
    })
    st.dataframe(recap, use_container_width=True)

st.markdown("---")
st.success("Analyse complete : EDA + ACP + Modelisation SVM + Prediction")
