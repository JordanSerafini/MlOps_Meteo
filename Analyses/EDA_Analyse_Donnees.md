# Première analyse des données — *Rain in Australia* (Kaggle)

> Dataset `Data/weatherAUS.csv` · Cible : `RainTomorrow` (pleut-il demain, ≥ 1 mm : Yes/No)
> Chiffres issus de `Data/eda_explore.py` → `Data/eda_stats.json` (calcul reproductible) et recoupés par le notebook `notebooks/01_analyse_meteo.ipynb`.

## 1. Vue d'ensemble

| Élément | Valeur |
|---|---|
| Observations | **145 460** lignes |
| Variables | **23** colonnes (1 date, 6 catégorielles, 16 numériques) |
| Période | **2007-11-01 → 2017-06-25** (~11 années civiles, 3 436 dates) |
| Stations météo | **49** localités (de 1 578 à 3 436 lignes ; moy. ~2 969) |
| Doublons exacts | **0** |

Données journalières multi-stations : température, pluie, évaporation, ensoleillement, vent (direction/vitesse), humidité, pression, nébulosité, à 9 h et 15 h. Deux colonnes dérivées : `RainToday` et la cible `RainTomorrow`.

## 2. Qualité des données & valeurs manquantes

Aucune ligne dupliquée, mais une **missingness hétérogène et structurée** :

| Colonne | % manquant | Colonne | % manquant |
|---|---|---|---|
| Sunshine | **48,0 %** | WindGustDir | 7,1 % |
| Evaporation | **43,2 %** | WindGustSpeed | 7,1 % |
| Cloud3pm | **40,8 %** | Humidity3pm | 3,1 % |
| Cloud9am | **38,4 %** | RainTomorrow (cible) | 2,3 % |
| Pressure9am / 3pm | ~10,3 % | Rainfall / RainToday | 2,2 % |

- **4 colonnes très lacunaires** (`Sunshine`, `Evaporation`, `Cloud9am/3pm`) : ce ne sont pas des trous aléatoires (certaines stations ne mesurent pas ces variables) → imputer globalement injecte du bruit.
- La **cible** est absente sur **3 267 lignes (2,25 %)** : ces lignes sont retirées (non imputables).
- Toutes les autres colonnes restent sous ~10 % → imputation raisonnable.

> ⚠️ `Sunshine` et `Cloud3pm` sont **parmi les meilleurs prédicteurs** (cf. §4) tout en étant les plus lacunaires : les **supprimer** (choix du notebook d'origine) sacrifie du signal. On préfère les **conserver et imputer** (médiane), en assumant l'incertitude — idéalement avec un indicateur « valeur manquante ».

## 3. Variable cible & déséquilibre

| Classe | Effectif | Part |
|---|---|---|
| No (pas de pluie) | 110 316 | **77,58 %** |
| Yes (pluie) | 31 877 | **22,42 %** |
| Manquant | 3 267 | (retirées) |

**Classe fortement déséquilibrée (~78/22).** Conséquence majeure : l'*accuracy* seule est trompeuse. Deux **baselines** fixent la barre à battre :

| Baseline | Accuracy | Idée |
|---|---|---|
| « toujours Non » | **77,58 %** | prédire la classe majoritaire |
| persistance « = RainToday » | **76,23 %** | demain = aujourd'hui |

La pluie de la veille est informative : **P(pluie demain \| il a plu) = 46,4 %** contre **15,2 %** sinon. `RainToday` (et `Rainfall`) sont donc utiles — mais à manier prudemment (risque de **fuite** si construits à partir d'infos non disponibles au moment de la prédiction). **Métrique à suivre : recall / F1 de la classe « pluie »**, pas l'accuracy.

## 4. Variables explicatives clés

Corrélations point-bisériales avec la cible (|corr| décroissant) :

| Variable | Corr. | Variable | Corr. |
|---|---|---|---|
| **Sunshine** | **−0,451** | Pressure9am | −0,246 |
| **Humidity3pm** | **+0,446** | Rainfall | +0,239 |
| **Cloud3pm** | **+0,382** | WindGustSpeed | +0,234 |
| Cloud9am | +0,317 | Pressure3pm | −0,226 |
| Humidity9am | +0,257 | Temp3pm | −0,192 |

Moyennes par classe (très discriminantes) :

| Variable | No (pas de pluie) | Yes (pluie) |
|---|---|---|
| Humidity3pm (%) | 46,5 | **68,8** |
| Sunshine (h) | 8,5 | **4,5** |
| Pressure3pm (hPa) | 1016,1 | **1012,3** |
| Rainfall (mm) | 1,27 | **6,14** |

**Interprétation physique cohérente** : pluie le lendemain ⇔ moins de soleil, humidité de l'après-midi plus élevée, plus de nuages, pression plus basse. Ces variables porteront l'essentiel du pouvoir prédictif.

## 5. Dimensions catégorielle, géographique & saisonnière

- **Direction du vent (`WindGustDir`)** : le taux de pluie demain varie de ~15 % (vents d'**Est** : E, ENE, ESE) à ~28–29 % (vents de secteur **Nord-Ouest** : NW, NNW, WNW). Variable pertinente.
- **Géographie** (taux de pluie demain par station) : très contrasté.
  - Plus pluvieuses : **Portland 36,6 %**, Walpole 33,7 %, Cairns 31,8 %, Dartmoor 31,3 %, NorfolkIsland 31,0 % (côtes humides).
  - Plus sèches : **Woomera 6,8 %**, Uluru 7,6 %, AliceSprings 8,1 %, Mildura 10,9 %, Cobar 12,9 % (intérieur désertique).
- **Saisonnalité** (hémisphère sud) : pic en **hiver austral** — mois le plus pluvieux **juillet (26,9 %)**, juin 26,2 %, août 25,3 % — creux en **janvier (19,3 %)**. → une feature `Month`/saison est justifiée.

## 6. Implications pour la modélisation

1. **Gérer le déséquilibre** : `class_weight="balanced"`, sur/sous-échantillonnage (SMOTE), ou **ajustement du seuil** de décision selon le coût métier (rater un jour de pluie vs fausse alerte).
2. **Cibler le bon score** : recall/F1 de la classe « pluie » et ROC-AUC, pas l'accuracy.
3. **Éviter la fuite de données** : imputation/standardisation **apprises sur le train uniquement** (pipeline scikit-learn), et vigilance sur `RainToday`/`Rainfall`.
4. **Missingness** : pour `Sunshine`/`Cloud`/`Evaporation`, ajouter des indicateurs de manquant plutôt qu'imputer en aveugle.
5. **Feature engineering temporel** : mois/saison (cyclique), et **validation temporelle** (ne pas prédire le passé avec le futur) pour un usage réaliste.
6. **Modèles** : régression logistique (référence) et Random Forest atteignent ~0,85 d'accuracy / ~0,87–0,89 de ROC-AUC mais ~0,50 de recall « pluie » → marge nette via les leviers ci-dessus, puis gradient boosting (XGBoost/LightGBM).

> Ces traitements seront ensuite industrialisés (scripts `process.py`/`train.py`/`evaluate.py`, tracking **MLflow**, versionnement **DVC**, **API** d'inférence) — cf. roadmap MLOps du projet.
