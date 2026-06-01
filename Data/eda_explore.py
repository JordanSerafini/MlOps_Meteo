#!/usr/bin/env python3
"""EDA exhaustif du dataset Kaggle "Rain in Australia" (weatherAUS.csv).

Produit des chiffres reels (rien d'invente) et dump l'ensemble dans eda_stats.json.
Objectif: nourrir un rapport sur la prediction de RainTomorrow.
"""
import json
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(HERE, "weatherAUS.csv")
OUT_JSON = os.path.join(HERE, "eda_stats.json")

# NA est deja un marqueur natif, mais on force pour etre explicite
df = pd.read_csv(CSV_PATH, na_values=["NA"])

stats = {}

# ---------------------------------------------------------------------------
# 1. Structure generale
# ---------------------------------------------------------------------------
stats["shape"] = {"n_rows": int(df.shape[0]), "n_cols": int(df.shape[1])}
stats["columns"] = list(df.columns)
stats["dtypes"] = {c: str(t) for c, t in df.dtypes.items()}
stats["n_duplicates"] = int(df.duplicated().sum())

# ---------------------------------------------------------------------------
# 2. Dates
# ---------------------------------------------------------------------------
df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
date_min = df["Date"].min()
date_max = df["Date"].max()
stats["dates"] = {
    "min": str(date_min.date()),
    "max": str(date_max.date()),
    "n_unique_dates": int(df["Date"].nunique()),
    "n_years_covered": int(df["Date"].dt.year.nunique()),
    "years": sorted(int(y) for y in df["Date"].dt.year.dropna().unique()),
}

# ---------------------------------------------------------------------------
# 3. Locations / stations
# ---------------------------------------------------------------------------
loc_counts = df["Location"].value_counts()
stats["locations"] = {
    "n_locations": int(df["Location"].nunique()),
    "min_rows_per_station": int(loc_counts.min()),
    "max_rows_per_station": int(loc_counts.max()),
    "station_min_rows": str(loc_counts.idxmin()),
    "station_max_rows": str(loc_counts.idxmax()),
    "mean_rows_per_station": round(float(loc_counts.mean()), 1),
}

# ---------------------------------------------------------------------------
# 4. Valeurs manquantes par colonne (% trie decroissant)
# ---------------------------------------------------------------------------
missing_pct = (df.isna().mean() * 100).sort_values(ascending=False)
stats["missing_pct"] = {c: round(float(v), 2) for c, v in missing_pct.items()}

# ---------------------------------------------------------------------------
# 5. Cible RainTomorrow
# ---------------------------------------------------------------------------
target = df["RainTomorrow"]
vc = target.value_counts(dropna=False)
n_yes = int(vc.get("Yes", 0))
n_no = int(vc.get("No", 0))
n_nan = int(target.isna().sum())
n_valid = n_yes + n_no
base_rate_yes = 100.0 * n_yes / n_valid  # taux de base = part de Yes
naive_always_no = 100.0 * n_no / n_valid  # baseline "toujours Non"
stats["target"] = {
    "counts": {"Yes": n_yes, "No": n_no, "NaN": n_nan},
    "pct_missing": round(100.0 * n_nan / len(df), 2),
    "base_rate_yes_pct": round(base_rate_yes, 2),
    "naive_always_no_pct": round(naive_always_no, 2),
}

# Cible binaire 0/1 pour correlations et conditions
y = target.map({"Yes": 1, "No": 0})

# ---------------------------------------------------------------------------
# 6. Baseline persistance: RainTomorrow == RainToday
# ---------------------------------------------------------------------------
mask = df["RainToday"].notna() & target.notna()
rt = df.loc[mask, "RainToday"]
tt = target.loc[mask]
persistence_acc = 100.0 * (rt.values == tt.values).mean()
# P(pluie demain | RainToday=Yes/No)
p_rain_given_today_yes = 100.0 * (tt[rt == "Yes"] == "Yes").mean()
p_rain_given_today_no = 100.0 * (tt[rt == "No"] == "Yes").mean()
stats["baseline_persistence"] = {
    "accuracy_pct": round(persistence_acc, 2),
    "n_rows_used": int(mask.sum()),
    "p_rain_tomorrow_given_raintoday_yes_pct": round(float(p_rain_given_today_yes), 2),
    "p_rain_tomorrow_given_raintoday_no_pct": round(float(p_rain_given_today_no), 2),
}

# ---------------------------------------------------------------------------
# 7. Colonnes numeriques: describe + correlation point-biseriale avec y
# ---------------------------------------------------------------------------
num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
desc = df[num_cols].describe().T
stats["numeric_describe"] = {
    c: {
        "min": round(float(desc.loc[c, "min"]), 3),
        "max": round(float(desc.loc[c, "max"]), 3),
        "mean": round(float(desc.loc[c, "mean"]), 3),
        "std": round(float(desc.loc[c, "std"]), 3),
        "count": int(desc.loc[c, "count"]),
    }
    for c in num_cols
}

# Correlation point-biseriale = Pearson(num, y_binaire), sur lignes communes non-nulles
pb_corr = {}
for c in num_cols:
    sub = pd.DataFrame({"x": df[c], "y": y}).dropna()
    if sub["x"].std() == 0 or len(sub) < 2:
        pb_corr[c] = None
    else:
        pb_corr[c] = round(float(sub["x"].corr(sub["y"])), 4)
# tri par |corr| decroissant (None en dernier)
pb_sorted = dict(
    sorted(pb_corr.items(), key=lambda kv: (kv[1] is None, -abs(kv[1]) if kv[1] is not None else 0))
)
stats["pointbiserial_corr"] = pb_sorted
stats["pointbiserial_corr_highlight"] = {
    c: pb_corr.get(c)
    for c in [
        "Humidity3pm", "Humidity9am", "Pressure3pm", "Sunshine",
        "Cloud3pm", "Rainfall", "WindGustSpeed", "Temp3pm",
    ]
    if c in pb_corr
}

# ---------------------------------------------------------------------------
# 8. Moyennes par classe RainTomorrow (Yes vs No)
# ---------------------------------------------------------------------------
class_means = {}
for c in ["Humidity3pm", "Sunshine", "Pressure3pm", "Rainfall"]:
    g = df.groupby(target)[c].mean()
    class_means[c] = {
        "No": round(float(g.get("No", np.nan)), 3),
        "Yes": round(float(g.get("Yes", np.nan)), 3),
    }
stats["class_means"] = class_means

# ---------------------------------------------------------------------------
# 9. WindGustDir: distribution + taux de pluie demain par direction
# ---------------------------------------------------------------------------
wgd_dist = df["WindGustDir"].value_counts(dropna=False)
wgd_rate = (
    df.assign(y=y)
    .dropna(subset=["WindGustDir", "y"])
    .groupby("WindGustDir")["y"]
    .agg(["mean", "count"])
)
wgd_rate["mean_pct"] = (wgd_rate["mean"] * 100).round(2)
wgd_rate_sorted = wgd_rate.sort_values("mean_pct", ascending=False)
stats["windgustdir"] = {
    "distribution": {str(k): int(v) for k, v in wgd_dist.items()},
    "rain_rate_pct": {str(k): float(v) for k, v in wgd_rate_sorted["mean_pct"].items()},
    "top3_wettest": {
        str(k): float(v) for k, v in wgd_rate_sorted["mean_pct"].head(3).items()
    },
    "bottom3_driest": {
        str(k): float(v) for k, v in wgd_rate_sorted["mean_pct"].tail(3).items()
    },
}

# ---------------------------------------------------------------------------
# 10. Geographie: taux de pluie demain par Location
# ---------------------------------------------------------------------------
loc_rate = (
    df.assign(y=y)
    .dropna(subset=["y"])
    .groupby("Location")["y"]
    .agg(["mean", "count"])
)
loc_rate["mean_pct"] = (loc_rate["mean"] * 100).round(2)
loc_rate_sorted = loc_rate.sort_values("mean_pct", ascending=False)
stats["geography"] = {
    "rain_rate_by_location_pct": {
        str(k): float(v) for k, v in loc_rate_sorted["mean_pct"].items()
    },
    "top5_wettest": {
        str(k): float(v) for k, v in loc_rate_sorted["mean_pct"].head(5).items()
    },
    "top5_driest": {
        str(k): float(v) for k, v in loc_rate_sorted["mean_pct"].tail(5).items()
    },
}

# ---------------------------------------------------------------------------
# 11. Saisonnalite: taux de pluie demain par mois
# ---------------------------------------------------------------------------
df["Month"] = df["Date"].dt.month
month_rate = (
    df.assign(y=y)
    .dropna(subset=["y", "Month"])
    .groupby("Month")["y"]
    .mean()
    * 100
).round(2)
stats["seasonality"] = {
    "rain_rate_by_month_pct": {int(k): float(v) for k, v in month_rate.items()},
    "wettest_month": int(month_rate.idxmax()),
    "wettest_month_pct": float(month_rate.max()),
    "driest_month": int(month_rate.idxmin()),
    "driest_month_pct": float(month_rate.min()),
}

# ---------------------------------------------------------------------------
# Dump JSON
# ---------------------------------------------------------------------------
with open(OUT_JSON, "w") as f:
    json.dump(stats, f, default=str, indent=2)

# ---------------------------------------------------------------------------
# Resume lisible
# ---------------------------------------------------------------------------
print("=" * 70)
print("EDA  weatherAUS.csv  (Rain in Australia)")
print("=" * 70)
print(f"Lignes x Colonnes : {stats['shape']['n_rows']} x {stats['shape']['n_cols']}")
print(f"Doublons          : {stats['n_duplicates']}")
print(f"Dates             : {stats['dates']['min']} -> {stats['dates']['max']} "
      f"({stats['dates']['n_years_covered']} annees, {stats['dates']['n_unique_dates']} dates uniques)")
print(f"Stations          : {stats['locations']['n_locations']} "
      f"({stats['locations']['min_rows_per_station']} a {stats['locations']['max_rows_per_station']} lignes)")
print()
print("--- Cible RainTomorrow ---")
print(f"  Yes={n_yes}  No={n_no}  NaN={n_nan}  (%manquant cible={stats['target']['pct_missing']}%)")
print(f"  Taux de base (Yes) = {stats['target']['base_rate_yes_pct']}%")
print(f"  Baseline 'toujours Non' = {stats['target']['naive_always_no_pct']}% accuracy")
print(f"  Baseline persistance (=RainToday) = {stats['baseline_persistence']['accuracy_pct']}% accuracy")
print(f"    P(pluie|RainToday=Yes) = {stats['baseline_persistence']['p_rain_tomorrow_given_raintoday_yes_pct']}%")
print(f"    P(pluie|RainToday=No)  = {stats['baseline_persistence']['p_rain_tomorrow_given_raintoday_no_pct']}%")
print()
print("--- Top 8 valeurs manquantes (%) ---")
for c, v in list(missing_pct.items())[:8]:
    print(f"  {c:16s} {v:5.1f}%")
print()
print("--- Correlations point-biseriales avec RainTomorrow (top 8 |corr|) ---")
for c, v in list(pb_sorted.items())[:8]:
    print(f"  {c:16s} {v}")
print()
print("--- Moyennes par classe (No vs Yes) ---")
for c, d in class_means.items():
    print(f"  {c:14s} No={d['No']:>8}   Yes={d['Yes']:>8}")
print()
print("--- Geo: top5 pluvieuses / top5 seches ---")
print("  Pluvieuses:", stats["geography"]["top5_wettest"])
print("  Seches    :", stats["geography"]["top5_driest"])
print()
print("--- Saisonnalite (taux pluie demain par mois %) ---")
print(" ", stats["seasonality"]["rain_rate_by_month_pct"])
print(f"  Mois le + pluvieux: {stats['seasonality']['wettest_month']} "
      f"({stats['seasonality']['wettest_month_pct']}%)  |  "
      f"+ sec: {stats['seasonality']['driest_month']} ({stats['seasonality']['driest_month_pct']}%)")
print()
print(f"JSON ecrit -> {OUT_JSON}")
