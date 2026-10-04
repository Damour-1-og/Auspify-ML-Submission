"""Shared helpers: load the Netflix dataset and build basic features."""
import os, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG, RES = os.path.join(ROOT, "figures"), os.path.join(ROOT, "results")
os.makedirs(FIG, exist_ok=True); os.makedirs(RES, exist_ok=True)
SEED = 42

# Merge the 14 raw ratings into 4 audience-maturity groups (NR / UR carry no maturity info)
RATING_GROUP = {
    "TV-Y": "Kids", "TV-Y7": "Kids", "TV-Y7-FV": "Kids", "TV-G": "Kids", "G": "Kids",
    "TV-PG": "Older Kids", "PG": "Older Kids",
    "TV-14": "Teens", "PG-13": "Teens",
    "TV-MA": "Adults", "R": "Adults", "NC-17": "Adults",
}

def load():
    df = pd.read_csv(os.path.join(ROOT, "data", "Dataset.csv"))
    df["genres"] = df["listed_in"].str.split(",").apply(lambda g: [x.strip() for x in g])
    df["n_genres"] = df["genres"].apply(len)
    df["has_director"] = (df["director"] != "Not Given").astype(int)
    df["countries"] = df["country"].apply(lambda s: [] if s == "Not Given" else [c.strip() for c in s.split(",")])
    df["primary_country"] = df["countries"].apply(lambda c: c[0] if c else "Unknown")
    d = pd.to_datetime(df["date_added"])
    df["year_added"], df["month_added"] = d.dt.year, d.dt.month
    df["duration_value"] = df["duration"].str.extract(r"(\d+)")[0].astype(int)   # minutes (movies) or seasons (TV)
    df["rating_group"] = df["rating"].map(RATING_GROUP)                          # NaN for NR / UR
    return df

def genre_matrix(df, top=None):
    """Multi-hot matrix of genres (all genres, or only the `top` most common)."""
    s = df["genres"].explode()
    cols = list(s.value_counts().index if top is None else s.value_counts().index[:top])
    return pd.DataFrame({g: df["genres"].apply(lambda L: int(g in L)) for g in cols}, index=df.index)

def country_matrix(df, top=15):
    cols = list(df["primary_country"].value_counts().index[:top])
    return pd.DataFrame({"country_" + c: (df["primary_country"] == c).astype(int) for c in cols}, index=df.index)

def _clean(o):
    """Replace NaN with None so the JSON file is valid."""
    if isinstance(o, dict): return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)): return [_clean(v) for v in o]
    if hasattr(o, "item"): o = o.item()
    if isinstance(o, float) and o != o: return None
    return o

def save_json(name, obj):
    with open(os.path.join(RES, name), "w") as f:
        json.dump(_clean(obj), f, indent=2, default=str)
