"""Task 1 - Netflix Content Recommendation System (content-based, TF-IDF + cosine similarity)"""
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from common import *

df = load().reset_index(drop=True)
rng = np.random.default_rng(SEED)

# ---- Step 1: prepare content features (each feature becomes a list of tokens) ----
def tok(v): return v.str.replace(" ", "_").str.replace("&", "and")
feat = {
    "genre":    df["genres"].apply(lambda L: "|".join(x.replace(" ", "_") for x in L)),
    "country":  df["countries"].apply(lambda L: "|".join(x.replace(" ", "_") for x in L)),
    "director": df["director"].where(df.has_director == 1, "").apply(lambda s: "|".join(x.strip().replace(" ", "_") for x in s.split(",")) if s else ""),
    "rating":   df["rating"],
    "type":     df["type"].str.replace(" ", "_"),
}
# importance of each feature in the similarity score (sum = 1)
WEIGHTS = {"genre": 0.50, "country": 0.15, "rating": 0.15, "type": 0.10, "director": 0.10}

# ---- Step 2: convert text to numbers (TF-IDF, one vectorizer per feature) ----
vecs = {k: TfidfVectorizer(tokenizer=lambda s: [t for t in s.split("|") if t], token_pattern=None, lowercase=False) for k in feat}
blocks = {k: vecs[k].fit_transform(feat[k]) for k in feat}

def combine(bl, w=WEIGHTS):
    X = hstack([np.sqrt(w[k]) * normalize(bl[k]) for k in w]).tocsr()
    return normalize(X)

X = combine(blocks)
print("Feature matrix:", X.shape)

# ---- Step 3 + 4: similarity scores and recommendations ----
title_idx = pd.Series(df.index, index=df["title"]).groupby(level=0).first()
def recommend(title, k=10):
    i = title_idx[title]
    sims = (X[i] @ X.T).toarray().ravel()          # cosine similarity (rows are L2-normalised)
    sims[i] = -1                                    # do not recommend the title itself
    top = np.argsort(-sims)[:k]
    out = df.loc[top, ["title", "type", "listed_in", "country", "rating"]].copy()
    out.insert(1, "similarity", np.round(sims[top], 3))
    return out

demo_candidates = ["Stranger Things", "Narcos", "Kung Fu Panda", "Sherlock", "Dangal", "Naruto", "The Crown", "Peaky Blinders"]
demo = [t for t in demo_candidates if t in title_idx.index][:4]
lines = []
for t in demo:
    r = recommend(t, 5)
    q = df.loc[title_idx[t]]
    print(f"\n=== Because you watched: {t} ({q['type']}, {q['listed_in']}) ===")
    print(r[["title", "similarity", "type", "listed_in"]].to_string(index=False))
    r.insert(0, "query", t); lines.append(r)
pd.concat(lines).to_csv(f"{RES}/task1_demo_recommendations.csv", index=False)

# ---- Step 5: evaluate recommendation quality ----
K = 10
def jaccard(a, b): return len(set(a) & set(b)) / len(set(a) | set(b))
sample = rng.choice(len(df), 1000, replace=False)
same_type, genre_jac, rand_jac = [], [], []
for i in sample:
    sims = (X[i] @ X.T).toarray().ravel(); sims[i] = -1
    top = np.argsort(-sims)[:K]
    same_type.append((df.loc[top, "type"] == df.loc[i, "type"]).mean())
    genre_jac.append(np.mean([jaccard(df.loc[i, "genres"], df.loc[j, "genres"]) for j in top]))
    rj = rng.choice(len(df), K, replace=False)
    rand_jac.append(np.mean([jaccard(df.loc[i, "genres"], df.loc[j, "genres"]) for j in rj]))
base_type = (df["type"] == df["type"].iloc[0]).mean()

# Non-circular test: hide ONE genre of the query title, then check if the
# recommendations (made from the other features) still contain that hidden genre.
multi = [i for i in rng.permutation(len(df)) if df.loc[i, "n_genres"] >= 2][:1500]
genre_share = df["genres"].explode().value_counts() / len(df)
def hit_rate(w):
    Xw = combine(blocks, w)
    hits, base = [], []
    for i in multi:
        hidden = df.loc[i, "genres"][rng.integers(df.loc[i, "n_genres"])]
        kept = [g for g in df.loc[i, "genres"] if g != hidden]
        qb = dict(blocks); qb = {k: blocks[k][i] for k in blocks}
        qb["genre"] = vecs["genre"].transform(["|".join(g.replace(" ", "_") for g in kept)])
        q = combine(qb, w)
        sims = (q @ Xw.T).toarray().ravel(); sims[i] = -1
        top = np.argsort(-sims)[:K]
        hits.append(np.mean([hidden in df.loc[j, "genres"] for j in top])); base.append(genre_share[hidden])
    return float(np.mean(hits)), float(np.mean(base))
hit_all, base_rate = hit_rate(WEIGHTS)
hit_genre_only, _ = hit_rate({"genre": 1.0, "country": 1e-9, "rating": 1e-9, "type": 1e-9, "director": 1e-9})

res = {
    "n_titles": len(df), "feature_matrix_shape": list(X.shape), "weights": WEIGHTS, "k": K,
    "same_type_in_top10": float(np.mean(same_type)), "same_type_random_baseline": float(df["type"].value_counts(normalize=True).pow(2).sum()),
    "genre_jaccard_top10": float(np.mean(genre_jac)), "genre_jaccard_random": float(np.mean(rand_jac)),
    "hidden_genre_precision_at10_all_features": hit_all, "hidden_genre_precision_at10_genre_only": hit_genre_only,
    "hidden_genre_random_baseline": base_rate, "demo_titles": demo,
}
print("\nEvaluation:", json.dumps(res, indent=1))
save_json("task1_results.json", res)

fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
ax[0].bar(["Random", "Recommender"], [res["genre_jaccard_random"], res["genre_jaccard_top10"]], color=["gray", "#1f77b4"])
ax[0].set_title("Genre overlap (Jaccard) of top-10"); ax[0].set_ylim(0, 1)
for i, v in enumerate([res["genre_jaccard_random"], res["genre_jaccard_top10"]]): ax[0].text(i, v + .02, f"{v:.2f}", ha="center")
vals = [res["hidden_genre_random_baseline"], res["hidden_genre_precision_at10_genre_only"], res["hidden_genre_precision_at10_all_features"]]
ax[1].bar(["Random", "Other genres\nonly", "All features"], vals, color=["gray", "#ff7f0e", "#1f77b4"])
ax[1].set_title("Hidden-genre test: precision@10"); ax[1].set_ylim(0, 1)
for i, v in enumerate(vals): ax[1].text(i, v + .02, f"{v:.2f}", ha="center")
plt.tight_layout(); plt.savefig(f"{FIG}/task1_evaluation.png", dpi=150); plt.close()
