"""Task 4 - Netflix Content Segmentation (unsupervised: K-Means + Agglomerative)"""
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, adjusted_rand_score
from common import *

df = load().reset_index(drop=True)

# ---- Step 1: prepare numerical + categorical features ----
ordinal = {"Kids": 0, "Older Kids": 1, "Teens": 2, "Adults": 3}
maturity = df["rating_group"].map(ordinal); maturity = maturity.fillna(maturity.median())
# duration is minutes for movies and seasons for shows, so standardise it inside each type
dur_z = df.groupby("type")["duration_value"].transform(lambda s: (s - s.mean()) / s.std())
num = pd.DataFrame({"is_tv_show": (df.type == "TV Show").astype(int), "release_year": df.release_year, "year_added": df.year_added,
                    "duration_z_in_type": dur_z, "maturity_level": maturity, "n_genres": df.n_genres})
cat = pd.concat([genre_matrix(df, top=12).add_prefix("genre_"), country_matrix(df, 6)], axis=1)
X = pd.concat([num, cat], axis=1)
Xs = StandardScaler().fit_transform(X)                      # data scaling (K-Means uses distances)
print("Feature matrix:", Xs.shape)

# ---- Step 2: apply clustering algorithms ----
ks = range(2, 11); inertia, sil = [], []
for k in ks:
    km = KMeans(k, n_init=10, random_state=SEED).fit(Xs)
    inertia.append(km.inertia_); sil.append(silhouette_score(Xs, km.labels_, sample_size=4000, random_state=SEED))
    print(f"k={k}: inertia={km.inertia_:,.0f}  silhouette={sil[-1]:.3f}")
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].plot(list(ks), inertia, "o-"); ax[0].set_title("Elbow method"); ax[0].set_xlabel("k"); ax[0].set_ylabel("Inertia")
ax[1].plot(list(ks), sil, "o-", color="#ff7f0e"); ax[1].set_title("Silhouette score"); ax[1].set_xlabel("k")
plt.tight_layout(); plt.savefig(f"{FIG}/task4_choose_k.png", dpi=150); plt.close()

K = 6      # chosen with elbow + silhouette + interpretability (see report)
km = KMeans(K, n_init=20, random_state=SEED).fit(Xs); df["cluster"] = km.labels_
agg = AgglomerativeClustering(K, linkage="ward").fit(Xs)
print("\nAgglomerative (Ward) silhouette:", round(silhouette_score(Xs, agg.labels_, sample_size=4000, random_state=SEED), 3),
      "| K-Means silhouette:", round(silhouette_score(Xs, km.labels_, sample_size=4000, random_state=SEED), 3),
      "| agreement (Adjusted Rand):", round(adjusted_rand_score(km.labels_, agg.labels_), 3))

# ---- Step 3: identify content groups  /  Step 5: interpret ----
rows = []
for c in range(K):
    s = df[df.cluster == c]
    g = s["genres"].explode().value_counts(normalize=True).head(3)
    rows.append({"cluster": c, "size": len(s), "share_%": round(100 * len(s) / len(df), 1), "tv_show_%": round(100 * (s.type == "TV Show").mean(), 1),
                 "mean_release_year": round(s.release_year.mean(), 1), "mean_year_added": round(s.year_added.mean(), 1),
                 "avg_movie_minutes": round(s[s.type == "Movie"].duration_value.mean(), 0) if (s.type == "Movie").any() else np.nan,
                 "avg_tv_seasons": round(s[s.type == "TV Show"].duration_value.mean(), 2) if (s.type == "TV Show").any() else np.nan,
                 "top_rating": s.rating.value_counts().index[0], "adults_%": round(100 * (s.rating_group == "Adults").mean(), 1),
                 "kids_%": round(100 * s.rating_group.isin(["Kids", "Older Kids"]).mean(), 1),
                 "top_countries": ", ".join(s.primary_country.value_counts().head(3).index),
                 "top_genres": "; ".join(f"{k} ({v:.0%})" for k, v in g.items())})
prof = pd.DataFrame(rows); pd.set_option("display.width", 250, "display.max_colwidth", 90)
print("\n", prof.T.to_string()); prof.to_csv(f"{RES}/task4_cluster_profiles.csv", index=False)

# ---- Step 4: visualise clusters (PCA to 2D) ----
pca = PCA(2, random_state=SEED); P = pca.fit_transform(Xs)
fig, ax = plt.subplots(figsize=(8, 6))
sc = ax.scatter(P[:, 0], P[:, 1], c=df.cluster, cmap="tab10", s=6, alpha=.6)
ax.legend(*sc.legend_elements(), title="Cluster", loc="best", fontsize=8)
ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.0%} of variance)"); ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.0%})"); ax.set_title("Netflix content clusters (PCA view)")
plt.tight_layout(); plt.savefig(f"{FIG}/task4_clusters_pca.png", dpi=150); plt.close()

fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
prof.set_index("cluster")["size"].plot.bar(ax=ax[0], rot=0, color="#1f77b4"); ax[0].set_title("Cluster sizes")
hm = df.assign(**{c: X[c] for c in ["is_tv_show", "maturity_level", "n_genres"]}).groupby("cluster")[["is_tv_show", "maturity_level", "release_year", "year_added", "n_genres"]].mean()
hm = (hm - hm.min()) / (hm.max() - hm.min())
im = ax[1].imshow(hm.T, cmap="viridis", aspect="auto"); ax[1].set_yticks(range(hm.shape[1])); ax[1].set_yticklabels(hm.columns); ax[1].set_xticks(range(K))
ax[1].set_title("Cluster profile (scaled 0-1)"); ax[1].set_xlabel("Cluster"); plt.colorbar(im, ax=ax[1])
plt.tight_layout(); plt.savefig(f"{FIG}/task4_cluster_profile.png", dpi=150); plt.close()

df[["show_id", "title", "type", "cluster"]].to_csv(f"{RES}/task4_title_clusters.csv", index=False)
save_json("task4_results.json", {"k": K, "silhouette_by_k": dict(zip(map(int, ks), sil)), "inertia_by_k": dict(zip(map(int, ks), inertia)),
    "kmeans_silhouette": float(silhouette_score(Xs, km.labels_, sample_size=4000, random_state=SEED)),
    "agg_silhouette": float(silhouette_score(Xs, agg.labels_, sample_size=4000, random_state=SEED)),
    "ari": float(adjusted_rand_score(km.labels_, agg.labels_)), "pca_var": pca.explained_variance_ratio_.tolist(),
    "profiles": prof.to_dict("records"), "n_features": Xs.shape[1]})
