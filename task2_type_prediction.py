"""Task 2 - Content Type Prediction (Movie vs TV Show)"""
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from sklearn.base import clone
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay
from common import *

df = load()
y = (df["type"] == "TV Show").astype(int)          # 1 = TV Show, 0 = Movie
print("Class balance:\n", df["type"].value_counts(normalize=True).round(3))

# ---- Step 1: select features ----
# 'duration' is NOT used: movies are "90 min", shows are "2 Seasons", so it gives the answer away (data leakage).
# We prove this below.
base = pd.DataFrame({"release_year": df.release_year, "year_added": df.year_added, "month_added": df.month_added,
                     "has_director": df.has_director, "n_genres": df.n_genres}, index=df.index)
# ---- Step 2: encode categorical variables ----
rating_oh = pd.get_dummies(df["rating"], prefix="rating").astype(int)       # one-hot
country_oh = country_matrix(df, top=15)                                     # one-hot of 15 most common countries
genres = genre_matrix(df).add_prefix("genre_")                              # multi-hot genres
X_no_genre = pd.concat([base, rating_oh, country_oh], axis=1)
X_full = pd.concat([X_no_genre, genres], axis=1)
print("Features without genres:", X_no_genre.shape[1], "| with genres:", X_full.shape[1])

# leakage demonstration
leak = DecisionTreeClassifier(max_depth=2, random_state=SEED)
Xl_tr, Xl_te, yl_tr, yl_te = train_test_split(df[["duration_value"]].assign(is_min=df.duration.str.contains("min").astype(int)), y, test_size=.2, stratify=y, random_state=SEED)
leak.fit(Xl_tr[["is_min"]], yl_tr)
leak_acc = accuracy_score(yl_te, leak.predict(Xl_te[["is_min"]]))
print("Leakage check - accuracy using only 'is the duration in minutes?':", round(leak_acc, 4))

# ---- Step 3: train classification models ----
models = {
    "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
    "KNN (k=15)": make_pipeline(StandardScaler(), KNeighborsClassifier(15)),
    "Decision Tree": DecisionTreeClassifier(max_depth=8, min_samples_leaf=5, random_state=SEED),
    "Random Forest": RandomForestClassifier(300, min_samples_leaf=2, random_state=SEED, n_jobs=-1),
    "Gradient Boosting": GradientBoostingClassifier(random_state=SEED),
}
cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
rows, fitted = [], {}
for fs_name, X in [("Without genres", X_no_genre), ("With genres", X_full)]:
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=.2, stratify=y, random_state=SEED)
    for name, m0 in models.items():
        m = clone(m0)                      # fresh copy so both feature sets keep their own fitted model
        m.fit(Xtr, ytr); pred = m.predict(Xte); proba = m.predict_proba(Xte)[:, 1]
        rows.append({"Features": fs_name, "Model": name,
                     "Accuracy": accuracy_score(yte, pred), "Precision": precision_score(yte, pred),
                     "Recall": recall_score(yte, pred), "F1": f1_score(yte, pred), "ROC_AUC": roc_auc_score(yte, proba),
                     "CV_Accuracy": cross_val_score(m, X, y, cv=cv, scoring="accuracy").mean()})
        fitted[(fs_name, name)] = (m, Xte, yte)
# ---- Step 4 + 5: evaluate and compare ----
res = pd.DataFrame(rows).round(4)
print(res.to_string(index=False)); res.to_csv(f"{RES}/task2_model_comparison.csv", index=False)
majority = float(max(y.mean(), 1 - y.mean()))
best = res[res.Features == "With genres"].sort_values("Accuracy", ascending=False).iloc[0]
best_nog = res[res.Features == "Without genres"].sort_values("Accuracy", ascending=False).iloc[0]
print("\nMajority-class baseline:", round(majority, 4)); print("Best (with genres):", best.Model, best.Accuracy)

# plots
fig, ax = plt.subplots(figsize=(9, 4.5))
w = .38; xs = np.arange(len(models))
for k, (fs_name, c) in enumerate([("Without genres", "#ff7f0e"), ("With genres", "#1f77b4")]):
    v = res[res.Features == fs_name].set_index("Model").loc[list(models)]["Accuracy"].values
    b = ax.bar(xs + (k - .5) * w, v, w, label=fs_name, color=c)
    for r_, val in zip(b, v): ax.text(r_.get_x() + r_.get_width() / 2, val + .005, f"{val:.3f}", ha="center", fontsize=8)
ax.axhline(majority, ls="--", color="gray"); ax.text(-.45, majority + .008, f"Always predicting 'Movie' = {majority:.3f}", ha="left", fontsize=8, color="gray")
ax.set_xticks(xs); ax.set_xticklabels(list(models), rotation=15); ax.set_ylim(.6, 1.09); ax.set_ylabel("Test accuracy"); ax.set_title("Task 2: model accuracy"); ax.legend(loc="upper center", ncol=2)
plt.tight_layout(); plt.savefig(f"{FIG}/task2_model_accuracy.png", dpi=150); plt.close()

m, Xte, yte = fitted[("Without genres", best_nog.Model)]   # headline model: genre names such as "TV Dramas" leak the type
fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
ConfusionMatrixDisplay(confusion_matrix(yte, m.predict(Xte)), display_labels=["Movie", "TV Show"]).plot(ax=ax[0], cmap="Blues", colorbar=False)
ax[0].set_title(f"Confusion matrix - {best_nog.Model} (no genres)")
rf = fitted[("Without genres", "Random Forest")][0]
imp = pd.Series(rf.feature_importances_, index=X_no_genre.columns).sort_values(ascending=False).head(10)[::-1]
ax[1].barh(imp.index, imp.values, color="#1f77b4"); ax[1].set_title("Top 10 features (Random Forest)")
plt.tight_layout(); plt.savefig(f"{FIG}/task2_confusion_importance.png", dpi=150); plt.close()
cm = confusion_matrix(yte, m.predict(Xte))
print("Headline model:", best_nog.Model, best_nog.Accuracy)
print("Top features (RF, no genres):", imp[::-1].round(3).to_dict())
save_json("task2_results.json", {"majority_baseline": majority, "leak_accuracy": leak_acc, "best_with_genres": best.to_dict(), "best_without_genres": best_nog.to_dict(),
          "confusion_matrix": cm.tolist(), "n_features_no_genre": X_no_genre.shape[1], "n_features_full": X_full.shape[1],
          "top_features": imp[::-1].round(4).to_dict(), "class_share_tv": float(y.mean())})
