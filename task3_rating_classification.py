"""Task 3 - Netflix Audience Rating Classification (multi-class)"""
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from sklearn.base import clone
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from common import *

df = load()

# ---- Step 1: analyze rating categories ----
raw_counts = df["rating"].value_counts()
print("Raw ratings:\n", raw_counts.to_string())
order = ["Kids", "Older Kids", "Teens", "Adults"]
d = df.dropna(subset=["rating_group"]).copy()                  # drop NR / UR (82 titles, no maturity information)
print("\nDropped NR/UR titles:", len(df) - len(d))
print("Audience groups:\n", d["rating_group"].value_counts().reindex(order).to_string())

fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
raw_counts.plot.bar(ax=ax[0], color="#1f77b4"); ax[0].set_title("14 raw rating labels (very unbalanced)"); ax[0].set_ylabel("Titles")
g = d["rating_group"].value_counts().reindex(order); g.plot.bar(ax=ax[1], color="#ff7f0e", rot=0); ax[1].set_title("4 audience groups used as target")
for i, v in enumerate(g): ax[1].text(i, v + 40, str(v), ha="center")
plt.tight_layout(); plt.savefig(f"{FIG}/task3_rating_distribution.png", dpi=150); plt.close()

# ---- Step 2: prepare training dataset ----
X = pd.concat([pd.DataFrame({"is_tv_show": (d.type == "TV Show").astype(int), "duration_value": d.duration_value,
                             "release_year": d.release_year, "year_added": d.year_added,
                             "n_genres": d.n_genres, "has_director": d.has_director}, index=d.index),
               country_matrix(d, 15), genre_matrix(d).add_prefix("genre_")], axis=1)
y = d["rating_group"]
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=.2, stratify=y, random_state=SEED)
print("\nTrain/test:", Xtr.shape, Xte.shape, "| features:", X.shape[1])

def evaluate(name, model, results):
    pred = model.predict(Xte)
    results.append({"Model": name, "Accuracy": accuracy_score(yte, pred),
                    "Macro_F1": f1_score(yte, pred, average="macro"), "Weighted_F1": f1_score(yte, pred, average="weighted")})
    return pred

# ---- Step 3: train classification models (default settings) ----
results, preds = [], {}
models = {"Baseline (most common class)": DummyClassifier(strategy="most_frequent"),
          "Decision Tree (default)": DecisionTreeClassifier(random_state=SEED),
          "Random Forest (default)": RandomForestClassifier(200, random_state=SEED, n_jobs=-1),
          "Hist. Gradient Boosting (default)": HistGradientBoostingClassifier(random_state=SEED)}
for n, m in models.items():
    m.fit(Xtr, ytr); preds[n] = evaluate(n, m, results)

# ---- Step 4: optimize (hyperparameter tuning, 5-fold CV, scoring = macro F1 so small groups matter) ----
cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
dt_search = RandomizedSearchCV(DecisionTreeClassifier(random_state=SEED, class_weight="balanced"),
    {"max_depth": [4, 6, 8, 10, 14, None], "min_samples_leaf": [1, 3, 5, 10, 20], "criterion": ["gini", "entropy"]},
    n_iter=25, cv=cv, scoring="f1_macro", random_state=SEED, n_jobs=-1).fit(Xtr, ytr)
rf_search = RandomizedSearchCV(RandomForestClassifier(random_state=SEED, n_jobs=-1, class_weight="balanced_subsample"),
    {"n_estimators": [200, 400], "max_depth": [10, 15, 20, None], "min_samples_leaf": [1, 2, 4], "max_features": ["sqrt", 0.3, 0.5]},
    n_iter=15, cv=cv, scoring="f1_macro", random_state=SEED, n_jobs=-1).fit(Xtr, ytr)
print("\nBest Decision Tree params:", dt_search.best_params_, "| CV macro-F1:", round(dt_search.best_score_, 4))
print("Best Random Forest params:", rf_search.best_params_, "| CV macro-F1:", round(rf_search.best_score_, 4))
preds["Decision Tree (tuned)"] = evaluate("Decision Tree (tuned)", dt_search.best_estimator_, results)
preds["Random Forest (tuned)"] = evaluate("Random Forest (tuned)", rf_search.best_estimator_, results)

# ---- Step 5: evaluate prediction accuracy ----
res = pd.DataFrame(results).round(4); print("\n", res.to_string(index=False)); res.to_csv(f"{RES}/task3_model_comparison.csv", index=False)
# headline model = best Decision Tree / Random Forest by macro-F1 (Gradient Boosting is only an extra comparison)
cand = res[res.Model.str.contains("Decision Tree|Random Forest")]
best_name = cand.sort_values("Macro_F1", ascending=False).iloc[0]["Model"]
best_model = {"Decision Tree (tuned)": dt_search.best_estimator_, "Random Forest (tuned)": rf_search.best_estimator_}.get(best_name) or models[best_name]
pred = preds[best_name]
print("\nBest model:", best_name)
rep = classification_report(yte, pred, labels=order, output_dict=True, zero_division=0)
print(classification_report(yte, pred, labels=order, zero_division=0))
pd.DataFrame(rep).T.round(3).to_csv(f"{RES}/task3_classification_report.csv")

# extra: why we merged the ratings - the same tuned forest on the 14 raw labels
raw = df["rating"]
Xr = pd.concat([pd.DataFrame({"is_tv_show": (df.type == "TV Show").astype(int), "duration_value": df.duration_value, "release_year": df.release_year,
                              "year_added": df.year_added, "n_genres": df.n_genres, "has_director": df.has_director}), country_matrix(df, 15), genre_matrix(df).add_prefix("genre_")], axis=1)
a, b, c, e = train_test_split(Xr, raw, test_size=.2, stratify=raw.where(raw.map(raw.value_counts()) > 5, "other"), random_state=SEED)
raw_acc = accuracy_score(e, clone(rf_search.best_estimator_).fit(a, c).predict(b))
print("Same forest on the 14 raw labels: accuracy", round(raw_acc, 4))

fig, ax = plt.subplots(1, 2, figsize=(12, 4.8))
ConfusionMatrixDisplay(confusion_matrix(yte, pred, labels=order), display_labels=order).plot(ax=ax[0], cmap="Blues", colorbar=False, xticks_rotation=20)
ax[0].set_title(f"Confusion matrix - {best_name}")
imp = pd.Series(best_model.feature_importances_, index=X.columns).sort_values(ascending=False).head(12)[::-1]
ax[1].barh(imp.index, imp.values, color="#1f77b4"); ax[1].set_title("Top 12 features")
plt.tight_layout(); plt.savefig(f"{FIG}/task3_confusion_importance.png", dpi=150); plt.close()

fig, ax = plt.subplots(figsize=(9, 4.2)); r = res.set_index("Model")
x_ = np.arange(len(r)); ax.bar(x_ - .2, r["Accuracy"], .4, label="Accuracy", color="#1f77b4"); ax.bar(x_ + .2, r["Macro_F1"], .4, label="Macro F1", color="#ff7f0e")
ax.set_xticks(x_); ax.set_xticklabels([m.replace(" (", "\n(") for m in r.index], fontsize=8); ax.set_ylim(0, 1); ax.legend(); ax.set_title("Task 3: model comparison")
plt.tight_layout(); plt.savefig(f"{FIG}/task3_model_comparison.png", dpi=150); plt.close()

save_json("task3_results.json", {"n_dropped": len(df) - len(d), "group_counts": d["rating_group"].value_counts().to_dict(), "raw_counts": raw_counts.to_dict(),
    "best_model": best_name, "table": res.to_dict("records"), "best_params_dt": dt_search.best_params_, "best_params_rf": rf_search.best_params_,
    "cv_f1_dt": dt_search.best_score_, "cv_f1_rf": rf_search.best_score_, "report": rep, "confusion": confusion_matrix(yte, pred, labels=order).tolist(),
    "top_features": imp[::-1].round(4).to_dict(), "raw14_accuracy": raw_acc, "n_features": X.shape[1]})
