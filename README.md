# Auspify ML Internship – Tasks 1–4 (Netflix dataset, Python)

Author: MUSABYIMANA Jean D'amour  |  GitHub: Damour-1-og

| Task | Script | Main result |
|---|---|---|
| 1. Content recommendation system | `code/task1_recommender.py` | TF-IDF + cosine similarity; top-10 genre overlap 0.90 (random 0.09) |
| 2. Movie vs TV Show prediction | `code/task2_type_prediction.py` | Random Forest 95.05% accuracy (baseline 69.7%) |
| 3. Audience rating classification | `code/task3_rating_classification.py` | Tuned Random Forest 64.5% accuracy, macro F1 0.64 (baseline 46.0%) |
| 4. Content segmentation | `code/task4_segmentation.py` | K-Means, 6 clusters (silhouette 0.17) |

Full report with explanations: **Auspify_ML_Report.docx**

## Run
```
pip install -r requirements.txt
./run_all.sh          # or: cd code && python task1_recommender.py   (same for task2, task3, task4)
```
Charts go to `figures/`, tables and numbers to `results/` (CSV/JSON + console output of each task).

## Notes
- Data leakage handled in Task 2: `duration` and genre names are excluded from the main model (see report).
- Task 3 merges 14 rating labels into 4 audience groups (Kids, Older Kids, Teens, Adults); NR/UR removed.
- Cluster names in Task 4 are an interpretation of the printed cluster profiles.

#Auspify #AuspifyTechnologies #AuspifyInternship #AuspifyProjects
