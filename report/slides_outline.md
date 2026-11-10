# Slides outline (technical slides, not pitching)

**Time:** 15 minutes presentation + 5 minutes questions. **Every member presents** and is graded individually; the
professor may ask any member, and teammates may help answer. Keep one idea per slide, large figures, little text.

Sections required by the professor: Introduction, Objectives, Methods, Experiments, Results, Discussion/Conclusion, References.

| # | Slide | Content | Figure / table | Speaker | Time |
|---|---|---|---|---|---|
| 1 | Title | Project name, course, team, advisor | — | Sonika | 0:15 |
| 2 | **Introduction** | Cambodia needs IT specialists; students choose with little guidance; gap: no model recommends a major *within IT* | — | Sonika | 1:15 |
| 3 | **Objectives** | Main goal and the 5 objectives (survey, key questions, scores, compare 4 models, web app) | — | Sonika | 0:45 |
| 4 | **Methods: data** | Our Google Form: 18 questions + major; 320 responses → 257 usable; 4 tracks; stratified 205 / 52 split | `class_balance.png` | Sonika | 1:30 |
| 5 | **Methods: features** | 14 ratings, 26 scenario columns, 3 scores (formulas); standardizing inside the pipeline (no leakage) | formulas | Thailong | 1:15 |
| 6 | **Methods: models and tuning** | Baseline + LR, KNN, RF, soft voting; settings searched; grid search with 5-fold CV | grid table | Thailong | 1:30 |
| 7 | **Experiments: protocol** | Metrics (accuracy, macro F1, top-2); selection rule; final exam once; ablation | small diagram | Sothon | 1:15 |
| 8 | **Results: model comparison** | All four ≈ 0.91 macro F1 vs baseline 0.12; tied → Logistic Regression | `model_comparison.png` | Sothon | 1:15 |
| 9 | **Results: final exam** | 86.5% accuracy (interval 76.9–96.2%), 85.2% macro F1, 90.4% top-2; baseline 32.7% | `confusion_matrix.png`, `per_track_scores.png` | Sothon | 1:30 |
| 10 | **Results: what helps** | Ablation: scenarios +6 points; scores not clearly | `ablation.png` | Rachana | 1:15 |
| 11 | **Results: which questions matter** | Correlation heatmap and importance (q18, q15, q17 on top) | `correlation_heatmap.png`, `feature_importance.png` | Rachana | 1:15 |
| 12 | **Results: the web app** | Live demo or screenshot: answers → top 2 tracks + SHAP reasons | screenshot | Thailong / Sothon | 1:30 |
| 13 | **Discussion** | Models tie; scenario answers mirror majors; small test set; reasons are not causes | — | Rachana | 1:00 |
| 14 | **Conclusion and future work** | What we built; limitations; more universities, year of study, Khmer UI | — | Rachana | 1:00 |
| 15 | **References** | The 10 references of the report | — | — | — |

Total ≈ 15 minutes. Speaking time: Sonika ≈ 3:45, Thailong ≈ 3:30 (with the demo), Sothon ≈ 4:30, Rachana ≈ 4:30 — rebalance if you like, but
everyone must speak and be ready to answer questions about **any** slide (see `docs/11_qa_sheet.md`).

## Rehearsal checklist

- Run the full talk once with a timer; cut to 15 minutes.
- Each member explains their own slides **without reading**, then explains one slide that is not theirs.
- Practice the 5 minutes of questions: someone asks from `docs/11_qa_sheet.md`, the speaker answers, a teammate adds.
- Test the demo on the presentation laptop: `.venv\Scripts\python app\app.py`; have a screenshot ready in case it fails.
