# 04 — `step3_models.py` and `step4_tuning.py`

**Owners: Sothon (step 3) and Thailong (step 4).**

## What these files do

Step 3 creates the models. Step 4 finds each model's best settings and builds the voting team.

## The models in plain words

- **Baseline** always answers the most common track. It is the floor: a real model must beat it.
- **Logistic Regression** gives each answer a weight toward each track and adds them up.
- **KNN** (K nearest neighbors) finds the K most similar students and lets them vote.
- **Random Forest** is many yes/no decision trees that vote.
- **Soft voting** averages the probabilities of the three tuned models above.

Logistic Regression and Random Forest use `class_weight="balanced"` so the small Cybersecurity
group is not ignored. KNN has no such option.

## Function by function

- **`build_candidates(feature_set="all")`** (step 3) returns the baseline and the 3 untuned
  models, each wrapped in a pipeline with the step-2 preprocessing.
- **`tune_candidates(candidates, X_train, y_train)`** (step 4) does the tuning. For each model it
  tries every combination of its settings ("grid search"), scores each with 5-fold cross-
  validation on the 205 practice students, and keeps the combination with the best macro F1. Then it
  builds the voting team from the 3 tuned models.

## What was tuned and what won (real run)

| Model | Settings tried | Best found |
|---|---|---|
| Logistic Regression | C: 0.01, 0.1, 1, 10 | C = 0.1 |
| KNN | neighbors 3, 5, 7, 9, 11, 15 × weights uniform/distance | 5 neighbors, uniform |
| Random Forest | trees 100/300 × depth none/5/10 × min leaf 1/2/4 | 100 trees, no depth limit, leaf 1 |

## Five likely questions

1. **What is tuning?** Models have settings (like oven temperature). The model learns from data, but
   *we* choose the settings. Tuning tries many and keeps the best.
2. **How does grid search work?** For each combination it runs 5-fold cross-validation: train on 4
   groups of students, test on the 5th, rotate 5 times, average. The best average wins.
3. **Why tune only on the 205 practice students?** The 52 hidden students are the final exam. Picking
   settings after seeing them would inflate the result.
4. **Why macro F1 for tuning?** It gives each track equal weight, so the model cannot win by ignoring
   Cybersecurity (see `10_metrics_explained.md`).
5. **What does soft voting mean?** Each of the three models gives probabilities for the 4 tracks. We
   average them and pick the highest. "Hard" voting would only count each model's single choice.

## Your one change

**Thailong:** add a value to one grid in `config.PARAM_GRIDS` (for example `C: 0.3` or `n_neighbors: 13`),
re-run `step5_train`, and check whether the best setting changes. Update the grid-size test in
`tests/test_config.py`. **Sothon:** add one more model to `build_candidates` (for example a
`DecisionTreeClassifier`) with a small grid in `config.PARAM_GRIDS`, update `tests/test_step3.py`, and explain
whether it is worth keeping.
