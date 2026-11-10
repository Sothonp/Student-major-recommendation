# 03 — `step2_features.py`

**Owner: Thailong** (Sothon co-owns `build_preprocessor`, where data leakage matters).

## What this file does

It turns the 18 answers into the 43 numbers the models learn from. Every model uses the
same preprocessing, so the comparison is fair.

## Function by function

- **`add_interest_scores(X)`** adds 3 combined scores to each student:
  - `build_vs_protect` = average of q7 and q12 (building software) minus q4 (cybersecurity).
    High = likes building, low = likes protecting. It separates Software from Cybersecurity.
  - `data_score` = average of q3 and q11 (data and patterns). Points to Data Science / AI.
  - `infra_score` = average of q5, q6 and q13 (networks, servers, systems). Points to Networking.

  Example: q7 = 5, q12 = 3, q4 = 5 gives build_vs_protect = (5 + 3) ÷ 2 − 5 = **−1**.
- **`build_preprocessor(feature_set)`** builds the pipeline with three parts:
  1. the scores above are added;
  2. the 14 ratings and 3 scores are **standardized** (rescaled to average 0 and similar spread);
  3. the 4 scenario questions become **one-hot** columns: one 0/1 column per possible answer (26).

  14 + 3 + 26 = **43 features**. `feature_set` lets the ablation use fewer groups:
  `"ratings"` (14), `"ratings+scores"` (17), `"ratings+scenarios"` (40), `"all"` (43).

## Five likely questions

1. **Why combine answers into scores?** Some tracks are alike (Software vs Cybersecurity). A
   combined score gives the model a clearer signal. We tested it instead of assuming: see Q4.
2. **What is standardizing, and why?** KNN and Logistic Regression compare sizes of numbers, so
   features on different scales would count unequally. Standardizing puts them on the same scale.
3. **What is one-hot encoding?** A scenario answer is text. We make one column per option: 1 if
   the student chose it, 0 if not. An answer never seen in training is ignored, not a crash.
4. **Did the scores help?** Not clearly. Ratings only vs ratings + scores (macro F1): Logistic
   Regression 0.847 → 0.841, Random Forest 0.823 → 0.824, KNN 0.807 → 0.832. Only KNN gained, and
   the folds vary by ±0.05, so it is within noise. We report that honestly.
5. **What is data leakage and where do you prevent it?** The scaler learns each column's average.
   If it learned from the test students, information would leak. The scaler is inside the
   pipeline, so in every cross-validation round it learns only from that round's training
   students. `test_scaler_is_fitted_on_train_only` checks it.

## Your one change

Add a fourth score, for example `creative_score` = average of q8 and q14 (interfaces and
creativity). Add it to `SCORE_COLS` and `add_interest_scores`, then update the tests that count features: the widths in
`tests/test_step2.py` (43 → 44 and 17 → 18; the others stay the same) and the 17-row correlation table in `tests/test_step7.py`
(17 → 18 rows, and the expected index). Run `pytest`, then `step6_evaluate`, and see whether the new score helps.
