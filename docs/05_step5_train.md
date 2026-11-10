# 05 — `step5_train.py`

**Owner: Sothon.** Run: `python -m pipeline.step5_train`

## What this file does

It runs steps 1 to 4, compares all models fairly, picks the winner, and saves everything the
later steps and the app need.

## Function by function

- **`cross_validate_models(models, X_train, y_train)`** scores each model with 5-fold cross-validation
  on the 205 practice students. For every model it reports the average and the spread (standard
  deviation) of accuracy, macro F1 and top-2 accuracy.
- **`select_best_model(cv_table)`** picks the winner:
  1. ignore the baseline;
  2. find the model with the highest average macro F1;
  3. any model within one standard deviation of it counts as **tied**;
  4. among the tied models choose the simplest: Logistic Regression < KNN < Random Forest < Soft Voting.
- **`main()`** runs everything and saves: `results/cv_results.csv`, `results/tuned_params.json`,
  `models/winner_train_only.joblib` (for the final exam), `models/best_model.joblib` (refit on all
  257 students, used by the app), `models/shap_background.joblib` and `models/model_info.json`.

## The real run

| Model | Accuracy | Macro F1 | Top-2 |
|---|---|---|---|
| Baseline | 0.322 | 0.122 | 0.620 |
| Logistic Regression | 0.907 | 0.910 ± 0.028 | 0.966 |
| KNN | 0.907 | 0.912 ± 0.038 | 0.966 |
| Random Forest | 0.902 | 0.905 ± 0.034 | 0.951 |
| Soft Voting | 0.902 | 0.905 ± 0.034 | 0.966 |

KNN has the highest average (0.912, spread 0.038), so any model above 0.874 is tied. All four are, so the
simplest, **Logistic Regression**, wins.

Two things to know about this table:
- **Random Forest and Soft Voting have identical accuracy and macro F1.** We checked: on all 205 practice students the
  voting team chooses the same track as Random Forest (Logistic Regression and KNN differ from it on only 1 student). Their
  top-2 scores differ, because their probabilities are not the same.
- **The baseline's top-2 (0.620) means "always suggest the two most common tracks"** (Data Science / AI and Networking).

## Five likely questions

1. **KNN scored highest. Why is Logistic Regression the winner?** The difference (0.912 vs 0.910) is far
   smaller than the fold-to-fold spread (±0.03 to 0.04), so it is noise. With a tie we prefer the simpler
   model: easier to explain, and less likely to be fitted to chance.
2. **Is Logistic Regression really better than the others?** No. They are statistically tied, and we say so.
3. **Why compare with cross-validation and not the test set?** If we chose using the test set it would
   no longer be a fair final exam.
4. **Why refit the winner on all 257 students for the app?** After the evaluation is finished, more
   data makes a slightly better model. The reported test scores come from the version trained on 205 only.
5. **What is the standard deviation here?** How much the score changes between the 5 rounds. A big spread
   means small score differences between models mean little.

## Your one change

Make `main()` print which models were tied and why the winner was chosen (for example "tied within 0.874:
LR, KNN, RF, SV"). Add a test for the new output and explain it in the presentation.
