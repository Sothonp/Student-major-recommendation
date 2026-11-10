# 06 — `step6_evaluate.py`

**Owner: Sothon.** Run: `python -m pipeline.step6_evaluate` (after step 5)

## What this file does

It is the final exam: the winner (trained on the 205 practice students) predicts the 52 hidden students. It
also runs the ablation experiments.

## Function by function

- **`evaluate_on_test(model, X_test, y_test, baseline_model=None)`** returns the exam results: accuracy,
  macro F1, top-2 accuracy, a 95% interval for accuracy, precision and recall for each track, and the
  confusion matrix. The interval comes from the **bootstrap**: re-draw the 52 students with replacement
  1000 times and see how far accuracy moves.
- **`run_ablation(X_train, y_train, best_params)`** removes groups of questions to see what each adds,
  for 3 models × 4 feature sets, using 5-fold cross-validation on the practice students.

## Results (real data, 52 test students)

| | Accuracy | Macro F1 | Top-2 |
|---|---|---|---|
| Logistic Regression | **86.5%** (95% interval 76.9%–96.2%) | **85.2%** | **90.4%** |
| Baseline | 32.7% | 12.3% | 61.5% |

| Track | Precision | Recall | Students |
|---|---|---|---|
| Cybersecurity | 0.67 | 0.86 | 7 |
| Data Science / AI | 1.00 | 0.82 | 17 |
| Networking | 0.86 | 0.80 | 15 |
| Software Development | 0.87 | 1.00 | 13 |

Mistakes: 1 Cybersecurity student → Software; 1 Data → Cybersecurity, 2 Data → Networking; 2 Networking →
Cybersecurity, 1 Networking → Software. Software was never missed.

**Ablation (macro F1, 5-fold CV):** Logistic Regression: ratings 0.847, + scores 0.841, + scenarios 0.910,
all 0.910. KNN: 0.807, 0.832, 0.902, 0.912. Random Forest: 0.823, 0.824, 0.900, 0.905.

## Five likely questions

1. **Why is the test accuracy (86.5%) lower than cross-validation (90.7%)?** The test set is only 52
   students, so the score is noisy: one student is about 2 points. 90.7% falls inside the 95% interval
   (76.9%–96.2%), so the two numbers agree.
2. **What is the confidence interval?** The range of accuracy we would plausibly see with different students
   of the same kind. It is wide because the test set is small.
3. **Which track is hardest?** Cybersecurity. Precision is 0.67 because 2 Networking and 1 Data student were
   labeled Cybersecurity. Only 7 test students are Cybersecurity, so this number is also noisy.
4. **Do the scenario questions help?** Yes, about +6 points (0.847 → 0.910 for Logistic Regression). Caution:
   the scenario options closely mirror the major names (for example "Build and manage cloud/network
   infrastructure"), so part of that gain is students restating their major. The ratings-only result
   (≈ 0.85) is the stricter number.
5. **Did the engineered scores help?** Not clearly (see `03_step2_features.md`). The gains are within the
   spread of the folds.

## Your one change

Add a 95% bootstrap interval for **macro F1** next to the one for accuracy (extend `_score`), update the
tests in `tests/test_step6.py`, and say what the interval tells us about the small Cybersecurity group.
