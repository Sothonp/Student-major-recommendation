# 07 — `step7_analysis.py`

**Owner: Rachana.** Run: `python -m pipeline.step7_analysis` (after steps 5 and 6)

## What this file does

It shows **which questions separate the four tracks** and draws the seven figures for the report.

## Function by function

- **`analyze_feature_signal(X_train, y_train)`** returns two tables, using the 205 practice students only:
  - **Correlation:** for each rating (q1 to q14) and each combined score, how strongly a high answer goes
    with being in each track. +1 = always together, −1 = opposite, 0 = no link.
  - **Importance:** a Random Forest looks at all 18 questions and reports how much it relied on each. The
    numbers add up to 1. The one-hot columns of a scenario question are added together so every question
    gets one number.
- **`plot_results()`** reads the saved result files and draws 7 PNG figures in `results/figures/`.

## What the real data shows

- Strongest links: `data_score` with Data Science / AI (**+0.73**), q3 with Data (+0.70), `infra_score` with
  Networking (+0.63), q6 with Networking (+0.60), q12 with Software (+0.60), q4 with Cybersecurity (+0.56).
  `build_vs_protect` is high for Software (+0.57) and negative for Cybersecurity (−0.47), which is exactly
  what it was designed to do.
- Most important questions: **q18 (0.17), q15 (0.14), q17 (0.10)**, then q5, q4, q3. The top three are the
  scenario questions.

## The seven figures

| File | Shows |
|---|---|
| `class_balance.png` | students per track (Cybersecurity is the smallest, 36) |
| `model_comparison.png` | average macro F1 of each model, with the spread as error bars |
| `confusion_matrix.png` | real track vs predicted track on the 52 test students |
| `correlation_heatmap.png` | which answers go with which track |
| `feature_importance.png` | which questions the Random Forest relies on |
| `ablation.png` | macro F1 when groups of questions are removed |
| `per_track_scores.png` | precision, recall and F1 of each track on the final exam |

## Five likely questions

1. **What is the difference between correlation and importance?** Correlation looks at one answer and
   one track at a time. Importance asks how much a model relies on a question when it uses all of them.
2. **Does correlation prove that interest causes the major?** No. It shows two things go together.
3. **Why are scenario questions the most important?** Each answer names an activity that matches a major
   (for example "Build a system that detects cyberattacks"). A student who already picked Cybersecurity
   will often choose it. That is useful, but also partly circular, which is why we also report the
   ratings-only result.
4. **Why is the data score so strongly linked to Data Science?** It averages q3 and q11, which ask about
   data directly, so students in that track rate them high.
5. **Why use the training students only?** To keep the test students unseen, as in every other step.

## Your change (already made for you by Sothon: read it, run its test, then explain it)

`plot_results` now draws a **seventh figure**, `per_track_scores.png`: for each track, three bars (precision, recall, F1) from the
final exam. It reads `results/per_track.csv`, and the code is the last block of `plot_results`: it loops over the three columns
and draws one set of bars for each, shifted sideways by one bar width.

**What the chart shows:** Software Development is the easiest track (recall 1.00: no student was missed). Data Science / AI has
perfect precision (1.00) but lower recall (0.82). Cybersecurity has the lowest precision (0.67), because three students from
other tracks were labeled Cybersecurity, and only 7 test students are Cybersecurity, so its bars are noisy.

**Test:** `test_plot_results_creates_seven_png_figures` in `tests/test_step7.py` (it now expects 7 files and writes a fake
`per_track.csv`).

**Your next step, if you want one of your own:** change the colors or add the number of students to each label, and commit it
under your own name.
