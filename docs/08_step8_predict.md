# 08 — `step8_predict.py`

**Owner: Thailong.**

## What this file does

It turns one student's 18 answers into **2 suggested tracks with percentages** and **3 reasons for each**.

## Function by function

- **`load_artifacts()`** loads the saved model, a sample of 50 past students, and the model info. If the model has
  not been trained it raises an error that tells you to run `python -m pipeline.step5_train`.
- **`predict_top2(model, answers)`** asks the model for the probability of each track and returns the best two,
  highest first, for example `[("Networking", 0.88), ("Data Science / AI", 0.08)]`.
- **`explain_prediction(model, background, answers, track)`** finds the 3 answers that pushed hardest toward
  that track and writes them as sentences, for example *You rated "servers, cloud computing and IT infrastructure"
  5 out of 5.*

## SHAP in plain words

SHAP answers: "for this student, how much did each answer push the result toward this track, compared with an
average student?" It tries the student's answers with some of them swapped for other students' answers and sees
how the probability moves. We explain the **18 questions** the student saw, not the 43 internal columns.

A technical detail: SHAP cannot do arithmetic on text, so scenario answers are turned into option numbers for SHAP
and turned back into text before the model sees them (`_encode` and `_decode`). We tested the explainer first:
the contributions add up exactly to the model's probability.

## Five likely questions

1. **What is SHAP and why did you choose it?** A method that shares out a prediction among the inputs. It was in our
   approved proposal and works for any model, so the explanation does not depend on which model wins.
2. **Do the reasons prove why the student fits the track?** No. They show which answers the *model* relied on. They are
   not causes.
3. **What do the percentages mean?** The model's confidence, based on the pattern of AUPP students in our survey. They
   are not guarantees, and the page says it is a suggestion, not a decision.
4. **Why can a reason be a low rating, like "software 2 out of 5" for Networking?** A low software interest does push
   toward Networking, so the model used it. For ratings of 2 or less the sentence says "only" ("You rated … only 2 out of 5")
   so it reads as a reason and not as a contradiction.
5. **Is the explanation the same every time?** Yes: the random seed is fixed (`seed=42`).

## Your one change

Add a "What does not fit" line: besides the 3 answers that pushed *toward* a track, show the one answer that pushed hardest
*against* it (the most negative SHAP value). Add it in `explain_prediction` (for example as an optional argument so the
existing calls keep working), show it on `app/templates/result.html`, add a test in `tests/test_step8.py`, and be ready to
explain why a student can still be a good fit even with one answer pointing the other way.
