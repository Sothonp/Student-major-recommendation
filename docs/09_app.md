# 09 — `app/` (the web app)

**Owner: Sothon.** Run: `python app\app.py`, then open http://127.0.0.1:5000

## What this does

A student answers the 18 questions and sees the top 2 tracks with a percentage bar and 3 reasons each, plus the note
that it is a suggestion, not a decision. **Nothing is stored**: no database and no login.

## Function by function

- **`questionnaire()`** (page `/`) shows the form: 14 rating questions as buttons 1 to 5 and 4 multiple-choice
  scenario questions, with the same wording as the Google Form. The major question is not shown.
- **`validate_answers(form)`** checks the submitted form and returns the clean answers plus a list of messages. A rating
  must be exactly 1 to 5 (so `4.5`, `abc` and an empty answer are rejected); a scenario answer must be one of the listed
  options.
- **`recommend()`** (page `/recommend`) validates, loads the model the first time it is needed, calls `predict_top2` and
  `explain_prediction`, and shows the result page. If answers are wrong it shows the form again with the messages and the
  earlier answers kept. If the model is missing it shows a friendly page with status 503 that says to run step 5.

Files: `app/templates/` (`index.html`, `result.html`, `error.html`) and `app/static/style.css`.

## Five likely questions

1. **Why Flask?** It is a small Python web library; the whole app is one file plus three templates.
2. **Is student data stored?** No. Answers are used for one prediction and then forgotten, which also protects privacy.
3. **What if someone submits a wrong or missing answer?** `validate_answers` rejects it and the form is shown again with
   a message; the app never crashes on bad input. Tests cover `4.5`, `abc`, empty, `6`, `0`, missing and unknown answers.
4. **What if the model is not trained yet?** The result page explains to run `python -m pipeline.step5_train`.
5. **Which model does the app use?** The winner (Logistic Regression), refit on all 257 students.

## Your one change

Add a one-sentence description of each track under its name on the result page (for example what a Cybersecurity student
studies). Put the text in `config.py`, show it in `result.html`, and extend `tests/test_app.py`.
