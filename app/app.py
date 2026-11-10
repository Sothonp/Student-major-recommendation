"""Step 9: the web app. A student answers 18 questions and sees 2 suggested IT tracks.

Owner: Sothon. Run it with:  python app/app.py   then open http://127.0.0.1:5000
Nothing a student types is stored. The model must be trained first (step 5).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # so `python app/app.py` finds `pipeline`

from flask import Flask, render_template, request

from pipeline import config
from pipeline.step8_predict import explain_prediction, load_artifacts, predict_top2

app = Flask(__name__)
_loaded = {}   # model files already read, keyed by the models folder


def _artifacts():
    """Load the model on first use. Until step 5 has run this raises, and nothing is cached."""
    key = str(config.MODELS_DIR)
    if key not in _loaded:
        _loaded[key] = load_artifacts()
    return _loaded[key]


def _form_page(answers, errors):
    """The questionnaire page, keeping earlier answers and showing any error messages."""
    def question(q):
        label = config.RATING_LABELS[q]
        return {"id": q, "n": q[1:], "text": label[0].upper() + label[1:]}

    return render_template(
        "index.html",
        interests=[question(q) for q in config.RATING_COLS[:8]],
        enjoyment=[question(q) for q in config.RATING_COLS[8:]],
        scenarios=[
            {"id": q, "n": q[1:], "text": config.SCENARIO_QUESTIONS[q], "options": config.SCENARIO_OPTIONS[q]}
            for q in config.SCENARIO_COLS
        ],
        answers=answers,
        errors=errors,
    )


def validate_answers(form):
    """Check the submitted form. Returns (clean_answers, error_messages).

    Ratings must be exactly the text "1" to "5" (so "4.5", "abc" and "" are rejected);
    scenario answers must be one of the listed options.
    """
    answers, errors = {}, []
    for q in config.RATING_COLS:
        value = form.get(q, "")
        if value in ("1", "2", "3", "4", "5"):
            answers[q] = int(value)
        else:
            errors.append(f"Question {q[1:]} needs a rating from 1 to 5.")
    for q in config.SCENARIO_COLS:
        value = form.get(q, "")
        if value in config.SCENARIO_OPTIONS[q]:
            answers[q] = value
        else:
            errors.append(f"Question {q[1:]} needs one of the listed answers.")
    return answers, errors


@app.get("/")
def questionnaire():
    """Show the 18 questions."""
    return _form_page({}, [])


@app.post("/recommend")
def recommend():
    """Check the answers, then show the top 2 tracks with their reasons."""
    answers, errors = validate_answers(request.form)
    if errors:
        return _form_page(request.form, errors)

    try:
        model, background, _ = _artifacts()
    except FileNotFoundError as problem:
        return render_template("error.html", message=str(problem)), 503

    results = [
        {
            "track": track,
            "percent": round(probability * 100),
            "reasons": explain_prediction(model, background, answers, track),
        }
        for track, probability in predict_top2(model, answers)
    ]
    return render_template("result.html", results=results)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000)
