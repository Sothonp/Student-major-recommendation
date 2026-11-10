"""Step 8: predict. Turn one student's answers into 2 suggested tracks and the reasons why.

Owner: Thailong. The web app calls the three public functions here.

How the reasons work (SHAP): for each of the 18 questions, SHAP asks "how much did this
answer push the result toward this track, compared with an average student?". The 3 answers
that pushed the most become the reasons.

SHAP cannot do arithmetic on text, so (found by testing) scenario answers are turned into
option numbers for SHAP and turned back into text before the model sees them. The explainer
is the Permutation explainer, which works for any model; it takes about 0.1 second.
"""
import json

import joblib
import numpy as np
import pandas as pd
import shap

from pipeline import config

SHAP_EVALS = 300   # how many model calls SHAP may use per explanation (more = steadier)


def _to_frame(answers):
    """One student's answers (a dict q1..q18) as a one-row table in the model's column order."""
    frame = pd.DataFrame([answers])[config.QUESTION_COLS].copy()
    frame[config.RATING_COLS] = frame[config.RATING_COLS].astype(int)
    return frame


def _encode(frame):
    """Table of answers -> numbers (scenario text becomes its option number)."""
    numbers = frame[config.QUESTION_COLS].copy()
    for q in config.SCENARIO_COLS:
        numbers[q] = numbers[q].map(config.SCENARIO_OPTIONS[q].index)
    return numbers.to_numpy(dtype=float)


def _decode(numbers):
    """Numbers -> table of answers (the reverse of _encode)."""
    frame = pd.DataFrame(np.asarray(numbers), columns=config.QUESTION_COLS)
    frame[config.RATING_COLS] = frame[config.RATING_COLS].round().astype(int)
    for q in config.SCENARIO_COLS:
        options = config.SCENARIO_OPTIONS[q]
        frame[q] = [options[int(i)] for i in frame[q].round()]
    return frame


def load_artifacts():
    """Load the saved model, the sample of past students for SHAP, and the model info.

    Raises FileNotFoundError (naming step 5) if the model has not been trained yet.
    """
    files = {
        "model": config.MODELS_DIR / "best_model.joblib",
        "background": config.MODELS_DIR / "shap_background.joblib",
        "info": config.MODELS_DIR / "model_info.json",
    }
    missing = [path.name for path in files.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing {', '.join(missing)} in {config.MODELS_DIR}. "
            "Train the model first: python -m pipeline.step5_train"
        )
    model = joblib.load(files["model"])
    background = joblib.load(files["background"])
    info = json.loads(files["info"].read_text())
    return model, background, info


def predict_top2(model, answers):
    """Return the two most likely tracks as [(track, probability), (track, probability)].

    answers is a dict with the keys q1..q18 (ratings as whole numbers 1-5, scenarios as text).
    The highest probability comes first.
    """
    probabilities = model.predict_proba(_to_frame(answers))[0]
    best_two = np.argsort(-probabilities)[:2]
    return [(str(model.classes_[i]), float(probabilities[i])) for i in best_two]


def explain_prediction(model, background, answers, track, n_reasons=3):
    """Return the answers that pushed hardest toward `track`, as plain sentences.

    Example: 'You rated "computer networks, network communication and connectivity" 5 out of 5.'
    """
    k = list(model.classes_).index(track)
    explainer = shap.explainers.Permutation(
        lambda numbers: model.predict_proba(_decode(numbers)),
        shap.maskers.Independent(_encode(background), max_samples=50),
        max_evals=SHAP_EVALS,
        seed=config.SEED,
    )
    pushes = explainer(_encode(_to_frame(answers))).values[0][:, k]   # 18 numbers, one per question
    strongest = np.argsort(-pushes)[:n_reasons]

    reasons = []
    for i in strongest:
        q = config.QUESTION_COLS[i]
        if q in config.RATING_COLS:
            reasons.append(f'You rated "{config.RATING_LABELS[q]}" {answers[q]} out of 5.')
        else:
            reasons.append(f'You chose "{answers[q]}" for {config.SCENARIO_LABELS[q]}.')
    return reasons
