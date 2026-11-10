"""Step 6: evaluate. The final exam on the hidden test students, plus the ablation experiments.

Owner: Sothon. Run it with:  python -m pipeline.step6_evaluate   (after step 5)
"""
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    top_k_accuracy_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate

from pipeline import config
from pipeline.step1_preprocessing import (
    clean_responses,
    load_responses,
    map_to_tracks,
    split_train_test,
)
from pipeline.step3_models import build_candidates

N_BOOTSTRAP = 1000


def _score(model, X, y):
    """Accuracy, macro F1, top-2 accuracy and a 95% bootstrap interval for accuracy."""
    predicted = model.predict(X)
    correct = (predicted == y).to_numpy()
    # bootstrap: re-draw the students with replacement 1000 times and see how accuracy moves
    rng = np.random.default_rng(config.SEED)
    draws = rng.integers(0, len(correct), size=(N_BOOTSTRAP, len(correct)))
    low, high = np.percentile(correct[draws].mean(axis=1), [2.5, 97.5])
    return {
        "accuracy": accuracy_score(y, predicted),
        "macro_f1": f1_score(y, predicted, average="macro", zero_division=0),
        "top2": top_k_accuracy_score(y, model.predict_proba(X), k=config.TOP_K, labels=config.TRACKS),
        "accuracy_ci_low": low,
        "accuracy_ci_high": high,
    }


def evaluate_on_test(model, X_test, y_test, baseline_model=None):
    """Score the model once on the hidden test students.

    Returns a dict with:
      "metrics"   one row per model ("model", and "baseline" if given)
      "per_track" precision, recall, F1 and number of students for each track
      "confusion" rows = real track, columns = predicted track
    """
    rows = {"model": _score(model, X_test, y_test)}
    if baseline_model is not None:
        rows["baseline"] = _score(baseline_model, X_test, y_test)

    predicted = model.predict(X_test)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_test, predicted, labels=config.TRACKS, zero_division=0
    )
    per_track = pd.DataFrame(
        {"precision": precision, "recall": recall, "f1": f1, "support": support},
        index=config.TRACKS,
    )
    confusion = pd.DataFrame(
        confusion_matrix(y_test, predicted, labels=config.TRACKS),
        index=config.TRACKS,
        columns=config.TRACKS,
    )
    return {
        "metrics": pd.DataFrame.from_dict(rows, orient="index")[
            ["accuracy", "macro_f1", "top2", "accuracy_ci_low", "accuracy_ci_high"]
        ],
        "per_track": per_track,
        "confusion": confusion,
    }


def run_ablation(X_train, y_train, best_params):
    """Check which question groups help, using 5-fold cross-validation on the training students.

    Each of the 3 single models (with its tuned settings from best_params) is tried on
    4 feature sets: ratings only, ratings + scores, ratings + scenarios, and all.
    Returns 12 rows: model, feature_set, accuracy_mean, macro_f1_mean, macro_f1_std.
    """
    folds = StratifiedKFold(n_splits=config.N_FOLDS, shuffle=True, random_state=config.SEED)
    rows = []
    for name in ["Logistic Regression", "KNN", "Random Forest"]:
        for feature_set in config.FEATURE_SETS:
            model = build_candidates(feature_set)[name].set_params(**best_params[name])
            scores = cross_validate(
                model, X_train, y_train, cv=folds, scoring={"accuracy": "accuracy", "macro_f1": "f1_macro"}
            )
            rows.append({
                "model": name,
                "feature_set": feature_set,
                "accuracy_mean": float(scores["test_accuracy"].mean()),
                "macro_f1_mean": float(scores["test_macro_f1"].mean()),
                "macro_f1_std": float(scores["test_macro_f1"].std(ddof=1)),
            })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    data = map_to_tracks(clean_responses(load_responses(config.DATA_PATH)))
    X_train, X_test, y_train, y_test = split_train_test(data)

    winner = joblib.load(config.MODELS_DIR / "winner_train_only.joblib")
    baseline = build_candidates()["Baseline"].fit(X_train, y_train)
    final_exam = evaluate_on_test(winner, X_test, y_test, baseline_model=baseline)
    tuned_params = json.loads((config.RESULTS_DIR / "tuned_params.json").read_text())
    ablation = run_ablation(X_train, y_train, tuned_params)

    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    final_exam["metrics"].to_csv(config.RESULTS_DIR / "test_results.csv")
    final_exam["per_track"].to_csv(config.RESULTS_DIR / "per_track.csv")
    final_exam["confusion"].to_csv(config.RESULTS_DIR / "confusion_matrix.csv")
    ablation.to_csv(config.RESULTS_DIR / "ablation.csv", index=False)
    data["track"].value_counts().rename_axis("track").rename("students").to_csv(
        config.RESULTS_DIR / "class_balance.csv"
    )

    print("Final exam on the hidden test students:")
    print(final_exam["metrics"].round(3).to_string())
    print("\nPer track:")
    print(final_exam["per_track"].round(3).to_string())
    print("\nConfusion matrix (rows = real, columns = predicted):")
    print(final_exam["confusion"].to_string())
    print("\nAblation (5-fold CV on training students):")
    print(ablation.round(3).to_string(index=False))
