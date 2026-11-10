"""Step 5: train. Compare all models, pick the winner, save everything the later steps need.

Owner: Sothon. Run it with:  python -m pipeline.step5_train
"""
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import make_scorer, top_k_accuracy_score
from sklearn.model_selection import StratifiedKFold, cross_validate

from pipeline import config
from pipeline.step1_preprocessing import (
    clean_responses,
    load_responses,
    map_to_tracks,
    split_train_test,
)
from pipeline.step3_models import build_candidates
from pipeline.step4_tuning import tune_candidates


def cross_validate_models(models, X_train, y_train):
    """Score every model with 5-fold cross-validation on the training students.

    Returns one row per model with the mean and standard deviation (across the 5 folds)
    of accuracy, macro F1 and top-2 accuracy. The standard deviation is the sample
    standard deviation (ddof=1).
    """
    folds = StratifiedKFold(n_splits=config.N_FOLDS, shuffle=True, random_state=config.SEED)
    scoring = {
        "accuracy": "accuracy",
        "macro_f1": "f1_macro",
        # top-2: right if the real track is one of the model's two best guesses;
        # listing the labels keeps it working even if a fold lacks a track
        "top2": make_scorer(
            top_k_accuracy_score,
            response_method="predict_proba",
            k=config.TOP_K,
            labels=config.TRACKS,
        ),
    }
    rows = {}
    for name, model in models.items():
        scores = cross_validate(model, X_train, y_train, cv=folds, scoring=scoring)
        row = {}
        for metric in scoring:
            values = np.asarray(scores[f"test_{metric}"])
            row[f"{metric}_mean"] = float(values.mean())
            row[f"{metric}_std"] = float(values.std(ddof=1))
        rows[name] = row
    table = pd.DataFrame.from_dict(rows, orient="index")
    table.index.name = "model"
    return table


def _tie_group(cv_table):
    """Find the best model and every model that is tied with it.

    Returns (best, cutoff, tied): the model with the highest mean macro F1, the score a
    model must reach to count as tied (best score minus the best model's standard
    deviation), and the tied models listed simplest first.
    """
    real = cv_table.drop(index="Baseline", errors="ignore")
    unknown = [name for name in real.index if name not in config.MODEL_SIMPLICITY]
    if unknown:
        raise ValueError(
            f"Model(s) {unknown} are missing from MODEL_SIMPLICITY in pipeline/config.py. "
            "Add each new model there (simplest first) so the tie rule can rank it."
        )
    best = real["macro_f1_mean"].idxmax()
    cutoff = real.loc[best, "macro_f1_mean"] - real.loc[best, "macro_f1_std"]
    within = set(real.index[real["macro_f1_mean"] >= cutoff])
    tied = [name for name in config.MODEL_SIMPLICITY if name in within]
    return best, cutoff, tied


def select_best_model(cv_table):
    """Pick the winner by macro F1, preferring the simpler model when scores are close.

    1. Ignore the baseline.
    2. The best model has the highest mean macro F1.
    3. Any model within one standard deviation of the best (using the best model's
       standard deviation) counts as tied with it.
    4. Among the tied models, take the simplest:
       Logistic Regression < KNN < Random Forest < Soft Voting.
    """
    return _tie_group(cv_table)[2][0]


def main():
    """Run steps 1-4, compare the models, choose the winner and save the results."""
    data = map_to_tracks(clean_responses(load_responses(config.DATA_PATH), verbose=True))
    X_train, X_test, y_train, y_test = split_train_test(data)

    candidates = build_candidates()
    baseline = candidates.pop("Baseline").fit(X_train, y_train)   # the baseline has no settings
    tuned = tune_candidates(candidates, X_train, y_train)

    models = {"Baseline": baseline}
    models.update({name: result["estimator"] for name, result in tuned.items()})
    cv_table = cross_validate_models(models, X_train, y_train)
    winner = select_best_model(cv_table)

    # the app uses the winner refitted on ALL students (train + test), after evaluation is done
    X_all = pd.concat([X_train, X_test])
    y_all = pd.concat([y_train, y_test])
    final_model = clone(models[winner]).fit(X_all, y_all)
    # a sample of past students: SHAP uses it as "the average student"
    background = X_train.sample(n=min(50, len(X_train)), random_state=config.SEED)

    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    tuned_params = {name: result["best_params"] for name, result in tuned.items()}

    cv_table.to_csv(config.RESULTS_DIR / "cv_results.csv")
    (config.RESULTS_DIR / "tuned_params.json").write_text(json.dumps(tuned_params, indent=2))
    joblib.dump(models[winner], config.MODELS_DIR / "winner_train_only.joblib")
    joblib.dump(final_model, config.MODELS_DIR / "best_model.joblib")
    joblib.dump(background, config.MODELS_DIR / "shap_background.joblib")
    info = {
        "winner": winner,
        "n_train": len(X_train),
        "n_total": len(X_all),
        "cv": cv_table.loc[winner].to_dict(),
        "tuned_params": tuned_params,
    }
    (config.MODELS_DIR / "model_info.json").write_text(json.dumps(info, indent=2))

    best, cutoff, tied = _tie_group(cv_table)
    print(cv_table.round(3).to_string())
    print(
        f"\nBest average macro F1: {best} "
        f"({cv_table.loc[best, 'macro_f1_mean']:.3f} +/- {cv_table.loc[best, 'macro_f1_std']:.3f})."
    )
    print(f"Cut-off for 'tied': {cutoff:.3f} (the best score minus its standard deviation).")
    print(f"Tied models (simplest first): {', '.join(tied)}.")
    print(f"Winner: {winner}  (trained on {len(X_train)} students, app model refit on {len(X_all)})")


if __name__ == "__main__":
    main()
