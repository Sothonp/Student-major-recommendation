"""Step 4: tuning. Find the best settings for each model, then build the voting team.

Owner: Thailong. "Tuning" = trying many combinations of a model's settings and keeping
the one with the best average macro F1 over 5-fold cross-validation (grid search).
Only the training students are used here; the hidden test students are never touched.
"""
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from pipeline import config


def tune_candidates(candidates, X_train, y_train, param_grids=None):
    """Tune each model, then build a soft-voting team from the 3 tuned models.

    candidates: dict of name -> untuned pipeline (the 3 single models, no baseline).
    param_grids: dict of name -> grid; None means config.PARAM_GRIDS (tests pass tiny grids).

    Returns a dict with the keys "Logistic Regression", "KNN", "Random Forest",
    "Soft Voting". Each value is {"estimator": fitted model, "best_params": dict,
    "cv_macro_f1": float}. The voting team has no settings of its own, so its
    best_params is {} and its cv_macro_f1 is None (step 5 scores it later).
    """
    grids = config.PARAM_GRIDS if param_grids is None else param_grids
    folds = StratifiedKFold(n_splits=config.N_FOLDS, shuffle=True, random_state=config.SEED)

    results = {}
    for name, pipeline in candidates.items():
        search = GridSearchCV(pipeline, grids[name], scoring="f1_macro", cv=folds)
        search.fit(X_train, y_train)   # refit=True: the best settings are refitted on all training rows
        results[name] = {
            "estimator": search.best_estimator_,
            "best_params": search.best_params_,
            "cv_macro_f1": float(search.best_score_),
        }

    # the team votes with probabilities; each member keeps its own tuned settings
    team = VotingClassifier(
        [(name, result["estimator"]) for name, result in results.items()], voting="soft"
    )
    team.fit(X_train, y_train)
    results["Soft Voting"] = {"estimator": team, "best_params": {}, "cv_macro_f1": None}
    return results
