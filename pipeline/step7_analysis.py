"""Step 7: analysis. Which questions separate the tracks, and the figures for the report.

Owner: Rachana. Run it with:  python -m pipeline.step7_analysis   (after steps 5 and 6)
"""
import matplotlib

matplotlib.use("Agg")  # draw to files, no window needed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from pipeline import config
from pipeline.step1_preprocessing import (
    clean_responses,
    load_responses,
    map_to_tracks,
    split_train_test,
)
from pipeline.step2_features import SCORE_COLS, add_interest_scores, build_preprocessor


def analyze_feature_signal(X_train, y_train):
    """Find which questions separate the four tracks (training students only).

    Returns a dict with two tables:
      "correlation"  17 rows (q1-q14 and the 3 scores) x 4 tracks. Each number is the
                     correlation between the answer and "this student is in that track"
                     (+1 = high answer goes with this track, -1 = the opposite, 0 = no link).
      "importance"   18 rows (q1-q18), sorted from most to least important, adding up to 1.
                     A Random Forest looks at all 18 questions; the one-hot columns of a
                     scenario question are added together so each question gets one number.
    """
    features = add_interest_scores(X_train)[config.RATING_COLS + SCORE_COLS].astype(float)
    correlation = pd.DataFrame(
        {track: features.corrwith((y_train == track).astype(float)) for track in config.TRACKS}
    )
    correlation.index.name = "feature"

    prep = build_preprocessor("ratings+scenarios")
    columns = prep.fit_transform(X_train)
    names = prep.named_steps["columns"].get_feature_names_out()   # e.g. "ratings__q4", "scenarios__q15_..."
    forest = RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=config.SEED)
    forest.fit(columns, y_train)
    question_of = [name.split("__")[1].split("_")[0] for name in names]
    importance = (
        pd.Series(forest.feature_importances_, index=question_of)
        .groupby(level=0).sum()
        .sort_values(ascending=False)
        .rename("importance")
        .to_frame()
    )
    importance.index.name = "question"
    return {"correlation": correlation, "importance": importance}


def plot_results():
    """Draw the seven report figures from the saved results files and return their paths.

    Reads (from config.RESULTS_DIR): class_balance, cv_results, confusion_matrix,
    correlation, feature_importance, ablation and per_track CSV files.
    Saves PNGs in results/figures/.
    """
    results = config.RESULTS_DIR
    figures = results / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    paths = []

    # 1. how many students in each track
    balance = pd.read_csv(results / "class_balance.csv")
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(balance["track"], balance["students"], color="#4C72B0")
    ax.set_ylabel("Students")
    ax.set_title("Students per track (257 usable survey responses)")
    plt.setp(ax.get_xticklabels(), rotation=15, ha="right")
    paths.append(figures / "class_balance.png")
    fig.savefig(paths[-1], dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 2. model comparison: average macro F1 over the 5 folds, with the spread as error bars
    cv = pd.read_csv(results / "cv_results.csv", index_col=0)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(cv.index, cv["macro_f1_mean"], yerr=cv["macro_f1_std"], capsize=4, color="#55A868")
    ax.set_ylabel("Macro F1 (5-fold cross-validation)")
    ax.set_ylim(0, 1)
    ax.set_title("Model comparison on the training students")
    plt.setp(ax.get_xticklabels(), rotation=15, ha="right")
    paths.append(figures / "model_comparison.png")
    fig.savefig(paths[-1], dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 3. confusion matrix: rows = real track, columns = predicted track
    confusion = pd.read_csv(results / "confusion_matrix.csv", index_col=0)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.imshow(confusion.to_numpy(), cmap="Blues")
    ax.set_xticks(range(len(confusion.columns)), confusion.columns, rotation=30, ha="right")
    ax.set_yticks(range(len(confusion.index)), confusion.index)
    for row in range(confusion.shape[0]):
        for col in range(confusion.shape[1]):
            ax.text(col, row, int(confusion.iat[row, col]), ha="center", va="center")
    ax.set_xlabel("Predicted track")
    ax.set_ylabel("Real track")
    ax.set_title("Confusion matrix on the hidden test students")
    paths.append(figures / "confusion_matrix.png")
    fig.savefig(paths[-1], dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 4. correlation heatmap: answers vs tracks
    correlation = pd.read_csv(results / "correlation.csv", index_col=0)
    fig, ax = plt.subplots(figsize=(6, 7))
    image = ax.imshow(correlation.to_numpy(), cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(correlation.columns)), correlation.columns, rotation=30, ha="right")
    ax.set_yticks(range(len(correlation.index)), correlation.index)
    fig.colorbar(image, ax=ax, label="Correlation")
    ax.set_title("Which answers go with which track")
    paths.append(figures / "correlation_heatmap.png")
    fig.savefig(paths[-1], dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 5. which questions the Random Forest relies on most
    importance = pd.read_csv(results / "feature_importance.csv", index_col=0)["importance"]
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.barh(importance.index[::-1], importance.to_numpy()[::-1], color="#C44E52")
    ax.set_xlabel("Importance (adds up to 1)")
    ax.set_title("Random Forest: importance of each question")
    paths.append(figures / "feature_importance.png")
    fig.savefig(paths[-1], dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 6. ablation: macro F1 for each model with each group of questions
    ablation = pd.read_csv(results / "ablation.csv")
    table = ablation.pivot(index="feature_set", columns="model", values="macro_f1_mean")
    table = table.reindex(list(config.FEATURE_SETS))
    fig, ax = plt.subplots(figsize=(8, 4.5))
    width = 0.8 / len(table.columns)
    for i, model in enumerate(table.columns):
        ax.bar(np.arange(len(table)) + i * width, table[model], width, label=model)
    ax.set_xticks(np.arange(len(table)) + width, table.index)
    ax.set_ylabel("Macro F1 (5-fold cross-validation)")
    ax.set_ylim(0, 1.15)   # empty space above the bars, so the legend covers nothing
    ax.set_title("Ablation: which groups of questions help")
    ax.legend(loc="upper center", ncol=3)
    paths.append(figures / "ablation.png")
    fig.savefig(paths[-1], dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 7. final exam, track by track: precision, recall and F1 for each of the 4 tracks
    per_track = pd.read_csv(results / "per_track.csv", index_col=0)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    width = 0.25
    for i, (column, name) in enumerate([("precision", "Precision"), ("recall", "Recall"), ("f1", "F1")]):
        ax.bar(np.arange(len(per_track)) + i * width, per_track[column], width, label=name)
    ax.set_xticks(np.arange(len(per_track)) + width, per_track.index, rotation=15, ha="right")
    ax.set_ylabel("Score on the hidden test students")
    ax.set_ylim(0, 1.15)   # room above the bars for the legend
    ax.set_title("Final exam by track: precision, recall and F1")
    ax.legend(loc="upper center", ncol=3)
    paths.append(figures / "per_track_scores.png")
    fig.savefig(paths[-1], dpi=150, bbox_inches="tight")
    plt.close(fig)

    return paths


if __name__ == "__main__":
    data = map_to_tracks(clean_responses(load_responses(config.DATA_PATH)))
    X_train, X_test, y_train, y_test = split_train_test(data)

    analysis = analyze_feature_signal(X_train, y_train)
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    analysis["correlation"].to_csv(config.RESULTS_DIR / "correlation.csv")
    analysis["importance"].to_csv(config.RESULTS_DIR / "feature_importance.csv")
    print("Most important questions:")
    print(analysis["importance"].head(8).round(3).to_string())

    for path in plot_results():
        print("saved", path)
