"""Step 1: preprocessing. Load the survey, clean it, map majors to tracks, split.

Owner: Sonika. Everything later in the pipeline starts from what this file returns.
"""
import pandas as pd
from sklearn.model_selection import train_test_split

from pipeline import config


def load_responses(path):
    """Read the Google Form spreadsheet and name the columns q1..q19.

    The first column is the Timestamp, which carries no information after the
    sheet was re-imported, so it is dropped. Columns are renamed by position
    because the real headers are long sentences.
    """
    df = pd.read_excel(path)
    df.columns = ["timestamp"] + [f"q{i}" for i in range(1, 20)]
    return df.drop(columns="timestamp")


def clean_responses(df):
    """Keep only complete rows with valid answers.

    - text answers lose stray spaces
    - ratings must be the whole numbers 1 to 5 (4.0 is fine, 4.5 and 6 are not)
    - any row with a blank answer is dropped
    """
    out = df.copy()
    for col in config.SCENARIO_COLS + [config.LABEL_COL]:
        out[col] = out[col].astype("string").str.strip().replace("", pd.NA)
    for col in config.RATING_COLS:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    out = out.dropna(subset=config.QUESTION_COLS + [config.LABEL_COL])
    out = out[out[config.RATING_COLS].isin([1, 2, 3, 4, 5]).all(axis=1)]

    out[config.RATING_COLS] = out[config.RATING_COLS].astype("int64")
    for col in config.SCENARIO_COLS + [config.LABEL_COL]:
        out[col] = out[col].astype(str)
    return out


def map_to_tracks(df):
    """Add a `track` column and set aside CS and ICT students (they have no single track)."""
    out = df.copy()
    out["track"] = out[config.LABEL_COL].map(config.MAJOR_TO_TRACK)
    return out.dropna(subset=["track"])


def split_train_test(df):
    """Split into training and hidden test students, keeping the track proportions.

    Returns X_train, X_test, y_train, y_test. X holds only the 18 questions.
    """
    X = df[config.QUESTION_COLS]
    y = df["track"]
    return train_test_split(
        X, y, test_size=config.TEST_SIZE, stratify=y, random_state=config.SEED
    )
