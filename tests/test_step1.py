import math

import numpy as np
import pandas as pd

from pipeline import config
from pipeline.step1_preprocessing import (
    clean_responses,
    load_responses,
    map_to_tracks,
    split_train_test,
)


def test_load_responses_renames_and_drops_timestamp(tmp_path, fake_raw):
    # a path with spaces and parentheses, like the real file
    path = tmp_path / "Student Major Recommendation (Responses).xlsx"
    sheet = fake_raw.copy()
    sheet.columns = [f"Question {i} with a long header" for i in range(1, 20)]
    sheet.insert(0, "Timestamp", 46336.6)
    sheet.to_excel(path, index=False)

    df = load_responses(path)

    assert list(df.columns) == [f"q{i}" for i in range(1, 20)]
    assert len(df) == len(fake_raw)
    assert df["q1"].tolist() == fake_raw["q1"].tolist()


def test_clean_responses_handles_quirks(fake_raw):
    df = fake_raw.head(10).copy()
    df[config.RATING_COLS] = df[config.RATING_COLS].astype(float)  # Excel stores ratings as floats
    df.loc[0, "q1"] = 4.0            # float rating -> kept as whole number 4
    df.loc[1, "q2"] = 6              # out of range -> dropped
    df.loc[2, "q3"] = 4.5            # not a whole number -> dropped
    df.loc[3, "q4"] = np.nan         # blank rating -> dropped
    df.loc[5, "q15"] = None          # blank scenario answer -> dropped
    df.loc[4, "q19"] = " Cybersecurity "  # stray spaces -> trimmed

    out = clean_responses(df)

    assert len(out) == 6
    assert out[config.RATING_COLS].isin([1, 2, 3, 4, 5]).all().all()
    assert all(pd.api.types.is_integer_dtype(out[q]) for q in config.RATING_COLS)
    assert "Cybersecurity" in out["q19"].tolist()
    assert " Cybersecurity " not in out["q19"].tolist()
    assert int(out.loc[0, "q1"]) == 4


def test_map_to_tracks(fake_raw):
    out = map_to_tracks(fake_raw)

    assert set(out["track"]) == set(config.TRACKS)
    assert not out["q19"].isin(config.EXCLUDED_MAJORS).any()
    assert len(out) == 160  # 4 tracks x 40, the 30 CS/ICT rows are set aside
    for major, track in config.MAJOR_TO_TRACK.items():
        assert (out.loc[out["q19"] == major, "track"] == track).all()


def test_split_train_test(fake_labeled):
    X_tr, X_te, y_tr, y_te = split_train_test(fake_labeled)

    assert len(X_te) == math.ceil(0.2 * len(fake_labeled))
    assert len(X_tr) + len(X_te) == len(fake_labeled)
    assert list(X_tr.columns) == config.QUESTION_COLS
    assert list(X_te.columns) == config.QUESTION_COLS
    assert set(y_tr) | set(y_te) == set(config.TRACKS)
    # each track keeps about 20% in the test set (within one row)
    for track in config.TRACKS:
        expected = 0.2 * (fake_labeled["track"] == track).sum()
        assert abs((y_te == track).sum() - expected) <= 1
    # the seed makes the split repeatable
    again = split_train_test(fake_labeled)
    assert again[1].index.tolist() == X_te.index.tolist()
