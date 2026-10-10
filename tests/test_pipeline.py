import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import pytest

from mauka.data import synthetic_series
from mauka.features import FEATURES, build_training, live_features
from mauka.model import evaluate_models, fit_for_app, positive_proba, time_split
from mauka.nudge import build_facts, fallback_nudge, find_windows, looks_grounded


@pytest.fixture(scope="module")
def df():
    return synthetic_series("2022-08-01", "2024-12-31")


def test_training_shapes_and_no_nans(df):
    X, y = build_training(df)
    assert list(X.columns) == FEATURES
    assert len(X) == len(y) > 5000
    assert not X.isna().any().any()
    assert set(y.unique()) == {0, 1}


def test_target_is_next_day_same_hour(df):
    from mauka.features import add_features
    d = add_features(df)
    t = pd.Timestamp("2023-03-10 08:00")
    assert d.loc[t, "target"] == d.loc[t + pd.Timedelta(days=1), "ok_now"]


def test_time_split_has_no_overlap(df):
    X, y = build_training(df)
    Xtr, ytr, Xte, yte = time_split(X, y)
    assert Xtr.index.max() + pd.Timedelta(days=1) < Xte.index.min()


def test_models_beat_trivial_scores(df):
    out = evaluate_models(df, models=("persistence", "climatology", "logistic", "hgb"))
    by = {r["model"]: r for r in out["results"]}
    assert set(by) == {"persistence", "climatology", "logistic", "hgb"}
    assert by["hgb"]["auc"] > 0.7
    assert by["hgb"]["brier"] < 0.25


def test_missing_tabpfn_is_reported_not_crashing(df, monkeypatch):
    import mauka.model as m
    monkeypatch.setattr(m, "tabpfn_available", lambda: False)
    out = m.evaluate_models(df, models=("hgb", "tabpfn"))
    assert "tabpfn" in out["skipped"]


def test_live_prediction_flow(df):
    full = synthetic_series("2024-06-01", "2025-01-12")
    recent = full.loc["2025-01-08":"2025-01-11"]
    model = fit_for_app(df, "hgb")
    X, today = live_features(recent, today=pd.Timestamp("2025-01-10"))
    p = positive_proba(model, X)
    assert today == pd.Timestamp("2025-01-10")
    assert len(p) == len(X) == 17 and ((0 <= p) & (p <= 1)).all()


def test_find_windows_and_fallback():
    hours = list(range(5, 22))
    probs = [0.1] * 3 + [0.8, 0.9, 0.7] + [0.2] * 5 + [0.6, 0.6] + [0.1] * 4
    w = find_windows(hours, probs)
    assert (w[0]["start"], w[0]["end"]) == (8, 11)
    facts = build_facts("Delhi", w, 140.0, 33.2)
    for lang in ("English", "Hinglish", "Hindi"):
        assert "08:00" in fallback_nudge(facts, lang)
    assert "all day" in fallback_nudge(build_facts("Delhi", [], 200, 40), "English")


def test_most_confident_window_comes_first():
    hours = list(range(5, 22))
    # a short very sure morning window and a long, less sure evening one
    probs = [0.9, 0.9, 0.9] + [0.1] * 8 + [0.55] * 6
    w = find_windows(hours, probs)
    assert (w[0]["start"], w[0]["end"]) == (5, 8)


def test_grounding_check_catches_made_up_text():
    facts = build_facts("Chennai", [{"start": 5, "end": 9, "hours": 4, "mean_p": 0.72}], 100, 30)
    good = "Tomorrow, 05:00 to 09:00 looks nice, with about a 72% chance of a good hour."
    bad_sun = "The sun is shining, so go out between 05:00 and 09:00."
    missing_time = "Tomorrow morning looks nice for a walk."
    assert looks_grounded(good, facts)
    assert not looks_grounded(bad_sun, facts)
    assert not looks_grounded(missing_time, facts)
