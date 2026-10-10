# The models, and a fair way to compare them.
#
# TabPFN is the main model. The others are baselines, so we can see if TabPFN
# is actually better than something simple:
#   persistence - "tomorrow will be like the same hour today"
#   climatology - how often that month and hour is usually good
#   logistic    - a simple linear model
#   hgb         - gradient boosted trees

import time

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, f1_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .features import build_training


class Persistence:
    def fit(self, X, y):
        return self

    def predict_proba(self, X):
        p = np.clip(X["ok_now"].to_numpy(), 0.02, 0.98)
        return np.column_stack([1 - p, p])


class Climatology:
    def fit(self, X, y):
        keys = pd.MultiIndex.from_arrays([X["month"], X["hour"]])
        self.rate = pd.Series(y.to_numpy(), index=keys).groupby(level=[0, 1]).mean()
        self.overall = float(y.mean())
        return self

    def predict_proba(self, X):
        keys = pd.MultiIndex.from_arrays([X["month"], X["hour"]])
        p = self.rate.reindex(keys).fillna(self.overall).to_numpy()
        p = np.clip(p, 0.02, 0.98)
        return np.column_stack([1 - p, p])


class TabPFNModel:
    # TabPFN works best on a few thousand rows, so we cap the training size.
    # We also predict in small batches to keep memory use low.

    def __init__(self, max_train=4000, batch=1500):
        self.max_train = max_train
        self.batch = batch

    def fit(self, X, y):
        from tabpfn import TabPFNClassifier  # only imported if we actually use it

        if len(X) > self.max_train:
            pick = np.random.default_rng(0).choice(len(X), self.max_train, replace=False)
            X, y = X.iloc[pick], y.iloc[pick]
        self.clf = TabPFNClassifier()
        self.clf.fit(X.to_numpy(dtype=float), y.to_numpy())
        return self

    def predict_proba(self, X):
        rows = X.to_numpy(dtype=float)
        parts = [self.clf.predict_proba(rows[i:i + self.batch])
                 for i in range(0, len(rows), self.batch)]
        proba = np.vstack(parts)
        if proba.shape[1] == 1:  # training data had only one class
            proba = np.column_stack([1 - proba[:, 0], proba[:, 0]])
        return proba


def tabpfn_available():
    try:
        import tabpfn  # noqa: F401
        return True
    except Exception:
        return False


def make_model(name, max_train=4000):
    if name == "persistence":
        return Persistence()
    if name == "climatology":
        return Climatology()
    if name == "logistic":
        return make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    if name == "hgb":
        return HistGradientBoostingClassifier(max_depth=4, learning_rate=0.08,
                                              max_iter=250, random_state=0)
    if name == "tabpfn":
        return TabPFNModel(max_train=max_train)
    raise ValueError(f"Unknown model: {name}")


def positive_proba(model, X):
    # chance that the hour is good
    p = model.predict_proba(X)
    return p[:, 1] if p.ndim == 2 and p.shape[1] > 1 else np.ravel(p)


def time_split(X, y, test_frac=0.25, gap_days=1):
    # train on the past, test on the most recent days.
    # the gap stops a training answer from overlapping with a test input.
    days = X.index.normalize()
    unique_days = days.unique().sort_values()
    cutoff = unique_days[int(len(unique_days) * (1 - test_frac))]
    train = days <= cutoff - pd.Timedelta(days=1 + gap_days)
    test = days >= cutoff
    return X[train], y[train], X[test], y[test]


def score(y_true, p):
    y_true = np.asarray(y_true)
    result = {
        "brier": float(brier_score_loss(y_true, p)),
        "accuracy": float(accuracy_score(y_true, p >= 0.5)),
        "f1": float(f1_score(y_true, p >= 0.5, zero_division=0)),
    }
    if len(np.unique(y_true)) > 1:
        result["auc"] = float(roc_auc_score(y_true, p))
    else:
        result["auc"] = float("nan")
    return result


def evaluate_models(df, models=("persistence", "climatology", "logistic", "hgb", "tabpfn"),
                    max_train=4000, max_test=None):
    X, y = build_training(df)
    X_train, y_train, X_test, y_test = time_split(X, y)

    if max_test and len(X_test) > max_test:
        pick = np.sort(np.random.default_rng(0).choice(len(X_test), max_test, replace=False))
        X_test, y_test = X_test.iloc[pick], y_test.iloc[pick]

    results, skipped = [], {}
    for name in models:
        if name == "tabpfn" and not tabpfn_available():
            skipped[name] = "tabpfn is not installed (pip install tabpfn)"
            continue
        start = time.time()
        try:
            model = make_model(name, max_train=max_train).fit(X_train, y_train)
            p = positive_proba(model, X_test)
        except Exception as e:
            skipped[name] = f"{type(e).__name__}: {e}"
            continue
        results.append({"model": name, **score(y_test, p), "seconds": round(time.time() - start, 1)})

    return {
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "train_base_rate": float(y_train.mean()),
        "test_base_rate": float(y_test.mean()),
        "test_period": [str(X_test.index.min().date()), str(X_test.index.max().date())],
        "results": results,
        "skipped": skipped,
    }


def fit_for_app(df, model_name, max_train=4000):
    # train on everything we have
    X, y = build_training(df)
    return make_model(model_name, max_train=max_train).fit(X, y)
