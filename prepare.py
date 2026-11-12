"""
Fixed evaluation harness. The agent must not modify this file.
Project setup: fill in every TODO(project) below, once. All project-specific configuration lives here.

train.py defines fit(train_df) -> predict_fn and passes it to evaluate(), which runs NUM_FOLDS-fold
cross-validation over the training data (rows sharing a GROUP value always stay in the same fold).
AUTORESEARCH_TEST=1 instead fits once on all training data and scores on the held-out test set.
"""

import os
import time

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Project configuration
# ---------------------------------------------------------------------------

TASK = "<TASK>"           # TODO(project): one or two sentences: what is predicted, from what
METRIC = "<METRIC>"       # TODO(project): what _metric computes, e.g. "RMSE of predicted RUL"
TARGET = "<TARGET>"       # TODO(project): name of the target column
GROUP = None              # TODO(project): column whose values must never be split across folds
                          # (e.g. "unit", "patient_id"), or None for row-level folds
TIME_BUDGET = 30          # TODO(project): seconds allowed per fit call
NUM_FOLDS = 5             # k for k-fold cross-validation (>= 2)
SPLIT_SEED = 0            # fixes the fold split; the model seed is SEED in train.py

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
TEST_MODE = os.environ.get("AUTORESEARCH_TEST") == "1"


def _load_train():
    """Return the full training table: feature columns, TARGET, and GROUP if set."""
    raise NotImplementedError("TODO(project): load the training data from DATA_DIR, e.g. pd.read_csv")


def _load_test():
    """Return the held-out test table, same columns as _load_train(). Only read when TEST_MODE."""
    raise NotImplementedError("TODO(project): load the test data from DATA_DIR")


def _make_queries(df, rng):
    """
    Turn held-out rows into (queries, targets): what predict_fn receives, and the truth to score against.
    The default hides the target and asks for one prediction per row. Override if a prediction is made
    from something other than a single row (e.g. truncating each held-out unit's history at a random point).
    """
    return df.drop(columns=[TARGET]).reset_index(drop=True), df[TARGET].to_numpy()


def _metric(preds, targets):
    """
    The single number the agent minimises, over all held-out predictions pooled.
    Must be lower-is-better: invert or negate higher-is-better scores (e.g. 1 - AUC).
    """
    raise NotImplementedError("TODO(project): e.g. RMSE, MAE, log loss, 1 - AUC")

# ---------------------------------------------------------------------------
# Evaluation (generic, do not change)
# ---------------------------------------------------------------------------

def _fit_and_predict(fit, train_df, queries, label):
    t0 = time.time()
    predict_fn = fit(train_df)
    fit_seconds = time.time() - t0
    preds = np.asarray(predict_fn(queries))
    assert len(preds) == len(queries), f"expected {len(queries)} predictions, got {len(preds)}"
    print(f"{label}: fit {fit_seconds:.1f}s (budget {TIME_BUDGET}s)", flush=True)
    return preds


def _folds(df, rng):
    """Split row indices into NUM_FOLDS validation folds, keeping each GROUP value in one fold."""
    keys = df[GROUP].to_numpy() if GROUP else np.arange(len(df))
    chunks = np.array_split(rng.permutation(np.unique(keys)), NUM_FOLDS)
    return [np.flatnonzero(np.isin(keys, chunk)) for chunk in chunks]


def evaluate(fit):
    """
    fit: train_df -> predict_fn. train_df has all columns including TARGET. Must finish within TIME_BUDGET.
    predict_fn: queries (see _make_queries) -> array of predictions, one per query.
    Returns _metric over all held-out predictions (lower is better).
    """
    assert NUM_FOLDS >= 2, "NUM_FOLDS must be at least 2"
    df = _load_train().reset_index(drop=True)
    rng = np.random.default_rng(SPLIT_SEED)
    if TEST_MODE:
        print("AUTORESEARCH_TEST=1: fitting on all training data, evaluating on the test set")
        queries, targets = _make_queries(_load_test(), rng)
        preds = _fit_and_predict(fit, df, queries, "test")
    else:
        preds, targets = [], []
        for k, val_idx in enumerate(_folds(df, rng)):
            queries, fold_targets = _make_queries(df.iloc[val_idx], rng)
            train_df = df.drop(index=val_idx).reset_index(drop=True)
            preds.append(_fit_and_predict(fit, train_df, queries, f"fold {k + 1}/{NUM_FOLDS}"))
            targets.append(fold_targets)
        preds, targets = np.concatenate(preds), np.concatenate(targets)
    return float(_metric(preds, targets))
