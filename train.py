"""
The model. This is the file the agent edits: fit(train_df) -> predict_fn, scored by prepare.evaluate.
Usage: uv run train.py
"""

import time

from prepare import TARGET, TIME_BUDGET, evaluate

SEED = 42  # pass to every random component (model init, sampling, shuffling)
# TODO(project): baseline hyperparameters


def fit(train):
    t_start = time.time()
    X = train.drop(columns=[TARGET])
    y = train[TARGET].to_numpy()

    # TODO(project): preprocessing and a simple, untuned baseline model.
    # The whole fit must finish within TIME_BUDGET. For iterative training, stop on the clock:
    #     while time.time() - t_start < TIME_BUDGET:
    #         ...one epoch / step...
    raise NotImplementedError("TODO(project): build and train the baseline model")

    def predict(queries):
        # TODO(project): apply the same preprocessing to queries; return one prediction per query row
        raise NotImplementedError

    return predict


if __name__ == "__main__":
    t_start = time.time()
    val_metric = evaluate(fit)
    print("---")
    print(f"val_metric:       {val_metric:.6f}")
    print(f"total_seconds:    {time.time() - t_start:.1f}")
