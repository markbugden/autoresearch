# Autoresearch

A minimal adaptation of [karpathy/autoresearch](https://github.com/karpathy/autoresearch) for tabular ML tasks. An agent edits `train.py`, trains for a fixed time budget, evaluates with k-fold cross-validation, and then decides to keep or discard the change.

- `prepare.py`: fixed harness. Holds all project configuration (task, metric, target, data loading) and the evaluation.
- `train.py`: the model, `fit(train_df) -> predict_fn`. The only file the agent edits during the loop.
- `program.md`: instructions for the agent. Generic, so no edits are needed.
- `pyproject.toml`: dependencies, managed with [uv](https://docs.astral.sh/uv/). The agent may only use what is listed here.

### Instructions for use
Copy `prepare.py`, `train.py`, `program.md` and `pyproject.toml` to your repo (if it already has a `pyproject.toml`, just `uv add numpy pandas`) and give the agent an instruction like (modify as required):
>Adapt `prepare.py` and `train.py` to the dataset located in the data folder. The task is to predict RUL, the metric is RMSE, and the time budget is 30s per training. Fill in every TODO(project), write a simple untuned baseline in `train.py`, `uv add` any packages it needs, and run it once with `uv run train.py` to check it works. Ask me if anything is unclear.

Before starting the loop:
- `uv add` any other packages you want the agent to be able to use (e.g. `scikit-learn`, `lightgbm`, `torch`). It cannot add dependencies itself.
- Review the adapted files and commit them, together with `pyproject.toml` and `uv.lock`. The loop reverts experiments with `git reset --hard`, so commit or stash any other work in the repo too.

Then start the loop with something like: "Read `program.md` and kick off a new experiment." The agent works on a new `autoresearch/<tag>` branch created from whichever branch you're on, so your branch is left untouched.


### Caution
The current instructions specify the agent should only stop when explicitly instructed by the user, which could result in a large token usage if left unrestricted. In practice, I find it useful to give the agent an upper limit on the number of iterations it should perform (100 seems to work well for the C-MAPSS dataset). 