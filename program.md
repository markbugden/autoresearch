# autoresearch

You are running an autonomous research loop: repeatedly edit `train.py`, evaluate it, and keep only the changes that improve the metric. The task (`TASK`), metric (`METRIC`) and per-fit time budget (`TIME_BUDGET`) are defined at the top of `prepare.py`.

## Setup

`prepare.py`, `train.py` and `program.md` have been copied into the user's project repo and adapted to it. Run every command from the directory that contains them. Work with the user to:

1. **Check the starting point**: no `TODO(project)` is left in `prepare.py` or `train.py`, these files, `pyproject.toml` and `uv.lock` are committed, and `git status` shows no modified tracked files. The loop reverts with `git reset --hard`, which would delete uncommitted work. If anything is missing, ask the user.
2. **Pick a run tag** based on today's date (e.g. `nov4`). The branch `autoresearch/<tag>` must not already exist.
3. **Create the branch**: `git checkout -b autoresearch/<tag>` from the current branch.
4. **Read the in-scope files**:
   - `prepare.py`: task, metric, data loading and k-fold evaluation. Read-only.
   - `train.py`: the file you edit. Defines `fit(train_df) -> predict_fn`.
   - The repo's README, if there is one, for context on the project.
5. **Check the data loads**: `uv run python -c "import prepare; print(prepare._load_train().shape)"`. If it fails, tell the user.
6. **Create `results.tsv`** containing only the header row (see Logging).
7. **Confirm with the user**, then start the loop.

## Rules

`evaluate()` runs `NUM_FOLDS`-fold cross-validation: for each fold it calls `fit` on the other folds and scores predictions on the held-out one. A run takes about `NUM_FOLDS × TIME_BUDGET` seconds. Launch it with `uv run train.py`.

**You CAN** change anything in `train.py`: features, preprocessing, the training target, model, loss, optimizer, hyperparameters.

**You CANNOT**:
- Modify `prepare.py`.
- Add dependencies or edit `pyproject.toml`. Use only the packages it already lists.
- Call `_load_test()`, read the files it loads, or set `AUTORESEARCH_TEST`. The test set is the user's held-out benchmark.
- Use any data in `fit` other than the `train_df` it receives, or otherwise get around `evaluate()`.
- Exceed `TIME_BUDGET` in any `fit` call (the harness prints each fold's time), or use excessive memory.

**The goal: the lowest `val_metric`.** It is always lower-is-better.

**Noise**: Establish the baseline first: run `train.py` unmodified, then with `SEED` in `train.py` set to 43 and 44, and log all three. Restore `SEED = 42` afterwards. The spread (max − min `val_metric`) is the noise level. Only count a change as an improvement if it beats the current best by more than that.

**Don't overfit the validation folds.** Every keep/discard decision leaks a little information about the validation data into `train.py`, and over hundreds of iterations it adds up. The seed spread measures model randomness, not how lucky the split is, so treat it as a lower bound on the real noise. Prefer changes with a clear reason to generalise over tweaks that just happen to help, and be suspicious of a long run of tiny gains.

**Simplicity**: All else being equal, simpler is better. A small gain that adds ugly complexity is not worth it. Removing code and getting equal or better results is a win.

## Output

A finished run ends with:

```
---
val_metric:       0.123456
total_seconds:    151.2
```

## Logging

Log every run to `results.tsv`, tab-separated (commas break in descriptions). Leave it untracked by git.

```
commit	val_metric	status	description
```

- `commit`: short hash (7 chars)
- `val_metric`: e.g. `0.123456`, or `nan` for crashes
- `status`: `keep`, `discard` or `crash`
- `description`: what the run tried

## The loop

LOOP FOREVER:

1. Edit `train.py` with one experimental idea, then commit only that file: `git commit train.py -m "<what it tries>"`.
2. Run `uv run train.py > run.log 2>&1` (redirect everything; don't let output flood your context).
3. Run `grep "^val_metric:" run.log`. If it prints nothing, the run crashed: read `tail -n 50 run.log`. Fix simple mistakes (a typo, a missing import) and re-run; if the idea itself is broken, log it as `crash`, revert, and move on.
4. Log the result in `results.tsv`.
5. If `val_metric` beats the best by more than the noise level, keep the commit. Otherwise `git reset --hard HEAD~1`.

**Timeout**: If a run takes more than twice `NUM_FOLDS × TIME_BUDGET` seconds, kill it, log it as `crash` and revert.

**Never stop**: Once the loop has started, do not ask the user whether to continue. They may be away and expect you to keep working until they interrupt you. If you run out of ideas, look at the data, re-read the in-scope files, combine near-misses, or try more radical changes.
