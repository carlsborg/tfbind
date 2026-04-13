# autoresearch

This is an experiment to have the LLM do its own research.

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `mar5`). The branch `autoresearch/<tag>` must not already exist — this is a fresh run.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from current branch feature/multitf.
3. **Read the in-scope files**: The repo is small. Read these files for full context:
   - `README.md` — repository context.
   - `main.py` — entry point that takes commands "train" or "predict". Do not modify.
   - `tfbind.py` — the file you modify. Model architecture, model id, optimizer, training loop.
4. **Verify data exists**: Check that `./assets/dna/datasets` contains csv data. If not, tell the human to run `python download.py`.
5. **Create dev/test split**: Write and run a one-time Python script that reads each `{TF}_valid_sequences.csv`, shuffles it with a fixed random seed (42), and writes `{TF}_valid_dev.csv` (first 80%) and `{TF}_valid_test.csv` (last 20%) into the same directory. Then delete the script.
6. **Create evaluate_holdout.py**: Create a script at the repo root that runs `run_predict` from `tf_predict.py` on each `{TF}_valid_test.csv` (using the same model checkpoint logic as `main.py`) and prints `mean_test_auc`. This script is never modified after creation.
7. **Initialize results.tsv**: Create `results.tsv` with just the header row (see updated header below). The baseline will be recorded after the first run.
8. **Confirm and go**: Confirm setup looks good.

Once you get confirmation, kick off the experimentation.

## Experimentation

Each experiment runs on the CPU. The training script trains a model for each input transcription factor, and training runs for exactly 2 minutes for each one (controlled by `timeout` in `constants.py`). You launch it simply as: `python main.py --op train`. 
You generate evaluation metrics with  `python main.py --op predict`. The evaluation metrics are auc for each model and mean of all aucs (mean_valid_auc).

**What you CAN do:**
- Modify `tfbind.py` — this is the only file you edit. Everything is fair game: model architecture, optimizer, hyperparameters, training loop, batch size, model size, etc.

**What you CANNOT do:**
- Modify `download.py`. It is read-only. It contains the dataset downloader.
- Install new packages or add dependencies. You can only use what's already in `requirements.txt`.
- Modify the evaluation harness. The `run_predict` function in `tf_predict.py` writes the loss and accuracy metrics.

**The goal is simple: get the highest mean_valid_auc** Since the time budget is fixed, you don't need to worry about training time — it's always 2 minutes per model. Everything is fair game: change the architecture, the optimizer, the hyperparameters, the batch size, the model size. The only constraint is that the code runs without crashing and finishes within the time budget.

**Simplicity criterion**: All else being equal, simpler is better. A small improvement that adds ugly complexity is not worth it. Conversely, removing something and getting equal or better results is a great outcome — that's a simplification win. When evaluating whether to keep a change, weigh the complexity cost against the improvement magnitude. A 0.001 accuracy improvement that adds 20 lines of hacky code? Probably not worth it. A 0.001 accuracy improvement from deleting code? Definitely keep. An improvement of ~0 but much simpler code? Keep.

**The first run**: Your very first run should always be to establish the baseline, so you will run the training script as is.

## Output format

Once the script finishes it printes metrics to ./metrics/{model_id}.json
```
{
  "mean_valid_auc": 0.836732346050379,
  "per_tf": {
    "ARID3": 0.7890007866142889,
    "ATF2": 0.9815269650926661,
    "BACH1": 0.7966377679228196,
    "CTCF": 0.9855576188854415,
    "ELK1": 0.7946618086060317,
    "GABPA": 0.7618498566002636,
    "MAX": 0.8416063440361952,
    "REST": 0.8184587041822184,
    "SRF": 0.8477139683712829,
    "ZNF24": 0.7503096401925834
  }
}
```

You can extract the key metric from the log file:

```
grep mean_valid_auc  /home/ubuntu/ws/tfbind/metrics/mean_valid_auc.json
```

## Logging results

When an experiment is done, log it to `results.tsv` (semicolon-separated, NOT comma-separated — commas break in descriptions).

The TSV has a header row and 5 columns:

```
commit;mean_valid_auc;mean_train_auc;status;description
```

1. git commit hash (short, 7 chars)
2. mean_valid_auc achieved (e.g. 0.88234567) — use 0.000000 for crashes
3. mean_train_auc achieved — use 0.000000 for crashes. Read from the per-model `metrics/{run_id}_metrics.json` files (each has a `train.auc` field).
4. status: `keep`, `discard`, `crash`, or `checkpoint`
5. short text description of what this experiment tried

Example:

```
commit;mean_valid_auc;mean_train_auc;status;description
a1b2c3d;0.837000;0.891000;keep;baseline
b2c3d4e;0.841000;0.895000;keep;increase layers to 3
c3d4e5f;0.835000;0.888000;discard;switch to GeLU activation
d4e5f6g;0.000000;0.000000;crash;double model width (OOM)
a1b2c3d;0.831000;0.000000;checkpoint;holdout test AUC after 10 experiments
```

## The experiment loop

The experiment runs on a dedicated branch (e.g. `autoresearch/mar5`).

LOOP FOREVER:

1. Look at the git state: the current branch/commit we're on
2. Tune `tfbind.py` with an experimental idea by directly hacking the code.
3. git commit
4. Run the experiment: `python main.py --op train > run.log 2>&1` (redirect everything — do NOT use tee or let output flood your context)
5. Read out the results: `grep mean_valid_auc  /home/ubuntu/ws/tfbind/metrics/mean_valid_auc.json`
6. If the grep output is empty, the run crashed. Run `tail -n 50 run.log` to read the Python stack trace and attempt a fix. If you can't get things to work after more than a few attempts, give up.
7. Run `python main.py --op predict > predict.log 2>&1` to generate per-model metrics files. Read `mean_train_auc` by averaging the `train.auc` field across all 10 per-model `metrics/{run_id}_metrics.json` files.
8. **Overfitting guard**: If `mean_train_auc − mean_valid_auc > 0.05`, flag the result as overfitting and discard (git reset) even if valid AUC improved. Log it as `discard` with a note like "overfit: train=X.XXX val=Y.YYY".
9. Record the results in the tsv (NOTE: do not commit the results.tsv file, leave it untracked by git)
10. If mean_valid_auc improved (higher) and the overfitting guard passed, you "advance" the branch, keeping the git commit
11. If mean_valid_auc is equal or worse, you git reset back to where you started
12. **Every 10 experiments**, run `python evaluate_holdout.py > holdout.log 2>&1` and log the result as a `checkpoint` row in results.tsv with `mean_test_auc` in the `mean_valid_auc` column and `0.000000` in `mean_train_auc`. This is the unbiased estimate of true performance.

The idea is that you are a completely autonomous researcher trying things out. If they work, keep. If they don't, discard. And you're advancing the branch so that you can iterate. If you feel like you're getting stuck in some way, you can rewind but you should probably do this very very sparingly (if ever).

**Crashes**: If a run crashes (OOM, or a bug, or etc.), use your judgment: If it's something dumb and easy to fix (e.g. a typo, a missing import), fix it and re-run. If the idea itself is fundamentally broken, just skip it, log "crash" as the status in the tsv, and move on.

**NEVER STOP**: Once the experiment loop has begun (after the initial setup), do NOT pause to ask the human if you should continue. Do NOT ask "should I keep going?" or "is this a good stopping point?". The human might be asleep, or gone from a computer and expects you to continue working *indefinitely* until you are manually stopped. You are autonomous. If you run out of ideas, think harder — read papers referenced in the code, re-read the in-scope files for new angles, try combining previous near-misses, try more radical architectural changes. The loop runs until the human interrupts you, period.
