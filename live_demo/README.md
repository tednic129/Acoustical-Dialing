# Live demo: acoustic pizza dialling

A local web page. It listens to the microphone (or plays recordings), finds each spoken digit,
and shows the answer of all models of 1.a, 1.b and 1.c. It then shows the dialled number.
Everything runs on your own computer. No internet is needed.

Use `py -3.12 -m` on Windows. On other systems use `python -m`.

## Set up (once)

1. Clone the repo. Go into the repo folder.
2. `py -3.12 -m pip install -r requirements.txt`
3. `py -3.12 -m common.prepare_data`
   It makes `data/processed/` from the WAV files (660 recordings).
4. `py -3.12 -m live_demo.build_models`
   It takes about 3 minutes. It makes `models/` (not in git).
   It must print: `All models equal the committed results.`

## Start

`py -3.12 -m live_demo.web`

Open `http://localhost:8765`. Ctrl+C stops the server.

## Logs

Each run ends with a log in `logs/demo/`:

| File | Content |
|---|---|
| `<date>_<time>_<language>_<source>_<voice>.md` | The report. It lists what worked, what failed and why. |
| `<date>_<time>_<language>_<source>_<voice>.json` | All data of the run. |
| `summary.md` | All runs of the folder and the pizza-test counters. |

`logs/demo/` stays on your computer. Git ignores it. Use it for dry runs.

## Logs that count (pizza test)

1. Start the server with your own log folder:
   `py -3.12 -m live_demo.web --log-dir logs/official/<your-name>`
2. Make the runs. Do not change any setting during the test.
3. Add only your own folder to git:
   `git add logs/official/<your-name>`
   Do not use `git add .`
4. Open a pull request from a branch `demo/logs-<your-name>`.

Each person writes only to their own folder. So the pull requests do not conflict.

## Join the logs of the team

After the pull requests are merged, pull `main`. Then run:

`py -3.12 -m live_demo.run_log logs/official`

It writes `logs/official/summary.md` with the counters of all folders below it.

## What to check before you join logs

- `Settings ID` must be the same in all logs. If it is different, the loop or the
  event extraction was not the same.
- `Commit` and `scikit-learn` should be the same.
- `Models ID` is a hash of the model files. It can be different on different computers even
  when the models are equal. The check that counts is the output of `build_models`.
