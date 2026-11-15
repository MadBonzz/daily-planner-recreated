# Synthetic histories for Plan Smarter

Ten fictional people have **79 completed tasks**, **20 editable open tasks**, and **20 later tasks held back for scoring**. Sixteen original PDFs are used by 34 records, and 55 completed tasks have a written completion note. The benchmark includes 17 declared patterns, one of which is a fixed 75-minute show. Three controls test a stable budget or an outcome that a user could not reasonably predict.

| History | Completed | What changes the budget |
| --- | ---: | --- |
| 01 · Problem sets between lab and home | 7 | Proving a recovery invariant versus filling in an event trace from the same handout |
| 02 · Shipping a trace tool | 7 | Implementing and testing a repair versus reporting on existing logs |
| 03 · Reading to present or to derive | 7 | Reproducing a mathematical derivation versus summarizing a paper’s empirical findings |
| 04 · GPU lab and errands | 7 | Debugging intermittent concurrent behavior versus a scripted sweep |
| 05 · Interview problems around class | 8 | Solving an unfamiliar invariant problem versus retyping a known solution |
| 06 · Neighborhood clinic volunteer | 9 | Capacity decisions with Mara versus routine inventory calls with Dev |
| 07 · Shared apartment | 8 | Hands-on dish cleanup versus laundry that encounters an unexpected problem |
| 08 · Community playhouse | 8 | A fixed show slot versus an open-ended preflight check |
| 09 · Robotics night school | 9 | Runs through a flaky USB bridge versus replaying saved traces |
| 10 · Evening signal-processing student | 9 | An online lecture with recurring distractions versus an offline problem sheet |

## Try a history in the app

Start the planner normally. Select **Dev** and choose a **Sample history**. Its completed tasks include the original time budget, recorded sessions, attached PDFs when supplied and any notes left at completion. Open **History details** to jump to the date of an earlier task, or work on today's open tasks. Edits persist separately for each sample. **Reset sample history** restores that sample. **Demo** opens the personal planner.

The ten event calendars in `schedules/` contain 410 task creations and timer actions. Each event identifies a task in the matching `histories/` file, its local minute after midnight, and a day offset ending on day zero. The task ID connects it to the existing attachment and synthetic ActivityWatch database. Each task's timer periods keep the original durations and breaks, so a window or phone event stays aligned after the calendar moves. For example, the apartment resident renews a filter subscription while the washer is running, and the interview student does the dishes during a break between two attempts at a problem. The calendars omit the held-out tasks.

In **Dev → Editable history → History details**, choose **Try a sample task** and click **Add sample task** to queue a new forecast on the selected plan date. The attached sample handout and bundled relevance results travel with its input. You can also use the CLI’s temporary profile. It recreates the chosen history and stages its held-out tasks on today's plan. Stopping the command discards that trial.

```sh
python -m examples.estimation.run list
python -m examples.estimation.run open 02
```

## Score a chronological replay

Run a single profile from its first creation through its last timer event. Choose a new output directory for each run. You can stop at an event count or pass a specific local date for day zero.

```sh
python -m examples.estimation.replay 06 --output data/results/local/clinic-live-01
python -m examples.estimation.replay 01 --actions 12 --offline --anchor-day 2026-11-15 --output data/results/local/replica-offline-01
```

The configured run uses the same forecasting, relevance and report generation as the Dev replay. Offline mode exercises the planner fallback, skips embeddings and pair decisions, and marks activity requiring a model label as uncertain. Each line of `events.jsonl` records one action, its observed timer state, available completed history for each forecast job, and changes to forecasts and reports. Task IDs match the profile fixtures. The `manifest.json` records fixture hashes and the anchor day. The `summary.json` compares the user's budget and the planner's **initial** forecast to recorded minutes on tasks completed within the selected events. It reports mean absolute error in minutes and counts model, fallback and scheduled forecasts separately.

## Score the estimator

From the repository root with the project environment activated, select a configured model or run the offline baseline. Use `--case TASK_ID` with a history ID or two-digit prefix to isolate one later task.

```sh
python -m examples.estimation.run validate
python -m examples.estimation.run evaluate 02
python -m examples.estimation.run evaluate --strict-model --json data/results/local/estimate-only-01/results.json
python -m examples.estimation.run evaluate --offline --json data/results/local/offline-01/results.json
```

The examples above use the synthetic reviewer's interval labels to test time estimation on repeatable history. Add `--classify-history` to test classification and estimation together.

```sh
python -m examples.estimation.run evaluate --classify-history --strict-model --json data/results/local/end-to-end-01/results.json
```

This reads the synthetic ActivityWatch databases, clips events to recorded work periods, applies exclusions, then sends sanitized historical intervals and task context (including PDF excerpts when supplied) to the configured model. The resulting labels supply the activity evidence for the held-out forecasts. `history_classification` records the calls, timings, failures, and task-level label totals separately from forecast calls. If the provider fails, affected intervals stay uncertain, the run is saved, and `--strict-model` exits nonzero.

Each forecast uses completed tasks strictly earlier than all the holdouts for that person. The sibling holdouts and open Dev tasks never enter the estimation history. Pre-task input comprises the task title, description, category, details, estimate, fixed-duration flag and extracted PDF content. Only past records can supply observed timings, completion notes and activity summaries. The private grading rubric, expected ratio, held-out completion note, recorded sessions and actual outcome are excluded from prediction.

The declared ratio compares the forecast with the user's initial budget. For example, 90 minutes for a 60-minute budget is 1.5×. Each patterned case declares an acceptable band before either model runs. A forecast on either endpoint counts as a hit. The report scores band agreement against this stated pattern and separately measures absolute error against what the fictional user actually spent. An unpredictable control can have no defensible band and is still scored on its actual duration. Tasks marked as fixed-duration preserve their scheduled minutes without calling the model.

The category-only baseline multiplies the original estimate by the median actual-to-estimated ratio for prior tasks in the same category. It retains the user estimate for a new or empty category. The offline benchmark exercises the planner's actual fallback. For online runs, successful model calls are marked `ok`, skipped model calls on scheduled tasks are marked `not_called`, and failed calls are marked `unavailable`. Strict evaluation writes the run and exits with a failure code if a required model response is missing.

The result JSON records the selected historical tasks, forecast range, calibration ratios, factual citations attached to cautions, and individual model responses with their prompt fingerprints. A caution earns source credit when it cites an earlier task named in the private rubric. That check is distinct from judging whether the wording accurately describes the underlying note or ActivityWatch observation.

## Inspect the inputs

The JSON files in `histories/` contain the independent profiles. `tasks` holds completed work, `open_tasks` gives Dev its editable starting state, and `holdouts` contains the prospective tasks and their private grading data. Attached handouts live in `pdfs/`. Several tasks may refer to different sections of the same handout. To regenerate the PDFs, install `reportlab` and run `python data/tasks/render_pdfs.py`.

Each history points to its own read-only ActivityWatch SQLite database in `activitywatch/`. The ten databases contain 625 fictional events. Their schema matches the v6 Rust store, including bucket metadata, a global event ID, the bucket foreign key, start and end times in Unix nanoseconds, JSON event payloads and the original time indexes. The databases contain only sensor observations. Timer sessions, pause boundaries and interval annotations stay in the history JSON. Past user budgets and actuals are part of the fixture. Model forecasts and run logs live under `data/results/`.

All ten histories contain reviewed distractions on previous tasks from both laptop and phone. In GPU lab, for example, two YouTube hockey clips during debugging total 11 minutes, and two visits to the FotMob phone app total 8 minutes. The Device column identifies foreground observations from the laptop, phone or both. When only a laptop idle heartbeat is available it shows Laptop; a gap or a private interval shows Unknown. Device attribution describes watcher evidence, not certainty about the user’s attention.

The clinic’s open flyer task includes Android activity overlapping a desktop editor. The apartment’s open laundry task has phone activity while the desktop is idle. Dev maps those events from the sample’s clock onto the running timer when a sample is loaded. On task review, the app queries the recorded windows, clips pauses and overlaps, applies exclusions, and sends only the sanitized intervals to the classifier. Time added after loading the sample has no synthetic observations.

[Benchmark runs and reports](../results/README.md) · [Original fixtures and report](https://github.com/MadBonzz/daily-planner-recreated/tree/1d1eea60ffacef970206b2120ae3808191941344/data)

Inspect a fixture with the SQLite CLI. The following query lists watcher types and event counts without displaying window titles.

```sh
sqlite3 -readonly data/tasks/activitywatch/06-neighborhood-clinic.db "SELECT b.type, b.client, count(e.id) FROM buckets b LEFT JOIN events e ON e.bucketrow=b.id GROUP BY b.id;"
```
