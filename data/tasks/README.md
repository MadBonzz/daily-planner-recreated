# Synthetic histories for the daily planner

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

For a manual forecast on the benchmark inputs, use the CLI's temporary profile. It recreates the chosen history and stages its held-out tasks on today's plan. Stopping the command discards that trial.

```sh
python -m examples.estimation.run list
python -m examples.estimation.run open 02
```

## Score the estimator

From the repository root with the project environment activated, select a configured model or run the offline baseline. Use `--case TASK_ID` with a history ID or two-digit prefix to isolate one later task.

```sh
python -m examples.estimation.run validate
python -m examples.estimation.run evaluate 02
python -m examples.estimation.run evaluate --strict-model --json data/results/local/my-model.json
python -m examples.estimation.run evaluate --offline --json data/results/local/offline.json
```

Each forecast uses completed tasks strictly earlier than all the holdouts for that person. The sibling holdouts and open Dev tasks never enter the estimation history. Pre-task input comprises the task title, description, category, details, estimate, fixed-duration flag and extracted PDF content. Only past records can supply observed timings, completion notes and activity summaries. The private grading rubric, expected ratio, held-out completion note, recorded sessions and actual outcome are excluded from prediction.

The declared ratio compares the forecast with the user's initial budget. For example, 90 minutes for a 60-minute budget is 1.5×. Each patterned case declares an acceptable band before either model runs. A forecast on either endpoint counts as a hit. The report scores band agreement against this stated pattern and separately measures absolute error against what the fictional user actually spent. An unpredictable control can have no defensible band and is still scored on its actual duration. Tasks marked as fixed-duration preserve their scheduled minutes without calling the model.

The category-only baseline multiplies the original estimate by the median actual-to-estimated ratio for prior tasks in the same category. It retains the user estimate for a new or empty category. The offline benchmark exercises the planner's actual fallback. For online runs, successful model calls are marked `ok`, skipped model calls on scheduled tasks are marked `not_called`, and failed calls are marked `unavailable`. Strict evaluation writes the run and exits with a failure code if a required model response is missing.

The result JSON records the selected historical tasks, forecast range, calibration ratios, factual citations attached to cautions, and individual model responses with their prompt fingerprints. A caution earns source credit when it cites an earlier task named in the private rubric. That check is distinct from judging whether the wording accurately describes the underlying note or ActivityWatch observation.

## Inspect the inputs

The JSON files in `histories/` contain the independent profiles. `tasks` holds completed work, `open_tasks` gives Dev its editable starting state, and `holdouts` contains the prospective tasks and their private grading data. Attached handouts live in `pdfs/`. Several tasks may refer to different sections of the same handout. To regenerate the PDFs, install `reportlab` and run `python data/tasks/render_pdfs.py`.

Ten per-profile ActivityWatch exports in `activitywatch/` contain 461 synthetic watcher events for the completed histories. Replaying them feeds window, browser and AFK events into the planner's normal timer and sanitization code. Relevance labels in the benchmark reports come from fixture annotations. In-progress and future tasks supply no activity evidence to their own estimates.

[Current benchmark results](../results/README.md) · [Original fixtures and report](https://github.com/MadBonzz/daily-planner-recreated/tree/1d1eea60ffacef970206b2120ae3808191941344/data)

To regenerate one raw export from its historical activity scenes, run the command below. The exported watcher buckets contain the fictional device’s events. Periods and relevance annotations belong to the separate profile JSON. During validation, a stale or mismatched checked-in export is rejected.

```sh
python -m examples.estimation.activity_fixture data/tasks/histories/10-signal-study.json data/tasks/activitywatch/10-signal-study.json
```
