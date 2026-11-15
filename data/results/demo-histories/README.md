# Five editable Dev histories

The five picker entries start with **89 completed and 30 unfinished tasks**. Each person has six unfinished tasks and three prospective tasks to add. The demo retains 79 completed task definitions from the ten-person benchmark and adds ten newly written tasks. The demo has its own dated schedules and newly authored sensor traces. PDF attachments and matching cached model requests reuse the existing assets. [Source mapping](../../tasks/demo/sources.json) · [Open a sample in the app](../../tasks/README.md)

## How much activity is visible?

The counts below come from a read-only SQLite integrity check and replay through the planner’s ordinary activity parser. Raw records include independent window, browser, AFK and Android watcher events, including events outside work periods. A task's raw count includes only distinct rows overlapping one of its timer periods. The report clips those rows to timers, combines overlapping sensors and also represents unobserved time. Counts in parentheses give the minimum and maximum across individual completed tasks.

| Measure | Ten original benchmarks | Five editable histories |
| --- | ---: | ---: |
| Completed / unfinished / addable tasks | 79 / 20 / 20 | 89 / 30 / 15 |
| Completed tasks with a written note | 55 | 71 |
| Calendar actions | 410 | 448 |
| Native SQLite watcher records | 625 | 1,911 |
| Median raw records per completed task | 5 (0–17) | 12 (1–65) |
| Median visible report intervals per completed task, including uncertainty | 4 (1–11) | 11 (1–62) |
| Median distinct foreground contexts per completed task | 3 (0–10) | 6 (0–14) |
| Median timer periods per completed task | 1 (1–2) | 1 (1–2) |
| Timed minutes with some visible foreground | 2,992 of 3,639 (82.2%) | 3,303 of 3,976 (83.1%) |
| Completed tasks with no foreground evidence | 12 | 1 |

A foreground context is an observed combination of app, window title, domain and phone app. An AFK heartbeat by itself supplies no foreground evidence. Every demo has laptop-only, phone-only, simultaneous laptop-and-phone, and unobserved intervals. Across completed tasks the review annotations classified approximately 3,086 minutes as work, 156 as distraction and 734 as uncertain. These are synthetic reviewer labels for exercising the app. [Per-task counts, device totals and label totals](coverage.json)

| Picker entry | Completed | Ready / paused / running | Watcher rows | Median raw / merged per task | Days with task starts | Completed with note | PDF references / unique files |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Nora · distributed systems lab | 16 | 4 / 2 / 0 | 444 | 14.5 / 14 | 18 | 10 | 12 / 4 |
| Ishan · paper reading and GPU research | 16 | 4 / 2 / 0 | 471 | 23 / 21 | 17 | 8 | 10 / 2 |
| Mae · clinic and community theater | 19 | 4 / 1 / 1 | 262 | 8 / 8 | 15 | 19 | 3 / 3 |
| Leila · interviews and signals evening class | 19 | 4 / 1 / 1 | 439 | 13 / 12 | 20 | 15 | 5 / 3 |
| Xavier · robotics lab and shared home | 19 | 4 / 2 / 0 | 295 | 8 / 8 | 20 | 19 | 4 / 4 |

Nora's 132-minute ring cursor repair includes 48 intervals across 12 contexts. Ishan’s partial-warp investigation includes 56 intervals across 13 contexts. Mae’s 50-minute fixed caption performance shows one sustained foreground window. Xavier’s five-minute mug rinse has one uncertain interval and no foreground. A task can therefore carry a reliable elapsed time with little activity evidence. The synthetic databases cover selected sessions in these fictional calendars.

## How much distraction evidence is available?

These counts use the saved, reviewed `off_task` intervals on completed tasks. The 1,911 raw watcher rows also include work, AFK, simultaneous device signals, gaps between tasks and unfinished sessions. Each classified distraction interval contributes its time once to the overall total.

| Reviewed distraction measure | Ten original benchmarks | Five editable histories |
| --- | ---: | ---: |
| Completed tasks with at least one distraction | 31 of 79 | 34 of 89 |
| Affected completed tasks per person | 3–4 | 6–8 |
| Off-task intervals | 38 | 41 |
| Off-task minutes | 146.00 | 156.18 |
| Distinct foreground domains / phone apps | 9 / 7 | 10 / 7 |
| Share of all recorded time | 4.0% | 3.9% |

The five calendars each combine more prior work, which explains much of the per-person increase. Leila has six affected tasks and 13 off-task intervals. Her 40 minutes of reviewed YouTube distractions include unrelated hockey and van-build clips. Her Fourier lecture footage on the same site remains available as work evidence. Ishan’s reply about the clipping step shows a 191-second cricket-scorecard visit. Mae’s two clinic-rota tasks contain separate crossword windows of 180 and 204 seconds. Xavier’s new rover crossing has 216 seconds on a GeoGuessr challenge. The JSON lists off-task minutes per domain and phone app for each person. A distraction label records what happened in that task. Any default site to block for a new task gets a separate Kev decision. [Per-person distraction counts](coverage.json) · [Default block-list comparison](default-sites.json)

## Calendar and dashboard checks

The five calendars span 22–26 days. Forty-two of 89 completed tasks were created on an earlier day than the first timer start. Median creation-to-start lead times range from 73 minutes for Leila to 2,382 minutes for Ishan. All 448 actions replayed chronologically with an anchor of November 15, 2026. All 113 forecast calls saw only work completed by their calendar action. This state check disabled inference and produced fallback forecasts. It tested the order in which evidence became available.

The editable-profile checks opened all five through the dashboard loader, navigated archived task dates, and verified descriptions, completion notes, original durations, PDFs and both devices in the analytics logs. They added all 15 sample tasks through the creation entry point. Each started Ready with zero recorded time and no leaked completion note, activity report or private rubric. The original ten remain available to the benchmark CLI. [Dashboard regression](../../../tests/test_demo_histories.py) · [Original benchmark snapshot](https://github.com/MadBonzz/daily-planner-recreated/tree/44fd7074382b0af20fe288c5ced065a671d4a2e6/data/tasks)

## Sample-add timing patterns

The prior ratios below summarize relevant historical actual time divided by the user's original budget. They describe a signal available before adding a task. The hypothetical target completion time is never part of the demo task.

| Addable task | Prior timing evidence | Contrast in the same calendar |
| --- | --- | --- |
| Nora’s bridge proof | Three proofs took 1.20×, 1.25× and 1.20× the budget. | Her two short write-ups from saved traces took 0.64× and 0.68×. |
| Ishan’s appendix proof | Three earlier derivations took 1.72–1.76×. | Two runs of a working GPU sweep took 0.98× and 1.02×. |
| Mae’s handover with Mara | Three capacity meetings took 1.30–1.37×. | Three prepared stock calls with Dev took 0.70–0.80×. |
| Leila’s triangular-wave lecture | Three prior lecture sessions took 1.34–1.38×. | Her two offline calculation exercises took 0.94× and 1.07×. |
| Xavier’s two physical rover crossings | Three old-bridge tests took 1.54–1.57×. | The rubric leaves the spare duvet’s timing open. Older laundry can inform ordinary hands-on effort, and the unknown fastening condition makes its outcome unpredictable. |

With the default comparisons, 12 of 45 open or addable demo tasks suggest blocking at least one website, compared with 8 of 40 in the archived histories. The recommendations cover seven domains, up from five. Xavier’s two rover tasks both suggest `geoguessr.com`. All five calendars also replayed their 448 actions with zero local inference calls, using simulated Cohere replies for the final forecast and activity labels. [Per-card site evidence](default-sites.json) · [Offline replay checks](packaged-workflows.json)

## Which past tasks reached the estimator?

We submitted all 15 prospective tasks through the dashboard’s sample-add entry point and ran the ordinary Qdrant search and Kev screening against each person’s completed tasks. The test reran search with the packaged Qwen embeddings and exact-request Kev responses. The final forecast model was disabled. These scores measure which duration comparisons it would receive. The rubric asks whether a past task’s operations, method and scope make its elapsed time useful. Of 252 eligible target–history pairs, 45 were labeled useful, 205 unrelated and two ambiguous pairs were left unscored. The labels for newly available pairs were fixed before inspecting the retrieval output. [Relevance rubric](addable-labels.json) · [Ranked scores and selected IDs for all 15 targets](coverage.json)

| Stage | True positives | False positives | False negatives | True negatives | Precision | Recall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen retrieval at 0.50 | 39 | 19 | 6 | 186 | 67.2% | 86.7% |
| Kev on Qwen’s 58 graded candidates | 30 | 0 | 9 | 19 | 100% | 76.9% |
| Qwen followed by Kev, all 250 graded pairs | 30 | 0 | 15 | 205 | 100% | 66.7% |

| Calendar | Useful comparisons in rubric | Retrieved by Qwen | Kept by Kev |
| --- | ---: | ---: | ---: |
| Nora | 10 | 8 | 4 |
| Ishan | 9 | 8 | 8 |
| Mae | 9 | 8 | 8 |
| Leila | 8 | 6 | 5 |
| Xavier | 9 | 9 | 5 |

Kev kept all three Mara meetings for Mae’s new Mara handover and all three Dev calls for her new Dev meeting. For Ishan’s appendix proof it kept four derivation tasks, including a related proof email at cosine 0.503, and rejected an empirical paper at 0.888. Xavier’s physical rover crossing illustrates the cost of screening. Qwen found all four useful runs at 0.549–0.626, but Kev kept only the caster crossing. Leila’s offline Fourier worksheet illustrates a retrieval miss. Neither previous calculation exercise reached Kev, which rejected the one candidate, a video lecture. Fourteen of the 15 new tasks received at least one relevant comparison. The sample shares task families with the archived benchmarks, so this measures the shipped profiles’ behavior.

## Reproduce

From the repository root with the planner requirements installed, run

```sh
python -m examples.estimation.audit_demo_histories data/results/local/demo-check.json --replay --retrieval
python -m pytest -q tests/test_demo_histories.py
```

The quality audit opens each v6 watcher store, checks its foreign keys, verifies its file hash against the supplied history, and checks that each completed task’s report accounts for all its timed seconds. Its chronology and retrieval checks disable final model inference. The original benchmark file directories are retained in place. [Validator](../../../examples/estimation/audit_demo_histories.py)
