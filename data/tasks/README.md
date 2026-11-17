# Synthetic histories for Plan Smarter

The Dev picker opens the five editable schedules in [demo/](demo/). Their task rows link to shared PDFs and include historical, in-progress and addable work. [demo/sources.json](demo/sources.json) records where a task also appears in the original benchmark. The picker and the benchmark have separate calendars, ActivityWatch stores and runtime workspaces.

## Try a history in the app

| Picker entry | Completed | Open | Try adding |
| --- | ---: | ---: | --- |
| Nora · distributed systems lab | 16 | 6 | A recovery proof or a short report from existing traces |
| Ishan · paper reading and GPU research | 16 | 6 | An appendix derivation or another working GPU timing sweep |
| Mae · clinic and community theater | 19 | 6 | An intake handover meeting with Mara or an inventory meeting with Dev |
| Leila · interviews and signals evening class | 19 | 6 | A sliding-window problem or an online Fourier lecture |
| Xavier · robotics lab and shared home | 19 | 6 | A physical rover run or tray-bake cleanup |

Start the planner normally. Select **Dev** and choose a **Sample history**. Its completed tasks include the original time budget, recorded sessions, attached PDFs when supplied and any notes left at completion. Open **History details** to jump to the date of an earlier task, or work on today's open tasks. Edits persist separately for each sample. **Reset sample history** restores that sample. **Demo** opens the personal planner.

Open an earlier date to revisit the task description, note and report; the Analytics page also shows dated laptop and phone intervals. The editable calendars and their ActivityWatch databases are in [demo/](demo/). Each calendar dates creation, start, pause, resume and completion relative to the day the profile is opened. Prospective sample tasks are available to add after loading.

In **Dev → Editable history → History details**, choose **Try a sample task** and click **Add sample task** to queue a new forecast on the selected plan date. The attached sample handout and bundled relevance results travel with its input.

## Original benchmark

The ten benchmark histories have **79 completed tasks**, **20 editable open tasks**, and **20 later tasks held back for scoring**. Sixteen original PDFs are used by 34 records, and 55 completed tasks have a written completion note. The benchmark includes 17 declared patterns, one of which is a fixed 75-minute show. Three controls test a stable budget or an outcome that a user could not reasonably predict.

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

The [benchmark retrieval data](retrieval/) includes a CSV for each of the 119 original sample tasks. Its 1,024 rows hold the document and query embeddings. The profile JSON records the two model-input hashes, the public source files and Kev’s pair decisions. Final forecasts are generated using the endpoint you configure.

## Verify the five Dev profiles

The demo combines exact cached Qwen and Kev outputs from the benchmark with [new model outputs](demo/models/). The submitted content, model, query role and decision prompt determine whether a packaged result applies. The Bionic launcher tries the live embedding first and uses a valid vector even if Bionic truncated the input. A failed live lookup can fall back to a packaged result for the exact request. The final forecast and activity classification go to your configured forecast endpoint. Editing task text, attaching another document, reviewing relevance, or changing activity labels can produce a new local-model request.

Run the following command to reconstruct the authored requests and check their coverage without calling a model:

```sh
python -m examples.estimation.package_demo --stage verify
```

To rebuild after changing fixtures, run the stages below from the repository root with the local Qwen and Kev endpoints configured. Each stage skips existing exact matches and writes new results atomically. The site stage requires the completed vector and pair stages. Use `--count-only` to preview missing inputs or `--history d1-systems-student` to build one profile.

```sh
python -m examples.estimation.package_demo --stage vectors
python -m examples.estimation.package_demo --stage pairs
python -m examples.estimation.package_demo --stage workspaces
python -m examples.estimation.package_demo --stage sites
```

## Inspect the inputs

The JSON files in `histories/` contain the original benchmark profiles. `tasks` holds completed work, `open_tasks` gives Dev its editable starting state, and `holdouts` contains the prospective tasks and their private grading data. Attached handouts live in `pdfs/`. Several tasks may refer to different sections of the same handout. To regenerate the PDFs, install `reportlab` and run `python data/tasks/render_pdfs.py`.

Each history points to its own read-only ActivityWatch SQLite database in `activitywatch/`. The ten databases contain 625 fictional events. Their schema matches the v6 Rust store, including bucket metadata, a global event ID, the bucket foreign key, start and end times in Unix nanoseconds, JSON event payloads and the original time indexes. The databases contain only sensor observations. Timer sessions, pause boundaries and interval annotations stay in the history JSON. Past user budgets and actuals are part of the fixture. Model forecasts and run logs live under `data/results/`.

All ten original benchmark histories contain reviewed distractions on previous tasks from both laptop and phone. In GPU lab, for example, two YouTube hockey clips during debugging total 11 minutes, and two visits to the FotMob phone app total 8 minutes. The Device column identifies foreground observations from the laptop, phone or both. When only a laptop idle heartbeat is available it shows Laptop; a gap or a private interval shows Unknown. Device attribution describes watcher evidence, not certainty about the user’s attention.

In the original benchmark, the clinic flyer has Android activity overlapping a desktop editor. The apartment laundry has phone activity while the desktop is idle. Dev maps those events from the sample’s clock onto the running timer when a sample is loaded. On task review, the app queries the recorded windows, clips pauses and overlaps, applies exclusions, and sends only the sanitized intervals to the classifier. Time added after loading the sample has no synthetic observations.

Inspect a fixture with the SQLite CLI. The following query lists watcher types and event counts without displaying window titles.

```sh
sqlite3 -readonly data/tasks/demo/activitywatch/d3-clinic-coordinator.db "SELECT b.type, b.client, count(e.id) FROM buckets b LEFT JOIN events e ON e.bucketrow=b.id GROUP BY b.id;"
```

## Website blocking suggestions

`focus-decisions.json` contains Kev’s decisions for websites labeled as distractions in a single completed comparison task. This original benchmark index includes the dated schedule, the 40 open and addable tasks across the ten archived histories, and the choice to manually select any completed task in the same history. A decision is keyed by the exact model request, including the task scope and observed tab titles.

This command verifies the keys and coverage using only the published samples.

```sh
python -m examples.estimation.package_focus --check
```

To regenerate the decisions with Kev loaded locally, run `python -m examples.estimation.package_focus`. An optional `--checkpoint PATH` saves each completed judgment for interrupted builds.
