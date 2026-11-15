# Plan Smarter

A local, daily-only planner for comparing the time you set aside with the time your work actually takes. Completed tasks become examples for future estimates. The interface and application logic run in Python.

## Run on macOS

The quickest demo uses the five bundled Dev histories and a Cohere API key. The planner, databases and retrieval cache run on your Mac. Cohere supplies the forecast and other chat responses. For fully local inference on Apple Silicon, see [Local LM Studio](#local-lm-studio). LM Studio supports Apple Silicon on macOS 14 or newer and recommends 16 GB or more of RAM. On an Intel Mac, use the hosted forecast configuration. [LM Studio system requirements](https://lmstudio.ai/docs/app/system-requirements)

1. Install Python 3.12 for macOS, or use `brew install python@3.12` if you have Homebrew. If `python3.12 --version` already reports 3.12, start at the clone command. [Python macOS installers](https://www.python.org/downloads/macos/) · [Homebrew formula](https://formulae.brew.sh/formula/python@3.12)

   ```sh
   git clone https://github.com/MadBonzz/daily-planner-recreated.git
   cd daily-planner-recreated
   python3.12 -m venv .venv
   .venv/bin/python -m pip install -r requirements.txt
   .venv/bin/python -m pip check
   ```

2. For hosted forecasts, copy the example only when creating a new `.env`. Edit the file in the repository root and set `COHERE_API` to your own key. The example already selects `command-r7b-12-2024`. If you have an existing `.env`, use the settings in [Choose the model endpoint](#choose-the-model-endpoint).

   ```sh
   cp -n .env.example .env
   nano .env
   ```

3. Start the app. Leave this Terminal window open, and visit the URL it prints. Ctrl+C stops the server.

   ```sh
   .venv/bin/python -m streamlit run app.py --server.address 127.0.0.1 --server.port 8501
   ```

4. Open **Dev → Sample history**, choose **Nora · distributed systems lab**, and keep **Editable history** selected. In **History details**, select a date and click **Open date** to inspect dated tasks, completion notes and ActivityWatch logs. Under **Try a sample task**, choose **Solve part B of the bridge handout** and click **Add sample task**. Its card initially shows **Estimating task time…**. Once the provider replies, check **Why this estimate?** for the model, comparisons, minutes, range and cautions. **Re-estimate** requests a fresh forecast. Under **Chronological replay**, **Apply next action** advances one scheduled action and **Run to end** runs the whole sequence. Each stage that needs a new forecast or model-labeled activity can make another hosted request.

5. To see a website suggestion, switch to **Ishan · paper reading and GPU research**. Under **History details → Try a sample task**, add **Fix nondeterministic segmented sums**. Once its estimate is ready, open **Websites to block during this task** on the card. Two related kernel-debugging sessions recorded 5 and 6 minutes on unrelated YouTube hockey videos. You can remove `youtube.com` under **Blocked websites** and click **Save websites**. [Block suggestions for the demo tasks](data/results/demo-histories/default-sites.json)

The supplied profiles include synthetic ActivityWatch stores, PDFs, precomputed Qwen vectors and Kev decisions. You can explore the supplied tasks without installing ActivityWatch or model weights. The [macOS smoke workflow](https://github.com/MadBonzz/daily-planner-recreated/actions/workflows/mac-smoke.yml) checks the clean install, picker and a full replay on both Apple Silicon and Intel, using a simulated hosted response. The Apple Silicon job also opens the PDF workspace in Chromium, asks a document question and checks the remaining time after pausing. Writing a new task, changing its text or choosing new comparisons can require fresh retrieval. When local models are unavailable the card reports any retrieval or provider error. **Demo** opens your separate personal plan. To check your own activity there, follow [ActivityWatch](#activitywatch).

To check the bundled inputs for Nora’s editable history and replay on your own machine, run the following from the repository. It verifies the precomputed embedding, relevance, workspace and site decisions without starting LM Studio or calling Cohere.

```sh
.venv/bin/python -m examples.estimation.package_demo --stage verify --history d1-systems-student
```

## Run on Windows

Use Python 3.11 or newer. After cloning the repository, open its directory in PowerShell and run:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py --server.address 127.0.0.1
```

If an existing install reports `No module named dotenv`, run the pip command above with the same virtual environment you use to start Streamlit. The package is named `python-dotenv`.

On Linux, use `python3 -m venv .venv` and `.venv/bin/python` for the last two commands. Open the local URL printed by Streamlit. The planner can record tasks and sessions before the optional services below are available.

## Choose the model endpoint

For Cohere, keep your existing `COHERE_API` key in the repository-root `.env` and add these two lines:

```dotenv
PLANNER_MODEL_ENDPOINT=https://api.cohere.ai/compatibility/v1/chat/completions
PLANNER_MODEL=command-r7b-12-2024
```

The hosted configuration uses **Command R7B** for estimates and activity classification. Command R7B and LM Studio use the same OpenAI-compatible structured request. The client checks each reply against the requested JSON Schema. [Cohere Command R7B](https://docs.cohere.com/v2/docs/command-r7b)

After pulling this update, install the updated requirements using your existing virtual environment and restart Streamlit. In **Options → Planner settings**, check the model and host. Create a task and expand **Why this estimate?** to see its source and explanation. To see a forecast with personal timing evidence using only your Cohere key, choose **Dev → Editable history**, open **History details**, pick **Try a sample task** and click **Add sample task**. The package includes the PDFs, Qwen vectors and Kev decisions for those examples. Choose **Re-estimate** to request a fresh Cohere forecast.

Task records, original PDFs and ActivityWatch databases stay on your computer. A hosted forecast or chat sends the selected task details, PDF passages, relevant history and completion notes needed for that request. Activity reports send filtered window titles, website domains and app names. Analytics questions can send scoped task and log data. Workspace chat can send the question, conversation, draft or selected document passages. Local inference uses the loopback services after the initial downloads and needs no Cohere key.

### Local LM Studio

On an Apple Silicon Mac running macOS 14+, install the LM Studio desktop app and launch it once, or install its headless daemon from Terminal. The official headless command also supports Linux. The commands below use the llama.cpp GGUF runtime and serve all three roles on loopback. [LM Studio download](https://lmstudio.ai/download) · [Headless installation](https://lmstudio.ai/docs/developer/core/headless) · [CLI](https://lmstudio.ai/docs/cli)

```sh
curl -fsSL https://lmstudio.ai/install.sh | bash
```

Open a new Terminal if the installer updates your shell path, then run `lms daemon up` for the headless install. Check `lms runtime ls`. Use `lms runtime get llama.cpp` to install the engine or `lms runtime update llama.cpp` to update it. Kev needs a release with the native System One decisions endpoint. [Runtime commands](https://lmstudio.ai/docs/cli/runtime/runtime) · [System One endpoint](https://lmstudio.ai/docs/developer/jev-compat/systemone)

Download the following GGUF files to your Mac and run the import commands from that folder. `--hard-link` keeps the downloads available without a second copy. If you downloaded to an external volume, use `--copy` to place a copy on the volume where LM Studio stores models. The benchmark used these exact quantizations. Only one Bonsai model is needed for normal use.

| Role | GGUF file |
| --- | --- |
| Embedding | [Qwen3-Embedding-0.6B-Q8_0.gguf](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B-GGUF/blob/370f27d7550e0def9b39c1f16d3fbaa13aa67728/Qwen3-Embedding-0.6B-Q8_0.gguf) |
| Decision | [Kev-4B-Q4_K_M.gguf](https://huggingface.co/ggml-org/Kev-4B-GGUF/blob/f4624b6bb7072f1bab726a622c1cad3f96030b9e/Kev-4B-Q4_K_M.gguf) |
| Forecast, activity and chat | [Ternary-Bonsai-4B-Q2_0_g64.gguf](https://huggingface.co/prism-ml/Ternary-Bonsai-4B-gguf/blob/a3eb42bafe873f9686bc97486c43b72ef7d75ec8/Ternary-Bonsai-4B-Q2_0_g64.gguf) |
| Optional 9B forecast model | [Ternary-Bonsai-2-9B-Q2_0.gguf](https://huggingface.co/prism-ml/Ternary-Bonsai-2-9B-gguf/blob/82265ccc78ddaac4700286a35e9badcc32376dd9/Ternary-Bonsai-2-9B-Q2_0.gguf) |

```sh
lms import Qwen3-Embedding-0.6B-Q8_0.gguf --user-repo Qwen/Qwen3-Embedding-0.6B-GGUF --hard-link --yes
lms import Kev-4B-Q4_K_M.gguf --user-repo ggml-org/Kev-4B-GGUF --hard-link --yes
lms import Ternary-Bonsai-4B-Q2_0_g64.gguf --user-repo prism-ml/Ternary-Bonsai-4B-gguf --hard-link --yes
lms ls
```

Load each model under the identifiers the planner uses. The first argument is the model key printed by `lms ls`. With these GGUFs, Qwen’s key starts with `text-embedding-`. If your catalog lists a different key, use the one paired with the exact file in `lms ls --json` and keep the identifier shown below. The commands select CPU inference. Append `--estimate-only` to a load command to estimate memory needs before loading. [Import options](https://lmstudio.ai/docs/cli/local-models/import) · [Load options](https://lmstudio.ai/docs/cli/local-models/load)

```sh
lms load text-embedding-qwen3-embedding-0.6b --gpu off --context-length 8192 --identifier qwen3-embedding-0.6b --yes
lms load kev-4b --gpu off --context-length 4096 --identifier kev-4b --yes
lms load ternary-bonsai-4b --gpu off --context-length 8192 --identifier ternary-bonsai-4b --yes
lms server start --bind 127.0.0.1 --port 1234
lms ps
```

Back in the repository, select the local forecast in `.env`. The embedding and decision endpoints default to the same loopback port.

```dotenv
PLANNER_MODEL_ENDPOINT=http://127.0.0.1:1234/v1/chat/completions
PLANNER_MODEL=ternary-bonsai-4b
PLANNER_EMBEDDING_MODEL=qwen3-embedding-0.6b
PLANNER_DECISION_MODEL=kev-4b
PLANNER_MODEL_TIMEOUT_SECONDS=360
```

From the repository root, check the three endpoints with a synthetic task and launch the app. To inspect local server traffic, run `lms log stream` in a separate Terminal. Long documents may need a larger Qwen context and longer inference can need a larger timeout. The default timeout is 90 seconds.

```sh
.venv/bin/python -m examples.estimation.check_local_models
.venv/bin/python -m streamlit run app.py --server.address 127.0.0.1
```

To try 9B, import `Ternary-Bonsai-2-9B-Q2_0.gguf` under `prism-ml/Ternary-Bonsai-2-9B-gguf`, unload the 4B forecaster with `lms unload ternary-bonsai-4b`, load the 9B file with `--identifier ternary-bonsai-2-9b`, and change `PLANNER_MODEL` accordingly. Leave Qwen and Kev loaded.

### Local relevance models

Qwen3-Embedding-0.6B searches prior tasks and Kev-4B judges each candidate. With a hosted forecaster you can still run just Qwen and Kev locally for your own new tasks. Precomputed vectors and decisions are keyed to the exact inputs. If a comparison needs fresh local inference and the models are unavailable, you can select completed comparisons under **Edit relevant tasks** and request a new forecast.

The planner embeds each task’s title, description, details and extracted document text together. Qwen encodes the full new task both with a retrieval instruction about comparable work, method and scope and in plain text. Qdrant compares each query with plain historical task vectors. A candidate goes to Kev once if either cosine reaches 0.50. The task card displays the larger score. You can raise this cutoff in `.env`. Category breaks tied cosine scores. The [threshold evaluation](data/results/retrieval/2026-11-15-qwen06b-kev4b/README.md) records the cosine sweep and the results for the earlier 0.70 cutoff. Check `.env.example` for local endpoint and threshold overrides.

## ActivityWatch

Run [ActivityWatch](https://activitywatch.net/) on the same computer. Enable its window and AFK watchers. Install its browser extension for active-tab details. On macOS the planner checks `~/Library/Application Support/activitywatch/aw-server-rust/sqlite.db`. On Windows it checks the desktop and synced stores listed below. If it cannot find a desktop store automatically, it reads the local ActivityWatch API on port 5600. Pause or Stop closes the timed period immediately. The **Finished?** prompt determines whether to add that task to estimation history.

On your Mac, start ActivityWatch and check its local interface at `http://127.0.0.1:5600`. In **System Settings → Privacy & Security → Accessibility**, allow ActivityWatch if window titles are missing. The browser watcher adds tab-level data. The planner opens the native Rust database read-only. If that store is absent, it can read other server layouts through the local ActivityWatch HTTP API. For a synced phone, set `PLANNER_ACTIVITYWATCH_PHONE_DB` to its local absolute path. Use **Options → Planner settings → Activity capture** for exclusions and to pause capture. Try a short personal task in **Demo** and open its **Activity report → Activity timeline** after finishing. Dev histories use the bundled synthetic logs. [ActivityWatch getting started](https://docs.activitywatch.net/en/latest/getting-started.html) · [Data directories](https://docs.activitywatch.net/en/latest/directories.html) · [macOS permissions](https://docs.activitywatch.net/en/latest/updating.html#macos)

| Windows source | Location checked automatically |
| --- | --- |
| Laptop ActivityWatch | `%LOCALAPPDATA%\activitywatch\aw-server-rust\sqlite.db` |
| Syncthing copy on the laptop | `%USERPROFILE%\Downloads\syncthing-windows-*\syncthing-windows-*\activity-watch-one-plus\<UUID>\test.db`, including the `<device>\<UUID>\test.db` layout |

The synced folder can contain either device's records. The planner identifies the Android foreground watcher by its bucket metadata. It reads SQLite in read-only mode, restricts each query to the timer's active periods, and applies exclusions to both devices. The classifier sees an interval ID, seconds, filtered laptop app, window or tab title, domain, laptop idle status and the concurrent Android app when available. Overlapping observations count once. An app in the foreground alone cannot establish what someone was doing. A classifier can mark conflicting signals uncertain.

For a different installation, set `PLANNER_ACTIVITYWATCH_DB` for the laptop database or `PLANNER_ACTIVITYWATCH_PHONE_DB` for a synced store with an Android watcher in the repository-root `.env`. Use literal paths, for example `D:/ActivityWatchSync/phone/test.db`. If a desktop database contains several computers, `PLANNER_ACTIVITYWATCH_HOSTNAME` selects one. The per-task timeline shows **Phone app** when a matching Android event overlaps the task. Past classified intervals can surface in **Why this estimate?** for a related task.

Starting a different task automatically pauses the previous one. Choosing **Continue later** after Stop saves the session and keeps the task open for another day.

Planning uses time spent across running sessions plus any time you enter manually. Pauses and gaps between sessions contribute no time. Choose **Edit time spent** on a ready, paused or completed task and enter the total so far. For example, enter 30 after working half an hour away from the timer. Extra minutes have no device timestamps and appear as uncertain in the activity report. Reducing the total removes the newest minutes first and trims the corresponding ActivityWatch windows.

Choose **Mark complete** to finish a task directly and optionally supply its duration and a completion note. A completion without a known duration stays out of timing comparisons. Once a running task reaches twice your original estimate, **Stop and review time** lets you correct the total before finishing or continuing later. ActivityWatch manages its own raw event storage and retention.

## Website blocking

On a task card, open **Websites to block during this task** to see the sites suggested by past activity. One off-task visit on a related task is enough for Kev to review the domain. For example, it can suggest blocking a chess puzzle during another proof-writing task. The card shows where the visit occurred and for how long. Kev also checks recurring visits before suggesting a block, so a platform used for past distractions can stay available when the current task needs it. A site used for work in the same comparisons or named in the new task stays available unless you choose to block it. Review the evidence, remove a suggestion or add a domain. Once you pair the browser, both **Start** and **Start with workspace** apply that task’s saved list. Pausing releases the sites and restores already-open tabs. Switching tasks applies the next task’s list.

Pair the extension once per Chrome or Edge profile under **Options → Planner settings → Browser blocking**. This works within the browser and requires no administrator access. [Install, pair and test the extension](browser_extension/README.md)

## Daily review and postponing work

On an unfinished task card, **Postpone** moves the task to tomorrow or a later date you choose. For a task already planned in the future, **Next day** means the day after its scheduled date. Postponing a running task pauses its timer immediately. You can resume it on its new date. Time already tracked remains visible on the days you worked, and the task keeps its estimate, attachments, and comparisons.

In **Demo → Options → Planner settings → Daily review reminder**, set a local time or turn reminders off. The default is **9:30 PM** on the computer running Plan Smarter. The live planner checks every 30 seconds while it is open. **Review this day** opens the plan, where you can finish tasks or postpone them. A dismissed reminder stays dismissed across tabs and restarts. A late opening can catch up on yesterday’s plan. Empty days are skipped.

On Windows, enable **Notify my Windows desktop even when the planner is closed** and save. Plan Smarter registers a daily task under your signed-in Windows account at limited privileges. The browser and Streamlit can then both be closed. Keep the project folder and Python environment in place. Delivery needs an awake computer and a signed-in desktop. The task can catch up after a missed start or login. Windows notification settings and Do Not Disturb control whether the balloon appears. If your organization blocks registration, the planner displays the reason and keeps the in-app reminder. Uncheck the desktop option and save to remove the scheduled task. Dev histories do not send personal reminders. [Windows daily triggers](https://learn.microsoft.com/en-us/powershell/module/scheduledtasks/new-scheduledtasktrigger) · [Missed-start settings](https://learn.microsoft.com/en-us/powershell/module/scheduledtasks/new-scheduledtasksettingsset)

## Task workspace

Choose **Start** for the ordinary timer. For reading, writing and other work the desk supports, **Start with workspace** starts the same timer and opens a new tab. **Resume with workspace** continues a paused timer. Reopening a reader URL leaves the timer in its current state.

Workspace chat is saved with its task and document. When you pause after chatting, the timer stops immediately and Plan Smarter asks the configured forecast model to review your progress. For example, “I finished the proof and still need to check the counterexample” tells the model which part remains. The reader and task card show the remaining minutes at pause, the revised total, and the message supporting the change. If the conversation gives too little evidence or the model is unavailable, the previous forecast stays visible. A pause without new workspace chat only stops the timer.

An attached PDF opens in a reader beside a chat panel. Select a document when a task has several PDFs. The assistant receives text from the selected document with page numbers. For longer PDFs it searches every readable page for the question and includes passages from across the document. Answers about the PDF show page references with verified quotations. Image-only PDFs can be viewed but need an extractable text layer for document chat. For a text-based task, the desk offers a saved draft and a task chat. Each task has its own draft, and each attachment has its own chat in the current tab.

Kev checks whether the work can be done in this desk and whether it needs physical action or an external tool. With no attached PDF, it checks whether the task needs a particular document or live page. You can use the writing desk for work such as drafting a message from the details you entered. Editable Dev histories include those decisions for their open tasks and addable examples. For a new task, the planner tries local Kev and then the configured chat model. If both are unavailable, the ordinary timer works and **Retry workspace check** lets you try again.

## Estimates

Select a saved category when creating a task. Use **Options → Planner settings** to add your own categories. Task name, description, duration and category are required. Extra details and PDFs are optional. Select **Fixed duration** when an event occupies a known block on the calendar. The planner uses that scheduled length directly. Leave this off for meetings whose actual length varies.

The task is saved immediately. Its card shows **Estimating task time…** while PDFs are parsed and the model runs. You can start the timer or add another task. The range and **Why this estimate?** appear automatically when the forecast is available.

The model forecasts the minutes, planning range and confidence from the task description, optional details and PDF passages, plus relevant completed work retrieved from local Qdrant. **Why this estimate?** shows the model’s workload assumptions, the recorded budgets and durations it cited, and what could put the task at either end of the range. With no usable timing match, the card identifies the gap and the model explains its estimate from the task scope. If the request fails, the card identifies the fallback. Older forecasts identify which steps were performed by the planner where that information was recorded.

Each card lists the completed tasks selected for comparison and the ones the forecast actually used. Choose **Edit relevant tasks** to add or remove comparisons and re-estimate. Your corrections supply examples for later forecasts and relevance decisions.

**Stop → No, continue later** pauses the session. **Yes, finish task** records the actual duration and optional note immediately. The activity report appears on the completed card as soon as the worker finishes. Its activity table lets you correct classifications and regenerate the report. Later reports and estimates can use these corrections. Original estimates for newly created tasks run first. When two flexible tasks overlap on the planner, Kev judges the pair once in a stable order. If one finishes with recorded time, a positive decision sends its duration directly into the other task’s next forecast. An unrelated completion leaves that forecast alone. Your explicit comparison choices take precedence. The planner keeps your budget and its initial forecast for comparison. A **Watch for** preview on the card highlights an earlier observation. Expand **Why this estimate? → What to watch for** to see the relevant completion notes or ActivityWatch observations and the tasks they came from. When there is no supporting history, the section says so. Choose **Re-estimate** on any unfinished task with a flexible duration to request a new forecast using the current endpoint and completed history. While it runs, the card labels the previous forecast. Your original forecast and timer are preserved.

PDF processing uses readable text from every page and selects passages relevant to the task. Image-only scans can be attached but supply no text.

Use the trash icon on a card to remove it from the plan. The confirmation defaults to keeping its data in memory. Checking **Also delete this task’s data from memory** erases its planner sessions, files, report, vectors and learned feedback. You can also erase a previously hidden task under **Options → Planner settings**. The planner reads ActivityWatch’s source databases independently of its own storage.

## Demo and Dev

**Demo** opens your personal plan. In **Dev**, select one of the five mixed-workload demo profiles and a history view. The original ten benchmark histories remain available to the evaluation tools.

* **Editable history** opens a populated calendar with dated tasks, descriptions, notes, PDFs and timer sessions. Go back a day with the arrow or choose a date under **History details**. The day shows tasks added to the plan and tasks worked on that day. Open a card to inspect the synthetic ActivityWatch intervals. Historical reports in this starting view use the fixture's reviewer labels. You can add tasks, finish work and revise classifications. Existing sample annotations refresh when you open a profile. **Reset sample history** restores the whole sample and discards edits in that copy.
* **Chronological replay** starts a separate copy with an empty planner. **Apply next action** or **Run to end** advances the schedule through real task creation, timers, model forecasts, completion and report generation. Each task's forecast history shows the initial prediction and subsequent revisions, including the evidence the model had at that time. You can leave and resume a replay, or restart it without affecting the editable calendar.

SQLite stores the tasks, attached PDFs, completion notes, work sessions, forecasts and activity reports. Qdrant stores the searchable local task index. Set `PLANNER_DATA_DIR` to choose a different data directory.

## Analytics & logs

Open **View → Analytics & logs** in either workspace. Choose completion dates and categories to compare your budgets or the original model forecasts with recorded timer time. The category chart shows the share of underestimates, overestimates and exact forecasts. Switch to **By task** and hover over a point to see its estimate, actual duration and error. The dashed line marks an exact forecast. The model view includes tasks with a recorded model-generated creation forecast.

Under **Activity logs**, explore activity by day and the sources of labeled distractions. Filter by date, task, device or classification. Each interval identifies contributing laptop or phone activity where available. Overlapping observations count once. Intervals spanning midnight are apportioned across the two days.

The chat at the bottom of the page shares a conversation between **Daily plan** and **Analytics & logs**. Ask “What distracts me during GPU programming?” or “I have 40 minutes before class. What could I finish, and what should I postpone?” Both pages retain the selected history filters. Changing the filters or erasing a task clears the previous conversation.

For analytics, the configured model selects a scoped metric or writes a read-only SQL query. It interprets the resulting records, including empty results. Expand the supporting data to see the model, scope, SQL and rows. For planning, the model ranks unfinished tasks using the selected date, task details and filtered historical accuracy. Python subtracts recorded progress from each estimate and checks the cumulative fit. You can specify available minutes or a local cutoff. Without an explicit budget, the answer shows the clock time up to midnight. Tasks that have already exhausted their estimate need a new remaining-time estimate. The suggestions leave the actual schedule in your control.

## Evaluate the models

Score the same unseen tasks against the original estimate, a category baseline, the offline planner, and the model-backed planner. Predeclared ranges check whether a forecast learns the expected personal correction. The original and enriched suites are documented separately. [Histories and commands](data/tasks/README.md) · [Benchmark results](data/results/README.md) · [0.50 vs. 0.70 and the pre-Qwen/Kev estimator](data/results/holdouts/2026-11-15-command-r7b-050/) · [Instructed vs. plain Qwen queries at 0.50](data/results/holdouts/2026-11-15-qwen-query-instruction/)

## Tests

From the repository root, run:

```sh
python -m pip install pytest
python -m pytest -q
```

The [component evaluation](data/results/component-audit/) includes controlled input changes, live local model results, false positives and false negatives, and end-to-end scheduling scenarios.

To run just the scheduling scenarios, use `python -m pytest -q tests/test_concurrent_planning.py tests/test_packaged_dev_schedule.py`. These cover rapid submissions while a forecast is running, switching active timers, a completion that triggers selective re-estimates, and chronological replays of the packaged histories. The replay tests use the packaged Qwen and Kev outputs, disable local model calls, and simulate the hosted endpoint for forecasts and activity reports. The [paired forecast evaluation](data/results/holdouts/2026-11-15-qwen-query-instruction/) records the live model calls used to measure estimation quality.
