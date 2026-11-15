# Plan Smarter

A local, daily-only planner for comparing the time you set aside with the time your work actually takes. Completed tasks become examples for future estimates. The interface and application logic run in Python.

## Run on Windows

Use Python 3.11 or newer. After cloning the repository, open its directory in PowerShell and run:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py --server.address 127.0.0.1
```

If an existing install reports `No module named dotenv`, run the pip command above with the same virtual environment you use to start Streamlit. The package is named `python-dotenv`.

On Linux or macOS, use `.venv/bin/python` for the last two commands. Open the local URL printed by Streamlit. The planner can record tasks and sessions before the optional services below are available.

## Choose the model endpoint

For Cohere, keep your existing `COHERE_API` key in the repository-root `.env` and add these two lines:

```dotenv
PLANNER_MODEL_ENDPOINT=https://api.cohere.ai/compatibility/v1/chat/completions
PLANNER_MODEL=command-r7b-12-2024
```

The hosted configuration uses **Command R7B** for estimates and activity classification. Command R7B and LM Studio use the same OpenAI-compatible structured request. The client checks each reply against the requested JSON Schema. [Cohere Command R7B](https://docs.cohere.com/v2/docs/command-r7b)

After pulling this update, install the updated requirements using your existing virtual environment and restart Streamlit. In **Options → Planner settings**, check the model and host. Create a task and expand **Why this estimate?** to see its source and explanation. To see a forecast with personal timing evidence using only your Cohere key, choose **Dev → Editable history**, open **History details**, pick **Try a sample task** and click **Add sample task**. The package includes the PDFs, Qwen vectors and Kev decisions for those examples. Choose **Re-estimate** to request a fresh Cohere forecast.

Task records, originals of PDFs and ActivityWatch logs are stored locally. Selecting a hosted endpoint sends relevant task descriptions, PDF text excerpts, relevant completion notes, and filtered activity observations to that provider for inference. No API key is needed for LM Studio.

### Local LM Studio

For local inference, remove the endpoint and model overrides or set them to `http://127.0.0.1:1234/v1/chat/completions` and `ternary-bonsai-4b`. The same timer, estimator, and activity review work with either configuration.

1. Install [LM Studio](https://lmstudio.ai/download) or its [headless Linux daemon](https://lmstudio.ai/docs/developer/core/headless). For headless Linux the official install command is `curl -fsSL https://lmstudio.ai/install.sh | bash`, followed by `lms daemon up`.
2. Download [Ternary-Bonsai-4B-Q2_0_g64.gguf](https://huggingface.co/prism-ml/Ternary-Bonsai-4B-gguf/blob/main/Ternary-Bonsai-4B-Q2_0_g64.gguf). The `_g64` suffix selects the group-64 layout compatible with the stock llama.cpp CPU engine. Import the file with `lms import <path-to-the-gguf>`. Use `lms ls` to find its model key.
3. Load it on CPU and start the local endpoint:

```sh
lms load <model-key> --gpu off --context-length 8192 --identifier ternary-bonsai-4b
lms server start --bind 127.0.0.1 --port 1234
```

The request timeout defaults to 90 seconds. Set `PLANNER_MODEL_TIMEOUT_SECONDS=360` in your .env if CPU inference needs longer.

LM Studio and the planner must run on the same host for this configuration. The initial software and model downloads require internet access.

### Local relevance models

The planner uses Qwen3-Embedding-0.6B to search task history and Kev-4B to judge each candidate. The ten bundled profiles include results for their sample tasks. New task text, a changed document, or new relevance feedback can require a fresh local model call. If those models are unavailable, Cohere can estimate from the task description and any completed comparisons you select under **Edit relevant tasks**. To enable automatic retrieval for your own tasks, run both local models through LM Studio. Install a version with the [System One API](https://lmstudio.ai/docs/developer/jev-compat/systemone) and an updated llama.cpp runtime.

Download [Qwen3-Embedding-0.6B-Q8_0.gguf](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B-GGUF/blob/370f27d7550e0def9b39c1f16d3fbaa13aa67728/Qwen3-Embedding-0.6B-Q8_0.gguf) and [Kev-4B-Q4_K_M.gguf](https://huggingface.co/ggml-org/Kev-4B-GGUF/blob/f4624b6bb7072f1bab726a622c1cad3f96030b9e/Kev-4B-Q4_K_M.gguf). Run the following commands from the folder containing both downloads.

```sh
lms daemon up
lms import Qwen3-Embedding-0.6B-Q8_0.gguf --user-repo Qwen/Qwen3-Embedding-0.6B-GGUF -y
lms import Kev-4B-Q4_K_M.gguf --user-repo ggml-org/Kev-4B-GGUF -y
lms load text-embedding-qwen3-embedding-0.6b --gpu off --context-length 8192 --identifier qwen3-embedding-0.6b
lms load kev-4b --gpu off --context-length 4096 --identifier kev-4b
lms server start --bind 127.0.0.1 --port 1234
```

The planner embeds each task’s title, description, details and extracted document text together. The search query carries a Qwen retrieval instruction about comparable work, method and scope. Historical documents use their plain task text. The local Qdrant index uses cosine similarity. Automatic retrieval sends each candidate with a cosine score of at least 0.50 to Kev. You can raise this cutoff in `.env`. Category breaks tied cosine scores. The [threshold evaluation](data/results/retrieval/2026-11-15-qwen06b-kev4b/README.md) records the cosine sweep and the results for the earlier 0.70 cutoff. Check `.env.example` for local endpoint and threshold overrides.

## ActivityWatch

Run [ActivityWatch](https://activitywatch.net/) on the same computer. Enable its window and AFK watchers. Install its browser extension for active-tab details. On Windows the planner checks the local ActivityWatch database and the synced stores listed below. If it cannot find a desktop store automatically, it reads the local ActivityWatch API on port 5600. Pause or Stop closes the timed period immediately. The **Finished?** prompt determines whether to add that task to estimation history.

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

On a task card, open **Websites to block during this task** to see the sites suggested by past activity. One off-task visit on a related task is enough for Kev to review the domain. For example, it can suggest blocking a chess puzzle during another proof-writing task. The card shows where the visit occurred and for how long. Repeated off-task use across related tasks also qualifies. A site used for work in the same comparisons or named in the new task stays available unless you choose to block it. Review the evidence, remove a suggestion or add a domain. Once you pair the browser, both **Start** and **Start with workspace** apply that task’s saved list. Pausing releases the sites and restores already-open tabs. Switching tasks applies the next task’s list.

Pair the extension once per Chrome or Edge profile under **Options → Planner settings → Browser blocking**. This works within the browser and requires no administrator access. [Install, pair and test the extension](browser_extension/README.md)

## Daily review and postponing work

On an unfinished task card, **Postpone** moves the task to tomorrow or a later date you choose. For a task already planned in the future, **Next day** means the day after its scheduled date. Postponing a running task pauses its timer immediately. You can resume it on its new date. Time already tracked remains visible on the days you worked, and the task keeps its estimate, attachments, and comparisons.

In **Demo → Options → Planner settings → Daily review reminder**, set a local time or turn reminders off. The default is **9:30 PM** on the computer running Plan Smarter. The live planner checks every 30 seconds while it is open. **Review this day** opens the plan, where you can finish tasks or postpone them. A dismissed reminder stays dismissed across tabs and restarts. A late opening can catch up on yesterday’s plan. Empty days are skipped.

On Windows, enable **Notify my Windows desktop even when the planner is closed** and save. Plan Smarter registers a daily task under your signed-in Windows account at limited privileges. The browser and Streamlit can then both be closed. Keep the project folder and Python environment in place. Delivery needs an awake computer and a signed-in desktop. The task can catch up after a missed start or login. Windows notification settings and Do Not Disturb control whether the balloon appears. If your organization blocks registration, the planner displays the reason and keeps the in-app reminder. Uncheck the desktop option and save to remove the scheduled task. Dev histories do not send personal reminders. [Windows daily triggers](https://learn.microsoft.com/en-us/powershell/module/scheduledtasks/new-scheduledtasktrigger) · [Missed-start settings](https://learn.microsoft.com/en-us/powershell/module/scheduledtasks/new-scheduledtasksettingsset)

## Task workspace

Choose **Start** for the ordinary timer. For reading, writing and other work the desk supports, **Start with workspace** starts the same timer and opens a new tab. **Resume with workspace** continues a paused timer. Reopening a reader URL leaves the timer in its current state.

Workspace chat is saved with its task and document. When you pause after chatting, the timer stops immediately and Plan Smarter asks the configured forecast model to review your progress. For example, “I finished the proof and still need to check the counterexample” tells the model which part remains. The reader and task card show the remaining minutes at pause, the revised total, and the message supporting the change. If the conversation gives too little evidence or the model is unavailable, the previous forecast stays visible. A pause without new workspace chat only stops the timer.

An attached PDF opens in a reader beside a chat panel. Select a document when a task has several PDFs. The assistant receives text from the selected document with page numbers. For longer PDFs it searches every readable page for the question and includes passages from across the document. Answers about the PDF show page references with verified quotations. Image-only PDFs can be viewed but need an extractable text layer for document chat. For a text-based task, the desk offers a saved draft and a task chat. Each task has its own draft, and each attachment has its own chat in the current tab.

Kev checks whether the work can be done in this desk and whether it needs physical action or an external tool. Editable Dev histories include those decisions for their open tasks and addable examples. For a new task, the planner tries local Kev and then the configured chat model. If both are unavailable, the ordinary timer works and **Retry workspace check** lets you try again.

## Estimates

Select a saved category when creating a task. Use **Options → Planner settings** to add your own categories. Task name, description, duration and category are required. Extra details and PDFs are optional. Select **Fixed duration** when an event occupies a known block on the calendar. The planner uses that scheduled length directly. Leave this off for meetings whose actual length varies.

The task is saved immediately. Its card shows **Estimating task time…** while PDFs are parsed and the model runs. You can start the timer or add another task. The range and **Why this estimate?** appear automatically when the forecast is available.

The model forecasts the minutes, planning range and confidence from the task description, optional details and PDF passages, plus relevant completed work retrieved from local Qdrant. **Why this estimate?** shows the model’s workload assumptions, the recorded budgets and durations it cited, and what could put the task at either end of the range. With no usable timing match, the card identifies the gap and the model explains its estimate from the task scope. If the request fails, the card identifies the fallback. Older forecasts identify which steps were performed by the planner where that information was recorded.

Each card lists the completed tasks selected for comparison and the ones the forecast actually used. Choose **Edit relevant tasks** to add or remove comparisons and re-estimate. Your corrections supply examples for later forecasts and relevance decisions.

**Stop → No, continue later** pauses the session. **Yes, finish task** records the actual duration and optional note immediately. The activity report appears on the completed card as soon as the worker finishes. Its activity table lets you correct classifications and regenerate the report. Later reports and estimates can use these corrections. Original estimates for newly created tasks run first. A completion triggers automatic refreshes only for unfinished tasks the pairwise decision model found related. The planner keeps your budget and its initial forecast for comparison. A **Watch for** preview on the card highlights an earlier observation. Expand **Why this estimate? → What to watch for** to see the relevant completion notes or ActivityWatch observations and the tasks they came from. When there is no supporting history, the section says so. Choose **Re-estimate** on any unfinished task with a flexible duration to request a new forecast using the current endpoint and completed history. While it runs, the card labels the previous forecast. Your original forecast and timer are preserved.

PDF processing uses readable text from every page and selects passages relevant to the task. Image-only scans can be attached but supply no text.

Use the trash icon on a card to remove it from the plan. The confirmation defaults to keeping its data in memory. Checking **Also delete this task’s data from memory** erases its planner sessions, files, report, vectors and learned feedback. You can also erase a previously hidden task under **Options → Planner settings**. The planner reads ActivityWatch’s source databases independently of its own storage.

## Demo and Dev

**Demo** opens your personal plan. In **Dev**, select one of ten profiles and a history view.

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

To run just the scheduling scenarios, use `python -m pytest -q tests/test_concurrent_planning.py tests/test_packaged_dev_schedule.py`. These cover rapid submissions while a forecast is running, switching active timers, a completion that triggers selective re-estimates, and chronological replays of all ten histories. The replay tests use the packaged Qwen and Kev outputs, disable local model calls, and simulate the hosted endpoint for forecasts and activity reports. The [paired forecast evaluation](data/results/holdouts/2026-11-15-qwen-query-instruction/) records the live model calls used to measure estimation quality.
