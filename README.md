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

After pulling this update, install the updated requirements using your existing virtual environment and restart Streamlit. In **Options → Planner settings**, check the model and host. Create a task and expand **Why this estimate?** to see its source and explanation. To try a prediction based on earlier work, choose **Dev**, select a sample history and add a related task.

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

Even with a hosted forecast model, the planner uses Qwen3-Embedding-0.6B to search task history and Kev-4B to decide whether each candidate is comparable. Both run through LM Studio on the planner's machine. Install a version with the [System One API](https://lmstudio.ai/docs/developer/jev-compat/systemone) and an updated llama.cpp runtime.

Download [Qwen3-Embedding-0.6B-Q8_0.gguf](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B-GGUF/blob/370f27d7550e0def9b39c1f16d3fbaa13aa67728/Qwen3-Embedding-0.6B-Q8_0.gguf) and [Kev-4B-Q4_K_M.gguf](https://huggingface.co/ggml-org/Kev-4B-GGUF/blob/f4624b6bb7072f1bab726a622c1cad3f96030b9e/Kev-4B-Q4_K_M.gguf). Run the following commands from the folder containing both downloads.

```sh
lms daemon up
lms import Qwen3-Embedding-0.6B-Q8_0.gguf --user-repo Qwen/Qwen3-Embedding-0.6B-GGUF -y
lms import Kev-4B-Q4_K_M.gguf --user-repo ggml-org/Kev-4B-GGUF -y
lms load text-embedding-qwen3-embedding-0.6b --gpu off --context-length 8192 --identifier qwen3-embedding-0.6b
lms load kev-4b --gpu off --context-length 4096 --identifier kev-4b
lms server start --bind 127.0.0.1 --port 1234
```

The planner requests one embedding for the combined title, description, details and extracted document text. The local Qdrant index uses cosine similarity. Automatic retrieval sends each candidate with a cosine score of at least 0.70 to Kev. You can raise this cutoff in `.env`. Category breaks tied cosine scores. The [threshold evaluation](data/results/retrieval/2026-11-15-qwen06b-kev4b/README.md) records the held-out results at 0.70 and the earlier exploratory cutoff sweep. Check `.env.example` for local endpoint and threshold overrides.

## ActivityWatch

Run [ActivityWatch](https://activitywatch.net/) on the same computer. Enable its window and AFK watchers. Install its browser extension for active-tab details. On Windows the planner checks the local ActivityWatch database and the synced stores listed below. If it cannot find a desktop store automatically, it reads the local ActivityWatch API on port 5600. Pause or Stop closes the timed period immediately. The **Finished?** prompt determines whether to add that task to estimation history.

| Windows source | Location checked automatically |
| --- | --- |
| Laptop ActivityWatch | `%LOCALAPPDATA%\activitywatch\aw-server-rust\sqlite.db` |
| Syncthing copy on the laptop | `%USERPROFILE%\Downloads\syncthing-windows-*\syncthing-windows-*\activity-watch-one-plus\<UUID>\test.db`, including the `<device>\<UUID>\test.db` layout |

The synced folder can contain either device's records. The planner identifies the Android foreground watcher by its bucket metadata. It reads SQLite in read-only mode, restricts each query to the timer's active periods, and applies exclusions to both devices. The classifier sees an interval ID, seconds, filtered laptop app, window or tab title, domain, laptop idle status and the concurrent Android app when available. Overlapping observations count once. An app in the foreground alone cannot establish what someone was doing. A classifier can mark conflicting signals uncertain.

For a different installation, set `PLANNER_ACTIVITYWATCH_DB` for the laptop database or `PLANNER_ACTIVITYWATCH_PHONE_DB` for a synced store with an Android watcher in the repository-root `.env`. Use literal paths, for example `D:/ActivityWatchSync/phone/test.db`. If a desktop database contains several computers, `PLANNER_ACTIVITYWATCH_HOSTNAME` selects one. The per-task timeline shows **Phone app** when a matching Android event overlaps the task. Past classified intervals can surface in **Why this estimate?** for a related task.

Starting a different task automatically pauses the previous one. Choosing **Continue later** after Stop saves the session and keeps the task open for another day.

The elapsed time used for planning is the sum of running periods across sessions. It includes interruptions during a running timer and excludes explicit pauses and gaps between sessions. The activity report gives a separate, approximate breakdown. Missing or ambiguous evidence is shown as uncertain. ActivityWatch manages its own raw event storage and retention.

## Estimates

Select a saved category when creating a task. Use **Options → Planner settings** to add your own categories. Task name, description, duration and category are required. Extra details and PDFs are optional. Select **Fixed duration** when an event occupies a known block on the calendar. The planner uses that scheduled length directly. Leave this off for meetings whose actual length varies.

The task is saved immediately. Its card shows **Estimating task time…** while PDFs are parsed and the model runs. You can start the timer or add another task. The range and **Why this estimate?** appear automatically when the forecast is available.

The model forecasts the minutes, planning range and confidence from the task description, optional details and PDF passages, plus relevant completed work retrieved from local Qdrant. **Why this estimate?** shows its rationale, confidence explanation and any cited comparisons. If the request fails, the card identifies the fallback. Older forecasts identify which steps were performed by the planner where that information was recorded.

Each card lists the completed tasks selected for comparison and the ones the forecast actually used. Choose **Edit relevant tasks** to add or remove comparisons and re-estimate. Your corrections supply examples for later forecasts and relevance decisions.

**Stop → No, continue later** pauses the session. **Yes, finish task** records the actual duration and optional note immediately. The activity report appears on the completed card as soon as the worker finishes. Its activity table lets you correct classifications and regenerate the report. Later reports and estimates can use these corrections. Original estimates for newly created tasks run first. A completion triggers automatic refreshes only for unfinished tasks the pairwise decision model found related. The planner keeps your budget and its initial forecast for comparison. A **Watch for** preview on the card highlights an earlier observation. Expand **Why this estimate? → What to watch for** to see the relevant completion notes or ActivityWatch observations and the tasks they came from. When there is no supporting history, the section says so. Choose **Re-estimate** on any unfinished task with a flexible duration to request a new forecast using the current endpoint and completed history. While it runs, the card labels the previous forecast. Your original forecast and timer are preserved.

PDF processing uses readable text from every page and selects passages relevant to the task. Image-only scans can be attached but supply no text.

Use the trash icon on a card to remove it from the plan. The confirmation defaults to keeping its data in memory. Checking **Also delete this task’s data from memory** erases its planner sessions, files, report, vectors and learned feedback. You can also erase a previously hidden task under **Options → Planner settings**. The planner reads ActivityWatch’s source databases independently of its own storage.

## Demo and Dev

**Demo** opens your personal plan. In **Dev**, select one of ten profiles and a history view.

* **Editable history** opens a populated calendar with dated tasks, descriptions, notes, PDFs and timer sessions. Go back a day with the arrow or choose a date under **History details**. The day shows tasks added to the plan and tasks worked on that day. Open a card to inspect the synthetic ActivityWatch intervals. Historical reports in this starting view use the fixture's reviewer labels. You can add tasks, finish work and revise classifications. If you loaded this profile on an earlier version, **Reset sample history** replaces its saved edits with the new calendar.
* **Chronological replay** starts a separate copy with an empty planner. **Apply next action** or **Run to end** advances the schedule through real task creation, timers, model forecasts, completion and report generation. Each task's forecast history shows the initial prediction and subsequent revisions, including the evidence the model had at that time. You can leave and resume a replay, or restart it without affecting the editable calendar.

SQLite stores the tasks, attached PDFs, completion notes, work sessions, forecasts and activity reports. Qdrant stores the searchable local task index. Set `PLANNER_DATA_DIR` to choose a different data directory.

## Analytics & logs

Open **View → Analytics & logs** in either workspace. Choose completion dates and categories to compare your budgets or the original model forecasts with recorded timer time. For each category, the table shows how often estimates were high, low or exact, and the average miss in minutes. An empty model column means that task has no recorded model-generated creation forecast.

Under **Activity logs**, select an activity day and task to inspect timestamps, window titles, apps, domains, classifications and their sources. Intervals spanning midnight are apportioned across the two days. **Ask about your history** answers questions such as “Which sites accounted for the most distraction time on my reading tasks?” The configured model proposes a read-only query over the filtered tasks and visible logs, then summarizes the result. Expand the supporting data to inspect the query and rows.

## Evaluate the models

Score the same unseen tasks against the original estimate, a category baseline, the offline planner, and the model-backed planner. Predeclared ranges check whether a forecast learns the expected personal correction. The original and enriched suites are documented separately. [Histories and commands](data/tasks/README.md) · [Benchmark results](data/results/README.md)

## Tests

From the repository root, run:

```sh
python -m pip install pytest
python -m pytest -q
```
