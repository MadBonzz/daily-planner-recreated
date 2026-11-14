# Daily Planner

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

The hosted configuration uses **Command R7B** for estimates and activity classification. [Cohere Command R7B](https://docs.cohere.com/v2/docs/command-r7b)

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

## ActivityWatch

Run [ActivityWatch](https://activitywatch.net/) on the same computer. Enable its window and AFK watchers. Install its browser extension for active-tab details. The planner reads stored events from the local API on port 5600. Pause or Stop closes the timed period immediately. The **Finished?** prompt determines whether to add that task to estimation history.

Starting a different task automatically pauses the previous one. Choosing **Continue later** after Stop saves the session and keeps the task open for another day.

The elapsed time used for planning is the sum of running periods across sessions. It includes interruptions during a running timer and excludes explicit pauses and gaps between sessions. The activity report gives a separate, approximate breakdown. Missing or ambiguous evidence is shown as uncertain. ActivityWatch manages its own raw event storage and retention.

## Estimates

Select a saved category when creating a task. Use **Options → Planner settings** to add your own categories. Task name, description, duration and category are required. Extra details and PDFs are optional. Select **Fixed duration** when an event occupies a known block on the calendar. The planner uses that scheduled length directly. Leave this off for meetings whose actual length varies.

The task is saved immediately. Its card shows **Estimating task time…** while PDFs are parsed and the model runs. You can start the timer or add another task. The range and **Why this estimate?** appear automatically when the forecast is available.

With no timed history, the model estimates independently from the description, details and relevant PDF passages. A failed request falls back to your budget with a ±50% range. With completed history, local Qdrant retrieves up to six candidates using hashed words and word pairs. The model compares the work involved. If its response reaches the token limit, the planner makes one shorter request for a decision per candidate. Python uses the selected tasks’ actual-to-estimated ratios to adjust the new budget. Intervals and confidence labels are planning heuristics.

When finishing, optionally record what affected the time. The completion adds its actual duration, note and filtered activity report to the history for future estimates. Open tasks can then receive an updated forecast. The planner retains your budget and its first forecast for comparison. A caution on a card identifies the earlier tasks and the specific observation behind it.

PDF processing uses readable text from every page and selects passages relevant to the task. Image-only scans can be attached but supply no text.

## Demo and Dev

**Demo** opens your daily planner, including tasks you have already saved. **Dev** lets you choose any of ten independent synthetic histories. The current day includes unfinished work. Open **History details** to inspect past tasks, their dates and recorded durations. Add a new task, finish an existing one, or switch profiles and return later. **Reset sample history** restores only the selected sample. Dev replays synthetic ActivityWatch data and keeps your personal activity separate.

SQLite stores the tasks, attached PDFs, completion notes, work sessions, forecasts and activity reports. Qdrant stores the searchable local task index. Set `PLANNER_DATA_DIR` to choose a different data directory.

## Evaluate the models

Score the same unseen tasks against the original estimate, a category baseline, the offline planner, and the model-backed planner. Predeclared ranges check whether a forecast learns the expected personal correction. The original and enriched suites are documented separately. [Histories and commands](data/tasks/README.md) · [Benchmark results](data/results/README.md)

## Tests

From the repository root, run:

```sh
python -m pip install pytest
python -m pytest -q
```
