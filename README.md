# Daily Planner

A local, daily-only planner for comparing the time you set aside with the time your work actually takes. Completed tasks become examples for future estimates. The interface and application logic run in Python.

## Run on Windows

Use Python 3.11 or newer. After cloning the repository, open its directory in PowerShell and run:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py --server.address 127.0.0.1
```

On Linux or macOS, use `.venv/bin/python` for the last two commands. Open the local URL printed by Streamlit. The planner can record tasks and sessions before the optional services below are available.

## Choose the model endpoint

For Cohere, keep your existing `COHERE_API` key in the repository-root `.env` and add these two lines:

```dotenv
PLANNER_MODEL_ENDPOINT=https://api.cohere.ai/compatibility/v1/chat/completions
PLANNER_MODEL=command-r7b-12-2024
```

The hosted configuration uses **Command R7B** for estimates and activity classification. [Cohere Command R7B](https://docs.cohere.com/v2/docs/command-r7b)

After pulling this update, install the updated requirements using your existing virtual environment and restart Streamlit. In **Options → Planner settings**, check the model and host. To test a model-assisted estimate, choose **Example history**, load the sample tasks and create a related task. Expand **Why this estimate?** to see the selected comparisons or the reason the app used a fallback.

Task records, originals of PDFs and ActivityWatch logs are stored locally. Selecting a hosted endpoint sends relevant task descriptions, PDF text excerpts, and filtered activity observations to that provider for inference. No API key is needed for LM Studio.

### Local LM Studio

For local inference, remove the endpoint and model overrides or set them to `http://127.0.0.1:1234/v1/chat/completions` and `ternary-bonsai-4b`. The same timer, estimator, and activity review work with either configuration.

1. Install [LM Studio](https://lmstudio.ai/download) or its [headless Linux daemon](https://lmstudio.ai/docs/developer/core/headless). For headless Linux the official install command is `curl -fsSL https://lmstudio.ai/install.sh | bash`, followed by `lms daemon up`.
2. Download [Ternary-Bonsai-4B-Q2_0_g64.gguf](https://huggingface.co/prism-ml/Ternary-Bonsai-4B-gguf/blob/main/Ternary-Bonsai-4B-Q2_0_g64.gguf). The `_g64` suffix selects the group-64 layout compatible with the stock llama.cpp CPU engine. Import the file with `lms import <path-to-the-gguf>`. Use `lms ls` to find its model key.
3. Load it on CPU and start the local endpoint:

```sh
lms load <model-key> --gpu off --context-length 8192 --identifier ternary-bonsai-4b
lms server start --bind 127.0.0.1 --port 1234
```

LM Studio and the planner must run on the same host for this configuration. The initial software and model downloads require internet access.

## ActivityWatch

Run [ActivityWatch](https://activitywatch.net/) on the same computer. Enable its window and AFK watchers. Install its browser extension for active-tab details. The planner reads stored events from the local API on port 5600. Pause or Stop closes the timed period immediately. The **Finished?** prompt determines whether to add that task to estimation history.

Starting a different task automatically pauses the previous one. Choosing **Continue later** after Stop saves the session and keeps the task open for another day.

The elapsed time used for planning is the sum of running periods across sessions. It includes interruptions during a running timer and excludes explicit pauses and gaps between sessions. The activity report gives a separate, approximate breakdown. Missing or ambiguous evidence is shown as uncertain. ActivityWatch manages its own raw event storage and retention.

## Estimates

The new-task form requires an original time budget and a description. You can add a category, extra details, and PDFs. The parser extracts text from every readable PDF page, then selects page-labeled passages that match the task and cover different sections. Scanned PDFs can be attached but supply no text for estimation. Local Qdrant retrieves up to six completed tasks by cosine similarity using hashed words and word pairs. This needs no second embedding model. The configured model selects comparable work using the descriptions, notes and document excerpts. Python computes the median actual/original ratio and applies a smaller correction when there are few matches. If inference is unavailable, text similarity or category history provides a fallback. The app keeps the original estimate and its own prediction separately. A first task gets a broad planning range around the entered estimate. Ranges and confidence labels are heuristics until evaluated on real history.

## Local data and example history

Under **Options → Workspace**, choose **Example history** and click **Load example history** to try the estimator with five labeled examples. Choose **My planner** for personal tasks. These workspaces use separate SQLite databases and Qdrant indexes under `data/`. Set `PLANNER_DATA_DIR` to change the parent directory. Example tasks demonstrate how a prediction changes after completion. Their outcomes are illustrative. SQLite holds the task descriptions, original PDFs, extracted text, saved predictions, sessions, and completed reports. The vector index can be rebuilt from the task history. Keep this directory local.

## Test the estimator

Open a disposable profile with synthetic history or score 20 held-out tasks against their recorded outcomes. The fixture runner uses your configured model and reports when it falls back. [Test histories and Windows commands](data/tasks/README.md)

## Tests

From the repository root, run:

```sh
python -m pip install pytest
python -m pytest -q
```
