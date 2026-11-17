# Plan Smarter

Plan Smarter estimates task duration from your description, attachments and similar work you’ve completed. Record the actual time, and the planner updates forecasts for related tasks. The app runs in Streamlit.

## Quick start

Install Python 3.12, then clone the repository.

```sh
git clone https://github.com/MadBonzz/daily-planner-recreated.git
cd daily-planner-recreated
```

**Windows (PowerShell)**

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
if (!(Test-Path .env)) { Copy-Item .env.example .env }
```

**macOS or Linux**

```sh
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
test -f .env || cp .env.example .env
```

### Run with Cohere

Set `COHERE_API` in `.env` to your key. The default setup uses bundled Qwen embeddings and Kev decisions for matching tasks. Cohere generates forecasts and answers model-backed chat questions.

| Platform | Start the app from the repository |
| --- | --- |
| Windows (PowerShell) | `.\.venv\Scripts\python.exe -m streamlit run app.py` |
| macOS or Linux | `.venv/bin/python -m streamlit run app.py` |

Open the local address Streamlit prints, usually <http://localhost:8501>. A hosted request can send relevant task text, document passages and activity records to the configured provider.

### Run locally on an Apple Silicon Mac

Download these models in LM Studio Bionic and start its local server on port **1235**.

| Model | Download | Used for |
| --- | --- | --- |
| Qwen3 Embedding 0.6B | `Qwen/Qwen3-Embedding-0.6B-GGUF`, Q8_0 | Finding similar task descriptions |
| Kev 4B | `ggml-org/Kev-4B-GGUF`, Q4_K_M | Deciding which tasks are related |
| Ternary Bonsai 2 4B | MLX model with key `ternary-bonsai-2-4b-mlx` | Forecasts, activity classification and chat |

Choose one launch command. To switch modes, stop Streamlit with **Ctrl+C** and run the other command.

| Mode | Command |
| --- | --- |
| Bundled Qwen and Kev results first, local Bonsai forecasts | `bash run-bionic-mac.sh --precomputed` |
| Live Qwen, Kev and Bonsai | `bash run-bionic-mac.sh` |

Use a dedicated Bionic server. The launcher automatically swaps one model at a time. New or edited inputs can need live Qwen or Kev. If live retrieval fails, an exact bundled match can supply the result. Both modes share saved tasks. For Cohere, stop the launcher and use the Streamlit command above.

## Try a sample history

**Demo** opens your personal planner. **Dev → Editable history** opens one of five populated calendars. Edits survive a restart. **History details → Reset sample history** restores the selected sample.

1. Open **Nora · distributed systems lab** and find **Attach the minimized trace to the lab issue**. Open **Why this estimate?** and note its estimate and cited tasks.
2. Mark **Pick up dish soap** complete and enter **20 minutes**. The trace task’s estimate should stay the same.
3. Mark **Minimize the marker-word failure** complete and enter **110 minutes**. Check the trace task again for the new comparison and forecast. The model determines the revised minutes.
4. Open **Analytics & logs** and ask “Which websites distract me most?” Expand **Query and supporting data** to inspect the records.

For a step-by-step replay, choose **Chronological replay → Apply next action**. [Sample histories](data/tasks/demo/) · [Latest walkthrough and observed results](data/results/latest.json)

## Working with tasks

- **Create and edit.** Use **Add task** or attach a PDF with **+** and send “Add my Paxos assignment for tomorrow. I expect four hours. Attach this PDF.” Wait for the upload to finish before sending. To revise it, use **Task details → Edit task** or ask in chat. “Postpone that task to Friday” keeps its files and recorded time.
- **Track time.** Start, pause or resume the timer. Use **Mark complete** to enter the total and an optional note.
- **Inspect estimates.** **Why this estimate?** shows the range and historical comparisons. **Edit relevant tasks** changes those comparisons, and **Re-estimate** requests a fresh forecast.
- **Review activity.** The sample histories include synthetic ActivityWatch logs. For personal tracking, run ActivityWatch and turn on **Options → Planner settings → Read ActivityWatch during timed work**. The optional Chrome or Edge extension can block selected sites while a task is running. [Browser setup and pairing](browser_extension/README.md)

Tasks, attachments and sessions are saved in SQLite. Qdrant stores the local task index. Ordinary launches use the repository’s `data/` directory. The Mac launcher uses `~/Library/Application Support/Plan Smarter/lmstudio-live`. Set `PLANNER_DATA_DIR` to choose a different location. [Configuration options](.env.example)
