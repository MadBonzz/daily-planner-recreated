# Test histories for the estimator

Ten independent synthetic histories contain 77 past tasks, two later tasks per history, and 20 original PDF attachments. All dates, prompts and outcomes are fictional. The two course-style histories use original problems and lab briefs.

## Try one in the planner

From the repository root, with your usual model settings:

```powershell
.\.venv\Scripts\python.exe -m examples.estimation.run open 02
```

The command opens an isolated profile at port 8502, recreates the earlier tasks and work sessions, and places the two unstarted test tasks on today's date. Open **Why this estimate?** on each task. They are about the same migration project: one calls for implementation and regression tests, the other for a write-up of saved results. The implementation PDF includes scope on its second page. Personal tasks and this temporary profile are stored separately. Stopping the command discards changes to the trial.

## Score the estimates

```powershell
.\.venv\Scripts\python.exe -m examples.estimation.run list
.\.venv\Scripts\python.exe -m examples.estimation.run evaluate 02 --case patch-tool
.\.venv\Scripts\python.exe -m examples.estimation.run evaluate --strict-model --json data/results/local/cohere-evaluation.json
```

Each test task uses only earlier tasks from the same history. The second test in a history sees the same previous records as the first. Its sibling's outcome, both test-task timings, and the author's test notes are withheld from prediction. Each full model run can make 20 hosted requests, one per case. The JSON includes the original budget, category baseline, planner estimate, actual duration, interval, selected evidence, and model status. Look for `model_status: ok` on the cases you intend to use for measuring model-assisted performance. Failed calls are scored using the application's fallback and identified in the output. With `--strict-model`, an unavailable model also produces a nonzero exit status after the report is saved.

The category baseline multiplies the user's estimate by the median actual-to-estimated ratio for earlier tasks in the same category. It uses the original estimate for a new category. The summary reports mean absolute error in minutes, the number of intervals covering the actual outcome, and average interval width. These fixtures test how the implementation behaves; real-world accuracy requires real completed tasks.

| ID | Prior tasks | What the history tests |
|---|---:|---|
| 01 | 8 | A new distributed safety proof compared with tracing an execution from the same handout |
| 02 | 7 | Implementing a C-to-Rust migration tool compared with reporting existing results |
| 03 | 8 | Deriving equations in a paper compared with reading an empirical evaluation |
| 04 | 8 | Debugging a GPU race, running scripted sweeps, and an already accurate estimate |
| 05 | 8 | Responding to open-ended design critique, routine edits and an unexpectedly long export |
| 06 | 8 | Cleaning inconsistent library exports compared with refreshing known reports |
| 07 | 7 | Adapting bilingual museum text for a new audience compared with proofreading approved text |
| 08 | 6 | Multi-visit repairs, routine inspections and time between sessions |
| 09 | 9 | Designing a new math lesson compared with adapting one for a returning cohort |
| 10 | 8 | A sparse personal backlog with weak comparisons and unfamiliar categories |

## Offline reference run

Add `--offline` to `evaluate` to deliberately exercise the fallback without any model calls. Across all 20 fixtures, mean absolute error was 34.45 minutes for the original budgets, 28.70 minutes for the category baseline and 31.35 minutes for the planner fallback. The fallback ranges covered 16 outcomes, with a mean width of 69.35 minutes. Run the same cases with your configured model to measure its effect on comparison selection.

The histories and hidden answers are editable JSON in `histories/`. Use `python -m examples.estimation.run validate` after making changes. The PDFs are ready to use in `pdfs/`. To rebuild them from the document specifications, install `reportlab` and run `python data/tasks/render_pdfs.py`.
