# Packaged sample retrieval

This directory contains one numeric CSV per public task and one JSON index per sample history. This lets an editable Dev workspace or chronological replay run its retrieval and relevance checks on a machine with only the final forecast API configured. A sample task can receive a different forecast each time it is added. The app uses the completed tasks available **at that point in the schedule** and sends the selected history to the configured forecaster.

Each task has a 1,024-dimensional Qwen3-Embedding-0.6B vector for use as a historical document and another for use as a new query. Open `<history-id>/<task-id>.csv` to inspect the two columns, `document` and `query`. Each of its 1,024 data rows holds the components at the same dimension. The numeric text round-trips to the original IEEE-754 binary32 output for both unit-length vectors. The per-history JSON index records each role’s exact input SHA-256 and the public source revision and SHA-256 of its history, schedule and referenced PDFs. It contains no PDF or task description text. Only the query receives the following instruction, in [Qwen’s retrieval format](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B#usage).

```text
Instruct: Retrieve past tasks whose work, methods, and scope are comparable and whose durations could help estimate this task.
Query: <combined title, description, details and extracted PDF text>
```

Each Kev-4B record specifies the target task, candidate task and boolean relevance decision. Coverage includes candidates reaching the 0.50 floor for either query style, concurrently open pairs from the dated schedules, and the open tasks and holdouts available to try in the dashboard. Matrix jobs always present the newer-created task to Kev as the target, including when it finishes before the worker can judge the pair. Repeated additions of the same sample have genuine pair decisions too. Hits are keyed by hashes of the complete model inputs. New workspace task IDs, dates and outcomes can change freely. Changing task text, the configured model or the judge’s examples requires a fresh model response for the affected request. A custom task with unavailable local models still gets a forecast from the configured forecast model using the evidence available to it.

There are 119 distinct sample tasks across the ten histories and 238 vectors. The current schedules, both variants of the query-instruction experiment, and editable sample additions require 351 distinct ordered Kev requests at the supported floor. Some bundles also retain earlier genuine decisions with the reverse task order. The app uses a record only when the entire request hash matches. The package includes a decision for a duplicate addition of each of the twenty optional tasks.

The sample vectors and decisions were generated locally with Qwen Q8_0 and Kev Q4_K_M. Lookups use the task-scope text hash or complete pair-decision request hash. The history database continues to determine whether a particular task is eligible for retrieval. Only eligible completed tasks from the active workspace can contribute timing comparisons.

## Regenerate

Install Qwen and Kev from the [local setup](../../../README.md#run-locally-on-an-apple-silicon-mac). For the build commands below, set the embedding and decision endpoints and model IDs for your server in the repository’s `.env`. [Configuration example](../../../.env.example) Fetch the public repository’s `main` branch and run both stages from its checkout with the published sample inputs. The builder verifies those files against `origin/main` before calling the models. To rebuild against an older published commit, add `--public-revision <commit>` to both commands and use the matching fixture files. These commands write a separate output directory for reviewing a build.

```sh
python -m examples.estimation.package_retrieval data/results/local/retrieval-build --stage vectors
python -m examples.estimation.package_retrieval data/results/local/retrieval-build --stage decisions
```

The decisions stage checkpoints each model response. Rerun it in the same directory to resume. The vectors stage rebuilds the vectors and clears the old decisions in its output directory.

To verify an export directly against the original public input revision, run the following command.

```sh
python -m examples.estimation.package_retrieval data/tasks/retrieval --stage verify --public-revision 2a0eda84ca4a3725933373b28f296cf50335f5dd
```
The verifier checks Git blob contents, all 238 embedding inputs, every numeric component, every complete Kev request hash and required pair coverage. This check makes no model calls.
