# Estimator benchmark results

Each suite has a report comparing the same held-out tasks across runs. Each run has its own directory with case-level predictions, timings and response status. Model-server retries are recorded with the run that issued them. Standalone probes have separate run directories.

| Suite | Cases | Reports and records |
| --- | ---: | --- |
| Chronological clinic replay · Cohere, Qwen3 and Kev | 40 actions, 9 completed tasks | [Run, forecast changes and accuracy](replay/2026-11-15-clinic-command-r7b-070/) |
| Local task relevance · Qwen3 and Kev | 19 holdouts, 150 within-profile pairs | [Threshold calibration and held-out decisions](retrieval/2026-11-15-qwen06b-kev4b/) |
| V3 · Model-generated point, range, confidence and warnings | 8 Cohere checks, 20 offline cases | [Report and run status](v3/README.md) · [Live Cohere checks](v3/runs/2026-11-14-cohere-direct-release/) |
| V2 · Frozen hybrid estimator, mixed daily histories, completion notes and ActivityWatch evidence | 20 in 10 histories | [Comparison report](v2/README.md) · [Runs](v2/runs/) |
| V1 · Original completed-task histories and PDFs | 20 in 10 histories | [Comparison report](v1/README.md) · [Paired case table](v1/cases.csv) · [Runs and probes](v1/runs/) |

[SQLite-based fallback and caution check](v2/runs/2026-11-14-sqlite-offline/README.md)

To score the current version, use the evaluation commands in the [sample history guide](../tasks/README.md). Give each new run its own directory, for example `data/results/local/trial-01/results.json`. The app loads sample inputs from `data/tasks/`.
