# Estimator benchmark results

Each suite has a report comparing the same held-out tasks across runs. Each run has its own directory with case-level predictions, timings and response status. Model-server retries are recorded with the run that issued them. Standalone probes have separate run directories.

| Suite | Cases | Reports and records |
| --- | ---: | --- |
| Qwen query instruction vs. bare query at 0.50 · Command R7B | 143 eligible pairs, 20 paired forecasts | [Retrieval, time accuracy and individual forecast changes](holdouts/2026-11-15-qwen-query-instruction/) |
| Analytics chat · live Command R7B | GPU distraction, filtered empty result, underestimation and longest-task queries | [Questions, SQL and supporting rows](analytics/2026-11-15-command-r7b-gpu/questions.json) · [Three-turn planning and analytics conversation](analytics/2026-11-15-command-r7b-gpu/shared-conversation.json) |
| Held-out forecast quality at 0.50 vs. 0.70 and the estimator before Qwen + Kev · Command R7B | 20 paired tasks in 10 histories | [Three-way forecast comparison and case records](holdouts/2026-11-15-command-r7b-050/) |
| Held-out forecast quality before and after Qwen + Kev · Command R7B | 20 paired tasks in 10 histories | [Paired forecast report and case records](holdouts/2026-11-15-command-r7b-paired-070/) |
| Chronological clinic replay · Cohere, Qwen3 and Kev | 40 actions, 9 completed tasks | [Run, forecast changes and accuracy](replay/2026-11-15-clinic-command-r7b-070/) |
| Local task relevance · Qwen3 and Kev | 19 holdouts, 150 within-profile pairs | [Threshold calibration and held-out decisions](retrieval/2026-11-15-qwen06b-kev4b/) |
| V3 · Model-generated point, range, confidence and warnings | 8 Cohere checks, 20 offline cases | [Report and run status](v3/README.md) · [Live Cohere checks](v3/runs/2026-11-14-cohere-direct-release/) |
| V2 · Frozen hybrid estimator, mixed daily histories, completion notes and ActivityWatch evidence | 20 in 10 histories | [Comparison report](v2/README.md) · [Runs](v2/runs/) |
| V1 · Original completed-task histories and PDFs | 20 in 10 histories | [Comparison report](v1/README.md) · [Paired case table](v1/cases.csv) · [Runs and probes](v1/runs/) |

[SQLite-based fallback and caution check](v2/runs/2026-11-14-sqlite-offline/README.md)

To score the current version, use the evaluation commands in the [sample history guide](../tasks/README.md). Give each new run its own directory, for example `data/results/local/trial-01/results.json`. The app loads sample inputs from `data/tasks/`.
