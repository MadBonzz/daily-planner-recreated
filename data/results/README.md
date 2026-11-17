# Estimator benchmark results

Each suite has a report comparing the same held-out tasks across runs. Each run has its own directory with case-level predictions, timings and response status. Model-server retries are recorded with the run that issued them. Standalone probes have separate run directories.

| Suite | Cases | Reports and records |
| --- | ---: | --- |
| Five editable demo histories | 89 completed tasks, 448 calendar actions and 15 prospective additions | [Watcher coverage, chronological checks and sample-task evidence](demo-histories/) |
| Expanded workspace-pause and selective re-estimation tests | 48 workspace conversations, 209 labeled active pairs and 96 ten-task timelines | [Scenarios, false positives and false negatives by component](component-scale/) |
| Component-by-component evaluation · Qwen, Kev and both Bonsai models | Retrieval, task relationships, forecasts, focus, activity, workspace and chat | [Component reports and per-run evidence](component-audit/) |
| Direct time forecasts with Ternary-Bonsai 4B, Ternary-Bonsai-2 9B and Command R7B | 20 paired tasks in 10 histories | [Accuracy, interval coverage, fallbacks and per-task responses](holdouts/2026-11-15-bonsai-direct-050/) |
| Qwen query instruction vs. bare query at 0.50 · Command R7B | 143 eligible pairs, 20 paired forecasts | [Retrieval, time accuracy and individual forecast changes](holdouts/2026-11-15-qwen-query-instruction/) |
| Analytics chat · live Command R7B | GPU distraction, filtered empty result, underestimation and longest-task queries | [Questions, SQL and supporting rows](analytics/2026-11-15-command-r7b-gpu/questions.json) · [Three-turn planning and analytics conversation](analytics/2026-11-15-command-r7b-gpu/shared-conversation.json) |
| Chronological clinic replay · Cohere, Qwen3 and Kev | 40 actions, 9 completed tasks | [Run, forecast changes and accuracy](replay/2026-11-15-clinic-command-r7b-070/) |
| Local task relevance · Qwen3 and Kev | 19 holdouts, 150 within-profile pairs | [Threshold calibration and held-out decisions](retrieval/2026-11-15-qwen06b-kev4b/) |

This checkout includes the latest recorded runs for each suite and any comparison records its scorer or replay needs. [Earlier runs in Git history](https://github.com/MadBonzz/daily-planner-recreated/tree/68ea21eb7a6d65ef8c9f6396a90081cde2685f55/data/results)

To score the current version, use the evaluation commands in the [sample history guide](../tasks/README.md). Give each new run its own directory, for example `data/results/local/trial-01/results.json`. The app loads sample inputs from `data/tasks/`.
