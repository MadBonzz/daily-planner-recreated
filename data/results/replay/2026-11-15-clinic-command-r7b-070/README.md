# Neighborhood clinic chronological replay

The replay applied all 40 scheduled actions to a fresh Dev workspace anchored on November 15, 2026. Nine tasks reached completion. Two remained open. Each creation, timer action and completion ran at its simulated timestamp. The estimator received the completed history available at that moment.

Command R7B supplied all 15 forecasts, including revisions. The ActivityWatch classification pipeline generated 9 completion reports. Qwen3-Embedding-0.6B Q8_0 and Kev-4B Q4_K_M handled 23 successful embedding calls and 19 successful pair decisions. Qdrant used a cosine cutoff of 0.70 for automatic comparisons.

## Initial estimates against recorded time

| Completed tasks | User mean absolute error | Planner mean absolute error |
| ---: | ---: | ---: |
| 9 | 6.00 min | 5.56 min |

Eight of nine initial forecasts matched the user's budget. For the third inventory call with Dev, the planner predicted 16 minutes against a 20-minute budget. The call took 16 minutes. The planner kept the original 30-minute budgets for all three capacity meetings with Mara, which took 39–41 minutes. The score uses each task's first saved forecast. The [threshold evaluation](../../retrieval/2026-11-15-qwen06b-kev4b/) assesses retrieval across profiles.

| Task in the fixture | User estimate (min) | Initial planner estimate (min) | Actual (min) |
| --- | ---: | ---: | ---: |
| mara-evening | 30 | 30 | 40 |
| dev-stock | 20 | 20 | 15 |
| mara-capacity | 30 | 30 | 41 |
| campus-posters | 35 | 35 | 34 |
| dev-packs | 20 | 20 | 14 |
| mara-late-room | 30 | 30 | 39 |
| dev-receipts | 20 | 16 | 16 |
| bus-pass | 15 | 15 | 12 |
| mara-dropoff | 10 | 10 | 5 |

## Inspect the run

At action 11, the forecast for `dev-packs` cited `dev-stock` and returned 20 minutes. The [event log](events.jsonl) records changes to forecasts, report totals and task status. Each job lists the historical tasks available to it. The [score summary](summary.json) contains the individual outcomes and the model/fallback breakdown. The [manifest](manifest.json) records settings and fixture hashes.

The two local models ran on CPU with LM Studio's bundled llama.cpp runtime. The test harness unloaded one set of weights before loading the other to fit the test machine's RAM. The application issued its normal HTTP requests for both roles and used Cohere's hosted endpoint for forecasting and classification. Forecasts and labels may change on a later run.

To repeat with the models configured as in the [planner setup](../../../../README.md#local-relevance-models), run from the repository root:

```sh
python -m examples.estimation.replay 06 --anchor-day 2026-11-15 --output data/results/local/clinic-replay-01
```
