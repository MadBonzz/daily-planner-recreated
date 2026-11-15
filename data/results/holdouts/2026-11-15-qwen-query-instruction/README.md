# Does instructing the Qwen retrieval query improve forecasts?

The same 20 synthetic tasks were scored with and without a task-retrieval instruction at cosine cutoff 0.50. On all tasks mean absolute error changed from **14.55 to 15.10 minutes**. On the five histories held out from the original retrieval calibration it changed from **14.10 to 13.70 minutes**. One forecast got closer to the recorded time, two got further away and seventeen tied.

| Forecast metric | Bare query | Instructed query |
| --- | ---: | ---: |
| 20 tasks, mean absolute error (min) | 14.55 | 15.10 |
| 19 flexible tasks, mean absolute error (min) | 15.32 | 15.89 |
| 10 tasks from retrieval-validation histories, mean absolute error (min) | 14.10 | 13.70 |
| Recorded duration inside the forecast interval | 5/20 | 6/20 |
| Warning cites an expected historical source | 3/5 | 3/5 |
| Cites at least one expected timing comparison | 13/19 | 13/19 |

## Retrieval

| Set and stage | Bare recall | Instructed recall | Bare precision | Instructed precision |
| --- | ---: | ---: | ---: | ---: |
| All 143 eligible pairs, Qwen at 0.50 | 40/48 (83.3%) | 39/48 (81.2%) | 63.5% | 61.9% |
| All 143 eligible pairs, Qwen and Kev | 26/48 (54.2%) | 28/48 (58.3%) | 96.3% | 96.6% |
| 67 validation pairs, Qwen at 0.50 | 19/24 (79.2%) | 18/24 (75.0%) | 76.0% | 69.2% |
| 67 validation pairs, Qwen and Kev | 15/24 (62.5%) | 16/24 (66.7%) | 100.0% | 100.0% |

Kev accepted three newly eligible matches. These were the Jensen check for the appendix derivation, the prior median run for the reference sweep and the two-target method for the shortest-window problem. The earlier stack retype for the interval-merge redo dropped below 0.50. All of the other changes to which pairs crossed the threshold involved candidates Kev rejected. Both versions passed 63 eligible pairs to the judge overall. For the transcript write-up, both authored comparisons cleared 0.50 in both arms and Kev still rejected both.

## Tasks whose forecast inputs changed

All cases below received distinct forecast requests because the prefix changed the set or rank of the historical comparisons. The other flexible tasks reused the same provider response across arms.

| Held-out task | Change in accepted history | Bare (min) | Instructed (min) | Actual (min) |
| --- | --- | ---: | ---: | ---: |
| Prepare the appendix board derivation | Added Verify the weighted-bin bound | 100 | 120 | 105 |
| Rerun the validated reference sweep | Added Repeat the archived shape sweep | 55 | 59 | 61 |
| Redo the saved interval merge after lunch | Removed Review the saved monotone stack solution | 25 | 25 | 19 |
| Solve a new shortest-window problem | Added Practice two non-overlapping target ranges | 45 | 45 | 62 |
| Set the handover rota with Mara | Rank order changed | 30 | 30 | 41 |
| Do the dishes after the tray bake | Rank order changed | 30 | 25 | 36 |
| Do the Sea Glass booth preflight | Rank order changed | 30 | 30 | 37 |

## Method

The sole intervention was prepending the following text to a new task’s complete embedding input. Qwen3-Embedding-0.6B Q8_0 always received the same bare text for historical documents. The threshold, Kev-4B Q4_K_M decision questions, historical activity, new estimator prompt and hosted Cohere Command R7B model were held fixed.

```text
Instruct: Retrieve past tasks whose work, methods, and scope are comparable and whose durations could help estimate this task.
Query: <complete task text>
```

Qdrant ranked actual 1,024-dimensional Qwen vectors from the portable fixtures. Kev’s exact recorded response was reused for a task pair whenever its complete decision input matched. The scorer verified both rankings against the cosine scores and both selection sets against the recorded pair decisions. Fixed-duration candidates were excluded just as they are in the app. The 75-minute show bypassed retrieval and forecasting in both arms. All 19 flexible tasks used model forecasts. Each target’s actual time and authored labels stayed outside the forecasting and retrieval requests.

Both arms used the production estimator at temperature zero. One live Cohere response was obtained for each unique set of messages, schema, token budget, endpoint and model. Responses for the bare baseline were requested in advance using its previously recorded selections and reused only on exact request matches. 12 of the 19 flexible tasks had identical initial provider requests across arms. The trial recorded 26 distinct provider requests and 26 actual attempts. Matched responses remove sampling variation for identical inputs. Distinct prompts each get one model outcome, so small time differences should be interpreted on this sample. Per-case latency includes replayed responses and cannot compare inference speed between arms.

Both arms use the updated forecast explanations and enriched histories in this release. Comparing the instructed result directly with an older forecast run also changes those inputs.

[All pair scores and decisions](pairs.json) · [Bare forecasts](bare/) · [Instructed forecasts](instructed/) · [Cohere receipts](requests.json) · [Full metrics](summary.json) · [Input fingerprints](manifest.json)

From the repository root, configure Cohere Command R7B, set `PLANNER_RELEVANCE_THRESHOLD=0.50`, and run the following command. The packaged Qwen vectors and Kev decisions cover both arms.

```sh
python -m examples.estimation.compare_instruction data/results/local/query-instruction-repeat
```
