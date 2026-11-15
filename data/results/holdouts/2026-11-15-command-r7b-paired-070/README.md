# Held-out time forecasts before and after the retrieval change

The earlier and current Plan Smarter releases were run on the same 20 held-out tasks, two per synthetic profile. Across those tasks, mean absolute error changed from **15.65 to 16.40 minutes**. The original user estimates averaged **15.75 minutes** of absolute error. One scheduled event has an exact, fixed duration in both versions.

| Metric | Earlier estimator | Qwen + Kev at 0.70 |
| --- | ---: | ---: |
| Mean absolute error, all 20 (min) | 15.65 | 16.40 |
| Mean absolute error, 19 flexible tasks (min) | 16.47 | 17.26 |
| Mean absolute error, 10 tasks in retrieval-validation profiles (min) | 12.10 | 14.00 |
| Mean absolute error, 18 flexible tasks with valid forecasts from both models (min) | 16.78 | 17.44 |
| Actual time within planning interval | 5/20 | 5/20 |
| Average planning interval width (min) | 11.65 | 11.10 |
| Flexible tasks in their authored target band | 2/16 | 2/16 |
| Warnings citing an expected historical source | 4/5 | 1/5 |
| Flexible forecasts citing a past task for timing | 19/19 | 9/19 |

On individual tasks the new forecast was closer on 3, further away on 6, and equally far away on 11.

The same-category median baseline averaged 13.55 minutes of absolute error. Among the 19 flexible tasks, the earlier estimator returned the original budget on 11, and the current estimator did so on 7.

For the transcript write-up, the earlier forecast cited two previous write-ups and predicted 30 minutes. The two authored comparisons scored 0.583 and 0.565 in the [saved cosine measurements](../../retrieval/2026-11-15-qwen06b-kev4b/embeddings.json), below 0.70. The new forecast had no comparisons and returned the user’s 40-minute budget. The task took 28 minutes. For the Mara handover, the earlier forecast cited a prior meeting and the newer one cited none. Both predicted 30 minutes for a meeting that took 41. Those cases show how relevance selection can change a forecast and how the forecaster can miss an overrun even with a relevant example.

The current estimator returned a valid model-authored forecast for all 19 flexible tasks. The earlier estimator returned 18 and used its fallback on one. The updated estimator improved response reliability on this run.

## Forecasts by task

A negative change means the current forecast was closer to the recorded duration.

| Profile | Held-out task | User (min) | Earlier (min) | Current (min) | Actual (min) | Change in absolute error (min) |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 01 | Solve part B of the bridge handout | 75 | 60 | 60 | 93 | +0 |
| 01 | Fill the replay table in part A | 30 | 30 | 25 | 23 | -5 |
| 02 | Implement first divergence for C and Rust traces | 85 | 85 | 85 | 138 | +0 |
| 02 | Explain the three saved replay transcripts | 40 | 30 | 40 | 28 | +10 |
| 03 | Prepare the appendix board derivation | 60 | 60 | 55 | 105 | +5 |
| 03 | Make one slide about the two ablations | 45 | 42 | 45 | 31 | +3 |
| 04 | Fix nondeterministic segmented sums | 85 | 100 | 100 | 132 | +0 |
| 04 | Rerun the validated reference sweep | 60 | 60 | 59 | 61 | +1 |
| 05 | Solve a new shortest-window problem | 50 | 45 | 45 | 62 | +0 |
| 05 | Redo the saved interval merge after lunch | 25 | 25 | 25 | 19 | +0 |
| 06 | Set the handover rota with Mara | 30 | 30 | 30 | 41 | +0 |
| 06 | Run the supply check with Dev | 20 | 20 | 20 | 15 | +0 |
| 07 | Do the dishes after the tray bake | 30 | 30 | 32 | 36 | -2 |
| 07 | Wash the spare duvet cover | 30 | 25 | 25 | 58 | +0 |
| 08 | Operate captions for Sea Glass at 19:00 | 75 | 75 | 75 | 75 | +0 |
| 08 | Do the Sea Glass booth preflight | 30 | 30 | 25 | 37 | +5 |
| 09 | Record the cable-cover crossing on rover C | 45 | 45 | 45 | 71 | +0 |
| 09 | Plot the ramp trial from yesterday’s recording | 40 | 30 | 35 | 38 | -5 |
| 10 | Watch the triangular-wave Fourier lecture | 40 | 45 | 45 | 54 | +0 |
| 10 | Complete the triangle-wave practice problems | 45 | 45 | 40 | 44 | +3 |

## Method and scope

Both revisions used hosted Cohere Command R7B at temperature zero. The evaluator held back each target’s actual time, interval grades, activity and completion feedback. Each predictor saw the same historical tasks, user budgets, PDF text and previously reviewed ActivityWatch labels. No simulated user corrections were injected. The new pipeline embedded the tasks with Qwen3-Embedding-0.6B Q8_0, filtered at cosine similarity 0.70 and passed the candidates individually through Kev-4B Q4_K_M. Local inference used the CPU runtime bundled with LM Studio, switching between the two local models to fit available memory. Both arms used their own production retrieval, prompts and output validation.

The ten profiles in this fixture have different task types and timing patterns. Five profiles were used in the earlier retrieval calibration, so the table also gives the result for the five profiles reserved for retrieval validation. One run on synthetic data measures behavior on these cases. It cannot identify the separate effects of the embedding threshold, the pairwise decision, or the changes to the forecasting prompt.

The following requests received an HTTP server or rate-limit error. Each failed request was eligible for one repeat with identical inputs. Original receipts, fallback estimates and first-pass totals are preserved alongside the final scored cases. No forecast was retried because of its numerical estimate.

- Before · `metrics-club` · api.cohere.ai returned HTTP 500 for command-r7b-12-2024.
- Before · `bridge-trace` · api.cohere.ai returned HTTP 500 for command-r7b-12-2024.

Counting the application's first response on every task, including any fallback, gives 15.65 minutes MAE before and 16.40 minutes after.

The full-suite score includes the estimates the app displayed after an unsuccessful request. The matched-model row includes the 18 flexible tasks where both pipelines succeeded.

- Earlier · `metrics-club` · The model could not supply a valid forecast. For comparison 2, explain which work both tasks require, or omit it.

The older Bonsai 4B hybrid benchmark recorded 10.60 minutes MAE on these same 20 tasks. In that version Bonsai chose comparisons and Python calculated the minutes. The model and estimator both differ from the paired Cohere runs. [Earlier benchmark](../../v2/README.md)

The results include aligned task IDs, forecast ranges, cited evidence, exact model replies and request hashes. [Earlier run](before/results.json) · [Current run](after/results.json) · [Paired metrics](summary.json)

To reproduce either arm, check out its revision, install that revision’s requirements, configure Command R7B, and run the evaluator with the default reviewed historical labels. The current revision also requires the two local relevance models.

```sh
python -m examples.estimation.run evaluate --strict-model --json data/results/local/held-out-rerun/results.json
```

Earlier revision [3ec53ef](https://github.com/MadBonzz/daily-planner-recreated/tree/3ec53efda053d69be2afa34d4b385b11f68b0d6e). Current revision [b69927d](https://github.com/MadBonzz/daily-planner-recreated/tree/b69927d3aa1d1ddf283276fe2e7666cccb05f1d6).
