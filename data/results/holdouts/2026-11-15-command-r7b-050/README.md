# Held-out forecasts with a 0.50 retrieval cutoff

This run compares the earlier estimator that used hashed words and phrases plus comparison rules with two versions using Qwen3-Embedding-0.6B and Kev-4B. The forecast model was Cohere Command R7B in all three runs. The 0.50 run used a fresh Qdrant index and live model requests on the same 20 held-out tasks. The pre-Qwen/Kev and 0.70 columns use the [saved paired runs](../2026-11-15-command-r7b-paired-070/).

| Measure | Before Qwen and Kev | Qwen + Kev, 0.70 | Qwen + Kev, 0.50 |
| --- | ---: | ---: | ---: |
| Mean absolute error on 20 tasks (min) | 15.65 | 16.40 | 14.75 |
| Mean absolute error on 19 flexible tasks (min) | 16.47 | 17.26 | 15.53 |
| Mean absolute error on the 10 tasks from retrieval-validation profiles (min) | 12.10 | 14.00 | 13.00 |
| Mean absolute error where all three produced model forecasts (min) | 16.78 (18 tasks) | 17.44 (18 tasks) | 15.61 (18 tasks) |
| Actual time inside planning range | 5/20 | 5/20 | 6/20 |
| Mean planning range width (min) | 11.65 | 11.10 | 9.35 |
| Flexible tasks in authored target band | 2/16 | 2/16 | 3/16 |
| Tasks citing at least one comparison named in the scoring rubric | 18/19 | 8/19 | 11/19 |
| Warnings citing an expected historical source | 4/5 | 1/5 | 3/5 |
| Flexible tasks using ≥1 timing comparison | 19/19 | 9/19 | 12/19 |
| Model-authored flexible forecasts | 18/19 | 19/19 | 19/19 |

Across all 20 tasks, 0.50 reduced mean absolute error by **0.90 minutes (5.8%)** against the pre-Qwen/Kev version. Six predictions were closer to the recorded time, four were further away and ten tied. The bridge proof alone contributed 15 of the 18 minutes of net error reduction. On the ten tasks from the five profiles held out during retrieval calibration, the original estimator remained more accurate, averaging **12.10 minutes** of error against **13.00** at 0.50. This single run on synthetic tasks gives mixed evidence for improvement beyond the benchmark.

Against 0.70, four predictions improved and sixteen tied. The users’ own budgets averaged 15.75 minutes of absolute error. Applying the median historical correction within each category had the lowest error of the methods compared, at 13.55 minutes.

## Task-level point estimates in minutes

| Held-out task | User | Before Qwen and Kev | 0.70 | 0.50 | Recorded |
| --- | ---: | ---: | ---: | ---: | ---: |
| Solve part B of the bridge handout | 75 | 60 | 60 | 75 | 93 |
| Fill the replay table in part A | 30 | 30 | 25 | 25 | 23 |
| Implement first divergence for C and Rust traces | 85 | 85 | 85 | 85 | 138 |
| Explain the three saved replay transcripts | 40 | 30 | 40 | 40 | 28 |
| Prepare the appendix board derivation | 60 | 60 | 55 | 60 | 105 |
| Make one slide about the two ablations | 45 | 42 | 45 | 45 | 31 |
| Fix nondeterministic segmented sums | 85 | 100 | 100 | 100 | 132 |
| Rerun the validated reference sweep | 60 | 60 | 59 | 59 | 61 |
| Solve a new shortest-window problem | 50 | 45 | 45 | 48 | 62 |
| Redo the saved interval merge after lunch | 25 | 25 | 25 | 25 | 19 |
| Set the handover rota with Mara | 30 | 30 | 30 | 30 | 41 |
| Run the supply check with Dev | 20 | 20 | 20 | 20 | 15 |
| Do the dishes after the tray bake | 30 | 30 | 32 | 32 | 36 |
| Wash the spare duvet cover | 30 | 25 | 25 | 25 | 58 |
| Operate captions for Sea Glass at 19:00 | 75 | 75 | 75 | 75 | 75 |
| Do the Sea Glass booth preflight | 30 | 30 | 25 | 35 | 37 |
| Record the cable-cover crossing on rover C | 45 | 45 | 45 | 45 | 71 |
| Plot the ramp trial from yesterday’s recording | 40 | 30 | 35 | 35 | 38 |
| Watch the triangular-wave Fourier lecture | 40 | 45 | 45 | 45 | 54 |
| Complete the triangle-wave practice problems | 45 | 45 | 40 | 40 | 44 |

## How retrieval affected the forecasts

In the five retrieval-validation profiles, Kev accepted 15 of the 24 labeled comparisons at 0.50, up from 6 at 0.70. Every comparison it accepted at either cutoff was a labeled match.

For the transcript write-up, [cosine scores](../../retrieval/2026-11-15-qwen06b-kev4b/embeddings.json) of 0.583 and 0.565 put both labeled write-ups above 0.50. [Kev rejected both](../../retrieval/2026-11-15-qwen06b-kev4b/decisions.json). Cohere again predicted 40 minutes with no comparison. The pre-Qwen/Kev forecast was 30 minutes and the work took 28.

For the Mara handover, Kev accepted three previous meetings at 0.50, up from zero at 0.70. Cohere cited one of those meetings and predicted 30 minutes. The older estimator and the 0.70 version also predicted 30. The meeting took 41 minutes.

For the Sea Glass booth preflight, Cohere’s rationale mentioned earlier playback and relay checks that took 38 and 43 minutes. Its structured timing citation pointed to the playback check. It forecast 35 minutes at 0.50, after forecasting 25 at 0.70. The pre-Qwen/Kev forecast was 30 and the work took 37.

## Run details

All history JSON, PDFs, and ActivityWatch fixture databases have identical Git trees across the three versions. Each case uses only earlier completed tasks. The evaluator withholds the target’s timer outcome, the scoring rubrics and the other holdout. Historical ActivityWatch reports use the same reviewed labels. No user feedback about relevance was injected. The sample includes two held-out tasks per profile and one fixed 75-minute show. The separate validation-profile row covers the five profiles held out of the original retrieval calibration.

The 0.50 run used 93 successful embedding requests (Qwen Q8_0), 63 successful decision requests (Kev Q4_K_M), and hosted Cohere for forecasting at temperature zero. The local weights ran one at a time on CPU with LM Studio’s bundled llama.cpp backend. The requested cutoff was the only estimator setting changed from 0.70. Provider responses can vary between runs. Of the 10 flexible cases whose first forecast prompt hash was identical at both cutoffs, 1 changed point estimate.

The [full results](results.json) record the tasks Kev accepted, timing citations, uncertainty ranges, warning sources, and provider responses. The [paired summary](summary.json) includes source revisions, input tree hashes, and each subgroup’s measurements. The earlier [0.45 experiment](../../retrieval/2026-11-15-qwen06b-kev4b/) measured which historical tasks were relevant. The pre-Qwen/Kev forecast baseline here comes from revision `3ec53ef`.

To reproduce, configure Command R7B, Qwen and Kev with the [planner setup](../../../../README.md#local-relevance-models), set `PLANNER_RELEVANCE_THRESHOLD=0.50`, and run the following command.

```sh
python -m examples.estimation.run evaluate --strict-model --json data/results/local/heldout-050-repeat/results.json
```
