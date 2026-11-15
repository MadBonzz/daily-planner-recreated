# V2 benchmark of model-selected comparisons and Python-computed estimates

This is the frozen hybrid estimator. The model selected comparable past tasks and optional warning sources. Python computed the forecast minutes, interval and confidence from the selected history. The current estimator asks the model for those values directly. Each of the ten profiles contributes two held-out tasks. Every method sees the user’s estimate in advance. The actual duration, target band and expected caution sources stay with the grader.

The profiles deliberately include close analogues, tasks that share a topic but differ in workload, and surprises. A 75-minute task to prove a crash-recovery invariant has three comparable proofs that took 1.20×, 1.25× and 1.20× their budgets. The target band is 1.14–1.30×. Three controls have no numeric target. One fixed 75-minute performance follows its schedule without calling a model.

## Forecast accuracy

| Method | Cases | Mean absolute error, min | In target band¹ | Actual inside forecast range | Mean range width, min | Cites expected caution source² |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| User estimate | 20 | 15.75 | 1/16 | — | — | — |
| Same-category median | 20 | 13.55 | 5/16 | — | — | — |
| [Offline planner](runs/2026-11-14-offline/results.json) | 20 | 13.60 | 1/16 | 15/20 | 38.6 | 0/5 |
| [Bonsai 4B first pass](runs/2026-11-14-bonsai-4b-first-pass/manifest.json) | 20 | 11.70 | 4/16 | 16/20 | 34.8 | 3/5 |
| [Bonsai 4B after compact retry](runs/2026-11-14-bonsai-4b-compact/manifest.json) | 20 | 10.60 | 9/16 | 17/20 | 28.3 | 5/5 |

¹ The 16 inferred tasks with an authored band are scored here. The fixed performance contributes to mean absolute error and range coverage, and is excluded from this column. A hit requires the **point estimate divided by the user estimate** to lie inside the task’s band. A wide planning range cannot earn a hit.

² Five tasks have a prespecified caution and a set of historical sources. This count checks whether a displayed caution cites at least one expected source. It does not establish that its wording captured the right issue. The remaining 15 tasks have no caution requirement.

## Paused 9B first pass

The 9B checkpoint contains 19 of 20 cases across the ten histories. Command responses validated for 18 cases, and the fixed performance used its scheduled duration. The remaining Fourier worksheet (`triangle-sheet`) has no saved result. On the 19 scored cases, mean absolute error was 11.11 minutes, and 17/19 planning intervals covered the actual time. The 4B table above scores all 20 cases. [Paused run manifest and individual records](runs/2026-11-14-bonsai-9b-first-pass-partial/manifest.json)

The separate Bonsai SQLite activity-classification probes had no completed case receipts at the pause.

## What the runs returned

**Bonsai 4B.** 6 of 19 initial requests produced usable responses. 13 reached the 480-token limit. A separate live compact request recovered 13/13 capped cases with a 120-token limit. After any retries, the model selected the comparison tasks for 18/19 forecasts. Other inferred forecasts used the app’s timing fallback. Median cumulative time in model calls across those 19 cases was 91 seconds.

## Where the evidence helped or misled the estimate

- The 4B used three previous crash-recovery proofs to move a 75-minute budget to 90 minutes. The work took 93. It flagged the earlier notes about constructing separate failure schedules. [Proof record](runs/2026-11-14-bonsai-4b-first-pass/cases/01-replica-proofs/bridge-proof.json)
- For a 30-minute meeting with Mara, the 4B suggested 40 minutes and surfaced the late starts in three earlier meetings. The actual time was 41. For Dev’s 20-minute meeting it suggested 15, exactly the recorded time. [Mara](runs/2026-11-14-bonsai-4b-first-pass/cases/06-neighborhood-clinic/mara-rooms.json) · [Dev](runs/2026-11-14-bonsai-4b-compact/cases/06-neighborhood-clinic/dev-april.json)
- The 4B correctly raised the Fourier video from 40 to 54 minutes. Its caution came from an earlier completion note about recommended clips. For a 40-minute offline robot-log replay, it imported USB-bridge problems from physical runs and predicted 62 minutes. The offline replay took 38. A real source can still be irrelevant to today’s task. [Fourier video](runs/2026-11-14-bonsai-4b-compact/cases/10-signal-study/triangle-video.json) · [Offline replay](runs/2026-11-14-bonsai-4b-compact/cases/09-rover-night/replay-ramp.json)
- The spare duvet cover took 58 minutes against the 4B’s 28-minute forecast. The broken closure was discovered during the task and required stitching. This was a control with no predeclared numeric target. [Laundry record](runs/2026-11-14-bonsai-4b-compact/cases/07-shared-apartment/duvet.json)

## Individual forecasts

| History · held-out task | User, min | Category, min | Offline, min | Bonsai 4B, min | Actual, min | Target ratio |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| 01 · Solve part B of the bridge handout | 75 | 90 | 82 | [90](runs/2026-11-14-bonsai-4b-first-pass/cases/01-replica-proofs/bridge-proof.json) | 93 | 1.14–1.30× |
| 01 · Fill the replay table in part A | 30 | 36 | 33 | [36](runs/2026-11-14-bonsai-4b-compact/cases/01-replica-proofs/bridge-trace.json) | 23 | 0.70–0.93× |
| 02 · Implement first divergence for C and Rust traces | 85 | 109 | 97 | [113](runs/2026-11-14-bonsai-4b-first-pass/cases/02-migration-lab/patch-tool.json) | 138 | 1.34–1.65× |
| 02 · Explain the three saved replay transcripts | 40 | 51 | 38 | [52](runs/2026-11-14-bonsai-4b-compact/cases/02-migration-lab/patch-note.json) | 28 | 0.60–0.84× |
| 03 · Prepare the appendix board derivation | 60 | 75 | 67 | [104](runs/2026-11-14-bonsai-4b-compact/cases/03-reading-depth/appendix-proof.json) | 105 | 1.48–1.86× |
| 03 · Make one slide about the two ablations | 45 | 56 | 42 | [56](runs/2026-11-14-bonsai-4b-compact/cases/03-reading-depth/metrics-club.json) | 31 | 0.60–0.87× |
| 04 · Fix nondeterministic segmented sums | 85 | 108 | 92 | [130](runs/2026-11-14-bonsai-4b-compact/cases/04-kernel-workbench/async-flicker.json) | 132 | 1.38–1.68× |
| 04 · Rerun the validated reference sweep | 60 | 76 | 60 | [92](runs/2026-11-14-bonsai-4b-compact/cases/04-kernel-workbench/sweep-again.json) | 61 | 0.90–1.11× |
| 05 · Solve a new shortest-window problem | 50 | 60 | 55 | [53](runs/2026-11-14-bonsai-4b-first-pass/cases/05-interview-practice/shortest-window.json) | 62 | 1.13–1.31× |
| 05 · Redo the saved interval merge after lunch | 25 | 30 | 28 | [20](runs/2026-11-14-bonsai-4b-first-pass/cases/05-interview-practice/editorial-review.json) | 19 | 0.60–0.87× |
| 06 · Set the handover rota with Mara | 30 | 32 | 32 | [40](runs/2026-11-14-bonsai-4b-first-pass/cases/06-neighborhood-clinic/mara-rooms.json) | 41 | 1.23–1.44× |
| 06 · Run the supply check with Dev | 20 | 21 | 18 | [15](runs/2026-11-14-bonsai-4b-compact/cases/06-neighborhood-clinic/dev-april.json) | 15 | 0.66–0.86× |
| 07 · Do the dishes after the tray bake | 30 | 28 | 31 | [36](runs/2026-11-14-bonsai-4b-compact/cases/07-shared-apartment/bake-dishes.json) | 36 | 1.15–1.28× |
| 07 · Wash the spare duvet cover | 30 | 28 | 29 | [28](runs/2026-11-14-bonsai-4b-compact/cases/07-shared-apartment/duvet.json) | 58 | control |
| 08 · Operate captions for Sea Glass at 19:00 | 75 | 75 | 75 | [75](runs/2026-11-14-bonsai-4b-first-pass/cases/08-community-playhouse/blue-show.json) | 75 | fixed |
| 08 · Do the Sea Glass booth preflight | 30 | 38 | 34 | [38](runs/2026-11-14-bonsai-4b-compact/cases/08-community-playhouse/sea-preflight.json) | 37 | 1.16–1.42× |
| 09 · Record the cable-cover crossing on rover C | 45 | 69 | 49 | [58](runs/2026-11-14-bonsai-4b-compact/cases/09-rover-night/bump-crossing.json) | 71 | 1.43–1.68× |
| 09 · Plot the ramp trial from yesterday’s recording | 40 | 62 | 40 | [62](runs/2026-11-14-bonsai-4b-compact/cases/09-rover-night/replay-ramp.json) | 38 | 0.85–1.12× |
| 10 · Watch the triangular-wave Fourier lecture | 40 | 54 | 42 | [54](runs/2026-11-14-bonsai-4b-compact/cases/10-signal-study/triangle-video.json) | 54 | 1.25–1.46× |
| 10 · Complete the triangle-wave practice problems | 45 | 60 | 53 | [53](runs/2026-11-14-bonsai-4b-first-pass/cases/10-signal-study/triangle-sheet.json) | 44 | 0.86–1.13× |

Each linked point opens the original case record, including the selected evidence, forecast range, caution sources, client and server timings, and raw model output. The app computes the point and range from the selected historical durations.

## Setup and replay

The forecast runs used zero GPU layers, temperature 0, an 8,192-token context and one inference slot in LM Studio’s llama.cpp Linux x86-64 AVX2 backend 2.66.0 (`b11966`). The 4B used `Ternary-Bonsai-4B-Q2_0_g64.gguf` and the 9B used `Ternary-Bonsai-2-9B-Q2_0.gguf` with reasoning off. CPU allocation changed during the first-pass measurements. The engines initially shared the machine and later ran on eight threads separately, so latency comparisons are descriptive. Weight digests and case timestamps are in the run manifests.

Each prediction used completed tasks only. Prior task reports came from replaying synthetic ActivityWatch events with authored interval labels. Classification probes, where listed, score the activity classifier separately. The frozen fixtures used native ActivityWatch JSON exports during these runs. The forecast run predates the SQLite conversion. Replaying the new SQLite stores reproduced all 79 prior reports. Holding the measured estimator constant, all 19 model-eligible holdouts produced identical JSON user messages from either representation. Later retrieval changes can alter which comparisons reach the model.

The same-category baseline multiplies the user’s estimate by the median actual-to-estimated ratio across previous tasks in that category. The offline planner exercises the production fallback. Coverage uses the app’s uncalibrated planning intervals. All outcomes are synthetic, and they cover only two held-out tasks per profile.

To evaluate the current estimator with a locally served model, configure `PLANNER_MODEL_ENDPOINT` and `PLANNER_MODEL` and run `python -m examples.estimation.run evaluate --strict-model --json <new-run-directory>/results.json`. Give every run its own directory. The 480-token first-pass responses and compact responses in these records came from separate measured requests. When the compact request ran later, it reused the saved first reply and matched the original request hash before issuing a new live request.

## Separate SQLite checks on the hybrid estimator

The hybrid app also ran against the new native stores. Command R7B made eight classification calls on the Fourier study history and one forecast call. It returned a 45-minute prediction for a task that took 54 minutes, cited an expected distraction source and completed all nine provider calls. [Cohere end-to-end record](runs/2026-11-14-cohere-r7b-end-to-end/results.json) · [Cohere regression cases](runs/2026-11-14-cohere-r7b-sqlite-smoke/README.md)

The SQLite offline evaluation ran all 20 holdouts and recorded 13.60 minutes mean absolute error. Four of the five caution cases cited an expected source. [Current offline run](runs/2026-11-14-sqlite-offline/results.json)
