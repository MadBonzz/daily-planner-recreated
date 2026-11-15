# Command R7B release verification

The configured Cohere endpoint returned a model-authored forecast in all **8 of 8** cases. The stored point estimate, interval, confidence, explanation and cited warning text match the accepted JSON replies. Across 9 calls, the typesetting case needed a second response to pass validation. No case displayed a local fallback.

| Task and response record | Estimate (min) | Interval (min) | Fixture actual (min) | Confidence | Timing comparisons | Warnings |
| --- | ---: | ---: | ---: | --- | ---: | ---: |
| [Update the March opening-hours flyer](paused-flyer.json) | 20 | 15–25 | — | medium | 1 | 1 |
| [Order two door wedges](ready-door.json) | 10 | 8–12 | — | medium | 0 | 0 |
| [Typeset the recovery argument](fix-proof.json) | 30 | 25–35 | — | medium | 1 | 1 |
| [Outline a ten minute talk](first-task.json) | 10 | 8–12 | — | medium | 0 | 0 |
| [Solve part B of the bridge handout](bridge-proof.json) | 60 | 55–65 | 93 | medium | 1 | 3 |
| [Record the cable-cover crossing on rover C](bump-crossing.json) | 45 | 40–50 | 71 | medium | 1 | 1 |
| [Set the handover rota with Mara](mara-rooms.json) | 30 | 25–35 | 41 | medium | 1 | 2 |
| [Watch the triangular-wave Fourier lecture](triangle-video.json) | 45 | 40–50 | 54 | medium | 1 | 1 |

The flyer used the earlier sign-printing task. Its warning referred to the tape needed for the old frame. The door-wedge errand had no timing comparisons or warnings. The rover forecast cited a USB bridge disconnect, and the Fourier forecast cited off-task video viewing. These observations came from synthetic completed work in the SQLite-backed histories. The released estimator resolved the saved task descriptions and warning sources from their IDs.

## Forecast quality

The 4 held-out tasks with known actual durations had a mean absolute error of 19.75 minutes. The predicted intervals covered 0 of 4 actual durations. Three open tasks and the no-history talk have no observed completion time. For the talk, the model allocated ten minutes to prepare slides and rehearse a ten-minute talk. That budget leaves no time for slide preparation. All eight forecasts reported medium confidence, so this sample does not establish calibrated confidence.

## Reproduction

The tested model was `command-r7b-12-2024` at Cohere’s compatibility endpoint. The estimator source has SHA-256 `e95c12c45cd3c386461217fe4f821399794be7625e7dabcdbf1bf7ef6a4df7f9`. The proof-handout case ran after resuming the session with the same estimator and profile. Each JSON file includes the actual provider replies, per-call duration and prompt fingerprint, alongside the forecast shown by the planner.

The historical forecast inputs used the fixtures' reviewed labels. A [separate end-to-end run](../2026-11-14-cohere-direct-end-to-end/results.json) classified synthetic device observations using the configured model before forecasting the Fourier case. Earlier contract probes remain in their own run directories.
