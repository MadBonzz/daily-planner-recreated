# Direct model forecasts

The release run used Cohere Command R7B on eight tasks. Each record pairs the model response with the forecast saved by Plan Smarter. [Release verification and case records](runs/2026-11-14-cohere-direct-release/README.md)

The separate Fourier test classified historical activity from the synthetic SQLite store before forecasting. It made eight classification calls and one forecast call. [End-to-end case](runs/2026-11-14-cohere-direct-end-to-end/results.json)

The offline fallback was measured across all 20 held-out tasks in the ten histories. [Offline results](runs/2026-11-14-offline-sqlite/results.json)

The Bonsai direct-forecast evaluation was paused during the first 4B case. No direct Bonsai forecast was saved, and the 9B direct run had not started. Existing results for the previous estimator are in the [V2 comparison](../v2/README.md).
