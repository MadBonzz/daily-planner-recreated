# Direct model forecasts

The release run used Cohere Command R7B on eight tasks. Each record pairs the model response with the forecast saved by Plan Smarter. [Release verification and case records](runs/2026-11-14-cohere-direct-release/README.md)

The separate Fourier test classified historical activity from the synthetic SQLite store before forecasting. It made eight classification calls and one forecast call. [End-to-end case](runs/2026-11-14-cohere-direct-end-to-end/results.json)

The offline fallback was measured across all 20 held-out tasks in the ten histories. [Offline results](runs/2026-11-14-offline-sqlite/results.json)

The November 14 Bonsai run was paused during its first 4B case. The later [paired Bonsai 4B, Bonsai 9B and Cohere evaluation](../holdouts/2026-11-15-bonsai-direct-050/) uses the instructed Qwen query and a 0.50 cutoff. Results for the earlier hybrid estimator are in the [V2 comparison](../v2/README.md).
