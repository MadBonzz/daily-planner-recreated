# Command R7B development probe

This early contract required the model to reproduce an exact quotation from each historical task. Several otherwise complete responses failed when the model paraphrased or combined source fields. The released code resolves source text locally using the cited task.

3 of 8 displayed forecasts matched a validated model response. The [release report](../2026-11-14-cohere-direct-release/README.md) documents the final contract.

| Case and record | Calls | Forecast source |
| --- | ---: | --- |
| [Solve part B of the bridge handout](bridge-proof.json) | 2 | fallback |
| [Record the cable-cover crossing on rover C](bump-crossing.json) | 2 | fallback |
| [Outline a ten minute talk](first-task.json) | 1 | direct |
| [Typeset the recovery argument](fix-proof.json) | 2 | fallback |
| [Set the handover rota with Mara](mara-rooms.json) | 2 | fallback |
| [Update the March opening-hours flyer](paused-flyer.json) | 2 | fallback |
| [Order two door wedges](ready-door.json) | 1 | direct |
| [Watch the triangular-wave Fourier lecture](triangle-video.json) | 2 | direct |
