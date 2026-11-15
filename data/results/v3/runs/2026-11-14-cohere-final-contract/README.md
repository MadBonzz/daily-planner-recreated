# Command R7B development probe

An intermediate response schema used an explicit zero length limit for arrays when no sources were available. Cohere returned HTTP 400 on the cold-start and door-wedge requests. The released schema omits those fields when no source can be cited.

6 of 8 displayed forecasts matched a validated model response. The [release report](../2026-11-14-cohere-direct-release/README.md) documents the final contract.

| Case and record | Calls | Forecast source |
| --- | ---: | --- |
| [Solve part B of the bridge handout](bridge-proof.json) | 1 | direct |
| [Record the cable-cover crossing on rover C](bump-crossing.json) | 1 | direct |
| [Outline a ten minute talk](first-task.json) | 1 | fallback |
| [Typeset the recovery argument](fix-proof.json) | 1 | direct |
| [Set the handover rota with Mara](mara-rooms.json) | 1 | direct |
| [Update the March opening-hours flyer](paused-flyer.json) | 1 | direct |
| [Order two door wedges](ready-door.json) | 1 | fallback |
| [Watch the triangular-wave Fourier lecture](triangle-video.json) | 1 | direct |
