# Command R7B smoke test

The configured Cohere compatibility endpoint was exercised with synthetic task histories and AW SQLite databases. An earlier request with a schema directly under `response_format.schema` returned HTTP 422. The corrected shared client uses `response_format.type = json_schema` and nests the schema in `response_format.json_schema.schema`. Successful responses passed the same local validator as LM Studio.

| Case | Suggested minutes | Observed behavior |
| --- | ---: | --- |
| [Rust migration](held-seed.json) | 91 | Cited two EOF fixture debugging notes. |
| [Pick up a card](pickup.json) | 15 | Kept the user's budget without a relevant prior task. |
| [Rover run](bump-crossing.json) | 58 | Cited earlier USB bridge disconnects. |
| [Meeting with Mara](mara-rooms.json) | 40 | A 30-minute invitation was adjusted using earlier meetings. |
| [Fourier lecture](triangle-video.json) | 45 | Cited prior off-task video titles, including hockey highlights. |
| [First task](first-task.json) | 7 | The hosted model supplied a 5–9 minute range for preparing six slides and rehearsing a ten-minute talk. Even the upper bound is shorter than the rehearsal. This case exposes a cold-start estimation error. |

The five forecasts with history use the fixture's reviewed historical interval labels. The [clinic](phone-and-laptop-06-neighborhood-clinic.json) and [laundry](phone-and-laptop-07-shared-apartment.json) files exercise real model classification on synthetic SQLite data. The clinic had 51 elapsed seconds, including 45 with a laptop window and Android Calendar open simultaneously. The second sample had 96 seconds, including 90 with the Clock app open on the phone and the laptop idle. These separate samples test call contracts and device attribution. The [manifest](manifest.json) records the provider, model, and timestamps.
