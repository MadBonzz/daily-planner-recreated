# Forecast measurements

Direct outputs alone receive model-quality credit. A scheduled event is counted separately and uses zero calls. Direction credit compares the change from the user's budget against the behavioral label.

## Published cases

| Model and version | Cases present | Direct | Fallback | Correct direction | Within labeled band | Citation precision / recall |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| bonsai-4b original | 20 | 16 | 3 | 3/15 | 3/15 | 93% / 68% |
| bonsai-4b revised | 20 | 19 | 0 | 12/18 | 10/18 | 94% / 54% |
| bonsai-9b original | 20 | 19 | 0 | 8/18 | 7/18 | 96% / 93% |

| Model and version | Increase TP / FP / FN | Decrease TP / FP / FN | Near budget TP / FP / FN | Citation TP / FP / FN | Reachable warning source hit / labeled |
| --- | --- | --- | --- | --- | --- |
| bonsai-4b original | 0 / 0 / 7 | 0 / 2 / 5 | 3 / 10 / 0 | 13 / 1 / 6 | 0/1 |
| bonsai-4b revised | 9 / 1 / 1 | 1 / 1 / 4 | 2 / 4 / 1 | 15 / 1 / 13 | 3/3 |
| bonsai-9b original | 2 / 0 / 8 | 5 / 9 / 0 | 1 / 1 / 2 | 26 / 1 / 2 | 0/3 |

The following rows restrict each within-model comparison to cases with a direct forecast in both versions. A case with no direction label remains eligible for other checks.

| Model | Common direct cases | Old → revised correct direction | Old → revised in behavioral band |
| --- | ---: | --- | --- |
| bonsai-4b | 16 | 3/15 → 9/15 | 3/15 → 7/15 |

## Fresh cases

| Model | Direct | Correct direction | Evidence TP / FP / FN | Warning TP / FP / FN | Cold low confidence | Explicitly acknowledged missing timing |
| --- | ---: | ---: | --- | --- | --- | --- |
| bonsai-4b | 8 | 6/6 | 13 / 0 / 2 | 1 / 2 / 0 | 1/1 | 1/1 |

Only the historical durations and derived arithmetic change across these requests. All three new tasks have a 40-minute budget. The first and third histories also share a mean and median of 60.

| Model | Slower history point | Faster history point | Both directions correct | Tight interval width | Variable interval width | Tight / variable confidence |
| --- | ---: | ---: | --- | ---: | ---: | --- |
| bonsai-4b | 58 | 22 | yes | 22 | 30 | low / high |

## One-sentence confidence trial

The two arms differ only by the confidence instruction. Each pair preserves the same task and historical mean and median. Ties are permitted. A narrower interval or higher confidence on the more variable history is an inversion. Indexing is the development pair, and supply counting checks transfer to a fresh task.

| Model and arm | Pair | Width, tight → variable | Width ordering | Confidence, tight → variable | Confidence ordering |
| --- | --- | --- | --- | --- | --- |
| bonsai-4b confidence-control | indexing | 15 → 70 | wider | low → low | tie |
| bonsai-4b confidence-control | supply count | 11 → 44 | wider | medium → low | lower |
| bonsai-4b confidence-trial | indexing | 15 → 35 | wider | high → medium | lower |
| bonsai-4b confidence-trial | supply count | 11 → 44 | wider | high → medium | lower |

## Integrated retrieval and forecasting

Both forecast versions use the same revised forecast prompt. The integrated arm searches the full history through production Qdrant and Kev, using shipped model results. The frozen arm receives the previously selected records.

| Model and case | Frozen supplied → cited | Integrated supplied → cited | Budget | Frozen point → integrated | Expected direction |
| --- | --- | --- | ---: | ---: | --- |
| bonsai-4b patch-note | none → none | suite-report → suite-report | 40 | 45 → 29 | down |
| bonsai-4b patch-tool | trace-tool → trace-tool | boundary-debug, trace-tool → trace-tool | 85 | 90 → 90 | up |

## Runtime

Shared CPU wall time includes queueing. The solo check uses the median-sized archived case, patch-tool, after other clients drain. One solo request cannot establish a general latency distribution.

| Model and workload | Calls | Retry calls | Truncations | Median call seconds | Calls over 90 seconds |
| --- | ---: | ---: | ---: | ---: | ---: |
| bonsai-4b original published | 22 | 3 | 6 | 316.88 | 21 |
| bonsai-4b revised published | 19 | 0 | 0 | 183.87 | 19 |
| bonsai-4b revised fresh | 8 | 0 | 0 | 141.08 | 7 |
| bonsai-4b solo patch-tool | 1 | 0 | 0 | 33.06 | 0 |
| bonsai-9b original published | 19 | 0 | 0 | 272.38 | 19 |

The following latency comparison checks primary HTTP payload hashes before comparing timings. The server is warm. Context reuse and generated length can differ.

| Model | Shared → solo wall seconds | Shared → solo generated tokens by attempt | Shared → solo cached input tokens by attempt |
| --- | ---: | --- | --- |
| bonsai-4b | 211.3 → 33.06 | [246] → [242] | [3] → [65] |
