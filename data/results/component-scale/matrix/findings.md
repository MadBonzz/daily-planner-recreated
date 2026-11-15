# Selective re-estimation results

The same 160 pair labels were reviewed independently by two assistants. All 41 positive and 119 negative labels agreed. Counts below use the pair or surviving-target opportunities named in each row. [Reference](reference.json) · [Full metrics and per-case scores](metrics.json)

| Component and denominator | TP | FP | FN | TN | Precision | Recall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen dual-query screen alone, 160 active pairs | 41 | 23 | 0 | 96 | 64.1% | 100.0% |
| Kev, 160 active pairs | 38 | 2 | 3 | 117 | 95.0% | 92.7% |
| Saved refresh vs semantic reference, 128 survivors | 34 | 1 | 2 | 91 | 97.1% | 94.4% |
| Saved refresh vs stored Kev decision, 128 survivors | 35 | 0 | 0 | 93 | 100.0% | 100.0% |
| Newly finished source reached the forecast vs reference | 36 | 2 | 3 | 103 | 94.7% | 92.3% |

The worker saved 35 refreshes over 48 waves on 16 boards. 8 waves included multiple completions before the worker drained. 91 unrelated survivor opportunities kept their prior forecast. There were 0 state or queue-order errors over 80 task additions and 128 survivor transitions. [Queue traces](queue.json)

### Completion waves that differed from the reference

| Completed task and actual | Open target | Observed effect |
| --- | --- | --- |
| Take up the uniform pants (58 min) | Mend the hem and zip on a skating costume | Missed refresh |
| Sync the kitchen interview (51 min) | Caption the workshop cut | Unnecessary refresh |
| Caption the workshop cut (77 min) | Deliver the kitchen sound and captions | Missed refresh |

Two coalesced waves illustrate why a correctly queued target also needs a source-level check.

- **Mend the hem and zip on a skating costume** refreshed as expected. Its forecast input omitted “Shorten the black trousers.”
- **Set up the spare electric guitar** refreshed as expected. Its forecast input included an unrelated source, “Swap passive bass pickups.”

### Kev disagreements

| Board | Task pair | Reference | Kev score |
| --- | --- | --- | ---: |
| textile-mending | Shorten the black trousers / Mend the hem and zip on a skating costume | Relevant | 0.218 |
| textile-mending | Take up the uniform pants / Mend the hem and zip on a skating costume | Relevant | 0.458 |
| instrument-room | Swap passive bass pickups / Set up the spare electric guitar | Unrelated | 0.558 |
| video-edit | Sync the kitchen interview / Caption the workshop cut | Unrelated | 0.555 |
| video-edit | Caption the workshop cut / Deliver the kitchen sound and captions | Relevant | 0.443 |

Actual Qwen embeddings were recorded for 80 task documents, their 80 instructed queries, and the shared permit-renewal document. 0 of the 80 tasks passed the dual-role .50 threshold for that unrelated historical source. Kev accepted 0. [Embedding receipts](qwen.json) · [Relevance receipts](kev.json)

## Interleaving stress

96 ten-card schedules included 960 submissions, 439 completions, 445 timer starts, 189 pauses, and 96 retroactive time corrections. The runner checked 5,930 persisted cell values, 2,496 creation snapshots, 1,476 current forecasts, and 521 correction-target decisions. 0 seeds failed. The test substitutes controlled model outputs and runs the real store and inference workers. [Per-seed counts](stress.json)

A separate single-board burst created 36 pending forecasts and 630 distinct matrix judgments. Six simultaneous completions produced 13 coalesced refreshes and left 17 other open forecasts untouched.
