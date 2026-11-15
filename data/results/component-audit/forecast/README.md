# Final forecast audit

The final forecast receives the task, selected historical records, timings, document excerpts and labeled observations. This audit freezes the published Qwen/Kev selections for the original 20 tasks. It measures how each Bonsai model uses the evidence it actually receives. The original records and the benchmark's full-pipeline results are in [the paired run](../../holdouts/2026-11-15-bonsai-direct-050/README.md).

## Baseline findings

The 4B model supplied 16 of the 19 flexible forecasts. The other three exhausted the output limit twice. It missed all seven labeled increases and all five decreases among the forecasts it authored. 9B supplied all 19. It detected 2/10 labeled increases and all 5 decreases, with 9 false decrease predictions. 4B called seven direction-labeled forecasts high confidence and got the direction right on two. Both models cited timing records much more reliably than they acted on the direction of the evidence. For example, 9B described a recurring 70–80% proof-work overrun and still lowered the next budget from 60 to 45 minutes.

Two of five expected warnings were unreachable from the comparison set. On the three reachable cases, 9B gave zero warnings. 4B authored a forecast for one reachable case and gave no warning. The other two were fallback calculations.

The revised prompt supplies the new budget multiplied by each prior actual/budget ratio as a same-scope reference. It asks the model to cite a concrete scope difference before discounting a repeated overrun, and to select warnings and evidence before writing the rationale. The model chooses the point, range, confidence, citations and warnings. The revised document input uses an explicitly assigned section when sibling headings identify it unambiguously. Results combine these changes on affected tasks. The stored selection of comparison IDs is identical across the published replay. Distinct secondary distraction sources now survive into the prompt. Malformed completed responses get one corrective request. Transport failures take the existing disclosed fallback path.

## Scoring

- **Direction.** The existing labeled bands define an expected increase, decrease or near-budget forecast. A predicted change greater than 5% in either direction counts as an increase or decrease. The confusion counts in [scores.json](scores.json) treat each direction as a separate one-vs-rest classification. A null precision or recall means its denominator is zero.
- **Citation.** Each supplied timing record is a candidate. The label identifies acceptable timing evidence for this task. A false positive means the forecast cited a supplied record outside the label set. These counts measure citation selection conditional on retrieval. Candidates omitted upstream never reached this component.
- **Warning.** On cases with a labeled obstacle, a hit requires a warning linked to an accepted source task. A second, stricter count also checks for terms describing the labeled obstacle in the advice. Read the case receipts to check the meaning of individual warnings. An unavailable source is counted separately. The original suite only labeled positive warning cases. The fresh scenarios include explicit negatives for neutral notes, removed obstacles and irrelevant sources. For these exhaustively labeled cases the detection counts charge both a false positive and a false negative when an unsupported warning replaces the expected one. Multiple warnings about the same supported obstacle share one credit.
- **CPU time.** Per-call wall times include waiting behind other requests. We report the median, calls exceeding the app's default 90-second timeout, retries and server-reported length truncations. The audit used an extended timeout. An isolated interactive request can see a different latency from the shared CPU run.
- **Validity.** We count model-authored outputs, identified fallbacks, ordered positive intervals, and fixed events completed with zero model calls. On the underspecified cold-start control, we also check for low confidence and explicit acknowledgement of missing historical comparisons.

The published direction labels describe synthetic behavioral expectations. No metric uses a particular invented completion time as ground truth for an individual. Fixed events and fallbacks are excluded from model-authored metrics.

## Fresh scenarios

| Task | Evidence and expectation |
| --- | --- |
| Dark-frame series | Two prior camera runs overran. Increase the budget and warn about controller pairing. |
| Shelve next cart | Both prior carts finished early. Decrease the budget and leave neutral completion notes out of warnings. |
| Export only | Earlier total included review and layout. Today's export omits both and took five minutes within the prior task. An old late-client warning would be a false positive. |
| First cabinet check | No prior timing and an undiagnosed noise. Give a positive interval, state the missing evidence and use low confidence. |
| Mail registration link | A user-selected comparison set includes sending an email and printing flyers. Cite the email, omit the printer jam and reduce the budget for the identical mailing. |
| Same indexing procedure (direction pair) | Identical 40-minute target, three historical 40-minute budgets, notes, methods and selections. Only durations differ (58/60/62 versus 22/24/26 minutes). The slower-history forecast should exceed the faster one. We also check direction relative to 40. |
| Same indexing procedure (uncertainty) | A third arm replaces the slower 58/60/62-minute history with 25/60/95. Mean, median and task stay fixed. We test whether the interval widens or confidence falls. |
| Reserve theatre slot | A 50-minute scheduled event uses zero forecast calls. |

Fresh cases deliberately exercise the app's user-edited comparison set, including one wrong selection. Their labels are held outside the inference payload. Code and fixtures are in [forecast_audit.py](../../../../examples/estimation/forecast_audit.py).

## Reproduction

With either model served locally, run the following from the repository root. Set the model alias and endpoint to the running server. Results are checkpointed per case in the chosen run directory. A transport failure stops the run. An invalid model reply and any corrective request remain in the case record.

```sh
python -m examples.estimation.forecast_audit --endpoint http://127.0.0.1:1243/v1/chat/completions --model ternary-bonsai-4b --run my-replay appendix-proof mara-rooms dev-april challenge
python -m examples.estimation.forecast_audit --endpoint http://127.0.0.1:1244/v1/chat/completions --model ternary-bonsai-2-9b --run my-replay appendix-proof mara-rooms dev-april challenge
python -m examples.estimation.score_forecast_audit
```

Use `published` in place of the named tasks to run all 20 original cases. The seed labels, retry state, per-case predictions and model attribution stay attached to each result.
