# Final forecast audit

The final forecast receives the task, selected historical records, timings, document excerpts and labeled observations. This audit freezes the published Qwen/Kev selections for the original 20 tasks. It measures how each Bonsai model uses the evidence it actually receives. The original records and the benchmark's full-pipeline results are in [the paired run](../../holdouts/2026-11-15-bonsai-direct-050/README.md).

[Behavioral metrics and runtime](metrics.md) · [Manual review of warning advice](warning-review.md) · [Per-case scores](scores.json)

## Baseline findings

The 4B model supplied 16 of the 19 flexible forecasts. The other three exhausted the output limit twice. It missed all seven labeled increases and all five decreases among the forecasts it authored. 9B supplied all 19. It detected 2/10 labeled increases and all 5 decreases, with 9 false decrease predictions. 4B called seven direction-labeled forecasts high confidence and got the direction right on two. Both models cited timing records much more reliably than they acted on the direction of the evidence. For example, 9B described a recurring 70–80% proof-work overrun and still lowered the next budget from 60 to 45 minutes.

Two of five expected warnings were unreachable from the comparison set. On the three reachable cases, 9B gave zero warnings. 4B authored a forecast for one reachable case and gave no warning. The other two were fallback calculations.

The revised prompt supplies the new budget multiplied by each prior actual/budget ratio as a same-scope reference. It asks the model to cite a concrete scope difference before discounting a repeated overrun, and to select warnings and evidence before writing the rationale. The model chooses the point, range, confidence, citations and warnings. The revised document input uses an explicitly assigned section when sibling headings identify it unambiguously. Results combine these changes on affected tasks. The stored selection of comparison IDs is identical across the published replay. Distinct secondary distraction sources now survive into the prompt. Malformed completed responses get one corrective request. Transport failures take the existing disclosed fallback path.

## Scoring

- **Direction.** The existing labeled bands define an expected increase, decrease or near-budget forecast. A predicted change greater than 5% in either direction counts as an increase or decrease. The confusion counts in [scores.json](scores.json) treat each direction as a separate one-vs-rest classification. A null precision or recall means its denominator is zero.
- **Citation.** Each supplied timing record is a candidate. The label identifies acceptable timing evidence for this task. A false positive means the forecast cited a supplied record outside the label set. These counts measure citation selection conditional on retrieval. Candidates omitted upstream never reached this component.
- **Warning.** On cases with a labeled obstacle, a hit requires a warning linked to an accepted source task. A second, stricter count also checks for terms describing the labeled obstacle in the advice. Read the case receipts to check the meaning of individual warnings. An unavailable source is counted separately. The original suite only labeled positive warning cases. The fresh scenarios include explicit negatives for neutral notes, removed obstacles and irrelevant sources. For these exhaustively labeled cases the detection counts charge both a false positive and a false negative when an unsupported warning replaces the expected one. Multiple warnings about the same supported obstacle share one credit. Fresh negative controls report how many actually had observations in the response contract. A negative case with zero supplied observations cannot produce a valid sourced warning.
- **CPU time.** Per-call wall times include waiting behind other requests. We report the median, calls exceeding the app's default 90-second timeout, retries and server-reported length truncations. The audit used an extended timeout. An isolated interactive request can see a different latency from the shared CPU run.
- **Validity.** We count model-authored outputs, identified fallbacks, ordered positive intervals, and fixed events completed with zero model calls. On the underspecified cold-start control, we also check for low confidence and explicit acknowledgement of missing historical comparisons.

The published direction labels describe synthetic behavioral expectations. No metric uses a particular invented completion time as ground truth for an individual. Fixed events and fallbacks are excluded from model-authored metrics.

### Copied advice

The first revision exposed a warning error that matching source IDs alone missed. On a late-arrival meeting, a stock check and dishwashing, 4B repeated the demonstration instruction “Test the pairing before the run.” Each warning cited an actual note, but none of those notes describes pairing. The scorer records each exact copied instruction with no pairing evidence in its source, even when a later clause mentions the correct topic. The topic-plus-source count and the count excluding copied advice are both available.

The separate warning trial removes the demonstration sentence, asks for an action tied to the recorded obstacle, and cautions against attributing elapsed time to a window switch. The development meeting and dishwashing cases test the observed failure. A neutral stock check and a camera task with genuine controller-pairing failures exercise both directions. [Frozen system instructions](prompt-versions.json)

### An unsupported scope claim

In the integrated 4B forecast for **Implement first divergence for C and Rust traces**, the target explicitly requires registering the tool in the repair loop. The cited earlier task requires wiring its error envelope into the CLI and testing truncated tails. Neither record establishes a parser-registration step that the new task can skip. Nevertheless, the model wrote, “The confidence_reason is that the shared work of handling iterator termination and truncation remains consistent, though the new task skips the parser registration step, which may reduce time slightly.” It raised the budget from 85 to 90 minutes, earning direction credit despite that unsupported justification. This is a reasoning error independent of warning-source selection. [Integrated 4B forecast](integrated/bonsai-4b/patch-tool.json) · [Task and earlier task descriptions](../../../tasks/histories/02-migration-lab.json)

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

## Controlled confidence trial

On the first revised 4B pass, increasing the spread of the indexing history widened its interval from 22 to 30 minutes. Confidence went up from low to high. The separate confidence trial compares the warnings-only prompt with the same prompt plus one sentence about consistency of comparable durations. The independent transfer case counts the same 16 stock bins with the same procedure and three 30-minute budgets. Earlier times are 39/40/41 in the tight arm and 18/40/62 in the broad arm. We record interval widths and ordinal confidence. Ties are reported separately. Increased confidence or a narrower interval on the broader history is an inversion. These comparisons test sensitivity and do not establish probabilistic calibration.

## Reproduction

With either model served locally, run the following from the repository root. Set the model alias and endpoint to the running server. Results are checkpointed per case in the chosen run directory. A transport failure stops the run. An invalid model reply and any corrective request remain in the case record.

```sh
python -m examples.estimation.forecast_audit --endpoint http://127.0.0.1:1243/v1/chat/completions --model ternary-bonsai-4b --run my-replay appendix-proof mara-rooms dev-april challenge
python -m examples.estimation.forecast_audit --endpoint http://127.0.0.1:1244/v1/chat/completions --model ternary-bonsai-2-9b --run my-replay appendix-proof mara-rooms dev-april challenge
python -m examples.estimation.score_forecast_audit
```

Use `--run warning-trial --warning-trial mara-rooms bake-dishes dev-april dark-frame-series` with the same endpoint and model. For the paired confidence control, use `--run confidence-control --warning-trial confidence`. For the one-sentence intervention, use `--run confidence-trial --confidence-trial confidence`. For the integrated retrieval comparison use `--run integrated --integrated patch-note patch-tool`. This calls the production search and per-candidate screening with the shipped Qwen and Kev outputs. A missing cached inference fails the check. To time one isolated call after other clients exit, use `--run solo --solo patch-tool`. Patch-tool had the median archived primary prompt size; the runner requires all server slots idle before and after the request.

Use `published` in place of the named tasks to run all 20 original cases. The seed labels, retry state, per-case predictions and model attribution stay attached to each result.
