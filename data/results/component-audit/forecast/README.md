# Final forecast audit

The final forecast receives the task, selected historical records, timings, document excerpts and labeled observations. This audit freezes the published Qwen/Kev selections for the original 20 tasks. It measures how each Bonsai model uses the evidence it actually receives. The original records and the benchmark's full-pipeline results are in [the paired run](../../holdouts/2026-11-15-bonsai-direct-050/README.md).

[Behavioral metrics and runtime](metrics.md) · [Manual review of warning advice](warning-review.md) · [Per-case scores](scores.json)

With the historical selections held fixed, **4B improved from 3/15 to 9/15** correct duration directions on cases where it supplied the forecast both times. It also supplied direct forecasts on all three cases where it had previously fallen back. **9B improved from 8/18 to 15/18**, including nine of ten expected increases. On the six new cases with direction labels both models chose the expected direction. Their warning decisions were less reliable. With an explicit negative warning rubric, 4B emitted false warnings on two cases and 9B on three.

The app now uses the combined warning-and-confidence instruction (SHA-256 `f5984b5789b7dfc0dc1100a771a256645c84c4fad174deb70c420fe149e41b70`). Direct tests of that instruction cover the eight targeted cases listed below. Each model also completed an isolated request using the primary instruction. The median-sized case took 33 seconds on 4B and 56 seconds on 9B, versus 211 and 508 seconds under concurrent load.

## Baseline findings

The 4B model supplied 16 of the 19 flexible forecasts. The other three exhausted the output limit twice. It missed all seven labeled increases and all five decreases among the forecasts it authored. 9B supplied all 19. It detected 2/10 labeled increases and all 5 decreases, with 9 false decrease predictions. 4B called seven direction-labeled forecasts high confidence and got the direction right on two. Both models cited timing records much more reliably than they acted on the direction of the evidence. For example, 9B described a recurring 70–80% proof-work overrun and still lowered the next budget from 60 to 45 minutes.

Two of five expected warnings were unreachable from the comparison set. On the three reachable cases, 9B gave zero warnings. 4B authored a forecast for one reachable case and gave no warning. The other two were fallback calculations.

The revised prompt supplies the new budget multiplied by each prior actual/budget ratio as a same-scope reference. It asks the model to cite a concrete scope difference before discounting a repeated overrun, and to select warnings and evidence before writing the rationale. The model chooses the point, range, confidence, citations and warnings. The revised document input uses an explicitly assigned section when sibling headings identify it unambiguously. Results combine these changes on affected tasks. The stored selection of comparison IDs is identical across the published replay. Distinct secondary distraction sources now survive into the prompt. Malformed completed responses get one corrective request. Transport failures take the existing disclosed fallback path.

## Which prompt ran on which cases

| Run | Forecast instruction | Cases directly executed per model |
| --- | --- | --- |
| Primary revised | First revision with the literal pairing example | All 20 published and 9 fresh cases, including two fixed events that bypass inference |
| Integrated and solo | Same first revision | Two integrated PDF cases with current Qwen/Kev retrieval, plus an isolated repeat of the frozen patch-tool request |
| Warning trial | Removed the example and required an action tied to the cited observation | Mara handover, dishwashing, Dev stock check and dark-frame series |
| Confidence control | Same warnings-only instruction | Tight and broad indexing histories, plus tight and broad supply-count histories |
| Confidence trial | Warnings-only instruction plus one sentence about variability and confidence | The same four timing cases as the confidence control |
| Combined warning replay | Same combined instruction as the confidence trial | Mara handover, dishwashing, Dev stock check and dark-frame series |

The **first revision** ran on all 29 cases per model. The reported direction, range, citation and CPU metrics for the primary cohort apply to that revision. The later prompts were evaluated on targeted panels. The warnings-only and combined prompts each had a four-task timing panel and a separate four-task warning panel. The manual review checks what the models actually advised on the warning panel. No full-cohort forecast quality score is available for the combined instruction.

The primary estimator source has SHA-256 `85f4e67e3549a4d7c6b7c0be4a148dcf0976321e2d1f3765de958fbad0165592` in both arms. The replay harness loads the frozen first-revision instruction by default, independently of later app prompt changes. [Primary source snapshot](https://github.com/MadBonzz/daily-planner-recreated/blob/9978d5a62d30cbe6f58b3eb3699e6acb91ac7723/estimator.py) · [Exact text of all three variants](prompt-versions.json)

## Scoring

- **Direction.** The existing labeled bands define an expected increase, decrease or near-budget forecast. A predicted change greater than 5% in either direction counts as an increase or decrease. The confusion counts in [scores.json](scores.json) treat each direction as a separate one-vs-rest classification. A null precision or recall means its denominator is zero.
- **Citation.** Each supplied timing record is a candidate. The label identifies acceptable timing evidence for this task. A false positive means the forecast cited a supplied record outside the label set. These counts measure citation selection conditional on retrieval. Candidates omitted upstream never reached this component.
- **Warning.** On cases with a labeled obstacle, a hit requires a warning linked to an accepted source task. A second, stricter count also checks for terms describing the labeled obstacle in the advice. Read the case receipts to check the meaning of individual warnings. An unavailable source is counted separately. The original suite only labeled positive warning cases. The fresh scenarios include explicit negatives for neutral notes, removed obstacles and irrelevant sources. For these exhaustively labeled cases the detection counts charge both a false positive and a false negative when an unsupported warning replaces the expected one. Multiple warnings about the same supported obstacle share one credit. Fresh negative controls report how many actually had observations in the response contract. A negative case with zero supplied observations cannot produce a valid sourced warning.
- **CPU time.** Per-call wall times include waiting behind other requests. We report the median, calls exceeding the app's default 90-second timeout, retries and server-reported length truncations. The audit used an extended timeout. An isolated interactive request can see a different latency from the shared CPU run.
- **Validity.** We count model-authored outputs, identified fallbacks, ordered positive intervals, and fixed events completed with zero model calls. On the underspecified cold-start control, we also check for low confidence and explicit acknowledgement of missing historical comparisons.

The published direction labels describe synthetic behavioral expectations. No metric uses a particular invented completion time as ground truth for an individual. Fixed events and fallbacks are excluded from model-authored metrics.

### Copied advice

Of 13 warnings per model on the 20 published tasks, 4B copied the pairing demonstration without supporting pairing evidence seven times and 9B did so twelve times. Matching source IDs alone missed this. On a late-arrival meeting, a stock check and dishwashing, 4B repeated the demonstration instruction “Test the pairing before the run.” Each warning cited an actual note, but none of those notes describes pairing. The scorer records each exact copied instruction with no pairing evidence in its source, even when a later clause mentions the correct topic. The topic-plus-source count and the count excluding copied advice are both available.

The separate warning trial removes the demonstration sentence, asks for an action tied to the recorded obstacle, and cautions against attributing elapsed time to a window switch. The development meeting and dishwashing cases test the observed failure. A neutral stock check and a camera task with genuine controller-pairing failures exercise both directions. [Frozen system instructions](prompt-versions.json)

### An unsupported scope claim

In the integrated 4B forecast for **Implement first divergence for C and Rust traces**, the target explicitly requires registering the tool in the repair loop. The cited earlier task requires wiring its error envelope into the CLI and testing truncated tails. Neither record establishes a parser-registration step that the new task can skip. Nevertheless, the model wrote, “The confidence_reason is that the shared work of handling iterator termination and truncation remains consistent, though the new task skips the parser registration step, which may reduce time slightly.” It raised the budget from 85 to 90 minutes, earning direction credit despite that unsupported justification. This is a reasoning error independent of warning-source selection. [Integrated 4B forecast](integrated/bonsai-4b/patch-tool.json) · [Task and earlier task descriptions](../../../tasks/histories/02-migration-lab.json)

On **Record the cable-cover crossing on rover C**, 9B first returned 45 minutes with lower and upper bounds both equal to 45. Validation triggered a second request that returned a 40–50-minute interval. Its explanation then called the task's document a “historical document excerpt” even though the model received zero historical comparisons. The corrected interval passed validation, and the historical attribution was wrong. [Both 9B responses](revised/bonsai-9b/bump-crossing.json)

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

## Context overflow

With 80 user-selected timing records and associated document excerpts, each provider rejected the request before forecasting. On the shared servers, 4B counted 25,294 input tokens and 9B counted 25,302 against a 6,144-token slot. Both returned HTTP 400. All 80 selections survived, and the app attributed the resulting estimate to its historical calculation. The original 4B message only showed HTTP 400. The revised message names the context limit and counts. The app still requires a larger context or a smaller selection for a direct model result in this case. These probes are excluded from forecast-quality metrics. [First 4B response](context-bonsai-4b-before.json) · [Updated 4B response](context-bonsai-4b.json) · [9B response](context-bonsai-9b.json)

## Controlled confidence trial

On the first revised 4B pass, increasing the spread of the indexing history widened its interval from 22 to 30 minutes. Confidence went up from low to high. The separate confidence trial compares the warnings-only prompt with the same prompt plus one sentence about consistency of comparable durations. The independent transfer case counts the same 16 stock bins with the same procedure and three 30-minute budgets. Earlier times are 39/40/41 in the tight arm and 18/40/62 in the broad arm. We record interval widths and ordinal confidence. Ties are reported separately. Increased confidence or a narrower interval on the broader history is an inversion. These comparisons test sensitivity and do not establish probabilistic calibration.

## Reproduction

The component run used LM Studio's bundled CPU llama.cpp backend 2.66.0 with eight threads, three 6,144-token slots (18,432 total), zero GPU layers, temperature zero and reasoning disabled. The 4B model used a 256 MiB prompt cache. 9B used no mmap and no RAM prompt cache. The harness allows 3,600 seconds per request so queued work can complete. That timeout does not change the app setting.

With either model served locally, run the following from the repository root. Set the model alias and endpoint to the running server. Results are checkpointed per case in the chosen run directory. A transport failure stops the run. An invalid model reply and any corrective request remain in the case record.

```sh
python -m examples.estimation.forecast_audit --endpoint http://127.0.0.1:1243/v1/chat/completions --model ternary-bonsai-4b --run my-replay appendix-proof mara-rooms dev-april challenge
python -m examples.estimation.forecast_audit --endpoint http://127.0.0.1:1244/v1/chat/completions --model ternary-bonsai-2-9b --run my-replay appendix-proof mara-rooms dev-april challenge
python -m examples.estimation.score_forecast_audit
```

Use `--run warning-trial --warning-trial mara-rooms bake-dishes dev-april dark-frame-series` with the same endpoint and model. For the paired confidence control, use `--run confidence-control --warning-trial confidence`. For the one-sentence intervention, use `--run confidence-trial --confidence-trial confidence`. For the integrated retrieval comparison use `--run integrated --integrated patch-note patch-tool`. This calls the production search and per-candidate screening with the shipped Qwen and Kev outputs. A missing cached inference fails the check. To time one isolated call after other clients exit, use `--run solo --solo patch-tool`. Patch-tool had the median archived primary prompt size; the runner requires all server slots idle before and after the request.

To repeat the four warning cases with the complete combined instruction, run `--run confidence-trial-warning-panel-local --confidence-trial mara-rooms bake-dishes dev-april dark-frame-series` with each model and endpoint.

For a rerun of the currently installed application prompt, add `--production` and choose a fresh run such as `--run production-local`. The same actual model aliases and endpoints apply to this mode. To replay the archived combined instruction after future app changes, use `--confidence-trial --run confidence-trial-local` with the requested case names.

Use `published` in place of the named tasks to run all 20 original cases. Use a fresh run name when replaying on a later source revision (for example, `warning-trial-local` or `confidence-trial-local`). This preserves existing receipts while selecting the same frozen instruction. The labels, retry state, per-case predictions and model attribution stay attached to each result.
