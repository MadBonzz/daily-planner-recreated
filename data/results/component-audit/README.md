# Plan Smarter component evaluation

The reference app is [c213b3f](https://github.com/MadBonzz/daily-planner-recreated/tree/c213b3f). Each experiment asks a separate question. Relevance means that one task's duration or obstacle can inform another task. A website suggestion additionally requires evidence about that website and whether the current task needs it. The same site or task can legitimately have different labels in these two evaluations.

| Component | What was scored |
| --- | --- |
| [Qwen candidate retrieval and Kev historical comparisons](retrieval/) | Threshold admissions and rejections on 143 labeled pairs, top result, query direction, model failures, user-reviewed evidence and offline cache fidelity |
| [PDF and metadata changes](document-inputs/) | Real PDF uploads with changed content, shared handouts with distinct assignments, page and section scope, controls for category and filename, index replacement and deletion |
| [Active-task matrix and completion queue](matrix/) | Every cell during board construction, selection of the task to refresh, revisions, corrections, retries, input order and the evidence supplied to the forecast model |
| [Bonsai forecasts](forecast/) | Direction of adjustments, cited evidence, grounded cautions, uncertainty, fixed events, fallback attribution and controlled changes to historical durations |
| [Website blocking and workspace eligibility](focus-workspace/) | False blocks, missed distractions, mixed-use domains, unavailable materials, user edits and prediction changes while a timer is running |
| [Activity reports and pause estimates](activity-progress/) | On-task, off-task and uncertain labels, device attribution, manual corrections, evidence of completed work and remaining-time updates |
| [Planner chat, analytics and PDF answers](chat/) | Task and site selections, date filters, arithmetic, supported answers, unsupported questions and page citations |
| [Clock boundaries and review reminders](time-boundaries/) | Crossing midnight, spring gaps, repeated fall times, real elapsed budgets and reminder deduplication in Ithaca |

## Which changes made a difference?

| Decision or scenario | Original | Revised | Interpretation |
| --- | --- | --- | --- |
| Qwen candidates on 143 labeled comparisons | 39 TP, 24 FP, 9 FN | 43 TP, 26 FP, 5 FN | Adding a bare query alongside the instructed query recovered four candidates at the 0.50 floor. |
| Qwen and Kev together on those 143 | 28 TP, 1 FP, 20 FN | 31 TP, 1 FP, 17 FN | Precision went from 96.6% to 96.9%, recall from 58.3% to 64.6%. Two gains came from previously inspected parts of a shared assignment. |
| Assigned work in 56 controlled PDF pairs | 25 TP, 0 FP, 3 FN | 28 TP, 0 FP, 0 FN | The original three misses reproduce one CUDA match under changes to prose, category and filename. Both versions passed all 28 pairs in the second packet. |
| Active-task decisions on five labeled boards | 7 TP, 0 FP, 0 FN | 7 TP, 0 FP, 0 FN | Stable ordering preserved accuracy. In ten dated schedules, three of eight positive peers previously dropped out of the forecast that their completion had triggered. |
| Website blocking on eight held-out cases | 3 TP, 2 FP, 0 FN | 3 TP, 0 FP, 0 FN | Required Twitch moderation and two generic New Tab visits account for the two false blocks. |
| Workspace eligibility on ten held-out cases | 4 TP, 0 FP, 0 FN | 4 TP, 0 FP, 0 FN | A development case fixed separately prevents offering a document reader for a PDF on another workstation. |
| Fifteen Ithaca clock scenarios | 6 passed | 15 passed | Tests cover DST gaps and repeated times, following up with a new time budget and reviewing the right day after a restart. |

The tags TP, FP and FN count accepted relevant items, accepted irrelevant items and missed relevant items, respectively. The component tables include true negatives and their denominators. The original Kev question outperformed its more narrowly worded alternative on the historical validation cases. Assigning a precise section of a PDF and changing the wording of a judge are separately measured interventions.

## Where the two Bonsai models differed

| Decision and test set | Bonsai 4B | Bonsai 9B |
| --- | --- | --- |
| Forecast moved in the expected direction, 18 labeled cases in the revised run | 12/18 | 15/18 |
| Direction on cases with a model answer in both the previous and revised run | 3/15 → 9/15 | 8/18 → 15/18 |
| Timing evidence cited on the revised run, precision / recall among supplied records | 94% / 54% | 96% / 96% |
| Adjustment direction on six fresh controls with a specified direction | 6/6 | 6/6 |
| Warnings on the fresh panel, TP / FP / FN | 1 / 2 / 0 | 1 / 3 / 0 |
| Activity label on 56 intervals from 10 native SQLite histories | 53/56 raw, 56/56 after the evidence check | 51/56 raw, 53/56 after the evidence check |
| Distraction on those intervals after the evidence check | 11 TP, 0 FP, 0 FN | 9 TP, 1 FP, 2 FN |
| Workspace-pause decisions on the two missed positive cases and three negative controls, final application | 2 TP, 0 FP, 0 FN, 3 TN | 2 TP, 0 FP, 0 FN, 3 TN |
| Model ranked the fitting open task first, three budgets on one board | 0/3 | 3/3 |
| User-facing list contained the correct fitting task on that board | 3/3 | 3/3 |

Nineteen of the 20 previously used tasks needed a model forecast and 18 had direction labels. Both revised models authored all 19. The frozen evaluation uses the same candidates in both forecast arms. With a 40-minute budget and only past durations changed from slower to faster, 4B moved from 58 to 22 minutes and 9B from 62 to 26. In a separate end-to-end check, recovering the suite-report comparison moved estimates for the archived-transcript write-up from 45 to 29 minutes for 4B and 40 to 25 for 9B. The target calls for less work than the historical implementation assignment in the same packet. [Frozen and integrated forecast results](forecast/metrics.md)

Both checkpoints produced valid classifications for all 56 eligible intervals. A blank Thunderbird inbox was the complete evidence on three rows. Replaying the same model responses with the sparse-evidence check corrected three false distractions for 4B and two for 9B. On the planner questions, Python checks cumulative available time separately from the model's ranking. That second check kept an oversized task out of the fitting list even when 4B put it first. [Activity errors and controls](activity-progress/) · [Planner chat stages](chat/)

On the pause follow-up, each model accepted the user's explicit completion at zero remaining minutes and revised a partial positive test result to 40 minutes remaining. Both attempted to reuse an old milestone on one negative control; quote validation preserved the earlier forecast. This diagnostic revisited known misses and cannot establish performance on new conversations. [Pause trials and per-case decisions](activity-progress/README.md)

The full forecast table measures the frozen revision. A later prompt change removed a hardware-specific example and asked the model to account for variation in comparable durations. We tested the exact release wording in four warning scenarios and four duration controls per model. Both sizes stopped copying pairing advice onto the three unrelated tasks and retained it for the camera task. Confidence decreased on the more variable history in three of four model–task pairs, compared with one of four under the warning-only wording. A sourced warning could still assert a causal link or claim that activities overlapped when the log only placed them in the same session. [Prompt versions](forecast/README.md) · [Manual warning review](forecast/warning-review.md)

## What happened when an active task finished?

Three open tasks started with 30-minute budgets. Kev related the signing and verification tasks. The memo was independent. Finishing the memo in 75 minutes queued zero re-estimates. Finishing signing in 90 minutes queued exactly one for verification. Both models cited the completed signing task.

| Estimate for the remaining verification task | At creation | After the independent memo | After the related signing task |
| --- | ---: | ---: | ---: |
| Bonsai 4B | 45 min | 45 min | 60 min |
| Bonsai 9B | 30 min | 30 min | 120 min |

This checks whether the right task changed and what evidence reached it. The verification task has no measured outcome to say whether 60 or 120 is closer. Under shared CPU load, the first 9B refresh timed out after 900 seconds and produced a labeled 40-minute fallback. Retrying that request with the same prior state produced the 120-minute model forecast. [Three-step replay and timeout record](matrix/README.md)

## Runtime and inputs

The local checkpoints were Qwen3-Embedding-0.6B Q8_0, Kev-4B Q4_K_M, Ternary-Bonsai-4B Q2_0_g64 and Ternary-Bonsai-2-9B Q2_0. The runner used LM Studio’s bundled llama.cpp CPU backend. The forecast comparison pins the upstream candidates so that a different Qwen result cannot masquerade as a forecasting improvement. The separate three-open-task replay exercises the stages together. Public fixture caches contain actual input-keyed embeddings or decisions. Controlled failure and race tests explicitly inject responses and are reported separately.

## Reading the numbers

A candidate rejected by Qwen never reaches the historical Kev judge in the app. The retrieval report scores both stages and the combined result. Active-task pairs go directly to Kev. A positive matrix cell is assessed separately from whether Bonsai moved its estimate after receiving the completed peer's duration. Browser judgments, workspace eligibility and vector retrieval are shared by both forecast configurations.

The ten stored histories are synthetic and have been used in previous experiments. Some probes are deliberately near-duplicates to isolate the effect of one input. These controls share inputs, so the number of independent scenarios is smaller than the row count. The component reports identify their control groups, development and validation cases, model-authored outputs, reused exact model responses and deterministic tests. Exact times in the histories are examples. The direction, evidence and state-transition checks score how the application responds to that evidence.
