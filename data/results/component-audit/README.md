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

## Runtime and inputs

The local checkpoints were Qwen3-Embedding-0.6B Q8_0, Kev-4B Q4_K_M, Ternary-Bonsai-4B Q2_0_g64 and Ternary-Bonsai-2-9B Q2_0. The runner used LM Studio’s bundled llama.cpp CPU backend. The forecast comparison pins the upstream candidates so that a different Qwen result cannot masquerade as a forecasting improvement. The separate three-open-task replay exercises the stages together. Public fixture caches contain actual input-keyed embeddings or decisions. Controlled failure and race tests explicitly inject responses and are reported separately.

## Reading the numbers

A candidate rejected by Qwen never reaches the historical Kev judge in the app. The retrieval report scores both stages and the combined result. Active-task pairs go directly to Kev. A positive matrix cell is assessed separately from whether Bonsai moved its estimate after receiving the completed peer's duration. Browser judgments, workspace eligibility and vector retrieval are shared by both forecast configurations.

The ten stored histories are synthetic and have been used in previous experiments. Some probes are deliberately near-duplicates to isolate the effect of one input. These controls share inputs, so the number of independent scenarios is smaller than the row count. The component reports identify their control groups, development and validation cases, model-authored outputs, reused exact model responses and deterministic tests. Exact times in the histories are examples. The direction, evidence and state-transition checks score how the application responds to that evidence.
