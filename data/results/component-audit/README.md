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

## Runtime and inputs

The local checkpoints were Qwen3-Embedding-0.6B Q8_0, Kev-4B Q4_K_M, Ternary-Bonsai-4B Q2_0_g64 and Ternary-Bonsai-2-9B Q2_0. The runner used LM Studio’s bundled llama.cpp CPU backend. The forecast comparison pins the upstream candidates so that a different Qwen result cannot masquerade as a forecasting improvement. The separate three-open-task replay exercises the stages together. Public fixture caches contain actual input-keyed embeddings or decisions. Controlled failure and race tests explicitly inject responses and are reported separately.

## Reading the numbers

A candidate rejected by Qwen never reaches the historical Kev judge in the app. The retrieval report scores both stages and the combined result. Active-task pairs go directly to Kev. A positive matrix cell is assessed separately from whether Bonsai moved its estimate after receiving the completed peer's duration. Browser judgments, workspace eligibility and vector retrieval are shared by both forecast configurations.

The ten stored histories are synthetic and have been used in previous experiments. Some probes are deliberately near-duplicates to isolate the effect of one input. These controls share inputs, so the number of independent scenarios is smaller than the row count. The component reports identify their control groups, development and validation cases, model-authored outputs, reused exact model responses and deterministic tests. Exact times in the histories are examples. The direction, evidence and state-transition checks score how the application responds to that evidence.
