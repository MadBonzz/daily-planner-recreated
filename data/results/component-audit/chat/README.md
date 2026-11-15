# Shared planner chat and document QA

The live probe sends ordinary questions through `ask_analytics` from the shared Tasks/Analytics chat and through `task_workspace.answer` for the selected attachment. Answers are checked against a known clock, four completed tasks, four activity intervals, the open plan and three PDFs already in the sample histories. No oracle duration estimates are needed. The question panel and answer checks live in `examples/estimation/audit_chat.py`. Development and held-out cases are identified in each run.

## Reproduced defects

| Component | Failing scenario at c213b3f | Correction |
| --- | --- | --- |
| Analytics aggregation | On two separate GPU tasks, two different browser tabs on the same sports domain took 3 and 4 minutes. Grouping by window and app made them look like two one-time distractions. | The site row shows 7 minutes across 2 tasks and 2 days. Unrelated windows without domains and simultaneous phone activity keep their own identities. |
| Date-scoped analytics | The router could select an all-dates accuracy summary for “on November 12.” Date-specific historical questions need a query capable of applying a date predicate. | ISO dates, month-first dates and day-first dates use the same date-aware routing contract as relative dates. |
| Time available | An explicit end of 12:00 AM produced zero minutes left at 6 PM. A client clock in a different timezone from the server used the server's cutoff. | Local wall times use the supplied zone. Midnight is the end of the plan day. A date with a daylight-saving transition keeps its actual duration. |
| Paused workspace progress | A task paused after 12.5 minutes had a 20-minute remaining forecast. Rounding its displayed total to 33 and subtracting the recorded time rounded the remainder to 21. | A 20-minute work window fits that task. Later activity reduces the remaining forecast from its pause snapshot. Zero left prompts for confirmation. |
| Reading workspace | A question asking for evidence specifically from page 2 sent all pages of a short PDF. An earlier page could supply an answer to a question the requested page could not answer. | Explicit page requests scope eligible passages and verifiable citations. An unavailable page gets a local answer. |

A separate PDF regression supplied a fabricated deadline and the exact word “the” as its citation. The previous check accepted it. Citation validation now requires a phrase from the cited page (or the whole readable passage for very short sources), and requests a correction on failure. This check verifies quotation and scope. Semantic support is still a model judgment.

Eleven of twelve new regression cases failed on c213b3f and all twelve passed after repair. They cover record aggregation, time-window arithmetic, workspace progress and page selection directly. The local and model-backed scenarios measure different stages, including which supporting rows the model saw.

## Running the model checks

Set `PLANNER_MODEL` and `PLANNER_MODEL_ENDPOINT` to the Bonsai instance. Run the following from the repository root. Use a distinct output filename per model and revision.

```shell
python -m examples.estimation.audit_chat data/results/component-audit/chat/after-example.json
```

Each checkpoint keeps the question, split, model, response, elapsed time and exact pass condition in the harness. Optional `--cache DIRECTORY` reuses a validated reply only when the endpoint, model ID, messages, schema and generation options are identical. Individual call records distinguish live and replayed outputs.

To recompute exact passes and each applicable confusion matrix, run the same command with `--score`. A planning positive is an open task that fits the explicit budget. A distraction positive is a site named in the user-facing answer for the requested category. A PDF positive is a factual answer with a verified quotation from the specified document. Exact PDF passes additionally require the expected fact. Scalar analytics are evaluated against the measured query result and its answer; precision on a scalar has no defined meaning here.
