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

Both chat modes also crashed on a schema-valid integer choice spelled `1.0`. Priority and distraction IDs now use integer indexing after response validation. Unknown and duplicate IDs cannot add a task or finding.

The first 4B run also returned an empty model ranking for a 20-minute plan. The app picked the correct fitting task through local budget arithmetic, so a fit-only metric passed. The ranking now requires a rank and explanation keyed to each open task ID and shows them in the plan evidence. An unavailable ranking explicitly falls back to local order and still computes the budget. The three-question priority rerun measures valid ranks with reasons and whether the model actually ranked the feasible task first.

Thirteen of fourteen new regression cases failed on c213b3f and all fourteen passed after repair. They cover record aggregation, time-window arithmetic, workspace progress and page selection directly. The local and model-backed scenarios measure different stages, including which supporting rows the model saw.

## Running the model checks

Set `PLANNER_MODEL` and `PLANNER_MODEL_ENDPOINT` to the Bonsai instance. Run the following from the repository root. Use a distinct output filename per model and revision.

```shell
python -m examples.estimation.audit_chat data/results/component-audit/chat/after-example.json
```

Each checkpoint keeps the question, split, model, response, elapsed time and exact pass condition in the harness. Optional `--cache DIRECTORY` reuses a validated reply only when the endpoint, model ID, messages, schema and generation options are identical. Individual call records distinguish live and replayed outputs.

To recompute exact passes and each applicable confusion matrix, run the same command with `--score`. A planning positive is an open task that fits the explicit budget. A distraction positive is a site named in the user-facing answer for the requested category. A PDF positive is a factual answer with a verified quotation from the specified document. Exact PDF passes additionally require the expected fact. Scalar analytics are evaluated against the measured query result and its answer; precision on a scalar has no defined meaning here.

## Live results

| Model | Five development answers before → after | Nine held-out answers | Ranks with a reason before → after (six slots) | Feasible task ranked first before → after (three questions) |
| --- | --- | --- | --- | --- |
| Bonsai 4B | 4/5 → 5/5 | 9/9 | 0/6 → 6/6 | 0/3 → 0/3 |

4B's original ranker returned an empty list for all three time budgets. After repair, it still placed the 60-minute kernel task ahead of the 15-minute receipt task for budgets of 20, 25 and 30 minutes. In the 20-minute case, the model invented a risk of blocking future work. In the other two cases its prose correctly said the kernel task could not fit, but it ranked that task first. Local cumulative-fit arithmetic placed only the receipts in the fitting section in all three rounds.

Across the final 4B panel, the PDF reader returned all three supported facts with verified excerpts and declined all four unsupported requests (3 TP, 0 FP, 0 FN, 4 TN). Both category-scoped distraction questions named exactly the expected site (2 TP, 0 FP, 0 FN, 4 TN). For the recurring GPU distraction, the original answer identified the right site but split its activity into 3- and 4-minute one-offs. After the correction it reported 7 minutes across two tasks on two days. The time-budget questions used three wordings on the same two-task board; each checked one feasible and one infeasible task. These data do not measure plan quality on a larger board.
