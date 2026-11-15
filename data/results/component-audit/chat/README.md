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

The first 4B run also returned an empty model ranking for a 20-minute plan. The app picked the correct fitting task through local budget arithmetic, so a fit-only metric passed. The first repair required a rank and explanation keyed to each open task ID and displayed them in the plan evidence. The later sequence experiment below tests whether that contract preserved the relationship between each task and its reason. An unavailable ranking explicitly falls back to local order and still computes the budget. The three-question priority rerun measures valid ranks with reasons and whether the model actually ranked the feasible task first.

Thirteen of fourteen new regression cases failed on c213b3f and all fourteen passed after repair. They cover record aggregation, time-window arithmetic, workspace progress and page selection directly. The local and model-backed scenarios measure different stages, including which supporting rows the model saw.

## Running the model checks

Set `PLANNER_MODEL` and `PLANNER_MODEL_ENDPOINT` to the Bonsai instance. Run the following from the repository root. Use a distinct output filename per model and revision.

```shell
python -m examples.estimation.audit_chat data/results/component-audit/chat/after-example.json
```

The before panel ran against commit `c213b3f` with the same audit driver copied into that checkout. To isolate the later prioritizer fix, pass `--case fit-20 --case budget-bare --case bare-half` and a fresh output path. The `priority-before` and `priority-final` results retain both rankings. The full panel uses non-planning answers from `after` and the three planning answers from `sequence-repair`. The intermediate per-task ranking contract and explicit open-plan check have separate receipts (`priority-final` and `scope-guard`). Running `python -m examples.estimation.audit_chat data/results/component-audit/chat --summarize` recomputes the combined metrics without inference.

Each checkpoint keeps the question, split, model, response, elapsed time and exact pass condition in the harness. Optional `--cache DIRECTORY` reuses a validated reply only when the endpoint, model ID, messages, schema and generation options are identical. Individual call records distinguish live and replayed outputs. The local models shared CPU inference slots with the other component evaluations; this run allowed 1,200 seconds per model request. Question wall times include queueing and identical prior calls can replay immediately.

To recompute exact passes and each applicable confusion matrix, run the same command with `--score`. A planning positive is an open task that fits the explicit budget. A distraction positive is a site named in the user-facing answer for the requested category. A PDF positive is a factual answer with a verified quotation from the specified document. Exact PDF passes additionally require the expected fact. Scalar analytics are evaluated against the measured query result and its answer; precision on a scalar has no defined meaning here.

## Live results

| Model | Five development answers before → final | Nine originally held-out answers, frozen → after targeted fixes | Task ranks with a reason, original → final (of 6) | Model ranked feasible task first, original → final (of 3) |
| --- | --- | --- | --- | --- |
| Bonsai 4B | 4/5 → 5/5 | 9/9 → 9/9 | 0/6 → 6/6 | 0/3 → 0/3 |
| Bonsai 9B | 4/5 → 5/5 | 8/9 → 9/9 | 4/6 → 6/6 | 2/3 → 3/3 |

The frozen score keeps the ordinary panel intact. The 9B miss in that panel led to the open-plan fix below. The repeated 25-minute question is therefore a targeted regression check in the final score. Both models returned the measured 12-minute mean underestimate for GPU work and the correct count of three completed tasks yesterday. Across all stages, 27 distinct 4B requests and 29 distinct 9B requests have successful live replies. Repeated identical requests are marked in the receipts. No chat requests recorded a transport error.

4B's original ranker returned an empty list for all three time budgets. After repair, it still placed the 60-minute kernel task ahead of the 15-minute receipt task for budgets of 20, 25 and 30 minutes. In the 20-minute case, the model invented a risk of blocking future work. In the other two cases its prose correctly said the kernel task could not fit, but it ranked that task first. Local cumulative-fit arithmetic placed only the receipts in the fitting section in all three rounds.

For each model in the final panel, the PDF reader returned all three supported facts with verified excerpts and declined all four unsupported requests (3 TP, 0 FP, 0 FN, 4 TN). Both category-scoped distraction questions named exactly the expected site (2 TP, 0 FP, 0 FN, 4 TN). For the recurring GPU distraction, the original answer identified the right site but split its activity into 3- and 4-minute one-offs. After the correction it reported 7 minutes across two tasks on two days. The time-budget questions used three wordings on the same two-task board; each checked one feasible and one infeasible task. These data do not measure plan quality on a larger board.

The first 9B header response was “4,” with a verified page quotation that explicitly states “four bytes.” The original answer checker required the unit in the answer itself and recorded a miss. The revised scoring rule accepts the bare count for this question across both models. `--score` recalculates PDF answer correctness from the unchanged replies. `scoring-notes.json` records the amendment.

### Budget query that selected completed history

For “I have 25 minutes. Which task fits?”, the original 9B router selected a historical SQL query. Its first query referenced a missing column and its repaired query returned “Profile a second warp,” a completed task. The frozen revised ranker could not help because the router never called it. `scope-before-ternary-bonsai-2-9b.json` replays the exact router, SQL repair and narrative through that version.

The follow-on check redirects an explicit prospective task-fit question with a work budget or clock cutoff to the open plan when the model selects historical data. The model still ranks the actual open tasks. The UI records the proposed tool and the scope correction. The 9B targeted rerun selected the paused “File receipts” (15 minutes remaining). Its two model reasons described the opposite task IDs, and the numeric rank still put “Kernel lab” first. The local time-budget check put the receipts alone in the fitting section. On the same question, 4B kept its model-selected planning route and its original ranking. `scope-guard-*.json` records both outcomes.

### Task sequence, reason binding and budget arithmetic

The two tasks are always “Kernel lab” (ID 1, 60 minutes remaining) and “File receipts” (ID 2, 15 minutes remaining). At each budget, only ID 2 fits. The three questions are paraphrases on this single board. A reason counts as present when the model returns text for a valid task ID. Feasible-first also requires the model to place ID 2 uniquely first. The page result checks the fitting task and the available minutes independently.

| Model and intervention | Task IDs with reasons (of 6) | Model placed feasible task first (of 3) | Page returned correct plan (of 3) |
| --- | ---: | ---: | ---: |
| 4B, original sequence | 0 | 0 | 3 |
| 4B, required per-ID ranks | 6 | 0 | 3 |
| 4B, per-ID ranks with scope check | 6 | 0 | 3 |
| 4B, original sequence plus one repair when incomplete | 6 | 0 | 3 |
| 9B, original sequence | 4 | 2 | 2 |
| 9B, required per-ID ranks | 4 | 0 | 2 |
| 9B, per-ID ranks with scope check | 6 | 0 | 3 |
| 9B, original sequence plus one repair when incomplete | 6 | 3 | 3 |

The required per-ID format introduced a 9B regression. With 20 minutes available, the model assigned its explanation of why task 2 fits to task 1 and assigned the explanation about task 1 overrunning to task 2. The same reversal occurred for the 25- and 30-minute questions once all three used the open plan. The app now accepts a complete ordered list with one reason for each ID, and makes a single request using required per-ID fields if the first reply omits or duplicates tasks, names an unknown ID, or cannot be validated. An unavailable endpoint still returns the local budget calculation. The app records the ranking source alongside the result.

On this board the final 9B sequence put the right reason on all six tasks and placed the receipts first all three times. For the formerly misrouted 25-minute question its new live reply compared 15 to 25 on ID 2 and 60 to 25 on ID 1. The other final ranking calls replayed previously captured, request-identical replies. Each 4B question needed the second call, which recovered the six explanations while retaining the rank-order failure above. Including local fit arithmetic is important for both time limits and interpreting this score. The `sequence-repair-*.json` files record each individual response and its live or replayed source.

After integrating the scope and ranking fixes, replaying all six questions through the production entry point reproduced each recorded request hash and the complete displayed result. The generation check also runs before any ranking repair and before rendering its reply. Three race tests cover deleting or excluding data while a complete sequence, incomplete sequence or repaired sequence is in flight.
