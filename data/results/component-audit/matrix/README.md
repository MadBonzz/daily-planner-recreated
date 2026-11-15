# Active-task matrix and selective re-estimation

A positive active-task decision means that the duration of either task could inform the other. The app stores one symmetric cell per co-active pair. It can leave a cell pending while Kev is unavailable. A newly completed task contributes only after a positive judgment and a recorded duration.

## Evidence reaching the forecast

We replayed the ten published schedules up to their recorded completion times, using their packaged Qwen vectors and Kev answers. There were eight positive co-active pairs in which at least one flexible task subsequently finished. Under the prior scheduling and retrieval rules, all eight would trigger a refresh and three of the eight sources would then fail the historical similarity cutoff. The remaining task would receive a new forecast without the timing evidence that had triggered it.

| Completed source | Remaining task | Instructed cosine | Bare cosine |
| --- | --- | ---: | ---: |
| Port the wraparound header reader | Implement the bounded mismatch reporter | 0.450 | 0.475 |
| Work out the epsilon-floor variance lemma | Verify the weighted-bin bound | 0.464 | 0.498 |
| Debug a mismatched pitch in the strided kernel | Make the shared scratch race reproducible | 0.386 | 0.466 |

The app now passes the recorded positive or negative Kev verdict through to the forecaster when a peer finishes. A user can override a verdict. Other historical comparisons still go through Qwen at the configured similarity floor and then Kev. The final model remains free to exclude a supplied comparison from its citations. A scheduled fixed-duration event enters the evidence only when the user explicitly chooses it.

The archived results measure consistency between stages. A positive Kev answer in an existing schedule is not an independently verified relevance label. [Per-pair values](archived-peer-decisions.json)

## Insertion order

The ten published schedules had 112 actual co-active pairs. Both directions of Kev's original call were available for 32. Three of those pairs returned opposite answers when we swapped the arguments. The original matrix sent the newest task first, so otherwise identical plans could acquire different matrix values.

For newly judged pairs we now put the two tasks in a stable order using their title, description, details and PDF text. User-reviewed examples for an active pair come from the same two categories in either order. Historical retrieval still asks whether a particular completed task can inform a particular new task. Already saved decisions and user corrections remain intact.

Across the 112 actual overlaps, reversing insertion order produced the same model input for all 112 under the patched policy. For 41 pairs only the opposite direction had been packaged previously. The planner consistently uses that exact saved answer whether a model is running or offline. New tasks use the canonical order. Consistency here measures reproducibility; relevance precision and recall use the labeled boards. [Archived orientation results](archived-orientation.json)

## Queue and memory scenarios

We copied the same ten regression checks to the prior checkout at `c213b3f` and ran them there and against the changes. Eight failed on the prior checkout. All ten passed after the fixes. The tests use real SQLite transactions and the normal estimate worker with controlled model responses. The existing queue, timer and report suites also passed.

| Scenario | Behavior checked |
| --- | --- |
| Three open tasks; one peer finishes in 10 or 90 minutes against a 30-minute budget | Completing the unrelated task leaves the original forecast untouched. Completing the related task sends its actual time, excludes the unrelated source, and saves the new forecast separately from the initial one. Both directions run as separate tests. |
| Two sources finish before all matrix cells are resolved | The worker processes other immediately due decisions for the same target before its refresh, so both durations fit into one forecast. |
| Kev is offline for one of those cells | The known positive source can update the target immediately. The unresolved cell backs off and gets a fresh decision later. |
| A scheduled event completes, then the user selects it manually | Automatic re-estimation ignores the scheduled event. Its later time corrections propagate to an explicit selection. |
| A report is rebuilt while a related forecast is in progress | Saving equivalent report content preserves the active forecast lease. Changed evidence still requests a refresh. |
| The same two same-title tasks are inserted in reverse order with reviewed examples | Kev receives the exact same structured payload, including the user feedback, despite different task IDs. |
| Only the reverse direction exists for an old packaged sample | Both insertion orders use the same archived answer without model inference. |
| Twelve seeded schedules interleave seven submissions, four completions, starts and worker ticks | Every creation forecast uses just the history available at submission. The final matrix and open-task forecast evidence agree with a separately maintained schedule. This run isolates transaction behavior using controlled model responses. |
| A user forgets a feedback example used by Kev | Memory deletion revokes the forecast, retires the old verdict, and removes the comparison when the new judgment rejects it. |

[Scenario tests](../../../../tests/test_matrix_lifecycle.py)

## Reproduction

From the repository root, set the standard local model endpoints. The matrix judge takes a new task and one active peer at a time. The five boards contain 30 unordered pairs with prespecified relevance labels. Their labels and development/validation assignments were written before running Kev. The boards cover same-title/different-work and analogous tasks with different wording. They give limited evidence about tasks connected only by a recurring obstacle. The initial forecasts in this stage are seeded so that the result isolates the active-task decision.

```sh
python -m examples.estimation.audit_matrix data/results/component-audit/matrix/kev-baseline.json
python -m examples.estimation.audit_matrix data/results/component-audit/matrix/kev-method-trial.json --method
python -m examples.estimation.audit_matrix data/results/component-audit/matrix/kev-canonical.json --canonical --reference data/results/component-audit/matrix/kev-baseline.json
python -m examples.estimation.audit_matrix data/results/component-audit/matrix/archived-orientation.json --archived-orientation
python -m examples.estimation.audit_matrix data/results/component-audit/matrix/archived-peer-decisions.json --archived-peers
python -m examples.estimation.audit_matrix data/results/component-audit/matrix/lifecycle-inputs.json --prepare
python -m examples.estimation.audit_matrix data/results/component-audit/matrix/lifecycle-4b.json --lifecycle --inputs data/results/component-audit/matrix/lifecycle-inputs.json
python -m pytest -q tests/test_matrix_lifecycle.py
```

Change the forecast model and output filename for the 9B lifecycle. The prep step records live Qwen and Kev responses keyed by the complete input payload. Each lifecycle uses a fresh store, reuses these exact responses, and calls the configured Bonsai forecaster. The two initial control cards with saved 30-minute forecasts, an unrelated historical bicycle repair, completion times, and a note about signing-token access are scenario inputs.

## Live run results

### Kev decisions on labeled active tasks

The development boards contain three positive and nine negative pairs. The validation boards contain four positive and fourteen negative pairs. Reference labels identify comparable work in the synthetic schedules.

| Configuration | Split | TP | FP | TN | FN | Precision | Recall |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Original insertion order | Development | 3 | 0 | 9 | 0 | 100% | 100% |
| Original insertion order | Validation | 4 | 0 | 14 | 0 | 100% | 100% |
| Timing-method prompt trial | Development | 3 | 0 | 9 | 0 | 100% | 100% |
| Timing-method prompt trial | Validation | 4 | 0 | 14 | 0 | 100% | 100% |
| Production stable order | Development | 3 | 0 | 9 | 0 | 100% | 100% |
| Production stable order | Validation | 4 | 0 | 14 | 0 | 100% | 100% |

All 20 additions per configuration created exactly one pending cell per previous open task and preserved existing decisions. Each step, request hash, probability and stored verdict is in the run file. [Original order](kev-baseline.json) · [Prompt trial](kev-method-trial.json) · [Stable order](kev-canonical.json)

### Three-task lifecycle with live Bonsai forecasts

A release memo and a package-signing task were open when a second package-signing task was added with a 30-minute budget. The independent memo finished in 75 minutes, then the related signing task finished in 90 minutes. Each model saw the same recorded Qwen embeddings and Kev verdicts. Only the new task received live Bonsai time forecasts.

| Model | Initial total | After unrelated completion | After related completion | Cited the 90-minute signing task | Model-authored forecasts |
| --- | ---: | ---: | ---: | --- | ---: |
| Bonsai 4B | 45 min | 45 min | 60 min | Yes | 2/2 |

The relevant completion tests whether the actual reaches the forecaster and whether the forecast moves in the expected direction. The new task has no labeled actual duration. [Frozen Qwen/Kev responses](lifecycle-inputs.json) · [4B transitions and receipts](lifecycle-4b.json) · [9B transitions and receipts](lifecycle-9b.json)
