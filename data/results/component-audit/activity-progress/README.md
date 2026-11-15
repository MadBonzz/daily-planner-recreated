# Activity classification and workspace progress at pause

The activity classifier predicts one of three labels per foreground interval. The pause forecaster makes a different decision, whether the chat contains new evidence that supports updating the task. Neither score uses an invented duration as ground truth.

## Tests and labels

The development and validation examples are in [the audit runner](../../../../examples/estimation/audit_activity_progress.py). Each split has eight visible activity intervals and four pause conversations. Development covers a usable reference and an unrelated video on the same domain, mixed device signals, a genuine blocker and hypothetical progress. Validation covers a different task domain, a copied instruction, a question after an earlier verified milestone, a new test result and explicit completion.

The additional SQLite replay uses one completed task per packaged history. It chooses the task with the most distinct model-eligible labels, breaking ties in favor of an off-task example, then fewer intervals, then fixture order. It loads the task and PDF context available when work began. The classifier receives sanitized ActivityWatch rows. The fixture annotations stay in the scoring process. This yields 56 model-eligible intervals, with 42 annotated as work, 11 as distraction and three as uncertain. Of these, 51 come from the laptop and five from the phone. The per-class scores treat each interval as one observation. They exclude time mechanically marked uncertain for lack of observations, privacy settings or laptop idle time. Reports retain those intervals for timer accounting. For model-eligible intervals, each output records how many received a valid model label. An unavailable reply yields uncertain in the app, which can coincide with an uncertain reference label.

**Positive** in the pause audit means the user directly established progress, a blocker or remaining work. A model failure leaves the previous estimate in place. The binary metric counts that as no revision, and each results file also reports how many assessments were unavailable. A good classification does not establish that the exact remaining-minute estimate was accurate.

## Report corrections and clock accounting

Across all 10 histories, [the accounting run](report-accounting.json) checked 79 completed tasks and 360 intervals against their recorded timer lengths. Of those tasks, 67 had editable foreground evidence. The audit corrected one activity group per editable task, changing 78 intervals. Applying edits to the stored report and rebuilding it from SQLite produced identical totals. The checks also verified that the corrections left elapsed time, switch counts, unrelated windows and excluded intervals unchanged. This component is deterministic and ran once for both models.

## Reproduced failures

| Scenario | At `c213b3f` | Changed behavior |
| --- | --- | --- |
| An interval exhausts the classifier's output budget in the first of multiple batches | All ten test intervals became uncertain | The classifier subdivides that batch. Nine intervals get labels and the irrecoverable interval stays uncertain with a warning. |
| One synced phone observation has malformed JSON | No diagnostic for missing phone evidence | The Android source uses the same row validation as the desktop reader. The report identifies the invalid event and retains the valid neighboring phone observation. |
| A user repeats a milestone from an earlier workspace session | A model could cite the same words again as fresh progress | An adjusted forecast needs a new user citation beyond the earlier evidence. |
| The user posts a long test trace followed by a progress statement | Head-only truncation hid the actual progress statement | Both ends of a long user message reach the model within the same input budget. Exact citations still have to match the original. |
| The model cites “finished the proof” inside “I have not yet finished the proof” | The shortened citation could justify lowering the estimate | The citation must retain the nearby negation, or the forecast is rejected and retried. |
| The model cites the full negated statement and labels it completed | The exact quote passed validation | The model must describe the open work or cite an actual completed step. |
| The model returns zero remaining minutes after only a partial milestone | The point forecast could drop to the time already recorded | Zero requires a fresh citation explicitly identified as completion of the whole task. |
| The model supplies a valid integer-valued JSON number such as `1.0` as its citation ID | A correct turn number was rejected | An integral number is normalized before indexing. Booleans and nonintegral numbers still fail validation. |

In the live development run, 4B called simultaneous work in a code editor and TikTok on a phone definite work. Its rationale only mentioned the editor. For simultaneous non-idle laptop and phone evidence the model now labels each device separately. The app counts the interval once and shows uncertain if the two assessments differ. An idle laptop does not override the foreground phone.

The eight regression cases above fail on the original revision and pass after the changes. They exercise parser and validator failure paths directly. The model accuracy measurements use live calls. The existing report-worker tests also cover correcting an interval, recomputing totals, showing the corrected label as a future example, avoiding excluded examples, and rejecting a stale generation.

## Follow-up checks

The opaque-dashboard test exposed an error beyond ordinary interval labeling. The 4B model correctly reused both earlier user corrections during another clinic stock task, but it invented work on three dashboards with no correction and carried two prior labels into an unrelated hiking task. For one of those windows the only observed title was “Dashboard,” while the model described football highlights.

The follow-up replays the *saved* classifier replies through a rule for generic browser pages. A specific window title, phone context, explicit override, an identified host in the task or rationale, or a matching past correction on comparable work can retain a definite label. Otherwise the app records uncertainty. This is a targeted diagnostic on the observed failure. The frozen live results in `after/` do not incorporate this rule. It does not remedy every opaque native application or independently verify everything a model says about a named site.

A separate retry experiment addresses a model returning a supported pause decision with an empty citation array. It keeps the initial call unchanged. The second call requires one citation object, and the validator still checks the precise user quote, its role, whether it is fresh and whether the minutes contradict it. It was rerun on the two identified misses; the report keeps those diagnostic results separate from the original held-out counts.

## Reproduction

For the before run, check out `c213b3f` in a separate worktree and copy the current audit runner into its `examples/estimation/` directory. This keeps the cases constant and imports the production code from the selected revision. Set `PLANNER_MODEL` and `PLANNER_MODEL_ENDPOINT` for the desired local Bonsai checkpoint. Use a different output file for each checkpoint, split and revision. From the repository, run the components individually.

```sh
python -m examples.estimation.audit_activity_progress activity-validation.json --split validation --component activity
python -m examples.estimation.audit_activity_progress progress-validation.json --split validation --component progress
python -m examples.estimation.audit_activity_progress accounting.json --split validation --component accounting
python -m examples.estimation.audit_activity_progress history-replay.json --split validation --component replay
python -m pytest -q tests/test_activity.py tests/test_activity_db.py tests/test_activity_fixture.py tests/test_report_worker.py tests/test_workspace_forecast.py tests/test_workspace_progress_scenarios.py
```
