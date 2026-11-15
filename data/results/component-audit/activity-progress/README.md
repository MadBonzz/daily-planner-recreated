# Activity classification and workspace progress at pause

The activity classifier predicts one of three labels per foreground interval. The pause forecaster makes a different decision, whether the chat contains new evidence that supports updating the task. Neither score uses an invented duration as ground truth.

## Tests and labels

The development and validation examples are in [the audit runner](../../../../examples/estimation/audit_activity_progress.py). Each split contains eight visible activity intervals and four pause conversations. Development covers a usable reference and an unrelated video on the same domain, mixed device signals, a genuine blocker and hypothetical progress. Validation covers a different task domain, a copied instruction, a question after an earlier verified milestone, a new test result and explicit completion.

The additional SQLite replay uses one completed task per packaged history. It chooses the task with the most distinct model-eligible labels, breaking ties in favor of an off-task example, then fewer intervals, then fixture order. It loads the task and PDF context available when work began. The classifier receives sanitized ActivityWatch rows. The fixture annotations stay in the scoring process. This yields 56 model-eligible intervals, with 42 annotated as work, 11 as distraction and three as uncertain. Of these, 51 come from the laptop and five from the phone. The per-class scores treat each interval as one observation. They exclude time mechanically marked uncertain for lack of observations, privacy settings or laptop idle time. Reports retain those intervals for timer accounting. For model-eligible intervals, each output records how many received a valid model label. An unavailable reply yields uncertain in the app, which can coincide with an uncertain reference label.

**Positive** in the pause audit means the user directly established progress, a blocker or remaining work. A model failure leaves the previous estimate in place. The binary metric counts that as no revision, and each results file also reports how many assessments were unavailable. A good classification does not establish that the exact remaining-minute estimate was accurate.

## Report corrections and clock accounting

Across all 10 histories, [the accounting run](report-accounting.json) checked 79 completed tasks and 360 intervals against their recorded timer lengths. Of those tasks, 67 had editable foreground evidence. The audit corrected one activity group per editable task, changing 78 intervals. Applying edits to the stored report and rebuilding it from SQLite produced identical totals. The checks also verified that the corrections left elapsed time, switch counts, unrelated windows and excluded intervals unchanged. This component is deterministic and ran once for both models.

## Frozen model runs

Each activity split contains three work, three distraction and two uncertain intervals. TP/FP/FN/TN below are one-versus-rest counts on foreground intervals.

| Model | Split | Correct before → after | Distraction TP/FP/FN/TN after | Uncertain TP/FP/FN/TN after |
| --- | --- | --- | --- | --- |
| 4B | Development | 6/8 → 8/8 | 3/0/0/5 | 2/0/0/6 |
| 4B | Validation | 6/8 → 7/8 | 3/1/0/4 | 1/0/1/6 |
| 9B | Development | 6/8 → 8/8 | 3/0/0/5 | 2/0/0/6 |
| 9B | Validation | 8/8 → 8/8 | 3/0/0/5 | 2/0/0/6 |

The ten-profile SQLite replay used the frozen classifier. The 56 eligible intervals included 42 work, 11 distraction and three uncertain observations.

| Model | Label | TP | FP | FN | TN | Precision | Recall |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 4B | on task | 42 | 0 | 0 | 14 | 100.0% | 100.0% |
| 4B | off task | 11 | 3 | 0 | 42 | 78.6% | 100.0% |
| 4B | uncertain | 0 | 0 | 3 | 53 | — | 0.0% |
| 9B | on task | 41 | 0 | 1 | 14 | 100.0% | 97.6% |
| 9B | off task | 9 | 3 | 2 | 42 | 75.0% | 81.8% |
| 9B | uncertain | 1 | 2 | 2 | 51 | 33.3% | 33.3% |

Each pause split contains two supported and two unsupported updates. The unavailable column counts cases where the app retained its prior estimate because the model could not produce a verified response.

| Model | Split | Supported TP/FP/FN/TN before | Supported TP/FP/FN/TN after | Unavailable before → after |
| --- | --- | --- | --- | --- |
| 4B | Development | 1/0/1/2 | 1/0/1/2 | 0 → 1 |
| 4B | Validation | 1/0/1/2 | 0/0/2/2 | 1 → 2 |
| 9B | Development | 1/0/1/2 | 1/0/1/2 | 0 → 0 |
| 9B | Validation | 0/0/2/2 | 0/0/2/2 | 0 → 0 |

On the two positive validation conversations, the frozen 4B attempted revisions but could not supply a valid quote or a positive remaining duration for outstanding writing. The frozen 9B returned valid decisions to keep the prior estimate. It requested external verification of the user’s explicit completion and dismissed a successful test because the writing was unfinished. Both models kept the original estimate on the copied instruction and the stale question.

## Correction transfer and retry outcomes

Each correction probe presented three opaque dashboards under three conditions. Two matching historical labels cover the related-task scenario. The other seven observations lack evidence of purpose. The targeted replay applies the later generic-page guard to the same saved model replies, with zero additional model calls.

| Model | Correct before | Correct after | Correct in targeted replay | New model calls in targeted replay |
| --- | ---: | ---: | ---: | ---: |
| 4B | 4/9 | 4/9 | 9/9 | 0 |
| 9B | 3/9 | 9/9 | 9/9 | 0 |

The follow-up also replays every eligible SQLite interval through the evidence guard. A native mailbox observation with only the title “Inbox” can identify the application but cannot establish the message or its purpose. The app retains a model label on an explicitly requested inbox check or a specific subject. It also preserves current user overrides and other-device evidence.

| Model | Correct raw model labels | Correct after evidence check | New errors | Raw → final distraction precision | Distraction recall after |
| --- | ---: | ---: | ---: | ---: | ---: |
| 4B | 53/56 | 56/56 | 0 | 78.6% → 100.0% | 100.0% |
| 9B | 51/56 | 53/56 | 0 | 75.0% → 90.0% | 81.8% |

All 112 eligible judgments in this native replay came from the two models. The app marked three 4B Inbox decisions and two 9B Inbox decisions uncertain. Both checkpoints preserved all true-positive distraction labels from their own raw runs. The 9B still classified a relevant terminal job-queue command as distraction and abstained on Steam and Ticketmaster phone foregrounds that the fixture labeled distractions. A phone app name alone provides less evidence of purpose than a specific window or page title.

A changed label is an application decision and is marked “insufficient evidence” in the report. The original model classifications and reasons remain in the evaluation receipts. The targeted results reuse the cases that motivated the guard and do not establish the same gain on unseen activities.

The targeted pause retry used fresh model calls on the partial and complete cases identified from 4B. Each checkpoint was rerun on those two conversations outside the frozen scores.

| Model | Scenario | Frozen assessment | With a single required retry citation | Reported remaining minutes |
| --- | --- | --- | --- | --- |
| 4B | partial | unavailable | unavailable | — |
| 4B | finished | unavailable | adjusted | 0 |
| 9B | partial | adjusted | adjusted | 40 |
| 9B | finished | insufficient evidence | insufficient evidence | — |

## First-hand workspace progress

The final paired trial repeated the two validation misses and three controls using cross-domain worked examples. It explicitly accepts first-hand progress reports and allows a positive revision with work outstanding. The controls used a copied instruction, an old milestone followed by a question, and a hypothetical completion. The remaining-minute columns record each model’s answers for the finished and partial cases. These durations are outputs, not timing ground truth.

| Model | Frozen positive revisions | Trial positive revisions | False updates | Direct model abstentions | Unsupported updates rejected by the app | Remaining minutes (finished, partial) |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| 4B | 0/2 | 2/2 | 0/3 | 2/3 | 1/3 | 0, 40 |
| 9B | 0/2 | 2/2 | 0/3 | 2/3 | 1/3 | 0, 40 |

The final prompt was adopted after both models increased the number of verified positive revisions and preserved the previous estimate on all three unsupported controls. Only a direct insufficient-evidence reply counts as the model abstaining in the table. An unsupported proposed revision that the citation guard caught is reported separately.
A supported revision can produce the same total minutes as before. Here the test checks that the app recognizes and cites fresh progress and keeps a positive estimate for work still open. All original quote, freshness and duration validators remain active. These five scenarios were selected after inspection of failures; the result measures that repair, not an independent held-out gain.

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

The follow-up replays the *saved* classifier replies through an evidence check for generic browser pages and the native mailbox window titled “Inbox.” A specific window title, phone context, explicit override, an identified host in the task or rationale, or a matching past correction on comparable work can retain a definite label. Otherwise the app records uncertainty. This is a targeted diagnostic on the observed failure. The frozen live results in `after/` do not incorporate this rule. Other native applications and generic mobile screens can still produce unsupported labels. Mentioning a named site in a rationale does not establish everything the model says about it.

A separate retry experiment addresses a model returning a supported pause decision with an empty citation array. It keeps the initial call unchanged. The second call requires one citation object, and the validator still checks the precise user quote, its role, whether it is fresh and whether the minutes contradict it. It was rerun on the two identified misses; the report keeps those diagnostic results separate from the original held-out counts.

## Reproduction

For the before run, check out `c213b3f` in a separate worktree and copy the audit runner from the [frozen benchmark revision](https://github.com/MadBonzz/daily-planner-recreated/tree/9978d5a62d30cbe6f58b3eb3699e6acb91ac7723) into its `examples/estimation/` directory. For the paired `after/` results, run the audit directly from that frozen revision. The two Bonsai checkpoints saw the same component inputs and response contracts. Set `PLANNER_MODEL` and `PLANNER_MODEL_ENDPOINT` for the desired local Bonsai checkpoint. Use a different output file for each checkpoint, split and revision. From the repository, run the components individually.

```sh
python -m examples.estimation.audit_activity_progress activity-validation.json --split validation --component activity
python -m examples.estimation.audit_activity_progress progress-validation.json --split validation --component progress
python -m examples.estimation.audit_activity_progress accounting.json --split validation --component accounting
python -m examples.estimation.audit_activity_progress history-replay.json --split validation --component replay
python -m examples.estimation.audit_activity_progress feedback-validation.json --split validation --component feedback
python -m pytest -q tests/test_activity.py tests/test_activity_db.py tests/test_activity_fixture.py tests/test_report_worker.py tests/test_workspace_forecast.py tests/test_workspace_progress_scenarios.py
```

The checkout below contains the original follow-on runners and their exact application code. From the current repository root, create it to replay saved judgments or rerun the pause cases independently. The evidence checks read existing model replies. The pause trials call the configured checkpoint. Supply the paths to your saved history-replay and feedback-validation files. The commands load the frozen instructions for the retry-only experiment.

```sh
git worktree add --detach ../plan-pause-audit 44fd7074382b0af20fe288c5ced065a671d4a2e6
cd ../plan-pause-audit
python -m examples.estimation.rescore_activity_labels history-replay.json history-evidence-check.json
python -m examples.estimation.rescore_activity_labels feedback-validation.json feedback-evidence-check.json
git show 9978d5a:workspace_forecast.py > frozen-workspace-forecast.py
python -m examples.estimation.retest_pause_citations pause-retry.json --frozen-module frozen-workspace-forecast.py
python -m examples.estimation.audit_pause_self_report pause-self-report.json
```
