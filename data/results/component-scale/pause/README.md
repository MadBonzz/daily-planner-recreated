# Workspace pause forecasts on eight new task families

This cohort checks whether Plan Smarter should revise an estimate when a user pauses an in-app workspace. It uses 48 new conversations. None are copied from the five cases used to tune the shipped self-report prompt. Each task family contributes six independent snapshots. The conversations vary the task and its language while keeping a balanced mix of supported and unsupported revisions within each family.

| Reference decision | Cases | Evidence |
| --- | ---: | --- |
| Keep the existing forecast | 24 | Copied instructions, stale milestones, assistant-only advice, another task, hypothetical work, unattributed status, or a timer without progress |
| Revise with work remaining | 16 | Partial deliverables, measured results, new blockers, or an explicit estimate of remaining active work |
| Revise to zero remaining | 8 | A first-hand report that the full deliverable, including the last required step, was finished |

The task families are audio restoration, bicycle repair, film captioning, seed inventory, expense reconciliation, field mapping, glaze preparation and newsletter editing. Each scenario starts independently with a pre-existing duration forecast, measured elapsed time and its own chat. Prior milestone cases include a previous pause and a boundary identifying which turns are fresh. A correct revision may leave the total numerically unchanged.

## Reference labels

An author specified the labels, supporting phrases and remaining-time checks before inference. A second AI reviewer saw shuffled neutral identifiers, task descriptions, chat turns and freshness boundaries. The reviewer did not receive the first labels, the evidence phrases, or model outputs. The independent three-way judgments agreed on all 48 cases (Cohen's κ = 1.00). These are synthetic examples with AI-authored reference labels. One task per family and six snapshots per task cannot establish a population-wide rate for that category. The inputs, [reference labels](reference.json) and [blind review](blind-adjudication.json) are published.

The decision metric compares the final application action, *revise* or *keep*, with the reference. The three-way table distinguishes declaring the whole task finished from recognizing partial progress. The runner also scores each first model reply prior to validation if the provider returned a schema-valid response. A transport failure, direct model abstention and a model update rejected for unusable evidence are reported separately. A rejected update preserves the user's time fields and counts as no revision at the application level.

Conditional checks require a nonnegative ordered interval, a fresh verbatim user quote and a total consistent with the minutes remaining plus recorded minutes. A stricter diagnostic checks whether a quote contains a distinct supporting phrase written before inference. That diagnostic can miss valid shorter quotes. Three conversations contain an explicit user estimate of total *remaining* active minutes. We check whether the model's planning range includes that stated duration. We do not compare a predicted minute to an invented actual.

## Run it

From revision `44fd707` with this runner and cohort, use the production forecaster and the specified checkpoint on a local OpenAI-compatible server. The full system prompt has SHA-256 `756d8805e6fdd2763a54f9addfbec5943b4e7d1ef86f70f02a3e5c127d9da979`. The conversation digest is `65affbaf966b52f81cf462c8934798a51fc39fd8347ce2b24ff106243af0b633` and the reference digest is `1cb4428ec37d755d27157f081109d5527e53770559317f283a5f5eddb70b6187`.

```sh
export PLANNER_MODEL_ENDPOINT=http://127.0.0.1:1243/v1/chat/completions
export PLANNER_MODEL=ternary-bonsai-4b
export PLANNER_MODEL_TIMEOUT_SECONDS=2400
python -m examples.estimation.scale_pause run 4b --workers 3
python -m examples.estimation.scale_pause score 4b
```

Use `ternary-bonsai-2-9b`, its serving port and the argument `9b` for the second model. The real method processes each case separately using temperature 0, its normal schema and a 900-token response limit. The first request or its one permitted repair may supply the accepted reply. Each request carries only the task, baseline and workspace chat. Gold labels, expected minutes and case identifiers stay outside the model prompt. The scripts record HTTP status, responding model, request hashes, generated structured answers and the forecast actually returned to the app. The runner can resume a stopped job. Use `--retry-transport` to repeat only transport failures and preserve the earlier receipts.
