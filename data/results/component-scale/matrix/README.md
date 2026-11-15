# Active-task relevance and selective re-estimation

This evaluation follows 16 five-task boards from submission through three waves of completions. It measures three distinct decisions: whether a pair is semantically relevant, which surviving tasks the planner actually refreshes, and how the final forecaster uses the new duration. [Measurements and failure cases](findings.md)

The assigned completion times drive the scenario. The remaining tasks have no measured finish time.

## Reference and design

Two assistant reviewers independently labeled the same 160 possible active-task pairs from their text before measuring Kev. The second received only titles, descriptions, user budgets and board names. The two sets agreed on all 41 relevant and 119 unrelated pairs. The criterion is whether the jobs share work of comparable scope or a concrete recurring obstacle that informs effort. A task that includes both hemming and zipper replacement can be relevant to both individual repairs. Those repairs have no shared method with each other. The fixtures include two boards with no related tasks, a five-task clique, single pairs, two disjoint pairs, three-task clusters, partially overlapping workflows and same-setting negatives. Both labels are judgments about this synthetic workload. [Reference boards and completion schedules](reference.json) · [Independent labels](second-labels.json)

Each board is created in a fresh SQLite store. At each addition the runner reads the physical matrix before and after the worker executes. It checks that each new cell starts pending, that Kev resolves it, and that adding a card leaves all prior cell values intact. Within a completion wave, all indicated tasks finish before the worker drains. The runner records the queued targets and the persisted forecast timeline for every survivor. The semantic denominator is all surviving tasks at each wave. It also measures the same transitions conditional on the cells Kev actually supplied, so that an erroneous relevance judgment can be distinguished from a scheduling error.

For the Bonsai runs, every board has a shared completed permit renewal to exercise production historical retrieval. Both Qwen query roles search its document vector. Every above-threshold hit receives the usual directional Kev check. The other comparisons come from the recorded Kev matrix as those tasks finish. Each board has one new task receiving a cold Bonsai forecast. Among the tasks still open after its last wave, we chose the one with the most reference peers, breaking ties alphabetically, to exercise a real cold-to-refresh sequence when possible. Its four other initial forecasts are seeded at the user's budget. Every eligible later refresh uses Bonsai. The runner records the final forecast source, the tasks sent and cited, raw request timings in minutes, and whether every independent card kept the same forecast and timeline length. A task is counted as cited if the model names it as a timing comparison or as the source of a warning. Point movement is scored separately. The Qwen vectors and Kev scores are shared across the two forecast models by exact input hash. [Qwen responses](qwen.json) · [Kev responses](kev.json)

The movement check applies only to the first new comparable evidence on a surviving task. Its source must finish at least 15% over or under budget, all newly comparable sources must agree in direction, and a combined deliverable paired with a single operation is excluded. The run measures movement relative to the prior point and separately tests which side of the user's budget the new point lands on. This measures a behavioral response to evidence. It does not supply ground truth for the new duration.

## Queue stress

The separate stress replay uses controlled judgments and forecast outputs to expose ordering problems. Ninety-six deterministic schedules draw ten cards from two boards, add four cards before the first worker tick, start and pause timers during inference, finish tasks in bursts, and edit a completed duration after the queue settles. An independent ledger records each task's creation-time history and every pair that was simultaneously open. The test reads actual SQLite rows and forecast versions and compares them with the ledger. [All replay seeds](stress.json)

## Reproduce

Run from the repository root with the normal application endpoint variables. Select the matching local forecaster before each forecast command. Each worker uses its own SQLite store and per-board request checkpoints.

```sh
python -m examples.estimation.scale_matrix vectors
python -m examples.estimation.scale_matrix kev
python -m examples.estimation.scale_matrix queue
python -m examples.estimation.scale_matrix stress
python -m examples.estimation.scale_matrix forecast --model-tag 4b --workers 2
python -m examples.estimation.scale_matrix forecast --model-tag 9b --workers 2
python -m examples.estimation.scale_matrix report
```

If Kev runs concurrently with Qwen, run the `prior` stage after Qwen completes and before unloading Kev. To rerun a board affected by a forecast transport error, add `--retry-transport` to its forecast command. The prior board outcome is retained and successful requests with identical inputs are reused.
