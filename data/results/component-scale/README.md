# Workspace pauses and selective re-estimation

These scenarios expand the [earlier component audit](../component-audit/). The reference labels were fixed before running the models. They describe synthetic situations and do not establish an individual user's future completion time.

| Question | Cases | What is measured |
| --- | ---: | --- |
| Did the user report enough fresh progress to revise the remaining time? | 48 conversations for each Bonsai model | False revisions, missed revisions, incorrect declarations of completion, evidence citations and consistency with explicit remaining time. [Inputs, labels and results](pause/) |
| Did the matrix link tasks with comparable work? | 16 five-task boards, 160 pair labels | Qwen’s cosine screen and Kev’s pair judgments against an independently reviewed reference. The app asks Kev about every pair of active tasks. [Board inputs](matrix/reference.json) |
| Did similar wording create extra links on mixed boards? | Two boards of eight and seven tasks, 49 further pairs | Kev found all seven labeled links and added seven extra links: 7 TP, 7 FP, 35 TN, 0 FN (50% precision, 100% recall). One example was linking someone writing a reconciliation script to someone reconciling actual dues. [Inputs](matrix/mixed-boards.json) · [Pair decisions and matrix snapshots](matrix/mixed-decisions.json) |
| Did finishing a task refresh exactly the affected forecasts? | 48 completion waves on the 16 boards, per model | Compare the worker’s actual targets with both the stored matrix and the reference relationships. For forecasts, check which completed tasks the model cited and the direction of a change when a direction is justified. [Matrix evaluation](matrix/) |
| Do additions, pauses, completions and time edits interfere with the queue? | 96 seeded ten-task timelines | 960 creations, 439 completions and 96 timing corrections. This state-machine test uses controlled relevance and forecast responses. [Per-timeline results](matrix/stress.json) |

On the 16 five-task boards, Kev accepted 38 of the 41 reference matches and added two unwanted links (95.0% precision, 92.7% recall). Across the 48 completion waves, the queue scheduled exactly the 35 refreshes implied by the stored matrix and left the other 93 forecasts untouched. Against the independent labels, there was one unnecessary refresh and two missed refreshes. [Per-board results and examples](matrix/findings.md)

The [five editable Dev histories](../../tasks/demo/) have a separate [fixture and calendar audit](../demo-histories/). Their ActivityWatch rows and task notes can be inspected in the running app.
