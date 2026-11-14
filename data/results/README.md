# Task estimation with Ternary-Bonsai 4B and 9B

On 20 held-out synthetic tasks, the original **4B planner missed by 19.05 minutes on average**, the Bonsai 2 **9B planner by 23.55 minutes**, and the user's own estimate by 34.45 minutes. Both models returned valid responses for all 20 final predictions after retries. On individual tasks, 4B was closer nine times, 9B was closer four times and the point estimates tied seven times.

| Forecast | Mean absolute error (min) | Median absolute error (min) | Mean forecast − actual (min) | Actual in range | Mean range width (min) |
| --- | ---: | ---: | ---: | ---: | ---: |
| User estimate | 34.45 | 36.0 | −26.25 | — | — |
| Category-only baseline | 28.70 | 12.0 | −24.80 | — | — |
| Planner with model offline | 31.35 | 32.0 | −24.95 | 16/20 | 69.35 |
| Ternary-Bonsai 4B | **19.05** | **11.0** | −14.55 | **17/20** | 63.85 |
| Ternary-Bonsai-2 9B | 23.55 | 14.5 | −18.95 | 15/20 | 59.70 |

Both models still tended to leave too little time. The 4B planner improved the user's point estimate on 13 tasks, worsened six and left one unchanged. The 9B planner improved 14, worsened five and left one unchanged. Its ranges were 4.15 minutes narrower on average and missed two more outcomes. Most confidence labels were low (18/20 for 4B and 19/20 for 9B). Neither model produced a high-confidence forecast.

[Paired case table](cases.csv) · [4B responses and retries](ternary-4b.json) · [9B responses and retries](ternary-9b.json) · [Offline baseline](offline.json) · [Synthetic histories and PDFs](../tasks/README.md)

## Mean absolute error by history

Each cell is the mean error in minutes on that history's two held-out tasks.

| History | User | Category only | Offline planner | 4B | 9B |
| --- | ---: | ---: | ---: | ---: | ---: |
| Replica protocols seminar | 25.0 | 22.5 | 23.5 | 11.5 | 17.0 |
| Migration tools lab | 32.0 | 32.5 | 32.0 | 12.5 | 10.0 |
| A journal club reading log | 25.0 | 23.5 | 24.0 | 9.0 | 10.0 |
| GPU kernel workbench | 34.0 | 38.0 | 36.0 | 4.5 | 6.5 |
| Studio critique and revision | 30.0 | 36.5 | 33.0 | 16.0 | 26.0 |
| Branch circulation analyst | 39.5 | 6.5 | 23.0 | 7.5 | 25.0 |
| Museum communications editor | 21.0 | 9.0 | 17.5 | 10.0 | 25.5 |
| Community garden equipment steward | 37.5 | 5.5 | 22.5 | 11.5 | 20.5 |
| Evening math tutor | 38.0 | 39.0 | 33.5 | 39.5 | 27.0 |
| Volunteer organizer's personal backlog | 62.5 | 74.0 | 68.5 | 68.5 | 68.0 |

## Where the selections mattered

- **An intermittent GPU bug.** Both models selected the past kernel race and warp-tail investigation. Those overruns moved the user's 60-minute budget to 124 minutes, three minutes short of the 127-minute outcome.
- **A migration write-up.** The 9B model picked two past tasks that involved writing up completed work. Those had taken 69% and 73% of their budgets. The planner lowered 45 minutes to 36, within two minutes of the 34-minute outcome. The 4B response selected no examples and its category fallback gave 44.
- **Reconciling library records.** The 9B model paired a difficult locker-mapping job with a quicker turnaway table, moving the user's 75 minutes to 94. The 4B model selected five records. Four had taken at least 1.7 times their estimates, moving its prediction to 129. The merge actually took 137 minutes.
- **Teaching an unfamiliar misconception.** The probability lesson took 135 minutes. The 4B model picked all six candidates, and the quicker adaptations and grading tasks outnumbered the two first-time lesson plans. The forecast dropped from 75 minutes to 62. The 9B model chose one first-time plan and one reuse case, giving 87 minutes. Both intervals topped out at 123.

## CPU inference and retries

Both runs used LM Studio’s stock Linux AVX2 llama.cpp engine 2.66.0 on an AMD EPYC 9V74 with GPU offload disabled. These times are per accepted forecast on this host.

| Model | Median latency | Mean latency | First-attempt successes | Additional primary attempts | Primary wall time including recovery |
| --- | ---: | ---: | ---: | ---: | ---: |
| 4B | 62.80 s | 71.54 s | 15/20 | 6 | 29 min 37 s |
| 9B | 45.29 s | 102.23 s | 3/20 | 17 | 29 min 03 s |

The 4B server used three CPU threads and one 8,192-token slot. Three requests hit a 90-second HTTP timeout. Three more produced immediate local engine or HTTP errors. The run retained successful responses and resumed with a 360-second timeout. The 15 initial successes span both timeout settings. Final scoring includes 15 model selections, four valid empty lists that led to category adjustments, and one valid empty list that retained the user's budget.

A separate [9B default-reasoning probe](ternary-9b-default-probe.json) timed out at 360 seconds before returning a forecast. The primary 9B run used short-answer mode (`--reasoning off`) with a 600-second timeout. Three accepted cases ran on a four-thread server configured with four 4,096-token slots. Following two simultaneous timeouts and a server exit, the remaining cases finished on a single 8,192-token slot. The first two used four threads and the final 15 used eight. Those final 15 averaged 43.94 seconds per forecast. The run records all 17 initial failed requests, including requests that reached an unavailable server. Nineteen final forecasts used model-selected examples. One valid empty list retained the original estimate.

These latencies include retrieval and validation. They exclude PDF extraction and loading the model. The mixed schedules, prompt cache and shared CPU load prevent a controlled speed comparison. The separate reasoning probe and cold-start smoke are outside the primary timing and accuracy tables.

## How the comparison works

Each of the ten fictional people has 6–9 completed tasks. Both models estimated the same two later tasks from each person's fixed history. Neither held-out task enters the other's history. The 77 past tasks and 20 outcomes are synthetic. Eleven held-out tasks have an attached PDF. Thirteen user estimates are below the eventual outcome and seven are above it.

The planner retrieves up to six candidates from the same history, using a local Qdrant index and a 512-dimensional hashed-text vector. The model sees the goal, category, notes and extracted PDF excerpts. It selects candidates and explains each selection. Historical timings, held-out outcomes and the fixture author's evaluation notes are excluded from the model prompt.

Python takes the median actual-to-user-estimate ratio among the selected tasks and multiplies the new task's user estimate by an adjusted ratio. One model-selected task applies one-third of the adjustment, two apply two-thirds and three or more apply it in full. A valid empty selection invokes the planner's same-category rule, or retains the user's estimate if the category has no history. A failed model call can use text similarity first. These paths are recorded separately in the raw results.

The category-only baseline applies the full median correction from all past tasks in the same nonempty category, or retains the user estimate when there are none. The offline baseline runs the production fallback with model access disabled. Range coverage counts inclusive endpoints. Mean width averages upper minus lower bound, in minutes. Ranges and confidence labels are planner heuristics with no target coverage probability.

All 11 PDF-bearing holdouts are tasks the fictional user underestimated, with a mean initial error of 54.36 minutes. For the nine holdouts without attachments the mean initial error is 10.11 minutes. Document availability and task difficulty therefore vary together in this set. A test of the effect of PDFs would need the same task scored with and without its document. Two outcomes per history also give too little information to rank occupations or calibrate confidence for a particular person.

The 4B file is the original Ternary-Bonsai release, and the 9B file is from the later Ternary-Bonsai-2 family. The benchmark changes the model generation, the model size and the serving configuration together. Accuracy differences describe the two configured planners.

## Separate first-task check

A [first-task smoke test](ternary-4b-cold-start.json) exercises the asynchronous no-history flow added by [commit 000c98f](https://github.com/MadBonzz/daily-planner-recreated/commit/000c98f). The planner immediately stored the synthetic migration task as pending, then passed its description, notes and PDF excerpt to 4B with no completed tasks. The user budgeted 85 minutes. The model returned 90 minutes with a 60–120-minute range after 56.6 seconds. The fixture's actual was 138 minutes. The regular 4B run used this same held-out task with its earlier history. Both benchmark averages count only the regular history-backed case.

## Reproduce the comparison

Use the 20 holdouts from [fixture commit 5a60ccf](https://github.com/MadBonzz/daily-planner-recreated/commit/5a60ccf) and a Python environment with the project requirements installed. The two exact model files are linked below. Their SHA-256 digests are also recorded in the run JSON.

| Download | SHA-256 |
| --- | --- |
| [Original 4B, Q2_0_g64](https://huggingface.co/prism-ml/Ternary-Bonsai-4B-gguf/blob/main/Ternary-Bonsai-4B-Q2_0_g64.gguf) | `9d968b04a3c9a794897bcc744c8072fb6a061c0e42efd03c989401ddf8baef0c` |
| [Bonsai 2 9B, Q2_0](https://huggingface.co/prism-ml/Ternary-Bonsai-2-9B-gguf/blob/82265ccc78ddaac4700286a35e9badcc32376dd9/Ternary-Bonsai-2-9B-Q2_0.gguf) | `1f884aa56aacd2bbbb2e4b7abaead2b1102b74a9c75a23fc33dab916813a384d` |

From the repository root, run the 4B evaluation with LM Studio's headless daemon and bundled CPU runtime. Replace the file path and the model key with the ones on your machine. The example writes to a new directory so the checked-in results stay available for comparison.

```sh
lms daemon up
lms import /path/to/Ternary-Bonsai-4B-Q2_0_g64.gguf
lms ls
lms load MODEL_KEY_FROM_LMS_LS --gpu off --context-length 8192 --identifier ternary-bonsai-4b
lms server start --bind 127.0.0.1 --port 1234
PLANNER_MODEL=ternary-bonsai-4b \
PLANNER_MODEL_ENDPOINT=http://127.0.0.1:1234/v1/chat/completions \
PLANNER_MODEL_TIMEOUT_SECONDS=360 \
python -m examples.estimation.run evaluate --strict-model --json rerun/4b.json
```

For 9B, start the LM Studio 2.66.0 AVX2 extension's `llama-server` executable (or a compatible build) in a separate terminal. This command uses port 1235 so it can be run alongside the 4B endpoint.

```sh
llama-server --host 127.0.0.1 --port 1235 \
  --alias ternary-bonsai-2-9b -m /path/to/Ternary-Bonsai-2-9B-Q2_0.gguf \
  -ngl 0 -c 8192 -np 1 -t 8 -tb 8 --no-warmup \
  --reasoning off
```

Once that server is ready, run the same estimator with the 9B endpoint. The repository CLI scores cases sequentially. It uses the same task inputs, selection prompt, model temperature of zero and output limit of 480 tokens as the published runs. Scheduling, prompt caching, and backend versions can change the observed timings and generated selections.

```sh
PLANNER_MODEL=ternary-bonsai-2-9b \
PLANNER_MODEL_ENDPOINT=http://127.0.0.1:1235/v1/chat/completions \
PLANNER_MODEL_TIMEOUT_SECONDS=600 \
python -m examples.estimation.run evaluate --strict-model --json rerun/9b.json

python -m examples.estimation.run evaluate --offline --json rerun/offline.json
```
