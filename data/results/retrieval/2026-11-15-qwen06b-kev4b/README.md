# Local task relevance evaluation

This evaluation measured automatic retrieval at a **minimum cosine similarity of 0.70**. At that setting Qwen sent eight of the 72 held-out pairs to Kev. Kev accepted all six labeled matches in those eight and rejected both distractors. Across the 24 labeled matches in the held-out set, end-to-end **recall was 25% and precision was 100%**. The other 18 matches fell below the cosine cutoff. Five of the nine flexible validation tasks received no automatic candidates.

The [operating point and eight individual Kev decisions](operating-point.json) were derived from saved model responses. The earlier experiment used a 0.45 cutoff, so it already included every candidate that clears 0.70. Rescoring used the original decisions for exactly those eight pairs.

| Cosine cutoff | Calibration recall | Calibration precision | Calibration F2 | Validation recall | Validation precision | Kev calls on validation |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.40 | 95.8% | 46.0% | 0.79 | 95.8% | 59.0% | 39 |
| 0.45 (experimental F2 maximum) | 95.8% | 54.8% | 0.83 | 87.5% | 61.8% | 34 |
| 0.50 | 87.5% | 55.3% | 0.78 | 79.2% | 76.0% | 25 |
| **0.70 (evaluated minimum)** | **29.2%** | **70.0%** | **0.33** | **25.0%** | **75.0%** | **8** |
| 0.80 | 8.3% | 50.0% | 0.10 | 8.3% | 50.0% | 4 |
| 0.85 | 8.3% | 50.0% | 0.10 | 8.3% | 50.0% | 4 |
| 0.90 | 8.3% | 66.7% | 0.10 | 4.2% | 100% | 1 |

## Inputs and scoring

Each of 19 flexible-duration holdouts was paired with every completed historical task for its user. One fixed-duration holdout bypasses estimation in the app and was excluded. The sweep also included seven negative-control pairs with past fixed-duration tasks. The app excludes those candidates automatically, and all seven scored below 0.40, so including them changed only the true-negative count. The benchmark embedded 98 distinct tasks, including the text extracted from their PDFs. Task text was serialized with the production field order. Correct matches came from the authored `evaluation.evidence_ids`. Outcome times, relevance labels and holdout notes never enter either model's input.

The five odd-numbered histories formed the calibration split (78 pairs). The five even-numbered histories formed the validation split (72 pairs). We swept cutoffs from 0.40 through 0.95 in steps of 0.05. The selection metric was calibration F2, which weights recall because the pairwise decision model can still reject an extra candidate. Ties favor higher precision and then the higher cutoff. The small synthetic sample and its evidence labels limit how well the scores generalize to individual users. See the individual scores, task text hashes and full grid in [embeddings.json](embeddings.json).

## Pair decisions

At 0.70, Qwen retained seven of 24 calibration matches and three additional candidates. On validation it retained six of 24 matches and two distractors. Two GPU tasks shared a handout, so both bug fixing and publishing a timing table scored highly against either new task. Kev correctly distinguished those deliverables. The three clinic meetings with Dev also passed both stages. The earlier clinic planning meetings with Mara scored between 0.60 and 0.65 against the new Mara rota task, leaving them available for manual selection.

In the original 0.45 experiment, 34 validation candidates reached Kev. It accepted 16 labeled matches and rejected all 13 distractors, yielding 66.7% recall and 100% precision at that experimental cutoff. The [complete decision log](decisions.json) preserves that original run.

## Model files and runtime

| Role | Model and quantization | SHA-256 |
| --- | --- | --- |
| Embed one combined task | [Qwen3-Embedding-0.6B Q8_0](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B-GGUF/tree/370f27d7550e0def9b39c1f16d3fbaa13aa67728) | `06507c7b42688469c4e7298b0a1e16deff06caf291cf0a5b278c308249c3e439` |
| Classify one pair | [Kev-4B Q4_K_M](https://huggingface.co/ggml-org/Kev-4B-GGUF/tree/f4624b6bb7072f1bab726a622c1cad3f96030b9e) | `33ae6b18926502b2209a1bf7d3b61a350d65938441515c686bb87d226a19eff9` |

The embedding run used LM Studio's bundled llama.cpp CPU runtime 2.66.0 (b11966), 8 threads, last-token pooling, 8,192-token context and the local `/v1/embeddings` endpoint. Qwen returned 1,024-dimensional vectors. A 495-token PDF-backed task produced the same vector through the headless daemon and directly through its bundled runtime (cosine similarity 1.0000). The decision stage uses the production pair question and examples with no prior user corrections, sent individually to the local `/v1/systemone` endpoint. The headless daemon processed the first 14 pairs on CPU with a 4,096-token context. Its server stopped during the next request. The checkpoint resumed on the same quantized weights and LM Studio llama.cpp runtime, using the same 4,096-token context, 8 threads, and a 512-token batch. Only successful pair decisions enter the scores.

## Reproduce

Load both models with the identifiers in the [planner setup](../../../../README.md#local-relevance-models). The original cutoff sweep predates the instruction-prefixed Qwen query. Create the historical checkout once from the repository root, then run its two stages in a fresh directory.

```sh
git worktree add --detach ../plan-calibration 791814ca57087b5624894e8b7acad0a212dd46c1
cd ../plan-calibration
python -m examples.estimation.calibrate_retrieval data/results/retrieval/my-run --stage embeddings
python -m examples.estimation.calibrate_retrieval data/results/retrieval/my-run --stage decisions
```

The decision stage checkpoints each evaluated pair in `decisions.json`. Re-running it on the same unchanged inputs resumes at the next unfinished pair.

The later 0.70 report reads the saved embeddings and Kev decisions. From the current repository root, create a checkout that contains its scorer and the original inputs.

```sh
git worktree add --detach ../plan-cutoff-070 e0e0fd3a5392f5a10dbc6d86e19398b367f07702
cd ../plan-cutoff-070
python -m examples.estimation.calibrate_retrieval data/results/retrieval/2026-11-15-qwen06b-kev4b --stage operating-point --threshold 0.70
```
