# Local task relevance evaluation

The selected cosine cutoff is **0.45**. Qwen3-Embedding-0.6B retained 23 of 24 labeled matches on the calibration profiles. On five separate validation profiles it retained 21 of 24 matches and passed 13 additional candidates to Kev. At 0.80 or 0.85, just two of the 24 validation matches reached the decision model.

| Cosine cutoff | Calibration recall | Calibration precision | Calibration F2 | Validation recall | Validation precision | Kev calls on validation |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.40 | 95.8% | 46.0% | 0.79 | 95.8% | 59.0% | 39 |
| **0.45** | **95.8%** | **54.8%** | **0.83** | **87.5%** | **61.8%** | **34** |
| 0.50 | 87.5% | 55.3% | 0.78 | 79.2% | 76.0% | 25 |
| 0.80 | 8.3% | 50.0% | 0.10 | 8.3% | 50.0% | 4 |
| 0.85 | 8.3% | 50.0% | 0.10 | 8.3% | 50.0% | 4 |
| 0.90 | 8.3% | 66.7% | 0.10 | 4.2% | 100% | 1 |

## Inputs and scoring

Each of 19 flexible-duration holdouts was paired with every completed historical task for its user. One fixed-duration holdout bypasses estimation in the app and was excluded. The sweep also included seven negative-control pairs with past fixed-duration tasks. The app excludes those candidates automatically, and all seven scored below 0.40, so including them changed only the true-negative count. The benchmark embedded 98 distinct tasks, including the text extracted from their PDFs. Task text was serialized with the production field order. Correct matches came from the authored `evaluation.evidence_ids`. Outcome times, relevance labels and holdout notes never enter either model's input.

The five odd-numbered histories formed the calibration split (78 pairs). The five even-numbered histories formed the validation split (72 pairs). We swept cutoffs from 0.40 through 0.95 in steps of 0.05. The selection metric was calibration F2, which weights recall because the pairwise decision model can still reject an extra candidate. Ties favor higher precision and then the higher cutoff. The small synthetic sample and its evidence labels limit how well the scores generalize to individual users. See the individual scores, task text hashes and full grid in [embeddings.json](embeddings.json).

Three authored matches fell below 0.45 on validation. The repeat GPU timing sweep scored 0.444 against the earlier median run. Two practice worksheets scored 0.427 and 0.368 against the triangle-wave worksheet. Manual additions can recover pairs that never reach Kev.

## Pair decisions on the validation profiles

Kev accepted **16 of 34 candidates**. Of the 24 authored matches across all 72 pairs, the two stages together recovered **16**. End-to-end precision was **100.0%** and recall was **66.7%**. The cosine filter missed three matches before Kev saw them. Kev rejected 5 more authored matches and accepted 0 unlabeled comparisons. See [all pair decisions](decisions.json).

For the validated GPU timing sweep, Kev kept the earlier timing-table task and rejected a high-similarity bug-fix task that shared the handout. For the write-up of saved replay transcripts, it rejected the previous transcript write-up despite the matching deliverable. A user who recognizes an omitted comparison can add it from the task's relevance editor.

## Model files and runtime

| Role | Model and quantization | SHA-256 |
| --- | --- | --- |
| Embed one combined task | [Qwen3-Embedding-0.6B Q8_0](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B-GGUF/tree/370f27d7550e0def9b39c1f16d3fbaa13aa67728) | `06507c7b42688469c4e7298b0a1e16deff06caf291cf0a5b278c308249c3e439` |
| Classify one pair | [Kev-4B Q4_K_M](https://huggingface.co/ggml-org/Kev-4B-GGUF/tree/f4624b6bb7072f1bab726a622c1cad3f96030b9e) | `33ae6b18926502b2209a1bf7d3b61a350d65938441515c686bb87d226a19eff9` |

The embedding run used LM Studio's bundled llama.cpp CPU runtime 2.66.0 (b11966), 8 threads, last-token pooling, 8,192-token context and the local `/v1/embeddings` endpoint. Qwen returned 1,024-dimensional vectors. A 495-token PDF-backed task produced the same vector through the headless daemon and directly through its bundled runtime (cosine similarity 1.0000). The decision stage uses the production pair question and examples with no prior user corrections, sent individually to the local `/v1/systemone` endpoint. The headless daemon processed the first 14 pairs on CPU with a 4,096-token context. Its server stopped during the next request. The checkpoint resumed on the same quantized weights and LM Studio llama.cpp runtime, using the same 4,096-token context, 8 threads, and a 512-token batch. Only successful pair decisions enter the scores.

## Reproduce

Load both models with the identifiers in the [planner setup](../../../../README.md#local-relevance-models). From the repository root, choose a fresh result directory and run both stages.

```sh
python -m examples.estimation.calibrate_retrieval data/results/retrieval/my-run --stage embeddings
python -m examples.estimation.calibrate_retrieval data/results/retrieval/my-run --stage decisions
```

The decision stage checkpoints each evaluated pair in `decisions.json`. Re-running it on the same unchanged inputs resumes at the next unfinished pair.
