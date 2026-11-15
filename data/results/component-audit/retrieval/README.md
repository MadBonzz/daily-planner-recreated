# Historical timing comparisons

We paired every flexible held-out task with every eligible completed task in its own history. The 19 targets produce 143 decisions, including 48 authored timing comparisons. The odd-numbered histories are the development split (76 pairs), and the even-numbered histories are the existing validation split (67). These are previously used synthetic histories. The models receive the task scope and PDF text. Target outcomes and evaluation labels are withheld. Bonsai 4B and 9B receive the same upstream retrieval decisions, so these component counts apply to both. The reference labels are the sample histories’ existing `evaluation.evidence_ids`. [Histories and scoring rubrics](../../../tasks/histories/) · [Baseline implementation](https://github.com/MadBonzz/daily-planner-recreated/tree/c213b3f)

## Candidate retrieval

Qwen3-Embedding-0.6B Q8_0 previously searched with only the instructed vector for the full task. We tested the complementary plain-text vector for the same full input, either vector passing, and a reverse search using the candidate as the instructed query. The production change admits either the instructed or the plain-text query at the existing 0.50 cosine floor, stores one vector per historical task, and sends each qualifying candidate to Kev once. The additional query embedding and vector search are the cost for a new custom task. Public samples already contain both vectors. Each query includes all the task's fields and extracted document content.

| Split / stage | TP | FP | FN | TN | Precision | Recall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Development, previous Qwen | 21 | 16 | 3 | 36 | 56.8% | 87.5% |
| Development, two Qwen queries | 23 | 18 | 1 | 34 | 56.1% | 95.8% |
| Validation, previous Qwen | 18 | 8 | 6 | 35 | 69.2% | 75.0% |
| Validation, two Qwen queries | 20 | 8 | 4 | 35 | 71.4% | 83.3% |
| All, previous Qwen | 39 | 24 | 9 | 71 | 61.9% | 81.3% |
| All, two Qwen queries | 43 | 26 | 5 | 69 | 62.3% | 89.6% |

Among the 18 targets with any authored comparison, the top result was relevant on 14 using the instructed query and 15 with both queries. Both retrieved at least one relevant example on 17 targets.

The reverse search found 42 matches and 25 distractors across all pairs. It would also require storing an instructed vector for every historical task. The two queries recovered both implementation comparisons for the C/Rust trace-tool task. Four held-out validation matches still fell below 0.50, including two pencil-and-calculator worksheet comparisons and a GPU race reproducer.

The cutoff itself had an independent boundary bug. Local Qdrant applies a strict greater-than comparison. An exact 0.50 scored pair disappeared at the default cutoff, and an identical vector scoring 1.0 disappeared at a configured 1.0. Both are now admitted and covered by local-index regression tests.

An endpoint returning a nonempty numeric vector of the wrong size posed a separate failure. For the configured Qwen model, a two-component first response could establish a two-dimensional index, breaking subsequent valid 1,024-component lookups. A regression seeds that legacy state and then supplies one malformed response and one valid candidate. The app now rebuilds the incompatible local index from eligible history, reports the skipped candidate, and returns the valid match. Custom model identifiers retain their own dimensions.

For all 19 targets, the checked-in audit verifies that the production Qdrant search returns the IDs, deduplicated counts and scores expected from the packaged numeric vectors. Four fresh embeddings from the live Qwen server, covering both input roles for a PDF-based coding task and a saved-logs reporting task, matched their stored vectors at cosine 1.0. [Per-pair scores](pairs.json) · [Fresh Qwen verification](live-qwen.json)

## Kev on completed-task comparisons

Kev-4B Q4_K_M was tested on the exact candidates returned by Qwen. Standalone judge rows exclude pairs Qwen filtered out. End-to-end rows count those pairs, including any missed positives. The scope update gives Kev the assigned section when the author explicitly requests just one named section of a larger handout.

| Split / stage | TP | FP | FN | TN | Precision | Recall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Earlier excerpts, Kev on 69 two-query candidates | 29 | 1 | 14 | 25 | 96.7% | 67.4% |
| Earlier excerpts, Kev on 28 validation candidates | 16 | 0 | 4 | 8 | 100% | 80.0% |
| Prior Qwen + Kev, all 143 pairs | 28 | 1 | 20 | 94 | 96.6% | 58.3% |
| Two-query Qwen + Kev with earlier excerpts, all 143 pairs | 29 | 1 | 19 | 94 | 96.7% | 60.4% |
| Two-query Qwen + Kev with earlier excerpts, 67 validation pairs | 16 | 0 | 8 | 43 | 100% | 66.7% |
| Final scoped Kev on 69 two-query candidates | 31 | 1 | 12 | 25 | 96.9% | 72.1% |
| Final scoped Kev on 28 validation candidates | 18 | 0 | 2 | 8 | 100% | 90.0% |
| Final Qwen + scoped Kev, all 143 pairs | 31 | 1 | 17 | 94 | 96.9% | 64.6% |
| Final Qwen + scoped Kev, 67 validation pairs | 18 | 0 | 6 | 43 | 100% | 75.0% |

Under the earlier excerpts, Kev rejected both newly admitted implementation comparisons and both reporting comparisons for the archived-transcript write-up. Named-section scoping recovered the first-bad-offset implementation for `patch-tool` and the cursor-table report for `patch-note`. The wrapped-header port and screenshot caption still did not pass Kev. The new end-to-end hit on development was the saved monotone-stack solution for a saved interval-merge redo, recovered by using both Qwen queries. The two software exercises were inspected while diagnosing the PDF issue. The improved result on the existing validation split is a regression check for this last intervention. [Named-section trial and its limits](../document-inputs/README.md)

The Part II archived-transcript target also illustrates a scope ambiguity in the reference Kev input. The 848-character PDF excerpt contains instructions for implementing the Part I tool despite the target explicitly assigning Part II. The earlier report’s excerpt likewise includes implementation passages. These excerpts identify a source of conflicting information. The two refusals by themselves cannot establish that the PDF caused those decisions.

One reported false positive exposes a mismatch in the labels. Kev selected the historical sheet-washing task for the spare duvet cover. Both explicitly count one laundry load and making a bed while excluding machine time. The duvet control has no expected evidence because its unexpected repair could not be predicted. This earlier task is a reasonable baseline for the ordinary work. The original labels are retained in every table. Accepting just the sheet comparison as valid would make final end-to-end counts 32 TP, 0 FP, 17 FN, 94 TN and would leave validation unchanged. Synthetic timing errors played no part in these decisions.

Exact previously recorded Kev calls also let us test pair order without another inference. Of 57 distinct public pairs with both orientations available, six changed when the tasks traded places. These include a fresh algorithm solve vs. retyping a saved solution, and a Fourier worksheet vs. a streamed Fourier lecture. These order checks cover a selected subset. They measure consistency across orientations. The 57 pairs lack independently authored symmetric labels. [Both decisions for every paired case](order-sensitivity.json)

The pipeline also used to drop both documents from examples of the user's explicit comparison corrections. This made a positive and negative correction with identical generic prose indistinguishable to the next task. Both sides now carry bounded, task-relevant passages. A regression checks that the checked and unchecked comparisons retain distinct document instructions in future Kev and forecast requests.

## Alternative Kev question

A second question emphasized transferable work steps, explicitly pointed out that different topics can involve the same effort, and gave a counterexample of implementing a fixture vs. writing up saved results. We used the same checkpoint, ordered pair, bounded PDF excerpts and few-shot examples. We chose the deployment policy on the 41 development candidates before validation responses arrived. A replacement had to recover more authored matches without increasing the number of false positives. The four cutoffs below apply to Kev's output score, independently of Qwen's 0.50 cosine cutoff.

| Judge on development candidates | TP | FP | FN | TN | Precision | Recall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Production question, 0.50 | 13 | 1 | 10 | 17 | 92.9% | 56.5% |
| Alternative, 0.25 | 19 | 8 | 4 | 10 | 70.4% | 82.6% |
| Alternative, 0.35 | 15 | 5 | 8 | 13 | 75.0% | 65.2% |
| Alternative, 0.50 | 12 | 1 | 11 | 17 | 92.3% | 52.2% |
| Alternative, 0.65 | 9 | 0 | 14 | 18 | 100.0% | 39.1% |

We retained the production question. At 0.50 the alternative recovered the paper-ablation comparison but lost a previously accepted invariant proof and an offline replay comparison. It correctly rejected the duvet/sheets pair by the strict authored rubric. It also introduced a clearer false positive by selecting a laptop replay at 0.599 for a live rover crossing, while rejecting all three prior physical trials at 0.207–0.326. [All development probe scores](development.json)

On the 28 validation candidates with frozen excerpts, the retained question had **16 TP, 0 FP, 4 FN, 8 TN** (100% precision, 80% recall). The alternative at 0.50 had **15 TP, 1 FP, 5 FN, 7 TN** (93.8% precision, 75% recall). It newly selected a streamed pulse-train lesson for a worksheet that explicitly requires pencil and calculator, and dropped the prior stride-kernel repair for the GPU debugging task. At 0.35 the alternative recovered one more true comparison than the retained question while adding two false positives. These numbers isolate the wording experiment from the section-extraction fix. [All validation probe scores](validation.json)

## Before-and-after regression checks

The same six newly added cases were run against commit `c213b3f` and the revised implementation. All six failed on the original checkout and passed after the fixes. They cover a candidate appearing through either or both queries, one query failing while the other succeeds, an exact score of 0.50, an identical vector at a configured floor of 1.0, recovery from a two-dimensional Qwen index alongside a malformed candidate, and a future task receiving two reviewed examples with identical prose but different PDF instructions. The broader retrieval, estimator, correction and dated public-replay suites passed all 75 cases. [Index and query tests](../../../../tests/test_clinic_relevance.py) · [User correction tests](../../../../tests/test_relevance_feedback.py)

## Reproduce

Run the five commands from the repository root.

```sh
python -m examples.estimation.audit_retrieval data/results/local/retrieval-audit --score
python -m examples.estimation.audit_retrieval data/results/local/retrieval-audit --verify-live-qwen
python -m examples.estimation.audit_retrieval data/results/local/retrieval-audit --split development
python -m examples.estimation.audit_retrieval data/results/local/retrieval-audit --split validation
python -m examples.estimation.audit_retrieval data/results/local/retrieval-audit --score
```

The reference fixture freezes the two questions, 119 bounded public task records, and 370 previously packaged Kev receipts. Reconstructed states and questions must reproduce their original SHA-256 before a baseline result is counted or a changed question is sent. This isolates the prompt experiment from later section-scoping changes. [Frozen request inputs and fingerprints](judge-inputs.json)

The first command replays the packaged, original numeric Qwen vectors and exact request-keyed Kev outputs. The next three exercise the live local models. The second score command includes the new Kev responses. The unchanged score threshold is 0.50. [Machine-readable confusion matrices](summary.json)
