# Does changing the PDF change which history the app selects?

The uploader receives real PDF bytes for each probe. The test recovers text through the planner's worker, encodes the complete input using Qwen3-Embedding-0.6B, searches the local Qdrant index and asks Kev to judge each pair separately. We also obtain Kev's verdict on pairs below the Qwen cutoff, to distinguish retrieval misses from judgment errors. The unchanged title, category, filename and repeated variants serve as controls. The two historical tasks in each profile share an identical multi-section packet but require different work.

| Input change | Expected behavior |
| --- | --- |
| Switch only the attached PDF on a generically worded task | The appropriate historical task switches with the assignment in the document. |
| Keep the same handout, change the named work in the title and description | Use the task concerning the requested work. |
| Remove the handout while retaining a self-contained description | The description can still identify a match. |
| Put the assignment only in the details, or refer to a specific page only | Identify the assigned portion of the packet. |
| Change only the filename or the category | Preserve decisions on the same requested work. |
| Replace a previously indexed PDF under the same ID | Replace the vector, and omit stale matches if embedding the edited content fails. |
| Correct only recorded time, then purge the task | Update the stored time without re-embedding. Purging removes the point. |

The development packet pairs programming a segmented GPU scan with interviewing a food-pantry volunteer. The validation packet pairs measuring an amplifier response with translating an oral history. Both use two potential history sources. Each split contains 28 labeled pairs across seven changes. Filename and category controls repeat the same semantic pair, so a repeated error in these controls counts as one underlying failure when diagnosing it.

The [index mutation trace](index-mutations.json) records the before/after vector hashes, observed metadata and final point count. The original ActivityWatch databases and user documents are outside this test. The corpus of realistic full-length assignment PDFs is also exercised in the [historical retrieval audit](../retrieval/).

## Results on the two controlled packets

| Stage across 56 pairs | TP | FP | FN | TN | Precision | Recall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen candidate search, original or revised | 28 | 22 | 0 | 6 | 56.0% | 100% |
| Qwen followed by original Kev inputs | 25 | 0 | 3 | 28 | 100% | 89.3% |
| Qwen followed by scoped Kev inputs | 28 | 0 | 0 | 28 | 100% | 100% |

Swapping only the attached PDF switched the final match in each packet. Holding the handout fixed and changing the assigned work also switched the match. The original judge dropped the CUDA coding match when the same packet described the volunteer interview. Renaming the PDF and changing only the category reproduced the same underlying miss. Restricting the excerpt to the explicitly assigned exercise fixed all three rows. Both pipelines got all 28 validation pairs right on the amplifier and translation packet. The two-query Qwen change made no difference to this small panel. Similarity admitted 22 irrelevant candidates, which Kev rejected. [Original answers](before.json) · [Revised answers](after.json)

## Results on the published multi-part assignments

Three sample tasks explicitly say to do only a named part of a shared handout. For each originally recorded Kev pair involving any of those tasks, we kept the question, task prose, candidate and model constant and replaced each affected document excerpt with the named section. Of 28 changed requests, ten have previously authored timing-relevance labels. The other 18 include active peers or self-comparisons and did not enter the precision or recall calculation.

| Kev input on ten labeled pairs | TP | FP | FN | TN | Precision | Recall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Original excerpts | 3 | 0 | 4 | 3 | 100% | 42.9% |
| Explicitly assigned section | 5 | 0 | 2 | 3 | 100% | 71.4% |

The Part II transcript write-up recovered the earlier suite report. The Part I mismatch-tool implementation recovered the boundary-debugging task. The latter needed both changes to reach the forecast. Its instructed similarity was 0.493, its bare similarity was 0.517, and Kev initially rejected it with the full handout. The new section raised the Kev score to 0.559. Two expected comparisons still failed, the codec caption for the write-up and the header-reader port for the implementation. All 18 ungraded decisions kept their prior verdict. The report example was inspected during development, so these are regression and diagnostic counts. [Exact request hashes, scores and previous decisions](named-sections.json)

The app uses the narrower excerpts for Kev and the final forecast when it can identify the assigned part unambiguously. Qwen still receives the whole document. Missing headings, ambiguous numbering across attachments and requests to cover more than one section preserve the broader context. Separate regression checks cover negated scope such as “not only page 2,” a reference to office hours on a different page, and late assigned pages in a long rubric. [Scope tests](../../../../tests/test_document_scope.py)

## Reproduce

Run from the repository root with the app's local endpoints set. On a machine that can hold one model at a time, first save vectors with Qwen, then start Kev. The cache is keyed by the SHA-256 of the exact instructed or bare input.

```sh
python -m examples.estimation.audit_document_retrieval output.json --prepare-vectors --vectors /tmp/planner-qwen-vectors.json
python -m examples.estimation.audit_document_retrieval output.json --vectors /tmp/planner-qwen-vectors.json
python -m examples.estimation.audit_document_retrieval index-mutations.json --index-mutations --vectors /tmp/planner-qwen-vectors.json
python -m examples.estimation.audit_section_scope named-sections.json
```
