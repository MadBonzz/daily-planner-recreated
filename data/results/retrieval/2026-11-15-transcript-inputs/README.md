# How task descriptions and PDFs changed transcript retrieval

The target was **Explain the three saved replay transcripts**. Its 101-character description asks for Part II only, a table from archived transcripts and an explanation of one checksum collision. Its attached lab handout contributes 8,115 characters of extracted text and also describes implementing a bounded mismatch reporter. All four tasks had empty optional details.

| Earlier task | Titles only | Titles and descriptions | Full inputs, including available PDFs |
| --- | ---: | ---: | ---: |
| Write up three saved replay transcripts | 0.933 | 0.658 | 0.565 |
| Annotate the saved decoder comparison | 0.510 | 0.639 | 0.583 |
| Implement the bounded mismatch reporter | 0.277 | 0.425 | 0.582 |

Adding descriptions moved the first pair below the 0.70 retrieval threshold. The earlier description asks for a cursor table and failure explanation. Adding both tasks’ different lab handouts lowered the score further. For the annotation task and implementation task, only the target had a PDF. Adding that handout moved the implementation candidate from 0.425 to 0.582, within 0.001 of the annotation candidate. The dataset labels the two write-ups as timing comparisons. These measurements describe this one query and three candidates.

The experiment used the production Qwen3-Embedding-0.6B Q8_0 model on CPU through LM Studio’s bundled llama.cpp runtime. Each comparison embedded both tasks using the indicated fields, with one vector per task, and measured cosine similarity. The full-input scores reproduce the original retrieval run. [Six-decimal scores and one-sided PDF checks](scores.json) · [Original scores for all candidates](../2026-11-15-qwen06b-kev4b/embeddings.json)
