# Website blocking and workspace eligibility

The two decisions use Kev-4B Q4_K_M. A positive website label means a whole-domain block is appropriate for the new task given the recorded browser evidence. A positive workspace label means the reader or plain-text desk can do the requested work with the material actually available. An off-task label on an overlapping laptop/phone interval does not establish which device caused it. A generic sign-in or dashboard title alone does not establish a blockable activity. An explicit whole-site work requirement takes precedence over unrelated past visits. For workspace tasks, drafting the requested content is meaningful progress even when it will be submitted elsewhere. An attached instruction sheet cannot supply an unavailable spreadsheet or let the desk annotate an image in place.

All cases, task descriptions, browser titles, expected labels and split assignments are in [cases.json](cases.json). These are authored examples for this audit.

The website evaluation creates completed tasks and ActivityWatch-style classified intervals in a temporary planner database, predicts with the related tasks supplied, calls the production site review, then reads the site's effective policy. Reports are already labeled and relations are given, which isolates the blocking decision from the activity classifier and retriever. Inputs cover single and repeated observations, named resources, a handout naming the site, mixed work and distraction use, and laptop/phone overlap. A case may have zero Kev calls when a filter or the old recurrence rule already decides it. Raw responses and actual calls are recorded individually. Workspace examples go directly through the production Kev request and response validator.

The original implementation is commit `c213b3f`. The harness also supports running just one split or one component. Use a fresh output path for each version, as a partially written run resumes previously recorded cases.

```bash
python -m examples.estimation.audit_focus_workspace --output data/results/local/focus-workspace.json
python -m pytest -q tests/test_focus_domains.py tests/test_focus_bridge.py tests/test_focus_judge.py tests/test_single_focus.py tests/test_task_workspace.py
```

## Kev decisions

| Component | Split | Version | TP | FP | FN | TN | Precision | Recall |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| focus | development | Baseline | 3 | 2 | 0 | 3 | 60.0% | 100.0% |
| focus | development | Website revision 1 | 2 | 0 | 1 | 5 | 100.0% | 66.7% |
| focus | development | Website final | 3 | 0 | 0 | 5 | 100.0% | 100.0% |
| focus | held out | Baseline | 3 | 2 | 0 | 3 | 60.0% | 100.0% |
| focus | held out | Website final | 3 | 0 | 0 | 5 | 100.0% | 100.0% |
| workspace | development | Baseline | 4 | 1 | 0 | 4 | 80.0% | 100.0% |
| workspace | development | Workspace revision 1 | 4 | 1 | 0 | 4 | 80.0% | 100.0% |
| workspace | development | Workspace revision 2 | 4 | 0 | 0 | 5 | 100.0% | 100.0% |
| workspace | development | Workspace final | 4 | 0 | 0 | 5 | 100.0% | 100.0% |
| workspace | held out | Baseline | 4 | 0 | 0 | 6 | 100.0% | 100.0% |
| workspace | held out | Workspace revision 2 | 4 | 0 | 0 | 6 | 100.0% | 100.0% |
| workspace | held out | Workspace final | 4 | 0 | 0 | 6 | 100.0% | 100.0% |

The first site and workspace revisions were tried only on development. The site candidate was chosen on development. The attachment-specific workspace branch was retained after checking a public demo task. A correct outcome on a filter-only case is a policy result. Only rows containing a live request measure Kev directly.

The following counts include only held-out website cases using a Kev decision.

- Baseline used Kev on 4 cases (2 TP, 0 FP, 0 FN, 2 TN).
- Website final used Kev on 7 cases (3 TP, 0 FP, 0 FN, 4 TN). 4 used an identical request from the earlier run.

A reused call records the source file, case and SHA-256 of the complete original request and answer. The audit runner resolves these references for offline replay. Inspect the requests and raw Kev scores in [Baseline](baseline.json), [Website revision 1](site-revision-one.json), [Website final](site-revision-two.json), [Workspace revision 1](workspace-revision-one.json), [Workspace revision 2](workspace-revision-two.json), [Workspace final](workspace-final.json).

## Changes

The original website policy automatically blocked domains after two related tasks. It blocked a site needed for a new task and treated repeated sign-in-only tabs as identifiable distractions. The first revision asked the same content-specificity question on both single and repeated visits. That wrongly rejected a named auction listing on two related tasks (specificity score 0.410). For repeated off-task visits, the final prompt asks whether every title is generic navigation and separately checks whether the site is required now. The repeated auction received a generic-title score of 0.171 and a site-needed score of 0.152 and became a suggestion again. Explicit unblocks now apply to overlapping automatic subdomain suggestions. Saving an unchecked pending site records that choice before a delayed decision arrives. The sample package includes 59 exact, genuine decisions covering chronological replay and manual combinations for the 40 open/addable tasks.

The original workspace gate allowed a PDF saved only on a workstation. On this example, the first revised question about a missing source scored 0.383, below the 0.50 cutoff. The second revision asks whether a particular document must be consulted. The database supplies the fact that there is no attachment. Kev scored the document requirement at 0.596 and the gate declined the workspace. During sample packaging, the revised general external-tool question also made reducing a live failure trace look eligible just because there was a handout. The final version preserves the exact original PDF-desk questions when an attachment exists and uses the tested source requirement on the writing-only desk. Each final workspace row replays a complete byte-equivalent Kev request and identifies its live source file.

All 60 targeted focus and workspace tests passed. All ten offline schedule replays passed. An interrupted workspace-fixture rebuild keeps the previous package available and can resume from a separate checkpoint. The timer and browser-policy regression starts the task while its forecast is pending, adds a user rule, commits a site review, and checks the resulting domain set through the same read-only transaction used by the extension. It also checks pause, resume, and archiving. Other regression tests cover report corrections, missing or malformed decisions, deleted evidence, pending-site edits, and negative choices.

## Workspace fallback

Both models were called through the normal workspace fallback after the test made the Kev endpoint unavailable. The writing task should open a desk and the workstation-only PDF should stay unavailable.

| Fallback | Missing PDF | Drafting text |
| --- | --- | --- |
| [Bonsai 4B](4b-fallback.json) | Ineligible | Eligible |
| [Bonsai 9B](9b-fallback.json) | Ineligible | Eligible |

Across the 40 shipped open or addable tasks, the number of eligible workspaces changed from 10 to 9. All 40 have exact packaged Kev decisions. [Per-task decisions](public-workspaces.json)

A final dashboard race appeared when workspace eligibility finished while the user clicked **Edit time spent**. Refreshing the entire app before the card handled its click could discard it. A deterministic test resolves the eligibility result between the initial check and the fragment poll during that click. It reproduced the missing dialog. The workspace offer now updates only its own fragment, and the same test opens the time editor. [Dashboard regression](../../../../tests/test_manual_time.py)
