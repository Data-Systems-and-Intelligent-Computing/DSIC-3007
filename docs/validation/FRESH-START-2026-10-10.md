# Fresh-clone evidence and long-document acquisition plan

## What is actually available

Pinned CPU retokenization processed the original 30,000 retained records without
changing text, adding special tokens, concatenating documents or including the
target in the input. Eligible predictors have 4095 preceding original tokens
and one subsequent target. Preparation is sealed at
data/prepared-inputs/retained-hostnames-2026-10-10. This successor corrects
the earlier preparation's removal of `www.` from C4 hostnames; raw files
and the earlier bundle remain intact. All 30,000 records were compared:
5,014 hostname labels changed and every other document field is identical.

Manifest SHA-256:
c1e94b872449958853c63c716504ed433cd2c29361bdbcc960950a9442a68fb7.

| Domain / retained split | Records | Full-context documents | Eligible predictors |
|---|---:|---:|---:|
| C4 calibration | 1,500 | 7 | 17,166 |
| C4 evaluation | 8,500 | 64 | 185,738 |
| Scientific calibration | 1,500 | 1,071 | 3,826,757 |
| Scientific evaluation | 8,500 | 6,095 | 21,683,764 |
| Code calibration | 1,500 | 99 | 488,080 |
| Code evaluation | 8,500 | 597 | 3,525,305 |

These are token eligibility counts, before source/variant exclusions.
The [source audit](SOURCE-AUDIT-2026-10-10.md) finds 70 eligible C4 hostnames,
692 eligible code repositories, no exact text duplicates, and one shared
scientific-publication URL requiring review.
Calibration cannot reach even 100 C4 documents or 100 code documents within
the retained calibration split. Combined C4 inventory has only 71 candidates,
so reassigning retained splits would also fail the 250-document freeze minimum.
Code's 696 combined candidates do not establish sufficient independent groups
for technical, calibration, probe, pilot and the 200-group main minimum.
The source review remains pending; no scientific sample is approved.

## Acquisition plan selected by Arya

Keep full 4096 input context. Add long documents in a **new** collection and
manifest; preserve the original retained files. No additional acquisition has
started. Before downloading, specify the bounded scan/document target and
source revisions for the supplement; do not infer them from missing old files.
Use the existing source families (C4 English validation, peS2o train,
StarCoderData Python train) and the exact pinned research tokenizer.

1. Read source revision, license/attribution/terms and native source fields from
   the retained source catalog. Keep full revision identities in the supplement.
2. Tokenize original standalone text with no external normalization or special
   tokens. Retain a record only when it has at least 4097 tokens, and preserve
   exact text/token hashes plus native metadata and known duplicate aliases.
3. Audit C4 hostnames, scientific paper identities/variants, repository aliases
   and known clones. Exclude duplicated content with explicit reasons.
   Preserve every eligible retained file of each selected code component.
4. Allocate source groups disjointly across technical, calibration, probe, pilot
   and main. Include original records in a combined manifest only if byte
   identities match. Keep exclusion decisions and the complete review.
5. Count document and source-group availability before generating GPU plans.
   Probe documents must support four eligible predictor positions. Main uses
   all available positions when a document has fewer than 64.

The study needs capacity for up to 500 calibration documents/domain; 32 fixed
probe documents/domain; 12 distinct technical groups/domain; up to 80 pilot
scientific/C4 documents or code groups; at least 200 scientific main documents
and 200 main code groups; and 200 C4 main documents. These partitions are
independent. Main allocation may exceed the minima after pilot planning.
Do not convert document demand into source-group demand without the audit.
No arbitrary oversampling multiplier or shorter-context substitute is selected.
The proposed 200,000 C4 / 40,000 code yielded-row scans and flagged-paper
decision are awaiting user answers in [collection-proposal.json](../../configs/collection-proposal.json).

## Implementation evidence and claim boundary

The native packed-expert hooks collect lossless routing and all eight individual
token-local effects. Raw complete events are journaled with input/model/runtime
identity; failures remain incomplete. Local analysis checks plan coverage and
uses source-group bootstrap with document-weighted estimands. The workflow
includes calibration freezing, technical runtime planning, exact main freezing,
and matched 2048 context collection/analysis.

Final checks: 46 CPU tests passed (9.32s); dependency lock consistency, Python
compilation and diff whitespace checks passed. Terraform 1.16.5 with Runpod
provider 1.0.1 passed formatting and schema validation. The CPU suite used
temporary collector dependencies, not a GPU container.

CPU tiny-model and synthetic sealed-bundle tests are regression evidence.
Terraform provider initialization/schema validation are local checks.
They do not establish CUDA/image compatibility, pretrained FP32 mechanics,
whole-model memory fit, scientific stability, pilot precision or causal results.
No R2 transfer, full-weight download, GPU experiment or paid resource launch
was performed during this implementation.

Commands and saved-data details are in [WORKFLOW.md](../../WORKFLOW.md).
