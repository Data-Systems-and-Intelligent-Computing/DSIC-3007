# Retained source audit and supplemental data proposal

All 30,000 retained records were checked, including short records that can
reveal source overlap. Private document-level evidence is sealed in
`data/source-audits/retained-hostnames-2026-10-10`; the count summary is
[SOURCE-AUDIT-2026-10-10.json](SOURCE-AUDIT-2026-10-10.json).

| Domain | Eligible documents | Eligible native groups | Documents supporting four probe positions |
|---|---:|---:|---:|
| C4 | 71 | 70 hostnames | 71 |
| Scientific | 7,166 | 7,166 paper IDs | 7,156 |
| Code | 696 | 692 repository IDs | 696 |

No exact text duplicates were found across the retained records. Two eligible
scientific records have the same publication URL, different paper IDs, and
different PDF hashes. The recommendation to exclude both is pending a user
answer; no automatic merge or exclusion occurred.

The old C4 calibration/evaluation labels overlap at 72 hostnames across all
records. No hostname appears in both eligible subsets, but the new five
scientific partitions still require explicit source-group allocation. Original
split labels alone are insufficient evidence for the new study's separation.

Code metadata supplies repository IDs and file paths but no external clone
genealogy. Case-insensitive repository-name links and recorded duplicate-source
aliases were checked; none linked different retained repository IDs. This does
not establish that all repositories are unrelated. No broader clone or article
similarity detector was specified, so those checks have not been called passed.
The allocation validator now enforces known transitive source-alias links even
when an excluded document connects two included units. Those units must have
the same reviewed canonical group and scientific stage.

## Corrected hostname identity

Preparation previously removed `www.` from C4 hostnames. The approved source
unit is the full hostname, so the implementation now preserves it. A new sealed
preparation at `data/prepared-inputs/retained-hostnames-2026-10-10` replaces that
derived representation. The earlier bundle and all raw corpus files remain
intact. Eligibility counts are unchanged.
Comparison of all 30,000 document records found 5,014 corrected hostname labels;
token IDs, text hashes, sampling eligibility and every other field are identical.

## Collection proposal awaiting answers

The proposal is [configs/collection-proposal.json](../../configs/collection-proposal.json).
Use the exact retained dataset revisions, scanning at most **200,000 C4 English
validation rows** and **40,000 StarCoderData Python training rows**. Retain every
unique document supporting 4096 input tokens plus its subsequent target.

Preparing capacity for the 500-document calibration ceiling, 32 probe documents,
12 technical groups, 80 pilot units, and 200 main units requires at least 824
documents per domain before whole-group allocation or exclusions. The retained
pool is therefore short by at least **753 C4 documents** and **128 code files**.
These are document-count lower bounds; independent group availability and
whole code group inclusion can increase demand. Pilot planning can also require
larger main allocations. The proposed scan limits do not guarantee sufficiency.

No supplement, R2 transfer, GPU experiment, or paid resource launch has occurred.
The collector and combined-input reader have offline regression coverage;
source streaming against these revisions and native scientific execution remain
unverified. Commands are documented in [WORKFLOW.md](../../WORKFLOW.md).
