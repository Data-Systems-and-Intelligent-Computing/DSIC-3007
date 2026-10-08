# Input Identity and Allocation Manifests

Folder ini eligible untuk Git, berbeda dari corpus/weights yang tetap ignored.
Catalog kecil tidak berisi raw source text atau credentials.

- [model-fp32.json](model-fp32.json): full HF commit dan expected per-file
  SHA-256/size untuk native FP32. Metadata/JSON sudah diperiksa; shards belum
  diunduh atau diverifikasi lokal. Record BF16 sebelumnya tetap provenance.
- [collection-10k-v1.json](collection-10k-v1.json): actual retained-file byte
  hashes/sizes, source revisions dan metadata-reported counts. R2 distribution
  masih planned; proposed keys tidak berarti objek sudah tersedia.

Audit/technical/probe/pilot/main allocation manifests belum dibuat. Field dan
sampling/group/token policies mengikuti [protocol](../../EXPERIMENT_PROTOCOL.md),
bukan defaults template historis. Outcome-independent audit mendahului allocation;
include stable source IDs, content hashes, groups, inclusion probabilities,
eligibility/exclusion reasons, seeds dan manifest version/hash.

Byte identity tidak memvalidasi licenses, leakage, independence atau runtime.
Public distribution memerlukan source permission/attribution/terms audit;
input acquisition harus memverifikasi SHA-256/size dari catalog yang dibekukan.
