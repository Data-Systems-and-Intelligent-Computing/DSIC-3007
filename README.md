# DSIC-3007 — Causal Audit of Domain-Conditioned Expert Routing

Penelitian ini menguji apakah statistik penggunaan expert yang dihitung per
domain lebih informatif tentang kontribusi kausal expert daripada statistik
global pada **OLMoE-1B-7B-0924 base**. Kontribusi diukur melalui perubahan
next-token loss setelah satu expert aktif diintervensi pada token dan layer
yang sama.

## Amendment — Path B selected, 2026-10-12

Arya selected **Path B** after comparing the original design, fresh feasibility
findings, and the supervisor's theory draft. The scope decision is recorded in
[configs/path-b-decision-v1.json](configs/path-b-decision-v1.json). This amendment
defines the active scope throughout this README. The earlier design is archived
separately; historical settings and raw evidence remain preserved.

- Primary remains original FP32 OLMoE, exact **4096 input tokens**, layer **15**,
  scientific/code equal-domain paired M2-minus-M1 discrimination, and all eight
  individual zero-contribution effects without replacement or renormalization.
- Main and pilot use **eight uniformly sampled nested positions/document**.
  Main retains at least **200 scientific documents and 200 whole code groups**;
  every eligible file in selected code groups remains included.
- C4 remains in calibration/probe and equal-domain M1. Main/pilot effects cover
  scientific/code. **C4 outcomes, lower-layer outcomes, and matched2048 sensitivity
  are deferred**; depth/C4/context-robustness claims are not part of this scope.
- Technical, calibration and probe scope becomes **layer15/4096**, retaining all
  three calibration domains, three seeds, the nested100/250/500 ladder, existing
  stability thresholds, and frozen numerical tolerances. New scoped acceptance
  must be established; old diagnostic PASS receipts do not grant full acceptance.
- Paired source-group inference, controls and target CI halfwidth **.005 nat**
  remain. The fallback policy if that precision cannot fit the budget is pending
  and must be settled before main outcomes, using the pilot planning evidence.
- The article is a goal; M3/M5, mean ablation and equivalence tests remain
  separate prospective decisions. Existing matched/mismatched comparisons can
  reuse the same complete eight-effect measurements.

**Implementation status:** the local migration is implemented in
[configs/research-path-b-v1.json](configs/research-path-b-v1.json) and the workflow
defaults to that version. Stage domains/layers, receipt scope, calibration family,
analysis and remaining-workflow budget admission now follow Path B. Original
`configs/research.json`, old sealed plans and raw evidence remain historical.
Native acceptance for the changed implementation is pending. Follow the
[migration and verification record](docs/validation/PATH-B-IMPLEMENTATION-2026-10-12.md).

The original **$0.20 / 15-minute diagnostic grant was consumed** by the failed
2026-10-13 benchmark attempt. The earlier **$0.60 retry was withdrawn**. Arya now authorized one
**$0.30 / 900-second user-run benchmark attempt**. Full scientific
collection is not authorized; invoices and remaining original $12/10h study
allowance are unreconciled.

## Calibration, Sampling dan Numerical Gates

Master seed **3007**; calibration position seeds **3008, 3009, 3010**.
Technical, calibration, probe, pilot dan main terpisah menurut audited source
groups: C4 hostname, scientific paper ID dan code repository/component ID.
Duplicate content, article variants, repository aliases dan known clones diaudit
sebelum sampling. Technical reserve tetap **12 groups/domain**, dengan empat
posisi deterministik/dokumen: first, last dan dua interior thirds, seluruhnya
pada layer 15 dengan delapan expert individual.

Fixed probe memakai **32 dokumen/domain × 4 posisi/dokumen**. Calibration memakai
nested **100→250→500 dokumen/domain**, satu full window/dokumen/seed, dan native
routing pada semua input positions. Utilization dirata-ratakan dengan bobot
dokumen sama lalu bobot tiga seed sama; M1 memberi bobot domain sama.

Stability tetap **L1 drift ≤0.05** atas 64 utilization values, tanpa dibagi delapan,
dan **top-8 serta bottom-8 Jaccard ≥0.75**, lolos pada **≥90%** perbandingan
domain/layer. Setiap candidate seed-pair dan consecutive averaged-table budget
comparison harus memenuhi aturan tersebut. Freeze 250 setelah 100→250 lolos,
atau 500 setelah 250→500 lolos; predecessor seed stability tidak diwajibkan.
Jika tidak ada candidate lolos, laporkan instability. Simpan seed/budget tables,
frozen rankings dan probe selection agreements.

Routing, survivor coefficients dan cleanup harus exact. FP32 mixture/logit
checks tetap **atol=1e-6, rtol=1e-5**; NLL tolerance **1e-6 nat**.
Receipt penerimaan baru untuk scope/implementation Path B masih diperlukan.

Desain sebelumnya dan inventory awal tersimpan di
[arsip desain 2026-10-10](docs/validation/README-HISTORICAL-DESIGN-2026-10-10.md).

## Fokus dan Pertanyaan Penelitian

**RQ utama:** apakah raw domain-conditioned utilization (M2) menghasilkan
tambahan high/low causal discrimination atas global utilization (M1),
di bawah single-expert token-local zero-output ablation?

Domain conditioning dapat mempertahankan pola penggunaan expert yang tertutup
oleh campuran domain. Namun, sering digunakan dalam suatu domain belum tentu
berarti kontribusinya lebih besar terhadap prediksi. Karena itu, routing
statistics diuji terhadap intervention effects pada data held-out.

Kontribusi penelitian adalah perbandingan paired M2-versus-M1 dengan global
baseline, intervensi, dan populasi evaluasi yang dinyatakan eksplisit. Penelitian
tidak mengusulkan routing frequency atau expert ablation sebagai metode baru.

Analisis pendukung membahas keunggulan M2 pada scientific/code, sinyal
per-token routing weight dan matched/mismatched domain rankings pada layer 15.
Perbandingan depth dan causal outcomes C4 ditunda.

## Model

Gunakan **OLMoE-1B-7B-0924 base**, dengan seluruh model weights tetap.
Arsitektur acuan memiliki 16 MoE layers, 64 experts/layer, dan top-8 routing
tanpa shared expert. Primary menggunakan model tanpa quantization dalam FP32.

Keputusan 2026-10-08 memilih **weights FP32 asli** dari branch `fp32`, dikunci
pada commit `c6f6954f182ffdd9f6ab4f4d772714b7dcd6836b`, bukan casting weights
BF16 lokal menjadi FP32. [Manifest model](data/manifests/model-fp32.json)
mencatat identitas file yang diharapkan. Diagnostic GPU terdahulu memakai
model pinned tersebut, tetapi bukan penerimaan penuh Path B. Preflight memakai
cache remote yang harus diverifikasi ulang; keberadaan manifest atau plan lokal
tidak membuktikan kesiapan cache/runtime saat ini.

Tidak ada fine-tuning, LoRA, router training, atau recovery training. Model
base dipilih karena outcome adalah next-token prediction pada natural corpus;
instruction-following bukan objek pengukuran penelitian ini.

## Domain dan Data

| Domain | Sumber | Peran |
|---|---|---|
| General web | C4 English validation | Calibration M1/M2, fixed probe dan technical checks; causal pilot/main ditunda. |
| Scientific text | peS2o train | Domain primary dan domain-conditioned calibration. |
| Source code | StarCoderData Python train | Domain primary dan domain-conditioned calibration. |

Domain ditentukan dari sumber dataset. Scientific dan code memberikan kontras
distribusi token terhadap general web. Natural text menyediakan target
next-token sebenarnya, sehingga core tidak memerlukan gold QA answers.

Retained inventory berisi 10.000 dokumen/domain: 1.500 calibration dan 8.500
evaluation-pool. Angka ini adalah persediaan; final eligible counts dan main
sample ditentukan setelah audit dan pilot. Supplements dan prepared successor
Path B tersimpan terpisah; inventory awal bukan batas seluruh eligible pool.

## Pemisahan Data dan Sampling

Calibration dan evaluation terpisah pada tingkat source document serta known
source groups. Audit duplicates, article variants dan repo/clone relationships
mendahului sampling untuk membatasi leakage dan ketergantungan sumber.

Technical checks, fixed probe, effect pilot dan main memakai source units
terpisah. Scientific memakai dokumen yang didukung audit independensi. Code
memakai whole eligible repository components: seluruh eligible documents dari
group yang terpilih ikut dianalisis. Ini mempertahankan dependensi antar-file
ketika sampling dan resampling.

Posisi dipilih uniform tanpa replacement menggunakan seeded nested sampling.
Ordinary whitespace, newline dan punctuation tetap eligible. Fully tied
events dan identical M1/M2 selections juga dipertahankan agar evaluasi tidak
mengondisikan sampel pada adanya perbedaan ranking.

## Konteks, Posisi dan Layer

Primary memakai **4096 input tokens**, dengan **4095 preceding
tokens** sebelum predictor terakhir dan target di luar window. Predictor dan target harus token asli yang eligible; window berasal
dari dokumen yang sama, tanpa concatenation antar-dokumen. Posisi dipilih lebih
dulu, kemudian konteks aslinya dibentuk menurut window policy.

- **Layer 15 saja:** seluruh technical, calibration, probe, pilot dan main.
- **Pilot dan main:** delapan uniform seeded nested positions/dokumen;
  dokumen dengan kurang dari delapan eligible positions memakai seluruhnya.
  Seluruh eligible files dalam selected code groups tetap disertakan.
- **Layer 0/7 dan matched 2048:** ditunda, termasuk matched 2k calibration.

Actual position counts dilaporkan. Delapan posisi mempertahankan pengukuran
berulang dalam dokumen dengan biaya lebih kecil daripada desain 64 posisi;
ketidakpastian aktual tetap harus dinilai melalui pilot.

## Metric Routing

### M1 — Global Utilization

Global utilization adalah campuran berbobot sama dari tiga calibration domain:

```text
M1(e) = [U_C4(e) + U_scientific(e) + U_code(e)] / 3
```

Target-domain calibration ikut dalam campuran; evaluation documents tidak ikut.
Bobot yang sama mencegah ukuran inventory suatu domain menentukan global
baseline. M1 bukan pooled token frequency dari seluruh corpus.

### M2 — Domain Utilization

Untuk tiap calibration document, hitung proporsi sampled eligible positions
yang memilih expert `e`. Kemudian rata-ratakan document means dengan bobot
dokumen sama dalam domain tersebut. Ini menghindari dominasi dokumen panjang.

Gunakan nested budgets 100→250→500 dokumen/domain dan tiga position seeds.
Satu full window/dokumen/seed mengamati semua input positions. Freeze budget
yang lebih besar pada comparison pertama yang lolos. Fixed-probe selection
agreement dilaporkan bersama frozen rankings. Calibration tanpa ablation.

Utilization adalah active-set membership frequency, bukan mean routing weight.
Dengan top-8 dan denominator konsisten, jumlah utilization semua expert adalah
delapan. Semua skor dihitung per layer.

### M0 dan M4 — Kontrol

**M0** memakai uniform random ordered pair dari dua distinct active experts,
dengan 64 seeded draws/event. Expected CD nol; finite draws tidak harus nol.
Randomization variability dilaporkan terpisah dari source-sampling uncertainty.

**M4** memakai native effective routing coefficient pada token/layer yang diuji.
Ini diagnostic positive control. Sinyal lemah memicu pemeriksaan mekanisme;
setelah mekanisme terbukti benar, hasil lemah tetap dapat menjadi temuan sah.

## Intervensi dan Outcome

Intervensi **A** men-zero-kan kontribusi satu naturally active expert pada satu
predictor token dan satu layer. Active membership dan coefficients survivor
lokal tetap, tanpa expert pengganti atau renormalization. Downstream computation
berjalan normal dan boleh merespons perubahan hidden state.

Outcome adalah teacher-forced **actual next-token NLL**, dalam nats, tanpa
router auxiliary loss:

```text
DeltaL_e = NLL setelah A pada expert e − baseline NLL
```

DeltaL positif berarti penghilangan kontribusi itu memperburuk prediksi lokal;
negatif berarti memperbaikinya. Pada setiap event yang diukur, kumpulkan delapan
expert effects melalui intervensi individual yang masing-masing dimulai dari
baseline yang sama. Efek tersebut dipakai bersama untuk metric comparisons
dan kontrol; delapan effects bukan delapan independent samples.

## Causal Discrimination dan Endpoint Primer

Setiap metric memilih high/low dari baseline-active set yang sama, berdasarkan
metric scores tanpa melihat intervention outcomes. Exact ties memakai shared
seeded ordering dengan high/low yang distinct.

```text
CD_m = DeltaL_high(m) − DeltaL_low(m)
delta = CD_M2 − CD_M1
```

CD positif menunjukkan high/low discrimination. Paired delta mengukur tambahan
discrimination M2 atas M1 pada posisi/layer yang sama.

Rata-ratakan delta dalam dokumen, lalu gunakan equal-document mean dalam domain.
**Endpoint tunggal** adalah aggregate layer 15:

```text
theta_primary = 0.5 × theta_scientific + 0.5 × theta_code
```

Code estimate tetap document-weighted atas files dalam selected groups, bukan
equal-repo mean. Equal domain weights menjaga scientific/code berperan sama
pada endpoint, meskipun final sample sizes dapat berbeda. C4 tidak masuk primary.

## Pilot dan Penetapan Main

Pilot memakai nested **20→40→80 scientific documents dan whole code groups**,
dengan **delapan posisi/dokumen**, sama dengan main. Semua eligible files dalam
selected code groups ikut. Extend jika ada kurang dari 10 independent groups
atau 95% CI halfwidth melebihi 0.25 × source-group SD; laporkan insufficiency
jika batas pilot terakhir belum cukup.

Cluster variance dan 2000 bootstrap-t draws menentukan precision projection;
measured full-technical runtime memilih qualifying allocation tercepat dari
reviewed inventory. Main freeze juga memerlukan ledger biaya/waktu seluruh
tahap tersisa, evidence per tahap, reconciled remaining limits dan ≥20% reserve.

Target planning adalah resolution **0.01 nats** dengan CI half-width **0.005**.
Main minimal 200 scientific documents dan 200 code groups; jumlah final dipilih
melalui aturan pilot yang mempertimbangkan precision, group safeguards dan
feasibility. Target tersebut belum membuktikan effect size atau achieved power.

Minimum-main dry-run lokal berisi **200 scientific documents + 205 code files
pada 200 whole code groups**, atau **3.240 windows / 25.920 expert effects yang
direncanakan**. Ini census desain, bukan observasi atau main allocation final.
Fallback jika target precision tidak muat budget masih harus diputuskan dengan
pilot evidence sebelum main outcomes.

Exact main IDs, positions dan analysis dibekukan sebelum main outcomes.
Perluasan pilot mengikuti projection rules; main tidak ditambah untuk mengejar
signifikansi atau membuat M2 menang.

## Analisis Statistik

Primary menggunakan paired two-sided **95% bootstrap-t**, conditional pada
frozen calibration. Resampling dilakukan atas whole scientific source groups dan whole
code groups, dengan semua files serta draw multiplicities dipertahankan.
Code bootstrap menghitung ulang document-weighted ratio dan standard error.

Document means dihitung sebelum inference. Domain cluster variance tetap
`G/(G−1) × Σg[Σi∈g(Di−Dbar)]²/N²`; primary variance
`0.25 × (V_scientific + V_code)`. Pilot projection mentransfer domain variance
menurut rasio pilot/main source-group counts pada budget delapan posisi yang sama.

Gunakan 10.000 final replicates dan 2000 planning replicates. CI memakai linear
quantiles; tail probabilities `(count+1)/(valid_draws+1)` sebelum Holm correction.
Laporkan degeneracy dan wajibkan ≥95% valid draws untuk interval. Expert effects,
layers dan repeated forwards tidak menambah independent source observations.

Scientific/code layer-15 advantages adalah dua confirmatory secondary claims
dengan Holm correction. Path B tidak mengoleksi main/pilot outcomes C4 atau
layer 0/7; tidak ada interval atau claim untuk outcomes yang ditunda tersebut.

Laporkan absolute CD_M1, CD_M2 dan paired difference beserta uncertainty.
Interval yang melintasi nol menunjukkan ketidakpastian; tidak membuktikan
equivalence. Core tidak menetapkan practical-null claim.

## Sensitivitas dan Batas Scope

Matched/mismatched domain rankings tetap tersedia dari complete eight-effect
measurements yang sama; reselecting pairs tidak memerlukan ablation tambahan.
C4 tetap menyumbang calibration untuk M1 dan mismatched-ranking controls.
Matched **2048**, causal outcomes C4 dan layer 0/7 ditunda. Path B tidak dapat
menyimpulkan robustness menurut panjang context, depth atau C4 causal outcomes.

M3 domain enrichment, M5 activation norm, mean ablation, equivalence tests,
B/C operators, layer 4/11, context 1024, equal-token/
token-pooled M1, tambahan domains/models dan Do-style scoring tetap di luar
core. Adopsinya membutuhkan keputusan ilmiah tersendiri dan tidak menggantikan
primary comparison yang sudah dipilih.

## Hipotesis dan Batas Interpretasi

Ekspektasi yang diuji adalah manfaat M2 pada scientific/code serta sinyal
positif M4 pada layer 15.
Ekspektasi ini tidak menjadi syarat hasil penelitian harus positif.

Keabsahan measurement bergantung pada native/no-op reconstruction, target
alignment, local zero-output, unchanged survivors dan cleanup yang benar.
Integrity failures dibedakan dari scientifically negative results.

Kesimpulan dibatasi pada tested OLMoE, source domains, context policy dan
intervensi A. High/low discrimination tidak membuktikan full-ranking accuracy,
unique domain knowledge, domain exclusivity, atau keamanan permanent pruning.
Domain sendiri tidak diintervensi; keuntungan M2 tidak mengisolasi causal
effect dari semantic domain terhadap semua source-associated properties.

README dan [configs/research-path-b-v1.json](configs/research-path-b-v1.json)
menetapkan studi aktif. Keputusan tercatat dalam
[path-b-decision-v1.json](configs/path-b-decision-v1.json); konfigurasi
`configs/research.json` adalah desain historis.
Detail evidence literatur tersedia
di [prior-work ledger](docs/literature/prior-work-ledger.md).

## Lokasi Kerja

`src/` contains preparation, routing/effect collection, local analysis and
private R2 transfer. The retained collector/profiler is in `src/collectors/`.
Root dependencies are pinned and locked separately from its environment.
CPU regression checks are implementation evidence; pretrained FP32 GPU
mechanism, fit and runtime verification remain pending.

`paper/` berisi naskah penelitian; `references/` berisi sumber paper,
sedangkan `docs/literature/` mencatat evidence literatur. Data retained ada di
`data/collection-10k-v1/`; target model FP32 di
`models/OLMoE-1B-7B-0924-fp32/` merupakan target local model path, bukan bukti
cache remote siap. Snapshot BF16 lama di
`models/OLMoE-1B-7B-0924/` tetap dipertahankan.
`runpod/` berisi bounded preflight runner dan default-disabled Terraform.
Prepared inputs aktif: `data/prepared-inputs/path-b-v1-2026-10-12/`;
sealed plans: `data/plans/path-b-v1-2026-10-12/`.
Local verification is recorded in the current execution note below. Path B
GPU attempts have not completed native acceptance or benchmark timings.
Tidak ada scientific results baru atau achieved-precision proof.

The fresh-clone workflow is in [WORKFLOW.md](WORKFLOW.md). Missing local-only
documents are intentional and are not prerequisites for this restart.
Ask Arya about unresolved scientific rules.

## Testing Biaya — L40S + Global Volume

Model preparation is **complete** on Global Volume `yelling_amber_shark`, ID
`cmv66f04p000007l8gngvd5k4`, at `/workspace-global/models/OLMoE-fp32`.
The successful retrieved setup receipt verifies all12 assets,27,679,451,403 bytes,
exact catalog sizes/SHA256 and F32 metadata. See the
[preparation receipt](docs/validation/global-model-preparation-success-2026-10-13.json).
Do not repeat setup/resume. Work/results use ephemeral local POSIX disk and are
retrieved before pod deletion; the Global Volume is retained. Region follows
L40S stock without a volume-region restriction. The pinned runtime is CUDA13.0;
benchmark hosts must advertise CUDA >=13.0 and pass actual pod read-back checks.

Arya instructed removal of repeated weight verification. Current v11 loads
weights directly from the prepared Global Volume without another SHA256/F32/
architecture verification pass. The existing completed preparation receipt is
required, but it is not fresh verification of current bytes. Cheap presence/size
checks remain. No model preparation is repeated. Native comparison gates,
FP32/eager, context4096/layer15 and all8 effects remain unchanged.

No local model copy, cache proof or disk resize is included; disk remains35GB.
The120/125/190s Path B timeout corrections remain. Changed loader source means
changed implementation identity; fresh native acceptance is pending.
The earlier $.60 proposal remains withdrawn. Current grant is one user-run
**$.30 / 900s** benchmark with no automatic retry. Setup/resume remain disabled.
At the configured guard rate and allowances, effective lease is about875.68s
including finalization/retrieval/cleanup. This is a planning guard, not an invoice
guarantee. Remaining study funds/invoices are unreconciled.

Offline source-package verification:

```bash
python -u runpod/path_b_preflight.py
```

User-run benchmark command (one attempt):

```bash
python -u runpod/path_b_preflight.py --run --benchmark-authorization-id global-path-b-benchmark-2026-10-13-v11
```

Durable log: `runpod/runs/latest/console.log`.
[Weight verification removal](docs/validation/PATH-B-WEIGHTS-REVERIFICATION-REMOVAL-2026-10-13.md) is the historical v10 change record.
Old frozen packages and raw receipts are preserved. No scientific results or
measured loading speed are claimed by this local fix.
