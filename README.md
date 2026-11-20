# DSIC-3007 — Causal Audit of Domain-Conditioned Expert Routing

Penelitian ini menguji apakah statistik penggunaan expert yang dihitung per
domain lebih informatif tentang kontribusi kausal expert daripada statistik
global pada **OLMoE-1B-7B-0924 base**. Kontribusi diukur melalui perubahan
next-token loss setelah satu expert aktif diintervensi pada token dan layer
yang sama.

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

## Proposal Seminar

Naskah Bab I–III untuk seminar proposal 2026 tersedia di
[`paper/proposal-sempro-v2/`](paper/proposal-sempro-v2/README.md), dengan PDF,
sumber LaTeX, bibliografi, dan lampiran protokol teknis. Proposal tersebut
mengembangkan pertanyaan lintas kedalaman pada OLMoE serta pemeriksaan
generalisasi eksploratif pada Qwen1.5-MoE-A2.7B BF16 dan DeepSeek-V2-Lite BF16.

Proposal seminar adalah dokumen akademik; konfigurasi dan README ini tetap
menjadi acuan untuk pipeline Path B yang dapat dijalankan. Path B saat ini
membatasi pengukuran kausal pada OLMoE layer 15 dan domain scientific/code.
Proposal mencatat diagnostik awal yang terbatas, bukan hasil penelitian utama.
Nilai target presisi pilot, toleransi numerik BF16, serta tanggal dan anggaran
GPU masih perlu ditetapkan sebelum tahap terkait.

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
model pinned tersebut, tetapi bukan penerimaan implementasi baru. Akuisisi
memverifikasi identitas file sekali; cache tersiap tidak di-hash ulang pada setiap
run. Keberadaan katalog lokal tidak membuktikan readiness runtime GPU.

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

README dan [config/research.json](config/research.json)
menetapkan studi aktif. Keputusan tercatat dalam
[path-b-decision-v1.json](docs/path-b-decision-v1.json); konfigurasi dan implementasi lama tersedia melalui recovery tag
`archive/path-b-before-rebuild-2026-10-13`.
Detail evidence literatur tersedia
di [prior-work ledger](docs/literature/prior-work-ledger.md).

## Implementasi dan Eksekusi

Implementasi baru memakai `config/`, `scripts/`, dan `terraform/`.
[WORKFLOW.md](WORKFLOW.md) menjelaskan konfigurasi, preparation, verifikasi GPU,
calibration, pilot, main, resume, retrieval, dan cleanup.
[Catatan rebuild](docs/rebuild.md) mencatat recovery, optimizations yang dipertahankan,
checks yang dijalankan, serta batas evidence. Parameter executable ada di
[config/research.json](config/research.json); aturan ilmiah tetap mengikuti README ini.

CPU/synthetic checks adalah evidence implementasi. Native acceptance pretrained
FP32 GPU, measured production throughput, calibration stability dan achieved
precision untuk implementasi baru belum ditetapkan. Historical findings tidak
menggantikan pengukuran tersebut.
