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

Analisis pendukung membahas keunggulan M2 pada masing-masing domain,
perbedaan descriptive menurut depth, sinyal per-token routing weight, serta
matched/mismatched domain rankings.

## Model

Gunakan **OLMoE-1B-7B-0924 base**, dengan seluruh model weights tetap.
Arsitektur acuan memiliki 16 MoE layers, 64 experts/layer, dan top-8 routing
tanpa shared expert. Primary menggunakan model tanpa quantization dalam FP32.

Keputusan 2026-10-08 memilih **weights FP32 asli** dari branch `fp32`, dikunci
pada commit `c6f6954f182ffdd9f6ab4f4d772714b7dcd6836b`, bukan casting weights
BF16 lokal menjadi FP32. [Manifest model](data/manifests/model-fp32.json)
mencatat identitas file yang diharapkan; weight shards belum diunduh atau
diverifikasi lokal. Config/tokenizer JSON dan remote identity metadata sudah
diperiksa. Detail perubahan ada pada dated amendment di protocol.

Tidak ada fine-tuning, LoRA, router training, atau recovery training. Model
base dipilih karena outcome adalah next-token prediction pada natural corpus;
instruction-following bukan objek pengukuran penelitian ini.

## Domain dan Data

| Domain | Sumber | Peran |
|---|---|---|
| General web | C4 English validation | Komponen global calibration dan generic diagnostic. |
| Scientific text | peS2o train | Domain primary dan domain-conditioned calibration. |
| Source code | StarCoderData Python train | Domain primary dan domain-conditioned calibration. |

Domain ditentukan dari sumber dataset. Scientific dan code memberikan kontras
distribusi token terhadap general web. Natural text menyediakan target
next-token sebenarnya, sehingga core tidak memerlukan gold QA answers.

Retained inventory berisi 10.000 dokumen/domain: 1.500 calibration dan 8.500
evaluation-pool. Angka ini adalah persediaan; final eligible counts dan main
sample ditentukan setelah audit dan pilot.

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

Primary memakai total window **4096 token**, dengan minimal **128 preceding
tokens**. Predictor dan target harus token asli yang eligible; window berasal
dari dokumen yang sama, tanpa concatenation antar-dokumen. Posisi dipilih lebih
dulu, kemudian konteks aslinya dibentuk menurut window policy.

- **Layer 15:** primary dan domain-specific confirmatory secondary; mulai
  **64 posisi/dokumen**, dengan escalation ke 128/256 jika pilot menunjukkan
  position-sampling noise terlalu besar.
- **Layer 0 dan 7:** descriptive secondary pada first **16 nested positions**
  setiap main document.

Dokumen dengan lebih sedikit eligible positions memakai seluruh posisi yang
tersedia. Actual counts dilaporkan. Layer 15 memberi satu primary setting,
sementara layer 0/7 mempertahankan pengamatan tentang depth tanpa memperbanyak
primary tests.

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

Gunakan seluruh eligible retained calibration documents. Position density
dimulai dengan assessment 128/256 dan dinaikkan jika stability criteria belum
terpenuhi. M1/M2 dibekukan bersama setelah density/seed stability dan fixed-probe
selection agreement lolos. Calibration mengamati native routing tanpa ablation.

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

Pilot mulai dengan 100 scientific documents, 100 whole code groups dan 100 C4
documents, pada 64 posisi/dokumen. Variance, position noise dan projection
uncertainty digunakan untuk menetapkan position budget serta main allocation.

Target planning adalah resolution **0.01 nats** dengan CI half-width **0.005**.
Main minimal 200 scientific documents dan 200 code groups; jumlah final dipilih
melalui aturan pilot yang mempertimbangkan precision, group safeguards dan
feasibility. Target tersebut belum membuktikan effect size atau achieved power.

Exact main IDs, positions dan analysis dibekukan sebelum main outcomes.
Perluasan pilot mengikuti projection rules; main tidak ditambah untuk mengejar
signifikansi atau membuat M2 menang.

## Analisis Statistik

Primary menggunakan paired two-sided **95% bootstrap-t**, conditional pada
frozen calibration. Resampling dilakukan atas scientific documents dan whole
code groups, dengan semua files serta draw multiplicities dipertahankan.
Code bootstrap menghitung ulang document-weighted ratio dan standard error.

Mulai dengan 20.000 replicates; prespecified Monte Carlo dan degeneracy rules
mengendalikan kebutuhan tambahan draws serta validitas interval. Expert effects,
layers dan repeated forwards tidak menambah independent source observations.

Scientific/code layer-15 advantages adalah dua confirmatory secondary claims
dengan Holm correction. Layer 0/7 dan C4 bersifat descriptive/diagnostic.
C4 diagnostic uncertainty mempertimbangkan hostname clusters sesuai safeguards.

Laporkan absolute CD_M1, CD_M2 dan paired difference beserta uncertainty.
Interval yang melintasi nol menunjukkan ketidakpastian; tidak membuktikan
equivalence. Core tidak menetapkan practical-null claim.

## Sensitivitas dan Batas Scope

Selected exploratory analyses mencakup token-weighted M2 dengan equal-domain
M1, matched/mismatched domain rankings, dan matched-policy context **2048**
pada subset yang ditetapkan. Context sensitivity memerlukan calibration yang
cocok serta intervention effects pada input baru. Reselecting expert pairs
pada input yang sama dapat memakai efek yang sudah diukur.

M3 domain enrichment, B/C operators, layer 4/11, context 1024, equal-token/
token-pooled M1, tambahan domains/models dan Do-style scoring tetap di luar
core. Adopsinya membutuhkan keputusan ilmiah tersendiri dan tidak menggantikan
primary comparison yang sudah dipilih.

## Hipotesis dan Batas Interpretasi

Ekspektasi yang diuji adalah manfaat M2 pada scientific/code, manfaat lebih
kecil pada C4, variasi menurut depth, dan sinyal positif M4 pada layer akhir.
Ekspektasi ini tidak menjadi syarat hasil penelitian harus positif.

Keabsahan measurement bergantung pada native/no-op reconstruction, target
alignment, local zero-output, unchanged survivors dan cleanup yang benar.
Integrity failures dibedakan dari scientifically negative results.

Kesimpulan dibatasi pada tested OLMoE, source domains, context policy dan
intervensi A. High/low discrimination tidak membuktikan full-ranking accuracy,
unique domain knowledge, domain exclusivity, atau keamanan permanent pruning.
Domain sendiri tidak diintervensi; keuntungan M2 tidak mengisolasi causal
effect dari semantic domain terhadap semua source-associated properties.

Detail aturan metode tersedia di [EXPERIMENT_PROTOCOL.md](EXPERIMENT_PROTOCOL.md)
yang mengonsolidasikan spesifikasi v1. Detail evidence literatur tersedia
di [prior-work ledger](docs/literature/prior-work-ledger.md).

## Lokasi Kerja

`src/` adalah satu root kode: collector/profiler aktif berada di
`src/collectors/`, dengan tes di `tests/collectors/`. Modul inference bawaan,
tes lainnya, `configs/` dan root `pyproject.toml` masih scaffold yang harus
dicocokkan dengan protokol sebelum digunakan sebagai implementasi aktif.
Dependency collector Python 3.12 terpisah dari planned inference runtime.

`paper/` berisi naskah penelitian; `references/` berisi sumber paper,
sedangkan `docs/literature/` mencatat evidence literatur. Data retained ada di
`data/collection-10k-v1/`; target model FP32 di
`models/OLMoE-1B-7B-0924-fp32/` belum diperoleh. Snapshot BF16 lama di
`models/OLMoE-1B-7B-0924/` tetap dipertahankan.
`runpod/` direncanakan untuk IaC/container, belum diimplementasikan.

Urutan kerja dan status terbaru ada di OPERATIONAL.md lokal. AGENTS.md,
EXPERIMENT_PROTOCOL.md, OPERATIONAL.md dan TODO.md Git-ignored sesuai arahan
Arya. README merangkum desain; detail metode/pelaksanaan tersedia pada dokumen
lokal tersebut dan tidak otomatis tersedia dari checkout Git.

## Reproducibility dan Distribusi Input

Policy 2026-10-08 menyiapkan kode, tests, dependency locks, README, ignore rules
dan small identity/evidence manifests untuk Git. Arahan terbaru Arya pada tanggal
yang sama mengembalikan AGENTS.md, EXPERIMENT_PROTOCOL.md dan OPERATIONAL.md
ke Git ignore. TODO, materi pribadi, reference PDFs, history, corpus, weights,
cache dan credentials tetap ignored. Git eligibility belum berarti perubahan
sudah di-commit.

Rencana dataset adalah snapshot di Cloudflare R2 dengan object paths berversi,
SHA-256, ukuran, source revisions dan metadata inventori yang tercatat di
[manifest collection](data/manifests/collection-10k-v1.json). URL/object keys
dan access policy belum tersedia; tidak ada upload atau public access yang
diklaim. Distribusi mengikuti audit license/attribution/terms per sumber,
termasuk persetujuan penerima ketika diwajibkan. Manifest byte identity tidak
membuktikan corpus eligibility, source independence atau bebas leakage.

Target alur adalah checkout code → acquire pinned inputs → verify checksums →
verification/calibration/effects → analysis of saved artifacts. Runner, resolved
inference environment dan clean-checkout execution masih pending. Sumber
weights berasal dari Hugging Face, diunduh pada full commit yang dikunci dan
diverifikasi sebelum model diload dari file lokal.
