# DSIC-3007 — Causal Audit of Domain-Conditioned Expert Routing

Penelitian ini menguji apakah statistik penggunaan expert yang dihitung per
domain lebih informatif tentang kontribusi kausal expert daripada statistik
global pada **OLMoE-1B-7B-0924 base**. Kontribusi diukur melalui perubahan
next-token loss setelah satu expert aktif diintervensi pada token dan layer
yang sama.

## Fokus dan Pertanyaan Penelitian

**Pertanyaan primer:** pada lapisan MoE terakhir OLMoE FP32, apakah M2 memberi tambahan
causal discrimination (CD) dibanding M1 pada agregat teks ilmiah dan kode berbobot sama?
Delta positif menunjukkan tambahan CD oleh M2.

Pertanyaan sekunder menguji selisih M2–M1 per domain pada lapisan pertama, tengah, dan
terakhir OLMoE. Enam kontras lapisan × domain memakai koreksi Holm. Perubahan CD menurut
kedalaman diringkas secara deskriptif, tanpa uji formal interaksi lapisan. Arah dan
besaran M2–M1 pada Qwen1.5-MoE-A2.7B BF16 serta DeepSeek-V2-Lite BF16 diperiksa secara
eksploratif dan terpisah per model.

Domain conditioning dapat mempertahankan pola penggunaan expert yang tertutup
oleh campuran domain. Namun, sering digunakan dalam suatu domain belum tentu
berarti kontribusinya lebih besar terhadap prediksi. Karena itu, routing
statistics diuji terhadap intervention effects pada data held-out.

Kontribusi penelitian adalah perbandingan paired M2-versus-M1 dengan global
baseline, intervensi, dan populasi evaluasi yang dinyatakan eksplisit. Penelitian
tidak mengusulkan routing frequency atau expert ablation sebagai metode baru.

Analisis sekunder yang dipra-spesifikasikan mencakup proporsi kejadian ketika pasangan
high/low M1 dan M2 berbeda beserta delta pada subset itu; Kendall tau-b untuk skor M1,
M2, M3, dan utilisasi per domain terhadap efek penghilangan; serta sensitivitas M1
leave-one-domain-out. Kendall melaporkan konkordansi, diskordansi, ties, tau-b, dan
proporsi pasangan tak ties yang konkordan. C4 menyumbang kalibrasi dan kontrol ranking;
outcome kausal C4 tidak dikumpulkan.

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

Generalisasi lintas model memakai checkpoint BF16 Qwen1.5-MoE-A2.7B dan
DeepSeek-V2-Lite, bukan bobot FP32 asli. Hasilnya eksploratif, dianalisis
terpisah, tidak digabungkan lintas tokenizer, dan tidak masuk keluarga Holm
OLMoE. Tidak ada fine-tuning, LoRA, router training, atau recovery training.
Outcome adalah next-token prediction pada natural corpus; instruction-following
bukan objek pengukuran.

## Domain dan Data

| Domain | Sumber | Peran |
|---|---|---|
| General web | C4 English validation | Calibration M1/M2, fixed probe dan technical checks; causal pilot/main ditunda. |
| Scientific text | peS2o train | Domain primary dan domain-conditioned calibration. |
| Source code | StarCoderData Python train | Domain primary dan domain-conditioned calibration. |

Domain ditentukan dari sumber dataset. Scientific dan code memberikan kontras
distribusi token terhadap general web. Natural text menyediakan target
next-token sebenarnya, sehingga core tidak memerlukan gold QA answers.

Snapshot persiapan OLMoE per 15 Oktober 2026 mencatat 34.591 record di seluruh partisi,
termasuk cadangan dan data yang dikecualikan. Angka berikut adalah ketersediaan
dokumen/kelompok sumber, bukan pengukuran yang sudah selesai.

| Partisi | C4 | Scientific | Code |
|---|---:|---:|---:|
| Verifikasi teknis | 12 / 12 | 12 / 12 | 14 / 12 |
| Kalibrasi tersedia | 500 / 483 | 500 / 500 | 500 / 488 |
| Probe tetap | 32 / 26 | 32 / 32 | 32 / 31 |
| Cadangan pilot | 0 / 0 | 80 / 80 | 80 / 80 |
| Inventory utama | 0 / 0 | 6.542 / 6.542 | 3.171 / 3.092 |

Jumlah kelompok dan dokumen berbeda pada sebagian partisi kode/C4. Alokasi akhir main
dibekukan setelah pilot; C4 tidak memiliki alokasi pilot atau main.

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

Lapisan OLMoE pertama, tengah, dan terakhir adalah 0, 7, dan 15; layer 15 menjadi
primer. Generalisasi BF16 memakai layer 0/11/23 pada Qwen1.5-MoE-A2.7B dan 1/13/26 pada
DeepSeek-V2-Lite (blok 0 DeepSeek padat). Setiap kombinasi model/layer yang dianalisis
memerlukan verifikasi dan kalibrasinya sendiri. Konfigurasi Path B yang dapat dijalankan
saat ini masih layer 15 saja; pengukuran multi-layer dan lintas model belum terlaksana.

Pilot dan main memilih hingga delapan posisi seragam bertingkat per dokumen dengan seed
utama 3007; dokumen dengan kurang dari delapan posisi eligible memakai seluruhnya. Semua
berkas eligible dalam kelompok kode terpilih tetap disertakan. Matched 2048 tetap di
luar rancangan.

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

Gunakan nested budgets 100→250→500 dokumen/domain dan seeds 3008, 3009, 3010. Satu full
window/dokumen/seed mengamati semua input positions. Untuk OLMoE, stabilitas ranking
mensyaratkan drift L1 ≤0,05 dan Jaccard top-8 serta bottom-8 masing-masing ≥0,75 pada
sedikitnya 90% perbandingan yang ditetapkan. Fixed-probe selection agreement dilaporkan
bersama ranking beku. Kalibrasi tanpa ablasi.

Utilization adalah active-set membership frequency, bukan mean routing weight.
Dengan top-8 dan denominator konsisten, jumlah utilization semua expert adalah
delapan. Semua skor dihitung per layer.

### M0 dan M3 — Kontrol

**M0** memakai uniform random ordered pair dari dua distinct active experts,
dengan 64 seeded draws/event. Expected CD nol; finite draws tidak harus nol.
Randomization variability dilaporkan terpisah dari source-sampling uncertainty.

**M3** memakai native effective routing coefficient pada token/layer yang diuji
sebagai kontrol positif diagnostik. Sinyal lemah memicu pemeriksaan mekanisme;
setelah mekanisme terbukti benar, hasil lemah tetap dapat menjadi temuan sah.
Kunci keluaran Path B saat ini masih bernama M4 dan perlu diselaraskan sebelum
pilot.

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

Pilot memakai nested **20→40→80 dokumen scientific dan kelompok kode**, dengan hingga
**delapan posisi/dokumen**, sama dengan main. Semua berkas eligible dalam kelompok kode
terpilih disertakan. Kecukupan dan alokasi dinilai dari proyeksi varians; tidak
digunakan aturan tambahan halfwidth dibagi akar jumlah kelompok.

Cluster variance dan 2000 bootstrap-t draws menentukan precision projection;
measured full-technical runtime memilih qualifying allocation tercepat dari
reviewed inventory. Main freeze juga memerlukan ledger biaya/waktu seluruh
tahap tersisa, evidence per tahap, reconciled remaining limits dan ≥20% reserve.

Main minimal 200 dokumen scientific dan 200 kelompok kode; jumlah final dipilih
melalui aturan pilot yang mempertimbangkan presisi, perlindungan jumlah kelompok,
inventory eligible, dan kelayakan komputasi. Target setengah lebar interval
primer ditetapkan sebagai fraksi eta dari ceiling oracle pilot. Nilai eta masih
placeholder sampai skala ceiling diketahui dan harus dibekukan sebelum hasil
utama. Target presisi tidak membuktikan effect size atau achieved power.

Fallback jika target precision tidak muat budget masih harus diputuskan dengan
pilot evidence sebelum main outcomes.

Sebelum pilot, protokol, rencana analisis, konfigurasi, identitas data dan
checkpoint, pemetaan lapisan, toleransi, alokasi awal, serta kode diberi tag
versi dan dicatat di OSF. Setelah pilot, alokasi utama dibekukan sebagai versi
bertanggal sebelum outcome utama. Exact main IDs, positions, dan analysis
dibekukan sebelum outcome; amendemen dicatat dengan tanggal dan alasan.
Perluasan pilot mengikuti projection rules; main tidak ditambah untuk mengejar
signifikansi atau membuat M2 menang.

## Analisis Statistik

Primary menggunakan paired two-sided **95% bootstrap-t**, conditional pada
kalibrasi yang dibekukan. Resampling mengambil kelompok sumber scientific dan
kelompok kode secara terpisah; semua file dan multiplicities dipertahankan.
Estimasi kode tetap berbobot dokumen, bukan rerata kelompok kode.

Document means dihitung sebelum inference. Domain cluster variance tetap
`G/(G−1) × Σg[Σi∈g(Di−Dbar)]²/N²`; primary variance
`0.25 × (V_scientific + V_code)`. Pilot projection mentransfer domain variance
menurut rasio pilot/main source-group counts pada budget delapan posisi yang sama.

Gunakan 10.000 final replicates dan 2000 planning replicates. CI memakai linear
quantiles; tail probabilities `(count+1)/(valid_draws+1)` sebelum Holm correction.
Laporkan degeneracy dan wajibkan ≥95% valid draws untuk interval. Expert effects,
layers dan repeated forwards tidak menambah independent source observations.

Satu uji primer dua sisi menguji delta agregat OLMoE layer 15. Enam kontras sekunder
Holm mencakup tiga lapisan OLMoE dikalikan dua domain. Estimasi tiap lapisan dilaporkan
dengan interval; pola kedalaman bersifat deskriptif dan tidak menjadi uji formal
interaksi. Generalisasi BF16 dilaporkan terpisah secara eksploratif tanpa koreksi lintas
model.

Untuk setiap kejadian, ceiling oracle adalah rentang efek penghilangan yang diukur,
`max(DeltaL_e) − min(DeltaL_e)` di antara expert aktif. Laporkan CD_M1, CD_M2, M0, dan
ceiling oracle beserta ketidakpastiannya. Delta primer dinormalisasi terhadap ceiling
oracle agregat dengan bobot dokumen dan domain yang sama; rasio dihitung ulang pada
setiap bootstrap resample dan tidak dilaporkan jika ceiling nol. Ceiling ini batas pada
pasangan yang diukur, bukan ranking yang dapat dipakai tanpa mengetahui efek intervensi.
Interval yang melintasi nol menunjukkan ketidakpastian, bukan bukti ekuivalensi.

## Sensitivitas dan Batas Scope

Matched/mismatched domain rankings tetap tersedia dari complete eight-effect
measurements yang sama; reselecting pairs tidak memerlukan ablation tambahan.
C4 tetap menyumbang kalibrasi M1 dan kontrol ranking mismatched, tetapi tidak menjadi
domain outcome kausal. Matched **2048** juga tetap ditunda. Kedalaman OLMoE kini dicakup
melalui lapisan 0/7/15 sesuai rancangan di atas; hasil lintas kedalaman baru tersedia
setelah tiap lapisan diverifikasi dan diukur.

Domain-enrichment score, M5 activation norm, mean ablation, equivalence tests,
B/C operators, layer 4/11, context 1024, equal-token/
token-pooled M1, tambahan domains/models dan Do-style scoring tetap di luar
core. Adopsinya membutuhkan keputusan ilmiah tersendiri dan tidak menggantikan
primary comparison yang sudah dipilih.

## Hipotesis dan Batas Interpretasi

Ekspektasi yang diuji adalah tambahan CD M2 atas M1 pada endpoint OLMoE layer 15 dan
sinyal positif kontrol koefisien routing native. Nama kontrol pada naskah dan kunci
keluaran Path B perlu diselaraskan sebelum pilot. Ekspektasi ini tidak menjadi syarat
hasil penelitian harus positif.

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

Diagnostik terbatas OLMoE FP32 pada 15 Oktober 2026 memakai A40, sembilan
jendela, dan hingga 72 perbandingan dengan ablasi native; selisih campuran,
logits, dan NLL tercatat nol pada kasus yang diperiksa. Ini bukan verifikasi
lengkap per lapisan, estimasi efek, atau hasil ilmiah. Kalibrasi, probe, pilot,
dan pengumpulan utama belum menghasilkan estimasi M2–M1. Tanggal dan anggaran
GPU-jam masih menunggu profil satu kejadian.

CPU/synthetic checks adalah evidence implementasi. Native acceptance pretrained
FP32 GPU, measured production throughput, calibration stability dan achieved
precision untuk implementasi baru belum ditetapkan. Historical findings tidak
menggantikan pengukuran tersebut.
