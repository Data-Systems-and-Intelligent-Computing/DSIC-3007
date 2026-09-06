# DSIC-3007 — Causal Audit of Domain-Conditioned Expert Routing in Mixture-of-Experts

Repository penelitian untuk topik **DSIC-3007**.

Judul kerja skripsi:

> **Validitas Kausal Statistik Routing Spesifik-Domain untuk Mengestimasi Kepentingan Expert pada Mixture-of-Experts**

Judul artikel yang direkomendasikan:

> **Causal Validity of Domain-Conditioned Routing Statistics for Expert Importance in Mixture-of-Experts Language Models**

Model utama:

> **OLMoE-1B-7B-0924 (base model)**

---

## Keputusan Pembimbing

Arah penelitian **Mixture-of-Experts (MoE)** diteruskan, tetapi fokusnya tidak boleh berhenti pada pertanyaan sederhana seperti:

> expert mana yang paling sering aktif?

Statistik routing hanya menunjukkan **asosiasi**. Penelitian ini harus menguji apakah expert yang tampak spesifik terhadap suatu domain secara observasional memang **lebih penting secara kausal** ketika expert tersebut diintervensi pada token dan layer yang sama.

Dengan demikian, inti penelitian adalah:

> **Apakah statistik routing yang dikondisikan pada domain memberikan bukti yang lebih baik tentang causal expert importance dibandingkan statistik routing global pada OLMoE?**

Penelitian ini tidak mengklaim bahwa:
- Mixture-of-Experts adalah konsep baru;
- expert specialization adalah konsep baru;
- expert ablation adalah metode baru;
- routing frequency secara langsung berarti knowledge tersimpan pada expert;
- hasil satu model dapat digeneralisasi ke seluruh MoE;
- zero ablation adalah satu-satunya definisi kausalitas yang mungkin.

Penelitian ini menguji validitas prediktif metrik routing **di bawah intervensi yang didefinisikan secara eksplisit**.

---

## Posisi Penelitian

Rantai logika penelitian:

```text
Mixture-of-Experts
        ↓
Sparse router memilih top-k expert per token
        ↓
Routing statistics menunjukkan pola observasional
        ↓
Beberapa expert tampak lebih sering aktif pada domain tertentu
        ↓
Apakah "sering aktif" berarti "penting"?
        ↓
Lakukan token-level single-expert ablation
        ↓
Ukur perubahan next-token loss
        ↓
Bandingkan kemampuan metric routing
dalam memprediksi causal expert importance
```

Perbedaan utama yang diuji:

```text
Global utilization
        vs
Domain-conditioned utilization
        vs
Domain enrichment
        vs
Per-token routing weight
```

---

## Research Question

### RQ Utama

**RQ1. Apakah utilization yang dihitung per domain memprediksi causal expert importance lebih baik daripada utilization global pada OLMoE?**

RQ1 adalah pusat penelitian. Semua eksperimen utama harus kembali menjawab pertanyaan ini.

### Sub-RQ

**RQ2.** Apakah validitas prediktif tersebut berbeda menurut domain dan kedalaman layer?

**RQ3.** Apakah domain enrichment, yaitu penggunaan expert pada domain target relatif terhadap domain lain, lebih informatif daripada raw domain utilization?

**RQ4.** Apakah per-token routing weight memberikan kontrol positif yang lebih kuat daripada statistik populasi global dan per-domain?

**RQ5.** Apakah expert yang diperingkat tinggi untuk satu domain tetap penting pada domain lain, atau menunjukkan causal selectivity terhadap matched domain?

RQ5 merupakan perluasan artikel. Jangan mengorbankan RQ1 demi menyelesaikan RQ5.

---

## Hipotesis

### H1 — Domain conditioning

Pada domain **scientific** dan **code**, skor domain-conditioned diharapkan menghasilkan causal discrimination yang lebih tinggi daripada global utilization, terutama pada layer akhir.

### H2 — Domain contrast

Keuntungan domain conditioning diperkirakan lebih kecil pada C4 karena C4 adalah general web corpus yang heterogen.

### H3 — Depth

Hubungan antara statistik routing dan causal impact diperkirakan berbeda menurut kedalaman layer.

### H4 — Positive control

Per-token routing weight pada layer akhir harus menunjukkan causal discrimination positif.

Bila H4 gagal, hasil utama **belum boleh diinterpretasikan** sebelum implementation fidelity, statistical power, dan lokasi hook diaudit.

Semua hipotesis boleh ditolak. Hasil nol tetap merupakan hasil penelitian yang sah apabila:
- kontrol positif bekerja;
- confidence interval cukup informatif;
- intervensi tervalidasi;
- dataset tidak leakage;
- protokol telah dibekukan sebelum main result dilihat.

---

## Apa yang Sebenarnya Disebut "Expert Importance"?

Penelitian ini tidak menggunakan activation frequency sebagai definisi final expert importance.

Causal expert importance didefinisikan melalui perubahan next-token loss ketika satu expert aktif diintervensi pada satu posisi token dan satu layer.

Baseline loss:

```text
L_t = -log p(x_{t+1} | x_{<=t})
```

Untuk expert aktif `e`, token `t`, layer `l`:

```text
DeltaL(e,t,l)
=
L_t(do(E_e,t,l = 0))
-
L_t
```

Interpretasi:

```text
DeltaL > 0
→ ablation memperburuk prediksi
→ expert membantu token tersebut

DeltaL ≈ 0
→ expert hampir tidak memberi perubahan lokal

DeltaL < 0
→ ablation justru memperbaiki prediksi lokal
```

Klaim kausal harus selalu dibaca sebagai:

> **causal under the specified token-level zero-ablation intervention**

bukan kausalitas universal terhadap seluruh perilaku model.

---

## Model Utama

Gunakan:

```text
OLMoE-1B-7B-0924
```

Karakteristik yang menjadi acuan eksperimen:

```text
~7B total parameters
~1B active parameters per token
16 MoE layers
64 experts per layer
top-8 routing
tanpa shared expert
```

Gunakan **base model**, bukan instruct model.

Alasan:
- outcome penelitian adalah next-token loss pada natural corpus;
- tidak diperlukan fine-tuning;
- tidak diperlukan training router;
- intervensi dilakukan saat inference;
- scope lebih realistis untuk satu bulan.

Aturan penting:

```text
NO fine-tuning
NO LoRA
NO router training
NO steering method baru
NO pruning recovery training
```

---

## Domain Eksperimen

Penelitian memakai tiga domain utama.

### D0 — General Web

Sumber:

```text
C4 Validation
```

Peran:

```text
generic / heterogeneous control
```

C4 dipertahankan karena sudah ada pada usulan awal, tetapi C4 saja tidak cukup untuk menguji domain specialization.

### D1 — Scientific Text

Sumber yang disarankan:

```text
peS2o
atau
subset arXiv
```

Peran:

```text
domain natural language dengan gaya,
struktur, dan terminologi khusus
```

### D2 — Source Code

Sumber yang disarankan:

```text
StarCoderData subset Python/GitHub
```

Peran:

```text
domain dengan distribusi token
yang sangat berbeda dari prose
```

Mengapa tiga domain ini?

1. Memberikan kontras yang kuat.
2. Domain label berasal dari sumber dataset, bukan anotasi subjektif.
3. Tidak memerlukan gold answer.
4. Tidak memerlukan annotator ahli.
5. Masih realistis untuk satu bulan.
6. Dapat digunakan untuk matched vs mismatched analysis di tahap artikel.

Jangan menambahkan Wikipedia, Books, domain hukum, domain medis, atau bahasa Indonesia sebelum main experiment selesai.

---

## Split Data

Satu kesalahan yang sangat berbahaya adalah menghitung domain statistics dari token yang nanti dipakai untuk causal evaluation.

Karena itu split dilakukan pada **document level**.

```text
source documents
        ↓
deterministic document-level split
        ├── calibration
        │      ↓
        │   hitung M1, M2, M3
        │
        └── evaluation
               ↓
           pilih posisi token
               ↓
           causal ablation
```

Aturan:

```text
calibration document != evaluation document
```

Metrik routing tidak boleh dihitung dari evaluation set.

---

## Ukuran Data Minimum

Per domain:

```text
Calibration:
>= 50,000 token

Evaluation pool:
>= 100 dokumen

Frozen intervention positions:
200 posisi token
```

Total minimum:

```text
3 domain × 200 posisi
=
600 posisi token
```

Layer tidak dihitung sebagai sampel independen baru karena posisi token yang sama digunakan kembali pada layer berbeda.

---

## Filtering Posisi Token

Posisi evaluasi harus:

- bukan padding;
- bukan token pertama;
- memiliki next token valid;
- berasal dari evaluation document;
- bukan whitespace/control-only token;
- mempunyai ranking metric yang tidak seluruhnya tie di antara active experts;
- dipilih memakai seed tetap;
- dibekukan sebelum hasil ablation dilihat.

Output posisi evaluasi disimpan di:

```text
data/samples/
```

---

## Statistik Routing yang Diaudit

### M0 — Random Active Expert

Kontrol negatif.

Pilih expert aktif secara acak dengan seed yang dibekukan.

Tujuan:

```text
metric ilmiah harus mengalahkan pemilihan acak
```

---

### M1 — Global Utilization

```text
U(e) = P(e aktif)
```

Tidak memakai informasi domain.

Peran:

```text
baseline observasional global
```

---

### M2 — Domain Utilization

```text
U_d(e) = P(e aktif | domain=d)
```

Ini adalah metrik utama yang diuji terhadap M1.

Pertanyaan utama:

```text
apakah conditioning pada domain
meningkatkan kemampuan prediksi
terhadap causal impact?
```

---

### M3 — Domain Enrichment

```text
E_d(e)
=
log(
  (U_d(e) + epsilon)
  /
  (U_not_d(e) + epsilon)
)
```

Tujuan:

membedakan antara:
- expert yang sering aktif pada semua domain;
- expert yang benar-benar enriched pada satu domain.

M3 adalah analisis sekunder tetapi sangat disarankan bila pipeline stabil.

---

### M4 — Per-Token Routing Weight

```text
g_e(x_t)
```

Peran:

```text
positive control
```

M4 bukan metric populasi yang dihitung sebelum forward pass.

Ia digunakan untuk memeriksa apakah protokol intervensi memiliki cukup power untuk menangkap sinyal lokal pada token yang sama.

---

## Intervensi Utama

Intervensi:

```text
token-level
single-expert
zero-output ablation
```

Untuk token `t`, layer `l`, expert aktif `e`:

```text
hanya kontribusi expert e
pada token t
di layer l
yang di-zero-kan
```

Tidak boleh:

```text
menghapus expert untuk seluruh sequence
mengubah model weights
melakukan rerouting
memilih expert pengganti
melakukan routing renormalization
mengubah expert lain
```

---

## Causal Discrimination

Untuk metric `m`, pada token dan layer yang sama:

```text
high(m)
=
expert aktif dengan skor metric tertinggi

low(m)
=
expert aktif dengan skor metric terendah
```

Hitung:

```text
CD_m,t,l
=
DeltaL_high(m),t,l
-
DeltaL_low(m),t,l
```

Interpretasi:

```text
CD > 0
→ metric berhasil memberi ranking yang memiliki causal discrimination

CD ≈ 0
→ ranking metric tidak membedakan causal impact

CD < 0
→ ranking metric berlawanan dengan causal effect lokal
```

---

## Endpoint Primer

Endpoint primer harus dibekukan sebelum main experiment.

Pada:

```text
Layer 15
Domain:
- scientific
- code
```

Bandingkan:

```text
DeltaCD
=
CD_M2
-
CD_M1
```

Pertanyaan:

```text
apakah domain utilization
lebih informatif secara kausal
daripada global utilization?
```

C4 digunakan sebagai generic control.

M3, depth profile, matched/mismatched analysis, dan layer tambahan adalah secondary/extension.

---

## Layer

Minimum viable thesis:

```text
Layer 0
Layer 7
Layer 15
```

Tujuan:

```text
awal
tengah
akhir
```

Article-ready:

```text
Layer 0
Layer 4
Layer 7
Layer 11
Layer 15
```

Jangan menjalankan seluruh layer lebih dahulu jika tiga layer minimum belum selesai.

---

## Verification Suite

Sebelum eksperimen ilmiah, implementasi harus lolos seluruh pemeriksaan berikut.

### V0 — No Intervention

Forward pass standar harus sesuai dengan implementasi model resmi.

### V1 — No-op Hook

Hook terpasang tetapi tidak mengubah tensor.

Expected:

```text
loss_noop ≈ loss_baseline
```

### V2 — Random Active Expert

Kontrol negatif.

### V3 — Global Utilization

Baseline observasional.

### V4 — Per-token Routing Weight

Kontrol positif.

### V5 — State Cleanup

Setelah hook dilepas:

```text
loss_after_cleanup ≈ baseline_loss
```

### V6 — Position Specificity

Ablation pada token `t`:

```text
tidak boleh diam-diam mengubah
semua posisi token di sequence
```

Jika salah satu verification gagal, main result belum boleh digunakan untuk kesimpulan ilmiah.

---

## Baseline dan Kontrol

| Kode | Kondisi | Peran |
|---|---|---|
| M0 | Random active expert | kontrol negatif |
| M1 | Global utilization | baseline statistik populasi |
| M2 | Domain utilization | metric utama |
| M3 | Domain enrichment | metric domain sekunder |
| M4 | Per-token routing weight | kontrol positif |
| No-op | Hook tanpa perubahan | implementation control |
| No intervention | Forward standar | baseline loss |
| Cleanup | Hook dilepas | state integrity |

---

## Alur Eksperimen

```text
Dataset tiga domain
C4 | Scientific | Code
        ↓
document-level split
        ↓
Calibration              Evaluation
    ↓                         ↓
routing statistics         pilih 200 posisi/domain
M1 M2 M3                      ↓
    ↓                     freeze positions
freeze metric tables          ↓
        \                    /
         \                  /
          OLMoE frozen base
                  ↓
        token + layer tertentu
                  ↓
          active top-8 experts
                  ↓
         rank menurut metric
         M0 M1 M2 M3 M4
                  ↓
           pilih high & low
                  ↓
      baseline next-token loss
                  ↓
    token-level expert ablation
                  ↓
      DeltaL_high / DeltaL_low
                  ↓
        causal discrimination
                  ↓
        CD_M2 - CD_M1
                  ↓
paired bootstrap + CI + depth/domain analysis
```

---

## Rencana Eksperimen

### E0 — Model Feasibility

Tujuan:
- memastikan GPU cukup;
- memastikan OLMoE dapat diload;
- membekukan model revision;
- membekukan precision;
- mengukur VRAM;
- mengukur runtime per batch/token.

Output:

```text
model revision
environment
VRAM requirement
runtime estimate
```

---

### E1 — Intervention Verification

Implementasikan token-level ablation.

Lakukan:
- baseline forward pass;
- no-op hook;
- state cleanup;
- position specificity;
- loss equivalence;
- small replication.

Satu replikasi sempit dapat memakai:

```text
OLMoE
layer 15
WikiText-2
n ≈ 50
```

Tujuannya bukan kontribusi penelitian, tetapi validasi mekanisme.

---

### E2 — Dataset Freeze

Bangun tiga loader:

```text
C4
Scientific
Code
```

Lakukan:
- deterministic document split;
- license audit;
- manifest;
- hash;
- dataset revision;
- seed freeze.

Jangan melanjutkan ke main experiment bila masih ada document leakage.

---

### E3 — Routing Statistics Calibration

Dari calibration split, hitung:

```text
M1 global utilization
M2 domain utilization
M3 domain enrichment
```

Buat:
- heatmap domain × expert;
- heatmap domain enrichment;
- expert coverage;
- tie-rate report.

Jangan menghitung M1-M3 dari evaluation tokens.

---

### E4 — Evaluation Position Freeze

Dari evaluation pool:

```text
200 posisi / domain
```

Total:

```text
600 posisi
```

Freeze:
- document ID;
- token position;
- token ID;
- next token ID;
- seed.

Jangan mengganti posisi setelah ablation result dilihat.

---

### E5 — Pilot

Gunakan:

```text
n ≈ 25 posisi/domain
layers 0, 7, 15
```

Tujuan:
- mengukur runtime;
- mendeteksi hook bug;
- mengaudit active expert ranking;
- memeriksa M4 positive control;
- mengecek heavy-tail loss.

Pilot hanya boleh mengubah masalah teknis.

Jangan mengubah hipotesis setelah pilot.

---

### E6 — Main Experiment

Jalankan:

```text
Domain:
C4
Scientific
Code

Layer:
0
7
15

Metric:
M0
M1
M2
M4
```

Untuk setiap token/layer/metric:

1. ambil 8 active experts;
2. rank active experts;
3. pilih expert high;
4. pilih expert low;
5. hitung baseline next-token loss;
6. zero-ablate high expert;
7. hitung high ablation loss;
8. cleanup;
9. zero-ablate low expert;
10. hitung low ablation loss;
11. cleanup;
12. hitung `DeltaL_high`;
13. hitung `DeltaL_low`;
14. hitung `CD`;
15. simpan event-level result.

Jangan hanya menyimpan rata-rata.

---

### E7 — Secondary / Article Extension

Setelah E6 selesai:

```text
M3 domain enrichment
layer 4
layer 11
matched vs mismatched domain
token-type/frequency robustness
clustered bootstrap per document
```

Model kedua hanya boleh dikerjakan bila seluruh analisis OLMoE selesai.

---

## Record Minimum per Posisi

Contoh:

```json
{
  "run_id": "...",
  "model_revision": "...",
  "dataset_revision": "...",
  "domain": "scientific",
  "document_id": "...",
  "position": 127,
  "layer": 15,
  "active_experts": [1, 4, 8, 13, 21, 34, 46, 57],
  "routing_weights": [0.21, 0.18, 0.15, 0.13, 0.11, 0.09, 0.07, 0.06],
  "metric": "domain_utilization",
  "high_expert": 13,
  "low_expert": 46,
  "baseline_loss": 2.14,
  "high_ablation_loss": 2.31,
  "low_ablation_loss": 2.18,
  "causal_discrimination": 0.13,
  "seed": 42
}
```

Simpan raw result dalam format JSONL atau Parquet yang tidak kehilangan event-level detail.

---

## Analisis Statistik

### Unit Analisis

Unit utama:

```text
token position
```

Bukan:
- expert;
- layer;
- repeated ablation;
- forward pass.

Delapan active expert pada token yang sama bukan delapan sampel independen.

### Analisis Primer

Hitung:

```text
CD_M1
CD_M2
DeltaCD = CD_M2 - CD_M1
```

pada posisi yang sama.

Laporkan:
- mean;
- median;
- standard deviation;
- paired effect difference;
- bootstrap 95% CI.

### Sensitivity Analysis

Jika distribusi heavy-tailed:

```text
Wilcoxon signed-rank
```

### Multiple Comparisons

Endpoint primer hanya satu keluarga utama.

Analisis:
- domain lain;
- layer lain;
- M3;
- matched/mismatched;

ditandai sebagai sekunder.

Gunakan Holm correction untuk kelompok uji sekunder yang sejenis.

### Hasil Nol

Tidak signifikan tidak otomatis berarti dua metric sama.

Jika ingin mengklaim efek praktis tidak ada:
- tentukan smallest effect size of interest;
- gunakan equivalence reasoning atau confidence interval yang cukup sempit.

---

## Grafik dan Tabel Wajib

1. Heatmap `P(expert active | domain)` untuk layer 0, 7, 15.
2. Heatmap domain enrichment.
3. Forest plot CD + 95% CI untuk M0-M4.
4. Interaction plot `depth × metric`, dipisahkan per domain.
5. Plot primer `CD_M2 - CD_M1` pada scientific dan code.
6. Tabel verification suite.
7. Tabel dataset manifest.
8. Tabel extreme/failure cases.
9. Distribusi `DeltaL`.
10. Tie-rate dan expert coverage per domain/layer.

Heatmap aktivasi **tidak cukup** sebagai hasil utama karena hanya menjawab association.

---

## Failure Analysis

Audit minimal 20 kasus.

Kategori yang perlu dicatat:

1. routing metric tinggi tetapi ablation hampir tidak berdampak;
2. routing metric rendah tetapi ablation berdampak besar;
3. M1 dan M2 memilih expert yang sama;
4. M1 dan M2 memilih expert berbeda tetapi CD hampir sama;
5. M4 positive control gagal;
6. punctuation-dominated tokens;
7. extremely frequent tokens;
8. rare code symbols;
9. long-document context;
10. very short scientific spans;
11. loss outlier;
12. expert tie;
13. hook cleanup mismatch;
14. position-specificity failure;
15. tokenization artifact.

Jangan hanya menyebut failure sebagai "noise". Kaitkan kembali ke:
- domain;
- token type;
- expert ranking;
- routing weight;
- layer;
- causal impact.

---

## Threats to Validity

### Construct Validity

Routing frequency bukan bukti bahwa expert "menyimpan pengetahuan domain".

Penelitian hanya menguji apakah metric routing tersebut memprediksi loss impact di bawah intervensi tertentu.

### Intervention Validity

Zero ablation dapat menghasilkan hidden state yang tidak sepenuhnya natural.

Karena itu:
- gunakan local token-level ablation;
- no-op control;
- position specificity;
- state cleanup.

### Data Leakage

M1/M2/M3 harus dihitung hanya dari calibration split.

### Domain Confounding

Scientific, code, dan C4 berbeda dalam:
- vocabulary;
- punctuation;
- formatting;
- document length;
- token frequency.

Jangan mengklaim seluruh perbedaan disebabkan "domain semantics".

### Model Validity

Satu model:

```text
OLMoE
```

tidak cukup untuk generalisasi seluruh MoE.

### Statistical Validity

Loss impact dapat heavy-tailed.

Token dari dokumen yang sama juga dapat berkorelasi.

Clustered bootstrap per document dapat ditambahkan untuk artikel.

### External Validity

Seluruh domain utama berbahasa Inggris.

Jangan menggeneralisasi ke Bahasa Indonesia tanpa eksperimen tambahan.

---

## Quality Gate — Minimum Skripsi

Penelitian minimal layak bila:

- [ ] OLMoE berhasil diload dan direvisi secara fixed.
- [ ] Intervention hook diverifikasi.
- [ ] No-op hook lolos.
- [ ] State cleanup lolos.
- [ ] Position specificity lolos.
- [ ] Tiga domain memiliki split calibration/evaluation.
- [ ] Manifest dataset dibekukan.
- [ ] 600 posisi evaluation dibekukan.
- [ ] Layer 0, 7, 15 selesai.
- [ ] M0, M1, M2, M4 selesai.
- [ ] Endpoint primer dihitung.
- [ ] Effect size dan 95% CI tersedia.
- [ ] Positive dan negative control berfungsi atau kegagalannya dijelaskan.
- [ ] Minimal 20 extreme/failure cases dianalisis.
- [ ] Satu command mereproduksi minimal satu tabel atau plot utama.
- [ ] Klaim dibatasi pada satu model dan tiga domain.

---

## Quality Gate — Submit Artikel

Sebelum submission:

- [ ] seluruh quality gate skripsi selesai;
- [ ] layer 0, 4, 7, 11, 15 selesai atau ada alasan metodologis kuat;
- [ ] M3 selesai;
- [ ] matched vs mismatched analysis selesai;
- [ ] token frequency/type robustness selesai;
- [ ] clustered bootstrap atau robustness statistik selesai;
- [ ] multiple-comparison plan terdokumentasi;
- [ ] related work terbaru diverifikasi;
- [ ] environment lock tersedia;
- [ ] seed, model revision, dataset revision tersedia;
- [ ] raw results tersedia;
- [ ] pembaca lain dapat menjalankan smoke test dari README;
- [ ] judul dan abstrak tidak melebihkan generalisasi satu model.

---

## Batas Scope Bulan Pertama

Jangan menambahkan:

- Policy-Gated RAG;
- fine-tuning;
- LoRA;
- router training;
- steering method baru;
- hallucination benchmark;
- jailbreak/safety benchmark;
- physical pruning;
- recovery training;
- lebih dari tiga domain utama;
- banyak model;
- dashboard;
- UI;
- agent;
- aplikasi web.

Semua dapat menjadi future work.

---

## Rencana Kerja 1 Bulan

### Minggu 1 — Feasibility dan Verification

Hari 1–2:
- akses GPU;
- freeze model revision;
- freeze precision;
- load OLMoE;
- baseline forward;
- catat VRAM dan runtime.

Hari 3–4:
- implement token-level ablation;
- no-op test;
- state cleanup;
- position specificity;
- replikasi kecil.

Hari 5–7:
- loader C4;
- loader scientific;
- loader code;
- document split;
- manifest v1;
- hash dan lisensi.

Gate minggu 1:

```text
model load OK
baseline loss OK
no-op OK
position specificity OK
cleanup OK
runtime feasible
```

### Minggu 2 — Routing Statistics dan Pilot

Hari 8–10:
- M1;
- M2;
- M3;
- heatmap;
- coverage;
- tie rate.

Hari 11–12:
- sample 200 posisi/domain;
- freeze evaluation list;
- pilot 25 posisi.

Hari 13–14:
- perbaiki bug teknis;
- freeze protocol;
- freeze analysis plan.

Gate minggu 2:

```text
no leakage
metric coverage cukup
positive control masuk akal
evaluation positions frozen
```

### Minggu 3 — Main Experiment

Hari 15–18:
- M0;
- M1;
- M2;
- M4;
- layer 0,7,15;
- C4, scientific, code.

Hari 19–21:
- M3;
- layer 4/11 bila compute cukup;
- checkpoint;
- hanya rerun technical failure.

Output:

```text
raw token-level causal impact table
```

### Minggu 4 — Statistik dan Penulisan

Hari 22–24:
- endpoint primer;
- bootstrap;
- CI;
- Holm correction;
- domain × depth plots.

Hari 25–26:
- 20 failure cases;
- token type/frequency checks.

Hari 27–28:
- Method;
- Results;
- Threats to Validity;
- artifact documentation.

Hari 29–30:
- fresh reproduction;
- freeze code/config/manifest/result;
- draft artikel v0.8;
- materi sidang.

---

## Struktur Repository

```text
DSIC-3007/
└── moe-domain-causal-audit/
    ├── README.md
    ├── pyproject.toml
    ├── .gitignore
    │
    ├── configs/
    │   ├── model.yaml
    │   ├── data.yaml
    │   └── experiment.yaml
    │
    ├── data/
    │   ├── manifests/
    │   │   ├── README.md
    │   │   └── manifest_template.csv
    │   └── samples/
    │       ├── README.md
    │       └── evaluation_positions_template.csv
    │
    ├── src/
    │   ├── __init__.py
    │   ├── data.py
    │   ├── routing_stats.py
    │   ├── interventions.py
    │   ├── verify.py
    │   ├── run_experiment.py
    │   └── analyze.py
    │
    ├── tests/
    │   ├── test_noop_hook.py
    │   ├── test_state_cleanup.py
    │   ├── test_position_specificity.py
    │   └── test_loss_equivalence.py
    │
    ├── results/
    │   ├── raw/
    │   ├── processed/
    │   └── figures/
    │
    └── paper/
        └── manuscript.md
```

---

## Setup Minimum

Buat environment:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Jalankan unit test:

```bash
pytest -q
```

Jangan menjalankan main experiment sebelum seluruh verification test berbasis model nyata selesai.

---

## Tools Penelitian

### Model / Deep Learning

```text
PyTorch
Transformers
Accelerate
Hugging Face Datasets
```

### Data

```text
datasets
pandas
NumPy
```

### Statistik

```text
SciPy
bootstrap custom / documented
scikit-learn bila diperlukan
```

### Visualisasi

```text
matplotlib
```

### Reproducibility

```text
Git
YAML config
seed
exact model revision
dataset revision
hash
JSONL/Parquet raw results
```

### Compute

GPU diperlukan karena OLMoE tetap merupakan model besar meskipun hanya sekitar 1B parameter aktif per token.

Sebelum hari ke-2 selesai, lakukan **compute gate**.

Jika model tidak dapat dijalankan dengan stabil, jangan menunggu minggu ke-2 untuk mengetahuinya.

---

## Catatan Pembimbing

Alur besarnya kira-kira seperti ini:

```text
C4 | Scientific | Code
        ↓
document-level split
        ↓
Calibration                   Evaluation
    ↓                             ↓
hitung routing stats          pilih 200 posisi/domain
M1 | M2 | M3                      ↓
    ↓                         freeze positions
freeze metric table               ↓
        \                         /
         \                       /
          OLMoE frozen base model
                  ↓
          token t pada layer l
                  ↓
          router memilih top-8
                  ↓
    rank expert dengan metric yang sama
      M0 | M1 | M2 | M3 | M4
                  ↓
       expert skor tinggi & rendah
                  ↓
       baseline next-token loss
                  ↓
        zero expert HIGH hanya
        pada token t, layer l
                  ↓
             DeltaL_high
                  ↓
           cleanup state
                  ↓
        zero expert LOW hanya
        pada token t, layer l
                  ↓
             DeltaL_low
                  ↓
    CD = DeltaL_high - DeltaL_low
                  ↓
       bandingkan CD_M2 vs CD_M1
                  ↓
   domain × layer × metric analysis
                  ↓
 paired bootstrap + 95% CI + failure audit
```

Secara eksperimen dapat dipahami seperti berikut.

1. **Mulai dari model yang sudah ada, bukan melatih MoE baru.** Gunakan OLMoE-1B-7B-0924 base dan bekukan revision, precision, tokenizer, dan environment. Tujuan minggu pertama adalah memastikan model dapat melakukan forward pass stabil dan routing information dapat diambil. Penelitian ini tidak membutuhkan fine-tuning karena outcome utamanya adalah perubahan next-token loss ketika expert yang sudah aktif diintervensi.

2. **Pisahkan data menjadi tiga domain yang benar-benar berbeda.** C4 berfungsi sebagai general-web control, scientific text menjadi domain natural language yang lebih khusus, sedangkan source code menjadi domain dengan distribusi token paling berbeda dari prose. Label domain berasal dari sumber dataset, bukan dari klasifikasi LLM. Ini penting karena kita tidak ingin membuat domain label menjadi variabel tambahan yang tidak pasti.

   Bentuk sederhananya:

```text
D0 → C4 Validation
D1 → Scientific text
D2 → Source code
```

3. **Setiap domain dibagi lagi pada level dokumen menjadi calibration dan evaluation.** Calibration hanya digunakan untuk menghitung statistik routing seperti global utilization, domain utilization, dan domain enrichment. Evaluation digunakan untuk causal ablation. Dokumen yang sama tidak boleh muncul di kedua bagian.

```text
source documents
   ├── calibration → hitung M1/M2/M3
   └── evaluation  → causal audit
```

   Kalau metrik domain dihitung dari token yang sama dengan token yang nanti diuji melalui ablation, hasilnya bisa terlalu optimistis karena terjadi leakage.

4. **Pada calibration split, kita belum melakukan intervensi.** Kita hanya mengamati router. Untuk setiap expert dihitung seberapa sering ia aktif secara global dan seberapa sering ia aktif pada domain tertentu.

```text
M1 = P(expert aktif)
M2 = P(expert aktif | domain)
M3 = enrichment terhadap domain lain
```

   Dari tahap ini kita bisa membuat heatmap, tetapi heatmap bukan hasil utama. Heatmap hanya menunjukkan asosiasi.

5. **Dari evaluation split, pilih 200 posisi token per domain dan bekukan posisinya sebelum ablation.** Total minimum adalah 600 posisi token. Posisi token yang sama digunakan pada layer 0, 7, dan 15 agar perbandingan antar-depth tetap paired. Jangan mengganti token setelah melihat hasil intervensi.

6. **Untuk setiap posisi token, lihat delapan expert yang benar-benar aktif.** OLMoE memakai top-8 routing. Kita tidak membandingkan semua 64 expert secara sembarang. Kita hanya membandingkan expert yang memang dipilih router untuk token tersebut. Lalu active experts itu diranking menurut M0, M1, M2, M3, dan M4.

   Contoh:

```text
active experts:
[1, 4, 8, 13, 21, 34, 46, 57]

M2 mengatakan:
expert 13 = paling tinggi
expert 46 = paling rendah
```

7. **Baru di sini eksperimen kausal dilakukan.** Hitung baseline next-token loss. Kemudian zero-kan kontribusi expert 13 hanya pada token dan layer yang sedang diuji. Ukur loss baru. Bersihkan hook. Lalu lakukan hal yang sama untuk expert 46.

```text
baseline loss = 2.14

ablate high expert
loss = 2.31
DeltaL_high = 0.17

ablate low expert
loss = 2.18
DeltaL_low = 0.04

CD_M2 = 0.17 - 0.04 = 0.13
```

   Bila `CD_M2 > 0`, berarti ranking domain-utilization pada token itu memiliki causal discrimination: expert yang diranking lebih tinggi memang lebih merusak prediksi ketika dihilangkan daripada expert yang diranking rendah.

8. **Pertanyaan utamanya bukan apakah M2 punya CD positif, tetapi apakah M2 lebih baik daripada M1.** Karena itu endpoint primer adalah:

```text
DeltaCD = CD_M2 - CD_M1
```

   terutama pada scientific dan code di layer 15.

   Jika positif secara konsisten dengan confidence interval yang mendukung, domain conditioning memberikan informasi kausal tambahan dibandingkan utilization global.

9. **M4 digunakan sebagai positive control.** Per-token routing weight sudah mengetahui kondisi token saat forward pass. Karena itu ia seharusnya lebih dekat ke causal effect lokal dibandingkan statistik populasi. Kalau M4 sama sekali tidak punya causal discrimination, jangan langsung menyimpulkan bahwa semua metric gagal. Bisa jadi hook salah, posisi token salah, loss salah, atau power eksperimen terlalu kecil.

10. **Depth menjadi faktor penting, tetapi bukan alasan memperbesar eksperimen tanpa batas.** Minimum layer 0, 7, dan 15 sudah cukup untuk mewakili awal, tengah, dan akhir. Kalau compute mencukupi, baru tambahkan layer 4 dan 11.

```text
awal   → layer 0
tengah → layer 7
akhir  → layer 15
```

11. **Hasil negatif tetap dapat menjadi hasil artikel.** Misalnya domain utilization memang menunjukkan heatmap yang sangat berbeda antara code dan scientific, tetapi ketika expert diablasi, `CD_M2` tidak lebih baik daripada `CD_M1`. Artinya, domain association tidak otomatis berarti causal importance. Ini justru temuan metodologis yang kuat jika kontrol positif bekerja.

12. **Failure analysis wajib.** Ambil setidaknya 20 token ekstrem. Periksa apakah kegagalan terkait punctuation, token sangat sering, simbol kode, panjang dokumen, expert tie, atau outlier loss. Jangan menyebut semua hal sebagai noise tanpa audit.

Untuk tools, **PyTorch + Transformers** adalah inti instrumentasi model. `datasets` dipakai untuk streaming/subset data, `pandas` dan `NumPy` untuk log serta analisis, `SciPy` untuk statistik, dan `matplotlib` untuk visualisasi. Tidak perlu membangun aplikasi, dashboard, RAG pipeline, atau sistem produksi. Yang paling mahal di penelitian ini justru **GPU inference + repeated ablation**, bukan data engineering.

Yang paling penting: penelitian ini bukan pertanyaan **“expert mana yang aktif pada domain code?”**, tetapi pertanyaan **“apakah expert yang tampak penting dari statistik routing domain benar-benar lebih penting ketika kita melakukan intervensi pada token dan layer yang sama, dan apakah metric domain itu lebih informatif daripada metric global?”** Itulah inti eksperimen DSIC-3007.
