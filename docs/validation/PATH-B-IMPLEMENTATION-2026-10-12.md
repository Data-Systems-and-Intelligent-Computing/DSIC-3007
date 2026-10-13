# Path B — keputusan dan langkah implementasi

Status: scope dipilih Arya2026-10-12; migrasi lokal telah diimplementasikan dan
180 tes lolos. Preflight GPU belum berjalan; live read-only check berhenti karena
stok L40S tidak tersedia di region volume US-MO-1. Ini bukan receipt penerimaan
native atau final scientific result.

Update dokumentasi 2026-10-13: README utama sudah konsisten dengan Path B;
desain lama dipindah ke arsip. Runner sekarang memakai successor immutable
`data/optimization/path-b-preflight-2026-10-13-v2/`, karena README ikut hash
paket. Paket v1, evidence dan settings ilmiah tetap dipertahankan. Lihat
[refresh record](PATH-B-README-REFRESH-2026-10-13.md).

## Keputusan yang sudah jelas

- Path B: original FP32,4096,layer15; scientific/code main dan pilot delapan
  posisi/dokumen; minimal200 scientific documents/200 whole code groups;
  all-eight individual effects dan exact cleanup tetap.
- C4 tetap calibration/probe untuk M1 tiga domain; C4/lower-layer outcomes dan
  matched2k deferred. Technical/calibration/probe scope15/4096 perlu receipt baru.
- Scientific minima, numerical tolerances, frozen raw evidence dan group
  resampling dipertahankan. Main allocation/freeze menunggu calibrated pilot.
- Artikel yang lebih luas adalah tujuan. Extension baru tidak otomatis diterima
  hanya karena muncul dalam draft teori. Existing matched/mismatched dapat reuse.
- Preflight berarti small GPU diagnostic, bukan main. Arya mengizinkan rentang
  $0.10–0.20 dan10–15min; hard maximum yang dicatat$0.20/15min, all phases hingga
  cleanup, dengan deadline lebih pendek jika cash cap mengharuskannya. Tidak ada
  automatic retry atau authorization full collection.

## Urutan kerja yang disepakati (rencana awal; hasil implementasi di bawah)

1. **Migrasi lokal dan versioned dry-run plans.** Materialize settings/policy B
   di package baru; jangan overwrite plans lama. `make_plan` perlu stage-specific
   domains/layers; `plan_main` perlu eight-position pilot, tanpa C4/lower-layer
   costs, serta whole-remaining-workflow budget admission. Expected minimum main
   census audit:200 scientific documents +205 code files/200 groups,3240 windows;
   ini reference check, bukan final allocation atau sample-size guarantee.
2. **Scope-specific gates dan offline analysis.** Update native/optimized receipts,
   collector receipt checks, calibration stability family dan reporting. Jangan
   mengubah `measure_input`/replay arithmetic untuk B. Pastikan receipt lama/scope
   berbeda tidak diam-diam mengizinkan B; absent outcomes dilabeli absent.
3. **Local regressions.** Whole-group preservation, nested sampling, target
   alignment, stage disjointness, receipt mismatch, eight-effect completion,
   failure cleanup, calibration provenance dan compatibility historical workflow.
   Last verified legacy suite:172 pass/18.93s; belum ada tes migrasi B.
4. **Persiapkan command preflight user-run.** Cache-only, prepared image/model,
   explicit case list, durable logs/progress, wall/cash deadline dan cleanup reserve.
   Periksa actual quote/storage/fees. Proposal lama1080s/$0.363 tidak muat cap baru
   dan harus diganti. Di nominal historical$1.11/h,$0.20 memberi648.65s sebelum
   biaya tambahan;15min sendiri cost~$0.2775. Ini arithmetic, bukan live invoice.
   Case count dan phase budgets belum dibekukan; diagnostic tidak menggantikan
   full technical acceptance atau pilot precision.
5. **Full acceptance → calibration/probe → pilot.** Hanya setelah paid envelope
   untuk tahap tersebut disetujui. Freeze rankings berdasarkan stability gates;
   pilot scientific/code20→40→80 units,8 posisi mengukur scientific variance.
6. **Main freeze → collection → analysis.** Pilih allocation dari pilot/remaining
   budget, freeze exact inputs dan claims sebelum main effects. Main tidak
   diperbesar atau dipotong berdasarkan apakah hasilnya signifikan.

## Keputusan yang belum diperlukan untuk pekerjaan lokal awal

- Sisa total study funds dan original ten-hour clock. Jawaban cap diagnostic
  bukan total remaining allowance; jangan reset historical spending/time.
- Jika .005-nat halfwidth tidak feasible: report achieved uncertainty sebagai
  precision-limited, atau stop insufficient. Arya meminta penjelasan bahwa ini
  menyangkut hasil akhir; belum memilih fallback. Bahas dengan pilot evidence
  sebelum main; jangan mengklaim precision-limited otomatis disetujui.
- Prospective M3/M5/TOST/mean-ablation dan paid breadth extension. Prioritaskan
  primary precision; pilih extension sebelum main results, tanpa significance peeking.

## Target source yang ditemukan sebelum migrasi

Historical `src/preparation.py::make_plan` mengunci domains ke `settings.domains`,
layers0/7/15 untuk technical/calibration/probe, serta first16 main positions untuk
lower layers. `src/planning.py::plan_main` memerlukan timings ketiga layer dan
C4 pilot diagnostic. `collect`, native/optimized validation dan calibration
analysis juga harus menerima scope baru melalui versioned identity.

Source awal diverifikasi terhadap graph snippets; git history hanya baseline
lama dan workspace mempunyai banyak uncommitted changes. Tidak ada source edits
dalam langkah pencatatan keputusan awal. Setelah instruksi `proceed`, source
dimigrasikan ke config baru; decision JSON tetap bukan executable settings file.

## Implementasi dan pemeriksaan aktual

Record machine-readable dan lokasi log durable:
[path-b-migration-2026-10-12.json](path-b-migration-2026-10-12.json).

- Config baru `configs/research-path-b-v1.json`; CLI workflow default ke B.
  `configs/research.json` tetap byte-identical sebagai config historical.
- `src/scope.py` mengikat settings, plan dan receipt; pilot/main scientific/code,
  technical/calibration/probe tiga domain, semua layer15/4096. Tidak ada 2k B.
  Receipt scope lama tidak dapat mengesahkan B. Model implementation identity
  kini memasukkan scope policy; receipt native baru tetap diperlukan.
- Prepared successor `data/prepared-inputs/path-b-v1-2026-10-12/` memakai salinan
  tokenized bytes lama tanpa retokenisasi. Review baru approved, tanpa shortfall;
  source groups tetap disjoint. C4 yang tidak dipakai tetap berada di raw inventory.
- Sealed plans baru: technical152 inputs; probe384; calibration100/250 masing-masing
  tiga seed (300/750 inputs per seed); pilot20/40/80 berisi320/640/1288 inputs.
  Minimum main dry-run tepat200 scientific documents +205 code files/200 whole
  code groups,3240 windows/25920 effects. Ini census/design plan, bukan main freeze
  atau scientific observations. Cal500 dapat dibuat hanya jika ladder memerlukannya.
- Planner menerima hanya complete eight-position pilot dan full technical timings,
  tanpa C4/lower-layer cost. Transfer variance tetap proyeksi whole-group dengan
  budget posisi sama, bukan power proof. `plan-main --remaining-workflow FILE`
  memerlukan ledger seluruh tahap tersisa, live quote, cash/time reconciliation
  serta minimal20% reserve sebelum main freeze. Tidak ada fallback precision otomatis.
- Calibration comparison family hanya layer15, tetap tiga domain/seed/ladder
  dan threshold lama. Reporter menandai outcomes deferred sebagai tidak dikoleksi.
  Tes menemukan dan memperbaiki NumPy boolean pada serialization penanda tie;
  metric selection dan aritmetika model tidak berubah.
- 180 tes lolos dalam37.73s, termasuk delapan regression B untuk nested sampling,
  whole-code groups, target outside, scope drift, calibration/probe family,
  diagnostic-only receipt, budget admission, main freeze delapan posisi, dan
  offline analysis/reporting dengan C4 tetap dalam M1 serta absent outcomes jelas.
  Fixture CPU/synthetic bukan bukti numerical fidelity pretrained4k.
- Paket immutable `data/optimization/path-b-preflight-2026-10-12-v1/` memiliki
  tiga exact4k inputs dari tiga domain/groups,24 independent native comparisons,
  warmup satu input, tiga measured collector events/all-eight effects.
  Frozen archive extraction, source/plan hashes dan offline verification PASS.

## Command preflight untuk Arya

```bash
cd /home/arya/School/TA/DSIC-3007
python runpod/path_b_preflight.py --check-live
```

Command itu read-only. Check terakhir pada2026-10-12T01:50:13Z berhenti:
`no current L40S stock in the replacement volume's region`. Tarif Secure L40S
$1.09/h, account pods0; region volume US-MO-1 tidak muncul sebagai available.
Registry/volume reads lolos sampai stock gate; cache-on-volume dan GPU runtime
belum diperiksa ulang. Tidak ada resource create/update/delete atau paid run.

Ketika live check lolos, jalankan **satu attempt** yang sudah diizinkan:

```bash
python runpod/path_b_preflight.py --run
```

`--run` mengulang read-only gate dahulu. Cap cash$0.20/wall900s; allowance tetap
$0.03 + hourly burn konservatif$1.11 memberikan lease551.35s, sudah mencakup
cleanup90s. Global cash deadline mengalahkan jumlah phase caps580s. Allowance
adalah bound perencanaan, bukan invoice/provider hard billing cap. Tidak ada
download weights, retry deployment, perubahan dtype atau numerical tolerance.

Console progress/heartbeat dan log durable ada di `runpod/runs/<attempt>/`:
`console.log`, `attempt.json`, `guard.log`, `result.json`, `cleanup.json`, dan
`results/` berisi native cases/progress, individual events, timing, GPU telemetry
dan runner receipts. `cleanup.json` mengonfirmasi deletion pod attempt sendiri;
volume user dipertahankan. Default tanpa flag memeriksa paket lokal saja.

Full acceptance/calibration/pilot/main belum diotorisasi. Setelah preflight,
gunakan receipt baru untuk merevisi forecast sebelum membahas paid envelope.
