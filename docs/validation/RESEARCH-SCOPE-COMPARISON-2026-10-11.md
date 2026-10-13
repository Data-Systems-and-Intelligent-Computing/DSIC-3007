# Perbandingan scope dan biaya — 2026-10-11

Update2026-10-12: Arya memilih scope Path B. Lihat
[decision](../../configs/path-b-decision-v1.json) dan
[migration plan](PATH-B-IMPLEMENTATION-2026-10-12.md). Script sudah dimigrasikan lokal;
precision fallback dan total remaining budget masih pending. Teks berikut
merekam perbandingan sebelum keputusan tersebut.

Status: bahan diskusi/proposal, bukan amendment atau otorisasi paid work.
Arya ingin paket eksperimen artikel yang lebih luas jika batas biaya/waktu
memungkinkan. Belum ada pilihan scope atau kebijakan precision-limited yang
dibekukan. Source, konfigurasi, frozen plans, dan raw receipts tidak diubah.

## Sumber dan batas evidence

- Authority saat ini: `README.md`, amendment 2026-10-10, dan `configs/research.json`.
- Proposal: `FRESH-FEASIBILITY-FINDINGS-2026-10-11.md` dan JSON companion.
- Masukan teori/literatur: `../draft-teori-dasar-arya.md`; beberapa detail memakai
  desain lama dan keputusan baru masih diusulkan.
- Timing berasal dari diagnostic L40S `dsic-screen-20261011T043533-f31ba85d`,
  enam windows/tiga kelompok. Forecast bukan pengukuran workflow lengkap,
  interval kepercayaan runtime, atau jaminan kecukupan statistik.
- Suite lokal diverifikasi ulang pada continuation: 172 passed in 18.93s,
  menggunakan `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/tmp/dsic3007-py314-test-deps python -m pytest -q`.
  Log: `/tmp/dsic3007-continuation-20261011-pytest.log`. CPU tests bukan GPU acceptance.

## README ke Path B

| Aspek | README | Proposal B | Konsekuensi ilmiah |
|---|---|---|---|
| Model/intervensi | Original FP32 OLMoE; zero satu expert tanpa pengganti/renormalisasi | Sama | Tidak mengganti model atau operator |
| Primary | Equal-domain scientific/code M2-minus-M1, layer15/4096 | Sama | Target expected contrast dipertahankan |
| Posisi main | 64/dokumen | 8/dokumen | Sampling variance bertambah; bukan realized estimate 64-posisi |
| Source units | Minimum 200 scientific documents dan 200 code groups | Sama | Code files dalam grup tetap dipertahankan; bukan sampel independen tambahan |
| Pilot | 2 posisi; scientific/code/C4 | 8 posisi; scientific/code | Lebih mahal, tetapi sesuai measurement design main yang diusulkan |
| Lower layers | 0/7, 16 nested positions | Dihapus dari outcomes/calibration scope | Kehilangan descriptive depth comparison dan validasi scope tersebut |
| C4 | Calibration/probe dan 200 diagnostic main documents | Calibration/probe saja | M1 tetap tiga domain; tanpa causal outcome/generalization C4 |
| Context | Primary4096 + matched2048 sensitivity | Primary4096;2048 deferred | Tidak ada claim context robustness |
| Eight effects | Semua expert aktif | Sama | Kontrol dan reranking pada active set tetap tersedia |
| Inference | Document means; whole-group bootstrap-t/Holm | Sama | Precision .005-nat belum dibuktikan; policy bila tidak tercapai perlu keputusan |
| Acceptance | 0/7/15 pada kedua contexts | Scope baru15/4096 | Threshold tetap; keluarga checks berubah; harus ada receipt baru |

Minimum candidate B: 200 scientific documents/groups; 205 code files dalam
200 code groups; 3,240 predictor windows; 25,920 single-expert effects.
Unit independen tetap source group. Final allocation bergantung pilot.

## Cost ledger

DERIVED/ASSUMED dari audit, rate historis $1.11/jam, termasuk20% reserve.
Rentang low/high mengikuti calibration250/pilot20 versus calibration500/pilot80,
bukan optimistic/pessimistic confidence bounds.

| Komponen | Original | B |
|---|---:|---:|
| Main4096 |48.999h|1.711h|
| Matched2048 outcome collection |7.438h|0|
| Calibration |1.888–3.775h|1.175–2.350h|
| Pilot |.063–.255h|.169–.680h|
| Complete GPU/allocation termasuk reserve |78.43–80.93h|6.05–8.07h|
| Nominal biaya |$87.06–89.83|$6.71–8.96|
| Tambahan offline allowance |.50h|.50h|

Pengurangan forecast total sekitar90–92%, terutama perubahan desain, bukan
speedup executor baru. Original→B tidak boleh disamakan dengan diagnostic
3→2-forward improvement yang sudah diterapkan.
Penghematan per-cut beririsan: lower-layer cut pada original main ~28.77h;
setelah itu C4 cut ~6.65h; scientific/code64→8 positions ~11.87h.
Jangan menjumlahkan C4 full-design cost16.15h dengan lower-layer saving.

## Memperluas bahan artikel

| Tambahan | Pengukuran GPU baru | Nilai dan batas |
|---|---|---|
| Matched/mismatched M2 | Tidak pada input yang sama | Sudah tersedia di analysis; menguji kesesuaian ranking/domain, bukan domain exclusivity |
| M3 enrichment | Tidak jika dibentuk dari frozen calibration dan eight effects | Di luar core; perlu definisi dan keputusan analisis prospectively |
| Full-active-set rank diagnostics | Tidak dengan eight effects yang lengkap | Supplementary; definisikan ties, aggregation, dan source-group uncertainty; bukan8 independent samples |
| Token-weighted calibration sensitivity | Tidak jika retained raw rows lengkap | README mengizinkan exploratory analysis; estimator berubah harus dilabeli |
| M5 activation norm | Bisa berbagi calibration forwards, tetapi recording/reductions baru | Overhead/peak/serialization UNKNOWN; norm tidak bisa dipulihkan dari route IDs/weights saja |
| TOST/SESOI | Tidak | Definisi margin/unit dan family sebelum main; broad CI mungkin tetap inconclusive |
| C4 main200 docs×8 positions |1,600 windows tambahan | Outcome-only .845h;20% reserve1.014h/$1.126. Bukan complete extension quote: pilot, planning/I/O/restart masih perlu accounting |
| Layer15-only matched2048 | Calibration/probe/technical + matched outcomes | Audit1.73–2.44h sebelum reserve/I/O, nominal~$2.30–3.25 dengan20% reserve; B+2k~$9.01–12.21, belum historical spend |
| Mean ablation/lower-layer subset | Ya | Full extension cost UNKNOWN; needs operator/scope validation and frozen representative subset |

Prioritas yang diusulkan: B dengan matched/mismatched dan supplementary diagnostics
dari evidence yang sama; pertimbangkan M3 dan M5 dengan prospective definitions.
Pilih maksimal satu paid breadth extension setelah complete forecast dan precision
admission, tanpa memilih ekstensi berdasarkan signifikansi hasil main.
Path C tetap alternatif radical, ~5.9–7.7 GPUh/$6.5–8.5 berdasarkan assumptions,
tetapi mengubah fixed4096 position population menjadi anchor×offset4081–4096
dan membutuhkan executor/assurance baru. Tidak jelas lebih murah daripada B.

## Rekonsiliasi draft dosen

Posisi novelty sebagai domain-conditioned extension relevan. Paper terdekat
memakai global metrics/token-level ablation; exact replication belum layak
diklaim karena corpus/context/runtime harus dicocokkan.
Sumber primer yang dicek: https://arxiv.org/html/2606.10703v2 (setup dan scope).

Detail yang perlu dikoreksi/disepakati: pooled-token M1 versus equal-domain/
equal-document M1; seed42 versus3007;200 posisi versus source-group minima;
token-percentile bootstrap versus current cluster bootstrap-t; full-tie exclusion
versus retention; BF16 tolerance versus frozen FP32 tolerances; downstream
computation boleh merespons intervensi; primary aggregate versus domain tests;
M3/M5/TOST/Wilcoxon belum otomatis diterima. M4 lemah tidak otomatis membuktikan
bug, dan hasil tidak signifikan tidak membuktikan practical equivalence.

Sisa cash, original ten-hour clock, biaya retained volume, live offer, calibration
serialization, pilot variance dan final sample allocation masih belum direkonsiliasi.
Perbandingan keputusan selanjutnya harus memasukkan claim gain/loss, precision,
independent source breadth, numerical risk, engineering effort, incremental
complete-workflow cost dan reusable raw evidence; jumlah eksperimen saja tidak cukup.
