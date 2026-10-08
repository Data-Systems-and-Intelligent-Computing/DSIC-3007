# Pengumpulan data penelitian

Collector dipindahkan dari scripts ke src/collectors pada 2026-10-07; tes ada
di tests/collectors. Environment direkonstruksi dari lock pada tanggal yang sama;
enam tes dan kedua module CLI --help lolos tanpa akses corpus/model.
Source, tes, `pyproject.toml`, `uv.lock` dan `.python-version` dipertahankan;
rekonstruksi environment dari root proyek dengan Python 3.12 dan uv:

```sh
uv sync --project src/collectors --locked
```

Ini memasang dependensi kolektor; bukan environment inference yang sudah
diverifikasi. Jangan menjalankan
ulang koleksi yang sudah ada hanya karena environment telah dibersihkan.

`download_ta_data.py` mengumpulkan dokumen unik per domain. Program tidak berjalan
saat di-import. Target pengumpulan yang disepakati adalah **10.000 dokumen unik
per domain** (web, ilmiah, kode; total 30.000), dengan maksimum **10 file per label
repo** untuk kode. Ini ukuran persediaan, bukan ukuran eksperimen utama atau
klaim kecukupan ilmiah. Kedua angka menjadi default CLI dan dapat diubah lewat
`--target-documents` serta `--max-files-per-repo`.
`--output-dir` tetap wajib ditentukan agar lokasi koleksi baru eksplisit.
Lihat opsi tanpa mengunduh data, setelah environment tersedia:

```sh
uv run --project src/collectors python -m src.collectors.download_ta_data --help
```

Profiler tersedia melalui `python -m src.collectors.profile_starcoder_repos`
dalam environment yang sama. Menjalankannya tanpa `--help` membaca sumber
remote; gunakan hanya ketika profiling memang diperlukan.

Contoh dari root proyek untuk koleksi baru ketika pengumpulan diperlukan.
Pilih direktori yang belum memiliki artefak; `data/collection-10k-v1` sudah
berisi koleksi yang dipertahankan dan tidak boleh dijadikan target ulang:

```sh
uv run --project src/collectors python -m src.collectors.download_ta_data --output-dir data/collection-new-run
```

Perintah tersebut mengunduh/membaca data sumber dan memuat tokenizer, bukan model
OLMoE. Default batas pembacaan tetap 100.000 record per domain. Jika dokumen unik
atau variasi repo tidak mencukupi, program gagal tanpa melonggarkan aturan atau
menambah batas otomatis. Domain sebelumnya yang telah selesai tetap tersimpan.

Prosedur: pin revisi sumber/tokenizer, acak urutan shard melalui streaming
Hugging Face, baca paling banyak `--max-source-rows` record yang dihasilkan
iterator, deduplikasi teks persis, acak kandidat dengan seed, lalu pilih jumlah
dokumen target. Untuk kode, batasi setiap label repo yang diketahui dan pisahkan
kelompok repo antara kalibrasi/evaluasi. Metadata salinan tetap disimpan.

`--calibration-ratio` sekarang berarti proporsi **dokumen** (default 0.15), bukan
token. Kelompok repo utuh dapat membuat proporsi aktual melampaui target;
proporsi aktual dokumen dan token dicatat. Jika pool tidak mencukupi, gagal
tanpa menulis file split. Artefak yang sudah ada tidak ditimpa.

Batas dan cakupan:

- Pool terbatas dari shard yang diacak bukan sampel uniform seluruh korpus.
- Seluruh pool kandidat disimpan di RAM; naikkan batas pembacaan dengan hati-hati.
- Batas record tidak membatasi byte jaringan atau read-ahead decoder Parquet.
- Batas file per repo menghitung juga alias pada duplikat yang ditemukan.
- `max_stars_repo_name` hanya label/proxy sumber. Fork dan near-duplicate belum
  dideteksi. Web/ilmiah masih dibagi per dokumen, bukan kelompok situs/penerbit.
- Seed dan revisi membuat prosedur dapat ditelusuri; simpan metadata run aktual.

Tes tanpa network/model/dataset asli, dari root proyek:

```sh
uv run --project src/collectors python -m unittest discover -s tests/collectors -p 'test_*.py' -v
```
