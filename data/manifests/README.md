# Dataset Manifests

Simpan manifest dataset yang sudah dibekukan sebelum eksperimen utama.

Minimal field:
- `domain`
- `source_dataset`
- `dataset_revision`
- `document_id`
- `split` (`calibration` / `evaluation`)
- `sha256` atau hash lain yang konsisten
- `license`
- `sampling_seed`

Jangan menaruh seluruh teks mentah bila lisensi sumber tidak mengizinkan redistribusi.
