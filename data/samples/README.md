# Frozen Evaluation Samples

Folder ini eligible untuk small frozen position/window manifests, bukan source
text. File CSV yang ada adalah template historis, bukan sampel main aktif.
Exact main positions belum dibekukan; ikuti [protocol](../../EXPERIMENT_PROTOCOL.md).

Minimal field:
- domain
- document_id
- position
- token_id
- next_token_id
- seed

Jumlah positions berasal dari eligible counts dan pilot-selected position budget
(mulai 64/document pada layer 15; first 16 pada layer 0/7). Minimum main adalah
200 scientific documents dan 200 whole code groups, bukan 200 posisi/domain.
