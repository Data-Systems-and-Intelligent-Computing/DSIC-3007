# DEC-011 Validation Evidence

**Executed locally 2026-10-03.** This record distinguishes completed CPU/file
checks from the experiment gates still pending under
[Operational Specification v1](../../EXPERIMENT_PROTOCOL.md).

## Synthetic statistical checks

The [standalone validation harness](validate_dec011.py) uses only generated
arrays and NumPy. It does not read corpus files, import model libraries, load
weights, initialize a GPU or use the network. No packages were installed.

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  python docs/validation/validate_dec011.py --output /tmp/dec011-screen-new.json
```

The output file must not exist: overwrite protection uses exclusive creation.
Reference results use Python 3.14.6, NumPy 2.4.6, seed 3007011, 200 source units
per domain, 2000 independently generated datasets/scenario and 1999 bootstrap
draws/dataset. Four scenarios comprise **8000 datasets**. Draws retain cluster
multiplicity and recompute the ratio SE; true primary mean is zero.

| Scenario | Empirical 95% CI coverage | 95% Wilson interval for coverage | Invalid trials |
|---|---:|---|---:|
| Normal, unequal group sizes, within-group dependence | 94.50% | 93.41–95.42% | 0 |
| Centered lognormal skew | 95.20% | 94.17–96.05% | 0 |
| Informative group sizes with known document-weighted truth | 94.75% | 93.68–95.64% | 0 |
| Finite-variance Student-t(3) heavy tails | 94.35% | 93.25–95.28% | 0 |

All four pass the predeclared 0.93–0.98 screen. The machine-readable
[result artifact](DEC-011-coverage-screen.json) preserves parameters, widths
and intervals. Unequal group sizes are 1–10 in these scenarios; they are not
measurements of retained repo components. The screen tests synthetic source-unit
sampling, not the entire allocation/data/position mechanism.

Six additional checks passed: singleton-cluster reduction to sample-mean SE,
unequal-group document weighting, duplicate-draw-slot variance, location
equivariance, positive-scale equivariance, and rejection of all-zero studentization.
A small-cluster fixture exposed degenerate draws; the harness correctly rejects
them rather than silently dropping them. This supports the prescribed degeneracy
policy and does not establish a universal minimum-cluster theorem.

## Local model identity

[Local artifact hashes](DEC-011-local-artifact-hashes.json) record byte sizes and
SHA-256 for all nine retained JSON/weight files, including the three complete
weight shards. Hashing streamed bytes without loading tensors. It establishes
the identity of local files for later comparison; it does not verify pretrained
forwards, CUDA compatibility or equivalence to an independently verified remote
revision. The model and tokenizer files were unchanged.

Installed package metadata and the installed OLMoE router/MoE source were read
without model loading. The source has native float32 router softmax, top-8
selection and optional normalization controlled by `norm_topk_prob`; runtime
must still verify the loaded implementation/config. The selected software
versions are recorded in the specification; collector versions remain separate.

## What remains unexecuted

Corpus audit/tokenization/allocation, license-use verification, actual eligible
counts, pretrained reconstruction/no-op/locality/cleanup tests, A40 throughput
and memory, calibration stability, pilot effects/variances, pilot-matched coverage,
main sample-size projection, scientific effects and final CI remain pending.
No actual research effect, hardware benchmark or advisor approval is claimed.
