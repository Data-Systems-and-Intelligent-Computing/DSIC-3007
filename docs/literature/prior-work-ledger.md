# Prior-Work Ledger — Supporting Evidence

**Reviewed for consolidation:** 2026-09-07. The current research claim is in
[README.md](../../README.md#fokus-dan-pertanyaan-penelitian).
This ledger records paper evidence and its limits; it does not maintain a second
current specification. PDF page references mean viewer pages. Versions below
identify the local copies, not an assertion that each is the latest publication.
Local PDF/history links refer to optional Git-ignored material. README is
Git-eligible and summarizes the scientific design; the detailed protocol and
operational guide remain local-only and Git-ignored by Arya's latest choice.

The earlier ledger is preserved in an
exact snapshot (path historis: `../history/2026-09-07-pre-consolidation/prior-work-ledger.md`; tidak lagi di workspace).
The consolidation uses the prior full-paper assessment and source passages
already checked; it does not claim a new exhaustive novelty search.

## L01 — OLMoE

**Reference:** Niklas Muennighoff et al., *OLMoE: Open Mixture-of-Experts Language
Models*. Local copy: arXiv:2409.02060v2, 2025-03-03; original release 2024.
[Local paper](../../references/background/olmoe-2025-open-mixture-of-experts.pdf).

**Literature evidence:** §5.3 Eq. 7 (p. 20) defines domain specialization by the
fraction of domain tokens routed to an expert. Figure 22 (p. 21) compares GitHub,
arXiv, Wikipedia, Books, and C4 routing. §5.4 relates domain and vocabulary
patterns. §2 and §4 explain the architecture and routing choices.

**Our interpretation:** raw M2 and domain heatmaps are established. Routing
frequency and weak co-activation do not by themselves prove functional
irreplaceability or low compensability. The current audit tests a narrower
interventional consequence of those observational patterns.

## L02 — Do et al.

**Reference:** Giang Do, Hung Le, Truyen Tran, *Do Domain-specific Experts exist
in MoE-based LLMs?* Local copy: Findings of ACL 2026, pp. 2324–2340;
ACL Anthology identifier `2026.findings-acl.110`.
[Local paper](../../references/core/do-2026-domain-specific-experts.pdf).

**Literature evidence:** §§3.2–3.3, Eqs. 5–10 (pp. 3–4), use embedding-gradient
salience to partition tokens and combine activation frequency with preference
for salient rather than common tokens. The Eq. 8 expert score is not raw domain
utilization. §3.4 defines DSMoE steering; §4.2 and Figure 1 include base OLMoE
among ten models. §§4.3–4.4 evaluate steering across domain tasks/benchmarks.
Appendix A.3.4 (p. 14), Tables 11–12 (p. 17), also studies a collaborative
expert during GPT-OSS-20B steering: accuracy is 52 for the original model, 56
with DSMoE, and 54 after the described collaborative-expert ablation.

**Our interpretation:** Do already supplies intervention evidence and expert
collaboration analysis, not only observational heatmaps. Its gradient-informed
score, steering manipulation, and task-level outcomes differ from a held-out
paired M2–M1 token-local zero-output audit. Its salient-token partition is not
itself evidence of cross-domain exclusivity. A future Do-style comparator uses
additional information; it cannot be substituted for M2 without changing the
comparison. Neither M2 success nor failure directly settles DSMoE's efficacy.

**Boundary to preserve:** the paper describes weight/logit steering and a
router-score-zero ablation; do not assume either is identical to A or native C.
“Domain-specific experts exist” and “steering a domain expert helps” are already
occupied claims.

## L03 — Engmann et al.

**Reference:** Leonard Engmann, Christian Medeiros Adriano, Holger Giese,
*From Observation to Intervention: A Causal Audit of Expert Importance in
Mixture-of-Experts Models*. Local copy: arXiv:2606.10703v2, 2026-06-16.
[Local paper](../../references/core/engmann-2026-observation-to-intervention.pdf).

**Literature evidence:** §2 (pp. 2–3) ranks active experts by population metrics,
zeros the output of the high/low expert at a single token/layer, and compares
next-token loss effects. It uses WikiText-2 and three MoE models, including base
OLMoE. §3 reports no population-metric cell passing its corrected dual-test
criterion. Table 4 (p. 7) gives the OLMoE layer-15 token-weight control:
Cohen's d = 0.231, 95% CI [0.09, 0.37], t-test p = 0.0013, Wilcoxon p = 0.019.
The stated corrected control threshold is 0.01. §4 acknowledges wide intervals
and distinguishes a token-level audit from deployed pruning pipelines.

**Our interpretation:** A and high/low token-level auditing are not new. The
paper does not directly test domain-conditioned utilization versus a global
baseline. Its null is scoped to the tested setup, not proof that all population
metrics are useless or exactly zero. Do not claim that both control tests pass
the corrected threshold, or that detecting the control proves adequate power
for a smaller M2–M1 contrast. Appendix A.3's redistribution analysis describes
correlation chains; it is not a direct implementation of our optional B.

## L04 — ESI

**Reference:** Anonymous authors, *Quantifying Expert Specialization for
Effective Pruning in Mixture-of-Experts Models*. Local copy identifies itself
as under review at ICLR 2026. An external persistent identifier and current
publication status are not verified in this ledger.
[Local paper](../../references/core/esi-2026-quantifying-expert-specialization.pdf).

**Literature evidence:** §§3.3–4.2 (pp. 4–5) define specialization/pruning and
compare preserving routing with zeroed removed-expert outputs against excluding
experts from routing and recomputing selection/weights. Tables 1–2 report
architecture-dependent results. Appendix C, Table 4 (pp. 12–13), explicitly
compares task-specific calibration against C4 for pruning on HellaSwag, MathQA,
and MedMCQA.

**Our interpretation:** neither domain-conditioned calibration nor the claim
that router handling after removal matters is sufficient novelty. The paper's
subset-pruning, merging, and task-accuracy setting differs from the current
single-event high/low discrimination comparison. Its intervention vocabulary
does not automatically establish equivalence to our exact A/B/C operators.

## L05 — CoR

**Reference:** Wentao Hu et al., *Awakening Dormant Experts: Counterfactual
Routing to Mitigate MoE Hallucinations*. Local copy: arXiv:2604.14246v2,
2026-04-29.
[Local paper](../../references/core/cor-2026-awakening-dormant-experts.pdf).

**Literature evidence:** §3.2.3, Eqs. 5–7 (p. 5), removes an originally active
expert on selected hard tokens, redistributes gating mass, and aggregates
counterfactual loss effects into CEI. §3.3 combines a causal prior with native
routing information and reallocates expert budgets across layers.

**Our interpretation:** individual-expert removal with redistribution and causal
expert scoring are occupied. The calibration population and adaptive routing
policy differ from the current M2–M1 audit. The stated normalization must not be
assumed identical to baseline-selected-mass-preserving B without resolving its
support and scaling. Positive removal damage does not by itself prove uniquely
stored knowledge. CoR's optimized routing is not our native target-exclusion C.

## L06 — Yoon et al.

**Reference:** Youngsik Yoon, Siwei Wang, Wei Chen, Jungseul Ok, *When Are Experts
Misrouted? Counterfactual Routing Analysis in Mixture-of-Experts Language
Models*. Local copy: arXiv:2605.07260v1, 2026-05-08.
[Local paper](../../references/core/yoon-2026-when-are-experts-misrouted.pdf).

**Literature evidence:** §3.1 (p. 3) compares a token's native route with 32
Gumbel-sampled equal-compute alternatives from a 32-expert candidate pool,
using verified self-generated reasoning trajectories. §§3.2–3.3 analyze token
difficulty, layers, domains, and models. Routing optimization is a further
part of the paper.

**Our interpretation:** same-token alternative-route evaluation and improvements
from route changes are established. This is not specifically the native
replacement after excluding one named active expert, nor an M2-versus-M1 audit
under zero-output ablation. Its route normalization and model/checkpoint choices
must not be imported as generic OLMoE implementation facts.

## L07 — Shazeer et al.

**Reference:** Noam Shazeer et al., *Outrageously Large Neural Networks: The
Sparsely-Gated Mixture-of-Experts Layer*. Local copy: arXiv:1701.06538v1,
2017-01-23.
[Local paper](../../references/background/shazeer-2017-sparsely-gated-moe.pdf).

**Literature evidence:** §2, Eqs. 1–5 (pp. 3–4), gives sparse mixtures and noisy
top-k gating; §4 (p. 5) treats load balancing.

**Our interpretation:** foundational sparse routing concepts belong in the
learning path. This architecture's gate normalization cannot substitute for
inspection of the selected OLMoE checkpoint and runtime.

## L08 — Switch Transformers

**Reference:** William Fedus, Barret Zoph, Noam Shazeer, *Switch Transformers:
Scaling to Trillion Parameter Models with Simple and Efficient Sparsity*.
Local copy: JMLR 23 (2022), arXiv:2101.03961v3, 2022-06-16. The filename's
2021 label reflects the earlier preprint, not this copy's publication date.
[Local paper](../../references/background/switch-transformers-2021.pdf).

**Literature evidence:** §2 (pp. 4–8) describes top-1 routing, capacity and
balancing. Appendix B (p. 29) discusses a second-choice routing alternative
for overflow.

**Our interpretation:** sparse routing and fallback routing are established.
Top-1/capacity behavior is not a specification for OLMoE's top-8 intervention.

## L09 — SteerMoE

**Reference:** Mohsen Fayyaz et al., *Steering MoE LLMs via Expert (De)Activation*.
Local copy: ICLR 2026; arXiv:2509.09660v2, 2026-02-12.
[Local paper](../../references/support/steermoe-2026.pdf).

**Literature evidence:** §3 (pp. 4–5) identifies activation differences between
conditions and steers experts; §§4–5 evaluate behavioral effects.

**Our interpretation:** conditional expert statistics plus inference-time
intervention are established. The behavioral selection and steering setup is
not the current source-domain M2–M1 token-local ablation comparison.

## L10 — SAFEx

**Reference:** Zhenglin Lai et al., *SAFEx: Analyzing Vulnerabilities of MoE-Based
LLMs via Stable Safety-critical Expert Identification*. Local copy:
arXiv:2506.17368v2, 2025-10-12.
[Local paper](../../references/support/safex-2025.pdf).

**Literature evidence:** §§2.3–2.5 (pp. 6–8) categorize experts using conditional
activation patterns and intervene through expert masking for safety analysis.

**Our interpretation:** expert categories based on conditional activation and
behavior-changing masking are not new. Safety-focused expert-group intervention
is distinct from the core TA's predictor-validity audit.

## L11 — MASCing

**Reference:** Jona te Lintelo et al., *MASCing: Configurable Mixture-of-Experts
Behavior via Activation Steering Masks*. Local copy: arXiv:2604.27818v1,
2026-04-30.
[Local paper](../../references/support/mascing-2026.pdf).

**Literature evidence:** §§4.1–4.3 (pp. 3–5) use a learned surrogate and steering
masks; §6.5 compares continuous routing-logit steering and forced selection.

**Our interpretation:** behavior control through routing masks and comparisons
of intervention handling already exist. Building another steering method is
not required to answer the current RQ.

## L12 — Rewiring Experts on the Fly

**Reference:** Guinan Su et al., *Rewiring Experts on the Fly: Continuous
Rerouting for Better Online Adaptation in Mixture-of-Expert Models*.
Local copy: arXiv:2510.14853v1, 2025-10-16.
[Local paper](../../references/support/rewiring-experts-on-the-fly-2025.pdf).

**Literature evidence:** §3 (pp. 3–4) learns additive routing-logit changes for
online adaptation; §6 (pp. 7–9) analyzes the method.

**Our interpretation:** inference-time routing adaptation is established, but
optimized logit changes are different from native target exclusion. This is
context and possible future work, not the core experiment.

## L13 — Depth-Aware Sensitivity

**Reference:** Pradeep Kumar Sharma, Shantanu Godbole, Hritvik Shrivastava,
*Depth-Aware Sensitivity Analysis of Mixture-of-Experts Models via
Magnitude-Based Expert Masking*. Local file is labeled 2026; an external
persistent identifier and publication status remain unverified here.
[Local paper](../../references/support/depth-aware-sensitivity-2026.pdf).

**Literature evidence:** §3.3 (pp. 4–5) and §4 (pp. 5–7) describe magnitude-based
expert masking and depth sensitivity on Qwen3.6 code translation; §5.4 (p. 10)
states limitations.

**Our interpretation:** depth-dependent masking effects are prior work, not a
new claim in isolation. Its whole-expert masks and task-level outcomes do not
establish the current paired token-local effect or a universal depth pattern.

## Use and remaining literature work

- Keep predictor, intervention, calibration population, evaluation population,
  aggregation, and outcome distinct in every comparison.
- Preserve the exact local version when citing a numerical result. Do not infer
  publication status from a filename or interpret a paper's conceptual notation
  as proof of the installed runtime's behavior.
- Record externally unresolved bibliographic metadata when verified; do not
  invent identifiers for ESI or the depth-aware paper.
- Refresh the closest-work search before article submission. Update the master
  if new evidence directly answers the full M2–M1 comparison.
- The [advisor audit](../../history/paper-workspace-2026-10-07/papers/advisor/Audit-DSIC3007.pdf) and
  [advisor update](../../history/paper-workspace-2026-10-07/papers/advisor/advisor-update-2026-09-06.md) are design
  provenance, not additional experimental evidence for the hypotheses.
