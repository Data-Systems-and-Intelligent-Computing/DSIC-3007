# Paired collector benchmark, 2026-10-11

The v6 user-run L40S diagnostic passed. All six selected inputs,144 independent
native expert comparisons,12 measured old/new pairs and48 measured raw events
completed.12 warmup raw events are separately accounted. Exact paired NLL, routes
and all192 matched effect values agree; native mixture/logit/NLL max errors are0.
This is diagnostic evidence, not a full-technical receipt or scientific result.

| Context | Layers | Old mean s/input | New mean s/input | Speed ratio old/new | Less elapsed |
|---|---|---:|---:|---:|---:|
| 4096 | 15 | 2.842 | 1.901 | 1.495x | 33.10% |
| 4096 | 15,0,7 | 13.518 | 12.605 | 1.072x | 6.75% |
| 2048 | 15 | 1.741 | 1.170 | 1.488x | 32.80% |
| 2048 | 15,0,7 | 8.166 | 7.630 | 1.070x | 6.56% |

These times include raw event and timing-row fsync, exclude progress fsync and
warmup/native validation. Each row pools three domain observations, one per domain.
Six windows derive from only three documents/groups. No repeated-run interval is
available;4096 always ran old-first and2048 new-first, confounding order/context.

At4096, candidate all-layer measurement averages12.578s: layer0 all-eight replay
6.946s and layer7 replay3.717s account for about85%; layer15 effects only0.019s.
Removing one full forward therefore helps layer15-only much more than all-layer
collection. Context/science/settings and full native cleanup remain unchanged.

Whole attempt:659.65s (10m59.65s), including cached hash verification82.84s, model
hash/load30.29s, warmups42.07s, native checks181.94s and both-arm collection plus
durable persistence149.19s. No model download. About172.84s of the total remains
setup/retrieval/teardown and other overhead beyond the benchmark child/cache stage.
Nominal compute+storage allowance is$0.20339; including the fixed$0.03 assumption
gives$0.23339, under the declared$0.35 bound. This is not an actual invoice.

The failed upload plus successful attempt total790.82 active seconds and nominal
compute+storage$0.24384 (or$0.30384 with both fixed assumptions). Their continuous
wall span is1629.82s (27m9.82s), including about839s outside active attempts. Do
not treat13m11s active time as elapsed wall time or ignore retained-volume billing.
Wallet and original ten-hour accounting boundary remain unverified.

Conditional main4096 illustration: apply these three-domain diagnostic means to
P=38,296/Q=9,677, scheduling28,619 layer15-only plus9,677 all-layer inputs. Old
58.932h becomes48.999h. This is transferred timing arithmetic, not measured main
collection, a universal bound or a complete forecast. Other stages/contexts remain
missing, so full-workflow verdict stays UNKNOWN and full collection is held.

Next local step: guarded unchanged expert-batch reuse profiling/prototype for
expensive downstream replay, with exact native input/shape/row/dependency guards.
Baseline/cleanup cost also needs attention; suffix optimization alone does not
establish ten-hour feasibility. No new paid work or research amendment is launched.

The run-end cleanup receipt confirms owned pod deleted; network volume retained.
Attempt.json still says creating; final result/runner/cleanup records establish
completion. These raw records are preserved, with the stale lifecycle label noted.

Raw identities, calculations and limitations: [JSON](optimization-paired-results-2026-10-11.json).
Independent audit: [review](optimization-paired-results-review-2026-10-11.json).

At the same nominal1.09/h compute plus0.02/h storage allowance, the48.999h
conditional main4096 scenario is about$54.39 before otherstages/reserve. This
is an assumption-based planning signal against both$12 and10h, not a priced
complete workflow or invoice. Independent raw-data review found no blocker.
