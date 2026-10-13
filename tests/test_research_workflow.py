"""Synthetic offline pipeline checks, without pretrained or GPU claims."""

from collections import defaultdict
from copy import deepcopy
from pathlib import Path
import struct

import numpy as np
import pytest

from src.analyze import (_choose_calibration_budget, _bootstrap_domain,
                        analyze_bundle)
from src.artifacts import atomic, canonical, digest, read_json, read_rows, seal, verify_bundle, write_rows
from src.preparation import make_plan, reviewed_documents
from src.workflow import shorten_plan


def test_freeze_uses_larger_budget_and_does_not_pool_away_a_failing_comparison():
    comparisons = []
    for budget in (100,250,500):
        for a,b in ((3008,3009),(3008,3010),(3009,3010)):
            comparisons.extend({"comparison":f"seed:{a}:{b}:budget:{budget}","pass":True} for _ in range(9))
    for a,b in ((100,250),(250,500)):
        comparisons.extend({"comparison":f"budget:{a}:{b}","pass":True} for _ in range(9))
    assert _choose_calibration_budget(comparisons,[100,250,500])[0] == 250
    comparisons[-18]["pass"] = False  # 8/9 fails the required 90% budget agreement.
    assert _choose_calibration_budget(comparisons,[100,250,500])[0] == 500
    comparisons[-1]["pass"] = False
    assert _choose_calibration_budget(comparisons,[100,250,500])[0] is None


def test_degeneracy_produces_json_serializable_status():
    result = _bootstrap_domain([{"source_group":"only","delta":1.}],20,3007,.95)
    assert result["ci"] is None and result["variance"] is None
    assert result["degenerate_replicates"] == 20
    canonical(result)


def test_candidate_calibration_gate_does_not_require_predecessor_seed_stability():
    comparisons=[]
    for budget in (100,250):
        for a,b in ((3008,3009),(3008,3010),(3009,3010)):
            comparisons.extend({"comparison":f"seed:{a}:{b}:budget:{budget}","pass":budget==250}
                               for _ in range(9))
    comparisons.extend({"comparison":"budget:100:250","pass":True} for _ in range(9))
    assert _choose_calibration_budget(comparisons,[100,250])[0]==250
    comparisons[-10]["pass"]=False
    assert _choose_calibration_budget(comparisons,[100,250])[0] is None


def prepared_fixture(tmp_path,group_count=3,file_count=2):
    prepared = tmp_path/"prepared"
    prepared.mkdir()
    documents=[]
    tokens=list(range(4100))
    for domain in ("c4","scientific","code"):
        for group in range(group_count):
            for file in range(file_count):
                identity=f"{domain}-{group}-{file}"
                documents.append({"document_id":identity,"dataset":"synthetic","domain":domain,
                    "revision":"fixture","source_group":f"{domain}-g{group}","token_ids":tokens,
                    "token_ids_sha256":digest(tokens),"eligible_positions":4,"content_sha256":digest(identity)})
    write_rows(prepared/"documents.jsonl",documents)
    atomic(prepared/"model-catalog.json",canonical({"fixture":True}))
    seal(prepared,"prepared-inputs",{"tokenizer_sha256":"fixture"})
    review=tmp_path/"review.json"
    atomic(review,canonical({"status":"approved","prepared_manifest_sha256":__import__("src.artifacts",fromlist=["file_digest"]).file_digest(prepared/"manifest.json"),
        "checks":{name:"pass" for name in ("duplicate_content","article_variants","repository_aliases_and_known_clones","source_group_disjointness")},
        "documents":{d["document_id"]:{"stage":"pilot","source_group":d["source_group"]} for d in documents}}))
    return prepared,review,documents


def test_pilot_document_budgets_and_complete_code_groups(tmp_path):
    prepared,review,_=prepared_fixture(tmp_path)
    settings=read_json(Path("configs/research.json"))
    output=tmp_path/"plan"
    make_plan(prepared,review,output,settings,"pilot",dict.fromkeys(settings["domains"],2))
    docs=list(read_rows(output/"documents.jsonl"))
    assert {d:sum(r["domain"]==d for r in docs) for d in settings["domains"]} == {"c4":2,"scientific":2,"code":4}
    code=defaultdict(list)
    for d in docs:
        if d["domain"]=="code":
            code[d["source_group"]].append(d)
    assert len(code)==2 and all(len(v)==2 for v in code.values())
    assert all(r["context_length"]==4096 and r["target_position"]==r["predictor_position"]+1
               for r in read_rows(output/"inputs.jsonl"))
    reviewed=read_json(review)
    reviewed["checks"]["article_variants"]="pending"
    review.write_bytes(canonical(reviewed))
    with pytest.raises(ValueError,match="audit checks"):
        make_plan(prepared,review,tmp_path/"bad",settings,"pilot",dict.fromkeys(settings["domains"],2))


def test_known_source_aliases_cannot_cross_stages_or_be_separate_clusters(tmp_path):
    prepared, review, docs = prepared_fixture(tmp_path, group_count=3, file_count=1)
    code = [d for d in docs if d["domain"] == "code"]
    code[0]["source_group_aliases"] = [code[1]["source_group"]]
    code[1]["duplicate_source_metadata"] = [{"max_stars_repo_name": code[2]["source_group"]}]
    # Replace the synthetic bundle before resealing it; production bundles stay immutable.
    (prepared / "manifest.json").unlink()
    (prepared / "documents.jsonl").unlink()
    write_rows(prepared / "documents.jsonl", docs)
    seal(prepared, "prepared-inputs", {"tokenizer_sha256": "fixture"})
    r = read_json(review)
    from src.artifacts import file_digest
    r["prepared_manifest_sha256"] = file_digest(prepared / "manifest.json")
    r["documents"][code[1]["document_id"]].update(stage="excluded", reason="fixture bridge")
    r["documents"][code[2]["document_id"]]["stage"] = "main"
    atomic(review, canonical(r))
    with pytest.raises(ValueError, match="known source aliases"):
        reviewed_documents(prepared, review)
    r["documents"][code[2]["document_id"]]["stage"] = "pilot"
    atomic(review, canonical(r))
    with pytest.raises(ValueError, match="known source aliases"):
        reviewed_documents(prepared, review)
    r["documents"][code[2]["document_id"]]["source_group"] = code[0]["source_group"]
    atomic(review, canonical(r))
    _, groups = reviewed_documents(prepared, review)
    assert len(groups["code", "pilot", code[0]["source_group"]]) == 2


def test_context_subset_preserves_whole_groups_and_predictors(tmp_path):
    prepared,review,_=prepared_fixture(tmp_path)
    reviewed=read_json(review)
    for item in reviewed["documents"].values():
        item["stage"]="main"
    review.write_bytes(canonical(reviewed))
    settings=read_json(Path("configs/research.json"))
    make_plan(prepared,review,tmp_path/"main",settings,"main",{"c4":6,"scientific":6,"code":3},positions=64)
    shorten_plan(tmp_path/"main",tmp_path/"short",subset=True)
    parents={r["input_id"]:r for r in read_rows(tmp_path/"main"/"inputs.jsonl")}
    docs=list(read_rows(tmp_path/"short"/"documents.jsonl"))
    assert len(docs)==6  # closest whole group: two of six documents/domain.
    for row in read_rows(tmp_path/"short"/"inputs.jsonl"):
        parent=parents[row["parent_input_id"]]
        assert row["target_id"]==parent["target_id"]
        assert row["predictor_position"]==parent["predictor_position"]
        assert row["input_ids"]==parent["input_ids"][-2048:]
    verify_bundle(tmp_path/"short","experiment-plan")


def effect_bundle(tmp_path,count=20,settings=None,source_meta=None):
    settings=settings or read_json(Path("configs/research.json"))
    root=tmp_path/"run"; plan=root/"plan"; cal=root/"calibration"
    plan.mkdir(parents=True); cal.mkdir()
    meta={"stage":"pilot","window":4096,"units":dict.fromkeys(settings["domains"],count),
          "prepared_manifest_sha256":"pool","source_review_sha256":"review","tokenizer_sha256":"tokenizer"}
    if source_meta:
        meta.update(source_meta)
    inputs=[]; events=[]
    for domain in settings["domains"]:
        for index in range(count):
            row={"input_id":f"{domain}-{index}","document_id":f"{domain}-{index}","dataset":"synthetic",
                "domain":domain,"revision":"fixture","source_group":f"{domain}-g{index}",
                "predictor_position":4095,"window_start":0,"window_end":4096,
                "target_id":9,"context_length":4096,"token_ids_sha256":"tokens",
                "sampling":{"seed":3007},"layers":[15],"input_ids":[1]*4096}
            inputs.append(row)
            value=.001*(index+1)
            baseline=2.
            event={k:v for k,v in row.items() if k not in ("layers","input_ids")}
            event.update(event_id=digest([row["input_id"],15]),layer=15,kind="effects",
                baseline_nll=baseline,routing={"expert_ids":list(range(8)),"weights":[.125]*8,
                    "weights_fp32_le_hex":struct.pack("<8f",*([.125]*8)).hex()},
                effects=[{"expert_id":e,"ablated_nll":baseline+e*value,
                          "delta_nll":(baseline+e*value)-baseline} for e in range(8)])
            events.append(event)
    write_rows(plan/"inputs.jsonl",inputs)
    atomic(plan/"research.json",canonical(settings))
    seal(plan,"experiment-plan",meta)
    a=np.linspace(.05,.2,64); a=a/a.sum()*8
    b=a[::-1].copy(); c=np.full(64,.125)
    tables={"scientific":{"15":a.tolist()},"code":{"15":b.tolist()},"c4":{"15":c.tolist()},
            "M1":{"15":np.mean([c,a,b],axis=0).tolist()}}
    atomic(cal/"frozen-tables.json",canonical({"status":"frozen","context_length":4096,"tables":tables}))
    executor={"runtime":"synthetic_cpu_fixture"}
    seal(cal,"calibration",{"executor":executor,"identities":{"source_pool":"pool","source_review":"review","tokenizer":"tokenizer"}})
    write_rows(root/"events.jsonl",events)
    seal(root,"measurements",{"stage":"pilot","executor":executor,"plan_metadata":meta,
                             "expected_event_ids":[r["event_id"] for r in events]})
    return root,settings


def test_sealed_raw_effect_analysis_is_offline_and_keeps_individual_effects(tmp_path):
    root,settings=effect_bundle(tmp_path)
    result=analyze_bundle(root,tmp_path/"analysis",replicates=200)
    assert result["report"]["inference_role"]=="pilot_planning"
    assert result["report"]["primary_equal_domain"]["ci"] is not None
    assert result["report"]["c4_diagnostic_uncertainty"]["confirmatory_test"] is False
    assert len(result["events"])==60 and all(len(r["delta_nll_by_expert"])==8 for r in result["events"])
    assert all(len(r["m0_pairs"])==64 for r in result["events"])
    assert (tmp_path/"analysis"/"bootstrap-draws.jsonl").stat().st_size>0
    assert read_json(tmp_path/"analysis"/"analysis.json")["secondary_holm"]["domains"]==["scientific","code"]
    changed=deepcopy(settings); changed["sampling"]["master_seed"]=42
    with pytest.raises(ValueError,match="settings differ"):
        analyze_bundle(root,settings=changed,replicates=20)


def test_pilot_runtime_planning_and_main_freeze(tmp_path):
    from src.artifacts import file_digest
    from src.planning import plan_main,freeze_main
    prepared,review,_=prepared_fixture(tmp_path,group_count=205,file_count=1)
    reviewed=read_json(review)
    for item in reviewed["documents"].values():
        item["stage"]="main"
    review.write_bytes(canonical(reviewed))
    settings=read_json(Path("configs/research.json"))
    identities={"prepared_manifest_sha256":file_digest(prepared/"manifest.json"),
                "source_review_sha256":file_digest(review),"tokenizer_sha256":"fixture"}
    pilot,_=effect_bundle(tmp_path,count=80,settings=settings,source_meta=identities)
    # Scale the synthetic effects below the precision target without changing
    # source counts or statistical rules.
    events=list(read_rows(pilot/"events.jsonl"))
    for row in events:
        for effect in row["effects"]:
            effect["ablated_nll"]=row["baseline_nll"]+effect["delta_nll"]*.001
            effect["delta_nll"]=effect["ablated_nll"]-row["baseline_nll"]
    (pilot/"manifest.json").unlink(); (pilot/"events.jsonl").unlink()
    write_rows(pilot/"events.jsonl",events)
    meta=dict(verify_bundle(pilot/"plan","experiment-plan")["metadata"])
    executor={"runtime":"synthetic_cpu_fixture"}
    seal(pilot,"measurements",{"stage":"pilot","executor":executor,"plan_metadata":meta,
         "expected_event_ids":[r["event_id"] for r in events]})
    cal=pilot/"calibration"
    cm=read_json(cal/"manifest.json"); (cal/"manifest.json").unlink()
    cm["metadata"]["identities"]={"source_pool":identities["prepared_manifest_sha256"],
                                 "source_review":identities["source_review_sha256"],"tokenizer":"fixture"}
    seal(cal,"calibration",cm["metadata"])
    # The parent measurement seal includes the calibration manifest too.
    (pilot/"manifest.json").unlink()
    seal(pilot,"measurements",{"stage":"pilot","executor":executor,"plan_metadata":meta,
         "expected_event_ids":[r["event_id"] for r in events]})
    technical=tmp_path/"technical"; technical_plan=technical/"plan"
    technical_plan.mkdir(parents=True)
    inputs=list(read_rows(pilot/"plan"/"inputs.jsonl"))
    for row in inputs:
        row["layers"]=[0,7,15]
    write_rows(technical_plan/"inputs.jsonl",inputs)
    atomic(technical_plan/"research.json",canonical(settings))
    techmeta=dict(meta,stage="technical")
    seal(technical_plan,"experiment-plan",techmeta)
    timed=[]
    for source in events:
        for layer in (0,7,15):
            row=dict(source,layer=layer,event_id=digest([source["input_id"],layer]),
                     shared_baseline_seconds=.1,shared_cleanup_seconds=.1,eight_interventions_seconds=.8)
            timed.append(row)
    write_rows(technical/"events.jsonl",timed)
    seal(technical,"measurements",{"stage":"technical","executor":executor,"plan_metadata":techmeta,
         "expected_event_ids":[r["event_id"] for r in timed]})
    report=plan_main(pilot,technical,cal,prepared,review,settings,tmp_path/"planning")
    assert report["status"]=="qualifying_allocation"
    assert report["selected"]["scientific_documents"]==200
    assert report["selected"]["code_source_groups"]==200
    frozen=freeze_main(tmp_path/"planning",prepared,review,settings,tmp_path/"frozen-main")
    assert frozen["metadata"]["main_freeze"]["settings_sha256"]==digest(settings)
    assert frozen["metadata"]["units"]=={"c4":200,"scientific":200,"code":200}
    shorten_plan(tmp_path/"frozen-main",tmp_path/"short-main",subset=True)
    assert read_json(tmp_path/"short-main"/"main-freeze.json")==read_json(tmp_path/"frozen-main"/"main-freeze.json")
