"""Offline scope/admission regressions; these are not pretrained GPU evidence."""

from copy import deepcopy
from pathlib import Path
import shutil
import struct

import pytest

from src.admission import REMAINING_STAGES, admit_remaining
from src.artifacts import atomic, canonical, digest, file_digest, read_json, read_rows, seal, write_rows
from src.preparation import make_plan, rebind_prepared
from src.scope import is_path_b, require_scope, scope_identity, validate_plan
from src.workflow import shorten_plan
from test_research_workflow import prepared_fixture


def settings():
    return read_json("configs/research-path-b-v1.json")


def fixture(tmp_path, stage="pilot", groups=5, files=2):
    prepared, review, docs = prepared_fixture(tmp_path, groups, files)
    (prepared/"manifest.json").unlink()
    (prepared/"documents.jsonl").unlink()
    for doc in docs:
        doc.update(token_ids=list(range(4110)), eligible_positions=14,
                   token_ids_sha256=digest(list(range(4110))))
    write_rows(prepared/"documents.jsonl",docs)
    atomic(prepared/"research.json",canonical(settings()))
    seal(prepared,"prepared-inputs",{"tokenizer_sha256":"fixture"})
    r=read_json(review)
    r["prepared_manifest_sha256"]=file_digest(prepared/"manifest.json")
    r["research_scope"]=scope_identity(settings())
    by_id={d["document_id"]:d for d in docs}
    for key,d in r["documents"].items():
        d["stage"]=stage
        d["domain"]=by_id[key]["domain"]
    atomic(review,canonical(r))
    return prepared,review,docs


def test_path_b_nested_eight_positions_whole_code_groups_and_external_targets(tmp_path):
    prepared,review,_=fixture(tmp_path)
    for n in (2,3):
        make_plan(prepared,review,tmp_path/f"pilot-{n}",settings(),"pilot",
                  {"scientific":n,"code":n})
    low=list(read_rows(tmp_path/"pilot-2/inputs.jsonl"))
    high=list(read_rows(tmp_path/"pilot-3/inputs.jsonl"))
    assert len(low)==48  # 2 science documents + 4 code files, each eight positions.
    assert {r["input_id"] for r in low}<={r["input_id"] for r in high}
    assert {r["domain"] for r in high}=={"scientific","code"}
    assert all(r["layers"]==[15] and r["input_ids"][-1]==r["predictor_position"] and
               r["target_id"]==r["predictor_position"]+1 and r["target_id"] not in r["input_ids"] for r in high)
    docs=list(read_rows(tmp_path/"pilot-3/documents.jsonl"))
    assert sum(d["domain"]=="code" for d in docs)==6
    m=read_json(tmp_path/"pilot-3/manifest.json")["metadata"]
    validate_plan(settings(),m,high)
    with pytest.raises(ValueError,match="position coverage"):
        validate_plan(settings(),m,high[1:])
    with pytest.raises(ValueError,match="domain budgets"):
        validate_plan(settings(),dict(m,units={"c4":3,"code":3,"scientific":3}),high)
    with pytest.raises(ValueError,match="matching versioned"):
        require_scope(settings(),{})
    with pytest.raises(ValueError,match="deferred"):
        shorten_plan(tmp_path/"pilot-3",tmp_path/"short")


def test_scope_rejects_numerical_or_protocol_drift():
    for key,value in (("main_positions_per_document",64),("main_secondary_layers",[0,7]),("main_c4_documents",200)):
        changed=settings(); changed["sampling"][key]=value
        with pytest.raises(ValueError,match="accepted scope"):
            is_path_b(changed)
    changed=settings(); changed["mechanics"]["nll_atol"]=.001
    with pytest.raises(ValueError,match="numerical contract"):
        is_path_b(changed)
    changed=settings(); changed["protocol"]["layers"]=[0,7,15]
    with pytest.raises(ValueError,match="scope policy"):
        is_path_b(changed)
    assert not is_path_b(read_json("configs/research.json"))


def test_rebinding_preserves_bytes_and_requires_new_review(tmp_path):
    prepared,review,_=fixture(tmp_path)
    before=file_digest(prepared/"documents.jsonl")
    parent=file_digest(prepared/"manifest.json")
    successor=tmp_path/"successor"
    result=rebind_prepared(prepared,successor,settings())
    assert file_digest(successor/"documents.jsonl")==before
    assert result["metadata"]["parent_prepared_manifest_sha256"]==parent
    assert file_digest(prepared/"manifest.json")==parent
    with pytest.raises(ValueError,match="exact prepared"):
        make_plan(successor,review,tmp_path/"bad",settings(),"pilot",{"scientific":2,"code":2})


def ledger():
    return {"research_scope":scope_identity(settings()), "identities":{"pilot":"p","technical":"t"},
        "reserve_fraction":.20,
        "remaining_stages":{k:{"gpu_seconds":0,"wall_seconds":0,"usd":0,
            "evidence_status":"bounded","evidence":"fixture explicit allowance"} for k in REMAINING_STAGES},
        "accounting":{"cash_boundary_confirmed":True,"time_boundary_confirmed":True,
            "price_evidence":"live_quote","quote_reference":"synthetic quote",
            "hourly_rate_usd":1,"remaining_budget_usd":2,"remaining_wall_seconds":8000}}


def test_remaining_admission_accounts_non_main_and_does_not_infer_zero():
    l=ledger(); ids=l["identities"]
    assert admit_remaining(settings(),3600,ids,None)["status"]=="unknown"
    assert admit_remaining(settings(),3600,ids,l)["feasibility_pass"]
    expensive=deepcopy(l); expensive["remaining_stages"]["calibration"]["gpu_seconds"]=3600
    assert admit_remaining(settings(),3600,ids,expensive)["status"]=="infeasible"
    for mutation in (lambda v:v["remaining_stages"].pop("storage"),
                     lambda v:v["accounting"].update(remaining_budget_usd=None),
                     lambda v:v.update(research_scope={}),
                     lambda v:v.update(reserve_fraction=0)):
        invalid=deepcopy(l); mutation(invalid)
        assert not admit_remaining(settings(),3600,ids,invalid)["feasibility_pass"]


def test_three_window_preflight_can_never_be_full_technical(tmp_path,monkeypatch):
    from src import model as loader
    from src import optimized_validation as v
    from runpod.path_b_preflight import prepare
    from runpod import screen_remote
    prepared,review,_=fixture(tmp_path,stage="technical",groups=12,files=1)
    plan=tmp_path/"technical"
    make_plan(prepared,review,plan,settings(),"technical",dict.fromkeys(settings()["domains"],12))
    package=tmp_path/"package"
    identity=prepare(package,plan)
    diagnostic=package/"plans/path-b-preflight-v1/4096"
    rows=list(read_rows(diagnostic/"inputs.jsonl"))
    assert len(rows)==3 and {r["domain"] for r in rows}==set(settings()["domains"])
    candidate=tmp_path/"candidate"
    shutil.copytree(package/"plans",candidate/"data/plans")
    (candidate/"configs").mkdir()
    for n in ("research.json","research-path-b-v1.json"):
        shutil.copyfile(Path("configs")/n,candidate/"configs"/n)
    assert screen_remote.validate_plans(identity,candidate)[0]==[candidate/"data/plans/path-b-preflight-v1/4096"]
    class Session:
        def acquire(self,*args):
            return object()
    monkeypatch.setattr(loader,"runtime_identity",lambda:{"fixture":"cpu"})
    monkeypatch.setattr(loader,"implementation_identity",lambda:"fixture")
    def case(model,row,*args,**kwargs):
        cases=[{"layer":15,"expert_id":e,"nll_abs_error":0.,
                "max_logit_abs_error":0.,"max_mixture_abs_error":0.} for e in range(8)]
        for i,c in enumerate(cases,1):
            kwargs["progress"]({"completed_comparisons":i,"latest_case":c})
        return {"cases":cases,"cleanup_pass":True}
    monkeypatch.setattr(v,"_input_case",case)
    monkeypatch.setattr(v,"_memory_metrics",lambda torch:{})
    result=v.validate_optimized([diagnostic],tmp_path/"model",tmp_path/"receipt.json",
        validation_scope="preflight",model_session=Session())
    assert result["status"]=="pass" and result["completed_expert_comparisons"]==24
    assert result["approval"]=="diagnostic_only" and result["research_scope"]==scope_identity(settings())
    with pytest.raises(ValueError,match="native mechanics receipt"):
        v.validate_optimized([diagnostic],tmp_path/"model",tmp_path/"full.json",validation_scope="full_technical")
    with pytest.raises(ValueError,match="benchmark screens"):
        v._validate_full_technical_pair([(diagnostic,read_json(diagnostic/"manifest.json"),settings(),rows)],
                                       file_digest(diagnostic/"source-review.json"))


def test_calibration_family_keeps_three_domains_three_seeds_and_layer15(tmp_path,monkeypatch):
    # Substitute JSONL disk I/O and memoize the identical synthetic routing
    # template. Scope coverage, comparisons, averaging, serialization and freeze
    # run normally; existing routing tests cover individual table arithmetic.
    from src import analyze as a
    values={}
    weights=struct.pack("<8f",*([.125]*8)).hex()
    window={"shape":[4096,8],"coefficient_dtype":"float32","expert_ids":[list(range(8))]*4096,
            "weights_fp32_le_hex":weights*4096}
    executor={"runtime":{"fixture":"cpu"}}
    checked=set()
    original_validate=a.validate_measurement
    def validate_identical_template(row):
        if row["kind"] not in checked:
            original_validate(row)
            checked.add(row["kind"])
    monkeypatch.setattr(a,"validate_measurement",validate_identical_template)
    import numpy as np
    table=np.r_[np.ones(8),np.zeros(56)]
    monkeypatch.setattr(a,"calibration_document_tables",lambda rows:{
        (r["sampling"]["seed"],r["domain"],r["layer"]):table for r in rows})
    def bundle(name,budget,seed,stage):
        root=tmp_path/name; plan=root/"plan"; plan.mkdir(parents=True)
        atomic(plan/"research.json",canonical(settings()))
        meta={"stage":stage,"window":4096,"seed":seed,"units":dict.fromkeys(settings()["domains"],budget),
            "research_scope":scope_identity(settings()),"prepared_manifest_sha256":"pool",
            "source_review_sha256":"review","tokenizer_sha256":"tokenizer"}
        rows=[];inputs=[]
        for domain in settings()["domains"]:
            for i in range(budget):
                for pos in range(4 if stage=="probe" else 1):
                    base={"input_id":f"{stage}-{domain}-{i}-{seed}-{pos}","document_id":f"{stage}-{domain}-{i}",
                        "dataset":"fixture","revision":"fixture","domain":domain,"source_group":f"{stage}-{domain}-{i}",
                        "predictor_position":4095+pos,"target_position":4096+pos,"window_start":pos,"window_end":4096+pos,
                        "context_length":4096,"local_predictor_position":4095,"target_id":2,"eligible_positions":10,
                        "token_ids_sha256":"tokens","sampling":{"seed":seed}}
                    inputs.append(dict(base,layers=[15],input_ids=[1]*4096))
                    event=dict(base,event_id=digest([base["input_id"],15]),layer=15,
                        kind="probe_routing" if stage=="probe" else "calibration_window",baseline_nll=2.,
                        routing={"expert_ids":list(range(8)),"weights":[.125]*8,"weights_fp32_le_hex":weights})
                    if stage=="calibration":
                        event["window_routing"]=window
                    rows.append(event)
        values[str(plan/"inputs.jsonl")]=inputs
        values[str(root/"events.jsonl")]=rows
        seal(plan,"experiment-plan",meta)
        seal(root,"measurements",{"stage":stage,"plan_metadata":meta,"executor":executor,
            "expected_event_ids":[r["event_id"] for r in rows]})
        return root
    monkeypatch.setattr(a,"read_rows",lambda p:iter(values[str(p)]))
    budgets={n:[bundle(f"cal-{n}-{seed}",n,seed,"calibration") for seed in (3008,3009,3010)] for n in (100,250)}
    probe=bundle("probe",32,3007,"probe")
    frozen=a.calibration_freeze(budgets,probe_bundle=probe,output_dir=tmp_path/"frozen")
    assert frozen["status"]=="frozen" and frozen["selected_document_budget"]==250
    assert set(frozen["tables"])=={"M1","scientific","code","c4"}
    assert all(set(t)=={"15"} for t in frozen["tables"].values())
    assert len(frozen["evidence"]["comparisons"])==21
    assert all(c["layer"]==15 for c in frozen["evidence"]["comparisons"])
    require_scope(settings(),frozen["manifest"]["metadata"])


def test_main_planning_requires_full_remaining_budget_and_freezes_eight_positions(tmp_path,monkeypatch):
    from src import planning as p
    prepared,review,docs=fixture(tmp_path,stage="main",groups=292,files=1)
    r=read_json(review)
    for doc in docs:
        index=int(doc["document_id"].split("-")[-2])
        r["documents"][doc["document_id"]]["stage"]=("technical" if index<12 else "pilot" if index<92 else "main")
    atomic(review,canonical(r))
    roots={}
    executor={"runtime":{"fixture":"cpu"}}
    for stage,units in (("pilot",{"scientific":80,"code":80}),("technical",dict.fromkeys(settings()["domains"],12))):
        root=tmp_path/stage; plan=root/"plan"
        manifest=make_plan(prepared,review,plan,settings(),stage,units)
        events=[]
        if stage=="technical":
            for row in read_rows(plan/"inputs.jsonl"):
                event={k:v for k,v in row.items() if k not in ("layers","input_ids")}
                event.update(layer=15,event_id=digest([row["input_id"],15]),kind="effects",baseline_nll=2.,
                    routing={"expert_ids":list(range(8)),"weights":[.125]*8,"weights_fp32_le_hex":struct.pack("<8f",*([.125]*8)).hex()},
                    effects=[{"expert_id":e,"ablated_nll":2.,"delta_nll":0.} for e in range(8)],
                    shared_baseline_seconds=.1,shared_cleanup_seconds=.1,eight_interventions_seconds=.01,input_total_seconds=.22)
                events.append(event)
        write_rows(root/"events.jsonl",events)
        seal(root,"measurements",{"stage":stage,"executor":executor,"plan_metadata":manifest["metadata"],
            "expected_event_ids":[e["event_id"] for e in events]})
        roots[stage]=root
    cal=tmp_path/"calibration"; cal.mkdir(); seal(cal,"calibration",{"fixture":True})
    fake={"report":{"pilot_planning":{"domains":{d:{"extend":False,"insufficient_at_maximum":False,"source_groups":80}
        for d in ("scientific","code")}},"domain_paired_delta":{d:{"variance":1e-10} for d in ("scientific","code")}},
        "documents":[{"domain":d,"document_id":f"{d}-{i}","source_group":f"{d}-{i}","layer":15,"mean_delta":i*1e-6}
        for d in ("scientific","code") for i in range(80)]}
    monkeypatch.setattr(p,"analyze_bundle",lambda *args,**kwargs:fake)
    report=p.plan_main(roots["pilot"],roots["technical"],cal,prepared,review,settings(),tmp_path/"planning-pending")
    assert report["status"]=="precision_qualifies_budget_unresolved"
    assert report["selected"]["scientific_documents"]==report["selected"]["code_source_groups"]==200
    assert report["selected"]["c4_documents"]==0
    with pytest.raises(ValueError,match="qualifying allocation"):
        p.freeze_main(tmp_path/"planning-pending",prepared,review,settings(),tmp_path/"bad-main")
    l=ledger(); l["identities"]={k:report[k] for k in ("pilot_manifest_sha256","technical_manifest_sha256","calibration_manifest_sha256")}
    report=p.plan_main(roots["pilot"],roots["technical"],cal,prepared,review,settings(),tmp_path/"planning",remaining_workflow=l)
    assert report["status"]=="qualifying_allocation"
    frozen=p.freeze_main(tmp_path/"planning",prepared,review,settings(),tmp_path/"main")
    assert frozen["metadata"]["units"]=={"scientific":200,"code":200}
    assert frozen["metadata"]["main_freeze"]["positions_per_document"]==8
    assert frozen["metadata"]["input_count"]==3200


def test_offline_path_b_effect_analysis_labels_absent_outcomes_and_keeps_m1_c4(tmp_path):
    import numpy as np
    from src.analyze import analyze_bundle
    prepared,review,_=fixture(tmp_path,groups=12,files=1)
    root=tmp_path/"effects"; plan=root/"plan"
    m=make_plan(prepared,review,plan,settings(),"pilot",{"scientific":12,"code":12})
    events=[]
    for row in read_rows(plan/"inputs.jsonl"):
        index=int(row["document_id"].split("-")[-2])
        value=.001*(index+1)+.00001*row["sample_order"]
        e={k:v for k,v in row.items() if k not in ("input_ids","layers")}
        e.update(event_id=digest([row["input_id"],15]),layer=15,kind="effects",baseline_nll=2.,
            routing={"expert_ids":list(range(8)),"weights":[.125]*8,"weights_fp32_le_hex":struct.pack("<8f",*([.125]*8)).hex()},
            effects=[{"expert_id":i,"ablated_nll":2.+i*value,"delta_nll":(2.+i*value)-2.} for i in range(8)])
        events.append(e)
    cal=root/"calibration"; cal.mkdir()
    a=np.linspace(.05,.20,64); a=a/a.sum()*8
    b=a[::-1].copy(); c=np.full(64,.125)
    tables={"scientific":{"15":a.tolist()},"code":{"15":b.tolist()},"c4":{"15":c.tolist()},
            "M1":{"15":np.mean([a,b,c],axis=0).tolist()}}
    atomic(cal/"frozen-tables.json",canonical({"status":"frozen","context_length":4096,
           "research_scope":scope_identity(settings()),"tables":tables}))
    executor={"runtime":{"fixture":"cpu"}}
    seal(cal,"calibration",{"executor":executor,"research_scope":scope_identity(settings()),
        "identities":{"source_pool":m["metadata"]["prepared_manifest_sha256"],
            "source_review":m["metadata"]["source_review_sha256"],"tokenizer":"fixture"}})
    write_rows(root/"events.jsonl",events)
    seal(root,"measurements",{"stage":"pilot","executor":executor,"plan_metadata":m["metadata"],
        "expected_event_ids":[e["event_id"] for e in events]})
    result=analyze_bundle(root,tmp_path/"analysis",replicates=100)
    assert result["report"]["c4_diagnostic_uncertainty"]["status"]=="not_collected_deferred_by_path_b"
    assert set(result["report"]["pilot_planning"]["domains"])=={"scientific","code"}
    assert len(result["events"])==192
    assert all(len(e["delta_nll_by_expert"])==8 for e in result["events"])
    assert all(set(e["mismatched_m2_cd"]) >= {"c4"} for e in result["events"])
    assert "results were not collected" in (tmp_path/"analysis/research-results.md").read_text()
