import copy
import json
import sqlite3
import zlib

import numpy as np
import pytest

from scripts.analyze import complete_events, plan_main
from scripts.common import Journal, load_prepared, read_json, sample_positions, window
from scripts.export_results import check, export
from scripts.prepare import audit_groups, partition, prepare


@pytest.fixture
def research():
    return copy.deepcopy(read_json("config/research.json"))


def doc(identity, domain="code", aliases=None, content=None, tokens=10, context=4):
    return {"id": identity, "domain": domain, "aliases": aliases or [identity], "content_sha256": content or identity,
            "file": "raw.jsonl", "line": 1, "tokens": tokens, "context": context}


def test_aliases_through_excluded_records_and_group_disjointness(research):
    rows = [doc("a", aliases=["repoA"], content="same"), doc("b", aliases=["repoB"], content="same"),
            doc("short", aliases=["repoB", "repoC"], tokens=2), doc("c", aliases=["repoC"])]
    audit_groups(rows)
    assert len({r["group"] for r in rows}) == 1
    assert sum(r.get("excluded") == "duplicate_content" for r in rows) == 1
    assert rows[2]["excluded"] == "too_short"


def test_partitions_keep_whole_code_groups(research):
    research["sampling"].update(technical_groups=1, probe_documents=1, pilot_ladder=[2])
    research["calibration"]["ladder"] = [2]
    rows = [doc(f"{domain}{g}-{f}", domain, [f"{domain}{g}"]) for domain in research["domains"] for g in range(10) for f in range(2 if domain == "code" else 1)]
    audit_groups(rows)
    assert not partition(rows, research)
    for group in {r["group"] for r in rows}:
        members = [r for r in rows if r["group"] == group and not r.get("excluded")]
        stages = {r["stage"] for r in members}
        assert stages <= {"calibration", "reserve"} or stages <= {"probe", "reserve"} or len(stages) == 1
        if "main" in stages or "pilot" in stages:
            assert len(stages) == 1


def test_nested_positions_and_exact_real_target():
    document = {"id": "d", "tokens": 30, "offset": 0}
    short = sample_positions(document, 2, 3007, 8)
    long = sample_positions(document, 8, 3007, 8)
    assert long[:2] == short
    assert len(set(long)) == 8
    for predictor in long:
        ids, target = window(document, predictor, np.arange(30), 8)
        np.testing.assert_array_equal(ids, np.arange(predictor - 7, predictor + 1))
        assert target == predictor + 1 and len(ids) == 8
    assert len(sample_positions({**document, "tokens": 10}, 8, 3007, 8)) == 2


def test_prepare_roundtrip_without_legacy_artifacts(tmp_path, research):
    research["context"] = 4
    research["sampling"].update(technical_groups=1, probe_documents=1, pilot_ladder=[1])
    research["calibration"]["ladder"] = [1, 2]
    paths = {}
    for domain in research["domains"]:
        file = tmp_path / f"{domain}.jsonl"
        rows = [{"text": chr(65 + i) * 10, "source_group": f"g{i}"} for i in range(8)]
        file.write_text("".join(json.dumps(r) + "\n" for r in rows))
        paths[domain] = [str(file)]
    tokenizer = type("Tokenizer", (), {"encode": lambda self, text, add_special_tokens: [ord(c) for c in text]})()
    execution = {"input_files": paths, "prepared_path": str(tmp_path / "prepared"), "tokenizer_path": str(tmp_path)}
    info = prepare(research, execution, tokenizer)
    # Same text across domains is conservatively excluded, never split across stages.
    assert info["exclusions"]["cross_domain_duplicate_component"] == 24
    loaded, docs, tokens = load_prepared(execution, research)
    assert loaded["id"] == info["id"] and len(docs) == 24
    assert len(tokens) == 240


def test_journal_resume_atomic_complete_event_and_lossless_routes(tmp_path, research):
    path = tmp_path / "run.sqlite"
    payload = {"active": list(range(8)), "ablated_nll": [1.] * 8, "effects": [.1] * 8, "cleanup_exact": True}
    ids = np.arange(16, dtype="<i4")
    routes = np.arange(128, dtype="u1").reshape(16, 8) % 64
    weights = np.arange(128, dtype="<f4").reshape(16, 8) / 128
    journal = Journal(path, research, {"id": "data"}, "pilot")
    with pytest.raises(ValueError, match="incomplete"):
        journal.append("bad", {**payload, "ablated_nll": [1.] * 7}, ids)
    assert journal.get("bad") is None
    journal.append("good", payload, ids, routes, weights)
    journal.close()
    resumed = Journal(path, research, {"id": "data"}, "pilot")
    assert resumed.get("good") == payload
    resumed.close()
    with sqlite3.connect(path) as db:
        i, r, w = db.execute("SELECT input,routes,weights FROM events").fetchone()
    assert zlib.decompress(i) == ids.tobytes()
    assert zlib.decompress(r) == routes.tobytes()
    assert zlib.decompress(w) == weights.tobytes()
    changed = copy.deepcopy(research)
    changed["seed"] += 1
    with pytest.raises(ValueError, match="Resume"):
        Journal(path, changed, {"id": "data"}, "pilot")


def test_analysis_rejects_other_frozen_calibration_and_incomplete_stage(tmp_path, research):
    path = tmp_path / "pilot.sqlite"
    calibration = {"id": "frozen"}
    journal = Journal(path, research, {"id": "data"}, "pilot", {"calibration": calibration})
    journal.db.execute("INSERT INTO metadata VALUES ('target',?)", (json.dumps({"keys": ["missing"], "units": 20}),))
    journal.db.commit()
    journal.close()
    with pytest.raises(ValueError, match="Incomplete stage"):
        complete_events(path, research, {"id": "data"}, calibration)
    with pytest.raises(ValueError, match="frozen calibration"):
        complete_events(path, research, {"id": "data"}, {"id": "another"})


def test_online_export_and_transfer_corruption(tmp_path, research):
    source, target = tmp_path / "source", tmp_path / "copy"
    source.mkdir()
    journal = Journal(source / "pilot.sqlite", research, {"id": "data"}, "pilot")
    journal.append("one", {"routing": True}, np.arange(8))
    export(source, target)
    check(target)
    assert sqlite3.connect(target / "pilot.sqlite").execute("SELECT count(*) FROM events").fetchone()[0] == 1
    journal.close()
    with (target / "pilot.sqlite").open("ab") as out:
        out.write(b"corrupted")
    with pytest.raises(ValueError, match="differs"):
        check(target)


def test_main_allocation_preserves_groups_precision_and_budget(research):
    documents = [{"id": f"{domain}{g}-{f}", "domain": domain, "group": f"{domain}{g}", "stage": "main", "tokens": 4107}
                 for domain in research["outcome_domains"] for g in range(250) for f in range(2 if domain == "code" else 1)]
    pilot = {d: {"variance": .00003, "groups": 80} for d in research["outcome_domains"]}
    pilot["primary"] = {"critical": 2.}
    benchmark = {"rates": {d: {"effects": {"max": .1}} for d in research["outcome_domains"]}}
    execution = {"remaining_usd": 10., "remaining_wall_hours": 1., "hourly_usd": 1.,
                 "remaining_overhead_seconds": 30., "remaining_offline_seconds": 60., "fixed_remaining_usd": .1, "reserve_fraction": .2}
    plan = plan_main(documents, pilot, {"sufficient": True}, benchmark, {"id": "cal"}, {"id": "data"}, research, execution)
    assert plan["units"] == {"scientific": 200, "code": 200}
    assert plan["projected_halfwidth"] <= research["analysis"]["target_halfwidth"]
    code = [row for row in plan["selection"] if row["domain"] == "code"]
    assert len(code) == 400 and len({row["group"] for row in code}) == 200
    assert all(len(row["predictors"]) == 8 for row in plan["selection"])
    assert plan["remaining_work"]["wall_hours"] > plan["remaining_work"]["total_hours"]
    with pytest.raises(ValueError, match="remaining funds/time"):
        plan_main(documents, pilot, {"sufficient": True}, benchmark, {"id": "cal"}, {"id": "data"}, research, {**execution, "remaining_usd": 0.})
    with pytest.raises(ValueError, match="Pilot is insufficient"):
        plan_main(documents, pilot, {"sufficient": False}, benchmark, {}, {}, research, execution)


def test_resume_rejects_changed_actual_window(tmp_path, research):
    journal = Journal(tmp_path / "events.sqlite", research, {"id": "data"}, "calibration")
    ids = np.arange(8)
    journal.append("event", {"target": 9}, ids)
    assert journal.get("event", ids, 9) == {"target": 9}
    with pytest.raises(ValueError, match="different input/target"):
        journal.get("event", ids + 1, 9)
    with pytest.raises(ValueError, match="different input/target"):
        journal.get("event", ids, 10)
    journal.close()
