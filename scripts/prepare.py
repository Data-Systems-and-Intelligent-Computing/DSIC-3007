"""Tokenize retained JSONL once, audit recorded source relations, and partition groups."""
from __future__ import annotations

from collections import Counter, defaultdict
import glob
import hashlib
import json
from pathlib import Path
from urllib.parse import urlparse

import numpy as np

from scripts.common import ROOT, digest, parser, seed_for, settings, write_json


def source_names(domain, row):
    """All recorded aliases participate, even on duplicate/ineligible records."""
    names = set(row.get("source_group_aliases", []) + row.get("group_aliases", []))
    if row.get("source_group"):
        names.add(row["source_group"])
    metadata = [row.get("source_metadata", row)] + row.get("duplicate_source_metadata", [])
    for meta in metadata:
        if domain == "c4":
            host = urlparse(meta.get("url", "")).hostname
            if host:
                names.add(host.lower().removeprefix("www."))
        elif domain == "code":
            for key in ("max_stars_repo_name", "repo_name", "repository_name"):
                if meta.get(key):
                    names.add(str(meta[key]).strip().lower())
            names.update(str(x).lower() for x in meta.get("repository_aliases", []))
        elif meta.get("id") is not None:
            names.add(str(meta["id"]))
            # Explicit variant identifiers, not arbitrary referenced-paper links.
            names.update(str(x) for x in meta.get("paper_aliases", []))
    return sorted(str(x).strip().lower() for x in names if str(x).strip())


def audit_groups(rows):
    parent = {}

    def find(x):
        parent.setdefault(x, x)
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    def join(a, b):
        a, b = find(a), find(b)
        parent[max(a, b)] = min(a, b)

    content = {}
    for row in rows:
        labels = [row["domain"] + ":" + name for name in row["aliases"]]
        if not labels:
            row["excluded"] = "missing_source_identity"
            continue
        for label in labels:
            join(labels[0], label)
        key = row["content_sha256"]
        if key in content:
            join(labels[0], content[key])
        content[key] = labels[0]
    for row in rows:
        if row["aliases"]:
            row["group"] = find(row["domain"] + ":" + row["aliases"][0])
    cross_domain = defaultdict(set)
    for row in rows:
        if row.get("group"):
            cross_domain[row["group"]].add(row["domain"])
    seen = set()
    for row in sorted(rows, key=lambda r: (r["id"], r["file"], r["line"])):
        if row.get("excluded"):
            continue
        if len(cross_domain[row["group"]]) > 1:
            row["excluded"] = "cross_domain_duplicate_component"
        elif row["tokens"] <= row["context"]:
            row["excluded"] = "too_short"
        elif (row["domain"], row["content_sha256"]) in seen:
            row["excluded"] = "duplicate_content"
        else:
            seen.add((row["domain"], row["content_sha256"]))
    # One eligible version per scientific paper component. Code retains all files.
    scientific = defaultdict(list)
    for row in rows:
        if row["domain"] == "scientific" and not row.get("excluded"):
            scientific[row["group"]].append(row)
    for variants in scientific.values():
        for row in sorted(variants, key=lambda r: r["id"])[1:]:
            row["excluded"] = "recorded_paper_variant"
    return rows


def partition(rows, research):
    """Reserve groups for each stage. Code pilot/main always retain whole groups."""
    for row in rows:
        row["stage"] = "excluded" if row.get("excluded") else "unassigned"
    shortages = {}
    for domain in research["domains"]:
        groups = defaultdict(list)
        for row in rows:
            if row["domain"] == domain and not row.get("excluded"):
                groups[row["group"]].append(row)
        order = sorted(groups, key=lambda g: seed_for(research["seed"], domain, g, "partition"))
        requests = [("technical", research["sampling"]["technical_groups"], "groups"),
                    ("probe", research["sampling"]["probe_documents"], "documents"),
                    ("calibration", max(research["calibration"]["ladder"]), "documents")]
        if domain in research["outcome_domains"]:
            requests.append(("pilot", max(research["sampling"]["pilot_ladder"]), "groups"))
        cursor = 0
        for stage, needed, unit in requests:
            selected = []
            count = 0
            while cursor < len(order) and count < needed:
                group = order[cursor]
                cursor += 1
                members = sorted(groups[group], key=lambda r: seed_for(research["seed"], r["id"], stage))
                selected.extend(members)
                count += 1 if unit == "groups" else len(members)
            if count < needed:
                shortages[f"{domain}/{stage}"] = {"available": count, "needed": needed, "unit": unit}
            for index, row in enumerate(selected):
                # Unused files in a calibration/probe group cannot leak into outcomes.
                row["stage"] = stage if unit == "groups" or index < needed else "reserve"
                row["stage_order"] = index
        for group in order[cursor:]:
            for row in groups[group]:
                row["stage"] = "main" if domain in research["outcome_domains"] else "reserve"
    return shortages


def prepare(research, execution, tokenizer):
    destination = Path(execution["prepared_path"])
    if destination.exists():
        raise ValueError(f"{destination} already exists; choose a new prepared_path")
    temporary = destination.with_name(destination.name + ".preparing")
    if temporary.exists():
        raise ValueError(f"Unfinished preparation at {temporary}; inspect it before removing/restarting")
    temporary.mkdir(parents=True)
    rows, sources, offset = [], [], 0
    with (temporary / "tokens.bin").open("wb") as token_file:
        for domain, patterns in execution["input_files"].items():
            files = sorted({p for pattern in patterns for p in glob.glob(str(ROOT / pattern))})
            if not files:
                raise ValueError(f"No raw inputs found for {domain}")
            for filename in files:
                checksum = hashlib.sha256()
                with Path(filename).open("rb") as stream:
                    for line_number, raw in enumerate(stream, 1):
                        checksum.update(raw)
                        row = json.loads(raw)
                        text = row["text"]
                        content_hash = hashlib.sha256(text.encode()).hexdigest()
                        if row.get("content_sha256", content_hash) != content_hash:
                            raise ValueError(f"Content checksum mismatch: {filename}:{line_number}")
                        ids = np.asarray(tokenizer.encode(text, add_special_tokens=False), dtype="<i4")
                        ids.tofile(token_file)
                        source_path = Path(filename).relative_to(ROOT) if Path(filename).is_relative_to(ROOT) else Path(filename)
                        rows.append({"id": digest([domain, content_hash]), "domain": domain,
                                     "content_sha256": content_hash, "aliases": source_names(domain, row),
                                     "file": str(source_path), "line": line_number,
                                     "offset": offset, "tokens": len(ids), "context": research["context"]})
                        offset += len(ids)
                        if len(rows) % 1000 == 0:
                            print(f"Prepared {len(rows)} documents / {offset} tokens", flush=True)
                source_path = Path(filename).relative_to(ROOT) if Path(filename).is_relative_to(ROOT) else Path(filename)
                sources.append({"path": str(source_path), "sha256": checksum.hexdigest()})
    audit_groups(rows)
    shortages = partition(rows, research)
    with (temporary / "documents.jsonl").open("w") as out:
        for row in rows:
            out.write(json.dumps(row, sort_keys=True) + "\n")
    with (temporary / "tokens.bin").open("rb") as stream:
        token_hash = hashlib.file_digest(stream, "sha256").hexdigest()
    tokenizer_files = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                       for p in Path(execution["tokenizer_path"]).glob("*.json")}
    info = {"research": research, "sources": sources, "tokenizer_files": tokenizer_files,
            "tokens_sha256": token_hash, "documents_id": digest(rows),
            "counts": dict(Counter(f"{r['domain']}/{r['stage']}" for r in rows)),
            "exclusions": dict(Counter(r["excluded"] for r in rows if r.get("excluded"))),
            "shortages": shortages,
            "audit_scope": "Exact content and recorded paper/repository/hostname aliases; unknown clones and unrecorded variants are not certified independent"}
    info["id"] = digest(info)
    write_json(temporary / "prepared.json", info)
    temporary.rename(destination)
    return info


def main():
    args = parser(__doc__).parse_args()
    research, execution = settings(args)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(execution["tokenizer_path"], local_files_only=True)
    info = prepare(research, execution, tokenizer)
    print(json.dumps({"counts": info["counts"], "shortages": info["shortages"], "audit_scope": info["audit_scope"]}, indent=2))


if __name__ == "__main__":
    main()
