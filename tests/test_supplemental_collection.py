import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from src import collection
from src.artifacts import read_json, read_rows, verify_bundle
from src.preparation import retained_documents, verify_catalog


REVISION = "a" * 40


class FakeStream:
    def __init__(self, rows):
        self.rows = rows
        self.yielded = 0

    def __iter__(self):
        for row in self.rows:
            self.yielded += 1
            yield row


class FakeTokenizer:
    is_fast = True

    def encode(self, text, add_special_tokens=False):
        assert add_special_tokens is False
        return list(range(int(text.split(":", 1)[0])))


def setup_inputs(root):
    model = {
        "repo_id": "allenai/OLMoE-1B-7B-0924",
        "revision": "b" * 40,
        "revision_type": "full_commit_sha",
        "files": [],
    }
    tokenizer_dir = root / "tokenizer"
    tokenizer_dir.mkdir()
    for name in collection.TOKENIZER_FILES:
        payload = (name + " pinned bytes").encode()
        (tokenizer_dir / name).write_bytes(payload)
        model["files"].append({"path": name, "size_bytes": len(payload),
                               "sha256": hashlib.sha256(payload).hexdigest()})
    model_path = root / "model-catalog.json"
    model_path.write_text(json.dumps(model))
    settings = {
        "sampling": {"master_seed": 3007, "probe_positions_per_document": 4},
        "context": {"primary_window": 4096, "minimum_preceding_tokens": 4095,
                    "excluded_predictor_target_ids": []},
    }
    return tokenizer_dir, model_path, settings


class SupplementalCollectionTests(unittest.TestCase):
    def collect(self, root, rows, *, limit=10, revision=REVISION):
        tokenizer_dir, model_path, settings = setup_inputs(root)
        stream = FakeStream(rows)
        config = {"repo_id": "org/pinned-dataset", "split": "train",
                  "text_column": "text", "source_group_column": "id"}
        with patch.object(collection, "_load_dataset", return_value=stream), \
                patch.object(collection, "_load_tokenizer", return_value=FakeTokenizer()):
            result = collection.collect_dataset(
                domain="scientific", config=config, revision=revision,
                max_source_rows=limit, tokenizer_dir=tokenizer_dir,
                model_catalog_path=model_path, settings=settings,
                output=root / "supplement")
        return stream, result

    def test_bounded_full_scan_keeps_all_eligible_and_exact_duplicate_aliases(self):
        rows = [
            {"text": "4100:0", "id": "paper-a", "license": "x"},
            {"text": "4097:1", "id": "paper-b"},
            {"text": "4100:0", "id": "paper-alias", "license": "y"},
            {"text": "short", "id": "paper-short"},
        ]
        # The fake tokenizer accepts numeric lengths only, so make the short row
        # a valid source text with fewer than the required 4097 tokens.
        rows[-1]["text"] = "100:short"
        stream, manifest = self.collect(Path(self._temp.name), rows, limit=4)
        root = Path(self._temp.name) / "supplement"
        self.assertEqual(stream.yielded, 4)
        self.assertEqual(manifest["kind"], "supplemental-collection")
        metadata = read_json(root / "scientific/metadata.json")
        self.assertEqual(metadata["coverage"]["rows_seen"], 4)
        self.assertEqual(metadata["coverage"]["unique_eligible_documents"], 2)
        self.assertEqual(metadata["coverage"]["duplicate_rows_collapsed"], 1)
        self.assertEqual(metadata["coverage"]["rows_below_4096_input_plus_target"], 1)
        self.assertEqual(metadata["coverage"]["documents_with_at_least_four_positions"], 1)
        docs = list(read_rows(root / "scientific/evaluation.jsonl"))
        self.assertEqual(len(docs), 2)
        self.assertEqual(docs[0]["content_sha256"], hashlib.sha256(b"4100:0").hexdigest())
        self.assertEqual(len(docs[0]["duplicate_source_metadata"]), 1)
        self.assertEqual(docs[0]["duplicate_source_metadata"][0]["id"], "paper-alias")
        self.assertTrue(all(d["num_tokens"] >= 4097 for d in docs))
        catalog = read_json(root / "catalog.json")
        verify_catalog(root, catalog)
        ingested = list(retained_documents(root, catalog))
        self.assertEqual(len(ingested), 2)
        self.assertEqual(ingested[0]["dataset"], "org/pinned-dataset")
        self.assertEqual(ingested[0]["revision"], REVISION)
        self.assertEqual(ingested[0]["text"], "4100:0")
        self.assertEqual([f["role"] for f in catalog["files"]],
                         ["evaluation", "collection_metadata"])
        self.assertEqual(verify_bundle(root, "supplemental-collection"), manifest)

    def setUp(self):
        self._temp = tempfile.TemporaryDirectory()

    def tearDown(self):
        self._temp.cleanup()

    def test_scan_does_not_consume_a_row_beyond_bound_or_make_partition(self):
        rows = [{"text": f"4100:{i}", "id": f"paper-{i}"} for i in range(6)]
        stream, _ = self.collect(Path(self._temp.name), rows, limit=3)
        root = Path(self._temp.name) / "supplement"
        self.assertEqual(stream.yielded, 3)
        metadata = read_json(root / "scientific/metadata.json")
        self.assertEqual(metadata["coverage"]["rows_seen"], 3)
        self.assertEqual(metadata["coverage"]["source_exhausted_before_bound"], False)
        docs = list(read_rows(root / "scientific/evaluation.jsonl"))
        self.assertEqual(len(docs), 3)
        self.assertFalse((root / "scientific/calibration.jsonl").exists())

    def test_code_keeps_more_than_ten_files_from_one_repo_and_forwards_pin(self):
        root = Path(self._temp.name)
        tokenizer_dir, model_path, settings = setup_inputs(root)
        rows = [{"content": f"4100:{i}", "max_stars_repo_name": "org/one-repo",
                 "max_stars_repo_path": f"src/file-{i}.py"} for i in range(12)]
        stream = FakeStream(rows)
        calls = {}
        config = {"repo_id": "org/pinned-code", "name": "python-config",
                  "split": "train", "data_dir": "python", "text_column": "content",
                  "source_group_column": "max_stars_repo_name"}

        def load_dataset(**kwargs):
            calls.update(kwargs)
            return stream

        with patch.object(collection, "_load_dataset", side_effect=load_dataset), \
                patch.object(collection, "_load_tokenizer", return_value=FakeTokenizer()):
            collection.collect_dataset(
                domain="code", config=config, revision=REVISION, max_source_rows=12,
                tokenizer_dir=tokenizer_dir, model_catalog_path=model_path,
                settings=settings, output=root / "code-supplement")

        self.assertEqual(calls, {"path": "org/pinned-code", "name": "python-config",
                                 "split": "train", "revision": REVISION,
                                 "streaming": True, "data_dir": "python"})
        docs = list(read_rows(root / "code-supplement/code/evaluation.jsonl"))
        self.assertEqual(len(docs), 12)
        self.assertEqual({d["source_metadata"]["max_stars_repo_name"] for d in docs},
                         {"org/one-repo"})
        self.assertEqual({d["source_metadata"]["max_stars_repo_path"] for d in docs},
                         {f"src/file-{i}.py" for i in range(12)})

    def test_requires_pinned_revision_and_verified_local_tokenizer(self):
        root = Path(self._temp.name)
        tokenizer_dir, model_path, settings = setup_inputs(root)
        config = {"repo_id": "org/data", "split": "train", "text_column": "text"}
        with self.assertRaisesRegex(ValueError, "40-character"):
            collection.collect_dataset(domain="scientific", config=config,
                revision="main", max_source_rows=1, tokenizer_dir=tokenizer_dir,
                model_catalog_path=model_path, settings=settings, output=root / "out")

        (tokenizer_dir / "tokenizer.json").write_text("changed")
        with self.assertRaisesRegex(ValueError, "does not match"):
            collection.collect_dataset(domain="scientific", config=config,
                revision=REVISION, max_source_rows=1, tokenizer_dir=tokenizer_dir,
                model_catalog_path=model_path, settings=settings, output=root / "out")

    def test_refuses_existing_output(self):
        root = Path(self._temp.name)
        tokenizer_dir, model_path, settings = setup_inputs(root)
        output = root / "existing"
        output.mkdir()
        with self.assertRaises(FileExistsError):
            collection.collect_dataset(domain="scientific",
                config={"repo_id": "org/data", "split": "train", "text_column": "text"},
                revision=REVISION, max_source_rows=1, tokenizer_dir=tokenizer_dir,
                model_catalog_path=model_path, settings=settings, output=output)

    def test_cli_accepts_explicit_bounds_and_reads_source_config(self):
        root = Path(self._temp.name)
        source_config = root / "source.json"
        source = {"repo_id": "org/pinned", "split": "validation",
                  "text_column": "text", "name": "en", "data_dir": "python"}
        source_config.write_text(json.dumps(source))
        settings_path = root / "research.json"
        settings_path.write_text(json.dumps({"sampling": {"master_seed": 3007}}))
        model_path = root / "model.json"
        model_path.write_text("{}")
        result = {"status": "complete", "kind": "supplemental-collection",
                  "metadata": {"rows_seen": 2}}
        with patch.object(collection, "collect_dataset", return_value=result) as collect:
            exit_code = collection.main([
                "--domain", "c4", "--source-config", str(source_config),
                "--revision", REVISION, "--max-source-rows", "17",
                "--tokenizer-dir", str(root / "tokenizer"),
                "--model-catalog", str(model_path),
                "--settings", str(settings_path),
                "--output", str(root / "output"),
            ])
        self.assertEqual(exit_code, 0)
        kwargs = collect.call_args.kwargs
        self.assertEqual(kwargs["config"], source)
        self.assertEqual(kwargs["revision"], REVISION)
        self.assertEqual(kwargs["max_source_rows"], 17)


if __name__ == "__main__":
    unittest.main()
