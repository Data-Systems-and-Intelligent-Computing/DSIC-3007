import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from src.collectors import download_ta_data as collector


def doc(text, repo="repo-a", aliases=()):
    return {"text": text, "num_tokens": len(text),
            "source_metadata": {"max_stars_repo_name": repo},
            "duplicate_source_metadata": [
                {"max_stars_repo_name": alias} for alias in aliases]}


class Stream:
    def __init__(self, rows):
        self.rows = rows
        self.consumed = 0

    def shuffle(self, **kwargs):
        self.shuffle_args = kwargs
        return self

    def __iter__(self):
        for row in self.rows:
            self.consumed += 1
            yield row


class CollectionTests(unittest.TestCase):
    def test_cap_counts_duplicate_aliases(self):
        docs = [doc("shared", "a", ("b",)), doc("a2", "a"),
                doc("b2", "b"), doc("c", "c"), doc("d", "d")]
        selected = collector.select_documents(
            docs, target_documents=3, max_files_per_repo=1,
            group_column="max_stars_repo_name")
        counts = {}
        for item in selected:
            for row in [item["source_metadata"], *item["duplicate_source_metadata"]]:
                repo = row["max_stars_repo_name"]
                counts[repo] = counts.get(repo, 0) + 1
        self.assertTrue(all(count <= 1 for count in counts.values()))

    def test_document_budget_and_determinism(self):
        docs = [doc(str(i) * (i + 1), str(i)) for i in range(20)]
        kwargs = dict(target_documents=8, max_files_per_repo=1,
                      group_column="max_stars_repo_name", seed=5)
        selected = collector.select_documents(docs, **kwargs)
        self.assertEqual(selected, collector.select_documents(docs, **kwargs))
        self.assertEqual(len(selected), 8)
        cal, evaluation = collector.split_documents(
            docs, calibration_ratio=.25, seed=42)
        self.assertEqual((len(cal), len(evaluation)), (5, 15))

    def test_duplicate_bridge_keeps_repositories_together(self):
        docs = [doc("bridge", "a", ("b",)), doc("a", "a"),
                doc("b", "b"), doc("c", "c")]
        cal, evaluation = collector.split_documents(
            docs, calibration_ratio=.5, seed=42,
            group_column="max_stars_repo_name")
        def labels(items):
            return {row["max_stars_repo_name"] for item in items
                    for row in [item["source_metadata"],
                                *item["duplicate_source_metadata"]]}
        self.assertFalse(labels(cal) & labels(evaluation))
        self.assertTrue(cal and evaluation)

    def run_collection(self, rows, output, **kwargs):
        stream = Stream(rows)
        tokenizer = SimpleNamespace(encode=lambda text, **kw: list(text))
        api = SimpleNamespace(dataset_info=lambda *a, **kw: SimpleNamespace(sha="revision"))
        with patch.object(collector, "load_dataset", return_value=stream), \
             patch.object(collector, "HfApi", return_value=api), \
             contextlib.redirect_stdout(io.StringIO()):
            collector.process_dataset(
                "code", collector.DATASETS["code"], tokenizer, "tokenizer-rev",
                output_dir=output, **kwargs)
        return stream

    def test_invalid_rows_cannot_escape_bound(self):
        stream = Stream([{"content": None}] * 30)
        with tempfile.TemporaryDirectory() as temp, \
             patch.object(collector, "load_dataset", return_value=stream), \
             patch.object(collector, "HfApi", return_value=SimpleNamespace(
                 dataset_info=lambda *a, **k: SimpleNamespace(sha="rev"))), \
             contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaises(RuntimeError):
                collector.process_dataset(
                    "code", collector.DATASETS["code"], None, "rev",
                    target_documents=2, max_files_per_repo=1,
                    max_source_rows=5, output_dir=Path(temp))
            self.assertEqual(stream.consumed, 5)
            self.assertFalse((Path(temp) / "code").exists())

    def test_metadata_dedup_and_full_pool_scan(self):
        rows = [{"content": "unique text " + str(i),
                 "max_stars_repo_name": "repo" + str(i),
                 "max_stars_repo_path": str(i) + ".py"} for i in range(8)]
        rows.append(dict(rows[0]))
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            stream = self.run_collection(rows, root, target_documents=4,
                                         max_files_per_repo=1, max_source_rows=9)
            meta = json.loads((root / "code/metadata.json").read_text())
            self.assertEqual(stream.consumed, 9)
            self.assertEqual(meta["documents_selected"], 4)
            self.assertEqual(meta["exact_duplicates_removed"], 1)
            self.assertEqual(meta["repository_groups_shared_across_splits"], 0)
            self.assertEqual(meta["resolved_revision"], "revision")
            records = []
            for name in ("calibration", "evaluation"):
                records.extend(json.loads(line) for line in
                               (root / f"code/{name}.jsonl").read_text().splitlines())
            self.assertEqual(len({r["content_sha256"] for r in records}), 4)
            self.assertTrue(all("max_stars_repo_path" in r["source_metadata"] for r in records))
            with self.assertRaises(FileExistsError):
                collector._ensure_outputs_are_new(root / "code")

    def test_inadequate_diversity_fails(self):
        with self.assertRaises(RuntimeError):
            collector.select_documents([doc(str(i)) for i in range(10)],
                                       target_documents=4, max_files_per_repo=2,
                                       group_column="max_stars_repo_name")
        with self.assertRaises(ValueError):
            collector.split_documents([doc("one"), doc("two")],
                                      calibration_ratio=.5, seed=42,
                                      group_column="max_stars_repo_name")


if __name__ == "__main__":
    unittest.main()
