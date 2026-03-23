"""Tests for sync_index and update_file fixes.

Run with: python -m unittest packages.leann-core.tests.test_sync -v
(from the LEANN root directory)
"""

import json
import os
import pickle
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch, call

import numpy as np

# Ensure leann is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from leann.api import LeannBuilder


def _make_index(tmp_dir, index_name="test_idx", dim=4, passages=None):
    """Create a minimal HNSW index on disk for testing."""
    try:
        import faiss
    except ImportError:
        from leann_backend_hnsw import faiss

    index_dir = Path(tmp_dir) / "indices"
    index_dir.mkdir(exist_ok=True)
    prefix = index_dir / index_name

    meta = {
        "backend_name": "hnsw",
        "embedding_model": "test-model",
        "embedding_mode": "sentence-transformers",
        "embedding_options": {},
        "backend_kwargs": {"is_compact": False},
        "is_compact": False,
        "is_pruned": False,
        "total_passages": 0,
        "dimensions": dim,
    }

    passages = passages or []

    offset_map = {}
    with open(f"{prefix}.passages.jsonl", "w") as pf:
        for p in passages:
            offset = pf.tell()
            offset_map[p["id"]] = offset
            pf.write(json.dumps(p) + "\n")

    meta["total_passages"] = len(passages)
    with open(f"{prefix}.meta.json", "w") as mf:
        json.dump(meta, mf)

    with open(f"{prefix}.passages.idx", "wb") as of:
        pickle.dump(offset_map, of)

    with open(f"{prefix}.ids.txt", "w") as idf:
        for p in passages:
            idf.write(p["id"] + "\n")

    hnsw = faiss.IndexHNSWFlat(dim, 16)
    if passages:
        vecs = np.random.rand(len(passages), dim).astype("float32")
        hnsw.add(vecs)
    faiss.write_index(hnsw, str(prefix) + ".index")

    return str(prefix)


def _write_file(path, content="hello world\nsome content here\n"):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content)
    return str(p)


# ---------------------------------------------------------------------------
# Issue 2: "No valid chunks to append" should be handled gracefully
# ---------------------------------------------------------------------------

class TestUpdateFileEmptyChunks(unittest.TestCase):
    """update_file should not raise when a file produces no valid text chunks."""

    @patch("leann.api.compute_embeddings")
    def test_whitespace_only_file_returns_zero_added(self, mock_embed):
        """A file with only whitespace should be skipped, not raise."""
        with tempfile.TemporaryDirectory() as tmp:
            index_path = _make_index(tmp)
            docs = Path(tmp) / "docs"
            docs.mkdir()
            ws_file = _write_file(docs / "empty.py", "   \n\n   \t  \n")

            builder = LeannBuilder(
                backend_name="hnsw",
                embedding_model="test-model",
                embedding_mode="sentence-transformers",
            )

            deleted, added = builder.update_file(index_path, ws_file)
            self.assertEqual(added, 0)
            mock_embed.assert_not_called()

    @patch("leann.api.compute_embeddings")
    def test_empty_file_returns_zero_added(self, mock_embed):
        """A completely empty file should be skipped gracefully."""
        with tempfile.TemporaryDirectory() as tmp:
            index_path = _make_index(tmp)
            docs = Path(tmp) / "docs"
            docs.mkdir()
            empty_file = _write_file(docs / "empty.txt", "")

            builder = LeannBuilder(
                backend_name="hnsw",
                embedding_model="test-model",
                embedding_mode="sentence-transformers",
            )

            deleted, added = builder.update_file(index_path, empty_file)
            self.assertEqual(added, 0)
            mock_embed.assert_not_called()

    @patch("leann.api.compute_embeddings")
    def test_deleted_file_returns_only_deleted_count(self, mock_embed):
        """If file was deleted, update_file should return (deleted, 0)."""
        with tempfile.TemporaryDirectory() as tmp:
            index_path = _make_index(tmp)
            nonexistent = str(Path(tmp) / "docs" / "gone.py")

            builder = LeannBuilder(
                backend_name="hnsw",
                embedding_model="test-model",
                embedding_mode="sentence-transformers",
            )

            deleted, added = builder.update_file(index_path, nonexistent)
            self.assertEqual(added, 0)
            mock_embed.assert_not_called()


# ---------------------------------------------------------------------------
# Issue 1: Sync speed — mtime/size caching
# ---------------------------------------------------------------------------

class TestSyncMtimeCaching(unittest.TestCase):
    """sync_index should use mtime+size to skip hash computation for unchanged files."""

    def test_hash_file_format_includes_mtime_and_size(self):
        """After sync, the hash file should store {path: {hash, mtime, size}}."""
        with tempfile.TemporaryDirectory() as tmp:
            index_path = _make_index(tmp)
            docs = Path(tmp) / "docs"
            docs.mkdir()
            _write_file(docs / "a.txt", "good content\n" * 20)

            builder = LeannBuilder(
                backend_name="hnsw",
                embedding_model="test-model",
                embedding_mode="sentence-transformers",
            )

            # Mock update_file to avoid actually embedding
            with patch.object(builder, "update_file", return_value=(0, 5)):
                builder.sync_index(index_path, str(docs))

            hash_file = Path(index_path).parent / f"{Path(index_path).name}.file_hashes.json"
            self.assertTrue(hash_file.exists())

            with open(hash_file) as f:
                hashes = json.load(f)

            self.assertEqual(len(hashes), 1)
            entry = list(hashes.values())[0]

            self.assertIsInstance(entry, dict)
            self.assertIn("hash", entry)
            self.assertIn("mtime", entry)
            self.assertIn("size", entry)
            self.assertIsInstance(entry["hash"], str)
            self.assertIsInstance(entry["mtime"], float)
            self.assertIsInstance(entry["size"], int)

    def test_unchanged_files_skip_hash_computation(self):
        """On second sync with no changes, _compute_file_hash should not be called."""
        with tempfile.TemporaryDirectory() as tmp:
            index_path = _make_index(tmp)
            docs = Path(tmp) / "docs"
            docs.mkdir()
            _write_file(docs / "a.txt", "good content\n" * 20)

            builder = LeannBuilder(
                backend_name="hnsw",
                embedding_model="test-model",
                embedding_mode="sentence-transformers",
            )

            # First sync
            with patch.object(builder, "update_file", return_value=(0, 5)):
                builder.sync_index(index_path, str(docs))

            # Second sync — file unchanged, should skip hash computation
            with patch.object(
                LeannBuilder, "_compute_file_hash",
                wraps=LeannBuilder._compute_file_hash,
            ) as mock_hash:
                builder2 = LeannBuilder(
                    backend_name="hnsw",
                    embedding_model="test-model",
                    embedding_mode="sentence-transformers",
                )
                stats = builder2.sync_index(index_path, str(docs))

                mock_hash.assert_not_called()
                self.assertEqual(stats["unchanged"], 1)
                self.assertEqual(stats["added"], 0)
                self.assertEqual(stats["modified"], 0)

    def test_modified_file_detected_by_mtime_change(self):
        """When a file's mtime changes, its hash should be recomputed."""
        with tempfile.TemporaryDirectory() as tmp:
            index_path = _make_index(tmp)
            docs = Path(tmp) / "docs"
            docs.mkdir()
            fpath = docs / "a.txt"
            _write_file(fpath, "original content\n" * 20)

            builder = LeannBuilder(
                backend_name="hnsw",
                embedding_model="test-model",
                embedding_mode="sentence-transformers",
            )

            # First sync
            with patch.object(builder, "update_file", return_value=(0, 5)):
                builder.sync_index(index_path, str(docs))

            # Modify the file
            time.sleep(0.05)
            _write_file(fpath, "modified content\n" * 20)

            # Second sync — should detect modification
            builder2 = LeannBuilder(
                backend_name="hnsw",
                embedding_model="test-model",
                embedding_mode="sentence-transformers",
            )
            with patch.object(builder2, "update_file", return_value=(5, 6)) as mock_update:
                stats = builder2.sync_index(index_path, str(docs))

            self.assertEqual(stats["modified"], 1)
            self.assertEqual(stats["unchanged"], 0)
            # update_file should have been called for the modified file
            mock_update.assert_called_once()

    def test_backward_compat_old_hash_format(self):
        """Old format {path: hash_str} should be migrated transparently."""
        with tempfile.TemporaryDirectory() as tmp:
            index_path = _make_index(tmp)
            index_dir = Path(index_path).parent
            index_name = Path(index_path).name

            docs = Path(tmp) / "docs"
            docs.mkdir()
            fpath = docs / "a.txt"
            _write_file(fpath, "content for backward compat test\n" * 20)

            # Write old-format hash file (plain string values)
            abs_path = str(fpath.resolve())
            file_hash = LeannBuilder._compute_file_hash(fpath)
            old_hashes = {abs_path: file_hash}
            hash_file = index_dir / f"{index_name}.file_hashes.json"
            with open(hash_file, "w") as f:
                json.dump(old_hashes, f)

            builder = LeannBuilder(
                backend_name="hnsw",
                embedding_model="test-model",
                embedding_mode="sentence-transformers",
            )

            # Sync — mtime won't match (old format uses 0.0), so hash is recomputed.
            # But content is the same, so hash matches → unchanged.
            stats = builder.sync_index(index_path, str(docs))
            self.assertEqual(stats["unchanged"], 1)

    def test_new_file_detected(self):
        """A new file added between syncs should be detected as added."""
        with tempfile.TemporaryDirectory() as tmp:
            index_path = _make_index(tmp)
            docs = Path(tmp) / "docs"
            docs.mkdir()
            _write_file(docs / "a.txt", "file a content\n" * 20)

            builder = LeannBuilder(
                backend_name="hnsw",
                embedding_model="test-model",
                embedding_mode="sentence-transformers",
            )

            # First sync with one file
            with patch.object(builder, "update_file", return_value=(0, 5)):
                builder.sync_index(index_path, str(docs))

            # Add a new file
            _write_file(docs / "b.txt", "file b content\n" * 20)

            builder2 = LeannBuilder(
                backend_name="hnsw",
                embedding_model="test-model",
                embedding_mode="sentence-transformers",
            )
            with patch.object(builder2, "update_file", return_value=(0, 3)) as mock_update:
                stats = builder2.sync_index(index_path, str(docs))

            self.assertEqual(stats["added"], 1)
            self.assertEqual(stats["unchanged"], 1)

    def test_deleted_file_detected(self):
        """A file removed between syncs should be detected as deleted."""
        with tempfile.TemporaryDirectory() as tmp:
            index_path = _make_index(tmp)
            docs = Path(tmp) / "docs"
            docs.mkdir()
            fpath = _write_file(docs / "a.txt", "file a content\n" * 20)
            _write_file(docs / "b.txt", "file b content\n" * 20)

            builder = LeannBuilder(
                backend_name="hnsw",
                embedding_model="test-model",
                embedding_mode="sentence-transformers",
            )

            # First sync with two files
            with patch.object(builder, "update_file", return_value=(0, 5)):
                with patch.object(builder, "delete_by_file", return_value=0):
                    builder.sync_index(index_path, str(docs))

            # Delete one file
            os.remove(fpath)

            builder2 = LeannBuilder(
                backend_name="hnsw",
                embedding_model="test-model",
                embedding_mode="sentence-transformers",
            )
            with patch.object(builder2, "delete_by_file", return_value=5) as mock_del:
                stats = builder2.sync_index(index_path, str(docs))

            self.assertEqual(stats["deleted"], 1)
            self.assertEqual(stats["unchanged"], 1)


# ---------------------------------------------------------------------------
# Sync resilience: individual file failures don't abort entire sync
# ---------------------------------------------------------------------------

class TestSyncResilience(unittest.TestCase):
    """sync_index should continue when individual files fail to process."""

    def test_sync_continues_on_file_failure(self):
        """If one file fails during sync, others should still be processed."""
        with tempfile.TemporaryDirectory() as tmp:
            index_path = _make_index(tmp)
            docs = Path(tmp) / "docs"
            docs.mkdir()
            _write_file(docs / "good.txt", "good content\n" * 20)
            _write_file(docs / "bad.txt", "bad content\n" * 20)

            builder = LeannBuilder(
                backend_name="hnsw",
                embedding_model="test-model",
                embedding_mode="sentence-transformers",
            )

            def selective_update(index_path, file_path):
                if "bad.txt" in file_path:
                    raise RuntimeError("Simulated failure")
                return (0, 5)

            with patch.object(builder, "update_file", side_effect=selective_update):
                stats = builder.sync_index(index_path, str(docs))

            self.assertEqual(stats.get("skipped", 0), 1)
            self.assertEqual(stats["added"], 1)

    def test_sync_saves_hashes_even_with_failures(self):
        """Hash file should be saved even when some files fail."""
        with tempfile.TemporaryDirectory() as tmp:
            index_path = _make_index(tmp)
            docs = Path(tmp) / "docs"
            docs.mkdir()
            _write_file(docs / "good.txt", "good content\n" * 20)
            _write_file(docs / "bad.txt", "bad content\n" * 20)

            builder = LeannBuilder(
                backend_name="hnsw",
                embedding_model="test-model",
                embedding_mode="sentence-transformers",
            )

            def selective_update(index_path, file_path):
                if "bad.txt" in file_path:
                    raise RuntimeError("Simulated failure")
                return (0, 5)

            with patch.object(builder, "update_file", side_effect=selective_update):
                builder.sync_index(index_path, str(docs))

            # Hash file should exist and contain both files
            hash_file = Path(index_path).parent / f"{Path(index_path).name}.file_hashes.json"
            self.assertTrue(hash_file.exists())
            with open(hash_file) as f:
                hashes = json.load(f)
            self.assertEqual(len(hashes), 2)


if __name__ == "__main__":
    unittest.main()
