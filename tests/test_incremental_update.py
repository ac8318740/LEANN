"""
Test incremental update functionality: remove-file, update-file, and sync commands.
"""

import json
import pytest
from pathlib import Path
from leann.cli import LeannCLI
from leann.api import LeannBuilder


# =============================================================================
# CLI Parser Tests
# =============================================================================


def test_cli_remove_file_accepts_required_args(tmp_path, monkeypatch):
    """Test that remove-file command parser accepts required arguments."""
    monkeypatch.chdir(tmp_path)

    cli = LeannCLI()
    parser = cli.create_parser()

    args = parser.parse_args(["remove-file", "my-index", "src/foo.py"])

    assert args.command == "remove-file"
    assert args.index_name == "my-index"
    assert args.file_path == "src/foo.py"


def test_cli_update_file_accepts_required_args(tmp_path, monkeypatch):
    """Test that update-file command parser accepts required arguments."""
    monkeypatch.chdir(tmp_path)

    cli = LeannCLI()
    parser = cli.create_parser()

    args = parser.parse_args(["update-file", "my-index", "src/foo.py"])

    assert args.command == "update-file"
    assert args.index_name == "my-index"
    assert args.file_path == "src/foo.py"


def test_cli_sync_accepts_required_args(tmp_path, monkeypatch):
    """Test that sync command parser accepts required arguments."""
    monkeypatch.chdir(tmp_path)

    cli = LeannCLI()
    parser = cli.create_parser()

    args = parser.parse_args(["sync", "my-index", "--docs", "./src"])

    assert args.command == "sync"
    assert args.index_name == "my-index"
    assert args.docs == "./src"


def test_cli_sync_accepts_file_types(tmp_path, monkeypatch):
    """Test that sync command accepts file type filters."""
    monkeypatch.chdir(tmp_path)

    cli = LeannCLI()
    parser = cli.create_parser()

    args = parser.parse_args(
        ["sync", "my-index", "--docs", "./src", "--file-types", ".py,.js"]
    )

    assert args.command == "sync"
    assert args.file_types == ".py,.js"


def test_cli_sync_accepts_include_hidden(tmp_path, monkeypatch):
    """Test that sync command accepts include-hidden flag."""
    monkeypatch.chdir(tmp_path)

    cli = LeannCLI()
    parser = cli.create_parser()

    args = parser.parse_args(["sync", "my-index", "--docs", "./src", "--include-hidden"])

    assert args.command == "sync"
    assert args.include_hidden is True


def test_cli_sync_defaults(tmp_path, monkeypatch):
    """Test that sync command has correct default values."""
    monkeypatch.chdir(tmp_path)

    cli = LeannCLI()
    parser = cli.create_parser()

    args = parser.parse_args(["sync", "my-index", "--docs", "./src"])

    assert args.include_hidden is False
    assert args.file_types is None


# =============================================================================
# API Unit Tests
# =============================================================================


def test_compute_file_hash():
    """Test that _compute_file_hash returns consistent SHA256 hashes."""
    import tempfile
    import os

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write("print('hello world')\n")
        temp_path = f.name

    try:
        hash1 = LeannBuilder._compute_file_hash(Path(temp_path))
        hash2 = LeannBuilder._compute_file_hash(Path(temp_path))

        # Same content should produce same hash
        assert hash1 == hash2
        # SHA256 hash is 64 hex characters
        assert len(hash1) == 64
        assert all(c in "0123456789abcdef" for c in hash1)
    finally:
        os.unlink(temp_path)


def test_compute_file_hash_different_content():
    """Test that different content produces different hashes."""
    import tempfile
    import os

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write("content1")
        path1 = f.name

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write("content2")
        path2 = f.name

    try:
        hash1 = LeannBuilder._compute_file_hash(Path(path1))
        hash2 = LeannBuilder._compute_file_hash(Path(path2))
        assert hash1 != hash2
    finally:
        os.unlink(path1)
        os.unlink(path2)


# =============================================================================
# Validation Tests
# =============================================================================


def _create_mock_index(tmp_path, index_name, meta):
    """Helper to create a minimal mock index for testing validation logic."""
    import pickle

    index_path = tmp_path / index_name

    # Create all required files
    with open(str(index_path) + ".meta.json", "w") as f:
        json.dump(meta, f)

    # Create empty passages file
    with open(str(index_path) + ".passages.jsonl", "w") as f:
        pass

    # Create empty offset file (pickle of empty dict)
    with open(str(index_path) + ".passages.idx", "wb") as f:
        pickle.dump({}, f)

    # Create dummy index file (just needs to exist for validation)
    stem = index_name.replace(".leann", "") if index_name.endswith(".leann") else index_name
    with open(tmp_path / f"{stem}.index", "wb") as f:
        f.write(b"dummy")

    return index_path


def test_delete_by_file_rejects_compact_index(tmp_path, monkeypatch):
    """Test that delete_by_file raises error for compact indices."""
    monkeypatch.chdir(tmp_path)

    meta = {
        "is_compact": True,
        "backend_name": "hnsw",
        "embedding_model": "test-model",
        "embedding_mode": "sentence-transformers",
    }
    index_path = _create_mock_index(tmp_path, "test.leann", meta)

    builder = LeannBuilder(backend_name="hnsw")

    with pytest.raises(ValueError, match="[Cc]ompact"):
        builder.delete_by_file(str(index_path), "/some/file.py")


def test_delete_by_file_rejects_pruned_index(tmp_path, monkeypatch):
    """Test that delete_by_file raises error for pruned indices."""
    monkeypatch.chdir(tmp_path)

    meta = {
        "is_compact": False,
        "is_pruned": True,
        "backend_name": "hnsw",
        "backend_kwargs": {"is_recompute": True},
    }
    index_path = _create_mock_index(tmp_path, "test.leann", meta)

    builder = LeannBuilder(backend_name="hnsw")

    with pytest.raises(ValueError, match="pruned"):
        builder.delete_by_file(str(index_path), "/some/file.py")


def test_delete_by_file_missing_index(tmp_path, monkeypatch):
    """Test that delete_by_file raises error for non-existent index."""
    monkeypatch.chdir(tmp_path)

    builder = LeannBuilder(backend_name="hnsw")

    with pytest.raises(FileNotFoundError):
        builder.delete_by_file(str(tmp_path / "nonexistent.leann"), "/some/file.py")


# =============================================================================
# Path Resolution Tests
# =============================================================================


def test_resolve_index_path_with_full_path():
    """Test that _resolve_index_path returns full paths as-is."""
    cli = LeannCLI()
    result = cli._resolve_index_path("/full/path/to/index.leann")
    assert result == "/full/path/to/index.leann"


def test_resolve_index_path_with_relative_path():
    """Test that _resolve_index_path returns relative paths with slashes as-is."""
    cli = LeannCLI()
    result = cli._resolve_index_path("./indexes/my-index.leann")
    assert result == "./indexes/my-index.leann"


def test_resolve_index_path_with_name_only(tmp_path, monkeypatch):
    """Test that _resolve_index_path expands simple names to full path."""
    monkeypatch.chdir(tmp_path)

    cli = LeannCLI()
    # Default indexes_dir is ~/leann-indexes
    result = cli._resolve_index_path("my-index")

    # Should expand to indexes_dir/my-index/documents.leann
    assert result.endswith("my-index/documents.leann")
    assert "my-index/my-index.leann" not in result  # Bug was here


# =============================================================================
# Sync Index Validation Tests
# =============================================================================


def test_sync_index_rejects_compact_index(tmp_path, monkeypatch):
    """Test that sync_index raises error for compact indices."""
    monkeypatch.chdir(tmp_path)

    meta = {
        "is_compact": True,
        "backend_name": "hnsw",
    }
    index_path = _create_mock_index(tmp_path, "test.leann", meta)

    builder = LeannBuilder(backend_name="hnsw")

    with pytest.raises(ValueError, match="[Cc]ompact"):
        builder.sync_index(str(index_path), str(tmp_path))


def test_sync_index_rejects_pruned_index(tmp_path, monkeypatch):
    """Test that sync_index raises error for pruned indices."""
    monkeypatch.chdir(tmp_path)

    meta = {
        "is_compact": False,
        "is_pruned": True,
        "backend_name": "hnsw",
        "backend_kwargs": {"is_recompute": True},
    }
    index_path = _create_mock_index(tmp_path, "test.leann", meta)

    builder = LeannBuilder(backend_name="hnsw")

    with pytest.raises(ValueError, match="pruned"):
        builder.sync_index(str(index_path), str(tmp_path))
