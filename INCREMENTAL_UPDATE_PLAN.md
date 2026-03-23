# LEANN Incremental Update Enhancement Plan

## Problem Statement

LEANN's current `update_index()` API only supports **appending** new documents. It cannot handle:
- **Edited files**: Old chunks remain in index alongside new chunks (duplicates/stale data)
- **Deleted files**: Old chunks persist indefinitely (stale data)

For development codebases where ~80% of changes are edits, this means frequent full rebuilds are still required, negating much of the benefit of incremental updates.

## Goal

Add `delete_by_file()` and `update_file()` methods to enable true incremental updates for all change types:

| Operation | Current | After Enhancement |
|-----------|---------|-------------------|
| Add new file | `update_index()` | `update_index()` |
| Edit file | Full rebuild required | `update_file()` |
| Delete file | Full rebuild required | `delete_by_file()` |

## Why This Matters

Cost comparison for a single file edit in a 9,000 chunk index:

| Approach | Chunks processed | Approx. cost |
|----------|------------------|--------------|
| Full rebuild | 9,000 | ~$0.50-1.00 |
| `update_file()` | ~20 | ~$0.01 |

**50-100x cheaper** for the common case (editing existing files).

## Technical Feasibility

### What Already Exists

1. **FAISS `remove_ids()`**: The underlying HNSW backend (FAISS) supports vector deletion
2. **File metadata**: Chunks already store source file path in `passage_sources`
3. **`update_index()`**: Appending new vectors to existing index already works

### What Needs to Be Built

```python
# In packages/leann-core/src/leann/api.py

def delete_by_file(self, index_path: str, file_path: str) -> int:
    """
    Remove all chunks originating from a specific file.

    Steps:
    1. Load index metadata to find chunk IDs where source_file == file_path
    2. Call faiss_index.remove_ids(chunk_ids)
    3. Remove corresponding entries from document store
    4. Update metadata

    Returns: Number of chunks deleted
    """

def update_file(self, index_path: str, file_path: str):
    """
    Re-index a single file (handles edits).

    Steps:
    1. delete_by_file(index_path, file_path)
    2. Chunk the updated file
    3. Generate embeddings for new chunks
    4. update_index() to add new chunks
    """
```

## Implementation Plan

### Phase 1: Understand Current Structure
- [ ] Read `api.py` to understand `LeannSearcher` and `LeannBuilder` classes
- [ ] Trace how chunks are stored (document store format)
- [ ] Trace how file paths are stored in metadata (`passage_sources`)
- [ ] Find where FAISS index is accessed

### Phase 2: Implement `delete_by_file()`
- [ ] Add method to find chunk IDs by source file path
- [ ] Add method to delete from FAISS using `remove_ids()`
- [ ] Add method to remove from document store
- [ ] Update index metadata after deletion
- [ ] Write tests

### Phase 3: Implement `update_file()`
- [ ] Combine delete + re-chunk + update_index
- [ ] Handle edge cases (file doesn't exist in index, file was deleted)
- [ ] Write tests

### Phase 4: Add CLI Commands
- [ ] `leann remove-file <index> <file>` - delete chunks from a file
- [ ] `leann update-file <index> <file>` - re-index a single file
- [ ] Update CLI help and documentation

### Phase 5: Contribution
- [ ] Test thoroughly with real indexes
- [ ] Clean up code, add docstrings
- [ ] Open PR to upstream (yichuan-w/LEANN)

## Branch Structure

```
main                    <- upstream main
pr-184-update-cli       <- PR #184 (adds `leann update` CLI command)
feature/delete-by-file  <- our work (based on pr-184)
```

## Setup Notes

This repo is installed in editable mode:
- `leann-core`: Editable from `/home/acoote/LEANN/packages/leann-core`
- `leann-backend-hnsw`: From PyPI (pre-built, no modifications needed)
- Changes to `api.py` take effect immediately

## Key Files

| File | Purpose |
|------|---------|
| `packages/leann-core/src/leann/api.py` | Main API - add methods here |
| `packages/leann-core/src/leann/cli.py` | CLI commands - add commands here |
| `packages/leann-backend-hnsw/.../faiss.py` | FAISS bindings (has `remove_ids()`) |

## References

- [PR #184](https://github.com/yichuan-w/LEANN/pull/184) - Adds `leann update` CLI (base for our work)
- [Issue #141](https://github.com/yichuan-w/LEANN/issues/141) - Feature request for reindex functionality
- [FAISS remove_ids](https://github.com/facebookresearch/faiss/wiki/FAQ#how-can-i-remove-elements-from-an-index) - FAISS deletion documentation
