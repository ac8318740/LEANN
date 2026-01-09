# LEANN CLI Incremental Commands - User Guide

## Overview

Three new commands have been added to the LEANN CLI for fine-grained index management. These commands enable incremental updates without rebuilding entire indexes.

## Commands

### 1. `remove-file` - Remove a File from Index

Remove all chunks from a specific file without affecting other indexed files.

#### Syntax
```bash
leann remove-file <index_name> <file_path>
```

#### Arguments
- `index_name`: Name of the index (e.g., "my-docs")
- `file_path`: Path to the file to remove (absolute or relative)

#### Examples
```bash
# Remove a Python file from the index
leann remove-file my-docs ./src/deprecated.py

# Remove using absolute path
leann remove-file my-docs /home/user/project/old_file.txt
```

#### Output
```
🗑️  Removing file from index: ./src/deprecated.py
✅ Removed 15 chunks from file: ./src/deprecated.py
```

#### Error Cases
- `❌ Index not found`: The specified index doesn't exist
- `❌ Cannot remove`: Index validation failed (wrong backend, compact mode)
- `❌ Not found`: The file doesn't exist or has no chunks in the index

---

### 2. `update-file` - Re-index a Single File

Delete old chunks and add new chunks for a single file. Useful when you've modified a file and want to update just that file in the index.

#### Syntax
```bash
leann update-file <index_name> <file_path>
```

#### Arguments
- `index_name`: Name of the index
- `file_path`: Path to the file to re-index

#### Examples
```bash
# Update a modified Python file
leann update-file my-docs ./src/utils.py

# Update a markdown file
leann update-file my-docs ./docs/README.md
```

#### Output
```
🔄 Updating file in index: ./src/utils.py
✅ Updated: deleted 12 chunks, added 15 chunks
```

#### Use Cases
- File content has changed
- You fixed a typo in documentation
- Code was refactored
- Want to refresh specific file without full rebuild

#### Error Cases
- `❌ Index not found`: The specified index doesn't exist
- `❌ Cannot update`: Index validation failed
- `❌ Not found`: The file doesn't exist

---

### 3. `sync` - Synchronize Index with Directory

Automatically detect changes in a directory and update the index accordingly. Uses content hashes to identify added, modified, and deleted files.

#### Syntax
```bash
leann sync <index_name> --docs <directory> [--file-types <extensions>] [--include-hidden]
```

#### Arguments
- `index_name`: Name of the index
- `--docs`: Directory to sync with (required)
- `--file-types`: Comma-separated file extensions to include (optional)
- `--include-hidden`: Include hidden files/directories (optional, default: false)

#### Examples

**Basic sync:**
```bash
# Sync entire source directory
leann sync my-docs --docs ./src
```

**Sync with file type filter:**
```bash
# Sync only Python and JavaScript files
leann sync my-docs --docs ./src --file-types .py,.js

# Sync only markdown files
leann sync my-docs --docs ./docs --file-types .md
```

**Include hidden files:**
```bash
# Include hidden configuration files
leann sync my-docs --docs ./config --include-hidden
```

#### Output
```
🔄 Syncing index 'my-docs' with './src'

📊 Sync complete:
   Added:     3 files (+45 chunks)
   Modified:  2 files
   Deleted:   1 files (-12 chunks)
   Unchanged: 25 files
```

#### What Happens During Sync

1. **Added Files**: New files in directory are indexed and added
2. **Modified Files**: Files with changed content are re-indexed (old chunks deleted, new chunks added)
3. **Deleted Files**: Files removed from directory have their chunks removed from index
4. **Unchanged Files**: Files with same content hash are skipped (no work needed)

#### Use Cases
- Daily/periodic index updates
- CI/CD pipeline integration
- Keep index in sync with evolving codebase
- Automated index maintenance

#### Error Cases
- `❌ Index not found`: The specified index doesn't exist
- `❌ Cannot sync`: Index validation failed
- `❌ Not found`: Directory doesn't exist

---

## Prerequisites

### Index Requirements

All three commands require:
1. **HNSW backend**: Only HNSW indices support incremental updates
2. **Non-compact mode**: Index must be built with `--no-compact` flag

### Creating a Compatible Index

```bash
# Build index with incremental update support
leann build my-docs --docs ./documents --no-compact

# Build with specific backend
leann build my-docs --docs ./src --backend-name hnsw --no-compact
```

If you have a compact index and need incremental updates:
```bash
# Rebuild with non-compact mode
leann build my-docs --docs ./documents --no-compact --force
```

---

## Workflows

### Daily Development Workflow

```bash
# Morning: Sync index with latest changes
leann sync my-code --docs ./src

# After editing a file
leann update-file my-code ./src/module.py

# After deleting obsolete files
leann remove-file my-code ./src/deprecated.py
```

### Continuous Integration

```bash
#!/bin/bash
# ci-update-index.sh

# Update search index with latest code
leann sync docs-index --docs ./docs --file-types .md
leann sync code-index --docs ./src --file-types .py,.js,.ts
```

### Selective Updates

```bash
# Sync only specific file types
leann sync my-docs --docs ./src --file-types .py

# Update only changed files in last day (using find)
find ./src -name "*.py" -mtime -1 | while read file; do
    leann update-file my-code "$file"
done
```

---

## Performance Considerations

### Speed Comparison

- **Full Rebuild**: `leann build` (slowest, processes all files)
- **Sync**: `leann sync` (medium, checks hashes, updates changed files)
- **Update File**: `leann update-file` (fast, single file)
- **Remove File**: `leann remove-file` (fast, single file)

### When to Use Each Command

| Scenario | Recommended Command |
|----------|---------------------|
| Initial index creation | `leann build` |
| Daily/periodic updates | `leann sync` |
| Single file edited | `leann update-file` |
| File deleted from project | `leann remove-file` |
| Major refactoring | `leann build --force` |
| Multiple scattered changes | `leann sync` |

---

## Technical Details

### Content Hashing (sync)

The `sync` command uses content hashes to detect changes:
- Each file's content is hashed
- Hashes are stored in index metadata
- On sync, current hashes are compared with stored hashes
- Only files with changed hashes are re-indexed

### Chunk Management

- Chunks include metadata: `file_path`, `file_name`, timestamps
- File identification uses absolute paths
- All chunks from a file share the same `file_path` metadata

### File Filters

File filters for `sync`:
```python
# Extension filter example
--file-types .py,.js  # Includes *.py and *.js files

# Hidden files filter
--include-hidden      # Includes files/dirs starting with '.'
```

Filters are combined:
1. Extension filter (if specified)
2. Hidden files filter (if not disabled)
3. Both filters must pass for file to be included

---

## Troubleshooting

### "Index not found"
```bash
# Check available indexes
leann list

# Ensure you're using the correct index name
leann sync my-docs --docs ./src  # Use exact name from 'leann list'
```

### "Cannot update: Index is compact"
```bash
# Rebuild without compact mode
leann build my-docs --docs ./documents --no-compact --force
```

### "Cannot remove: Only HNSW indices support updates"
```bash
# Check index backend
cat .leann/indexes/my-docs/documents.leann.meta.json | grep backend_name

# Rebuild with HNSW backend
leann build my-docs --docs ./documents --backend-name hnsw --no-compact --force
```

### File path issues
```bash
# Use absolute paths to avoid confusion
leann update-file my-docs /home/user/project/file.py

# Or use resolve to get absolute path
leann update-file my-docs $(realpath ./src/file.py)
```

---

## Examples

### Example 1: Code Documentation Index

```bash
# Initial setup
leann build code-docs --docs ./src ./docs --no-compact --file-types .py,.md

# Daily sync
leann sync code-docs --docs ./src --file-types .py
leann sync code-docs --docs ./docs --file-types .md

# Fix specific file after edit
leann update-file code-docs ./docs/API.md
```

### Example 2: Project Knowledge Base

```bash
# Build knowledge base
leann build kb --docs ./knowledge ./docs ./config --no-compact

# Update after meeting (new notes added)
leann sync kb --docs ./knowledge

# Remove outdated document
leann remove-file kb ./knowledge/deprecated/old_process.md
```

### Example 3: Multi-language Codebase

```bash
# Build full index
leann build app-index --docs ./src --no-compact

# Sync only Python files
leann sync app-index --docs ./src/backend --file-types .py

# Sync only TypeScript files
leann sync app-index --docs ./src/frontend --file-types .ts,.tsx

# Update a specific config file
leann update-file app-index ./config/settings.yaml
```

---

## Integration with Other LEANN Commands

These commands work seamlessly with existing LEANN operations:

```bash
# Build initial index
leann build my-docs --docs ./documents --no-compact

# Update incrementally
leann sync my-docs --docs ./documents

# Search updated index
leann search my-docs "your query"

# Ask questions with updated data
leann ask my-docs "your question" --interactive

# Verify index contents
leann list
```

---

## Best Practices

1. **Use `--no-compact` from the start** if you plan to use incremental updates
2. **Use `sync` for routine updates** - it's automatic and efficient
3. **Use `update-file` for quick single-file fixes** during development
4. **Use `remove-file` when deleting files** to keep index clean
5. **Periodically rebuild** indexes completely to ensure consistency
6. **Test with small directories first** before syncing large codebases
7. **Use file type filters** to focus on relevant files and improve performance

---

## See Also

- `leann build --help` - Build initial indexes
- `leann update --help` - Add new documents to existing index
- `leann list` - Show all available indexes
- `leann search --help` - Search in indexes
