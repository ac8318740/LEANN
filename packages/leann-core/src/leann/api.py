"""
This file contains the core API for the LEANN project, now definitively updated
with the correct, original embedding logic from the user's reference code.
"""

import json
import logging
import os
import pickle
import re
import subprocess
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Literal, Optional, Union

import numpy as np
from leann_backend_hnsw.convert_to_csr import prune_hnsw_embeddings_inplace

from leann.interactive_utils import create_api_session
from leann.interface import LeannBackendSearcherInterface

from .chat import get_llm
from .embedding_server_manager import EmbeddingServerManager
from .interface import LeannBackendFactoryInterface
from .metadata_filter import MetadataFilterEngine
from .registry import BACKEND_REGISTRY

logger = logging.getLogger(__name__)


def get_registered_backends() -> list[str]:
    """Get list of registered backend names."""
    return list(BACKEND_REGISTRY.keys())


def compute_embeddings(
    chunks: list[str],
    model_name: str,
    mode: str = "sentence-transformers",
    use_server: bool = True,
    port: Optional[int] = None,
    is_build=False,
    provider_options: Optional[dict[str, Any]] = None,
) -> np.ndarray:
    """
    Computes embeddings using different backends.

    Args:
        chunks: List of text chunks to embed
        model_name: Name of the embedding model
        mode: Embedding backend mode. Options:
            - "sentence-transformers": Use sentence-transformers library (default)
            - "mlx": Use MLX backend for Apple Silicon
            - "openai": Use OpenAI embedding API
            - "gemini": Use Google Gemini embedding API
        use_server: Whether to use embedding server (True for search, False for build)

    Returns:
        numpy array of embeddings
    """
    if use_server:
        # Use embedding server (for search/query)
        if port is None:
            raise ValueError("port is required when use_server is True")
        return compute_embeddings_via_server(chunks, model_name, port=port)
    else:
        # Use direct computation (for build_index)
        from .embedding_compute import (
            compute_embeddings as compute_embeddings_direct,
        )

        return compute_embeddings_direct(
            chunks,
            model_name,
            mode=mode,
            is_build=is_build,
            provider_options=provider_options,
        )


def compute_embeddings_via_server(chunks: list[str], model_name: str, port: int) -> np.ndarray:
    """Computes embeddings using sentence-transformers.

    Args:
        chunks: List of text chunks to embed
        model_name: Name of the sentence transformer model
    """
    logger.info(
        f"Computing embeddings for {len(chunks)} chunks using SentenceTransformer model '{model_name}' (via embedding server)..."
    )
    import msgpack
    import numpy as np
    import zmq

    # Connect to embedding server
    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    socket.connect(f"tcp://localhost:{port}")

    # Send chunks to server for embedding computation
    request = chunks
    socket.send(msgpack.packb(request))

    # Receive embeddings from server
    response = socket.recv()
    embeddings_list = msgpack.unpackb(response)

    # Convert back to numpy array
    embeddings = np.array(embeddings_list, dtype=np.float32)

    socket.close()
    context.term()

    return embeddings


@dataclass
class SearchResult:
    id: str
    score: float
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


class PassageManager:
    def __init__(
        self, passage_sources: list[dict[str, Any]], metadata_file_path: Optional[str] = None
    ):
        self.offset_maps: dict[str, dict[str, int]] = {}
        self.passage_files: dict[str, str] = {}
        # Avoid materializing a single gigantic global map to reduce memory
        # footprint on very large corpora (e.g., 60M+ passages). Instead, keep
        # per-shard maps and do a lightweight per-shard lookup on demand.
        self._total_count: int = 0
        self.filter_engine = MetadataFilterEngine()  # Initialize filter engine

        # Derive index base name for standard sibling fallbacks, e.g., <index_name>.passages.*
        index_name_base = None
        if metadata_file_path:
            meta_name = Path(metadata_file_path).name
            if meta_name.endswith(".meta.json"):
                index_name_base = meta_name[: -len(".meta.json")]

        for source in passage_sources:
            assert source["type"] == "jsonl", "only jsonl is supported"
            passage_file = source.get("path", "")
            index_file = source.get("index_path", "")  # .idx file

            # Fix path resolution - relative paths should be relative to metadata file directory
            def _resolve_candidates(
                primary: str,
                relative_key: str,
                default_name: Optional[str],
                source_dict: dict[str, Any],
            ) -> list[Path]:
                """
                Build an ordered list of candidate paths. For relative paths specified in
                metadata, prefer resolution relative to the metadata file directory first,
                then fall back to CWD-based resolution, and finally to conventional
                sibling defaults (e.g., <index_base>.passages.idx / .jsonl).
                """
                candidates: list[Path] = []
                # 1) Primary path
                if primary:
                    p = Path(primary)
                    if p.is_absolute():
                        candidates.append(p)
                    else:
                        # Prefer metadata-relative resolution for relative paths
                        if metadata_file_path:
                            candidates.append(Path(metadata_file_path).parent / p)
                        # Also consider CWD-relative as a fallback for legacy layouts
                        candidates.append(Path.cwd() / p)
                # 2) metadata-relative explicit relative key (if present)
                if metadata_file_path and source_dict.get(relative_key):
                    candidates.append(Path(metadata_file_path).parent / source_dict[relative_key])
                # 3) metadata-relative standard sibling filename
                if metadata_file_path and default_name:
                    candidates.append(Path(metadata_file_path).parent / default_name)
                return candidates

            # Build candidate lists and pick first existing; otherwise keep last candidate for error message
            idx_default = f"{index_name_base}.passages.idx" if index_name_base else None
            idx_candidates = _resolve_candidates(
                index_file, "index_path_relative", idx_default, source
            )
            pas_default = f"{index_name_base}.passages.jsonl" if index_name_base else None
            pas_candidates = _resolve_candidates(passage_file, "path_relative", pas_default, source)

            def _pick_existing(cands: list[Path]) -> str:
                for c in cands:
                    if c.exists():
                        return str(c.resolve())
                # Fallback to last candidate (best guess) even if not exists; will error below
                return str(cands[-1].resolve()) if cands else ""

            index_file = _pick_existing(idx_candidates)
            passage_file = _pick_existing(pas_candidates)

            if not Path(index_file).exists():
                raise FileNotFoundError(f"Passage index file not found: {index_file}")

            with open(index_file, "rb") as f:
                offset_map: dict[str, int] = pickle.load(f)
                self.offset_maps[passage_file] = offset_map
                self.passage_files[passage_file] = passage_file
                self._total_count += len(offset_map)

    def get_passage(self, passage_id: str) -> dict[str, Any]:
        # Fast path: check each shard map (there are typically few shards).
        # This avoids building a massive combined dict while keeping lookups
        # bounded by the number of shards.
        for passage_file, offset_map in self.offset_maps.items():
            try:
                offset = offset_map[passage_id]
                with open(passage_file, encoding="utf-8") as f:
                    f.seek(offset)
                    return json.loads(f.readline())
            except KeyError:
                continue
        raise KeyError(f"Passage ID not found: {passage_id}")

    def filter_search_results(
        self,
        search_results: list[SearchResult],
        metadata_filters: Optional[dict[str, dict[str, Union[str, int, float, bool, list]]]],
    ) -> list[SearchResult]:
        """
        Apply metadata filters to search results.

        Args:
            search_results: List of SearchResult objects
            metadata_filters: Filter specifications to apply

        Returns:
            Filtered list of SearchResult objects
        """
        if not metadata_filters:
            return search_results

        logger.debug(f"Applying metadata filters to {len(search_results)} results")

        # Convert SearchResult objects to dictionaries for the filter engine
        result_dicts = []
        for result in search_results:
            result_dicts.append(
                {
                    "id": result.id,
                    "score": result.score,
                    "text": result.text,
                    "metadata": result.metadata,
                }
            )

        # Apply filters using the filter engine
        filtered_dicts = self.filter_engine.apply_filters(result_dicts, metadata_filters)

        # Convert back to SearchResult objects
        filtered_results = []
        for result_dict in filtered_dicts:
            filtered_results.append(
                SearchResult(
                    id=result_dict["id"],
                    score=result_dict["score"],
                    text=result_dict["text"],
                    metadata=result_dict["metadata"],
                )
            )

        logger.debug(f"Filtered results: {len(filtered_results)} remaining")
        return filtered_results

    def __len__(self) -> int:
        return self._total_count


class LeannBuilder:
    def __init__(
        self,
        backend_name: str,
        embedding_model: str = "facebook/contriever",
        dimensions: Optional[int] = None,
        embedding_mode: str = "sentence-transformers",
        embedding_options: Optional[dict[str, Any]] = None,
        **backend_kwargs,
    ):
        self.backend_name = backend_name
        # Normalize incompatible combinations early (for consistent metadata)
        if backend_name == "hnsw":
            is_recompute = backend_kwargs.get("is_recompute", True)
            is_compact = backend_kwargs.get("is_compact", True)
            if is_recompute is False and is_compact is True:
                warnings.warn(
                    "HNSW with is_recompute=False requires non-compact storage. Forcing is_compact=False.",
                    UserWarning,
                    stacklevel=2,
                )
                backend_kwargs["is_compact"] = False

        backend_factory: Optional[LeannBackendFactoryInterface] = BACKEND_REGISTRY.get(backend_name)
        if backend_factory is None:
            raise ValueError(f"Backend '{backend_name}' not found or not registered.")
        self.backend_factory = backend_factory
        self.embedding_model = embedding_model
        self.dimensions = dimensions
        self.embedding_mode = embedding_mode
        self.embedding_options = embedding_options or {}

        # Check if we need to use cosine distance for normalized embeddings
        normalized_embeddings_models = {
            # OpenAI models
            ("openai", "text-embedding-ada-002"),
            ("openai", "text-embedding-3-small"),
            ("openai", "text-embedding-3-large"),
            # Voyage AI models
            ("voyage", "voyage-2"),
            ("voyage", "voyage-3"),
            ("voyage", "voyage-large-2"),
            ("voyage", "voyage-multilingual-2"),
            ("voyage", "voyage-code-2"),
            # Cohere models
            ("cohere", "embed-english-v3.0"),
            ("cohere", "embed-multilingual-v3.0"),
            ("cohere", "embed-english-light-v3.0"),
            ("cohere", "embed-multilingual-light-v3.0"),
        }

        # Also check for patterns in model names
        is_normalized = False
        current_model_lower = embedding_model.lower()
        current_mode_lower = embedding_mode.lower()

        # Check exact matches
        for mode, model in normalized_embeddings_models:
            if (current_mode_lower == mode and current_model_lower == model) or (
                mode in current_mode_lower and model in current_model_lower
            ):
                is_normalized = True
                break

        # Check patterns
        if not is_normalized:
            # OpenAI patterns
            if "openai" in current_mode_lower or "openai" in current_model_lower:
                if any(
                    pattern in current_model_lower
                    for pattern in ["text-embedding", "ada", "3-small", "3-large"]
                ):
                    is_normalized = True
            # Voyage patterns
            elif "voyage" in current_mode_lower or "voyage" in current_model_lower:
                is_normalized = True
            # Cohere patterns
            elif "cohere" in current_mode_lower or "cohere" in current_model_lower:
                if "embed" in current_model_lower:
                    is_normalized = True

        # Handle distance metric
        if is_normalized and "distance_metric" not in backend_kwargs:
            backend_kwargs["distance_metric"] = "cosine"
            warnings.warn(
                f"Detected normalized embeddings model '{embedding_model}' with mode '{embedding_mode}'. "
                f"Automatically setting distance_metric='cosine' for optimal performance. "
                f"Normalized embeddings (L2 norm = 1) should use cosine similarity instead of MIPS.",
                UserWarning,
                stacklevel=2,
            )
        elif is_normalized and backend_kwargs.get("distance_metric", "").lower() != "cosine":
            current_metric = backend_kwargs.get("distance_metric", "mips")
            warnings.warn(
                f"Warning: Using '{current_metric}' distance metric with normalized embeddings model "
                f"'{embedding_model}' may lead to suboptimal search results. "
                f"Consider using 'cosine' distance metric for better performance.",
                UserWarning,
                stacklevel=2,
            )

        self.backend_kwargs = backend_kwargs
        self.chunks: list[dict[str, Any]] = []

    def add_text(self, text: str, metadata: Optional[dict[str, Any]] = None):
        if metadata is None:
            metadata = {}
        passage_id = metadata.get("id", str(len(self.chunks)))
        chunk_data = {"id": passage_id, "text": text, "metadata": metadata}
        self.chunks.append(chunk_data)

    def build_index(self, index_path: str):
        if not self.chunks:
            raise ValueError("No chunks added.")

        # Filter out invalid/empty text chunks early to keep passage and embedding counts aligned
        valid_chunks: list[dict[str, Any]] = []
        skipped = 0
        for chunk in self.chunks:
            text = chunk.get("text", "")
            if isinstance(text, str) and text.strip():
                valid_chunks.append(chunk)
            else:
                skipped += 1
        if skipped > 0:
            print(
                f"Warning: Skipping {skipped} empty/invalid text chunk(s). Processing {len(valid_chunks)} valid chunks"
            )
            self.chunks = valid_chunks
            if not self.chunks:
                raise ValueError("All provided chunks are empty or invalid. Nothing to index.")
        if self.dimensions is None:
            self.dimensions = len(
                compute_embeddings(
                    ["dummy"],
                    self.embedding_model,
                    self.embedding_mode,
                    use_server=False,
                    provider_options=self.embedding_options,
                )[0]
            )
        path = Path(index_path)
        index_dir = path.parent
        index_name = path.name
        index_dir.mkdir(parents=True, exist_ok=True)
        passages_file = index_dir / f"{index_name}.passages.jsonl"
        offset_file = index_dir / f"{index_name}.passages.idx"
        offset_map = {}
        with open(passages_file, "w", encoding="utf-8") as f:
            try:
                from tqdm import tqdm

                chunk_iterator = tqdm(self.chunks, desc="Writing passages", unit="chunk")
            except ImportError:
                chunk_iterator = self.chunks

            for chunk in chunk_iterator:
                offset = f.tell()
                json.dump(
                    {
                        "id": chunk["id"],
                        "text": chunk["text"],
                        "metadata": chunk["metadata"],
                    },
                    f,
                    ensure_ascii=False,
                )
                f.write("\n")
                offset_map[chunk["id"]] = offset
        with open(offset_file, "wb") as f:
            pickle.dump(offset_map, f)
        texts_to_embed = [c["text"] for c in self.chunks]
        embeddings = compute_embeddings(
            texts_to_embed,
            self.embedding_model,
            self.embedding_mode,
            use_server=False,
            is_build=True,
            provider_options=self.embedding_options,
        )
        string_ids = [chunk["id"] for chunk in self.chunks]
        current_backend_kwargs = {**self.backend_kwargs, "dimensions": self.dimensions}
        builder_instance = self.backend_factory.builder(**current_backend_kwargs)
        builder_instance.build(embeddings, string_ids, index_path, **current_backend_kwargs)

        # Persist ID map AFTER backend.build() to ensure it's authoritative
        # The backend may also write this file, but we overwrite with the correct data
        idmap_file = (
            index_dir
            / f"{index_name[: -len('.leann')] if index_name.endswith('.leann') else index_name}.ids.txt"
        )
        with open(idmap_file, "w", encoding="utf-8") as f:
            for sid in string_ids:
                f.write(str(sid) + "\n")
        logger.info("Wrote %d IDs to %s", len(string_ids), idmap_file)
        leann_meta_path = index_dir / f"{index_name}.meta.json"
        meta_data = {
            "version": "1.0",
            "backend_name": self.backend_name,
            "embedding_model": self.embedding_model,
            "dimensions": self.dimensions,
            "backend_kwargs": self.backend_kwargs,
            "embedding_mode": self.embedding_mode,
            "passage_sources": [
                {
                    "type": "jsonl",
                    # Preserve existing relative file names (backward-compatible)
                    "path": passages_file.name,
                    "index_path": offset_file.name,
                    # Add optional redundant relative keys for remote build portability (non-breaking)
                    "path_relative": passages_file.name,
                    "index_path_relative": offset_file.name,
                }
            ],
        }

        if self.embedding_options:
            meta_data["embedding_options"] = self.embedding_options

        # Add storage status flags for HNSW backend
        if self.backend_name == "hnsw":
            is_compact = self.backend_kwargs.get("is_compact", True)
            is_recompute = self.backend_kwargs.get("is_recompute", True)
            meta_data["is_compact"] = is_compact
            meta_data["is_pruned"] = bool(is_recompute)

        # Track total passages for sync/update operations
        meta_data["total_passages"] = len(self.chunks)

        with open(leann_meta_path, "w", encoding="utf-8") as f:
            json.dump(meta_data, f, indent=2)

    def build_index_from_embeddings(self, index_path: str, embeddings_file: str):
        """
        Build an index from pre-computed embeddings stored in a pickle file.

        Args:
            index_path: Path where the index will be saved
            embeddings_file: Path to pickle file containing (ids, embeddings) tuple
        """
        # Load pre-computed embeddings
        with open(embeddings_file, "rb") as f:
            data = pickle.load(f)

        if not isinstance(data, tuple) or len(data) != 2:
            raise ValueError(
                f"Invalid embeddings file format. Expected tuple with 2 elements, got {type(data)}"
            )

        ids, embeddings = data

        if not isinstance(embeddings, np.ndarray):
            raise ValueError(f"Expected embeddings to be numpy array, got {type(embeddings)}")

        if len(ids) != embeddings.shape[0]:
            raise ValueError(
                f"Mismatch between number of IDs ({len(ids)}) and embeddings ({embeddings.shape[0]})"
            )

        # Validate/set dimensions
        embedding_dim = embeddings.shape[1]
        if self.dimensions is None:
            self.dimensions = embedding_dim
        elif self.dimensions != embedding_dim:
            raise ValueError(f"Dimension mismatch: expected {self.dimensions}, got {embedding_dim}")

        logger.info(
            f"Building index from precomputed embeddings: {len(ids)} items, {embedding_dim} dimensions"
        )

        # Ensure we have text data for each embedding
        if len(self.chunks) != len(ids):
            # If no text chunks provided, create placeholder text entries
            if not self.chunks:
                logger.info("No text chunks provided, creating placeholder entries...")
                for id_val in ids:
                    self.add_text(
                        f"Document {id_val}",
                        metadata={"id": str(id_val), "from_embeddings": True},
                    )
            else:
                raise ValueError(
                    f"Number of text chunks ({len(self.chunks)}) doesn't match number of embeddings ({len(ids)})"
                )

        # Build file structure
        path = Path(index_path)
        index_dir = path.parent
        index_name = path.name
        index_dir.mkdir(parents=True, exist_ok=True)
        passages_file = index_dir / f"{index_name}.passages.jsonl"
        offset_file = index_dir / f"{index_name}.passages.idx"

        # Write passages and create offset map
        offset_map = {}
        with open(passages_file, "w", encoding="utf-8") as f:
            for chunk in self.chunks:
                offset = f.tell()
                json.dump(
                    {
                        "id": chunk["id"],
                        "text": chunk["text"],
                        "metadata": chunk["metadata"],
                    },
                    f,
                    ensure_ascii=False,
                )
                f.write("\n")
                offset_map[chunk["id"]] = offset

        with open(offset_file, "wb") as f:
            pickle.dump(offset_map, f)

        # Build the vector index using precomputed embeddings
        string_ids = [str(id_val) for id_val in ids]
        current_backend_kwargs = {**self.backend_kwargs, "dimensions": self.dimensions}
        builder_instance = self.backend_factory.builder(**current_backend_kwargs)
        builder_instance.build(embeddings, string_ids, index_path)

        # Persist ID map AFTER backend.build() to ensure it's authoritative
        idmap_file = (
            index_dir
            / f"{index_name[: -len('.leann')] if index_name.endswith('.leann') else index_name}.ids.txt"
        )
        with open(idmap_file, "w", encoding="utf-8") as f:
            for sid in string_ids:
                f.write(str(sid) + "\n")
        logger.info("Wrote %d IDs to %s", len(string_ids), idmap_file)

        # Create metadata file
        leann_meta_path = index_dir / f"{index_name}.meta.json"
        meta_data = {
            "version": "1.0",
            "backend_name": self.backend_name,
            "embedding_model": self.embedding_model,
            "dimensions": self.dimensions,
            "backend_kwargs": self.backend_kwargs,
            "embedding_mode": self.embedding_mode,
            "passage_sources": [
                {
                    "type": "jsonl",
                    # Preserve existing relative file names (backward-compatible)
                    "path": passages_file.name,
                    "index_path": offset_file.name,
                    # Add optional redundant relative keys for remote build portability (non-breaking)
                    "path_relative": passages_file.name,
                    "index_path_relative": offset_file.name,
                }
            ],
            "built_from_precomputed_embeddings": True,
            "embeddings_source": str(embeddings_file),
        }

        if self.embedding_options:
            meta_data["embedding_options"] = self.embedding_options

        # Add storage status flags for HNSW backend
        if self.backend_name == "hnsw":
            is_compact = self.backend_kwargs.get("is_compact", True)
            is_recompute = self.backend_kwargs.get("is_recompute", True)
            meta_data["is_compact"] = is_compact
            meta_data["is_pruned"] = bool(is_recompute)

        # Track total passages for sync/update operations
        meta_data["total_passages"] = len(self.chunks)

        with open(leann_meta_path, "w", encoding="utf-8") as f:
            json.dump(meta_data, f, indent=2)

        logger.info(f"Index built successfully from precomputed embeddings: {index_path}")

    def update_index(self, index_path: str):
        """Append new passages and vectors to an existing HNSW index."""
        if not self.chunks:
            raise ValueError("No new chunks provided for update.")

        path = Path(index_path)
        index_dir = path.parent
        index_name = path.name
        index_prefix = path.stem

        meta_path = index_dir / f"{index_name}.meta.json"
        passages_file = index_dir / f"{index_name}.passages.jsonl"
        offset_file = index_dir / f"{index_name}.passages.idx"
        index_file = index_dir / f"{index_prefix}.index"

        if not meta_path.exists() or not passages_file.exists() or not offset_file.exists():
            raise FileNotFoundError("Index metadata or passage files are missing; cannot update.")
        if not index_file.exists():
            raise FileNotFoundError(f"HNSW index file not found: {index_file}")

        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)
        backend_name = meta.get("backend_name")
        if backend_name != self.backend_name:
            raise ValueError(
                f"Index was built with backend '{backend_name}', cannot update with '{self.backend_name}'."
            )

        meta_backend_kwargs = meta.get("backend_kwargs", {})
        index_is_compact = meta.get("is_compact", meta_backend_kwargs.get("is_compact", True))
        if index_is_compact:
            raise ValueError(
                "Compact HNSW indices do not support in-place updates. Rebuild required."
            )

        distance_metric = meta_backend_kwargs.get(
            "distance_metric", self.backend_kwargs.get("distance_metric", "mips")
        ).lower()
        needs_recompute = bool(
            meta.get("is_pruned")
            or meta_backend_kwargs.get("is_recompute")
            or self.backend_kwargs.get("is_recompute")
        )

        with open(offset_file, "rb") as f:
            offset_map: dict[str, int] = pickle.load(f)
        existing_ids = set(offset_map.keys())

        valid_chunks: list[dict[str, Any]] = []
        for chunk in self.chunks:
            text = chunk.get("text", "")
            if not isinstance(text, str) or not text.strip():
                continue
            metadata = chunk.setdefault("metadata", {})
            passage_id = chunk.get("id") or metadata.get("id")
            if passage_id and passage_id in existing_ids:
                raise ValueError(f"Passage ID '{passage_id}' already exists in the index.")
            valid_chunks.append(chunk)

        if not valid_chunks:
            raise ValueError("No valid chunks to append.")

        texts_to_embed = [chunk["text"] for chunk in valid_chunks]
        embeddings = compute_embeddings(
            texts_to_embed,
            self.embedding_model,
            self.embedding_mode,
            use_server=False,
            is_build=True,
            provider_options=self.embedding_options,
        )

        embedding_dim = embeddings.shape[1]
        expected_dim = meta.get("dimensions")
        if expected_dim is not None and expected_dim != embedding_dim:
            raise ValueError(
                f"Dimension mismatch during update: existing index uses {expected_dim}, got {embedding_dim}."
            )

        from leann_backend_hnsw import faiss  # type: ignore

        embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)
        if distance_metric == "cosine":
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            norms[norms == 0] = 1
            embeddings = embeddings / norms

        index = faiss.read_index(str(index_file))
        if hasattr(index, "is_recompute"):
            index.is_recompute = needs_recompute
            print(f"index.is_recompute: {index.is_recompute}")
        if getattr(index, "storage", None) is None:
            if index.metric_type == faiss.METRIC_INNER_PRODUCT:
                storage_index = faiss.IndexFlatIP(index.d)
            else:
                storage_index = faiss.IndexFlatL2(index.d)
            index.storage = storage_index
            index.own_fields = True
            # Faiss expects storage.ntotal to reflect the existing graph's
            # population (even if the vectors themselves were pruned from disk
            # for recompute mode).  When we attach a fresh IndexFlat here its
            # ntotal starts at zero, which later causes IndexHNSW::add to
            # believe new "preset" levels were provided and trips the
            # `n0 + n == levels.size()` assertion.  Seed the temporary storage
            # with the current ntotal so Faiss maintains the proper offset for
            # incoming vectors.
            try:
                storage_index.ntotal = index.ntotal
            except AttributeError:
                # Older Faiss builds may not expose ntotal as a writable
                # attribute; in that case we fall back to the default behaviour.
                pass
        if index.d != embedding_dim:
            raise ValueError(
                f"Existing index dimension ({index.d}) does not match new embeddings ({embedding_dim})."
            )

        passage_meta_mode = meta.get("embedding_mode", self.embedding_mode)
        passage_provider_options = meta.get("embedding_options", self.embedding_options)

        # Assign IDs to chunks that don't already have one
        # Preserve existing IDs (e.g., UUIDs from update_file) to avoid overwrites
        base_id = index.ntotal
        for offset, chunk in enumerate(valid_chunks):
            existing_id = chunk.get("id") or chunk.get("metadata", {}).get("id")
            if existing_id:
                # Preserve existing ID (e.g., UUID from update_file)
                chunk["id"] = existing_id
                chunk.setdefault("metadata", {})["id"] = existing_id
            else:
                # Assign new sequential ID only if no ID exists
                new_id = str(base_id + offset)
                chunk.setdefault("metadata", {})["id"] = new_id
                chunk["id"] = new_id

        # Append passages/offsets before we attempt index.add so the ZMQ server
        # can resolve newly assigned IDs during recompute. Keep rollback hooks
        # so we can restore files if the update fails mid-way.
        rollback_passages_size = passages_file.stat().st_size if passages_file.exists() else 0
        offset_map_backup = offset_map.copy()

        try:
            with open(passages_file, "a", encoding="utf-8") as f:
                for chunk in valid_chunks:
                    offset = f.tell()
                    json.dump(
                        {
                            "id": chunk["id"],
                            "text": chunk["text"],
                            "metadata": chunk.get("metadata", {}),
                        },
                        f,
                        ensure_ascii=False,
                    )
                    f.write("\n")
                    offset_map[chunk["id"]] = offset

            with open(offset_file, "wb") as f:
                pickle.dump(offset_map, f)

            server_manager: Optional[EmbeddingServerManager] = None
            server_started = False
            requested_zmq_port = int(os.getenv("LEANN_UPDATE_ZMQ_PORT", "5557"))

            try:
                if needs_recompute:
                    server_manager = EmbeddingServerManager(
                        backend_module_name="leann_backend_hnsw.hnsw_embedding_server"
                    )
                    server_started, actual_port = server_manager.start_server(
                        port=requested_zmq_port,
                        model_name=self.embedding_model,
                        embedding_mode=passage_meta_mode,
                        passages_file=str(meta_path),
                        distance_metric=distance_metric,
                        provider_options=passage_provider_options,
                    )
                    if not server_started:
                        raise RuntimeError(
                            "Failed to start HNSW embedding server for recompute update."
                        )
                    if actual_port != requested_zmq_port:
                        logger.warning(
                            "Embedding server started on port %s instead of requested %s. "
                            "Using reassigned port.",
                            actual_port,
                            requested_zmq_port,
                        )
                    if hasattr(index.hnsw, "set_zmq_port"):
                        index.hnsw.set_zmq_port(actual_port)
                    elif hasattr(index, "set_zmq_port"):
                        index.set_zmq_port(actual_port)

                if needs_recompute:
                    for i in range(embeddings.shape[0]):
                        print(f"add {i} embeddings")
                        index.add(1, faiss.swig_ptr(embeddings[i : i + 1]))
                else:
                    index.add(embeddings.shape[0], faiss.swig_ptr(embeddings))
                faiss.write_index(index, str(index_file))

                # Update IDs file with new chunk IDs
                # Read existing IDs and append new ones
                ids_file = (
                    index_dir
                    / f"{index_name[: -len('.leann')] if index_name.endswith('.leann') else index_name}.ids.txt"
                )
                existing_ids = []
                if ids_file.exists():
                    with open(ids_file, encoding="utf-8") as f:
                        existing_ids = [line.strip() for line in f if line.strip()]
                # Append new chunk IDs
                new_ids = [chunk["id"] for chunk in valid_chunks]
                all_ids = existing_ids + new_ids
                with open(ids_file, "w", encoding="utf-8") as f:
                    for id_str in all_ids:
                        f.write(str(id_str) + "\n")
                logger.info("Updated IDs file with %d new IDs (total: %d)", len(new_ids), len(all_ids))
            finally:
                if server_started and server_manager is not None:
                    server_manager.stop_server()

        except Exception:
            # Roll back appended passages/offset map to keep files consistent.
            if passages_file.exists():
                with open(passages_file, "rb+") as f:
                    f.truncate(rollback_passages_size)
            offset_map = offset_map_backup
            with open(offset_file, "wb") as f:
                pickle.dump(offset_map, f)
            raise

        meta["total_passages"] = len(offset_map)
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        logger.info(
            "Appended %d passages to index '%s'. New total: %d",
            len(valid_chunks),
            index_path,
            len(offset_map),
        )

        self.chunks.clear()

        if needs_recompute:
            prune_hnsw_embeddings_inplace(str(index_file))

    def delete_by_file(self, index_path: str, file_path: str) -> int:
        """
        Remove all chunks originating from a specific file by rebuilding the index.

        Since FAISS HNSW doesn't support in-place deletion, this method rebuilds
        the index with only the chunks that should be kept.

        Args:
            index_path: Path to the LEANN index (e.g., "my_index.leann")
            file_path: Exact path of source file whose chunks should be removed.
                       Will be normalized to absolute path for comparison.

        Returns:
            Number of chunks deleted

        Raises:
            ValueError: If index uses compact format (not supported)
            FileNotFoundError: If index files are missing
        """
        # 1. Load and validate index metadata
        path = Path(index_path)
        index_dir = path.parent
        index_name = path.name
        index_prefix = path.stem

        meta_path = index_dir / f"{index_name}.meta.json"
        passages_file = index_dir / f"{index_name}.passages.jsonl"
        offset_file = index_dir / f"{index_name}.passages.idx"
        index_file = index_dir / f"{index_prefix}.index"
        ids_file = (
            index_dir
            / f"{index_name[: -len('.leann')] if index_name.endswith('.leann') else index_name}.ids.txt"
        )

        if not meta_path.exists() or not passages_file.exists() or not offset_file.exists():
            raise FileNotFoundError("Index metadata or passage files are missing; cannot delete.")
        if not index_file.exists():
            raise FileNotFoundError(f"Index file not found: {index_file}")

        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)

        backend_name = meta.get("backend_name")
        if backend_name != "hnsw":
            raise ValueError(
                f"delete_by_file() currently only supports HNSW backend, got '{backend_name}'."
            )

        meta_backend_kwargs = meta.get("backend_kwargs", {})
        index_is_compact = meta.get("is_compact", meta_backend_kwargs.get("is_compact", True))
        if index_is_compact:
            raise ValueError(
                "Compact HNSW indices do not support deletion. Rebuild with is_compact=False."
            )

        # Check if embeddings are stored (needed for reconstruction)
        is_pruned = bool(meta.get("is_pruned"))
        is_recompute = bool(meta_backend_kwargs.get("is_recompute"))
        if is_pruned or is_recompute:
            raise ValueError(
                "Cannot delete from indices with pruned embeddings (is_pruned=True or is_recompute=True). "
                "Embeddings must be stored in the index for reconstruction during rebuild."
            )

        # 2. Find chunks to delete
        with open(offset_file, "rb") as f:
            offset_map: dict[str, int] = pickle.load(f)

        # Normalize target file path for comparison
        target_path = Path(file_path).resolve()

        chunks_to_keep: list[tuple[str, dict[str, Any]]] = []
        chunks_to_delete_set: set[str] = set()

        with open(passages_file, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                chunk = json.loads(line)
                chunk_id = chunk.get("id", "")
                metadata = chunk.get("metadata", {})

                # Check both file_path and source fields
                stored_path = metadata.get("file_path") or metadata.get("source", "")

                # Normalize stored path for comparison
                should_delete = False
                if stored_path:
                    try:
                        normalized_stored = Path(stored_path).resolve()
                        if normalized_stored == target_path:
                            should_delete = True
                    except (OSError, ValueError):
                        # Invalid path format, skip
                        pass

                if should_delete:
                    chunks_to_delete_set.add(chunk_id)
                else:
                    chunks_to_keep.append((chunk_id, chunk))

        # 3. Early return if no chunks match or if all chunks would be deleted
        if not chunks_to_delete_set:
            logger.info("No chunks found for file '%s' in index '%s'.", file_path, index_path)
            return 0

        if not chunks_to_keep:
            raise ValueError(
                f"All {len(chunks_to_delete_set)} chunks in the index are from file '{file_path}'. "
                "Cannot delete all chunks - this would leave an empty index."
            )

        logger.info(
            "Found %d chunks to delete from file '%s' in index '%s'. Rebuilding index with %d kept chunks.",
            len(chunks_to_delete_set),
            file_path,
            index_path,
            len(chunks_to_keep),
        )

        # 4. Load FAISS index and ID mapping
        from leann_backend_hnsw import faiss  # type: ignore

        index = faiss.read_index(str(index_file))

        original_id_list: list[str] = []
        if ids_file.exists():
            with open(ids_file, encoding="utf-8") as f:
                original_id_list = [line.strip() for line in f if line.strip()]
        else:
            raise FileNotFoundError(
                f"ID mapping file not found: {ids_file}. "
                "Cannot safely delete without reliable ID mapping. "
                "Rebuild the index with a newer LEANN version that creates .ids.txt files."
            )

        # Build mapping from string ID to FAISS integer ID
        id_to_faiss_idx = {str_id: idx for idx, str_id in enumerate(original_id_list)}

        # 5. Reconstruct embeddings for kept chunks
        logger.info("Reconstructing embeddings for %d kept chunks...", len(chunks_to_keep))

        kept_embeddings = []
        kept_string_ids = []
        embedding_dim = index.d

        for chunk_id, _ in chunks_to_keep:
            if chunk_id not in id_to_faiss_idx:
                logger.warning("Chunk ID '%s' not found in ID mapping, skipping.", chunk_id)
                continue

            faiss_id = id_to_faiss_idx[chunk_id]
            embedding = np.zeros(embedding_dim, dtype=np.float32)

            try:
                index.reconstruct(faiss_id, faiss.swig_ptr(embedding))
                kept_embeddings.append(embedding)
                kept_string_ids.append(chunk_id)
            except Exception as e:
                raise RuntimeError(
                    f"Failed to reconstruct embedding for chunk '{chunk_id}' (FAISS ID {faiss_id}). "
                    f"This typically means embeddings were pruned from the index. Error: {e}"
                ) from e

        if not kept_embeddings:
            raise ValueError("No embeddings could be reconstructed. Cannot rebuild index.")

        kept_embeddings_array = np.array(kept_embeddings, dtype=np.float32)
        logger.info("Successfully reconstructed %d embeddings.", len(kept_embeddings_array))

        # 6. Build new index with kept chunks
        logger.info("Building new index with kept chunks...")

        # Get backend kwargs from metadata (not self.backend_kwargs)
        backend_kwargs_for_rebuild = meta_backend_kwargs.copy()
        backend_kwargs_for_rebuild["dimensions"] = embedding_dim
        backend_kwargs_for_rebuild["is_compact"] = meta.get(
            "is_compact", meta_backend_kwargs.get("is_compact", False)
        )
        backend_kwargs_for_rebuild["is_recompute"] = meta.get(
            "is_pruned", meta_backend_kwargs.get("is_recompute", False)
        )

        # Get backend factory
        backend_factory = BACKEND_REGISTRY.get(backend_name)
        if backend_factory is None:
            raise ValueError(f"Backend '{backend_name}' not found in registry.")

        # Build new index to temporary location first
        # The backend's build() uses Path(index_path).stem to get the prefix, then appends .index/.ids.txt
        # Path.stem only removes the LAST extension, so:
        #   - If we pass "documents.index.rebuild", stem = "documents.index"
        #   - Backend creates: "documents.index.index" and "documents.index.ids.txt"
        # We use "_rebuild" (underscore) to avoid the stem stripping issue
        temp_prefix = str(index_file).replace(".index", "_rebuild")
        temp_index_actual = index_dir / f"{Path(temp_prefix).stem}.index"
        temp_ids_path = index_dir / f"{Path(temp_prefix).stem}.ids.txt"

        builder_instance = backend_factory.builder(**backend_kwargs_for_rebuild)
        builder_instance.build(
            kept_embeddings_array, kept_string_ids, temp_prefix, **backend_kwargs_for_rebuild
        )

        # Explicitly write IDs file - don't rely on backend's error-swallowing code
        # The backend has silent exception handling that can leave the file incomplete
        with open(temp_ids_path, "w", encoding="utf-8") as f:
            for kept_id in kept_string_ids:
                f.write(str(kept_id) + "\n")

        logger.info("Successfully built new index with %d chunks.", len(kept_string_ids))

        # 7. Rewrite passages and metadata atomically
        passages_file_tmp = passages_file.with_suffix(".jsonl.tmp")
        offset_file_tmp = offset_file.with_suffix(".idx.tmp")

        try:
            # Write new passages.jsonl with kept chunks only
            new_offset_map: dict[str, int] = {}
            with open(passages_file_tmp, "w", encoding="utf-8") as f:
                for chunk_id, chunk in chunks_to_keep:
                    if chunk_id in kept_string_ids:  # Only write chunks that were successfully reconstructed
                        offset = f.tell()
                        json.dump(chunk, f, ensure_ascii=False)
                        f.write("\n")
                        new_offset_map[chunk_id] = offset

            # Write new passages.idx
            with open(offset_file_tmp, "wb") as f:
                pickle.dump(new_offset_map, f)

            # Atomic renames to replace original files
            # First rename index and IDs files (these were built to temp location)
            temp_index_actual.replace(index_file)
            temp_ids_path.replace(ids_file)

            # Then rename passages files
            passages_file_tmp.replace(passages_file)
            offset_file_tmp.replace(offset_file)

        except Exception as e:
            # Cleanup temp files on failure
            for tmp_file in [passages_file_tmp, offset_file_tmp, temp_index_actual, temp_ids_path]:
                if tmp_file.exists():
                    tmp_file.unlink()
            raise RuntimeError(f"Failed to rewrite index files: {e}") from e

        # 8. Update metadata
        meta["total_passages"] = len(new_offset_map)
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        deleted_count = len(chunks_to_delete_set)
        logger.info(
            "Deleted %d chunks from file '%s' in index '%s'. Remaining: %d",
            deleted_count,
            file_path,
            index_path,
            len(new_offset_map),
        )

        return deleted_count

    def update_file(self, index_path: str, file_path: str) -> tuple[int, int]:
        """
        Re-index a single file by removing old chunks and adding new ones.

        This is the primary method for handling file edits - it removes stale
        chunks from the previous version and indexes the current version.

        Args:
            index_path: Path to the LEANN index
            file_path: Path to the file to re-index

        Returns:
            Tuple of (chunks_deleted, chunks_added)

        Raises:
            ValueError: If index uses compact format
            FileNotFoundError: If index or file not found
        """
        # Step 1: Delete old chunks
        deleted_count = self.delete_by_file(index_path, file_path)

        # Clear any stale chunks from previous operations
        self.chunks.clear()

        # Step 2: Check if file still exists
        file_path_obj = Path(file_path)
        if not file_path_obj.exists():
            logger.info("File '%s' was deleted. Removed %d chunks.", file_path, deleted_count)
            return (deleted_count, 0)

        # Step 3: Load index metadata for chunking settings
        path = Path(index_path)
        index_dir = path.parent
        index_name = path.name

        meta_path = index_dir / f"{index_name}.meta.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"Index metadata not found: {meta_path}")

        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)

        # Extract embedding configuration and sync to self for update_index()
        embedding_model = meta.get("embedding_model")
        embedding_mode = meta.get("embedding_mode", "sentence-transformers")
        embedding_options = meta.get("embedding_options", {})

        # Sync embedding settings to self so update_index() uses the correct model
        if embedding_model:
            self.embedding_model = embedding_model
        self.embedding_mode = embedding_mode
        self.embedding_options = embedding_options

        # Step 4: Load and chunk the file
        logger.info("Loading file '%s' for re-indexing...", file_path)

        try:
            from llama_index.core import SimpleDirectoryReader

            # Load the single file
            documents = SimpleDirectoryReader(
                input_files=[str(file_path_obj.resolve())], filename_as_id=True
            ).load_data()

            if not documents:
                logger.warning("Could not load file '%s', no chunks to add.", file_path)
                return (deleted_count, 0)

        except Exception as e:
            logger.error("Failed to load file '%s': %s", file_path, e)
            raise RuntimeError(f"Failed to load file '{file_path}': {e}") from e

        # Step 5: Chunk the file using appropriate settings
        # Determine if this is a code file
        code_file_exts = {
            ".py",
            ".js",
            ".ts",
            ".jsx",
            ".tsx",
            ".java",
            ".cpp",
            ".c",
            ".h",
            ".hpp",
            ".cs",
            ".go",
            ".rs",
            ".rb",
            ".php",
            ".swift",
            ".kt",
            ".scala",
        }
        is_code_file = file_path_obj.suffix.lower() in code_file_exts

        # Use code-optimized chunking settings for code files, doc settings for others
        # These are reasonable defaults matching CLI behavior
        if is_code_file:
            chunk_size = 512
            chunk_overlap = 50
        else:
            chunk_size = 256
            chunk_overlap = 128

        from llama_index.core.node_parser import SentenceSplitter

        node_parser = SentenceSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separator="\n" if is_code_file else " ",
            paragraph_separator="\n\n",
        )

        import uuid

        new_chunks: list[dict[str, Any]] = []
        for doc in documents:
            nodes = node_parser.get_nodes_from_documents([doc])
            for node in nodes:
                # Generate unique ID for each chunk to avoid conflicts with existing chunks
                chunk_metadata = {
                    "id": str(uuid.uuid4()),
                    "file_path": str(file_path_obj.resolve()),
                    "file_name": file_path_obj.name,
                }

                # Add optional metadata if available
                if "creation_date" in doc.metadata:
                    chunk_metadata["creation_date"] = doc.metadata["creation_date"]
                if "last_modified_date" in doc.metadata:
                    chunk_metadata["last_modified_date"] = doc.metadata["last_modified_date"]

                new_chunks.append({"text": node.get_content(), "metadata": chunk_metadata})

        if not new_chunks:
            logger.warning("No chunks created from file '%s'", file_path)
            return (deleted_count, 0)

        logger.info("Created %d new chunks from file '%s'", len(new_chunks), file_path)

        # Step 6: Add new chunks using existing builder's add_text method
        for chunk in new_chunks:
            self.add_text(chunk["text"], metadata=chunk["metadata"])

        # Step 7: Update the index (appends to existing index)
        try:
            self.update_index(index_path)
            added_count = len(new_chunks)
            logger.info(
                "Updated file '%s': deleted %d, added %d chunks",
                file_path,
                deleted_count,
                added_count,
            )
            return (deleted_count, added_count)
        except Exception as e:
            logger.error("Failed to update index after chunking file '%s': %s", file_path, e)
            raise RuntimeError(f"Failed to update index: {e}") from e

    @staticmethod
    def _compute_file_hash(file_path: Path) -> str:
        """Compute SHA256 hash of file contents."""
        import hashlib

        sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256.update(chunk)
        return sha256.hexdigest()

    def sync_index(
        self,
        index_path: str,
        docs_path: str,
        file_filter: Optional[Callable[[str], bool]] = None,
    ) -> dict[str, int]:
        """
        Sync an index with a directory, detecting and handling file changes.

        Uses content hashes to detect:
        - New files: Added to index
        - Modified files: Old chunks deleted, new chunks added
        - Deleted files: Chunks removed from index

        Note: The file_filter is applied to the current directory scan. Files that
        previously passed the filter but no longer do will be treated as deleted.
        This means changing the filter will cause previously-indexed files to be
        removed if they don't match the new filter.

        Args:
            index_path: Path to the LEANN index
            docs_path: Path to the directory to sync
            file_filter: Optional function to filter files (returns True to include).
                        Applied to absolute file paths.

        Returns:
            Dict with counts: {"added": N, "modified": N, "deleted": N,
                              "unchanged": N, "chunks_added": N, "chunks_deleted": N}

        Raises:
            ValueError: If index uses compact format or has pruned embeddings
            FileNotFoundError: If index or directory not found
        """
        # 1. Validate index exists and is compatible
        path = Path(index_path)
        index_dir = path.parent
        index_name = path.name

        meta_path = index_dir / f"{index_name}.meta.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"Index metadata not found: {meta_path}")

        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)

        # Check if index supports updates
        backend_name = meta.get("backend_name")
        if backend_name != "hnsw":
            raise ValueError(f"sync_index() currently only supports HNSW backend, got '{backend_name}'.")

        meta_backend_kwargs = meta.get("backend_kwargs", {})
        index_is_compact = meta.get("is_compact", meta_backend_kwargs.get("is_compact", True))
        if index_is_compact:
            raise ValueError(
                "Compact HNSW indices do not support sync operations. Rebuild with is_compact=False."
            )

        # Check if embeddings are stored (needed for deletion/reconstruction)
        is_pruned = bool(meta.get("is_pruned"))
        is_recompute = bool(meta_backend_kwargs.get("is_recompute"))
        if is_pruned or is_recompute:
            raise ValueError(
                "Cannot sync indices with pruned embeddings (is_pruned=True or is_recompute=True). "
                "Sync requires stored embeddings for reconstruction. Rebuild with is_recompute=False."
            )

        # 2. Load existing file hashes (or empty dict if first sync)
        hash_file = index_dir / f"{index_name}.file_hashes.json"
        if hash_file.exists():
            with open(hash_file, encoding="utf-8") as f:
                old_hashes = json.load(f)
        else:
            old_hashes = {}

        logger.info("Syncing index '%s' with directory '%s'", index_path, docs_path)
        if old_hashes:
            logger.info("Loaded %d file hashes from previous sync", len(old_hashes))
        else:
            logger.info("First sync - treating all files as new")

        # 3. Scan current directory and compute new hashes
        docs_dir = Path(docs_path)
        if not docs_dir.exists():
            raise FileNotFoundError(f"Directory not found: {docs_path}")
        if not docs_dir.is_dir():
            raise ValueError(f"Not a directory: {docs_path}")

        new_hashes: dict[str, str] = {}
        logger.info("Scanning directory and computing file hashes...")

        for file_path in docs_dir.rglob("*"):
            if not file_path.is_file():
                continue

            abs_path = str(file_path.resolve())

            # Apply file filter if provided
            if file_filter and not file_filter(abs_path):
                continue

            try:
                file_hash = self._compute_file_hash(file_path)
                new_hashes[abs_path] = file_hash
            except (OSError, IOError) as e:
                logger.warning("Failed to read file '%s': %s. Skipping.", abs_path, e)
                continue

        logger.info("Found %d files in directory", len(new_hashes))

        # 4. Categorize files
        old_files = set(old_hashes.keys())
        new_files = set(new_hashes.keys())

        deleted_files = old_files - new_files
        added_files = new_files - old_files
        common_files = old_files & new_files
        modified_files = {f for f in common_files if old_hashes[f] != new_hashes[f]}
        unchanged_files = common_files - modified_files

        logger.info(
            "Change summary: %d added, %d modified, %d deleted, %d unchanged",
            len(added_files),
            len(modified_files),
            len(deleted_files),
            len(unchanged_files),
        )

        # 5. Process changes
        stats = {
            "added": 0,
            "modified": 0,
            "deleted": 0,
            "unchanged": len(unchanged_files),
            "chunks_added": 0,
            "chunks_deleted": 0,
        }

        # Handle deleted files
        for file_path in deleted_files:
            try:
                deleted_count = self.delete_by_file(index_path, file_path)
                stats["deleted"] += 1
                stats["chunks_deleted"] += deleted_count
                logger.info("Deleted %d chunks from removed file '%s'", deleted_count, file_path)
            except Exception as e:
                logger.error("Failed to delete chunks for file '%s': %s", file_path, e)
                raise

        # Handle modified files
        for file_path in modified_files:
            try:
                deleted, added = self.update_file(index_path, file_path)
                stats["modified"] += 1
                stats["chunks_deleted"] += deleted
                stats["chunks_added"] += added
                logger.info(
                    "Updated modified file '%s': deleted %d, added %d chunks",
                    file_path,
                    deleted,
                    added,
                )
            except Exception as e:
                logger.error("Failed to update file '%s': %s", file_path, e)
                raise

        # Handle new files
        for file_path in added_files:
            try:
                # Use update_file which handles non-existent files gracefully
                # (delete returns 0, then adds new chunks)
                deleted, added = self.update_file(index_path, file_path)
                stats["added"] += 1
                stats["chunks_added"] += added
                logger.info("Added new file '%s': %d chunks", file_path, added)
            except Exception as e:
                logger.error("Failed to add file '%s': %s", file_path, e)
                raise

        # 6. Save new hashes
        try:
            with open(hash_file, "w", encoding="utf-8") as f:
                json.dump(new_hashes, f, indent=2)
            logger.info("Saved %d file hashes to '%s'", len(new_hashes), hash_file)
        except (OSError, IOError) as e:
            logger.error("Failed to save hash file: %s", e)
            raise

        logger.info(
            "Sync complete: %d added, %d modified, %d deleted, %d unchanged. "
            "Chunks: +%d, -%d",
            stats["added"],
            stats["modified"],
            stats["deleted"],
            stats["unchanged"],
            stats["chunks_added"],
            stats["chunks_deleted"],
        )

        return stats


class LeannSearcher:
    def __init__(self, index_path: str, enable_warmup: bool = False, **backend_kwargs):
        # Fix path resolution for Colab and other environments
        if not Path(index_path).is_absolute():
            index_path = str(Path(index_path).resolve())

        self.meta_path_str = f"{index_path}.meta.json"
        if not Path(self.meta_path_str).exists():
            parent_dir = Path(index_path).parent
            print(
                f"Leann metadata file not found at {self.meta_path_str}, and you may need to rm -rf {parent_dir}"
            )
            # highlight in red the filenotfound error
            raise FileNotFoundError(
                f"Leann metadata file not found at {self.meta_path_str}, \033[91m you may need to rm -rf {parent_dir}\033[0m"
            )
        with open(self.meta_path_str, encoding="utf-8") as f:
            self.meta_data = json.load(f)
        backend_name = self.meta_data["backend_name"]
        self.embedding_model = self.meta_data["embedding_model"]
        # Support both old and new format
        self.embedding_mode = self.meta_data.get("embedding_mode", "sentence-transformers")
        self.embedding_options = self.meta_data.get("embedding_options", {})
        # Delegate portability handling to PassageManager
        self.passage_manager = PassageManager(
            self.meta_data.get("passage_sources", []), metadata_file_path=self.meta_path_str
        )
        # Preserve backend name for conditional parameter forwarding
        self.backend_name = backend_name
        backend_factory = BACKEND_REGISTRY.get(backend_name)
        if backend_factory is None:
            raise ValueError(f"Backend '{backend_name}' not found.")
        final_kwargs = {**self.meta_data.get("backend_kwargs", {}), **backend_kwargs}
        final_kwargs["enable_warmup"] = enable_warmup
        if self.embedding_options:
            final_kwargs.setdefault("embedding_options", self.embedding_options)
        self.backend_impl: LeannBackendSearcherInterface = backend_factory.searcher(
            index_path, **final_kwargs
        )

    def search(
        self,
        query: str,
        top_k: int = 5,
        complexity: int = 64,
        beam_width: int = 1,
        prune_ratio: float = 0.0,
        recompute_embeddings: bool = True,
        pruning_strategy: Literal["global", "local", "proportional"] = "global",
        expected_zmq_port: int = 5557,
        metadata_filters: Optional[dict[str, dict[str, Union[str, int, float, bool, list]]]] = None,
        batch_size: int = 0,
        use_grep: bool = False,
        provider_options: Optional[dict[str, Any]] = None,
        **kwargs,
    ) -> list[SearchResult]:
        """
        Search for nearest neighbors with optional metadata filtering.

        Args:
            query: Text query to search for
            top_k: Number of nearest neighbors to return
            complexity: Search complexity/candidate list size, higher = more accurate but slower
            beam_width: Number of parallel search paths/IO requests per iteration
            prune_ratio: Ratio of neighbors to prune via approximate distance (0.0-1.0)
            recompute_embeddings: Whether to fetch fresh embeddings from server vs use stored codes
            pruning_strategy: Candidate selection strategy - "global" (default), "local", or "proportional"
            expected_zmq_port: ZMQ port for embedding server communication
            metadata_filters: Optional filters to apply to search results based on metadata.
                Format: {"field_name": {"operator": value}}
                Supported operators:
                - Comparison: "==", "!=", "<", "<=", ">", ">="
                - Membership: "in", "not_in"
                - String: "contains", "starts_with", "ends_with"
                Example: {"chapter": {"<=": 5}, "tags": {"in": ["fiction", "drama"]}}
            **kwargs: Backend-specific parameters

        Returns:
            List of SearchResult objects with text, metadata, and similarity scores
        """
        # Handle grep search
        if use_grep:
            return self._grep_search(query, top_k)

        logger.info("🔍 LeannSearcher.search() called:")
        logger.info(f"  Query: '{query}'")
        logger.info(f"  Top_k: {top_k}")
        logger.info(f"  Metadata filters: {metadata_filters}")
        logger.info(f"  Additional kwargs: {kwargs}")

        # Smart top_k detection and adjustment
        # Use PassageManager length (sum of shard sizes) to avoid
        # depending on a massive combined map
        total_docs = len(self.passage_manager)
        original_top_k = top_k
        if top_k > total_docs:
            top_k = total_docs
            logger.warning(
                f"  ⚠️  Requested top_k ({original_top_k}) exceeds total documents ({total_docs})"
            )
            logger.warning(f"  ✅ Auto-adjusted top_k to {top_k} to match available documents")

        zmq_port = None

        start_time = time.time()
        if recompute_embeddings:
            zmq_port = self.backend_impl._ensure_server_running(
                self.meta_path_str,
                port=expected_zmq_port,
                **kwargs,
            )
            del expected_zmq_port
        zmq_time = time.time() - start_time
        logger.info(f"  Launching server time: {zmq_time} seconds")

        start_time = time.time()

        # Extract query template from stored embedding_options with fallback chain:
        # 1. Check provider_options override (highest priority)
        # 2. Check query_prompt_template (new format)
        # 3. Check prompt_template (old format for backward compat)
        # 4. None (no template)
        query_template = None
        if provider_options and "prompt_template" in provider_options:
            query_template = provider_options["prompt_template"]
        elif "query_prompt_template" in self.embedding_options:
            query_template = self.embedding_options["query_prompt_template"]
        elif "prompt_template" in self.embedding_options:
            query_template = self.embedding_options["prompt_template"]

        query_embedding = self.backend_impl.compute_query_embedding(
            query,
            use_server_if_available=recompute_embeddings,
            zmq_port=zmq_port,
            query_template=query_template,
        )
        logger.info(f"  Generated embedding shape: {query_embedding.shape}")
        embedding_time = time.time() - start_time
        logger.info(f"  Embedding time: {embedding_time} seconds")

        start_time = time.time()
        backend_search_kwargs: dict[str, Any] = {
            "complexity": complexity,
            "beam_width": beam_width,
            "prune_ratio": prune_ratio,
            "recompute_embeddings": recompute_embeddings,
            "pruning_strategy": pruning_strategy,
            "zmq_port": zmq_port,
        }
        # Only HNSW supports batching; forward conditionally
        if self.backend_name == "hnsw":
            backend_search_kwargs["batch_size"] = batch_size

        # Merge any extra kwargs last
        backend_search_kwargs.update(kwargs)

        results = self.backend_impl.search(
            query_embedding,
            top_k,
            **backend_search_kwargs,
        )
        search_time = time.time() - start_time
        logger.info(f"  Search time in search() LEANN searcher: {search_time} seconds")
        logger.info(f"  Backend returned: labels={len(results.get('labels', [[]])[0])} results")

        enriched_results = []
        if "labels" in results and "distances" in results:
            logger.info(f"  Processing {len(results['labels'][0])} passage IDs:")
            # Python 3.9 does not support zip(strict=...); lengths are expected to match
            for i, (string_id, dist) in enumerate(
                zip(results["labels"][0], results["distances"][0])
            ):
                try:
                    passage_data = self.passage_manager.get_passage(string_id)
                    enriched_results.append(
                        SearchResult(
                            id=string_id,
                            score=dist,
                            text=passage_data["text"],
                            metadata=passage_data.get("metadata", {}),
                        )
                    )

                    # Color codes for better logging
                    GREEN = "\033[92m"
                    BLUE = "\033[94m"
                    YELLOW = "\033[93m"
                    RESET = "\033[0m"

                    # Truncate text for display (first 100 chars)
                    display_text = passage_data["text"]
                    logger.info(
                        f"   {GREEN}✓{RESET} {BLUE}[{i + 1:2d}]{RESET} {YELLOW}ID:{RESET} '{string_id}' {YELLOW}Score:{RESET} {dist:.4f} {YELLOW}Text:{RESET} {display_text}"
                    )
                except KeyError:
                    RED = "\033[91m"
                    RESET = "\033[0m"
                    logger.error(
                        f"   {RED}✗{RESET} [{i + 1:2d}] ID: '{string_id}' -> {RED}ERROR: Passage not found!{RESET}"
                    )

        # Apply metadata filters if specified
        if metadata_filters:
            logger.info(f"  🔍 Applying metadata filters: {metadata_filters}")
            enriched_results = self.passage_manager.filter_search_results(
                enriched_results, metadata_filters
            )

        # Define color codes outside the loop for final message
        GREEN = "\033[92m"
        RESET = "\033[0m"
        logger.info(f"  {GREEN}✓ Final enriched results: {len(enriched_results)} passages{RESET}")
        return enriched_results

    def _find_jsonl_file(self) -> Optional[str]:
        """Find the .jsonl file containing raw passages for grep search"""
        index_path = Path(self.meta_path_str).parent
        potential_files = [
            index_path / "documents.leann.passages.jsonl",
            index_path.parent / "documents.leann.passages.jsonl",
        ]

        for file_path in potential_files:
            if file_path.exists():
                return str(file_path)
        return None

    def _grep_search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        """Perform grep-based search on raw passages"""
        jsonl_file = self._find_jsonl_file()
        if not jsonl_file:
            raise FileNotFoundError("No .jsonl passages file found for grep search")

        try:
            cmd = ["grep", "-i", "-n", query, jsonl_file]
            result = subprocess.run(cmd, capture_output=True, text=True, check=False)

            if result.returncode == 1:
                return []
            elif result.returncode != 0:
                raise RuntimeError(f"Grep failed: {result.stderr}")

            matches = []
            for line in result.stdout.strip().split("\n"):
                if not line:
                    continue
                parts = line.split(":", 1)
                if len(parts) != 2:
                    continue

                try:
                    data = json.loads(parts[1])
                    text = data.get("text", "")
                    score = text.lower().count(query.lower())

                    matches.append(
                        SearchResult(
                            id=data.get("id", parts[0]),
                            text=text,
                            metadata=data.get("metadata", {}),
                            score=float(score),
                        )
                    )
                except json.JSONDecodeError:
                    continue

            matches.sort(key=lambda x: x.score, reverse=True)
            return matches[:top_k]

        except FileNotFoundError:
            raise RuntimeError(
                "grep command not found. Please install grep or use semantic search."
            )

    def _python_regex_search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        """Fallback regex search"""
        jsonl_file = self._find_jsonl_file()
        if not jsonl_file:
            raise FileNotFoundError("No .jsonl file found")

        pattern = re.compile(re.escape(query), re.IGNORECASE)
        matches = []

        with open(jsonl_file, encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                if pattern.search(line):
                    try:
                        data = json.loads(line.strip())
                        matches.append(
                            SearchResult(
                                id=data.get("id", str(line_num)),
                                text=data.get("text", ""),
                                metadata=data.get("metadata", {}),
                                score=float(len(pattern.findall(data.get("text", "")))),
                            )
                        )
                    except json.JSONDecodeError:
                        continue

        matches.sort(key=lambda x: x.score, reverse=True)
        return matches[:top_k]

    def cleanup(self):
        """Explicitly cleanup embedding server resources.
        This method should be called after you're done using the searcher,
        especially in test environments or batch processing scenarios.
        """
        backend = getattr(self.backend_impl, "embedding_server_manager", None)
        if backend is not None:
            backend.stop_server()

    # Enable automatic cleanup patterns
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        try:
            self.cleanup()
        except Exception:
            pass

    def __del__(self):
        try:
            self.cleanup()
        except Exception:
            # Avoid noisy errors during interpreter shutdown
            pass


class LeannChat:
    def __init__(
        self,
        index_path: str,
        llm_config: Optional[dict[str, Any]] = None,
        enable_warmup: bool = False,
        searcher: Optional[LeannSearcher] = None,
        **kwargs,
    ):
        if searcher is None:
            self.searcher = LeannSearcher(index_path, enable_warmup=enable_warmup, **kwargs)
            self._owns_searcher = True
        else:
            self.searcher = searcher
            self._owns_searcher = False
        self.llm = get_llm(llm_config)

    def ask(
        self,
        question: str,
        top_k: int = 5,
        complexity: int = 64,
        beam_width: int = 1,
        prune_ratio: float = 0.0,
        recompute_embeddings: bool = True,
        pruning_strategy: Literal["global", "local", "proportional"] = "global",
        llm_kwargs: Optional[dict[str, Any]] = None,
        expected_zmq_port: int = 5557,
        metadata_filters: Optional[dict[str, dict[str, Union[str, int, float, bool, list]]]] = None,
        batch_size: int = 0,
        use_grep: bool = False,
        **search_kwargs,
    ):
        if llm_kwargs is None:
            llm_kwargs = {}
        search_time = time.time()
        results = self.searcher.search(
            question,
            top_k=top_k,
            complexity=complexity,
            beam_width=beam_width,
            prune_ratio=prune_ratio,
            recompute_embeddings=recompute_embeddings,
            pruning_strategy=pruning_strategy,
            expected_zmq_port=expected_zmq_port,
            metadata_filters=metadata_filters,
            batch_size=batch_size,
            **search_kwargs,
        )
        search_time = time.time() - search_time
        logger.info(f"  Search time: {search_time} seconds")
        context = "\n\n".join([r.text for r in results])
        prompt = (
            "Here is some retrieved context that might help answer your question:\n\n"
            f"{context}\n\n"
            f"Question: {question}\n\n"
            "Please provide the best answer you can based on this context and your knowledge."
        )

        print("The context provided to the LLM is:")
        print(f"{'Relevance':<10} | {'Chunk id':<10} | {'Content':<60} | {'Source':<80}")
        print("-" * 150)
        for r in results:
            chunk_relevance = f"{r.score:.3f}"
            chunk_id = r.id
            chunk_content = r.text[:60]
            chunk_source = r.metadata.get("source", "")[:80]
            print(
                f"{chunk_relevance:<10} | {chunk_id:<10} | {chunk_content:<60} | {chunk_source:<80}"
            )
        ask_time = time.time()
        ans = self.llm.ask(prompt, **llm_kwargs)
        ask_time = time.time() - ask_time
        logger.info(f"  Ask time: {ask_time} seconds")
        return ans

    def start_interactive(self):
        """Start interactive chat session."""
        session = create_api_session()

        def handle_query(user_input: str):
            response = self.ask(user_input)
            print(f"Leann: {response}")

        session.run_interactive_loop(handle_query)

    def cleanup(self):
        """Explicitly cleanup embedding server resources.

        This method should be called after you're done using the chat interface,
        especially in test environments or batch processing scenarios.
        """
        # Only stop the embedding server if this LeannChat instance created the searcher.
        # When a shared searcher is passed in, avoid shutting down the server to enable reuse.
        if getattr(self, "_owns_searcher", False) and hasattr(self.searcher, "cleanup"):
            self.searcher.cleanup()

    # Enable automatic cleanup patterns
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        try:
            self.cleanup()
        except Exception:
            pass

    def __del__(self):
        try:
            self.cleanup()
        except Exception:
            pass
