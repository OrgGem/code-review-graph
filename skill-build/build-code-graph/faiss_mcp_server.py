from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

try:
    import faiss
except ImportError:  # pragma: no cover - dependency check at runtime
    faiss = None

try:
    import numpy as np
except ImportError:  # pragma: no cover - dependency check at runtime
    np = None

try:
    from mcp.server.fastmcp import FastMCP
except ImportError:  # pragma: no cover - dependency check at runtime
    FastMCP = None  # type: ignore[assignment]

mcp = FastMCP("faiss-code-index") if FastMCP is not None else None
_INDEX: faiss.Index | None = None
_METADATA: dict | None = None
_DIM: int = 384
_SERVER_INDEX_DIR = ".code-review-graph/faiss-index"


def _hash_embed(text: str, dim: int) -> np.ndarray:
    if np is None:
        raise RuntimeError("numpy is required. Install with: python3 -m pip install numpy")

    vec = np.zeros(dim, dtype=np.float32)
    tokens = re.findall(r"[A-Za-z0-9_]+|[^\sA-Za-z0-9_]", text.lower())
    for token in tokens:
        idx = hash(token) % dim
        vec[idx] += 1.0
    norm = float(np.linalg.norm(vec))
    if norm > 0:
        vec /= norm
    return vec


def _ensure_loaded(index_dir: Path) -> None:
    global _INDEX, _METADATA, _DIM

    if _INDEX is not None and _METADATA is not None:
        return

    index_path = index_dir / "code.index"
    metadata_path = index_dir / "metadata.json"

    if not index_path.exists() or not metadata_path.exists():
        raise FileNotFoundError(
            f"Missing index files in {index_dir}. Run build_faiss_index.py first."
        )

    if faiss is None:
        raise RuntimeError("faiss-cpu is required. Install with: python3 -m pip install faiss-cpu")

    _INDEX = faiss.read_index(str(index_path))
    _METADATA = json.loads(metadata_path.read_text(encoding="utf-8"))
    _DIM = int(_METADATA.get("dimension", 384))


def search_code_chunks(query: str, top_k: int = 8) -> dict:
    """Semantic-like code chunk search using local FAISS index."""
    index_dir_env = Path(_SERVER_INDEX_DIR)
    _ensure_loaded(index_dir_env)

    assert _INDEX is not None
    assert _METADATA is not None

    q = _hash_embed(query, _DIM).reshape(1, -1)
    scores, ids = _INDEX.search(q, top_k)

    chunks = _METADATA.get("chunks", [])
    hits = []
    for rank, (score, idx) in enumerate(zip(scores[0], ids[0]), start=1):
        if idx < 0 or idx >= len(chunks):
            continue
        chunk = chunks[idx]
        hits.append(
            {
                "rank": rank,
                "score": float(score),
                "path": chunk.get("path"),
                "start_line": chunk.get("start_line"),
                "end_line": chunk.get("end_line"),
                "preview": chunk.get("preview"),
            }
        )

    return {
        "query": query,
        "top_k": top_k,
        "total_chunks": len(chunks),
        "results": hits,
    }


if mcp is not None:
    mcp.tool()(search_code_chunks)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run MCP server for FAISS code index")
    parser.add_argument(
        "--index-dir",
        default=".code-review-graph/faiss-index",
        help="Directory containing code.index and metadata.json",
    )
    args = parser.parse_args()

    global _SERVER_INDEX_DIR
    _SERVER_INDEX_DIR = str(Path(args.index_dir).resolve())

    if mcp is None:
        raise RuntimeError("mcp package is required. Install with: python3 -m pip install mcp")

    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
